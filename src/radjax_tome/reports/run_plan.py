from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from radjax_tome.backends.base import (
    TeacherBackendConfig,
    TeacherBatchInput,
    gpu_batch_size_candidates,
    resolve_exemplar_capture_policy,
    resolve_gpu_batch_size_policy,
)
from radjax_tome.corpora import corpus_provenance_from_manifest, read_corpus_manifest
from radjax_tome.io.json import write_json
from radjax_tome.provenance import (
    teacher_model_provenance_summary,
    validate_teacher_model_provenance,
)
from radjax_tome.reports.runtime_doctor import build_runtime_doctor_report

GPU_RUN_PLAN_SCHEMA = "gpu_run_plan_v1"
GPU_RUN_PLAN_FILENAME = "run_plan.json"
GPU_RUN_PLANNER_VERSION = "spec_4_5_gpu_run_planner_v1"
AUTO_BATCH_SELECTION_POLICY = "largest_passing_batch_size_with_safety_margin_v1"
_ARTIFACT_WARNING_THRESHOLD_BYTES = 1_000_000_000
_DENSE_LARGE_CORPUS_EXAMPLES = 1_000
_DTYPE_BYTES = 4

ProbeCandidateRunner = Callable[
    [TeacherBackendConfig, int],
    Mapping[str, object],
]


@dataclass(frozen=True)
class GPURunPlanConfig:
    backend_config: TeacherBackendConfig
    dataset_path: Path
    corpus_manifest_path: Path | None = None
    teacher_model_provenance_path: Path | None = None
    max_examples: int | None = None
    exemplar_selection_enabled: bool = False
    exemplar_fulfillment_policy: str = "auto"
    strict_provenance: bool = False
    max_artifact_bytes: int | None = None
    fail_on_warnings: bool = False
    selection_integration_policy: str = "global_only_v1"
    total_selected_exemplar_budget: int | None = None
    fingerprint_corridor_budget_fraction: str = "0.50"
    fingerprint_corridor_budget_max: int | None = None
    fingerprint_corridor_mode_cap: int = 10
    fingerprint_corridor_candidate_pool_cap: int = 4
    require_full_selected_budget: bool = True


def build_gpu_run_plan(
    config: GPURunPlanConfig,
    *,
    probe_candidate_runner: ProbeCandidateRunner | None = None,
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    backend_config = config.backend_config
    doctor = build_runtime_doctor_report(
        backend_config,
        exemplar_selection_enabled=config.exemplar_selection_enabled,
        exemplar_fulfillment_policy=config.exemplar_fulfillment_policy,
    )
    _add_doctor_blockers(doctor, blockers, warnings)

    dataset = _dataset_summary(config.dataset_path, blockers)
    corpus = _corpus_provenance_summary(
        config.corpus_manifest_path,
        dataset_path=config.dataset_path,
        blockers=blockers,
        warnings=warnings,
        strict=config.strict_provenance,
    )
    model = _teacher_model_provenance_summary(
        config.teacher_model_provenance_path,
        blockers=blockers,
        warnings=warnings,
        strict=config.strict_provenance,
    )
    auto_probe = _auto_batch_probe_summary(
        backend_config,
        probe_candidate_runner=probe_candidate_runner,
        blockers=blockers,
        warnings=warnings,
    )
    probe_results = tuple(auto_probe.get("candidate_results", ()) or ())
    resolved = resolve_gpu_batch_size_policy(
        backend_config,
        probe_results=probe_results if probe_results else None,
        payload=None,
    )
    resolved_policy = _resolved_batch_policy(resolved, auto_probe)
    selected_batch_size = resolved_policy.get("effective_gpu_batch_size")
    memory = _memory_estimates(
        backend_config,
        selected_batch_size=(
            int(selected_batch_size) if selected_batch_size is not None else None
        ),
        observed_peak_memory_bytes=auto_probe.get("observed_peak_memory_bytes"),
    )
    artifact = _artifact_estimates(
        backend_config,
        dataset_count=dataset.get("num_examples"),
        max_examples=config.max_examples,
    )
    _add_estimate_warnings(
        backend_config,
        artifact,
        warnings,
        max_artifact_bytes=config.max_artifact_bytes,
        blockers=blockers,
    )
    capture = _capture_mode_estimates(
        backend_config,
        selection_enabled=config.exemplar_selection_enabled,
        fulfillment_policy=config.exemplar_fulfillment_policy,
        warnings=warnings,
    )
    if config.fail_on_warnings and warnings:
        blockers.append("fail_on_warnings enabled and planner emitted warnings")

    status = "fail" if blockers else "warn" if warnings else "pass"
    return {
        "schema_version": GPU_RUN_PLAN_SCHEMA,
        "status": status,
        "blockers": blockers,
        "warnings": warnings,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "planner_version": GPU_RUN_PLANNER_VERSION,
        "backend": backend_config.backend_id,
        "runtime": backend_config.runtime_mode,
        "target_policy": backend_config.target_policy,
        "model": {
            "model_id": backend_config.model_id,
            "teacher_model_provenance_path": _path_or_none(
                config.teacher_model_provenance_path
            ),
        },
        "tokenizer": {
            "tokenizer_id": backend_config.tokenizer_id,
        },
        "dataset": dataset,
        "corpus_provenance": corpus,
        "teacher_model_provenance": model,
        "environment": _environment_from_doctor(doctor),
        "doctor_diagnostics": doctor,
        "requested_batch_policy": _requested_batch_policy(backend_config),
        "resolved_batch_policy": resolved_policy,
        "auto_batch_probe": auto_probe,
        "memory_estimates": memory,
        "artifact_estimates": artifact,
        "estimate_notes": ["memory and artifact estimates are rough"],
        "capture_mode_estimates": capture,
        "recommended_command": _recommended_command(backend_config, resolved_policy),
        "claims_not_made": _claims_not_made(),
    }


def write_gpu_run_plan(plan: Mapping[str, Any], path: Path) -> None:
    write_json(path, dict(plan))


def render_gpu_run_plan_summary(plan: Mapping[str, Any], output: Path) -> list[str]:
    auto_probe = plan.get("auto_batch_probe", {})
    resolved = plan.get("resolved_batch_policy", {})
    artifact = plan.get("artifact_estimates", {})
    if not isinstance(auto_probe, Mapping):
        auto_probe = {}
    if not isinstance(resolved, Mapping):
        resolved = {}
    if not isinstance(artifact, Mapping):
        artifact = {}
    return [
        (
            f"status={plan.get('status')} output={output} "
            f"selected_batch_size={resolved.get('effective_gpu_batch_size')}"
        ),
        f"probe_status={auto_probe.get('probe_status')}",
        f"largest_passing_batch_size={auto_probe.get('largest_passing_batch_size')}",
        f"first_failing_batch_size={auto_probe.get('first_failing_batch_size')}",
        f"estimated_artifact_bytes={artifact.get('estimated_total_artifact_bytes')}",
        f"warnings={len(plan.get('warnings', ()) or ())}",
    ]


def run_gpu_torch_auto_batch_probe(
    config: TeacherBackendConfig,
    *,
    probe_candidate_runner: ProbeCandidateRunner | None = None,
) -> dict[str, Any]:
    candidates = gpu_batch_size_candidates(config)
    results: list[dict[str, object]] = []
    runner = probe_candidate_runner or _run_real_gpu_torch_probe_candidate
    for candidate in candidates:
        result = dict(runner(config, candidate))
        result.setdefault("candidate_batch_size", candidate)
        result.setdefault("status", "pass" if result.get("success") else "fail")
        result.setdefault("success", result["status"] == "pass")
        results.append(result)
        if result["status"] == "fail":
            break
    return _summarize_probe_results(results)


def _auto_batch_probe_summary(
    config: TeacherBackendConfig,
    *,
    probe_candidate_runner: ProbeCandidateRunner | None,
    blockers: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    if config.gpu_batch_size_mode != "auto":
        return {
            "probe_status": "skipped",
            "candidate_count": 0,
            "candidate_results": [],
            "largest_passing_batch_size": None,
            "first_failing_batch_size": None,
            "selected_batch_size": None,
            "selection_policy": AUTO_BATCH_SELECTION_POLICY,
            "skip_reason": "gpu_batch_size_mode is not auto",
            "probe_performed": False,
        }
    if config.backend_id != "gpu_torch":
        blockers.append("auto GPU batch probing requires backend gpu_torch")
        return {
            "probe_status": "fail",
            "candidate_count": 0,
            "candidate_results": [],
            "largest_passing_batch_size": None,
            "first_failing_batch_size": None,
            "selected_batch_size": None,
            "selection_policy": AUTO_BATCH_SELECTION_POLICY,
            "probe_performed": False,
        }
    probe = run_gpu_torch_auto_batch_probe(
        config,
        probe_candidate_runner=probe_candidate_runner,
    )
    if probe["probe_status"] == "fail":
        blockers.append("auto batch probe failed: no candidate batch size passed")
    elif probe.get("largest_passing_batch_size") == config.gpu_batch_size_auto_min:
        warnings.append("auto batch probe only passed the minimum candidate")
    elif probe.get("first_failing_batch_size") is None:
        warnings.append(
            "all auto batch probe candidates passed; max may not be optimal"
        )
    return probe


def _run_real_gpu_torch_probe_candidate(
    config: TeacherBackendConfig,
    candidate_batch_size: int,
) -> dict[str, object]:
    from dataclasses import replace

    from radjax_tome.backends.gpu_torch import GPUTorchTeacherEmissionBackend

    probe_config = replace(config, batch_size=candidate_batch_size)
    batch = TeacherBatchInput(
        example_ids=tuple(f"probe-{index}" for index in range(candidate_batch_size)),
        texts=tuple(
            "RADJAX Tome GPU batch probe." for _ in range(candidate_batch_size)
        ),
    )
    started = time.perf_counter()
    try:
        backend = GPUTorchTeacherEmissionBackend(probe_config)
        result = backend.emit_batch(batch)
        _synchronize_torch_if_available()
        duration_ms = int((time.perf_counter() - started) * 1000)
        metadata = result.metadata
        return {
            "candidate_batch_size": candidate_batch_size,
            "status": "pass",
            "success": True,
            "failure_stage": None,
            "failure_reason": None,
            "duration_ms": duration_ms,
            "device_name": metadata.get("torch_device"),
            "observed_memory_allocated_bytes": _cuda_memory("memory_allocated"),
            "observed_memory_reserved_bytes": _cuda_memory("memory_reserved"),
        }
    except Exception as exc:  # pragma: no cover - real GPU path is host-specific
        duration_ms = int((time.perf_counter() - started) * 1000)
        return {
            "candidate_batch_size": candidate_batch_size,
            "status": "fail",
            "success": False,
            "failure_stage": _classify_probe_failure(exc),
            "failure_reason": str(exc),
            "duration_ms": duration_ms,
            "device_name": None,
            "oom_or_device_failure": _is_oom_or_device_error(exc),
        }


def _summarize_probe_results(results: list[dict[str, object]]) -> dict[str, Any]:
    largest_pass: int | None = None
    first_fail: int | None = None
    observed_peak: int | None = None
    for result in results:
        candidate = int(result["candidate_batch_size"])
        if result.get("status") == "pass":
            largest_pass = candidate
        elif first_fail is None:
            first_fail = candidate
        for key in (
            "observed_memory_allocated_bytes",
            "observed_memory_reserved_bytes",
        ):
            value = result.get(key)
            if value is not None:
                observed_peak = max(observed_peak or 0, int(value))
    selected = largest_pass
    status = "pass" if largest_pass is not None else "fail"
    return {
        "probe_status": status,
        "candidate_count": len(results),
        "candidate_results": results,
        "largest_passing_batch_size": largest_pass,
        "first_failing_batch_size": first_fail,
        "selected_batch_size": selected,
        "selection_policy": AUTO_BATCH_SELECTION_POLICY,
        "probe_performed": True,
        "observed_peak_memory_bytes": observed_peak,
    }


def _dataset_summary(path: Path, blockers: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "dataset_path": str(path),
        "exists": path.is_file(),
        "num_examples": None,
        "max_examples_effective": None,
    }
    if not path.is_file():
        blockers.append(f"dataset path missing: {path}")
        return summary
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows += 1
    summary["num_examples"] = rows
    summary["size_bytes"] = path.stat().st_size
    summary["sha256"] = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    return summary


def _corpus_provenance_summary(
    path: Path | None,
    *,
    dataset_path: Path,
    blockers: list[str],
    warnings: list[str],
    strict: bool,
) -> dict[str, Any]:
    if path is None:
        message = "no corpus manifest; run planning has weaker provenance"
        (blockers if strict else warnings).append(message)
        return {"provided": False, "status": "missing"}
    try:
        manifest = read_corpus_manifest(path)
        provenance = corpus_provenance_from_manifest(path)
        if dataset_path.is_file():
            dataset_hash = (
                "sha256:" + hashlib.sha256(dataset_path.read_bytes()).hexdigest()
            )
            if manifest.get("corpus_hash") != dataset_hash:
                blockers.append("corpus manifest hash does not match dataset")
                return {
                    "provided": True,
                    "status": "fail",
                    "corpus_manifest_path": str(path),
                    "dataset_hash": dataset_hash,
                    **provenance,
                }
        return {
            "provided": True,
            "status": "pass",
            "corpus_manifest_path": str(path),
            **provenance,
        }
    except Exception as exc:
        blockers.append(f"corpus manifest invalid: {exc}")
        return {"provided": True, "status": "fail", "corpus_manifest_path": str(path)}


def _teacher_model_provenance_summary(
    path: Path | None,
    *,
    blockers: list[str],
    warnings: list[str],
    strict: bool,
) -> dict[str, Any]:
    if path is None:
        message = "no teacher model provenance; run planning has weaker provenance"
        (blockers if strict else warnings).append(message)
        return {"provided": False, "status": "missing"}
    report = validate_teacher_model_provenance(path)
    if report.status == "fail":
        blockers.extend(
            f"teacher model provenance invalid: {item}" for item in report.blockers
        )
        return {
            "provided": True,
            "status": "fail",
            "teacher_model_provenance_path": str(path),
        }
    summary = teacher_model_provenance_summary(path)
    return {
        "provided": True,
        "status": report.status,
        "warnings": list(report.warnings),
        **summary,
    }


def _add_doctor_blockers(
    doctor: Mapping[str, Any],
    blockers: list[str],
    warnings: list[str],
) -> None:
    if bool(doctor.get("can_emit", False)):
        return
    stage = str(doctor.get("failure_stage") or "")
    if stage == "model_load":
        warnings.append("doctor could not load local model in metadata-only preflight")
        return
    blockers.append(f"doctor preflight failed: {stage}: {doctor.get('failure_reason')}")


def _memory_estimates(
    config: TeacherBackendConfig,
    *,
    selected_batch_size: int | None,
    observed_peak_memory_bytes: object,
) -> dict[str, Any]:
    if selected_batch_size is None:
        return {
            "estimate_available": False,
            "reason": "selected_batch_size_unavailable",
            "estimate_confidence": "rough",
            "estimated_dense_logits_bytes": None,
            "estimated_compact_payload_bytes": None,
            "estimated_workspace_bytes": None,
            "estimated_total_probe_payload_bytes": None,
            "observed_peak_memory_bytes": None,
        }
    dense = (
        selected_batch_size * config.sequence_length * config.vocab_size * _DTYPE_BYTES
    )
    compact = _compact_payload_estimate(config, selected_batch_size)
    workspace = dense if config.target_policy != "dense_logits" else 0
    observed = (
        int(observed_peak_memory_bytes)
        if observed_peak_memory_bytes is not None
        else None
    )
    return {
        "estimate_available": True,
        "estimate_confidence": "rough",
        "estimated_dense_logits_bytes": dense,
        "estimated_compact_payload_bytes": compact,
        "estimated_workspace_bytes": workspace,
        "estimated_total_probe_payload_bytes": dense + compact + workspace,
        "observed_peak_memory_bytes": observed,
    }


def _compact_payload_estimate(config: TeacherBackendConfig, batch_size: int) -> int:
    positions = batch_size * config.sequence_length
    top_k = max(1, min(config.top_k, config.vocab_size))
    if config.target_policy == "dense_logits":
        return 0
    if config.target_policy == "topk_with_tail_v0":
        return positions * (top_k * 3 + 3) * _DTYPE_BYTES
    if config.target_policy == "cascaded_soft_labels_v1":
        return positions * (top_k * 3 + config.num_buckets + 3) * _DTYPE_BYTES
    if config.target_policy == "dynamic_cascaded_soft_labels_v1":
        dynamic_k = max(1, min(config.dynamic_top_k_max, config.vocab_size))
        return positions * (dynamic_k * 4 + config.num_buckets + 4) * _DTYPE_BYTES
    if config.target_policy == "corridor_exemplar_v1":
        capture = resolve_exemplar_capture_policy(config, actual_batch_size=batch_size)
        mode = capture["exemplar_capture_mode_effective"]
        key = (
            "estimated_two_pass_total_bytes"
            if mode == "two_pass_sparse_exemplar"
            else "estimated_one_pass_candidate_bytes"
        )
        return int(capture[key])
    return 0


def _artifact_estimates(
    config: TeacherBackendConfig,
    *,
    dataset_count: object,
    max_examples: int | None,
) -> dict[str, Any]:
    count = int(dataset_count) if dataset_count is not None else None
    effective = (
        min(count, max_examples) if count is not None and max_examples else count
    )
    if effective is None:
        return {
            "estimate_available": False,
            "reason": "dataset_count_unavailable",
            "estimate_confidence": "rough",
        }
    per_example = _compact_payload_estimate(config, 1)
    if config.target_policy == "dense_logits":
        per_example = config.sequence_length * config.vocab_size * _DTYPE_BYTES
    sidecar = 32_768
    shard = int(effective * per_example)
    return {
        "estimate_available": True,
        "estimate_confidence": "rough",
        "dataset_count": count,
        "max_examples": max_examples,
        "num_examples_effective": effective,
        "estimated_shard_bytes": shard,
        "estimated_sidecar_bytes": sidecar,
        "estimated_total_artifact_bytes": shard + sidecar,
    }


def _capture_mode_estimates(
    config: TeacherBackendConfig,
    *,
    selection_enabled: bool,
    fulfillment_policy: str,
    warnings: list[str],
) -> dict[str, Any]:
    capture = resolve_exemplar_capture_policy(config)
    effective = str(capture["exemplar_capture_mode_effective"])
    if config.target_policy == "corridor_exemplar_v1":
        if effective == "two_pass_sparse_exemplar":
            warnings.append(
                "two-pass sparse exemplar capture may require a rerun/selected pass"
            )
        elif not selection_enabled:
            warnings.append(
                "one-pass candidate capture is not a final corpus-global selector"
            )
    return {
        "exemplar_capture_mode_requested": capture["exemplar_capture_mode_requested"],
        "exemplar_capture_mode_effective": effective,
        "requires_second_pass": effective == "two_pass_sparse_exemplar",
        "score_pass_only": effective == "two_pass_sparse_exemplar",
        "selection_enabled": selection_enabled,
        "fulfillment_policy": fulfillment_policy,
        "capture_policy": capture.get("exemplar_capture_policy"),
        "auto_policy_reason": capture.get("auto_policy_reason"),
    }


def _add_estimate_warnings(
    config: TeacherBackendConfig,
    artifact: Mapping[str, Any],
    warnings: list[str],
    *,
    max_artifact_bytes: int | None,
    blockers: list[str],
) -> None:
    total = artifact.get("estimated_total_artifact_bytes")
    count = artifact.get("num_examples_effective")
    if (
        config.target_policy == "dense_logits"
        and count is not None
        and int(count) >= _DENSE_LARGE_CORPUS_EXAMPLES
    ):
        warnings.append("dense logits selected for a larger corpus")
    if total is not None and int(total) > _ARTIFACT_WARNING_THRESHOLD_BYTES:
        warnings.append("estimated artifact size is very large")
    if (
        max_artifact_bytes is not None
        and total is not None
        and int(total) > max_artifact_bytes
    ):
        blockers.append("estimated artifact size exceeds max_artifact_bytes")


def _requested_batch_policy(config: TeacherBackendConfig) -> dict[str, Any]:
    return {
        "gpu_batch_size_mode_requested": config.gpu_batch_size_mode,
        "gpu_batch_size_preset": config.gpu_batch_size_preset,
        "gpu_batch_size_custom": config.gpu_batch_size_custom,
        "gpu_batch_size_auto_min": config.gpu_batch_size_auto_min,
        "gpu_batch_size_auto_max": config.gpu_batch_size_auto_max,
    }


def _resolved_batch_policy(
    resolved: Mapping[str, Any],
    auto_probe: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "gpu_batch_size_mode_effective": resolved["gpu_batch_size_mode_effective"],
        "effective_gpu_batch_size": _effective_batch_size_for_plan(
            resolved,
            auto_probe,
        ),
        "batch_selection_reason": _batch_selection_reason(resolved, auto_probe),
        "probe_required": resolved["gpu_batch_size_probe_required"],
        "probe_performed": bool(auto_probe.get("probe_performed", False)),
    }


def _effective_batch_size_for_plan(
    resolved: Mapping[str, Any],
    auto_probe: Mapping[str, Any],
) -> object | None:
    if (
        auto_probe.get("probe_performed")
        and auto_probe.get("probe_status") == "fail"
        and auto_probe.get("selected_batch_size") is None
    ):
        return None
    return resolved["effective_gpu_batch_size"]


def _batch_selection_reason(
    resolved: Mapping[str, Any],
    auto_probe: Mapping[str, Any],
) -> str:
    if (
        auto_probe.get("probe_performed")
        and auto_probe.get("probe_status") == "fail"
        and auto_probe.get("selected_batch_size") is None
    ):
        return "auto_batch_probe_failed_no_passing_candidate"
    if auto_probe.get("probe_performed"):
        return str(auto_probe.get("selection_policy"))
    mode = resolved["gpu_batch_size_mode_effective"]
    if mode == "custom":
        return "user_custom_batch_size"
    if mode == "preset":
        return "preset_batch_size"
    return "auto_batch_size_requires_probe"


def _environment_from_doctor(doctor: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: doctor.get(key)
        for key in (
            "python_version",
            "platform",
            "torch_available",
            "torch_version",
            "transformers_available",
            "transformers_version",
            "cuda_available",
            "cuda_device_count",
            "cuda_device_names",
            "torch_cuda_version",
            "mps_available",
        )
    }


def _recommended_command(
    config: TeacherBackendConfig,
    resolved: Mapping[str, Any],
) -> str | None:
    batch = resolved.get("effective_gpu_batch_size")
    if batch is None:
        return None
    return (
        "radjax-tome build "
        f"--teacher-backend {config.backend_id} "
        f"--runtime-mode {config.runtime_mode} "
        f"--target-policy {config.target_policy} "
        f"--teacher-model {config.model_id} "
        f"--batch-size {batch} "
        "--dataset ./corpus_out/corpus.jsonl "
        "--corpus-manifest ./corpus_out/corpus_manifest.json "
        "--teacher-model-provenance ./teacher_model_provenance.json "
        "--output ./gpu_tome"
    )


def _claims_not_made() -> dict[str, bool]:
    return {
        "no_production_build_performed": True,
        "no_full_corpus_run": True,
        "no_model_download": True,
        "no_network_verification": True,
        "no_multidevice_scheduling": True,
        "no_tpu_jax": True,
        "estimates_not_exact": True,
    }


def _path_or_none(path: Path | None) -> str | None:
    return str(path) if path is not None else None


def _classify_probe_failure(exc: Exception) -> str:
    text = str(exc).lower()
    if "out of memory" in text or "oom" in text:
        return "oom"
    if "model" in text:
        return "model_load"
    if "tokenizer" in text:
        return "tokenizer_load"
    if "device" in text or "cuda" in text or "mps" in text:
        return "device"
    return "probe"


def _is_oom_or_device_error(exc: Exception) -> bool:
    return _classify_probe_failure(exc) in {"oom", "device"}


def _synchronize_torch_if_available() -> None:
    try:
        import torch
    except Exception:  # pragma: no cover - optional dependency
        return
    cuda = getattr(torch, "cuda", None)
    if callable(getattr(cuda, "is_available", None)) and cuda.is_available():
        cuda.synchronize()


def _cuda_memory(method: str) -> int | None:
    try:
        import torch
    except Exception:  # pragma: no cover - optional dependency
        return None
    cuda = getattr(torch, "cuda", None)
    if not (callable(getattr(cuda, "is_available", None)) and cuda.is_available()):
        return None
    reader = getattr(cuda, method, None)
    if not callable(reader):
        return None
    try:
        return int(reader())
    except Exception:  # pragma: no cover - optional dependency
        return None


def config_to_dict(config: GPURunPlanConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["dataset_path"] = str(config.dataset_path)
    payload["corpus_manifest_path"] = _path_or_none(config.corpus_manifest_path)
    payload["teacher_model_provenance_path"] = _path_or_none(
        config.teacher_model_provenance_path
    )
    return payload
