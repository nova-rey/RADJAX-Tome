from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from radjax_tome.backends import TeacherBackendConfig
from radjax_tome.builder.backend_textbook import (
    BackendTeacherTextbookBuildConfig,
    build_streaming_backend_teacher_textbook,
)
from radjax_tome.builder.teacher_textbook import validate_teacher_textbook
from radjax_tome.corpora import validate_corpus_artifact
from radjax_tome.io.json import read_json_object, write_json
from radjax_tome.provenance import validate_teacher_model_provenance
from radjax_tome.reports import (
    GPURunPlanConfig,
    TomeParityConfig,
    build_gpu_run_plan,
    build_runtime_doctor_report,
    compare_tome_artifacts,
    write_gpu_run_plan,
    write_tome_parity_report,
)
from radjax_tome.tome import write_cover_page

PRODUCTION_BUILD_REPORT_SCHEMA = "production_build_report_v1"
PRODUCTION_BUILD_REPORT_FILENAME = "production_build_report.json"


@dataclass(frozen=True)
class ProductionBuildConfig:
    teacher_model: str
    dataset_path: Path
    corpus_manifest_path: Path
    teacher_model_provenance_path: Path
    output_dir: Path
    tokenizer_id: str | None = None
    teacher_backend: str = "gpu_torch"
    runtime_mode: str = "cpu_gpu"
    target_policy: str = "corridor_exemplar_v1"
    sequence_length: int = 16
    vocab_size: int = 32
    top_k: int = 8
    num_buckets: int = 4
    gpu_batch_size_mode: str = "auto"
    gpu_batch_size_preset: int = 8
    gpu_batch_size_custom: int | None = None
    gpu_batch_size_auto_min: int = 1
    gpu_batch_size_auto_max: int = 64
    shard_size_examples: int = 1024
    max_examples: int | None = None
    resume: bool = False
    overwrite: bool = False
    strict_provenance: bool = True
    fail_on_plan_warnings: bool = False
    no_build_if_plan_warn: bool = False
    max_artifact_bytes: int | None = None
    run_plan_path: Path | None = None
    production_report_path: Path | None = None
    parity_left: Path | None = None
    parity_report_path: Path | None = None
    run_manifest_path: Path | None = None
    progress_log_path: Path | None = None


def build_production_gpu_tome(config: ProductionBuildConfig) -> dict[str, Any]:
    created_at = _now()
    report_path = _production_report_path(config)
    run_plan_path = _run_plan_path(config)
    parity_report_path = _parity_report_path(config)
    blockers: list[str] = []
    warnings: list[str] = []
    already_complete = _already_complete(config)

    _validate_required_inputs(config, blockers)
    if blockers:
        report = _production_report(
            config,
            created_at=created_at,
            completed_at=_now(),
            status="fail",
            blockers=blockers,
            warnings=warnings,
            doctor_report={},
            run_plan_path=run_plan_path,
            run_plan={},
            effective_batch_size=None,
            already_complete=already_complete,
            parity_report_path=parity_report_path,
            parity_status="not_run",
        )
        write_production_build_report(report, report_path)
        return report

    output_has_artifact = _has_existing_artifact(config)
    if output_has_artifact and not (config.resume or config.overwrite):
        blockers.append("output exists; use --resume or --overwrite")
        report = _production_report(
            config,
            created_at=created_at,
            completed_at=_now(),
            status="fail",
            blockers=blockers,
            warnings=warnings,
            doctor_report={},
            run_plan_path=run_plan_path,
            run_plan={},
            effective_batch_size=None,
            already_complete=already_complete,
            parity_report_path=parity_report_path,
            parity_status="not_run",
        )
        write_production_build_report(report, report_path)
        return report
    if already_complete:
        validation = validate_teacher_textbook(config.output_dir)
        if validation.status == "pass":
            report = _production_report(
                config,
                created_at=created_at,
                completed_at=_now(),
                status="pass",
                blockers=[],
                warnings=[],
                doctor_report={},
                run_plan_path=run_plan_path,
                run_plan={"status": "not_run"},
                effective_batch_size=None,
                already_complete=True,
                parity_report_path=parity_report_path,
                parity_status="not_run",
                validation_status="pass",
                build_status="already_complete",
            )
            write_production_build_report(report, report_path)
            return report
        report = _production_report(
            config,
            created_at=created_at,
            completed_at=_now(),
            status="fail",
            blockers=list(validation.blockers),
            warnings=list(validation.warnings),
            doctor_report={},
            run_plan_path=run_plan_path,
            run_plan={"status": "not_run"},
            effective_batch_size=None,
            already_complete=True,
            parity_report_path=parity_report_path,
            parity_status="not_run",
            validation_status=validation.status,
            build_status="already_complete_invalid",
        )
        write_production_build_report(report, report_path)
        return report
    if output_has_artifact and config.overwrite:
        shutil.rmtree(config.output_dir)

    backend_config = _backend_config(config)
    doctor_report = build_runtime_doctor_report(backend_config)
    plan = build_gpu_run_plan(
        GPURunPlanConfig(
            backend_config=backend_config,
            dataset_path=config.dataset_path,
            corpus_manifest_path=config.corpus_manifest_path,
            teacher_model_provenance_path=config.teacher_model_provenance_path,
            max_examples=config.max_examples,
            strict_provenance=config.strict_provenance,
            max_artifact_bytes=config.max_artifact_bytes,
            fail_on_warnings=config.fail_on_plan_warnings,
        )
    )
    write_gpu_run_plan(plan, run_plan_path)
    warnings.extend(str(item) for item in plan.get("warnings", ()))
    run_plan_status = str(plan.get("status"))
    if run_plan_status == "fail":
        blockers.extend(str(item) for item in plan.get("blockers", ()))
    if run_plan_status == "warn" and config.no_build_if_plan_warn:
        blockers.append("run plan status is warn and no_build_if_plan_warn is enabled")
    effective_batch_size = _effective_batch_size(plan)
    if effective_batch_size is None:
        blockers.append("run plan did not select an effective batch size")
    if blockers:
        report = _production_report(
            config,
            created_at=created_at,
            completed_at=_now(),
            status="fail",
            blockers=blockers,
            warnings=warnings,
            doctor_report=doctor_report,
            run_plan_path=run_plan_path,
            run_plan=plan,
            effective_batch_size=effective_batch_size,
            already_complete=already_complete,
            parity_report_path=parity_report_path,
            parity_status="not_run",
        )
        write_production_build_report(report, report_path)
        return report

    try:
        build_report = build_streaming_backend_teacher_textbook(
            _streaming_config(config, effective_batch_size)
        )
    except Exception as exc:
        blockers.append(str(exc))
        report = _production_report(
            config,
            created_at=created_at,
            completed_at=_now(),
            status="fail",
            blockers=blockers,
            warnings=warnings,
            doctor_report=doctor_report,
            run_plan_path=run_plan_path,
            run_plan=plan,
            effective_batch_size=effective_batch_size,
            already_complete=already_complete,
            parity_report_path=parity_report_path,
            parity_status="not_run",
        )
        write_production_build_report(report, report_path)
        return report

    validation = validate_teacher_textbook(config.output_dir)
    write_cover_page(config.output_dir)
    parity_status = "not_run"
    if config.parity_left is not None and validation.status == "pass":
        parity_report = compare_tome_artifacts(
            config.parity_left,
            config.output_dir,
            TomeParityConfig(max_examples=config.max_examples),
            left_label="baseline",
            right_label="production",
        )
        write_tome_parity_report(parity_report, parity_report_path)
        parity_status = parity_report.status
        if parity_report.status == "fail":
            blockers.extend(parity_report.blockers)
        elif parity_report.status == "warn":
            warnings.extend(parity_report.warnings)

    if validation.status != "pass":
        blockers.extend(validation.blockers)
    status = "fail" if blockers else "warn" if warnings else "pass"
    report = _production_report(
        config,
        created_at=created_at,
        completed_at=_now(),
        status=status,
        blockers=blockers,
        warnings=warnings,
        doctor_report=doctor_report,
        run_plan_path=run_plan_path,
        run_plan=plan,
        effective_batch_size=effective_batch_size,
        already_complete=already_complete,
        parity_report_path=parity_report_path,
        parity_status=parity_status,
        validation_status=validation.status,
        build_status=getattr(build_report, "status", None),
    )
    write_production_build_report(report, report_path)
    return report


def write_production_build_report(report: dict[str, Any], path: Path) -> None:
    write_json(path, report)


def render_production_build_summary(report: dict[str, Any]) -> list[str]:
    return [
        (
            f"status={report.get('status')} output={report.get('output_dir')} "
            f"effective_batch_size={report.get('effective_batch_size')}"
        ),
        f"run_plan_status={report.get('run_plan_status')}",
        f"validation_status={report.get('validation_status')}",
        f"parity_status={report.get('parity_status')}",
        f"already_complete={str(report.get('already_complete')).lower()}",
        f"warnings={len(report.get('warnings', ()) or ())}",
        f"blockers={len(report.get('blockers', ()) or ())}",
    ]


def _validate_required_inputs(
    config: ProductionBuildConfig,
    blockers: list[str],
) -> None:
    for label, path in (
        ("dataset", config.dataset_path),
        ("corpus manifest", config.corpus_manifest_path),
        ("teacher model provenance", config.teacher_model_provenance_path),
    ):
        if not path.is_file():
            blockers.append(f"{label} path missing: {path}")
    if not blockers:
        corpus_report = validate_corpus_artifact(config.corpus_manifest_path.parent)
        blockers.extend(
            f"corpus manifest invalid: {item}" for item in corpus_report.blockers
        )
        teacher_report = validate_teacher_model_provenance(
            config.teacher_model_provenance_path
        )
        blockers.extend(
            f"teacher model provenance invalid: {item}"
            for item in teacher_report.blockers
        )


def _backend_config(config: ProductionBuildConfig) -> TeacherBackendConfig:
    tokenizer_id = config.tokenizer_id or config.teacher_model
    return TeacherBackendConfig(
        backend_id=config.teacher_backend,
        runtime_mode=config.runtime_mode,  # type: ignore[arg-type]
        target_policy=config.target_policy,  # type: ignore[arg-type]
        model_id=config.teacher_model,
        tokenizer_id=tokenizer_id,
        sequence_length=config.sequence_length,
        batch_size=1,
        vocab_size=config.vocab_size,
        top_k=config.top_k,
        num_buckets=config.num_buckets,
        gpu_batch_size_mode=config.gpu_batch_size_mode,
        gpu_batch_size_preset=config.gpu_batch_size_preset,
        gpu_batch_size_custom=config.gpu_batch_size_custom,
        gpu_batch_size_auto_min=config.gpu_batch_size_auto_min,
        gpu_batch_size_auto_max=config.gpu_batch_size_auto_max,
        local_files_only=True,
        allow_downloads=False,
        fallback_policy="error",
    )


def _streaming_config(
    config: ProductionBuildConfig,
    effective_batch_size: int,
) -> BackendTeacherTextbookBuildConfig:
    return BackendTeacherTextbookBuildConfig(
        output_dir=config.output_dir,
        dataset_path=config.dataset_path,
        teacher_backend=config.teacher_backend,
        runtime_mode=config.runtime_mode,
        target_policy=config.target_policy,
        teacher_model_id=config.teacher_model,
        tokenizer_id=config.tokenizer_id or config.teacher_model,
        sequence_length=config.sequence_length,
        batch_size=effective_batch_size,
        max_examples=config.max_examples,
        vocab_size=config.vocab_size,
        top_k=config.top_k,
        num_buckets=config.num_buckets,
        gpu_batch_size_mode=config.gpu_batch_size_mode,
        gpu_batch_size_preset=config.gpu_batch_size_preset,
        gpu_batch_size_custom=config.gpu_batch_size_custom,
        gpu_batch_size_auto_min=config.gpu_batch_size_auto_min,
        gpu_batch_size_auto_max=config.gpu_batch_size_auto_max,
        fallback_policy="error",
        local_files_only=True,
        allow_downloads=False,
        overwrite=False,
        corpus_manifest_path=config.corpus_manifest_path,
        teacher_model_provenance_path=config.teacher_model_provenance_path,
        streaming=True,
        resume=config.resume,
        shard_size_examples=config.shard_size_examples,
        progress_log_path=config.progress_log_path,
        run_manifest_path=config.run_manifest_path,
    )


def _production_report(
    config: ProductionBuildConfig,
    *,
    created_at: str,
    completed_at: str,
    status: str,
    blockers: list[str],
    warnings: list[str],
    doctor_report: dict[str, Any],
    run_plan_path: Path,
    run_plan: dict[str, Any],
    effective_batch_size: int | None,
    already_complete: bool,
    parity_report_path: Path,
    parity_status: str,
    validation_status: str | None = None,
    build_status: str | None = None,
) -> dict[str, Any]:
    artifact_summary = _artifact_summary(config.output_dir, run_plan)
    return {
        "schema_version": PRODUCTION_BUILD_REPORT_SCHEMA,
        "status": status,
        "blockers": blockers,
        "warnings": warnings,
        "created_at": created_at,
        "completed_at": completed_at,
        "command": "production-build",
        "output_dir": str(config.output_dir),
        "inputs": _inputs(config),
        "doctor_summary": _doctor_summary(doctor_report),
        "run_plan_path": str(run_plan_path),
        "run_plan_status": run_plan.get("status"),
        "effective_batch_size": effective_batch_size,
        "streaming_build": True,
        "resume_requested": config.resume,
        "already_complete": already_complete,
        "run_manifest_path": str(_run_manifest_path(config)),
        "progress_log_path": str(_progress_log_path(config)),
        "validation_report_path": str(config.output_dir / "validation_report.json"),
        "validation_status": validation_status or _validation_status(config.output_dir),
        "cover_page_path": str(config.output_dir / "cover_page.json"),
        "parity_report_path": str(parity_report_path) if config.parity_left else None,
        "parity_status": parity_status,
        "build_status": build_status,
        "artifact_summary": artifact_summary,
        "claims_not_made": {
            "no_model_download": True,
            "no_network_verification": True,
            "no_silent_cpu_fallback": True,
            "no_multidevice_scheduling": True,
            "no_tpu_jax": True,
            "no_unverified_parity_claim": config.parity_left is None,
        },
    }


def _inputs(config: ProductionBuildConfig) -> dict[str, Any]:
    return {
        "teacher_model": config.teacher_model,
        "tokenizer_id": config.tokenizer_id or config.teacher_model,
        "dataset": str(config.dataset_path),
        "corpus_manifest": str(config.corpus_manifest_path),
        "teacher_model_provenance": str(config.teacher_model_provenance_path),
        "teacher_backend": config.teacher_backend,
        "runtime_mode": config.runtime_mode,
        "target_policy": config.target_policy,
        "sequence_length": config.sequence_length,
        "vocab_size": config.vocab_size,
        "top_k": config.top_k,
        "num_buckets": config.num_buckets,
        "allow_downloads": False,
        "local_files_only": True,
    }


def _doctor_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report.get(key)
        for key in (
            "can_emit",
            "failure_stage",
            "failure_reason",
            "dependency_status",
            "torch_available",
            "transformers_available",
            "cuda_available",
            "mps_available",
        )
    }


def _artifact_summary(output_dir: Path, run_plan: dict[str, Any]) -> dict[str, Any]:
    metadata_path = output_dir / "metadata.json"
    metadata = read_json_object(metadata_path) if metadata_path.is_file() else {}
    artifact_estimates = run_plan.get("artifact_estimates", {})
    if not isinstance(artifact_estimates, dict):
        artifact_estimates = {}
    return {
        "num_examples": metadata.get("num_examples"),
        "shard_count": metadata.get("shard_count"),
        "target_type": metadata.get("target_type"),
        "dtype": metadata.get("dtype"),
        "estimated_artifact_bytes": artifact_estimates.get(
            "estimated_total_artifact_bytes"
        ),
        "actual_shard_bytes": _actual_shard_bytes(output_dir),
    }


def _actual_shard_bytes(output_dir: Path) -> int | None:
    shards_dir = output_dir / "shards"
    if not shards_dir.is_dir():
        return None
    return sum(path.stat().st_size for path in shards_dir.glob("shard-*.npz"))


def _effective_batch_size(plan: dict[str, Any]) -> int | None:
    resolved = plan.get("resolved_batch_policy", {})
    if not isinstance(resolved, dict):
        return None
    value = resolved.get("effective_gpu_batch_size")
    return int(value) if value is not None else None


def _validation_status(output_dir: Path) -> str | None:
    report_path = output_dir / "validation_report.json"
    if not report_path.is_file():
        return None
    return read_json_object(report_path).get("status")


def _already_complete(config: ProductionBuildConfig) -> bool:
    manifest_path = _run_manifest_path(config)
    if not config.resume or not manifest_path.is_file():
        return False
    return read_json_object(manifest_path).get("status") == "complete"


def _has_existing_artifact(config: ProductionBuildConfig) -> bool:
    output_dir = config.output_dir
    return any(
        path.exists()
        for path in (
            output_dir / "metadata.json",
            _run_manifest_path(config),
            output_dir / "shards",
        )
    )


def _run_plan_path(config: ProductionBuildConfig) -> Path:
    return config.run_plan_path or config.output_dir / "run_plan.json"


def _run_manifest_path(config: ProductionBuildConfig) -> Path:
    return config.run_manifest_path or config.output_dir / "run_manifest.json"


def _progress_log_path(config: ProductionBuildConfig) -> Path:
    return config.progress_log_path or config.output_dir / "progress_log.jsonl"


def _production_report_path(config: ProductionBuildConfig) -> Path:
    return (
        config.production_report_path
        or config.output_dir / PRODUCTION_BUILD_REPORT_FILENAME
    )


def _parity_report_path(config: ProductionBuildConfig) -> Path:
    return config.parity_report_path or config.output_dir / "parity_report.json"


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
