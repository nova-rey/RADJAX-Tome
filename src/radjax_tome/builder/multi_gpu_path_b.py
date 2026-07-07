from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from radjax_tome.corpora import corpus_provenance_from_manifest
from radjax_tome.io.json import read_json_object, write_json
from radjax_tome.provenance import (
    sha256_file,
    teacher_model_provenance_summary,
    validate_teacher_model_provenance,
)

MULTI_GPU_PATH_B_REPORT_FILENAME = "multi_gpu_path_b_report.json"
MULTI_GPU_PATH_B_REPORT_SCHEMA = "multi_gpu_path_b_report_v1"
MULTI_GPU_WORKER_MANIFEST_FILENAME = "multi_gpu_worker_manifest.json"
MULTI_GPU_WORKER_MANIFEST_SCHEMA = "multi_gpu_worker_manifest_v1"
PATH_B_CANDIDATE_RECORD_SCHEMA = "path_b_candidate_record_v1"
EXPERIMENTAL_WARNING = (
    "experimental multi-GPU Path B scheduling is enabled; single-GPU "
    "production-build remains the recommended path."
)
ASSIGNMENT_POLICY = "round_robin_shards_v1"
DEVICE_POLICY = "explicit_devices_only_v1"
MERGE_POLICY = "score_desc_tie_break_key_example_position_assignment_worker_v1"
FAKE_EXECUTION_MODE = "fake_for_scheduler_test"
REAL_EXECUTION_BLOCKER = (
    "real multi-GPU backend candidate execution is not implemented in 4.7.a; "
    "use --fake-workers for the scheduler harness"
)


@dataclass(frozen=True)
class MultiGPUPathBConfig:
    teacher_model: str
    dataset_path: Path
    corpus_manifest_path: Path
    teacher_model_provenance_path: Path
    output_dir: Path
    devices: tuple[str, ...] | str
    tokenizer_id: str | None = None
    target_policy: str = "corridor_exemplar_v1"
    sequence_length: int = 128
    batch_size_per_device: int = 4
    shard_size_examples: int = 1024
    max_examples: int | None = None
    top_k: int = 8
    num_buckets: int = 4
    resume: bool = False
    overwrite: bool = False
    dry_run: bool = False
    fake_workers: bool = False


def run_multi_gpu_path_b_candidate_harness(
    config: MultiGPUPathBConfig,
) -> dict[str, Any]:
    created_at = _now()
    output_dir = config.output_dir
    report_path = output_dir / MULTI_GPU_PATH_B_REPORT_FILENAME
    manifest_path = output_dir / MULTI_GPU_WORKER_MANIFEST_FILENAME
    blockers: list[str] = []
    warnings = [EXPERIMENTAL_WARNING]
    selected_devices = normalize_multi_gpu_devices(config.devices, blockers)
    _validate_config(config, selected_devices, blockers)

    if output_dir.exists() and config.overwrite and not config.resume:
        _remove_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_tmp_files(output_dir)

    corpus_provenance = _corpus_provenance(config.corpus_manifest_path, blockers)
    teacher_provenance = _teacher_provenance(
        config.teacher_model_provenance_path,
        blockers,
    )
    examples = _read_examples(config.dataset_path, config.max_examples, blockers)
    assignments = _build_assignments(
        examples,
        selected_devices,
        shard_size_examples=config.shard_size_examples,
    )
    existing_manifest = _read_existing_manifest(manifest_path) if config.resume else {}
    completed_by_id = _completed_assignment_by_id(existing_manifest)
    completed_assignments: list[dict[str, Any]] = []
    failed_assignments: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    candidate_execution_mode = FAKE_EXECUTION_MODE

    if not config.fake_workers and not config.dry_run:
        blockers.append(REAL_EXECUTION_BLOCKER)

    if blockers:
        report = _report(
            config,
            created_at=created_at,
            completed_at=_now(),
            status="fail",
            blockers=blockers,
            warnings=warnings,
            selected_devices=selected_devices,
            assignments=assignments,
            completed_assignments=[],
            failed_assignments=[],
            candidate_records=[],
            corpus_provenance=corpus_provenance,
            teacher_provenance=teacher_provenance,
            candidate_execution_mode=candidate_execution_mode,
        )
        _write_manifest(
            config,
            created_at=created_at,
            status="failed",
            selected_devices=selected_devices,
            assignments=assignments,
            completed_assignments=[],
            failed_assignments=[],
            corpus_provenance=corpus_provenance,
            teacher_provenance=teacher_provenance,
            candidate_execution_mode=candidate_execution_mode,
        )
        write_json(report_path, report)
        return report

    for assignment in assignments:
        assignment_id = str(assignment["assignment_id"])
        existing = completed_by_id.get(assignment_id)
        if existing and _completed_output_is_valid(config.output_dir, existing):
            completed = {**assignment, **existing, "status": "complete"}
            completed_assignments.append(completed)
            candidate_records.extend(
                _read_jsonl_records(config.output_dir / existing["output_path"])
            )
            continue
        if config.dry_run:
            completed = {
                **assignment,
                "status": "planned",
                "output_path": None,
                "sha256": None,
                "completed_at": None,
            }
            completed_assignments.append(completed)
            continue
        try:
            records = _fake_candidate_records(config, assignment, examples)
            output_path = _candidate_output_path(assignment)
            absolute_output_path = config.output_dir / output_path
            _write_jsonl(absolute_output_path, records)
            completed = {
                **assignment,
                "status": "complete",
                "output_path": output_path.as_posix(),
                "sha256": sha256_file(absolute_output_path),
                "completed_at": _now(),
            }
            completed_assignments.append(completed)
            candidate_records.extend(records)
        except Exception as exc:  # pragma: no cover - defensive only
            failed_assignments.append(
                {
                    **assignment,
                    "status": "failed",
                    "failure_reason": str(exc),
                    "completed_at": _now(),
                }
            )

    _write_worker_sidecars(config.output_dir, selected_devices, completed_assignments)
    merged = merge_path_b_candidate_records(candidate_records)
    merged_path = config.output_dir / "merged_candidates.jsonl"
    _write_jsonl(merged_path, merged["records"])
    status = "fail" if failed_assignments else "warn" if config.dry_run else "pass"
    _write_manifest(
        config,
        created_at=str(existing_manifest.get("created_at") or created_at),
        status="complete" if not failed_assignments else "failed",
        selected_devices=selected_devices,
        assignments=assignments,
        completed_assignments=completed_assignments,
        failed_assignments=failed_assignments,
        corpus_provenance=corpus_provenance,
        teacher_provenance=teacher_provenance,
        candidate_execution_mode=candidate_execution_mode,
    )
    report = _report(
        config,
        created_at=created_at,
        completed_at=_now(),
        status=status,
        blockers=[],
        warnings=warnings,
        selected_devices=selected_devices,
        assignments=assignments,
        completed_assignments=completed_assignments,
        failed_assignments=failed_assignments,
        candidate_records=candidate_records,
        corpus_provenance=corpus_provenance,
        teacher_provenance=teacher_provenance,
        candidate_execution_mode=candidate_execution_mode,
        merged_leaderboard_path=merged_path,
    )
    write_json(report_path, report)
    return report


def normalize_multi_gpu_devices(
    devices: tuple[str, ...] | list[str] | str,
    blockers: list[str] | None = None,
) -> tuple[str, ...]:
    blockers = blockers if blockers is not None else []
    raw_items: list[str]
    if isinstance(devices, str):
        raw_items = devices.split(",")
    else:
        raw_items = []
        for item in devices:
            raw_items.extend(str(item).split(","))
    normalized: list[str] = []
    for raw_item in raw_items:
        item = raw_item.strip()
        if not item:
            continue
        if item.isdigit():
            item = f"cuda:{item}"
        if not item.startswith("cuda:") or not item.removeprefix("cuda:").isdigit():
            blockers.append(f"unsupported multi-GPU device identifier: {raw_item}")
            continue
        if item not in normalized:
            normalized.append(item)
    if not normalized:
        blockers.append("explicit --devices list is required")
    return tuple(normalized)


def build_path_b_assignments(
    *,
    num_examples_effective: int,
    shard_size_examples: int,
    devices: tuple[str, ...],
) -> list[dict[str, Any]]:
    if shard_size_examples < 1:
        raise ValueError("shard_size_examples must be positive")
    if not devices:
        raise ValueError("at least one device is required")
    assignments = []
    shard_id = 0
    for start in range(0, num_examples_effective, shard_size_examples):
        end = min(start + shard_size_examples, num_examples_effective)
        worker_id = shard_id % len(devices)
        assignments.append(
            {
                "assignment_id": f"assignment-{shard_id:05d}",
                "worker_id": f"worker-{worker_id:03d}",
                "device": devices[worker_id],
                "shard_id": shard_id,
                "example_start_index": start,
                "example_end_index_exclusive": end,
                "num_examples": end - start,
                "status": "planned",
            }
        )
        shard_id += 1
    return assignments


def merge_path_b_candidate_records(
    records: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    ordered = sorted(
        (dict(record) for record in records),
        key=lambda record: (
            -float(record.get("score", 0.0)),
            str(record.get("tie_break_key", "")),
            int(record.get("example_index", 0)),
            int(record.get("position_index", 0)),
            str(record.get("assignment_id", "")),
            str(record.get("worker_id", "")),
        ),
    )
    return {
        "schema_version": "path_b_candidate_merge_v1",
        "deterministic_merge_policy": MERGE_POLICY,
        "candidate_record_count": len(ordered),
        "records": ordered,
    }


def render_multi_gpu_path_b_summary(report: dict[str, Any]) -> list[str]:
    return [
        (
            f"status={report.get('status')} output={report.get('output_dir')} "
            f"assignments_completed={report.get('assignments_completed')}"
        ),
        f"experimental={str(report.get('experimental')).lower()}",
        f"recommended_path={report.get('recommended_path')}",
        f"candidate_execution_mode={report.get('candidate_execution_mode')}",
        f"device_count={report.get('device_count')}",
        f"candidate_record_count={report.get('candidate_record_count')}",
        f"warnings={len(report.get('warnings', ()) or ())}",
        f"blockers={len(report.get('blockers', ()) or ())}",
    ]


def _validate_config(
    config: MultiGPUPathBConfig,
    selected_devices: tuple[str, ...],
    blockers: list[str],
) -> None:
    if config.target_policy != "corridor_exemplar_v1":
        blockers.append("multi-gpu Path B currently supports corridor_exemplar_v1 only")
    if config.batch_size_per_device < 1:
        blockers.append("batch_size_per_device must be positive")
    if config.shard_size_examples < 1:
        blockers.append("shard_size_examples must be positive")
    if config.dataset_path.is_file() is False:
        blockers.append(f"dataset path missing: {config.dataset_path}")
    if config.corpus_manifest_path.is_file() is False:
        blockers.append(f"corpus manifest path missing: {config.corpus_manifest_path}")
    if config.teacher_model_provenance_path.is_file() is False:
        blockers.append(
            "teacher model provenance path missing: "
            f"{config.teacher_model_provenance_path}"
        )
    if config.output_dir.exists() and not (config.resume or config.overwrite):
        blockers.append("output exists; use --resume or --overwrite")
    if selected_devices and not config.fake_workers and not config.dry_run:
        blockers.extend(_device_availability_blockers(selected_devices))


def _device_availability_blockers(devices: tuple[str, ...]) -> list[str]:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception:
        return ["torch is required for real multi-GPU candidate execution"]
    if not bool(torch.cuda.is_available()):
        return ["CUDA is unavailable for requested real multi-GPU devices"]
    device_count = int(torch.cuda.device_count())
    blockers = []
    for device in devices:
        index = int(device.removeprefix("cuda:"))
        if index >= device_count:
            blockers.append(f"requested CUDA device unavailable: {device}")
    return blockers


def _remove_output_dir(path: Path) -> None:
    import shutil

    shutil.rmtree(path)


def _remove_stale_tmp_files(output_dir: Path) -> None:
    workers_dir = output_dir / "workers"
    if not workers_dir.is_dir():
        return
    for tmp_path in workers_dir.rglob("*.tmp"):
        tmp_path.unlink()


def _corpus_provenance(path: Path, blockers: list[str]) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "missing", "corpus_manifest_path": str(path)}
    try:
        return {"status": "pass", **corpus_provenance_from_manifest(path)}
    except Exception as exc:
        blockers.append(f"corpus manifest invalid: {exc}")
        return {"status": "fail", "corpus_manifest_path": str(path)}


def _teacher_provenance(path: Path, blockers: list[str]) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "missing", "teacher_model_provenance_path": str(path)}
    report = validate_teacher_model_provenance(path)
    if report.status == "fail":
        blockers.extend(
            f"teacher model provenance invalid: {item}" for item in report.blockers
        )
        return {"status": "fail", "teacher_model_provenance_path": str(path)}
    return {"status": report.status, **teacher_model_provenance_summary(path)}


def _read_examples(
    path: Path,
    max_examples: int | None,
    blockers: list[str],
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    examples = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if max_examples is not None and len(examples) >= max_examples:
                break
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                blockers.append(f"dataset JSONL row {index} invalid: {exc}")
                continue
            if not isinstance(payload, dict):
                blockers.append(f"dataset JSONL row {index} must be an object")
                continue
            example_id = str(payload.get("example_id") or f"example-{index}")
            examples.append(
                {
                    "example_id": example_id,
                    "example_index": index,
                    "text": str(payload.get("text") or ""),
                }
            )
    return examples


def _build_assignments(
    examples: list[dict[str, Any]],
    selected_devices: tuple[str, ...],
    *,
    shard_size_examples: int,
) -> list[dict[str, Any]]:
    if not selected_devices or shard_size_examples < 1:
        return []
    return build_path_b_assignments(
        num_examples_effective=len(examples),
        shard_size_examples=shard_size_examples,
        devices=selected_devices,
    )


def _read_existing_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return read_json_object(path)
    except Exception:
        return {}


def _completed_assignment_by_id(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = manifest.get("completed_assignments", ())
    if not isinstance(records, list):
        return {}
    return {
        str(record.get("assignment_id")): dict(record)
        for record in records
        if isinstance(record, dict)
    }


def _completed_output_is_valid(output_dir: Path, assignment: dict[str, Any]) -> bool:
    output_path_value = assignment.get("output_path")
    expected_hash = assignment.get("sha256")
    if not output_path_value or not expected_hash:
        return False
    output_path = output_dir / str(output_path_value)
    return output_path.is_file() and sha256_file(output_path) == expected_hash


def _candidate_output_path(assignment: dict[str, Any]) -> Path:
    return (
        Path("workers")
        / str(assignment["worker_id"])
        / f"{assignment['assignment_id']}-candidates.jsonl"
    )


def _fake_candidate_records(
    config: MultiGPUPathBConfig,
    assignment: dict[str, Any],
    examples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    start = int(assignment["example_start_index"])
    end = int(assignment["example_end_index_exclusive"])
    records = []
    for example in examples[start:end]:
        example_index = int(example["example_index"])
        score = round(1.0 / (example_index + 1), 12)
        records.append(
            {
                "schema_version": PATH_B_CANDIDATE_RECORD_SCHEMA,
                "assignment_id": assignment["assignment_id"],
                "worker_id": assignment["worker_id"],
                "device": assignment["device"],
                "shard_id": assignment["shard_id"],
                "example_id": example["example_id"],
                "example_index": example_index,
                "position_index": 0,
                "score": score,
                "score_source_policy": "fake_entropy_score_v1",
                "candidate_payload": {
                    "candidate_execution_mode": FAKE_EXECUTION_MODE,
                    "sequence_length": config.sequence_length,
                    "top_k": config.top_k,
                    "num_buckets": config.num_buckets,
                },
                "tie_break_key": (
                    f"{example_index:012d}:000000:{assignment['assignment_id']}:"
                    f"{assignment['worker_id']}"
                ),
            }
        )
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_worker_sidecars(
    output_dir: Path,
    devices: tuple[str, ...],
    completed_assignments: list[dict[str, Any]],
) -> None:
    for worker_index, device in enumerate(devices):
        worker_id = f"worker-{worker_index:03d}"
        worker_dir = output_dir / "workers" / worker_id
        write_json(
            worker_dir / "device.json",
            {
                "worker_id": worker_id,
                "device": device,
                "device_policy": DEVICE_POLICY,
            },
        )
        worker_assignments = [
            assignment
            for assignment in completed_assignments
            if assignment.get("worker_id") == worker_id
        ]
        _write_jsonl(worker_dir / "assignments.jsonl", worker_assignments)


def _write_manifest(
    config: MultiGPUPathBConfig,
    *,
    created_at: str,
    status: str,
    selected_devices: tuple[str, ...],
    assignments: list[dict[str, Any]],
    completed_assignments: list[dict[str, Any]],
    failed_assignments: list[dict[str, Any]],
    corpus_provenance: dict[str, Any],
    teacher_provenance: dict[str, Any],
    candidate_execution_mode: str,
) -> None:
    manifest = {
        "schema_version": MULTI_GPU_WORKER_MANIFEST_SCHEMA,
        "status": status,
        "created_at": created_at,
        "updated_at": _now(),
        "experimental": True,
        "experimental_warning": EXPERIMENTAL_WARNING,
        "dataset_path": str(config.dataset_path),
        "dataset_hash": sha256_file(config.dataset_path)
        if config.dataset_path.is_file()
        else None,
        "corpus_manifest_path": str(config.corpus_manifest_path),
        "corpus_hash": corpus_provenance.get("source_corpus_hash"),
        "teacher_model_provenance_path": str(config.teacher_model_provenance_path),
        "teacher_model_hashes": {
            "config_hash": teacher_provenance.get("config_hash"),
            "tokenizer_hash": teacher_provenance.get("tokenizer_hash"),
            "weights_hash": teacher_provenance.get("weights_hash"),
            "model_directory_hash": teacher_provenance.get("model_directory_hash"),
        },
        "target_policy": config.target_policy,
        "sequence_length": config.sequence_length,
        "batch_size_per_device": config.batch_size_per_device,
        "shard_size_examples": config.shard_size_examples,
        "device_policy": DEVICE_POLICY,
        "selected_devices": list(selected_devices),
        "assignment_policy": ASSIGNMENT_POLICY,
        "assignments": assignments,
        "completed_assignments": completed_assignments,
        "failed_assignments": failed_assignments,
        "resume_supported": True,
        "candidate_execution_mode": candidate_execution_mode,
        "claims_not_made": _claims_not_made(),
    }
    write_json(config.output_dir / MULTI_GPU_WORKER_MANIFEST_FILENAME, manifest)


def _report(
    config: MultiGPUPathBConfig,
    *,
    created_at: str,
    completed_at: str,
    status: str,
    blockers: list[str],
    warnings: list[str],
    selected_devices: tuple[str, ...],
    assignments: list[dict[str, Any]],
    completed_assignments: list[dict[str, Any]],
    failed_assignments: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    corpus_provenance: dict[str, Any],
    teacher_provenance: dict[str, Any],
    candidate_execution_mode: str,
    merged_leaderboard_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": MULTI_GPU_PATH_B_REPORT_SCHEMA,
        "status": status,
        "blockers": blockers,
        "warnings": warnings,
        "created_at": created_at,
        "completed_at": completed_at,
        "experimental": True,
        "experimental_warning": EXPERIMENTAL_WARNING,
        "recommended_path": "single_gpu_production_build",
        "output_dir": str(config.output_dir),
        "dataset": str(config.dataset_path),
        "corpus_provenance": corpus_provenance,
        "teacher_model_provenance": teacher_provenance,
        "target_policy": config.target_policy,
        "selected_devices": list(selected_devices),
        "device_count": len(selected_devices),
        "device_policy": DEVICE_POLICY,
        "batch_size_per_device": config.batch_size_per_device,
        "shard_size_examples": config.shard_size_examples,
        "assignment_policy": ASSIGNMENT_POLICY,
        "assignments_planned": len(assignments),
        "assignments_completed": len(completed_assignments),
        "assignments_failed": len(failed_assignments),
        "worker_manifest_path": str(
            config.output_dir / MULTI_GPU_WORKER_MANIFEST_FILENAME
        ),
        "candidate_record_count": len(candidate_records),
        "merged_leaderboard_path": (
            str(merged_leaderboard_path) if merged_leaderboard_path else None
        ),
        "deterministic_merge_policy": MERGE_POLICY,
        "resume_supported": True,
        "single_device_plan_path": None,
        "per_device_probe_status": "not_run",
        "planner_warning": "multi-GPU per-device planning is experimental",
        "candidate_execution_mode": candidate_execution_mode,
        "claims_not_made": _claims_not_made(),
    }


def _claims_not_made() -> dict[str, bool]:
    return {
        "no_default_multi_gpu_recommendation": True,
        "no_ddp": True,
        "no_model_parallelism": True,
        "no_combined_vram": True,
        "no_network_verification": True,
        "no_model_download": True,
        "no_tpu_jax": True,
        "no_full_multi_gpu_burn_validation": True,
    }


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
