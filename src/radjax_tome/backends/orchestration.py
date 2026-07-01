from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from radjax_tome.backends.base import (
    CpuOrchestrationMode,
    TeacherBatchInput,
    TeacherEmissionBackend,
    TeacherEmissionResult,
)

_ORCHESTRATION_MODES = {"auto", "serial", "staged"}


@dataclass(frozen=True)
class BackendBatchEnvelope:
    sequence_id: int
    batch: TeacherBatchInput
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sequence_id < 0:
            raise ValueError("sequence_id must be >= 0")


@dataclass(frozen=True)
class BackendRunConfig:
    cpu_orchestration_mode: CpuOrchestrationMode = "auto"
    auto_serial_max_batches: int = 1
    auto_serial_max_examples: int = 8
    max_workers: int = 1
    queue_depth: int = 1

    def __post_init__(self) -> None:
        if self.cpu_orchestration_mode not in _ORCHESTRATION_MODES:
            raise ValueError("cpu_orchestration_mode must be auto, serial, or staged")
        if self.auto_serial_max_batches < 0:
            raise ValueError("auto_serial_max_batches must be >= 0")
        if self.auto_serial_max_examples < 0:
            raise ValueError("auto_serial_max_examples must be >= 0")
        if self.max_workers <= 0:
            raise ValueError("max_workers must be > 0")
        if self.queue_depth <= 0:
            raise ValueError("queue_depth must be > 0")


@dataclass(frozen=True)
class BackendRunResult:
    requested_cpu_orchestration_mode: CpuOrchestrationMode
    effective_cpu_orchestration_mode: CpuOrchestrationMode
    results: tuple[tuple[int, TeacherEmissionResult], ...]
    metadata: Mapping[str, object]


def run_backend_batches(
    backend: TeacherEmissionBackend,
    batches: Iterable[BackendBatchEnvelope],
    config: BackendRunConfig | None = None,
) -> BackendRunResult:
    run_config = config or BackendRunConfig()
    envelopes = tuple(batches)
    _validate_unique_sequence_ids(envelopes)
    example_count = sum(len(envelope.batch.texts) for envelope in envelopes)
    effective_mode, auto_reason = _resolve_mode(
        run_config,
        batch_count=len(envelopes),
        example_count=example_count,
    )
    if effective_mode == "serial":
        results = _run_serial(backend, envelopes)
    elif effective_mode == "staged":
        results = _run_staged(backend, envelopes, run_config)
    else:
        raise ValueError(f"unsupported cpu_orchestration_mode: {effective_mode}")
    ordered_results = tuple(sorted(results, key=lambda item: item[0]))
    sequence_ids = tuple(envelope.sequence_id for envelope in envelopes)
    output_ids = tuple(sequence_id for sequence_id, _ in ordered_results)
    metadata = {
        "requested_cpu_orchestration_mode": run_config.cpu_orchestration_mode,
        "effective_cpu_orchestration_mode": effective_mode,
        "batch_count": len(envelopes),
        "example_count": example_count,
        "sequence_id_min": min(sequence_ids) if sequence_ids else None,
        "sequence_id_max": max(sequence_ids) if sequence_ids else None,
        "input_order_was_sorted": sequence_ids == tuple(sorted(sequence_ids)),
        "output_order_sorted": output_ids == tuple(sorted(output_ids)),
        "auto_reason": auto_reason,
        "runner_kind": "backend_batch_orchestrator_v1",
        "max_workers": run_config.max_workers,
        "queue_depth": run_config.queue_depth,
        "staged_is_performance_optimized": False,
    }
    return BackendRunResult(
        requested_cpu_orchestration_mode=run_config.cpu_orchestration_mode,
        effective_cpu_orchestration_mode=effective_mode,
        results=ordered_results,
        metadata=metadata,
    )


def _validate_unique_sequence_ids(envelopes: tuple[BackendBatchEnvelope, ...]) -> None:
    seen: set[int] = set()
    duplicates: set[int] = set()
    for envelope in envelopes:
        if envelope.sequence_id in seen:
            duplicates.add(envelope.sequence_id)
        seen.add(envelope.sequence_id)
    if duplicates:
        duplicate_list = ", ".join(str(item) for item in sorted(duplicates))
        raise ValueError(f"duplicate sequence_id values: {duplicate_list}")


def _resolve_mode(
    config: BackendRunConfig,
    *,
    batch_count: int,
    example_count: int,
) -> tuple[CpuOrchestrationMode, str]:
    if config.cpu_orchestration_mode == "serial":
        return "serial", "requested_serial"
    if config.cpu_orchestration_mode == "staged":
        return "staged", "requested_staged"
    if batch_count <= config.auto_serial_max_batches:
        return "serial", "tiny_batch_count"
    if example_count <= config.auto_serial_max_examples:
        return "serial", "tiny_example_count"
    return "staged", "normal_workload"


def _run_serial(
    backend: TeacherEmissionBackend,
    envelopes: tuple[BackendBatchEnvelope, ...],
) -> tuple[tuple[int, TeacherEmissionResult], ...]:
    return tuple(
        (envelope.sequence_id, backend.emit_batch(envelope.batch))
        for envelope in envelopes
    )


def _run_staged(
    backend: TeacherEmissionBackend,
    envelopes: tuple[BackendBatchEnvelope, ...],
    config: BackendRunConfig,
) -> tuple[tuple[int, TeacherEmissionResult], ...]:
    # Spec 3.3D establishes the staged control-flow shape and metadata without
    # claiming historical throughput optimization or adding queue machinery.
    staged_inputs = tuple(envelopes[: config.queue_depth]) + tuple(
        envelopes[config.queue_depth :]
    )
    return tuple(
        (envelope.sequence_id, backend.emit_batch(envelope.batch))
        for envelope in staged_inputs
    )
