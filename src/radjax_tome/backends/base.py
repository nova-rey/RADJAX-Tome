from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import ceil
from typing import Any, Literal, Protocol

import numpy as np

from radjax_tome.targets import TargetStoreMetadata

RuntimeMode = Literal["cpu", "cpu_gpu", "cpu_tpu"]
CpuOrchestrationMode = Literal["auto", "serial", "staged"]
TargetPolicy = Literal[
    "dense_logits",
    "topk_with_tail_v0",
    "cascaded_soft_labels_v1",
    "dynamic_cascaded_soft_labels_v1",
    "corridor_exemplar_v1",
]
SupportStatus = Literal[
    "unsupported",
    "planned",
    "supported",
    "supported_debug",
    "optimized",
    "historical_reference_exists",
]
FallbackPolicy = Literal["error", "auto"]


@dataclass(frozen=True)
class TeacherBackendConfig:
    backend_id: str
    runtime_mode: RuntimeMode = "cpu"
    cpu_orchestration_mode: CpuOrchestrationMode = "auto"
    target_policy: TargetPolicy = "dense_logits"
    model_id: str = "fake-deterministic-teacher"
    tokenizer_id: str = "fake-deterministic-tokenizer"
    sequence_length: int = 8
    batch_size: int = 1
    vocab_size: int = 32
    top_k: int = 8
    num_buckets: int = 4
    exemplar_top_n: int = 1
    exemplar_source_policy: str = "dynamic_cascaded_soft_labels_v1"
    exemplar_selection_policy: str = "entropy_top_n_v1"
    exemplar_capture_mode: str = "one_pass_candidate"
    exemplar_first_pass_score_policy: str = "entropy_score_v1"
    exemplar_second_pass_source_policy: str = "dynamic_cascaded_soft_labels_v1"
    exemplar_sparse_selection_top_n: int = 1
    exemplar_sparse_selection_fraction: float | None = None
    exemplar_auto_num_examples: int | None = None
    exemplar_auto_expected_selected_fraction: float | None = None
    exemplar_auto_available_disk_budget_bytes: int | None = None
    exemplar_auto_teacher_inference_cost_hint: str | None = None
    corridor_payload_flavor: str = "production_v1"
    dynamic_top_k_min: int = 1
    dynamic_top_k_max: int = 32
    dynamic_mass_threshold: float = 0.95
    dynamic_top_k_policy: str = "mass_threshold_v1"
    gpu_vocab_chunk_size: int | None = None
    gpu_enable_vocab_chunking: bool = False
    gpu_batch_size_mode: str = "preset"
    gpu_batch_size_preset: int = 8
    gpu_batch_size_custom: int | None = None
    gpu_batch_size_auto_min: int = 1
    gpu_batch_size_auto_max: int = 64
    gpu_batch_size_warning_threshold: int = 64
    gpu_batch_size_probe_policy: str = "exponential_probe_v1"
    gpu_batch_size_midpoint_refinement: bool = True
    local_files_only: bool = True
    allow_downloads: bool = False
    fallback_policy: FallbackPolicy = "error"
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TeacherBatchInput:
    example_ids: tuple[str, ...]
    texts: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "example_ids", tuple(self.example_ids))
        object.__setattr__(self, "texts", tuple(self.texts))
        if len(self.example_ids) != len(self.texts):
            raise ValueError("example_ids and texts must have the same length")


@dataclass(frozen=True)
class TeacherEmissionResult:
    backend_id: str
    runtime_mode: RuntimeMode
    target_policy: TargetPolicy
    input_ids: Any
    attention_mask: Any
    payload: Mapping[str, Any]
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class BackendCapability:
    backend_id: str
    backend_family: str
    runtime_mode: RuntimeMode
    target_policy: TargetPolicy
    status: SupportStatus
    optimized: bool
    implemented_now: bool
    notes: str

    def __post_init__(self) -> None:
        if self.optimized and not self.implemented_now:
            raise ValueError("optimized capabilities must also be implemented_now")


class TeacherBackend(Protocol):
    backend_id: str
    vocab_size: int

    def emit_logits(self, input_ids: np.ndarray) -> np.ndarray:
        """Emit logits shaped [batch, sequence, vocab]."""


class TeacherTargetEmitter(Protocol):
    name: str

    def build_metadata(
        self,
        *,
        num_examples: int,
        sequence_length: int,
    ) -> TargetStoreMetadata: ...

    def emit_targets(
        self,
        *,
        num_examples: int,
        sequence_length: int,
    ) -> Mapping[str, np.ndarray]: ...


class TeacherEmissionBackend(Protocol):
    backend_id: str
    backend_family: str
    runtime_mode: RuntimeMode

    def capabilities(self) -> tuple[BackendCapability, ...]: ...

    def emit_batch(self, batch: TeacherBatchInput) -> TeacherEmissionResult: ...

    def close(self) -> None: ...

    def metadata(self) -> dict[str, object]: ...


def resolve_exemplar_capture_policy(
    config: TeacherBackendConfig,
    *,
    actual_batch_size: int | None = None,
    effective_vocab_size: int | None = None,
) -> dict[str, object]:
    num_examples, missing_inputs = _exemplar_policy_num_examples(
        config,
        actual_batch_size=actual_batch_size,
    )
    expected_fraction = _exemplar_expected_selected_fraction(config)
    if config.exemplar_auto_expected_selected_fraction is None:
        missing_inputs.append("expected_selected_exemplar_fraction")
    if config.exemplar_auto_available_disk_budget_bytes is None:
        missing_inputs.append("available_disk_budget_bytes")
    if config.exemplar_auto_teacher_inference_cost_hint is None:
        missing_inputs.append("teacher_inference_cost_hint")

    estimates = _exemplar_capture_estimates(
        config,
        num_examples=num_examples,
        expected_selected_exemplar_fraction=expected_fraction,
        effective_vocab_size=effective_vocab_size,
    )
    requested = config.exemplar_capture_mode
    if requested in {"one_pass_candidate", "two_pass_sparse_exemplar"}:
        return {
            **estimates,
            "exemplar_capture_mode_requested": requested,
            "exemplar_capture_mode_effective": requested,
            "exemplar_capture_policy": "manual_exemplar_capture_policy_v1",
            "manual_override_used": True,
            "auto_policy_reason": f"manual override forced {requested}",
            "expected_selected_exemplar_fraction": expected_fraction,
            "available_disk_budget_bytes": (
                config.exemplar_auto_available_disk_budget_bytes
            ),
            "auto_policy_inputs_missing": tuple(missing_inputs),
        }
    if requested != "auto":
        raise ValueError(
            "exemplar_capture_mode must be 'auto', 'one_pass_candidate', "
            "or 'two_pass_sparse_exemplar'"
        )

    budget = config.exemplar_auto_available_disk_budget_bytes
    one_pass_bytes = estimates["estimated_one_pass_candidate_bytes"]
    token_positions = num_examples * config.sequence_length
    if budget is not None and one_pass_bytes > budget:
        effective = "two_pass_sparse_exemplar"
        reason = "one-pass candidate estimate exceeds available disk budget"
    elif one_pass_bytes >= 512 * 1024 * 1024:
        effective = "two_pass_sparse_exemplar"
        reason = "one-pass candidate estimate is very large"
    elif token_positions >= 1_000_000 and expected_fraction <= 0.25:
        effective = "two_pass_sparse_exemplar"
        reason = "large corpus with sparse expected selected exemplar fraction"
    else:
        effective = "one_pass_candidate"
        reason = "one-pass candidate estimate is small enough for default policy"
    return {
        **estimates,
        "exemplar_capture_mode_requested": requested,
        "exemplar_capture_mode_effective": effective,
        "exemplar_capture_policy": "auto_exemplar_capture_policy_v1",
        "manual_override_used": False,
        "auto_policy_reason": reason,
        "expected_selected_exemplar_fraction": expected_fraction,
        "available_disk_budget_bytes": budget,
        "auto_policy_inputs_missing": tuple(missing_inputs),
    }


GPU_BATCH_SIZE_PRESETS = (1, 2, 4, 8, 16, 32, 64)
_GPU_BATCH_SIZE_MODES = {"preset", "custom", "auto"}
_GPU_BATCH_SIZE_PROBE_POLICY = "exponential_probe_v1"
_BATCH_SIZE_WARNING_REASON = (
    "custom batch size exceeds warning threshold; may exceed accelerator memory "
    "depending on model size, vocab size, sequence length, target policy, "
    "exemplar source policy, and exemplar capture mode"
)
_TIME_PER_EXAMPLE_REGRESSION_FACTOR = 1.25
_WRITE_TIME_REGRESSION_FACTOR = 1.25


def validate_gpu_batch_size_policy_config(config: TeacherBackendConfig) -> None:
    if config.gpu_batch_size_mode not in _GPU_BATCH_SIZE_MODES:
        raise ValueError("gpu_batch_size_mode must be preset, custom, or auto")
    if config.gpu_batch_size_preset not in GPU_BATCH_SIZE_PRESETS:
        raise ValueError("gpu_batch_size_preset must be one of 1, 2, 4, 8, 16, 32, 64")
    if config.gpu_batch_size_custom is not None and config.gpu_batch_size_custom < 1:
        raise ValueError("gpu_batch_size_custom must be None or >= 1")
    if config.gpu_batch_size_mode == "custom" and config.gpu_batch_size_custom is None:
        raise ValueError("gpu_batch_size_custom must be set when mode is custom")
    if config.gpu_batch_size_auto_min < 1:
        raise ValueError("gpu_batch_size_auto_min must be >= 1")
    if config.gpu_batch_size_auto_max < config.gpu_batch_size_auto_min:
        raise ValueError("gpu_batch_size_auto_max must be >= gpu_batch_size_auto_min")
    if config.gpu_batch_size_warning_threshold < 1:
        raise ValueError("gpu_batch_size_warning_threshold must be >= 1")
    if config.gpu_batch_size_probe_policy != _GPU_BATCH_SIZE_PROBE_POLICY:
        raise ValueError("gpu_batch_size_probe_policy must be 'exponential_probe_v1'")


def gpu_batch_size_candidates(config: TeacherBackendConfig) -> tuple[int, ...]:
    candidate = config.gpu_batch_size_auto_min
    candidates = [candidate]
    while candidate < config.gpu_batch_size_auto_max:
        candidate = min(candidate * 2, config.gpu_batch_size_auto_max)
        if candidate != candidates[-1]:
            candidates.append(candidate)
    return tuple(candidates)


def resolve_gpu_batch_size_policy(
    config: TeacherBackendConfig,
    *,
    probe_results: tuple[Mapping[str, object], ...] | None = None,
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    validate_gpu_batch_size_policy_config(config)
    measured_bytes = _payload_nbytes(payload)
    measured_available = measured_bytes is not None
    base = _gpu_batch_size_base_metadata(
        config,
        measured_output_bytes=measured_bytes,
        measured_output_bytes_available=measured_available,
    )
    mode = config.gpu_batch_size_mode
    if mode == "preset":
        effective = config.gpu_batch_size_preset
        return {
            **base,
            "gpu_batch_size_mode_effective": "preset",
            "requested_gpu_batch_size": config.gpu_batch_size_preset,
            "effective_gpu_batch_size": effective,
            "gpu_batch_size_candidates_tried": (),
            "gpu_batch_size_last_good": None,
            "gpu_batch_size_failure_at": None,
            "gpu_batch_size_failure_reason": None,
            "gpu_batch_size_probe_required": False,
            "gpu_batch_size_auto_failed": False,
            "gpu_batch_size_warning_emitted": (
                effective > config.gpu_batch_size_warning_threshold
            ),
            "gpu_batch_size_warning_reason": (
                _BATCH_SIZE_WARNING_REASON
                if effective > config.gpu_batch_size_warning_threshold
                else None
            ),
            "gpu_batch_size_manual_override_used": False,
        }
    if mode == "custom":
        effective = int(config.gpu_batch_size_custom or 1)
        warning = effective > config.gpu_batch_size_warning_threshold
        return {
            **base,
            "gpu_batch_size_mode_effective": "custom",
            "requested_gpu_batch_size": effective,
            "effective_gpu_batch_size": effective,
            "gpu_batch_size_candidates_tried": (),
            "gpu_batch_size_last_good": None,
            "gpu_batch_size_failure_at": None,
            "gpu_batch_size_failure_reason": None,
            "gpu_batch_size_probe_required": False,
            "gpu_batch_size_auto_failed": False,
            "gpu_batch_size_warning_emitted": warning,
            "gpu_batch_size_warning_reason": _BATCH_SIZE_WARNING_REASON
            if warning
            else None,
            "gpu_batch_size_manual_override_used": True,
        }
    return _resolve_auto_gpu_batch_size_policy(
        config,
        probe_results=probe_results or (),
        base=base,
    )


def _gpu_batch_size_base_metadata(
    config: TeacherBackendConfig,
    *,
    measured_output_bytes: int | None,
    measured_output_bytes_available: bool,
) -> dict[str, object]:
    estimated_bytes = _estimated_batch_output_bytes(config)
    ratio = (
        estimated_bytes / measured_output_bytes
        if measured_output_bytes not in {None, 0}
        else None
    )
    return {
        "gpu_batch_size_mode_requested": config.gpu_batch_size_mode,
        "gpu_batch_size_policy": "gpu_batch_size_policy_v1",
        "gpu_batch_size_preset": config.gpu_batch_size_preset,
        "gpu_batch_size_custom": config.gpu_batch_size_custom,
        "gpu_batch_size_auto_min": config.gpu_batch_size_auto_min,
        "gpu_batch_size_auto_max": config.gpu_batch_size_auto_max,
        "gpu_batch_size_probe_policy": config.gpu_batch_size_probe_policy,
        "gpu_batch_size_candidate_policy": config.gpu_batch_size_probe_policy,
        "gpu_batch_size_candidates": gpu_batch_size_candidates(config),
        "gpu_batch_size_midpoint_refinement": (
            config.gpu_batch_size_midpoint_refinement
        ),
        "batch_size_policy_uses_estimates": True,
        "estimated_bytes_are_calibrated": False,
        "estimated_batch_output_bytes": estimated_bytes,
        "measured_output_bytes_available": measured_output_bytes_available,
        "measured_output_bytes": measured_output_bytes,
        "measured_compact_bytes_transferred_to_host": measured_output_bytes,
        "estimated_to_measured_bytes_ratio": ratio,
        "gpu_batch_size_target_policy": config.target_policy,
        "gpu_batch_size_exemplar_source_policy": config.exemplar_source_policy,
        "gpu_batch_size_exemplar_capture_mode": config.exemplar_capture_mode,
        "multidevice_enabled": False,
        "batch_partition_strategy": "single_device",
        "device_count": 1,
        "device_ids": (),
        "global_batch_size": None,
        "per_device_batch_size": None,
        "multidevice_policy_version": None,
        "measured_gpu_peak_memory_available": False,
        "measured_gpu_peak_memory_bytes": None,
    }


def _resolve_auto_gpu_batch_size_policy(
    config: TeacherBackendConfig,
    *,
    probe_results: tuple[Mapping[str, object], ...],
    base: dict[str, object],
) -> dict[str, object]:
    if not probe_results:
        return {
            **base,
            "gpu_batch_size_mode_effective": "auto",
            "requested_gpu_batch_size": None,
            "effective_gpu_batch_size": config.gpu_batch_size_auto_min,
            "gpu_batch_size_candidates_tried": (),
            "gpu_batch_size_last_good": None,
            "gpu_batch_size_failure_at": None,
            "gpu_batch_size_failure_reason": None,
            "gpu_batch_size_probe_required": True,
            "gpu_batch_size_auto_failed": False,
            "gpu_batch_size_warning_emitted": False,
            "gpu_batch_size_warning_reason": None,
            "gpu_batch_size_manual_override_used": False,
        }

    candidates_tried = tuple(
        int(result["candidate_batch_size"]) for result in probe_results
    )
    last_good: int | None = None
    last_good_time: float | None = None
    last_good_write_time: float | None = None
    failure_at: int | None = None
    failure_reason: str | None = None
    midpoint_candidate: int | None = None

    for index, result in enumerate(probe_results):
        candidate = int(result["candidate_batch_size"])
        success = bool(result.get("success", False))
        candidate_failure = _probe_failure_reason(
            result,
            last_good_time=last_good_time,
            last_good_write_time=last_good_write_time,
        )
        if not success or candidate_failure is not None:
            failure_at = candidate
            failure_reason = str(
                result.get("failure_reason") or candidate_failure or "unknown"
            )
            if config.gpu_batch_size_midpoint_refinement and last_good is not None:
                midpoint_candidate = _successful_midpoint_candidate(
                    probe_results[index + 1 :],
                    last_good=last_good,
                    failure_at=failure_at,
                )
            break
        last_good = candidate
        last_good_time = _float_or_none(result.get("time_per_example_seconds"))
        last_good_write_time = _float_or_none(result.get("write_time_seconds"))

    if failure_at is None and probe_results:
        final = probe_results[-1]
        if bool(final.get("success", False)):
            last_good = int(final["candidate_batch_size"])
    effective = midpoint_candidate if midpoint_candidate is not None else last_good
    auto_failed = effective is None
    if effective is None:
        effective = config.gpu_batch_size_auto_min
    return {
        **base,
        "gpu_batch_size_mode_effective": "auto",
        "requested_gpu_batch_size": None,
        "effective_gpu_batch_size": effective,
        "gpu_batch_size_candidates_tried": candidates_tried,
        "gpu_batch_size_last_good": last_good,
        "gpu_batch_size_failure_at": failure_at,
        "gpu_batch_size_failure_reason": failure_reason,
        "gpu_batch_size_probe_required": False,
        "gpu_batch_size_auto_failed": auto_failed,
        "gpu_batch_size_warning_emitted": False,
        "gpu_batch_size_warning_reason": None,
        "gpu_batch_size_manual_override_used": False,
    }


def _probe_failure_reason(
    result: Mapping[str, object],
    *,
    last_good_time: float | None,
    last_good_write_time: float | None,
) -> str | None:
    if bool(result.get("oom_or_device_failure", False)):
        return str(result.get("failure_reason") or "oom")
    if not bool(result.get("success", False)):
        return str(result.get("failure_reason") or "unknown")
    time_per_example = _float_or_none(result.get("time_per_example_seconds"))
    if (
        last_good_time is not None
        and time_per_example is not None
        and time_per_example > last_good_time * _TIME_PER_EXAMPLE_REGRESSION_FACTOR
    ):
        return "time_per_example_regression"
    write_time = _float_or_none(result.get("write_time_seconds"))
    if (
        last_good_write_time is not None
        and write_time is not None
        and write_time > last_good_write_time * _WRITE_TIME_REGRESSION_FACTOR
    ):
        return "output_write_time_regression"
    return None


def _successful_midpoint_candidate(
    probe_results: tuple[Mapping[str, object], ...],
    *,
    last_good: int,
    failure_at: int,
) -> int | None:
    midpoint = (last_good + failure_at) // 2
    for result in probe_results:
        candidate = int(result["candidate_batch_size"])
        if candidate != midpoint:
            continue
        if (
            bool(result.get("success", False))
            and _probe_failure_reason(
                result,
                last_good_time=None,
                last_good_write_time=None,
            )
            is None
        ):
            return candidate
    return None


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _payload_nbytes(payload: Mapping[str, object] | None) -> int | None:
    if payload is None:
        return None
    total = 0
    found = False
    for value in payload.values():
        nbytes = getattr(value, "nbytes", None)
        if nbytes is None:
            continue
        total += int(nbytes)
        found = True
    return total if found else None


def _estimated_batch_output_bytes(config: TeacherBackendConfig) -> int:
    return int(config.batch_size * config.sequence_length * config.vocab_size * 4)


def _exemplar_policy_num_examples(
    config: TeacherBackendConfig,
    *,
    actual_batch_size: int | None,
) -> tuple[int, list[str]]:
    if config.exemplar_auto_num_examples is not None:
        return config.exemplar_auto_num_examples, []
    if actual_batch_size is not None:
        return actual_batch_size, ["num_examples"]
    return config.batch_size, ["num_examples"]


def _exemplar_expected_selected_fraction(config: TeacherBackendConfig) -> float:
    if config.exemplar_auto_expected_selected_fraction is not None:
        return config.exemplar_auto_expected_selected_fraction
    if config.exemplar_sparse_selection_fraction is not None:
        return config.exemplar_sparse_selection_fraction
    return 0.01


def _exemplar_capture_estimates(
    config: TeacherBackendConfig,
    *,
    num_examples: int,
    expected_selected_exemplar_fraction: float,
    effective_vocab_size: int | None,
) -> dict[str, int]:
    sequence_length = config.sequence_length
    vocab_size = (
        effective_vocab_size if effective_vocab_size is not None else config.vocab_size
    )
    exemplar_top_n = min(config.exemplar_top_n, sequence_length)
    bytes_per_scalar = 4
    per_position_arrays = 9
    per_example_arrays = 1
    per_exemplar_arrays = 2
    source_policy_multiplier = 1
    if config.exemplar_source_policy == "cascaded_soft_labels_v1":
        source_policy_multiplier = max(1, min(config.top_k, vocab_size))
    if config.exemplar_source_policy == "dense_logits":
        source_policy_multiplier = max(1, min(vocab_size, config.dynamic_top_k_max))
    estimated_one_pass = num_examples * (
        sequence_length
        * per_position_arrays
        * bytes_per_scalar
        * source_policy_multiplier
        + per_example_arrays * bytes_per_scalar
        + exemplar_top_n * per_exemplar_arrays * bytes_per_scalar
    )
    estimated_score = num_examples * 8 * bytes_per_scalar
    selected_examples = max(1, ceil(num_examples * expected_selected_exemplar_fraction))
    estimated_selected = selected_examples * (
        sequence_length
        * per_position_arrays
        * bytes_per_scalar
        * source_policy_multiplier
        + per_example_arrays * bytes_per_scalar
        + exemplar_top_n * per_exemplar_arrays * bytes_per_scalar
    )
    return {
        "estimated_one_pass_candidate_bytes": int(estimated_one_pass),
        "estimated_two_pass_score_bytes": int(estimated_score),
        "estimated_two_pass_selected_bytes": int(estimated_selected),
        "estimated_two_pass_total_bytes": int(estimated_score + estimated_selected),
    }
