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
