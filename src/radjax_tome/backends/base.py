from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
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
