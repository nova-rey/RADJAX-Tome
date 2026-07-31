"""Typed public handoffs and errors for selected-exemplar delivery.

The delivery implementation remains in ``exemplar_delivery`` during M6C;
this module is intentionally dependency-light and owns its stable data
boundary so rerun, assembly, and validation can split without a flag day.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from radjax_tome.backends import TeacherBackendConfig
from radjax_tome.builder.corridor_artifacts import CorridorArtifactBuildResult
from radjax_tome.builder.long_tail import (
    DEFAULT_LONG_TAIL_WARNING_K,
    DEFAULT_PERVERSE_TAIL_WARNING_K,
    DEFAULT_VERY_LONG_TAIL_WARNING_K,
)
from radjax_tome.builder.teacher_textbook import TinyTextExample
from radjax_tome.targets.store import TeacherTargetStore

EXEMPLAR_DELIVERY_REPORT_FILENAME = "delivery_report.json"
EXEMPLAR_DELIVERY_REPORT_SCHEMA = "selected_exemplar_delivery_report_v1"
EXEMPLAR_DELIVERY_PARITY_REPORT_SCHEMA = "exemplar_delivery_parity_report_v1"
LEADERBOARD_REPORT_FILENAME = "leaderboard_report.json"
SELECTED_EXEMPLARS_FILENAME = "selected_exemplars.json"
CURRICULUM_ROUTES_FILENAME = "selected_routes.json"
CURRICULUM_ROUTES_SCHEMA = "selected_exemplar_curriculum_routes_v1"
SELECTED_LINKAGE_MISMATCH = (
    "selected exemplar linkage mismatch: selected record/payload does not match "
    "source candidate coordinate"
)
ONE_PASS_PRUNED_CANDIDATE = "one_pass_pruned_candidate"
TWO_PASS_RERUN_SELECTED = "two_pass_rerun_selected"
NATIVE_C6_PATH_B_EXECUTION = "native_c6_path_b_v1"
EXEMPLAR_SCORE_POLICY = "entropy_top_n_v1"
DeliveryProgressCallback = Callable[[dict[str, Any]], None]


class SelectedExemplarDeliveryError(ValueError):
    """Preserves a machine-readable coordinate trace for delivery failures."""

    def __init__(self, diagnostic: dict[str, Any]) -> None:
        self.diagnostic = diagnostic
        super().__init__(
            f"{SELECTED_LINKAGE_MISMATCH}: {json.dumps(diagnostic, sort_keys=True)}"
        )


class SelectedRerunCudaOOMError(RuntimeError):
    """Native selected rerun exhausted the final microbatch size."""

    def __init__(self, diagnostic: dict[str, Any]) -> None:
        self.diagnostic = diagnostic
        super().__init__(
            "selected rerun CUDA OOM at batch size 1: "
            + json.dumps(diagnostic, sort_keys=True)
        )


@dataclass(frozen=True)
class ExemplarDeliveryConfig:
    artifact_dir: Path
    dataset_path: Path
    delivery_path: str = TWO_PASS_RERUN_SELECTED
    selection_enabled: bool = False
    leaderboard_capacity: int = 16
    selected_exemplar_budget: int | None = None
    selected_exemplar_fraction: float | None = None
    retain_unselected_exemplar_payloads: bool = True
    score_policy: str = EXEMPLAR_SCORE_POLICY
    sequence_length: int = 16
    vocab_size: int = 32
    top_k: int = 8
    num_buckets: int = 4
    max_examples: int | None = None
    backend_config: TeacherBackendConfig | None = None
    selected_rerun_batch_size: int = 1
    track_timing: bool = False
    long_tail_warning_k: int = DEFAULT_LONG_TAIL_WARNING_K
    very_long_tail_warning_k: int = DEFAULT_VERY_LONG_TAIL_WARNING_K
    perverse_tail_warning_k: int = DEFAULT_PERVERSE_TAIL_WARNING_K
    reject_perverse_exemplars: bool = False
    primary_selected_exemplar_budget: int | None = None
    long_tail_side_board_cap: int = 128
    perverse_tail_side_board_cap: int = 32
    include_long_tail_in_primary: bool = False
    include_perverse_tail_in_primary: bool = False
    include_perverse_tail_in_student: bool = False
    progress_callback: DeliveryProgressCallback | None = None
    authoritative_selection: bool = False
    authoritative_records: tuple[dict[str, Any], ...] | None = None
    execution_mode: str = "legacy_delivery_v1"
    rerun_metrics: dict[str, Any] | None = None
    delivery_authority_hash: str | None = None


@dataclass(frozen=True)
class PreparedSelectedDelivery:
    """In-memory handoff between native selected-delivery phases."""

    config: ExemplarDeliveryConfig
    created_at: str
    delivery_started: float
    store: TeacherTargetStore
    examples: tuple[TinyTextExample, ...]
    manifest: dict[str, Any]
    selected_records: list[dict[str, Any]]
    selected_payloads: list[dict[str, Any]]
    rerun_selected_example_count: int
    rerun_selected_example_ids: list[str]
    selection_wall_seconds: float
    payload_wall_seconds: float
    selected_example_count: int
    tail_summary: dict[str, Any]
    selected_board_summary: dict[str, Any]
    selected_records_by_board: dict[str, list[dict[str, Any]]]
    corridors_dir: Path
    leaderboards_dir: Path
    selected_dir: Path
    curriculum_dir: Path
    corridor_result: CorridorArtifactBuildResult | None = None
