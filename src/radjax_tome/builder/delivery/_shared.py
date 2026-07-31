from __future__ import annotations

# ruff: noqa: F401
import hashlib
import json
import os
import platform
import resource
import shutil
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from radjax_tome.backends import (
    TeacherBatchInput,
    create_backend,
)
from radjax_tome.builder.corridor_artifacts import (
    build_corridor_artifacts,
    validate_corridor_artifacts,
)
from radjax_tome.builder.exemplar_delivery_contracts import (
    CURRICULUM_ROUTES_FILENAME,
    CURRICULUM_ROUTES_SCHEMA,
    EXEMPLAR_DELIVERY_PARITY_REPORT_SCHEMA,
    EXEMPLAR_DELIVERY_REPORT_FILENAME,
    EXEMPLAR_DELIVERY_REPORT_SCHEMA,
    EXEMPLAR_SCORE_POLICY,
    LEADERBOARD_REPORT_FILENAME,
    NATIVE_C6_PATH_B_EXECUTION,
    ONE_PASS_PRUNED_CANDIDATE,
    SELECTED_EXEMPLARS_FILENAME,
    SELECTED_LINKAGE_MISMATCH,
    TWO_PASS_RERUN_SELECTED,
    ExemplarDeliveryConfig,
    PreparedSelectedDelivery,
    SelectedExemplarDeliveryError,
    SelectedRerunCudaOOMError,
)
from radjax_tome.builder.exemplar_selection import (
    PATH_A_FULFILLMENT_POLICY,
    PATH_B_FULFILLMENT_POLICY,
    build_exemplar_selection_manifest,
)
from radjax_tome.builder.long_tail import (
    LONG_TAIL_UNCERTAINTY_BOARD,
    PERVERSE_TAIL_DIAGNOSTIC_BOARD,
    PRIMARY_SELECTED_BOARD,
    LongTailPolicy,
    is_perverse_long_tail,
    long_tail_diagnostics,
    long_tail_summary,
    selected_board_for_long_tail,
    semantic_tail_tag,
)
from radjax_tome.builder.teacher_textbook import TinyTextExample
from radjax_tome.io.json import read_json_object, write_json
from radjax_tome.quantization import (
    ENTROPY_PARITY_QUANTIZATION_STEP,
    entropy_absolute_delta,
    entropy_parity_close,
)
from radjax_tome.targets.store import TeacherTargetStore

_SIDE_SELECTED_BOARD_IDS = (
    LONG_TAIL_UNCERTAINTY_BOARD,
    PERVERSE_TAIL_DIAGNOSTIC_BOARD,
)
_REQUIRED_SELECTED_PAYLOAD_FIELDS = (
    "selected_example_id",
    "selected_position",
    "selected_score",
    "score_selected_position_entropy",
    "score_top_token_id",
    "source_shard_id",
    "source_row",
    "source_position",
    "source_score",
    "source_top_token_id",
    "source_score_policy",
    "payload_ref",
    "selected_policy",
    "source_delivery_path",
    "top_token_ids",
    "top_log_probs",
    "top_probs",
    "top_selection_mask",
    "effective_top_k",
    "top_mass",
    "tail_mass",
    "bucket_masses",
    "teacher_entropy",
    "sequence_length",
    "vocab_size",
    "num_buckets",
    "dynamic_top_k",
    "dynamic_mass_threshold",
    "dynamic_top_k_max",
    "top_k_saturated",
    "long_tail_class",
    "long_tail_warnings",
    "effective_top_k_fraction_of_vocab",
    "semantic_tail_tag",
    "selected_board",
    "corridor_mode_id",
    "corridor_fingerprint_id",
    "corridor_assignment_status",
)

_ONE_PASS_CANDIDATE_PAYLOAD_ARRAYS = (
    "exemplar_source_policy_ids",
    "exemplar_source_top_token_ids",
    "exemplar_source_top_log_probs",
    "exemplar_source_top_probs",
    "exemplar_source_top_selection_mask",
    "exemplar_source_effective_top_k",
    "exemplar_source_top_mass",
    "exemplar_source_tail_mass",
    "exemplar_source_bucket_masses",
)


__all__ = [name for name in globals() if not name.startswith("__")]
