"""Leaf helpers shared by production stage implementations.

This module intentionally has no dependency on :mod:`builder.production`.
The public production facade supplies its compatibility configuration and run
state objects structurally to the stages.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def hash_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def selection_integration_hash(config: Any) -> str:
    """Preserve the fixed 25-field selection-authority projection."""

    payload = {
        "selection_integration_policy": config.selection_integration_policy,
        "teacher_model": config.teacher_model,
        "tokenizer_id": config.tokenizer_id or config.teacher_model,
        "dataset_path": str(config.dataset_path),
        "corpus_manifest_path": str(config.corpus_manifest_path),
        "target_policy": config.target_policy,
        "sequence_length": config.sequence_length,
        "vocab_size": config.vocab_size,
        "top_k": config.top_k,
        "num_buckets": config.num_buckets,
        "dynamic_top_k_min": config.dynamic_top_k_min,
        "dynamic_top_k_max": config.dynamic_top_k_max,
        "dynamic_mass_threshold": config.dynamic_mass_threshold,
        "selected_rerun_batch_size": config.selected_rerun_batch_size,
        "total_selected_exemplar_budget": config.total_selected_exemplar_budget,
        "fingerprint_corridor_budget_fraction": (
            config.fingerprint_corridor_budget_fraction
        ),
        "fingerprint_corridor_budget_max": config.fingerprint_corridor_budget_max,
        "fingerprint_corridor_mode_cap": config.fingerprint_corridor_mode_cap,
        "fingerprint_corridor_candidate_pool_cap": (
            config.fingerprint_corridor_candidate_pool_cap
        ),
        "require_full_selected_budget": config.require_full_selected_budget,
        "c2_schema": "radjax.c2_corridor_candidate_leaderboards.v1",
        "c3_schema": "radjax.c3_corridor_coverage_plan.v1",
        "c4_schema": "radjax.c4_corridor_global_claims.v1",
        "c5_schema": "radjax.multi_role_selected_exemplar.v1",
        "delivery_path": config.exemplar_delivery_path,
        "full_width_composition_cap": {
            "numerator": config.full_width_cap_numerator,
            "denominator": config.full_width_cap_denominator,
        },
    }
    return hash_payload(payload)


def native_c6_path_b_enabled(config: Any) -> bool:
    from radjax_tome.builder.c6_integration import C6_SELECTION_INTEGRATION_POLICY

    return (
        config.selection_integration_policy == C6_SELECTION_INTEGRATION_POLICY
        and config.target_policy == "corridor_exemplar_v1"
        and config.exemplar_selection_enabled
        and config.exemplar_delivery_path == "two_pass_rerun_selected"
    )


def exemplar_capture_mode(config: Any) -> str:
    return (
        "two_pass_sparse_exemplar"
        if config.exemplar_delivery_path == "two_pass_rerun_selected"
        else "one_pass_candidate"
    )
