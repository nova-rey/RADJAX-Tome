"""Versioned semantic authority hashes for the native C6 score surface.

V1 is intentionally retained as the historical raw-byte recipe.  V2 binds
schema-defined semantic projections while recording the same raw inputs
separately for integrity diagnostics.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from radjax_tome.builder.corridor_artifacts import (
    ASSIGNMENT_POLICY,
    ASSIGNMENT_STORAGE_KIND,
    CORRIDOR_ASSIGNMENTS_SCHEMA,
    CORRIDOR_MODES_SCHEMA,
)
from radjax_tome.builder.exemplar_selection import (
    EXEMPLAR_SELECTION_MANIFEST_SCHEMA,
    validate_exemplar_selection_manifest,
)
from radjax_tome.io.json import read_json_object
from radjax_tome.targets.schema import target_store_metadata_from_dict

AUTHORITY_HASH_V1 = "radjax.c6.score_pass_authority.v1"
AUTHORITY_HASH_V2 = "radjax.c6.score_pass_authority.v2"
AUTHORITY_MANIFEST_SCHEMA_V1 = "radjax.c6_selection_authority.v1"
AUTHORITY_MANIFEST_SCHEMA_V2 = "radjax.c6_selection_authority.v2"

RAW_ARTIFACT_DIGEST_PATHS = {
    "metadata.json": "metadata.json",
    "corridors/mode_assignments.json": "corridors/mode_assignments.json",
    "corridors/corridor_modes.json": "corridors/corridor_modes.json",
    "c6/production_global_selector.json": "c6/production_global_selector.json",
}

_METADATA_CORPUS_KEYS = (
    "source_corpus_hash",
    "source_corpus_manifest_hash",
    "source_corpus_schema_version",
    "source_corpus_num_examples",
    "source_corpus_num_sources",
    "source_corpus_normalization_policy",
    "source_corpus_chunking_policy",
    "source_corpus_deduplication_policy",
)
_METADATA_TARGET_PARAM_KEYS = (
    "target_policy",
    "backend_id",
    "requested_backend_id",
    "runtime_mode",
    "artifact_emission_path",
    "student_consumption_ready",
    "experimental_target_schema",
    "production_global_selector",
    "dynamic_top_k_min",
    "dynamic_top_k_max",
    "dynamic_mass_threshold",
    "corridor_stat_top_k",
    "min_corridor_stat_top_k",
    "teacher_model_provenance_schema",
    "teacher_model_source_kind",
    "teacher_model_identity_confidence",
    "teacher_model_provenance_mode",
    "teacher_model_name",
    "teacher_model_name_source",
    "teacher_model_revision",
    "teacher_model_revision_source",
    "teacher_model_config_hash",
    "teacher_model_tokenizer_hash",
    "teacher_model_weights_hash",
    "teacher_model_directory_hash",
    "teacher_model_network_used",
    "streaming_build",
    "resume_supported",
    "shard_size_examples",
    "resume_config_hash",
    "native_c6_path_b_execution",
    "exemplar_selector_policy",
    "exemplar_selection_enabled",
    "exemplar_selection_manifest_schema",
    "exemplar_fulfillment_policy",
    "selection_application",
    "deduplication_policy",
    "duplicate_candidate_count",
    "backfill_success_count",
    "score_aware_budget_trimming",
    "budget_trimming_policy",
    "budget_applied",
    "num_candidates_seen",
    "num_unique_examples_selected",
    "num_unique_positions_selected",
    "semantic_diversity_used",
    "utility_calibrated",
    "retention_policy",
    "rerun_manifest_ready",
    "selected_pass_rerun_performed",
    "selected_from_existing_capture",
)
_MODE_SEMANTIC_KEYS = (
    "schema_version",
    "mode_policy",
    "corridor_mode_policy",
    "corridor_max_modes",
    "corridor_stat_top_k",
    "min_corridor_stat_top_k",
    "tracked_stats",
    "corridor_observation_basis",
    "degraded_corridor_export",
    "corridor_positions_available",
    "corridor_positions_used",
    "mode_count",
    "modes",
)
_ASSIGNMENT_ARRAYS = (
    "position_example_index",
    "position",
    "mode_id",
    "weight",
    "fingerprint_index",
)
_SELECTOR_SEMANTIC_KEYS = (
    "schema_version",
    "selection_policy",
    "capture_mode",
    "fulfillment_policy",
    "selection_application",
    "num_candidates_seen",
    "num_boards",
    "total_board_capacity",
    "num_board_winners",
    "num_unique_examples_selected",
    "num_unique_positions_selected",
    "deduplication_policy",
    "duplicate_candidate_count",
    "backfill_attempt_count",
    "backfill_success_count",
    "boards_with_backfill",
    "runner_up_pool_multiplier",
    "score_aware_budget_trimming",
    "budget_trimming_policy",
    "budget_requested_examples",
    "budget_requested_fraction",
    "budget_applied",
    "budget_trimmed_example_count",
    "budget_trimmed_position_count",
    "production_global_selector",
    "semantic_diversity_used",
    "utility_calibrated",
    "retention_policy",
    "boards",
    "selected_examples",
    "candidate_filter",
    "candidate_filter_rejected_count",
    "num_candidates_seen_before_filter",
)


@dataclass(frozen=True)
class AuthorityHashSet:
    """Both authority versions plus their retained raw-byte integrity digests."""

    raw_artifact_digests: dict[str, str]
    score_pass_authority_hash_v1: str
    score_pass_authority_hash_v2: str


def hash_payload(payload: Mapping[str, Any]) -> str:
    """Return the historical compact sorted JSON SHA-256 representation."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def raw_artifact_digests(artifact_dir: Path) -> dict[str, str]:
    """Return raw-byte integrity digests for the four authority inputs."""

    root = artifact_dir.resolve()
    return {
        key: _file_sha256(root / relative_path)
        for key, relative_path in RAW_ARTIFACT_DIGEST_PATHS.items()
    }


def score_pass_authority_hash_v1(
    raw_digests: Mapping[str, str],
    *,
    selection_integration_config_hash: str,
) -> str:
    """Preserve the historical v1 recipe exactly for legacy verification."""

    return hash_payload(
        {
            "metadata_sha256": raw_digests["metadata.json"],
            "assignment_manifest_sha256": raw_digests[
                "corridors/mode_assignments.json"
            ],
            "modes_sha256": raw_digests["corridors/corridor_modes.json"],
            "selector_sha256": raw_digests["c6/production_global_selector.json"],
            "selection_integration_config_hash": selection_integration_config_hash,
        }
    )


def authority_hashes_for_artifact(
    artifact_dir: Path,
    *,
    selection_integration_config_hash: str,
) -> AuthorityHashSet:
    """Calculate v1 lineage and v2 semantic authority from one artifact."""

    raw_digests = raw_artifact_digests(artifact_dir)
    return AuthorityHashSet(
        raw_artifact_digests=raw_digests,
        score_pass_authority_hash_v1=score_pass_authority_hash_v1(
            raw_digests,
            selection_integration_config_hash=selection_integration_config_hash,
        ),
        score_pass_authority_hash_v2=score_pass_authority_hash_v2(
            artifact_dir,
            selection_integration_config_hash=selection_integration_config_hash,
        ),
    )


def score_pass_authority_hash_v2(
    artifact_dir: Path,
    *,
    selection_integration_config_hash: str,
) -> str:
    """Hash the v2 schema-defined semantic authority projection."""

    return hash_payload(
        semantic_authority_projection_v2(
            artifact_dir,
            selection_integration_config_hash=selection_integration_config_hash,
        )
    )


def semantic_authority_projection_v2(
    artifact_dir: Path,
    *,
    selection_integration_config_hash: str,
) -> dict[str, Any]:
    """Return the explicit semantic preimage used by authority-hash v2."""

    root = artifact_dir.resolve()
    metadata = read_json_object(root / "metadata.json")
    assignments = read_json_object(root / "corridors" / "mode_assignments.json")
    modes = read_json_object(root / "corridors" / "corridor_modes.json")
    selector = read_json_object(root / "c6" / "production_global_selector.json")
    return {
        "authority_hash_contract_version": AUTHORITY_HASH_V2,
        "metadata": _metadata_projection(metadata),
        "mode_assignments": _assignment_projection(root, assignments),
        "modes": _modes_projection(modes),
        "selector": _selector_projection(selector),
        "selection_integration_config_hash": selection_integration_config_hash,
    }


def authority_hash_contract_version(authority: Mapping[str, Any]) -> str:
    """Resolve a manifest's authority contract without reinterpreting legacy data."""

    version = authority.get("score_pass_authority_contract_version")
    if version is None:
        return AUTHORITY_HASH_V1
    if version not in {AUTHORITY_HASH_V1, AUTHORITY_HASH_V2}:
        raise ValueError(f"unsupported score-pass authority contract: {version!r}")
    return str(version)


def _metadata_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    metadata = target_store_metadata_from_dict(dict(payload))
    source = _allowed_mapping(metadata.source, _METADATA_CORPUS_KEYS)
    target_params = _allowed_mapping(
        metadata.target_params, _METADATA_TARGET_PARAM_KEYS
    )
    for key in _METADATA_CORPUS_KEYS:
        if key in target_params:
            source[key] = target_params.pop(key)
    return {
        "schema_version": metadata.schema_version,
        "target_store_version": metadata.target_store_version,
        "model_id": metadata.model_id,
        "model_family": metadata.model_family,
        "tokenizer_id": metadata.tokenizer_id,
        "tokenizer_hash": metadata.tokenizer_hash,
        "vocab_size": metadata.vocab_size,
        "target_type": metadata.target_type,
        "dtype": metadata.dtype,
        "sequence_length": metadata.sequence_length,
        "num_examples": metadata.num_examples,
        "shard_count": metadata.shard_count,
        "corpus_provenance": source,
        "target_params": target_params,
    }


def _assignment_projection(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    if manifest.get("schema_version") != CORRIDOR_ASSIGNMENTS_SCHEMA:
        raise ValueError("authority-hash v2 requires corridor_mode_assignments_v3")
    if manifest.get("assignment_policy") != ASSIGNMENT_POLICY:
        raise ValueError("authority-hash v2 requires the canonical assignment policy")
    if manifest.get("storage_kind") != ASSIGNMENT_STORAGE_KIND:
        raise ValueError("authority-hash v2 requires packed_numpy_v1 assignments")
    if manifest.get("full_assignment_retained") is not True:
        raise ValueError("authority-hash v2 requires full packed assignments")
    arrays = manifest.get("arrays")
    if not isinstance(arrays, Mapping):
        raise ValueError("authority-hash v2 assignment arrays are missing")
    count = _positive_int(manifest.get("num_assignments"), "num_assignments")
    example_count = _positive_int(manifest.get("num_examples"), "num_examples")
    array_projection = {
        name: _assignment_array_projection(root, arrays, name, count)
        for name in _ASSIGNMENT_ARRAYS
    }
    examples = _assignment_examples_projection(root, manifest, example_count)
    return {
        "schema_version": str(manifest["schema_version"]),
        "assignment_policy": str(manifest["assignment_policy"]),
        "storage_kind": str(manifest["storage_kind"]),
        "corridor_observation_basis": manifest.get("corridor_observation_basis"),
        "full_assignment_retained": True,
        "num_assignments": count,
        "num_examples": example_count,
        "arrays": array_projection,
        "examples": examples,
    }


def _assignment_array_projection(
    root: Path,
    arrays: Mapping[str, Any],
    name: str,
    count: int,
) -> dict[str, Any]:
    spec = arrays.get(name)
    if not isinstance(spec, Mapping):
        raise ValueError(f"authority-hash v2 assignment array is missing: {name}")
    path = _artifact_relative_path(root, spec.get("path"), f"assignment array {name}")
    array = np.load(path, allow_pickle=False)
    if array.ndim != 1 or int(array.shape[0]) != count:
        raise ValueError(f"authority-hash v2 assignment array shape is invalid: {name}")
    if str(np.dtype(array.dtype)) != str(spec.get("dtype")):
        raise ValueError(f"authority-hash v2 assignment array dtype is invalid: {name}")
    if list(array.shape) != spec.get("shape"):
        raise ValueError(
            f"authority-hash v2 assignment array shape spec is invalid: {name}"
        )
    return {
        "dtype": _canonical_dtype(array.dtype),
        "shape": list(array.shape),
        "semantic_sha256": _array_semantic_sha256(array),
    }


def _assignment_examples_projection(
    root: Path,
    manifest: Mapping[str, Any],
    expected_count: int,
) -> list[dict[str, Any]]:
    spec = manifest.get("examples_metadata")
    if not isinstance(spec, Mapping):
        raise ValueError("authority-hash v2 examples metadata is missing")
    if (
        _positive_int(spec.get("num_examples"), "examples_metadata.num_examples")
        != expected_count
    ):
        raise ValueError("authority-hash v2 examples metadata count is invalid")
    path = _artifact_relative_path(root, spec.get("path"), "examples metadata")
    rows: list[dict[str, Any]] = []
    for expected_index, line in enumerate(
        path.read_text(encoding="utf-8").splitlines()
    ):
        payload = json.loads(line)
        if (
            not isinstance(payload, Mapping)
            or payload.get("example_index") != expected_index
            or not isinstance(payload.get("example_id"), str)
        ):
            raise ValueError("authority-hash v2 examples metadata is invalid")
        rows.append(
            {
                "example_index": expected_index,
                "example_id": payload["example_id"],
            }
        )
    if len(rows) != expected_count:
        raise ValueError("authority-hash v2 examples metadata count does not match")
    return rows


def _modes_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != CORRIDOR_MODES_SCHEMA:
        raise ValueError("authority-hash v2 requires corridor_modes_v2")
    missing = [key for key in _MODE_SEMANTIC_KEYS if key not in payload]
    if missing:
        raise ValueError(f"authority-hash v2 corridor modes missing fields: {missing}")
    modes = payload.get("modes")
    if not isinstance(modes, list) or int(payload["mode_count"]) != len(modes):
        raise ValueError("authority-hash v2 corridor modes are invalid")
    return {key: payload[key] for key in _MODE_SEMANTIC_KEYS}


def _selector_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    validate_exemplar_selection_manifest(payload)
    if payload.get("schema_version") != EXEMPLAR_SELECTION_MANIFEST_SCHEMA:
        raise ValueError("authority-hash v2 selector schema is unsupported")
    return {key: payload[key] for key in _SELECTOR_SEMANTIC_KEYS if key in payload}


def _artifact_relative_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"authority-hash v2 {label} path is missing")
    path = (root / value).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"authority-hash v2 {label} path is invalid")
    return path


def _array_semantic_sha256(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(
        array.astype(array.dtype.newbyteorder("<"), copy=False)
    )
    return "sha256:" + hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _canonical_dtype(dtype: np.dtype[Any]) -> str:
    return np.dtype(dtype).newbyteorder("<").str


def _allowed_mapping(
    source: Mapping[str, str], keys: tuple[str, ...]
) -> dict[str, str]:
    return {key: source[key] for key in keys if key in source}


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"authority-hash v2 {label} must be a positive integer")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()
