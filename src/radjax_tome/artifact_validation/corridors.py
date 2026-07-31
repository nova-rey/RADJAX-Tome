"""Validation of persisted corridor artifacts without production-builder imports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from radjax_tome.backends import MIN_CORRIDOR_STAT_TOP_K
from radjax_tome.io.json import read_json_object

CORRIDOR_MODE_POLICY = "stat_bands_v0"
ASSIGNMENT_POLICY = "full_token_position_stat_bands_v0"
CORRIDOR_ASSIGNMENTS_SCHEMA = "corridor_mode_assignments_v3"
LEGACY_CORRIDOR_ASSIGNMENTS_SCHEMA = "corridor_mode_assignments_v2"
ASSIGNMENT_STORAGE_KIND = "packed_numpy_v1"
FULL_TOKEN_POSITION_CORRIDOR = "full_token_position_corridor"
BOUNDED_FULL_SURFACE_SKETCH = "bounded_full_surface_sketch"
SCORE_SELECTED_POSITION_ONLY = "score_selected_position_only"
DEFAULT_CORRIDOR_MAX_MODES = 256
CORRIDOR_TRACKED_STATS = (
    "entropy",
    "top1_margin",
    "top8_mass",
    "top32_mass",
    "tail_mass",
)


@dataclass(frozen=True)
class CorridorArtifactValidationResult:
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    corridor_artifact_ok: bool = False
    corridor_fingerprints_ok: bool = False
    corridor_modes_ok: bool = False
    corridor_mode_count: int = 0
    corridor_fingerprint_count: int = 0
    corridor_observation_basis: str | None = None
    degraded_corridor_export: bool | None = None
    corridor_positions_available: int = 0
    corridor_positions_used: int = 0
    corridor_mode_policy: str | None = None
    corridor_stat_top_k: int = 0
    corridor_assignment_storage_kind: str | None = None
    corridor_assignment_count: int = 0

    @property
    def ok(self) -> bool:
        return not self.blockers


def validate_corridor_artifacts(
    output_dir: Path,
    *,
    selected_records: list[dict[str, Any]] | None = None,
    selected_payloads: list[dict[str, Any]] | None = None,
    expected_selected_count: int | None = None,
) -> CorridorArtifactValidationResult:
    corridors_dir = output_dir / "corridors"
    required = {
        "summary": corridors_dir / "corridor_summary.json",
        "fingerprints": corridors_dir / "corridor_fingerprints.json",
        "modes": corridors_dir / "corridor_modes.json",
        "assignments": corridors_dir / "mode_assignments.json",
    }
    blockers: list[str] = []
    warnings: list[str] = []
    missing = [label for label, path in required.items() if not path.is_file()]
    if missing:
        if "modes" in missing:
            blockers.append(
                "corridor_exemplar_v1 selected-only run did not emit corridor modes"
            )
        blockers.extend(
            f"corridor artifact missing {label}: {path.name}"
            for label, path in required.items()
            if label in missing
        )
        return CorridorArtifactValidationResult(tuple(blockers), tuple(warnings))
    try:
        summary = read_json_object(required["summary"])
        fingerprints = read_json_object(required["fingerprints"])
        modes = read_json_object(required["modes"])
        assignments = read_json_object(required["assignments"])
    except (OSError, ValueError) as exc:
        return CorridorArtifactValidationResult(
            (f"corridor artifact invalid JSON: {exc}",), tuple(warnings)
        )
    fingerprint_count = int(summary.get("fingerprint_count") or 0)
    mode_count = int(summary.get("mode_count") or 0)
    observation_basis = summary.get("corridor_observation_basis")
    degraded = bool(summary.get("degraded_corridor_export"))
    positions_available = int(summary.get("corridor_positions_available") or 0)
    positions_used = int(summary.get("corridor_positions_used") or 0)
    num_examples_scored = int(summary.get("num_examples_scored") or 0)
    num_positions_scored = int(summary.get("num_positions_scored") or 0)
    mode_policy = str(
        summary.get("corridor_mode_policy") or summary.get("mode_policy") or ""
    )
    max_modes = int(summary.get("corridor_max_modes") or DEFAULT_CORRIDOR_MAX_MODES)
    stat_top_k = int(summary.get("corridor_stat_top_k") or 0)
    storage_kind = summary.get("corridor_assignment_storage_kind")
    assignment_count = int(summary.get("corridor_assignment_count") or 0)
    _validate_summary(
        summary,
        blockers,
        observation_basis=observation_basis,
        degraded=degraded,
        positions_used=positions_used,
        positions_available=positions_available,
        num_examples_scored=num_examples_scored,
        num_positions_scored=num_positions_scored,
        fingerprint_count=fingerprint_count,
        mode_count=mode_count,
        mode_policy=mode_policy,
        max_modes=max_modes,
        stat_top_k=stat_top_k,
        storage_kind=storage_kind,
        assignment_count=assignment_count,
    )
    if fingerprints.get("fingerprint_count") != fingerprint_count:
        blockers.append("corridor_fingerprints fingerprint_count mismatch")
    if modes.get("mode_count") != mode_count:
        blockers.append("corridor_modes mode_count mismatch")
    if modes.get("mode_policy") != CORRIDOR_MODE_POLICY:
        blockers.append("corridor_modes.mode_policy must be stat_bands_v0")
    if modes.get("tracked_stats") != list(CORRIDOR_TRACKED_STATS):
        blockers.append("corridor_modes.tracked_stats mismatch")
    if int(modes.get("corridor_stat_top_k") or 0) < MIN_CORRIDOR_STAT_TOP_K:
        blockers.append("corridor_modes.corridor_stat_top_k must be >= 32")
    mode_ids = _validate_modes_payload(modes, blockers)
    if (
        expected_selected_count is not None
        and int(summary.get("selected_exemplar_count") or -1) != expected_selected_count
    ):
        blockers.append("corridor_summary selected_exemplar_count mismatch")
    _validate_selected_links(
        selected_records or (), blockers, source="selected", valid_mode_ids=mode_ids
    )
    _validate_selected_links(
        selected_payloads or (), blockers, source="payload", valid_mode_ids=mode_ids
    )
    schema = assignments.get("schema_version")
    if schema == CORRIDOR_ASSIGNMENTS_SCHEMA:
        _validate_packed_assignment_manifest(
            output_dir,
            assignments,
            mode_ids=mode_ids,
            blockers=blockers,
            expected_count=positions_used,
            expected_examples=num_examples_scored,
        )
    elif schema == LEGACY_CORRIDOR_ASSIGNMENTS_SCHEMA:
        warnings.append(
            "corridor mode assignments use legacy giant-json storage; "
            "packed_numpy_v1 is preferred"
        )
        items = assignments.get("assignments", [])
        if not isinstance(items, list):
            blockers.append("mode_assignments.assignments must be a list")
        else:
            if int(assignments.get("num_assignments") or -1) != len(items):
                blockers.append("mode_assignments.num_assignments mismatch")
            for item in items:
                if not isinstance(item, dict):
                    blockers.append("mode_assignments contains non-object assignment")
                    break
                if item.get("mode_id") not in mode_ids:
                    blockers.append("mode_assignments references nonexistent mode_id")
                    break
    else:
        blockers.append("mode_assignments schema_version mismatch")
    ok = not blockers
    return CorridorArtifactValidationResult(
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        corridor_artifact_ok=ok,
        corridor_fingerprints_ok=fingerprint_count >= 1 and ok,
        corridor_modes_ok=mode_count >= 1 and ok,
        corridor_mode_count=mode_count,
        corridor_fingerprint_count=fingerprint_count,
        corridor_observation_basis=str(observation_basis)
        if observation_basis is not None
        else None,
        degraded_corridor_export=degraded,
        corridor_positions_available=positions_available,
        corridor_positions_used=positions_used,
        corridor_mode_policy=mode_policy,
        corridor_stat_top_k=stat_top_k,
        corridor_assignment_storage_kind=str(storage_kind)
        if storage_kind is not None
        else None,
        corridor_assignment_count=assignment_count,
    )


def _validate_summary(
    summary: dict[str, Any], blockers: list[str], **values: Any
) -> None:
    if summary.get("corridor_artifact_built") is not True:
        blockers.append("corridor_summary.corridor_artifact_built is not true")
    if summary.get("corridor_modes_built") is not True:
        blockers.append("corridor_summary.corridor_modes_built is not true")
    basis = values["observation_basis"]
    if basis == SCORE_SELECTED_POSITION_ONLY or values["degraded"]:
        blockers.append(
            "corridor artifact was built from score-selected positions only; "
            "full corridor fingerprint export is missing"
        )
    if basis not in {
        FULL_TOKEN_POSITION_CORRIDOR,
        BOUNDED_FULL_SURFACE_SKETCH,
        SCORE_SELECTED_POSITION_ONLY,
    }:
        blockers.append("corridor_summary.corridor_observation_basis is invalid")
    if (
        basis == FULL_TOKEN_POSITION_CORRIDOR
        and values["positions_used"] != values["num_positions_scored"]
    ):
        blockers.append(
            "corridor_summary.corridor_positions_used must equal "
            "num_positions_scored for full token-position export"
        )
    if values["positions_used"] <= values["num_examples_scored"]:
        blockers.append(
            "corridor_summary.corridor_positions_used must exceed num_examples_scored"
        )
    if values["positions_available"] < values["positions_used"]:
        blockers.append(
            "corridor_summary.corridor_positions_available must be >= positions_used"
        )
    if values["fingerprint_count"] < 1:
        blockers.append("corridor_summary.fingerprint_count must be >= 1")
    if values["mode_count"] < 1:
        blockers.append("corridor_summary.mode_count must be >= 1")
    if values["mode_policy"] != CORRIDOR_MODE_POLICY:
        blockers.append(
            "corridor modes use deprecated fingerprint_group_v1 pseudo-mode policy; "
            "expected stat_bands_v0"
        )
    if summary.get("corridor_tracked_stats") != list(CORRIDOR_TRACKED_STATS):
        blockers.append("corridor_summary.corridor_tracked_stats mismatch")
    if values["stat_top_k"] < MIN_CORRIDOR_STAT_TOP_K:
        blockers.append("corridor_summary.corridor_stat_top_k must be >= 32")
    if int(summary.get("min_corridor_stat_top_k") or 0) != MIN_CORRIDOR_STAT_TOP_K:
        blockers.append("corridor_summary.min_corridor_stat_top_k mismatch")
    if values["storage_kind"] != ASSIGNMENT_STORAGE_KIND:
        blockers.append(
            "corridor_summary.corridor_assignment_storage_kind must be packed_numpy_v1"
        )
    if values["assignment_count"] != values["positions_used"]:
        blockers.append(
            "corridor_summary.corridor_assignment_count must equal positions used"
        )
    if summary.get("selected_exemplars_linked_to_corridor_modes") is not True:
        blockers.append(
            "corridor_summary.selected_exemplars_linked_to_corridor_modes is not true"
        )
    if values["mode_count"] > values["max_modes"]:
        blockers.append("corridor_summary.mode_count exceeds corridor_max_modes")


def _validate_selected_links(
    items: Any, blockers: list[str], *, source: str, valid_mode_ids: set[Any]
) -> None:
    if not isinstance(items, (list, tuple)):
        return
    for item in items:
        if not isinstance(item, dict):
            continue
        if "corridor_mode_id" not in item or "corridor_fingerprint_id" not in item:
            blockers.append(f"{source} selected exemplar missing corridor linkage")
            return
        if item.get("corridor_assignment_status") != "linked":
            blockers.append(f"{source} selected exemplar corridor assignment missing")
            return
        if item.get("corridor_mode_id") not in valid_mode_ids:
            blockers.append(f"{source} selected exemplar references invalid mode_id")
            return


def _validate_modes_payload(
    modes_payload: dict[str, Any], blockers: list[str]
) -> set[Any]:
    items = modes_payload.get("modes", [])
    if not isinstance(items, list):
        blockers.append("corridor_modes.modes must be a list")
        return set()
    mode_ids: set[Any] = set()
    for mode in items:
        if not isinstance(mode, dict):
            blockers.append("corridor_modes contains non-object mode")
            continue
        mode_id = mode.get("mode_id")
        mode_ids.add(mode_id)
        if mode.get("mode_policy") != CORRIDOR_MODE_POLICY:
            blockers.append("corridor mode entry policy must be stat_bands_v0")
            continue
        key = mode.get("mode_key")
        if not isinstance(key, dict) or not {
            "entropy_bin",
            "top1_margin_bin",
            "top32_mass_bin",
        }.issubset(key):
            blockers.append("corridor mode entry missing stat band mode_key")
        bounds = mode.get("bounds")
        if not isinstance(bounds, dict):
            blockers.append("corridor mode entry missing bounds")
            continue
        for stat in CORRIDOR_TRACKED_STATS:
            stat_bounds = bounds.get(stat)
            if not isinstance(stat_bounds, dict):
                blockers.append(f"corridor mode bounds missing {stat}")
            elif not {"min", "max", "mean"}.issubset(stat_bounds):
                blockers.append(f"corridor mode bounds incomplete for {stat}")
    return mode_ids


def _validate_packed_assignment_manifest(
    output_dir: Path,
    manifest: dict[str, Any],
    *,
    mode_ids: set[Any],
    blockers: list[str],
    expected_count: int,
    expected_examples: int,
) -> None:
    if manifest.get("assignment_policy") != ASSIGNMENT_POLICY:
        blockers.append("mode_assignments.assignment_policy mismatch")
    if manifest.get("storage_kind") != ASSIGNMENT_STORAGE_KIND:
        blockers.append("mode_assignments.storage_kind must be packed_numpy_v1")
    if manifest.get("full_assignment_retained") is not True:
        blockers.append("mode_assignments.full_assignment_retained is not true")
    if int(manifest.get("num_assignments") or -1) != expected_count:
        blockers.append("mode_assignments.num_assignments mismatch")
    if int(manifest.get("num_examples") or -1) != expected_examples:
        blockers.append("mode_assignments.num_examples mismatch")
    arrays = manifest.get("arrays")
    if not isinstance(arrays, dict):
        blockers.append("mode_assignments.arrays must be an object")
        return
    loaded: dict[str, np.ndarray] = {}
    for name, dtype in {
        "position_example_index": np.int32,
        "position": np.int32,
        "mode_id": np.int32,
        "weight": np.float32,
    }.items():
        spec = arrays.get(name)
        if not isinstance(spec, dict):
            blockers.append(f"mode_assignments missing array spec: {name}")
            continue
        path = output_dir / str(spec.get("path", ""))
        if not path.is_file():
            blockers.append(f"mode_assignments array missing: {name}")
            continue
        array = np.load(path, allow_pickle=False)
        loaded[name] = array
        if array.dtype != np.dtype(dtype):
            blockers.append(f"mode_assignments array dtype mismatch: {name}")
        if tuple(array.shape) != (expected_count,):
            blockers.append(f"mode_assignments array shape mismatch: {name}")
        if spec.get("dtype") != str(np.dtype(dtype)):
            blockers.append(f"mode_assignments manifest dtype mismatch: {name}")
        if spec.get("shape") != [expected_count]:
            blockers.append(f"mode_assignments manifest shape mismatch: {name}")
    if set(("position_example_index", "position", "mode_id", "weight")).issubset(
        loaded
    ):
        if np.any(loaded["position"] < 0):
            blockers.append("mode_assignments position contains negative values")
        index = loaded["position_example_index"]
        if np.any(index < 0) or np.any(index >= expected_examples):
            blockers.append("mode_assignments position_example_index out of range")
        if np.any(~np.isfinite(loaded["weight"])) or np.any(loaded["weight"] < 0.0):
            blockers.append("mode_assignments weight must be finite and nonnegative")
        valid = np.asarray(sorted(int(item) for item in mode_ids), dtype=np.int32)
        if not np.isin(loaded["mode_id"], valid).all():
            blockers.append("mode_assignments references nonexistent mode_id")
    examples = manifest.get("examples_metadata")
    if not isinstance(examples, dict):
        blockers.append("mode_assignments.examples_metadata must be an object")
        return
    if not (output_dir / str(examples.get("path", ""))).is_file():
        blockers.append("mode_assignments examples_metadata missing")
    if int(examples.get("num_examples") or -1) != expected_examples:
        blockers.append("mode_assignments examples_metadata num_examples mismatch")
