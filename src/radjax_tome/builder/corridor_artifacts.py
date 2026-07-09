from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from radjax_tome.backends import MIN_CORRIDOR_STAT_TOP_K
from radjax_tome.builder.teacher_textbook import TinyTextExample
from radjax_tome.io.json import read_json_object, write_json
from radjax_tome.targets.store import TeacherTargetStore

FINGERPRINT_POLICY = "top_token_entropy_confidence_v1"
CORRIDOR_MODE_POLICY = "stat_bands_v0"
MODE_POLICY = CORRIDOR_MODE_POLICY
ASSIGNMENT_POLICY = "full_token_position_stat_bands_v0"
CORRIDOR_SUMMARY_SCHEMA = "corridor_summary_v3"
CORRIDOR_FINGERPRINTS_SCHEMA = "corridor_fingerprints_v1"
CORRIDOR_MODES_SCHEMA = "corridor_modes_v2"
CORRIDOR_ASSIGNMENTS_SCHEMA = "corridor_mode_assignments_v3"
LEGACY_CORRIDOR_ASSIGNMENTS_SCHEMA = "corridor_mode_assignments_v2"
ASSIGNMENT_STORAGE_KIND = "packed_numpy_v1"
FULL_TOKEN_POSITION_CORRIDOR = "full_token_position_corridor"
BOUNDED_FULL_SURFACE_SKETCH = "bounded_full_surface_sketch"
SCORE_SELECTED_POSITION_ONLY = "score_selected_position_only"
DEFAULT_CORRIDOR_MAX_MODES = 256
DEFAULT_ENTROPY_BINS = (0.0, 1.0, 2.5, 4.0, 8.0, math.inf)
DEFAULT_TOP1_MARGIN_BINS = (0.0, 0.05, 0.15, 0.35, 1.0, math.inf)
DEFAULT_TOP32_MASS_BINS = (0.0, 0.5, 0.75, 0.9, 1.0, math.inf)
CORRIDOR_TRACKED_STATS = (
    "entropy",
    "top1_margin",
    "top8_mass",
    "top32_mass",
    "tail_mass",
)
_CORRIDOR_MIN_WIDTH = 1e-6
CorridorProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class CorridorArtifactBuildResult:
    summary_path: Path
    fingerprints_path: Path
    modes_path: Path
    assignments_path: Path
    human_summary_path: Path
    fingerprint_count: int
    mode_count: int
    selected_exemplars_linked: bool
    observation_basis: str
    degraded: bool
    positions_available: int
    positions_used: int
    corridor_stat_top_k: int
    assignment_storage_kind: str
    assignment_count: int

    def report_fields(self) -> dict[str, Any]:
        return {
            "corridor_artifact_built": True,
            "corridor_modes_built": True,
            "corridor_observation_basis": self.observation_basis,
            "degraded_corridor_export": self.degraded,
            "corridor_positions_available": self.positions_available,
            "corridor_positions_used": self.positions_used,
            "corridor_observation_count": self.positions_used,
            "corridor_summary_path": str(self.summary_path),
            "corridor_fingerprints_path": str(self.fingerprints_path),
            "corridor_modes_path": str(self.modes_path),
            "corridor_mode_assignments_path": str(self.assignments_path),
            "corridor_fingerprint_count": self.fingerprint_count,
            "corridor_mode_count": self.mode_count,
            "corridor_mode_policy": CORRIDOR_MODE_POLICY,
            "corridor_max_modes": DEFAULT_CORRIDOR_MAX_MODES,
            "corridor_tracked_stats": list(CORRIDOR_TRACKED_STATS),
            "corridor_stat_top_k": self.corridor_stat_top_k,
            "min_corridor_stat_top_k": MIN_CORRIDOR_STAT_TOP_K,
            "corridor_assignment_storage_kind": self.assignment_storage_kind,
            "corridor_assignment_count": self.assignment_count,
            "selected_exemplars_linked_to_corridor_modes": (
                self.selected_exemplars_linked
            ),
        }


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

    def report_fields(self) -> dict[str, Any]:
        return {
            "corridor_artifact_ok": self.corridor_artifact_ok,
            "corridor_fingerprints_ok": self.corridor_fingerprints_ok,
            "corridor_modes_ok": self.corridor_modes_ok,
            "corridor_mode_count": self.corridor_mode_count,
            "corridor_fingerprint_count": self.corridor_fingerprint_count,
            "corridor_observation_basis": self.corridor_observation_basis,
            "degraded_corridor_export": self.degraded_corridor_export,
            "corridor_positions_available": self.corridor_positions_available,
            "corridor_positions_used": self.corridor_positions_used,
            "corridor_mode_policy": self.corridor_mode_policy,
            "corridor_max_modes": DEFAULT_CORRIDOR_MAX_MODES,
            "corridor_tracked_stats": list(CORRIDOR_TRACKED_STATS),
            "corridor_stat_top_k": self.corridor_stat_top_k,
            "min_corridor_stat_top_k": MIN_CORRIDOR_STAT_TOP_K,
            "corridor_assignment_storage_kind": self.corridor_assignment_storage_kind,
            "corridor_assignment_count": self.corridor_assignment_count,
        }


@dataclass(frozen=True)
class CorridorObservationExtraction:
    observations: list[_Observation]
    observation_basis: str
    positions_available: int
    positions_used: int
    degraded: bool
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _Observation:
    example_id: str
    position: int
    top_token_id: int
    entropy: float
    confidence: float
    top1_margin: float
    top8_mass: float
    top32_mass: float
    tail_mass: float
    length: int
    source_policy_id: int

    @property
    def key(self) -> tuple[str, int]:
        return (self.example_id, self.position)

    @property
    def stat_values(self) -> dict[str, float]:
        return {
            "entropy": self.entropy,
            "top1_margin": self.top1_margin,
            "top8_mass": self.top8_mass,
            "top32_mass": self.top32_mass,
            "tail_mass": self.tail_mass,
        }


def build_corridor_artifacts(
    *,
    output_dir: Path,
    examples: tuple[TinyTextExample, ...],
    selected_records: list[dict[str, Any]],
    selected_payloads: list[dict[str, Any]],
    delivery_path: str,
    non_selected_exemplar_payload_retained: bool,
    fingerprint_policy: str = FINGERPRINT_POLICY,
    mode_policy: str = MODE_POLICY,
    allow_degraded_score_only: bool = False,
    progress_callback: CorridorProgressCallback | None = None,
) -> CorridorArtifactBuildResult:
    store = TeacherTargetStore.open(output_dir)
    positions_total = store.metadata.num_examples * store.metadata.sequence_length
    _notify_corridor_progress(
        progress_callback,
        phase="corridor_export",
        event="started",
        positions_processed=0,
        positions_total=positions_total,
        modes_discovered=0,
        fingerprints_discovered=0,
        assignment_storage_kind=ASSIGNMENT_STORAGE_KIND,
    )
    extraction = _corridor_observations(
        store,
        examples,
        allow_degraded_score_only=allow_degraded_score_only,
    )
    _notify_corridor_progress(
        progress_callback,
        phase="corridor_export",
        event="observations_extracted",
        positions_processed=extraction.positions_used,
        positions_total=extraction.positions_available,
        modes_discovered=0,
        fingerprints_discovered=0,
        assignment_storage_kind=ASSIGNMENT_STORAGE_KIND,
    )
    observations = extraction.observations
    fingerprints, observation_to_fingerprint = _fingerprints(
        observations,
        target_type="corridor_exemplar_v1",
        source_target_type=store.metadata.target_type,
        num_examples_scored=store.metadata.num_examples,
        num_positions_scored=store.metadata.num_examples
        * store.metadata.sequence_length,
        fingerprint_policy=fingerprint_policy,
    )
    modes, observation_to_mode = _modes(
        observations,
        mode_policy=mode_policy,
    )
    _notify_corridor_progress(
        progress_callback,
        phase="corridor_export",
        event="modes_discovered",
        positions_processed=extraction.positions_used,
        positions_total=extraction.positions_available,
        modes_discovered=len(modes),
        fingerprints_discovered=len(fingerprints),
        assignment_storage_kind=ASSIGNMENT_STORAGE_KIND,
    )
    linked = _link_selected_exemplars(
        selected_records,
        selected_payloads,
        observation_to_fingerprint=observation_to_fingerprint,
        observation_to_mode=observation_to_mode,
    )
    selected_linked = bool(selected_records) and linked == len(selected_records)

    corridors_dir = output_dir / "corridors"
    corridors_dir.mkdir(parents=True, exist_ok=True)
    corridor_stat_top_k = _store_corridor_stat_top_k(store)
    summary_path = corridors_dir / "corridor_summary.json"
    fingerprints_path = corridors_dir / "corridor_fingerprints.json"
    modes_path = corridors_dir / "corridor_modes.json"
    assignments_path = corridors_dir / "mode_assignments.json"
    human_summary_path = corridors_dir / "corridor_summary.txt"
    assignments_manifest = _write_packed_mode_assignments(
        output_dir=output_dir,
        observations=observations,
        observation_to_fingerprint=observation_to_fingerprint,
        observation_to_mode=observation_to_mode,
        observation_basis=extraction.observation_basis,
    )
    _notify_corridor_progress(
        progress_callback,
        phase="corridor_export",
        event="assignments_written",
        positions_processed=extraction.positions_used,
        positions_total=extraction.positions_available,
        modes_discovered=len(modes),
        fingerprints_discovered=len(fingerprints),
        assignment_storage_kind=ASSIGNMENT_STORAGE_KIND,
    )

    summary = {
        "schema_version": CORRIDOR_SUMMARY_SCHEMA,
        "corridor_artifact_built": True,
        "corridor_fingerprints_retained": True,
        "corridor_modes_built": True,
        "corridor_observation_basis": extraction.observation_basis,
        "degraded_corridor_export": extraction.degraded,
        "corridor_positions_available": extraction.positions_available,
        "corridor_positions_used": extraction.positions_used,
        "corridor_observation_count": extraction.positions_used,
        "fingerprint_count": len(fingerprints),
        "mode_count": len(modes),
        "num_examples_scored": store.metadata.num_examples,
        "num_positions_scored": store.metadata.num_examples
        * store.metadata.sequence_length,
        "fingerprint_policy": fingerprint_policy,
        "mode_policy": mode_policy,
        "corridor_mode_policy": mode_policy,
        "corridor_max_modes": DEFAULT_CORRIDOR_MAX_MODES,
        "corridor_tracked_stats": list(CORRIDOR_TRACKED_STATS),
        "corridor_stat_top_k": corridor_stat_top_k,
        "min_corridor_stat_top_k": MIN_CORRIDOR_STAT_TOP_K,
        "corridor_assignment_storage_kind": ASSIGNMENT_STORAGE_KIND,
        "corridor_assignment_count": assignments_manifest["num_assignments"],
        "selected_exemplar_count": len(selected_records),
        "selected_exemplars_linked_to_modes": selected_linked,
        "selected_exemplars_linked_to_corridor_modes": selected_linked,
        "non_selected_exemplar_payload_retained": (
            non_selected_exemplar_payload_retained
        ),
        "delivery_path": delivery_path,
    }
    write_json(summary_path, summary)
    write_json(
        fingerprints_path,
        {
            "schema_version": CORRIDOR_FINGERPRINTS_SCHEMA,
            "target_type": "corridor_exemplar_v1",
            "source_target_type": "corridor_exemplar_score_pass_v1",
            "source_store_target_type": store.metadata.target_type,
            "corridor_observation_basis": extraction.observation_basis,
            "degraded_corridor_export": extraction.degraded,
            "corridor_positions_available": extraction.positions_available,
            "corridor_positions_used": extraction.positions_used,
            "corridor_observation_count": extraction.positions_used,
            "num_examples_scored": store.metadata.num_examples,
            "num_positions_scored": store.metadata.num_examples
            * store.metadata.sequence_length,
            "fingerprint_policy": fingerprint_policy,
            "fingerprint_count": len(fingerprints),
            "fingerprints": fingerprints,
        },
    )
    write_json(
        modes_path,
        {
            "schema_version": CORRIDOR_MODES_SCHEMA,
            "mode_policy": mode_policy,
            "corridor_mode_policy": mode_policy,
            "corridor_max_modes": DEFAULT_CORRIDOR_MAX_MODES,
            "corridor_stat_top_k": corridor_stat_top_k,
            "min_corridor_stat_top_k": MIN_CORRIDOR_STAT_TOP_K,
            "tracked_stats": list(CORRIDOR_TRACKED_STATS),
            "corridor_observation_basis": extraction.observation_basis,
            "degraded_corridor_export": extraction.degraded,
            "corridor_positions_available": extraction.positions_available,
            "corridor_positions_used": extraction.positions_used,
            "mode_count": len(modes),
            "modes": modes,
        },
    )
    write_json(assignments_path, assignments_manifest)
    human_summary_path.write_text(
        _human_summary(
            num_examples=store.metadata.num_examples,
            num_positions=store.metadata.num_examples * store.metadata.sequence_length,
            corridor_positions_used=extraction.positions_used,
            observation_basis=extraction.observation_basis,
            degraded=extraction.degraded,
            mode_count=len(modes),
            mode_policy=mode_policy,
            assignment_storage_kind=ASSIGNMENT_STORAGE_KIND,
            fingerprint_count=len(fingerprints),
            selected_count=len(selected_records),
            selected_payload_retained=bool(selected_payloads),
            non_selected_payload_retained=non_selected_exemplar_payload_retained,
            delivery_path=delivery_path,
        ),
        encoding="utf-8",
    )
    return CorridorArtifactBuildResult(
        summary_path=summary_path,
        fingerprints_path=fingerprints_path,
        modes_path=modes_path,
        assignments_path=assignments_path,
        human_summary_path=human_summary_path,
        fingerprint_count=len(fingerprints),
        mode_count=len(modes),
        selected_exemplars_linked=selected_linked,
        observation_basis=extraction.observation_basis,
        degraded=extraction.degraded,
        positions_available=extraction.positions_available,
        positions_used=extraction.positions_used,
        corridor_stat_top_k=corridor_stat_top_k,
        assignment_storage_kind=ASSIGNMENT_STORAGE_KIND,
        assignment_count=int(assignments_manifest["num_assignments"]),
    )


def _notify_corridor_progress(
    progress_callback: CorridorProgressCallback | None,
    **payload: Any,
) -> None:
    if progress_callback is not None:
        progress_callback(payload)


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
        return CorridorArtifactValidationResult(
            blockers=tuple(blockers),
            warnings=tuple(warnings),
        )

    try:
        summary = read_json_object(required["summary"])
        fingerprints = read_json_object(required["fingerprints"])
        modes = read_json_object(required["modes"])
        assignments = read_json_object(required["assignments"])
    except (OSError, ValueError) as exc:
        return CorridorArtifactValidationResult(
            blockers=(f"corridor artifact invalid JSON: {exc}",),
            warnings=tuple(warnings),
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
    corridor_stat_top_k = int(summary.get("corridor_stat_top_k") or 0)
    assignment_storage_kind = summary.get("corridor_assignment_storage_kind")
    assignment_count = int(summary.get("corridor_assignment_count") or 0)
    if summary.get("corridor_artifact_built") is not True:
        blockers.append("corridor_summary.corridor_artifact_built is not true")
    if summary.get("corridor_modes_built") is not True:
        blockers.append("corridor_summary.corridor_modes_built is not true")
    if observation_basis == SCORE_SELECTED_POSITION_ONLY or degraded:
        blockers.append(
            "corridor artifact was built from score-selected positions only; "
            "full corridor fingerprint export is missing"
        )
    if observation_basis not in {
        FULL_TOKEN_POSITION_CORRIDOR,
        BOUNDED_FULL_SURFACE_SKETCH,
        SCORE_SELECTED_POSITION_ONLY,
    }:
        blockers.append("corridor_summary.corridor_observation_basis is invalid")
    if observation_basis == FULL_TOKEN_POSITION_CORRIDOR:
        if positions_used != num_positions_scored:
            blockers.append(
                "corridor_summary.corridor_positions_used must equal "
                "num_positions_scored for full token-position export"
            )
    if positions_used <= num_examples_scored:
        blockers.append(
            "corridor_summary.corridor_positions_used must exceed num_examples_scored"
        )
    if positions_available < positions_used:
        blockers.append(
            "corridor_summary.corridor_positions_available must be >= positions_used"
        )
    if fingerprint_count < 1:
        blockers.append("corridor_summary.fingerprint_count must be >= 1")
    if mode_count < 1:
        blockers.append("corridor_summary.mode_count must be >= 1")
    if mode_policy != CORRIDOR_MODE_POLICY:
        blockers.append(
            "corridor modes use deprecated fingerprint_group_v1 pseudo-mode policy; "
            "expected stat_bands_v0"
        )
    if summary.get("corridor_tracked_stats") != list(CORRIDOR_TRACKED_STATS):
        blockers.append("corridor_summary.corridor_tracked_stats mismatch")
    if corridor_stat_top_k < MIN_CORRIDOR_STAT_TOP_K:
        blockers.append("corridor_summary.corridor_stat_top_k must be >= 32")
    if int(summary.get("min_corridor_stat_top_k") or 0) != MIN_CORRIDOR_STAT_TOP_K:
        blockers.append("corridor_summary.min_corridor_stat_top_k mismatch")
    if assignment_storage_kind != ASSIGNMENT_STORAGE_KIND:
        blockers.append(
            "corridor_summary.corridor_assignment_storage_kind must be packed_numpy_v1"
        )
    if assignment_count != positions_used:
        blockers.append(
            "corridor_summary.corridor_assignment_count must equal positions used"
        )
    if summary.get("selected_exemplars_linked_to_corridor_modes") is not True:
        blockers.append(
            "corridor_summary.selected_exemplars_linked_to_corridor_modes is not true"
        )
    if mode_count > max_modes:
        blockers.append("corridor_summary.mode_count exceeds corridor_max_modes")
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
        selected_records or (),
        blockers,
        source="selected",
        valid_mode_ids=mode_ids,
    )
    _validate_selected_links(
        selected_payloads or (),
        blockers,
        source="payload",
        valid_mode_ids=mode_ids,
    )
    assignment_schema = assignments.get("schema_version")
    if assignment_schema == CORRIDOR_ASSIGNMENTS_SCHEMA:
        _validate_packed_assignment_manifest(
            output_dir,
            assignments,
            mode_ids=mode_ids,
            blockers=blockers,
            expected_count=positions_used,
            expected_examples=num_examples_scored,
        )
    elif assignment_schema == LEGACY_CORRIDOR_ASSIGNMENTS_SCHEMA:
        warnings.append(
            "corridor mode assignments use legacy giant-json storage; "
            "packed_numpy_v1 is preferred"
        )
        assignment_items = assignments.get("assignments", [])
        if not isinstance(assignment_items, list):
            blockers.append("mode_assignments.assignments must be a list")
        else:
            if int(assignments.get("num_assignments") or -1) != len(assignment_items):
                blockers.append("mode_assignments.num_assignments mismatch")
            for item in assignment_items:
                if not isinstance(item, dict):
                    blockers.append("mode_assignments contains non-object assignment")
                    break
                if item.get("mode_id") not in mode_ids:
                    blockers.append("mode_assignments references nonexistent mode_id")
                    break
    else:
        blockers.append("mode_assignments schema_version mismatch")
    corridor_artifact_ok = not blockers
    return CorridorArtifactValidationResult(
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        corridor_artifact_ok=corridor_artifact_ok,
        corridor_fingerprints_ok=fingerprint_count >= 1 and not blockers,
        corridor_modes_ok=mode_count >= 1 and not blockers,
        corridor_mode_count=mode_count,
        corridor_fingerprint_count=fingerprint_count,
        corridor_observation_basis=(
            str(observation_basis) if observation_basis is not None else None
        ),
        degraded_corridor_export=degraded,
        corridor_positions_available=positions_available,
        corridor_positions_used=positions_used,
        corridor_mode_policy=mode_policy,
        corridor_stat_top_k=corridor_stat_top_k,
        corridor_assignment_storage_kind=(
            str(assignment_storage_kind)
            if assignment_storage_kind is not None
            else None
        ),
        corridor_assignment_count=assignment_count,
    )


def _corridor_observations(
    store: TeacherTargetStore,
    examples: tuple[TinyTextExample, ...],
    *,
    allow_degraded_score_only: bool = False,
) -> CorridorObservationExtraction:
    observations: list[_Observation] = []
    warnings: list[str] = []
    example_offset = 0
    positions_available = 0
    found_full_corridor_arrays = False
    for shard_id in range(store.metadata.shard_count):
        arrays = store.read_shard(shard_id)
        row_count = int(np.asarray(arrays["input_ids"]).shape[0])
        if _has_full_corridor_arrays(arrays):
            found_full_corridor_arrays = True
            shard_observations, shard_available, shard_warnings = (
                _full_corridor_observations_from_shard(
                    arrays,
                    examples=examples,
                    example_offset=example_offset,
                )
            )
            observations.extend(shard_observations)
            positions_available += shard_available
            warnings.extend(shard_warnings)
        else:
            positions_available += row_count * store.metadata.sequence_length
        example_offset += row_count
    if found_full_corridor_arrays:
        return CorridorObservationExtraction(
            observations=observations,
            observation_basis=FULL_TOKEN_POSITION_CORRIDOR,
            positions_available=positions_available,
            positions_used=len(observations),
            degraded=False,
            warnings=tuple(warnings),
        )
    if allow_degraded_score_only:
        return _score_selected_observations(
            store,
            examples,
            positions_available=max(
                positions_available,
                store.metadata.num_examples * store.metadata.sequence_length,
            ),
        )
    raise ValueError(
        "corridor_exemplar_v1 selected-only run did not retain full token-position "
        "corridor arrays"
    )


def _has_full_corridor_arrays(arrays: dict[str, np.ndarray]) -> bool:
    return all(
        name in arrays
        for name in (
            "corridor_top_token_ids",
            "corridor_teacher_entropy",
            "corridor_entropy",
            "corridor_top1_margin",
            "corridor_top8_mass",
            "corridor_top32_mass",
            "corridor_tail_mass",
            "corridor_confidence",
        )
    )


def _full_corridor_observations_from_shard(
    arrays: dict[str, np.ndarray],
    *,
    examples: tuple[TinyTextExample, ...],
    example_offset: int,
) -> tuple[list[_Observation], int, tuple[str, ...]]:
    top_ids = np.asarray(arrays["corridor_top_token_ids"])
    entropy = np.asarray(arrays["corridor_entropy"])
    top1_margin = np.asarray(arrays["corridor_top1_margin"])
    top8_mass = np.asarray(arrays["corridor_top8_mass"])
    top32_mass = np.asarray(arrays["corridor_top32_mass"])
    tail_mass = np.asarray(arrays["corridor_tail_mass"])
    confidence = np.asarray(arrays["corridor_confidence"])
    rows, sequence_length = top_ids.shape
    observations: list[_Observation] = []
    warnings: list[str] = []
    positions_available = 0
    for row in range(rows):
        example_index = example_offset + row
        example_id = (
            examples[example_index].example_id
            if example_index < len(examples)
            else f"row-{example_index:06d}"
        )
        length, length_warning = _corridor_row_length(
            arrays,
            row=row,
            sequence_length=sequence_length,
        )
        if length_warning is not None:
            warnings.append(length_warning)
        valid_length = min(max(length, 0), sequence_length)
        positions_available += valid_length
        for position in range(valid_length):
            observations.append(
                _Observation(
                    example_id=example_id,
                    position=position,
                    top_token_id=int(top_ids[row, position]),
                    entropy=float(entropy[row, position]),
                    confidence=float(confidence[row, position]),
                    top1_margin=float(top1_margin[row, position]),
                    top8_mass=float(top8_mass[row, position]),
                    top32_mass=float(top32_mass[row, position]),
                    tail_mass=float(tail_mass[row, position]),
                    length=valid_length,
                    source_policy_id=_source_policy_id(arrays, row, position),
                )
            )
    return observations, positions_available, tuple(sorted(set(warnings)))


def _corridor_row_length(
    arrays: dict[str, np.ndarray],
    *,
    row: int,
    sequence_length: int,
) -> tuple[int, str | None]:
    if "corridor_lengths" in arrays:
        return int(np.asarray(arrays["corridor_lengths"])[row]), None
    if "attention_mask" in arrays:
        return int(np.sum(np.asarray(arrays["attention_mask"])[row])), None
    return sequence_length, "corridor_lengths missing; used full sequence length"


def _source_policy_id(
    arrays: dict[str, np.ndarray],
    row: int,
    position: int,
) -> int:
    if "exemplar_source_policy_ids" in arrays:
        return int(np.asarray(arrays["exemplar_source_policy_ids"])[row, position])
    if "score_source_policy_ids" in arrays:
        return int(np.asarray(arrays["score_source_policy_ids"])[row])
    return 0


def _score_selected_observations(
    store: TeacherTargetStore,
    examples: tuple[TinyTextExample, ...],
    *,
    positions_available: int,
) -> CorridorObservationExtraction:
    observations: list[_Observation] = []
    example_offset = 0
    for shard_id in range(store.metadata.shard_count):
        arrays = store.read_shard(shard_id)
        row_count = int(np.asarray(arrays["input_ids"]).shape[0])
        for row in range(row_count):
            example_index = example_offset + row
            example_id = (
                examples[example_index].example_id
                if example_index < len(examples)
                else f"row-{example_index:06d}"
            )
            observations.append(_score_observation_from_row(arrays, row, example_id))
        example_offset += row_count
    return CorridorObservationExtraction(
        observations=observations,
        observation_basis=SCORE_SELECTED_POSITION_ONLY,
        positions_available=positions_available,
        positions_used=len(observations),
        degraded=True,
        warnings=(
            "corridor artifact was built from score-selected positions only; "
            "full corridor fingerprint export is missing",
        ),
    )


def _validate_selected_links(
    items: Any,
    blockers: list[str],
    *,
    source: str,
    valid_mode_ids: set[Any],
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


def _score_observation_from_row(
    arrays: dict[str, np.ndarray],
    row: int,
    example_id: str,
) -> _Observation:
    position = int(np.asarray(arrays["score_selected_position"])[row])
    length = int(np.asarray(arrays["score_lengths"])[row])
    if "score_top_token_id" in arrays:
        top_token_id = int(np.asarray(arrays["score_top_token_id"])[row])
    elif "corridor_top_token_ids" in arrays:
        top_token_id = int(np.asarray(arrays["corridor_top_token_ids"])[row, position])
    else:
        top_token_id = int(np.asarray(arrays["input_ids"])[row, position])
    return _Observation(
        example_id=example_id,
        position=position,
        top_token_id=top_token_id,
        entropy=float(np.asarray(arrays["score_selected_position_entropy"])[row]),
        confidence=float(
            np.asarray(arrays["score_confidence_at_selected_position"])[row]
        ),
        top1_margin=_score_stat_at_position(
            arrays,
            "corridor_top1_margin",
            row=row,
            position=position,
            default=0.0,
        ),
        top8_mass=_score_stat_at_position(
            arrays,
            "corridor_top8_mass",
            row=row,
            position=position,
            default=0.0,
        ),
        top32_mass=_score_stat_at_position(
            arrays,
            "corridor_top32_mass",
            row=row,
            position=position,
            default=0.0,
        ),
        tail_mass=_score_stat_at_position(
            arrays,
            "corridor_tail_mass",
            row=row,
            position=position,
            default=1.0,
        ),
        length=length,
        source_policy_id=int(np.asarray(arrays["score_source_policy_ids"])[row]),
    )


def _score_stat_at_position(
    arrays: dict[str, np.ndarray],
    name: str,
    *,
    row: int,
    position: int,
    default: float,
) -> float:
    if name not in arrays:
        return default
    return float(np.asarray(arrays[name])[row, position])


def _store_corridor_stat_top_k(store: TeacherTargetStore) -> int:
    raw = store.metadata.target_params.get("corridor_stat_top_k")
    if raw is None:
        return 0
    return int(raw)


def _fingerprints(
    observations: list[_Observation],
    *,
    target_type: str,
    source_target_type: str,
    num_examples_scored: int,
    num_positions_scored: int,
    fingerprint_policy: str,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], str]]:
    groups: dict[tuple[int, str, str, str], list[_Observation]] = {}
    for observation in observations:
        signature = (
            observation.top_token_id,
            _entropy_bucket(observation.entropy),
            _confidence_bucket(observation.confidence),
            _position_bucket(observation.position, observation.length),
        )
        groups.setdefault(signature, []).append(observation)
    fingerprints: list[dict[str, Any]] = []
    observation_to_fingerprint: dict[tuple[str, int], str] = {}
    sorted_groups = sorted(
        groups.items(),
        key=lambda item: (
            -len(item[1]),
            item[0][0],
            item[0][1],
            item[0][2],
            item[0][3],
        ),
    )
    for index, (signature, members) in enumerate(sorted_groups):
        fingerprint_id = f"fp_{index:06d}"
        for member in members:
            observation_to_fingerprint[member.key] = fingerprint_id
        entropies = [member.entropy for member in members]
        confidences = [member.confidence for member in members]
        representatives = sorted(
            members,
            key=lambda item: (-item.entropy, item.example_id, item.position),
        )[:3]
        fingerprints.append(
            {
                "fingerprint_id": fingerprint_id,
                "signature": {
                    "top_token_ids": [signature[0]],
                    "top_token_ranks": [0],
                    "bucket": "_".join(signature[1:]),
                    "entropy_bucket": signature[1],
                    "confidence_bucket": signature[2],
                    "position_bucket": signature[3],
                },
                "count": len(members),
                "mean_entropy": float(np.mean(entropies, dtype=np.float32)),
                "max_entropy": float(np.max(entropies)),
                "mean_confidence": float(np.mean(confidences, dtype=np.float32)),
                "positions_observed": len(members),
                "example_count": len({member.example_id for member in members}),
                "representatives": [
                    {
                        "example_id": member.example_id,
                        "position": member.position,
                        "entropy": member.entropy,
                        "confidence": member.confidence,
                    }
                    for member in representatives
                ],
                "target_type": target_type,
                "source_target_type": source_target_type,
                "num_examples_scored": num_examples_scored,
                "num_positions_scored": num_positions_scored,
                "fingerprint_policy": fingerprint_policy,
            }
        )
    return fingerprints, observation_to_fingerprint


def _modes(
    observations: list[_Observation],
    *,
    mode_policy: str,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], int]]:
    if mode_policy != CORRIDOR_MODE_POLICY:
        raise ValueError(f"unsupported corridor mode policy: {mode_policy}")
    groups: dict[tuple[int, int, int], list[_Observation]] = {}
    for observation in observations:
        groups.setdefault(_stat_band_mode_key(observation.stat_values), []).append(
            observation
        )
    if len(groups) > DEFAULT_CORRIDOR_MAX_MODES:
        raise ValueError(
            "corridor mode count exceeds DEFAULT_CORRIDOR_MAX_MODES for "
            f"{CORRIDOR_MODE_POLICY}: {len(groups)}"
        )
    modes: list[dict[str, Any]] = []
    observation_to_mode: dict[tuple[str, int], int] = {}
    denominator = max(len(observations), 1)
    for mode_id, (mode_key, members) in enumerate(sorted(groups.items())):
        for member in members:
            observation_to_mode[member.key] = mode_id
        representatives = sorted(
            members,
            key=lambda item: (-item.entropy, item.example_id, item.position),
        )[:3]
        entropy_bin, top1_margin_bin, top32_mass_bin = mode_key
        modes.append(
            {
                "mode_id": mode_id,
                "name": (
                    f"{CORRIDOR_MODE_POLICY}/e{entropy_bin}_m{top1_margin_bin}_"
                    f"t{top32_mass_bin}"
                ),
                "description": f"{CORRIDOR_MODE_POLICY} teacher-side corridor mode.",
                "mode_key": {
                    "entropy_bin": entropy_bin,
                    "top1_margin_bin": top1_margin_bin,
                    "top32_mass_bin": top32_mass_bin,
                },
                "record_count": len(members),
                "count": len(members),
                "share": float(len(members)) / float(denominator),
                "bounds": _mode_bounds(members),
                "representative_examples": [
                    {
                        "example_id": item.example_id,
                        "position": item.position,
                    }
                    for item in representatives
                ],
                "mode_policy": mode_policy,
            }
        )
    return modes, observation_to_mode


def _link_selected_exemplars(
    selected_records: list[dict[str, Any]],
    selected_payloads: list[dict[str, Any]],
    *,
    observation_to_fingerprint: dict[tuple[str, int], str],
    observation_to_mode: dict[tuple[str, int], int],
) -> int:
    linked = 0
    for collection in (selected_records, selected_payloads):
        for item in collection:
            key = (
                str(item.get("selected_example_id")),
                int(item.get("selected_position", -1)),
            )
            fingerprint_id = observation_to_fingerprint.get(key)
            mode_id = observation_to_mode.get(key)
            item["corridor_fingerprint_id"] = fingerprint_id
            item["corridor_mode_id"] = mode_id
            is_linked = mode_id is not None
            item["corridor_assignment_status"] = "linked" if is_linked else "missing"
            if collection is selected_records and is_linked:
                linked += 1
    return linked


def _mode_assignments(
    observations: list[_Observation],
    *,
    observation_to_fingerprint: dict[tuple[str, int], str],
    observation_to_mode: dict[tuple[str, int], int],
) -> list[dict[str, Any]]:
    assignments: list[dict[str, Any]] = []
    for observation in sorted(
        observations,
        key=lambda item: (item.example_id, item.position),
    ):
        mode_id = observation_to_mode[observation.key]
        assignments.append(
            {
                "example_id": observation.example_id,
                "position": observation.position,
                "mode_id": mode_id,
                "fingerprint_id": observation_to_fingerprint.get(observation.key),
                "weight": 1.0,
                "source": "corridor_position",
            }
        )
    return assignments


def _write_packed_mode_assignments(
    *,
    output_dir: Path,
    observations: list[_Observation],
    observation_to_fingerprint: dict[tuple[str, int], str],
    observation_to_mode: dict[tuple[str, int], int],
    observation_basis: str,
) -> dict[str, Any]:
    assignments_dir = output_dir / "corridors" / "mode_assignments"
    assignments_dir.mkdir(parents=True, exist_ok=True)
    count = len(observations)
    example_indices: dict[str, int] = {}
    position_example_index = np.empty((count,), dtype=np.int32)
    positions = np.empty((count,), dtype=np.int32)
    mode_ids = np.empty((count,), dtype=np.int32)
    weights = np.ones((count,), dtype=np.float32)
    fingerprint_indices = np.full((count,), -1, dtype=np.int32)
    fingerprint_id_to_index: dict[str, int] = {}
    for index, observation in enumerate(observations):
        example_index = example_indices.setdefault(
            observation.example_id,
            len(example_indices),
        )
        position_example_index[index] = example_index
        positions[index] = observation.position
        mode_ids[index] = observation_to_mode[observation.key]
        fingerprint_id = observation_to_fingerprint.get(observation.key)
        if fingerprint_id is not None:
            fingerprint_indices[index] = fingerprint_id_to_index.setdefault(
                fingerprint_id,
                len(fingerprint_id_to_index),
            )
    arrays = {
        "position_example_index": position_example_index,
        "position": positions,
        "mode_id": mode_ids,
        "weight": weights,
        "fingerprint_index": fingerprint_indices,
    }
    for name, array in arrays.items():
        np.save(assignments_dir / f"{name}.npy", array)
    metadata_path = assignments_dir / "examples_metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for example_id, example_index in sorted(
            example_indices.items(),
            key=lambda item: item[1],
        ):
            handle.write(
                json.dumps(
                    {
                        "example_index": example_index,
                        "example_id": example_id,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return {
        "schema_version": CORRIDOR_ASSIGNMENTS_SCHEMA,
        "assignment_policy": ASSIGNMENT_POLICY,
        "storage_kind": ASSIGNMENT_STORAGE_KIND,
        "corridor_observation_basis": observation_basis,
        "full_assignment_retained": True,
        "num_assignments": count,
        "num_examples": len(example_indices),
        "arrays": {
            name: {
                "path": f"corridors/mode_assignments/{name}.npy",
                "dtype": str(array.dtype),
                "shape": list(array.shape),
            }
            for name, array in arrays.items()
        },
        "examples_metadata": {
            "path": "corridors/mode_assignments/examples_metadata.jsonl",
            "num_examples": len(example_indices),
        },
    }


def _stat_band_mode_key(stats: dict[str, float]) -> tuple[int, int, int]:
    return (
        _bin_index(stats["entropy"], DEFAULT_ENTROPY_BINS),
        _bin_index(stats["top1_margin"], DEFAULT_TOP1_MARGIN_BINS),
        _bin_index(stats["top32_mass"], DEFAULT_TOP32_MASS_BINS),
    )


def _bin_index(value: float, bins: tuple[float, ...]) -> int:
    clamped = max(float(value), 0.0)
    for index, upper_bound in enumerate(bins[1:]):
        if clamped < upper_bound:
            return index
    return max(len(bins) - 2, 0)


def _mode_bounds(members: list[_Observation]) -> dict[str, dict[str, float]]:
    return {
        stat: _stat_bounds(
            [observation.stat_values[stat] for observation in members],
            clamp_unit=stat != "entropy",
        )
        for stat in CORRIDOR_TRACKED_STATS
    }


def _stat_bounds(values: list[float], *, clamp_unit: bool) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float32)
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    mean = float(np.mean(array, dtype=np.float32))
    if maximum - minimum < _CORRIDOR_MIN_WIDTH:
        midpoint = (maximum + minimum) / 2.0
        minimum = midpoint - (_CORRIDOR_MIN_WIDTH / 2.0)
        maximum = midpoint + (_CORRIDOR_MIN_WIDTH / 2.0)
    if clamp_unit:
        minimum = max(0.0, min(1.0, minimum))
        maximum = max(0.0, min(1.0, maximum))
        mean = max(0.0, min(1.0, mean))
    else:
        minimum = max(0.0, minimum)
        maximum = max(0.0, maximum)
        mean = max(0.0, mean)
    return {"min": minimum, "max": maximum, "mean": mean}


def _validate_modes_payload(
    modes_payload: dict[str, Any],
    blockers: list[str],
) -> set[Any]:
    mode_items = modes_payload.get("modes", [])
    if not isinstance(mode_items, list):
        blockers.append("corridor_modes.modes must be a list")
        return set()
    mode_ids: set[Any] = set()
    for mode in mode_items:
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
                continue
            if not {"min", "max", "mean"}.issubset(stat_bounds):
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
    num_assignments = int(manifest.get("num_assignments") or -1)
    num_examples = int(manifest.get("num_examples") or -1)
    if num_assignments != expected_count:
        blockers.append("mode_assignments.num_assignments mismatch")
    if num_examples != expected_examples:
        blockers.append("mode_assignments.num_examples mismatch")
    arrays = manifest.get("arrays")
    if not isinstance(arrays, dict):
        blockers.append("mode_assignments.arrays must be an object")
        return
    loaded: dict[str, np.ndarray] = {}
    expected_arrays = {
        "position_example_index": np.int32,
        "position": np.int32,
        "mode_id": np.int32,
        "weight": np.float32,
    }
    for name, dtype in expected_arrays.items():
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
    if set(expected_arrays).issubset(loaded):
        if np.any(loaded["position"] < 0):
            blockers.append("mode_assignments position contains negative values")
        example_index = loaded["position_example_index"]
        if np.any(example_index < 0) or np.any(example_index >= expected_examples):
            blockers.append("mode_assignments position_example_index out of range")
        if np.any(~np.isfinite(loaded["weight"])) or np.any(loaded["weight"] < 0.0):
            blockers.append("mode_assignments weight must be finite and nonnegative")
        valid_mode_ids = np.asarray(
            sorted(int(item) for item in mode_ids),
            dtype=np.int32,
        )
        if not np.isin(loaded["mode_id"], valid_mode_ids).all():
            blockers.append("mode_assignments references nonexistent mode_id")
    examples_metadata = manifest.get("examples_metadata")
    if not isinstance(examples_metadata, dict):
        blockers.append("mode_assignments.examples_metadata must be an object")
        return
    metadata_path = output_dir / str(examples_metadata.get("path", ""))
    if not metadata_path.is_file():
        blockers.append("mode_assignments examples_metadata missing")
    if int(examples_metadata.get("num_examples") or -1) != expected_examples:
        blockers.append("mode_assignments examples_metadata num_examples mismatch")


def _entropy_bucket(value: float) -> str:
    if value < 1.0:
        return "low"
    if value < 3.0:
        return "mid"
    if value < 6.0:
        return "high"
    return "very_high"


def _confidence_bucket(value: float) -> str:
    if value >= 0.75:
        return "high"
    if value >= 0.35:
        return "mid"
    if value >= 0.10:
        return "low"
    return "very_low"


def _position_bucket(position: int, length: int) -> str:
    if length <= 1 or position >= length - 1:
        return "final"
    ratio = float(position) / float(max(length - 1, 1))
    if ratio < 0.33:
        return "early"
    if ratio < 0.66:
        return "middle"
    return "late"


def _human_summary(
    *,
    num_examples: int,
    num_positions: int,
    corridor_positions_used: int,
    observation_basis: str,
    degraded: bool,
    mode_count: int,
    mode_policy: str,
    assignment_storage_kind: str,
    fingerprint_count: int,
    selected_count: int,
    selected_payload_retained: bool,
    non_selected_payload_retained: bool,
    delivery_path: str,
) -> str:
    lines = [
        "This Tome artifact contains:",
        f"  - {num_examples:,} scored examples",
        f"  - {num_positions:,} scored token positions",
        f"  - {corridor_positions_used:,} corridor positions used for fingerprinting",
        f"  - {mode_count:,} discovered corridor modes",
        f"  - corridor mode policy: {mode_policy}",
        f"  - corridor assignment storage: {assignment_storage_kind}",
        f"  - {fingerprint_count:,} corridor fingerprints",
        f"  - {selected_count:,} selected exemplars",
        "  - selected exemplar payloads retained: "
        f"{'yes' if selected_payload_retained else 'no'}",
        "  - non-selected exemplar payloads retained: "
        f"{'yes' if non_selected_payload_retained else 'no'}",
        f"  - corridor observation basis: {observation_basis}",
        f"  - delivery path: {delivery_path}",
    ]
    if degraded:
        lines.extend(
            (
                "",
                "WARNING: corridor export is degraded.",
                "Only score-selected positions were used for corridor modes.",
                "This is not a full fingerprint corridor artifact.",
            )
        )
    lines.append("")
    return "\n".join(lines)
