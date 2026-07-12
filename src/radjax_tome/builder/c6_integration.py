"""C6 integration contracts for corridor-first production artifacts.

The module is deliberately independent of teacher execution.  It validates and
reports the handoff from C4/C5 to delivery, curriculum, audit, and packaging
surfaces without changing C1-C5 selection math.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from radjax_tome.fingerprint.corridor_claims import (
    GLOBAL_BOARD_SUPPLY_SCHEMA,
    CorridorGlobalClaimResult,
)
from radjax_tome.fingerprint.multi_role_selection import (
    MultiRoleSelectionArtifact,
    payload_key_for_coordinate,
    validate_multi_role_selection_artifact,
)
from radjax_tome.io.json import read_json_object, write_json
from radjax_tome.targets.store import TeacherTargetStore

C6_SELECTION_INTEGRATION_POLICY = "corridor_first_global_backfill_v1"
GLOBAL_ONLY_SELECTION_POLICY = "global_only_v1"
C6_COVERAGE_REPORT_SCHEMA = "radjax.fingerprint_corridor_coverage.v1"
C6_VALIDATION_SCHEMA = "radjax.c6_integrated_selection_validation.v1"
C6_FEATURE_EXPORT_SCHEMA = "radjax.c6_corridor_feature_export.v1"
CURRICULUM_ROUTES_SCHEMA = "selected_exemplar_curriculum_routes_v1"


class C6IntegrationError(ValueError):
    """Actionable C6 provenance, parity, or package-integration error."""


def export_corridor_candidate_features(
    *,
    artifact_dir: Path,
    output_dir: Path,
) -> Path:
    """Stream strict C2 features from this Tome's packed corridor artifact."""

    assignments = read_json_object(artifact_dir / "corridors" / "mode_assignments.json")
    modes_payload = read_json_object(artifact_dir / "corridors" / "corridor_modes.json")
    arrays = assignments.get("arrays")
    if not isinstance(arrays, Mapping):
        raise C6IntegrationError("corridor assignments are missing packed arrays")
    required = ("position_example_index", "position", "mode_id")
    if any(name not in arrays for name in required):
        raise C6IntegrationError("corridor assignments are missing required arrays")
    metadata = assignments.get("examples_metadata")
    if not isinstance(metadata, Mapping) or not metadata.get("path"):
        raise C6IntegrationError("corridor assignments are missing examples metadata")
    example_ids = _assignment_example_ids(artifact_dir / str(metadata["path"]))
    mode_specs = {
        int(item["mode_id"]): item
        for item in modes_payload.get("modes", [])
        if isinstance(item, Mapping) and item.get("mode_id") is not None
    }
    if not mode_specs:
        raise C6IntegrationError("corridor modes are missing")
    position_examples = np_load(artifact_dir, arrays, "position_example_index")
    positions = np_load(artifact_dir, arrays, "position")
    mode_ids = np_load(artifact_dir, arrays, "mode_id")
    if not (len(position_examples) == len(positions) == len(mode_ids)):
        raise C6IntegrationError(
            "packed corridor assignment arrays have mismatched lengths"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "candidate_features.jsonl"
    temporary = output_dir / ".candidate_features.jsonl.tmp"
    store = TeacherTargetStore.open(artifact_dir)
    shard_ranges = _shard_ranges(store)
    cached_shard_id: int | None = None
    cached_shard: dict[str, np.ndarray] | None = None
    with temporary.open("w", encoding="utf-8") as handle:
        for assignment_index in range(len(mode_ids)):
            example_index = int(position_examples[assignment_index])
            position = int(positions[assignment_index])
            mode_id = int(mode_ids[assignment_index])
            if example_index < 0 or example_index >= len(example_ids):
                raise C6IntegrationError("corridor assignment example index is invalid")
            mode = mode_specs.get(mode_id)
            if mode is None:
                raise C6IntegrationError("corridor assignment references unknown mode")
            shard_id, row = _source_row_for_example(shard_ranges, example_index)
            if cached_shard_id != shard_id:
                cached_shard_id = shard_id
                cached_shard = store.read_shard(shard_id)
            if cached_shard is None:  # pragma: no cover - guarded above
                raise C6IntegrationError("corridor source shard is unavailable")
            shard = cached_shard
            entropy = float(np.asarray(shard["corridor_entropy"])[row, position])
            membership = _entropy_membership(entropy, mode)
            record = {
                "features": {
                    "candidate_id": example_ids[example_index],
                    "position": position,
                    "corridor_mode_id": mode_id,
                    "assignment_status": "linked",
                    "membership_strength": membership,
                    "core_distance": 1.0 - membership,
                    "mode_support": int(mode.get("record_count") or 0),
                    "difficulty_score": _entropy_difficulty(
                        entropy,
                        vocab_size=store.metadata.vocab_size,
                    ),
                },
                "fidelity": "derived",
                "source_artifact_schema": "radjax.corridor_artifact.v3",
                "membership_derivation": "mode_entropy_center_distance_v1",
                "core_distance_derivation": "one_minus_membership_strength_v1",
                "difficulty_derivation": "entropy_over_log_vocab_v1",
                "normalization_parameters": {
                    "membership": "abs(entropy-mode_mean)/max(mode_half_width,1e-6)",
                    "difficulty": "clamp(entropy/log(vocab_size),0,1)",
                    "assignment_manifest_sha256": _sha256(
                        artifact_dir / "corridors" / "mode_assignments.json"
                    ),
                    "modes_sha256": _sha256(
                        artifact_dir / "corridors" / "corridor_modes.json"
                    ),
                },
            }
            handle.write(
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            )
    os.replace(temporary, destination)
    manifest = {
        "schema_version": C6_FEATURE_EXPORT_SCHEMA,
        "feature_path": destination.name,
        "feature_sha256": _sha256(destination),
        "assignment_manifest_sha256": _sha256(
            artifact_dir / "corridors" / "mode_assignments.json"
        ),
        "modes_sha256": _sha256(artifact_dir / "corridors" / "corridor_modes.json"),
        "source_artifact": str(artifact_dir),
        "record_count": int(len(mode_ids)),
        "fidelity": "derived",
        "production_grade": True,
    }
    write_json(output_dir / "manifest.json", manifest)
    return destination


def load_curriculum_route_records(artifact_dir: Path) -> list[dict[str, Any]]:
    """Load the explicit delivery-produced curriculum routes."""

    payload = read_json_object(artifact_dir / "curriculum" / "selected_routes.json")
    if payload.get("schema_version") != CURRICULUM_ROUTES_SCHEMA:
        raise C6IntegrationError("curriculum routes schema is unsupported")
    routes = payload.get("routes")
    if not isinstance(routes, list) or any(
        not isinstance(item, dict) for item in routes
    ):
        raise C6IntegrationError("curriculum routes must be an object list")
    return [dict(item) for item in routes]


def _assignment_example_ids(path: Path) -> list[str]:
    rows: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            rows.append(str(payload["example_id"]))
    return rows


def np_load(root: Path, arrays: Mapping[str, Any], name: str) -> np.ndarray:
    descriptor = arrays[name]
    if not isinstance(descriptor, Mapping) or not descriptor.get("path"):
        raise C6IntegrationError(f"corridor assignment array is invalid: {name}")
    return np.load(root / str(descriptor["path"]), allow_pickle=False, mmap_mode="r")


def _shard_ranges(store: TeacherTargetStore) -> list[tuple[int, int, int]]:
    offset = 0
    ranges: list[tuple[int, int, int]] = []
    for shard_id in range(store.metadata.shard_count):
        count = int(np.asarray(store.read_shard(shard_id)["input_ids"]).shape[0])
        ranges.append((offset, offset + count, shard_id))
        offset += count
    return ranges


def _source_row_for_example(
    shard_ranges: Sequence[tuple[int, int, int]],
    example_index: int,
) -> tuple[int, int]:
    for start, end, shard_id in shard_ranges:
        if start <= example_index < end:
            return shard_id, example_index - start
    raise C6IntegrationError("corridor assignment references unavailable example")


def _entropy_membership(entropy: float, mode: Mapping[str, Any]) -> float:
    bounds = mode.get("bounds")
    if not isinstance(bounds, Mapping):
        raise C6IntegrationError("corridor mode is missing bounds")
    entropy_bounds = bounds.get("entropy")
    if not isinstance(entropy_bounds, Mapping):
        raise C6IntegrationError("corridor mode is missing entropy bounds")
    minimum = float(entropy_bounds["min"])
    maximum = float(entropy_bounds["max"])
    mean = float(entropy_bounds["mean"])
    width = max(abs(mean - minimum), abs(maximum - mean), 1e-6)
    return max(0.0, min(1.0, 1.0 - (abs(entropy - mean) / width)))


def _entropy_difficulty(entropy: float, *, vocab_size: int) -> float:
    return max(0.0, min(1.0, entropy / max(math.log(max(vocab_size, 2)), 1e-6)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def c5_records_for_delivery(
    selected: MultiRoleSelectionArtifact,
    *,
    delivery_path: str,
) -> list[dict[str, Any]]:
    """Translate verified C5 passports into the existing delivery record shape."""

    records: list[dict[str, Any]] = []
    for index, record in enumerate(selected.records, start=1):
        passport = dict(record.source_passport)
        required = (
            "source_shard_id",
            "source_row",
            "source_position",
            "source_score",
            "source_top_token_id",
        )
        missing = [field for field in required if field not in passport]
        if missing:
            raise C6IntegrationError(
                f"C5 source passport missing delivery fields for "
                f"{record.example_id}:{record.position}: {', '.join(missing)}"
            )
        source_position = int(passport["source_position"])
        if source_position != record.position:
            raise C6IntegrationError(
                f"C5 source passport position mismatch for {record.example_id}"
            )
        source_score = float(passport["source_score"])
        source_top_token_id = int(passport["source_top_token_id"])
        payload_ref = dict(passport.get("payload_slot") or {})
        payload_ref.update(
            {
                "kind": (
                    "corridor_exemplar_score_pass_v1"
                    if delivery_path == "two_pass_rerun_selected"
                    else "one_pass_candidate_v1"
                ),
                "source_shard_id": int(passport["source_shard_id"]),
                "source_row": int(passport["source_row"]),
                "source_position": source_position,
                "source_score": source_score,
                "source_top_token_id": source_top_token_id,
                "c5_authoritative_coordinate": True,
            }
        )
        rank_by_board: dict[str, int] = {}
        scores_by_board: dict[str, float] = {}
        for obligation in record.selection_obligations:
            if obligation.role == "global_board":
                rank_by_board[obligation.source_id] = obligation.rank
                if obligation.score is not None:
                    scores_by_board[obligation.source_id] = float(obligation.score)
        records.append(
            {
                "rank": index,
                "selection_index": record.selection_index,
                "selected_example_id": record.example_id,
                "selected_position": record.position,
                "selected_score": source_score,
                "score_selected_position_entropy": source_score,
                "score_top_token_id": source_top_token_id,
                "source_shard_id": int(passport["source_shard_id"]),
                "source_row": int(passport["source_row"]),
                "source_position": source_position,
                "source_score": source_score,
                "source_top_token_id": source_top_token_id,
                "source_score_policy": passport.get(
                    "source_score_policy", "entropy_top_n_v1"
                ),
                "payload_ref": payload_ref,
                "c5_authoritative_coordinate": True,
                "selected_policy": passport.get(
                    "source_score_policy", "entropy_top_n_v1"
                ),
                "source_delivery_path": delivery_path,
                "selected_board": "primary",
                "mode_key": (
                    record.represented_fingerprint_corridor_ids[0]
                    if record.represented_fingerprint_corridor_ids
                    else None
                ),
                "corridor_mode_id": (
                    record.represented_fingerprint_corridor_ids[0]
                    if record.represented_fingerprint_corridor_ids
                    else None
                ),
                "corridor_fingerprint_id": passport.get("corridor_fingerprint_id"),
                "corridor_assignment_status": passport.get(
                    "corridor_assignment_status",
                    "linked"
                    if record.represented_fingerprint_corridor_ids
                    else "not_applicable",
                ),
                "winning_boards": list(record.global_board_ids),
                "rank_by_board": rank_by_board,
                "scores_by_board": scores_by_board,
                "selection_roles": list(record.selection_roles),
                "selection_obligations": [
                    obligation.to_dict() for obligation in record.selection_obligations
                ],
                "payload_identity": dict(record.payload_identity),
            }
        )
    return records


def build_corridor_coverage_report(
    claims: CorridorGlobalClaimResult,
    selected: MultiRoleSelectionArtifact,
    *,
    c2_summary: Mapping[str, Any] | None = None,
    c3_summary: Mapping[str, Any] | None = None,
    global_supply: Mapping[str, Any] | None = None,
    delivery_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute the user-facing C6 coverage report from authoritative records."""

    validation = validate_multi_role_selection_artifact(
        selected,
        claims=claims,
        production_grade=claims.production_grade,
    )
    if validation.status == "fail":
        raise C6IntegrationError(
            "cannot report invalid C5 selection: " + "; ".join(validation.blockers)
        )
    corridor_claims = claims.corridor_claims
    global_claims = claims.global_claims
    mode_counts = Counter(str(item.corridor_mode_id) for item in corridor_claims)
    board_counts = Counter(item.board_id for item in global_claims)
    c3 = dict(c3_summary or {})
    allocation_rows = c3.get("mode_allocations", [])
    allocation_by_mode = {
        str(item["mode_id"]): dict(item)
        for item in allocation_rows
        if isinstance(item, Mapping) and item.get("mode_id") is not None
    }
    mode_ids = sorted(
        set(mode_counts) | set(allocation_by_mode),
        key=lambda item: int(item),
    )
    c2 = dict(c2_summary or {})
    requested_budget = claims.policy.total_selected_exemplar_budget
    corridor_budget = len(corridor_claims)
    global_budget = requested_budget - corridor_budget
    source = dict(claims.source_provenance)
    c2_source = dict(source.get("c2") or {})
    c3_source = dict(source.get("c3") or {})
    global_source = dict(source.get("global") or {})
    unresolved = sum(
        lineage.replacement_rank is None for lineage in claims.backfill_lineage
    )
    report = {
        "schema_version": C6_COVERAGE_REPORT_SCHEMA,
        "selection_integration_policy": C6_SELECTION_INTEGRATION_POLICY,
        "total_selected_budget": requested_budget,
        "corridor_budget_requested": int(
            c3.get("corridor_budget_ceiling", corridor_budget)
        ),
        "corridor_budget_ceiling": int(
            c3.get("corridor_budget_ceiling", corridor_budget)
        ),
        "corridor_budget_actual": corridor_budget,
        "global_budget": global_budget,
        "corridor_mode_count_observed": len(mode_ids),
        "corridor_modes_capacity_positive": int(
            c3.get("modes_with_positive_capacity", len(mode_ids))
        ),
        "corridor_modes_allocated": sum(
            int(item.get("allocated_slots") or 0) > 0
            for item in allocation_by_mode.values()
        ),
        "corridor_modes_fulfilled": sum(count > 0 for count in mode_counts.values()),
        "corridor_modes_uncovered": [
            {
                "mode_id": int(mode_id),
                "reason": allocation_by_mode.get(mode_id, {}).get(
                    "zero_allocation_reason", "unfulfilled"
                ),
            }
            for mode_id in mode_ids
            if mode_counts[mode_id] == 0
        ],
        "coverage_fraction": (
            0.0 if requested_budget == 0 else corridor_budget / requested_budget
        ),
        "claims_by_mode": {
            mode: {
                "allocated_slots": int(
                    allocation_by_mode.get(mode, {}).get("allocated_slots") or 0
                ),
                "claimed_count": mode_counts[mode],
                "fulfilled": mode_counts[mode]
                == int(allocation_by_mode.get(mode, {}).get("allocated_slots") or 0),
                "zero_allocation_reason": allocation_by_mode.get(mode, {}).get(
                    "zero_allocation_reason"
                ),
            }
            for mode in mode_ids
        },
        "candidate_pool_occupancy": c2.get("pool_occupancy", {}),
        "collision_count": len(claims.collision_obligations),
        "global_backfill_count": sum(claim.backfilled for claim in global_claims),
        "unresolved_backfill_count": unresolved,
        "multi_role_coordinate_count": int(
            selected.summary.get("multi_role_coordinate_count", 0)
        ),
        "counts_by_global_board": dict(sorted(board_counts.items())),
        "zero_allocation_reasons": list(c3.get("zero_allocation_reasons", [])),
        "feature_fidelity": c2_source.get("feature_fidelity"),
        "feature_derivation": {
            "c2_policy_id": c2_source.get("policy_id"),
            "c2_source_artifact_id": c2_source.get("source_artifact_id"),
        },
        "c2_schema_version": "radjax.c2_corridor_candidate_leaderboards.v1",
        "c3_schema_version": "radjax.c3_corridor_coverage_plan.v1",
        "c4_schema_version": "radjax.c4_corridor_global_claims.v1",
        "c5_schema_version": selected.schema_version,
        "c2_artifact_hash": c2_source.get("source_artifact_hash"),
        "c3_artifact_hash": c3_source.get("source_artifact_hash"),
        "c4_claims_sha256": selected.c4_claims_sha256,
        "c5_production_grade": selected.production_grade,
        "production_grade": bool(
            claims.production_grade
            and selected.production_grade
            and global_source.get("production_grade", False)
        ),
        "selected_unique_count": len(selected.records),
        "selected_obligation_count": int(selected.summary.get("obligation_count", 0)),
        "selected_coordinate_set_authoritative": "c5",
        "delivery_path": (delivery_report or {}).get("delivery_path"),
        "payload_count": (delivery_report or {}).get("num_selected_exemplars"),
        "claims_not_made": {
            "student_training_quality": True,
            "t4_rehearsal_executed": True,
            "tpu_jax_execution": True,
        },
        "t4_rehearsal_status": "not_executed",
    }
    if global_supply is not None:
        report["global_supply_schema_version"] = global_supply.get("schema_version")
        report["global_supply_source_production_grade"] = (
            global_supply.get("source_provenance", {}) or {}
        ).get("production_grade")
    return report


def validate_integrated_selection_contract(
    claims: CorridorGlobalClaimResult | None,
    selected: MultiRoleSelectionArtifact,
    *,
    legacy_records: Sequence[Mapping[str, Any]] | None = None,
    payload_records: Sequence[Mapping[str, Any]] | None = None,
    source_passports: Sequence[Mapping[str, Any]] | None = None,
    curriculum_records: Sequence[Mapping[str, Any]] | None = None,
    package_records: Sequence[Mapping[str, Any]] | None = None,
    audit_report: Mapping[str, Any] | None = None,
    production_grade: bool = True,
) -> dict[str, Any]:
    """Strictly compare every available C6 surface to the C5 coordinate set."""

    blockers: list[str] = []
    warnings: list[str] = []
    c5_validation = validate_multi_role_selection_artifact(
        selected,
        claims=claims,
        production_grade=production_grade,
    )
    blockers.extend(c5_validation.blockers)
    warnings.extend(c5_validation.warnings)
    expected = {(record.example_id, record.position) for record in selected.records}
    if len(expected) != len(selected.records):
        blockers.append("C5 coordinate set is not unique")
    if production_grade:
        if claims is not None and not claims.production_grade:
            blockers.append("production C6 requires production-grade C4/C5 sources")
        if not selected.production_grade:
            blockers.append("production C6 requires production-grade C4/C5 sources")
        if not source_passports:
            blockers.append("production C6 requires real source passports")
    _compare_surface("legacy projection", expected, legacy_records, blockers)
    _compare_surface("payload manifest", expected, payload_records, blockers)
    _validate_curriculum_routes(expected, curriculum_records, blockers)
    _compare_surface("package selected set", expected, package_records, blockers)
    if payload_records is not None:
        payload_keys: list[str] = []
        for item in payload_records:
            coordinate = _coordinate(item)
            if coordinate is None:
                continue
            payload_key = item.get("payload_key")
            if payload_key is None:
                payload_key = item.get("payload_identity", {}).get("payload_key")
            if payload_key is not None:
                payload_keys.append(str(payload_key))
            elif coordinate in expected:
                payload_keys.append(payload_key_for_coordinate(*coordinate))
        if len(payload_keys) != len(set(payload_keys)):
            blockers.append("payload manifest contains duplicate payload identities")
        if payload_records and len(payload_records) != len(expected):
            blockers.append("payload manifest count does not match C5 unique count")
    if source_passports is not None:
        passport_coordinates = set()
        for passport in source_passports:
            coordinate = _coordinate(passport)
            if coordinate is None:
                blockers.append("source passport is missing canonical identity")
                continue
            passport_coordinates.add(coordinate)
            if coordinate in expected:
                if not all(field in passport for field in ("example_id", "position")):
                    blockers.append("source passport identity is incomplete")
        if not expected.issubset(passport_coordinates):
            blockers.append("source passport index does not cover the C5 set")
        if len(passport_coordinates) != len(source_passports):
            blockers.append("source passports contain duplicates")
        for record in selected.records:
            passport = next(
                (
                    item
                    for item in source_passports
                    if _coordinate(item) == (record.example_id, record.position)
                ),
                None,
            )
            if passport is None:
                continue
            for field in ("source_shard_id", "source_row", "source_position"):
                if field not in passport:
                    blockers.append(
                        f"source passport missing required production field: {field}"
                    )
            if record.represented_fingerprint_corridor_ids:
                if (
                    passport.get("corridor_mode_id")
                    != record.represented_fingerprint_corridor_ids[0]
                ):
                    blockers.append("source passport corridor mode mismatch")
                if passport.get("corridor_assignment_status") != "linked":
                    blockers.append("source passport corridor assignment is not linked")
    if audit_report is not None:
        audit_count = _first_int(
            audit_report.get("selected_count"),
            audit_report.get("selected_unique_count"),
        )
        if audit_count is not None and audit_count != len(expected):
            blockers.append("selected-linkage audit count does not match C5")
        if audit_report.get("status") == "fail":
            blockers.append("selected-linkage audit status is fail")
    return {
        "schema_version": C6_VALIDATION_SCHEMA,
        "status": "fail" if blockers else ("warn" if warnings else "pass"),
        "blockers": list(dict.fromkeys(blockers)),
        "warnings": list(dict.fromkeys(warnings)),
        "selected_unique_count": len(expected),
        "selected_obligation_count": selected.summary.get("obligation_count", 0),
        "multi_role_coordinate_count": selected.summary.get(
            "multi_role_coordinate_count", 0
        ),
        "coordinate_set_authority": "c5",
        "curriculum_route_count": (
            len(curriculum_records) if curriculum_records is not None else None
        ),
    }


def write_corridor_coverage_report(
    report: Mapping[str, Any],
    path: str | Path,
) -> Path:
    """Atomically write a C6 coverage report."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    write_json(temporary, dict(report))
    os.replace(temporary, destination)
    return destination


def export_production_global_board_supply(
    selector_manifest: Mapping[str, Any],
    *,
    source_artifact_id: str,
    source_artifact_hash: str,
) -> dict[str, Any]:
    """Export ranked supply only from the production selector state.

    The development ``exemplar_selection_manifest_v1`` adapter is rejected
    unless the selector explicitly exports production-ranked supply.
    """

    if selector_manifest.get("selection_policy") != (
        "multi_leaderboard_exemplar_selector_v1"
    ):
        raise C6IntegrationError("unsupported global selector policy")
    if selector_manifest.get("production_global_selector") is not True:
        raise C6IntegrationError(
            "development global selector manifest cannot be used for production C6"
        )
    raw_boards = selector_manifest.get("boards")
    if not isinstance(raw_boards, list) or not raw_boards:
        raise C6IntegrationError("production global selector must export boards")
    boards: list[dict[str, Any]] = []
    for board in raw_boards:
        if not isinstance(board, Mapping):
            raise C6IntegrationError(
                "production global selector board must be an object"
            )
        ranked = board.get("ranked_candidates")
        if not isinstance(ranked, list) or not ranked:
            raise C6IntegrationError(
                "production global selector must export ranked_candidates"
            )
        capacity = int(board.get("capacity", len(ranked)))
        if capacity < 1:
            raise C6IntegrationError(
                "production global selector capacity must be positive"
            )
        candidates = []
        identities: set[tuple[str, int]] = set()
        previous_score: float | None = None
        for rank, candidate in enumerate(ranked, start=1):
            if not isinstance(candidate, Mapping):
                raise C6IntegrationError("global ranked candidate must be an object")
            if candidate.get("rank") is not None and int(candidate["rank"]) != rank:
                raise C6IntegrationError(
                    "global ranked candidate ranks must be contiguous"
                )
            example_id = str(candidate["example_id"])
            position = int(candidate["selected_position"])
            score = float(candidate["score"])
            if not example_id or position < 0:
                raise C6IntegrationError("global ranked candidate identity is invalid")
            if not math.isfinite(score):
                raise C6IntegrationError("global ranked candidate score must be finite")
            if previous_score is not None and score > previous_score:
                raise C6IntegrationError(
                    "global ranked candidates must be score descending"
                )
            identity = (example_id, position)
            if identity in identities:
                raise C6IntegrationError("global ranked candidates contain duplicates")
            identities.add(identity)
            previous_score = score
            candidates.append(
                {
                    "example_id": example_id,
                    "position": position,
                    "rank": rank,
                    "score": score,
                    "eligible": bool(candidate.get("eligible", True)),
                }
            )
        boards.append(
            {
                "board_id": str(board["board_id"]),
                "priority": int(board.get("priority", 0)),
                "requested_slots": capacity,
                "candidates": candidates,
            }
        )
    return {
        "schema_version": GLOBAL_BOARD_SUPPLY_SCHEMA,
        "source_provenance": {
            "source_artifact_id": source_artifact_id,
            "source_artifact_hash": source_artifact_hash,
            "selector_policy": selector_manifest["selection_policy"],
            "selector_schema_version": selector_manifest["schema_version"],
            "production_grade": True,
        },
        "boards": boards,
    }


def _compare_surface(
    label: str,
    expected: set[tuple[str, int]],
    records: Sequence[Mapping[str, Any]] | None,
    blockers: list[str],
) -> None:
    if records is None:
        return
    coordinates = [_coordinate(item) for item in records]
    if any(coordinate is None for coordinate in coordinates):
        blockers.append(f"{label} contains a record without canonical identity")
        return
    actual = {coordinate for coordinate in coordinates if coordinate is not None}
    if len(actual) != len(coordinates):
        blockers.append(f"{label} contains duplicate coordinates")
    if actual != expected:
        blockers.append(f"{label} coordinate set does not match C5")
    if len(records) != len(expected) and label != "curriculum union":
        blockers.append(f"{label} count does not match C5 unique count")


def _validate_curriculum_routes(
    expected: set[tuple[str, int]],
    routes: Sequence[Mapping[str, Any]] | None,
    blockers: list[str],
) -> None:
    if routes is None:
        return
    coordinates = [_coordinate(route) for route in routes]
    if any(coordinate is None for coordinate in coordinates):
        blockers.append("curriculum routes contain a record without canonical identity")
        return
    actual = {coordinate for coordinate in coordinates if coordinate is not None}
    if actual != expected:
        blockers.append("curriculum route coordinate union does not match C5")
    route_keys: set[tuple[str, int, str]] = set()
    for route, coordinate in zip(routes, coordinates, strict=True):
        if coordinate is None:
            continue
        board = route.get("curriculum_board")
        if not isinstance(board, str) or not board:
            blockers.append("curriculum route is missing curriculum_board")
            continue
        route_key = (*coordinate, board)
        if route_key in route_keys:
            blockers.append("curriculum routes contain duplicate board routes")
        route_keys.add(route_key)
        payload_key = route.get("payload_key")
        if payload_key is not None and payload_key != payload_key_for_coordinate(
            *coordinate
        ):
            blockers.append("curriculum route payload identity does not match C5")
    if len(routes) < len(expected):
        blockers.append("curriculum route count is below C5 unique count")


def _coordinate(item: Mapping[str, Any]) -> tuple[str, int] | None:
    example_id = item.get("example_id", item.get("selected_example_id"))
    position = item.get("position", item.get("selected_position"))
    if example_id is None or position is None:
        return None
    try:
        return str(example_id), int(position)
    except (TypeError, ValueError):
        return None


def _first_int(*values: Any) -> int | None:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None
