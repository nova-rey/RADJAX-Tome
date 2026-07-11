"""C6 integration contracts for corridor-first production artifacts.

The module is deliberately independent of teacher execution.  It validates and
reports the handoff from C4/C5 to delivery, curriculum, audit, and packaging
surfaces without changing C1-C5 selection math.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from radjax_tome.fingerprint.corridor_claims import (
    GLOBAL_BOARD_SUPPLY_SCHEMA,
    CorridorGlobalClaimResult,
)
from radjax_tome.fingerprint.multi_role_selection import (
    MultiRoleSelectionArtifact,
    payload_key_for_coordinate,
    validate_multi_role_selection_artifact,
)
from radjax_tome.io.json import write_json

C6_SELECTION_INTEGRATION_POLICY = "corridor_first_global_backfill_v1"
GLOBAL_ONLY_SELECTION_POLICY = "global_only_v1"
C6_COVERAGE_REPORT_SCHEMA = "radjax.fingerprint_corridor_coverage.v1"
C6_VALIDATION_SCHEMA = "radjax.c6_integrated_selection_validation.v1"


class C6IntegrationError(ValueError):
    """Actionable C6 provenance, parity, or package-integration error."""


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
    mode_ids = sorted(mode_counts, key=lambda item: int(item))
    c3 = dict(c3_summary or {})
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
        "corridor_modes_allocated": len(mode_ids),
        "corridor_modes_fulfilled": len(mode_ids),
        "corridor_modes_uncovered": list(c3.get("uncovered_modes", [])),
        "coverage_fraction": (
            0.0 if requested_budget == 0 else corridor_budget / requested_budget
        ),
        "claims_by_mode": {
            mode: {"claimed_count": mode_counts[mode], "fulfilled": True}
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
    }
    if global_supply is not None:
        report["global_supply_schema_version"] = global_supply.get("schema_version")
        report["global_supply_source_production_grade"] = (
            global_supply.get("source_provenance", {}) or {}
        ).get("production_grade")
    return report


def validate_integrated_selection_contract(
    claims: CorridorGlobalClaimResult,
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
        if not claims.production_grade or not selected.production_grade:
            blockers.append("production C6 requires production-grade C4/C5 sources")
        if not source_passports:
            blockers.append("production C6 requires real source passports")
    _compare_surface("legacy projection", expected, legacy_records, blockers)
    _compare_surface("payload manifest", expected, payload_records, blockers)
    _compare_surface("curriculum union", expected, curriculum_records, blockers)
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
        if passport_coordinates != expected:
            blockers.append("source passport coordinate set does not match C5")
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
