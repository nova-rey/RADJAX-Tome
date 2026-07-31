"""Builder-independent validation of the C5/C6 artifact handoff."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from radjax_tome.fingerprint.corridor_claims import CorridorGlobalClaimResult
from radjax_tome.fingerprint.multi_role_selection import (
    MultiRoleSelectionArtifact,
    payload_key_for_coordinate,
    validate_multi_role_selection_artifact,
)
from radjax_tome.io.json import read_json_object

C6_VALIDATION_SCHEMA = "radjax.c6_integrated_selection_validation.v1"
CURRICULUM_ROUTES_SCHEMA = "selected_exemplar_curriculum_routes_v1"


class C6IntegrationError(ValueError):
    """Actionable C6 provenance, parity, or package-integration error."""


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
        selected, claims=claims, production_grade=production_grade
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
                identity = item.get("payload_identity", {})
                payload_key = (
                    identity.get("payload_key")
                    if isinstance(identity, Mapping)
                    else None
                )
            payload_keys.append(
                str(payload_key)
                if payload_key is not None
                else payload_key_for_coordinate(*coordinate)
            )
        if len(payload_keys) != len(set(payload_keys)):
            blockers.append("payload manifest contains duplicate payload identities")
        if payload_records and len(payload_records) != len(expected):
            blockers.append("payload manifest count does not match C5 unique count")
    if source_passports is not None:
        passport_coordinates: set[tuple[str, int]] = set()
        for passport in source_passports:
            coordinate = _coordinate(passport)
            if coordinate is None:
                blockers.append("source passport is missing canonical identity")
                continue
            passport_coordinates.add(coordinate)
            if coordinate in expected and not all(
                field in passport for field in ("example_id", "position")
            ):
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
        "curriculum_route_count": len(curriculum_records)
        if curriculum_records is not None
        else None,
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
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return None
