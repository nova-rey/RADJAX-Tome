"""Producer-artifact validation adapters used by package materialization.

This keeps production-specific audit and C6 compatibility calls outside the
packaging layer.  It is not a second validation authority: native audit and
C6 validators retain their established semantics and reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from radjax_tome.artifact_validation.long_tail import long_tail_summary
from radjax_tome.artifact_validation.selection import (
    load_curriculum_route_records,
    validate_integrated_selection_contract,
)
from radjax_tome.artifact_validation.teacher_textbook import validate_teacher_textbook
from radjax_tome.io.json import read_json_object


def validate_full_debug_producer(root: Path) -> tuple[str, tuple[str, ...]]:
    report = validate_teacher_textbook(root)
    return report.status, tuple(report.blockers)


def audit_selected_package(root: Path, *, profile: str) -> Any:
    from radjax_tome.audit import audit_selected_linkage

    return audit_selected_linkage(root, strict=True, profile=profile)


def summarize_long_tail(records: list[dict[str, Any]]) -> dict[str, Any]:
    return long_tail_summary(records)


def validate_c6_package_parity(
    root: Path,
    *,
    audit_report: dict[str, Any],
    read_selected_payloads: Any,
) -> dict[str, Any] | None:
    c5_root = root / "c6" / "multi-role-selection"
    if not c5_root.is_dir():
        return None
    from radjax_tome.fingerprint.multi_role_selection import (
        load_multi_role_selection_artifact,
    )

    selected = load_multi_role_selection_artifact(c5_root, production_grade=False)
    leaderboard = read_json_object(root / "leaderboards" / "selected_exemplars.json")
    records = list(leaderboard.get("selected_exemplars") or [])
    payloads = read_selected_payloads(root)
    routes = load_curriculum_route_records(root)
    return validate_integrated_selection_contract(
        None,
        selected,
        legacy_records=records,
        payload_records=payloads,
        source_passports=[dict(record.source_passport) for record in selected.records],
        curriculum_records=routes,
        package_records=records,
        audit_report=audit_report,
        production_grade=False,
    )
