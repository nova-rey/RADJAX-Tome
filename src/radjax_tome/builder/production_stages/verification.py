"""Native validation, linkage, reconciliation, and cover stage operations."""

from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
from typing import Any

from radjax_tome.audit import audit_selected_linkage, write_selected_linkage_audit
from radjax_tome.builder.c6_integration import (
    build_corridor_coverage_report,
    load_curriculum_route_records,
    validate_integrated_selection_contract,
    write_corridor_coverage_report,
)
from radjax_tome.builder.production_stages.evidence import native_file_evidence
from radjax_tome.builder.teacher_textbook import (
    validate_teacher_textbook,
    write_teacher_textbook_validation_report,
)
from radjax_tome.io.json import read_json_object, write_json
from radjax_tome.tome import write_cover_page


def finalize_c6_selection(
    config: Any,
    context: dict[str, Any],
    *,
    delivery_report: dict[str, Any] | None,
    audit_report: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    claims = context["claims"]
    selected = context["selected"]
    legacy: list[Mapping[str, Any]] = []
    payloads: list[Mapping[str, Any]] = []
    selected_path = config.output_dir / "leaderboards" / "selected_exemplars.json"
    payload_path = config.output_dir / "selected_exemplars" / "payload_index.json"
    if selected_path.is_file():
        legacy = list(read_json_object(selected_path).get("selected_exemplars") or [])
    if payload_path.is_file():
        payloads = list(read_json_object(payload_path).get("selected_exemplars") or [])
    else:
        legacy_payload_path = (
            config.output_dir / "selected_exemplars" / "selected-exemplars-00000.json"
        )
        if legacy_payload_path.is_file():
            payloads = list(
                read_json_object(legacy_payload_path).get("selected_exemplars") or []
            )
    try:
        curriculum_records: list[Mapping[str, Any]] = load_curriculum_route_records(
            config.output_dir
        )
    except (OSError, TypeError, ValueError) as exc:
        curriculum_records = [{"curriculum_load_error": str(exc)}]
    if delivery_report is None:
        validation = {
            "schema_version": "radjax.c6_integrated_selection_validation.v1",
            "status": "fail",
            "blockers": ["C6 selected delivery did not complete"],
            "warnings": [],
            "selected_unique_count": len(selected.records),
            "selected_obligation_count": selected.summary.get("obligation_count", 0),
            "coordinate_set_authority": "c5",
        }
    else:
        validation = validate_integrated_selection_contract(
            claims,
            selected,
            legacy_records=legacy,
            payload_records=payloads,
            source_passports=context["source_passports"],
            curriculum_records=curriculum_records,
            audit_report=audit_report,
            production_grade=True,
        )
    coverage = build_corridor_coverage_report(
        claims,
        selected,
        c2_summary=context.get("c2_summary"),
        c3_summary=context.get("c3_summary"),
        global_supply=context.get("global_supply"),
        delivery_report=delivery_report,
    )
    coverage["integrated_validation_status"] = validation["status"]
    coverage["integrated_validation_path"] = (
        "reports/c6_integrated_selection_validation.json"
    )
    return validation, coverage


def native_validation_linkage_operation(state: Any, inputs: Any) -> Any:
    from radjax_tome.builder.native_path_b.contracts import StageResult
    from radjax_tome.builder.native_path_b.verification import ValidationLinkageHandoff

    output = state.config.output_dir
    state.progress.validation_started()
    started = perf_counter()
    validation = validate_teacher_textbook(output)
    write_teacher_textbook_validation_report(
        validation, output / "validation_report.json"
    )
    wall_seconds = perf_counter() - started
    state.progress.memory_checkpoint("validation_complete")
    state.progress.validation_completed(validation.status)
    linkage_audit = audit_selected_linkage(output, strict=True)
    write_selected_linkage_audit(linkage_audit, output / "selected_linkage_audit.json")
    evidence = native_file_evidence(
        "validation_linkage",
        (output / "validation_report.json", output / "selected_linkage_audit.json"),
        prior=inputs.assembly_evidence,
    )
    return StageResult(
        status="pass",
        value=ValidationLinkageHandoff(
            value={
                **inputs.assembly,
                "validation": validation,
                "linkage_audit": linkage_audit,
                "validation_wall_seconds": wall_seconds,
            },
            stage_evidence=evidence,
        ),
        evidence=evidence,
    )


def native_reconciliation_cover_operation(state: Any, inputs: Any) -> Any:
    from radjax_tome.builder.native_path_b.contracts import StageResult
    from radjax_tome.builder.native_path_b.verification import (
        ReconciliationCoverHandoff,
    )

    output = state.config.output_dir
    value = inputs.validation
    c6_validation, c6_coverage = finalize_c6_selection(
        state.config,
        value["context"],
        delivery_report=value["delivery_report"],
        audit_report=value["linkage_audit"].to_dict(),
    )
    audit_payload = value["linkage_audit"].to_dict()
    audit_payload["c6_integration"] = {
        "status": c6_validation["status"],
        "selected_unique_count": c6_validation["selected_unique_count"],
        "selected_obligation_count": c6_validation["selected_obligation_count"],
        "coordinate_set_authority": "c5",
    }
    write_json(output / "selected_linkage_audit.json", audit_payload)
    (output / "reports").mkdir(parents=True, exist_ok=True)
    write_json(
        output / "reports" / "c6_integrated_selection_validation.json", c6_validation
    )
    write_corridor_coverage_report(
        c6_coverage, output / "reports" / "fingerprint_corridor_coverage.json"
    )
    if c6_validation["status"] == "fail":
        state.blockers.extend(str(item) for item in c6_validation["blockers"])
    if value["validation"].status == "pass":
        write_cover_page(output)
    evidence = native_file_evidence(
        "reconciliation_cover",
        (
            output / "reports" / "c6_integrated_selection_validation.json",
            output / "reports" / "fingerprint_corridor_coverage.json",
            output / "cover_page.json",
        ),
        prior=inputs.validation_evidence,
    )
    return StageResult(
        status="pass",
        value=ReconciliationCoverHandoff(
            value={**value, "c6_validation": c6_validation}, stage_evidence=evidence
        ),
        evidence=evidence,
    )
