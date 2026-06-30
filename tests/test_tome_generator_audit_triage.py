from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from radjax_tome.audit.triage import (
    build_migration_map,
    compute_spec3_gate,
    render_migration_map_markdown,
    triage_file_match,
    triage_missing_test,
)


def test_missing_producer_core_file_becomes_must_migrate_before_spec3() -> None:
    item = triage_file_match(
        _match("src/qrwkv_xla/targets/store.py", "producer_core"),
        high_risk=True,
    )

    assert item.bucket == "must_migrate_tome_before_spec3"
    assert item.blocks_spec3 is True


def test_missing_student_training_file_becomes_belongs_student() -> None:
    item = triage_file_match(
        _match("src/qrwkv_xla/training/student_backend.py", "producer_cli"),
        high_risk=True,
    )

    assert item.bucket == "belongs_student"
    assert item.proposed_destination_repo == "RADJAX-Student"


def test_missing_contract_schema_file_becomes_belongs_contract() -> None:
    item = triage_file_match(
        _match("src/qrwkv_xla/contracts/vocab.py", "producer_core"),
        high_risk=True,
    )

    assert item.bucket == "belongs_contract"
    assert item.blocks_spec3 is False


def test_mixed_producer_student_file_becomes_mixed_requires_split() -> None:
    item = triage_file_match(
        _match("src/qrwkv_xla/fingerprint/real_teacher.py", "mixed_producer_consumer"),
        high_risk=True,
    )

    assert item.bucket == "mixed_requires_split"
    assert item.blocks_spec3 is True


def test_deprecated_item_can_be_waived_as_historical() -> None:
    item = triage_file_match(
        _match("docs/deprecated_teacher_export.md", "producer_doc"),
        high_risk=False,
    )

    assert item.bucket == "historical_deprecated"


def test_duplicate_or_merged_item_points_to_new_path() -> None:
    item = triage_file_match(
        _match(
            "src/qrwkv_xla/artifacts/teacher_textbook_builder.py",
            "producer_core",
            new_paths=("src/radjax_tome/builder/teacher_textbook.py",),
        ),
        high_risk=False,
    )

    assert item.bucket == "duplicate_or_merged"
    assert "teacher_textbook.py" in item.proposed_destination_path_or_area


def test_high_risk_blocker_is_included_in_blocker_table(tmp_path: Path) -> None:
    audit = _audit(tmp_path)
    _add_match(audit, "src/qrwkv_xla/targets/store.py", "producer_core")
    _write_audit(tmp_path / "audit.json", audit)

    migration_map = build_migration_map(tmp_path / "audit.json")

    assert migration_map.high_risk_by_bucket["must_migrate_tome_before_spec3"] == 1
    assert migration_map.spec3_gate.passed is False


def test_missing_producer_test_is_linked_to_migration_bucket() -> None:
    item = triage_missing_test(
        {
            "old_path": "tests/test_topk_tail_textbook.py",
            "status": "missing",
            "new_paths": [],
        }
    )

    assert item.test_bucket == "must_port_before_spec3"
    assert item.linked_migration_bucket == "must_migrate_tome_before_spec3"


def test_spec3_gate_fails_when_high_risk_producer_gaps_exist() -> None:
    item = triage_file_match(
        _match("src/qrwkv_xla/targets/store.py", "producer_core"),
        high_risk=True,
    )

    gate = compute_spec3_gate((item,), ())

    assert gate.passed is False
    assert gate.high_risk_must_migrate_tome_before_spec3 == 1


def test_spec3_gate_can_pass_when_gaps_are_waived_or_nonblocking() -> None:
    contract = triage_file_match(
        _match("src/qrwkv_xla/contracts/vocab.py", "contract_only"),
        high_risk=True,
    )

    gate = compute_spec3_gate((contract,), ())

    assert gate.passed is True


def test_markdown_report_contains_roadmap_and_spec3_gate(tmp_path: Path) -> None:
    audit = _audit(tmp_path)
    _add_match(audit, "src/qrwkv_xla/contracts/vocab.py", "contract_only")
    _write_audit(tmp_path / "audit.json", audit)
    migration_map = build_migration_map(tmp_path / "audit.json")

    markdown = render_migration_map_markdown(migration_map)

    assert "## Short-Term Roadmap" in markdown
    assert "## Spec 3 Gate" in markdown


def test_fail_on_untriaged_high_risk_behavior(tmp_path: Path) -> None:
    audit = _audit(tmp_path)
    audit["file_matches"].append(
        {
            "old_path": "src/qrwkv_xla/mystery.py",
            "old_role": "producer_cli",
            "old_symbols_or_entrypoints": [],
            "new_status": "needs_human_review",
            "new_path_or_paths": [],
            "migration_confidence": "low",
            "notes": "",
            "action_required": "",
        }
    )
    audit["high_risk_missing_items"].append(
        {
            "old_path": "src/qrwkv_xla/mystery.py",
            "role": "producer_cli",
            "status": "needs_human_review",
            "action_required": "review",
        }
    )
    audit_path = tmp_path / "audit.json"
    _write_audit(audit_path, audit)
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "triage_tome_generator_audit.py"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--audit-json",
            str(audit_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--overwrite",
            "--fail-on-untriaged-high-risk",
        ],
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1


def _match(
    old_path: str,
    old_role: str,
    *,
    new_status: str = "missing",
    new_paths: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "old_path": old_path,
        "old_role": old_role,
        "old_symbols_or_entrypoints": [],
        "new_status": new_status,
        "new_path_or_paths": list(new_paths),
        "migration_confidence": "high",
        "notes": "",
        "action_required": "",
    }


def _audit(tmp_path: Path) -> dict[str, object]:
    del tmp_path
    return {
        "summary": {
            "producer_relevant_old_files": 0,
            "missing": 0,
            "partial": 0,
            "producer_relevant_old_tests": 0,
            "missing_tests": 0,
        },
        "file_matches": [],
        "high_risk_missing_items": [],
        "test_inventory": [],
    }


def _add_match(audit: dict[str, object], path: str, role: str) -> None:
    audit["file_matches"].append(_match(path, role))  # type: ignore[index,union-attr]
    audit["high_risk_missing_items"].append(  # type: ignore[index,union-attr]
        {
            "old_path": path,
            "role": role,
            "status": "missing",
            "action_required": "review",
        }
    )


def _write_audit(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
