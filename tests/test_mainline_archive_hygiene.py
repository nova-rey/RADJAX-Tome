from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POINTERS = ROOT / "docs/TOME_ARCHIVE_POINTERS.md"
LEDGER = ROOT / "docs/TOME_MAINLINE_HYGIENE_LEDGER.json"
ARCHIVE_BRANCHES = ("archive/tome-migration-audit", "archive/tome-large-docs")
REQUIRED_LEDGER_KEYS = {
    "path",
    "main_action",
    "archive_branch",
    "classification",
    "reason",
    "replacement_or_pointer",
}
VALID_ACTIONS = {"removed_from_main", "kept_on_main", "summarized_on_main"}
VALID_CLASSIFICATIONS = {
    "historical_artifact",
    "large_generated_doc",
    "quarantine_reference",
    "forensic_audit_payload",
    "active_runtime_code",
    "active_test_code",
    "active_doc",
    "uncertain_keep",
}


def test_quarantine_tree_is_absent_on_main() -> None:
    assert not (ROOT / "quarantine").exists()


def test_archive_pointer_doc_mentions_both_archive_branches() -> None:
    text = POINTERS.read_text(encoding="utf-8")

    for branch in ARCHIVE_BRANCHES:
        assert branch in text
    assert (
        "git fetch origin archive/tome-migration-audit:archive/tome-migration-audit"
        in text
    )
    assert "git fetch origin archive/tome-large-docs:archive/tome-large-docs" in text
    assert "git switch main" in text


def test_mainline_hygiene_ledger_is_valid_compact_json() -> None:
    payload = json.loads(LEDGER.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "radjax_tome.mainline_hygiene_ledger.v1"
    assert payload["summary"]["status"] == "complete"
    assert set(payload["summary"]["archive_branches"]) == set(ARCHIVE_BRANCHES)
    assert len(LEDGER.read_text(encoding="utf-8").splitlines()) < 350

    entries = payload["entries"]
    assert entries
    for entry in entries:
        assert REQUIRED_LEDGER_KEYS <= entry.keys()
        assert entry["main_action"] in VALID_ACTIONS
        assert entry["classification"] in VALID_CLASSIFICATIONS
        assert entry["reason"]
        assert entry["replacement_or_pointer"]


def test_ledger_records_key_mainline_cleanup_decisions() -> None:
    entries = {
        entry["path"]: entry
        for entry in json.loads(LEDGER.read_text(encoding="utf-8"))["entries"]
    }

    assert entries["quarantine/"]["main_action"] == "removed_from_main"
    assert entries["quarantine/"]["classification"] == "quarantine_reference"
    assert (
        entries["docs/TOME_GENERATOR_BULK_MIGRATION_MANIFEST.json"]["classification"]
        == "large_generated_doc"
    )
    assert (
        entries["scripts/prove_tome_generation_capabilities.py"]["main_action"]
        == "kept_on_main"
    )


def test_active_python_files_do_not_import_quarantine_or_qrwkv_xla() -> None:
    forbidden = {"quarantine", "qrwkv_xla"}

    for path in _active_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
                assert names.isdisjoint(forbidden), path.relative_to(ROOT)
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in forbidden, path.relative_to(
                    ROOT
                )


def _active_python_files() -> list[Path]:
    paths: list[Path] = []
    for root_name in ("src", "scripts", "tests"):
        for path in (ROOT / root_name).rglob("*.py"):
            if "__pycache__" not in path.parts:
                paths.append(path)
    return paths
