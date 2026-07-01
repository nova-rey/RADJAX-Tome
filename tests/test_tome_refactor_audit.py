from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from radjax_tome.audit.refactor_surface import (
    SCORECARD_CATEGORIES,
    VALID_SEVERITIES,
    run_refactor_audit,
    write_refactor_audit,
)

ROOT = Path(__file__).resolve().parents[1]
SUBPROCESS_ENV = {"PYTHONPATH": str(ROOT / "src")}
REQUIRED_TOP_LEVEL = {
    "schema_version",
    "repo",
    "commit",
    "generated_at",
    "summary",
    "scorecard",
    "file_metrics",
    "boundary_findings",
    "duplication_findings",
    "api_findings",
    "script_findings",
    "test_findings",
    "doc_findings",
    "checklist",
}
REQUIRED_MARKDOWN_SECTIONS = (
    "## Executive Summary",
    "## Spec 3 Readiness",
    "## Modularity Scorecard",
    "## Top Refactor Checklist",
    "## Must Fix Before Spec 3",
    "## Should Fix Before Production Burns",
    "## Can Wait",
    "## File Size and Complexity Hotspots",
    "## Boundary Findings",
    "## Duplication Findings",
    "## API Surface Findings",
    "## Script Thinness Findings",
    "## Test Suite Findings",
    "## Documentation Hygiene",
    "## Recommended Follow-up Specs",
)
REQUIRED_CHECKLIST_FIELDS = {
    "id",
    "title",
    "severity",
    "category",
    "files",
    "problem",
    "evidence",
    "recommended_change",
    "expected_benefit",
    "risk",
    "suggested_spec",
    "must_preserve",
}


def test_refactor_audit_script_runs(tmp_path: Path) -> None:
    json_out = tmp_path / "audit.json"
    md_out = tmp_path / "audit.md"
    script = ROOT / "scripts" / "audit_tome_refactor_surface.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo-root",
            str(ROOT),
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
        ],
        cwd=ROOT,
        env=SUBPROCESS_ENV,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "status=complete" in result.stdout
    assert json_out.is_file()
    assert md_out.is_file()


def test_refactor_audit_json_has_required_shape(tmp_path: Path) -> None:
    payload = _run_tmp_audit(tmp_path)

    assert REQUIRED_TOP_LEVEL <= payload.keys()
    assert payload["schema_version"] == "refactor_audit_v1"
    assert payload["summary"]["status"] == "complete"


def test_refactor_audit_markdown_has_required_sections(tmp_path: Path) -> None:
    json_out = tmp_path / "audit.json"
    md_out = tmp_path / "audit.md"
    audit = run_refactor_audit(ROOT)
    write_refactor_audit(audit, json_out=json_out, md_out=md_out)
    markdown = md_out.read_text(encoding="utf-8")

    for section in REQUIRED_MARKDOWN_SECTIONS:
        assert section in markdown


def test_file_metrics_include_active_python_files(tmp_path: Path) -> None:
    payload = _run_tmp_audit(tmp_path)
    paths = {item["path"] for item in payload["file_metrics"]}

    assert "src/radjax_tome/targets/store.py" in paths
    assert "scripts/audit_tome_refactor_surface.py" in paths


def test_checklist_items_have_required_fields_and_valid_severities(
    tmp_path: Path,
) -> None:
    payload = _run_tmp_audit(tmp_path)

    assert payload["checklist"]
    for item in payload["checklist"]:
        assert REQUIRED_CHECKLIST_FIELDS <= item.keys()
        assert item["severity"] in VALID_SEVERITIES


def test_scorecard_categories_are_present(tmp_path: Path) -> None:
    payload = _run_tmp_audit(tmp_path)
    categories = {item["category"] for item in payload["scorecard"]}

    assert set(SCORECARD_CATEGORIES) == categories


def test_optional_dependency_imports_are_classified(tmp_path: Path) -> None:
    payload = _run_tmp_audit(tmp_path)
    imports = payload["optional_dependency_imports"]

    assert any(item["import"] == "transformers" for item in imports)
    assert all(item["classification"] for item in imports)


def test_excluded_paths_are_not_scanned(tmp_path: Path) -> None:
    payload = _run_tmp_audit(tmp_path)
    paths = [item["path"] for item in payload["file_metrics"]]

    assert not any(path.startswith("quarantine/") for path in paths)
    assert not any(path.startswith("artifacts/") for path in paths)
    assert not any(".venv/" in path for path in paths)


def test_medium_or_low_candidates_do_not_fail_audit(tmp_path: Path) -> None:
    payload = _run_tmp_audit(tmp_path)

    assert payload["summary"]["status"] == "complete"
    assert payload["summary"]["spec3_blocked"] is False
    assert payload["summary"]["medium_count"] + payload["summary"]["low_count"] >= 1


def _run_tmp_audit(tmp_path: Path) -> dict[str, object]:
    json_out = tmp_path / "audit.json"
    md_out = tmp_path / "audit.md"
    audit = run_refactor_audit(ROOT)
    write_refactor_audit(audit, json_out=json_out, md_out=md_out)
    return json.loads(json_out.read_text(encoding="utf-8"))
