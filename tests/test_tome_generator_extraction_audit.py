from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from radjax_tome.audit.extraction_inventory import (
    classify_file,
    render_markdown,
    run_extraction_audit,
    scan_repo,
    write_audit_reports,
)


def test_keyword_scanner_finds_obvious_producer_files(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "old", package="qrwkv_xla")
    _write(
        repo / "src/qrwkv_xla/artifacts/teacher_textbook_builder.py",
        "class TeacherTextbookBuildConfig: pass\ndef build_teacher_textbook(): pass\n",
    )

    items = scan_repo(repo)

    assert any(item.path.endswith("teacher_textbook_builder.py") for item in items)
    assert any(item.role == "producer_core" for item in items)


def test_student_only_file_is_not_producer_core() -> None:
    role = classify_file(
        "src/qrwkv_xla/training/student_backend.py",
        "student backend optimizer checkpoint train step",
    )

    assert role == "student_only"


def test_producer_cli_gets_classified_as_producer_cli() -> None:
    role = classify_file(
        "scripts/export_teacher_targets.py",
        "export_teacher_targets teacher target_type dense_logits",
        ("export_teacher_targets", "target_type", "dense_logits"),
    )

    assert role == "producer_cli"


def test_old_file_with_obvious_new_equivalent_is_marked_migrated(
    tmp_path: Path,
) -> None:
    old = _repo(tmp_path / "old", package="qrwkv_xla")
    new = _repo(tmp_path / "new", package="radjax_tome")
    _write(
        old / "src/qrwkv_xla/targets/store.py",
        "class TeacherTargetStore: pass\n",
    )
    _write(
        new / "src/radjax_tome/targets/store.py",
        "class TeacherTargetStore: pass\n",
    )

    report = run_extraction_audit(old, new)

    match = _match(report, "src/qrwkv_xla/targets/store.py")
    assert match["new_status"] == "migrated"


def test_old_file_without_equivalent_is_marked_missing(tmp_path: Path) -> None:
    old = _repo(tmp_path / "old", package="qrwkv_xla")
    new = _repo(tmp_path / "new", package="radjax_tome")
    _write(
        old / "src/qrwkv_xla/artifacts/extra_teacher_export.py",
        "def export_teacher_targets(): pass\n",
    )

    report = run_extraction_audit(old, new)

    match = _match(report, "src/qrwkv_xla/artifacts/extra_teacher_export.py")
    assert match["new_status"] == "missing"


def test_mixed_producer_consumer_file_is_marked_mixed() -> None:
    role = classify_file(
        "src/qrwkv_xla/fingerprint/capture.py",
        "capture_fingerprint FingerprintCapture capture_summary optimizer train step",
        ("capture_fingerprint", "FingerprintCapture", "capture_summary"),
    )

    assert role == "mixed_producer_consumer"


def test_audit_json_schema_includes_required_top_level_fields(tmp_path: Path) -> None:
    old = _repo(tmp_path / "old", package="qrwkv_xla")
    new = _repo(tmp_path / "new", package="radjax_tome")
    _write(old / "scripts/build_teacher_textbook.py", "build_teacher_textbook\n")
    _write(new / "scripts/build_teacher_textbook.py", "build_teacher_textbook\n")

    report = run_extraction_audit(old, new).to_dict()

    for key in (
        "summary",
        "old_candidates",
        "new_candidates",
        "file_matches",
        "cli_inventory",
        "test_inventory",
        "recommendation",
    ):
        assert key in report


def test_audit_markdown_includes_recommendation_section(tmp_path: Path) -> None:
    old = _repo(tmp_path / "old", package="qrwkv_xla")
    new = _repo(tmp_path / "new", package="radjax_tome")
    report = run_extraction_audit(old, new)

    markdown = render_markdown(report)

    assert "## Recommendation" in markdown


def test_missing_producer_tests_are_counted(tmp_path: Path) -> None:
    old = _repo(tmp_path / "old", package="qrwkv_xla")
    new = _repo(tmp_path / "new", package="radjax_tome")
    _write(
        old / "tests/test_teacher_textbook_builder.py",
        "def test_build_teacher_textbook():\n    assert 'TeacherTextbook'\n",
    )

    report = run_extraction_audit(old, new)

    assert report.summary["producer_relevant_old_tests"] == 1
    assert report.summary["missing_tests"] == 1


def test_fail_on_blockers_logic_detects_high_risk_missing_items(
    tmp_path: Path,
) -> None:
    old = _repo(tmp_path / "old", package="qrwkv_xla")
    new = _repo(tmp_path / "new", package="radjax_tome")
    output = tmp_path / "out"
    _write(
        old / "src/qrwkv_xla/artifacts/missing_builder.py",
        "def build_teacher_textbook(): pass\n",
    )

    report = run_extraction_audit(old, new)
    write_audit_reports(report, output, overwrite=True)

    assert report.blockers_before_spec3
    assert (output / "extraction_audit.json").is_file()


def test_script_runs_against_synthetic_repos(tmp_path: Path) -> None:
    old = _repo(tmp_path / "old", package="qrwkv_xla")
    new = _repo(tmp_path / "new", package="radjax_tome")
    _write(old / "scripts/build_teacher_textbook.py", "build_teacher_textbook\n")
    _write(new / "scripts/build_teacher_textbook.py", "build_teacher_textbook\n")
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "audit_tome_generator_extraction.py"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--old-repo",
            str(old),
            "--new-repo",
            str(new),
            "--output-dir",
            str(tmp_path / "report"),
            "--overwrite",
        ],
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads((tmp_path / "report" / "extraction_audit.json").read_text())
    assert "summary" in payload


def test_real_audit_runs_when_archived_repo_is_configured(tmp_path: Path) -> None:
    import os

    old_repo = os.environ.get("QRWKV_XLA_OLD_REPO")
    if not old_repo:
        pytest.skip("QRWKV_XLA_OLD_REPO not set")

    report = run_extraction_audit(
        Path(old_repo),
        Path(__file__).resolve().parents[1],
    )

    assert report.summary["producer_relevant_old_files"] > 0


def _repo(root: Path, *, package: str) -> Path:
    (root / "src" / package).mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "tests").mkdir()
    return root


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _match(report, old_path: str) -> dict[str, object]:
    for match in report.to_dict()["file_matches"]:
        if match["old_path"] == old_path:
            return match
    raise AssertionError(f"missing match for {old_path}")
