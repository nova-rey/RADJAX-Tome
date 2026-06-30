from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from radjax_tome.audit.closure import run_closure_audit, write_closure_audit

ROOT = Path(__file__).resolve().parents[1]


def _write_repo_pair(tmp_path: Path) -> tuple[Path, Path]:
    old = tmp_path / "old"
    new = tmp_path / "new"
    (old / "src/qrwkv_xla").mkdir(parents=True)
    (new / "src/radjax_tome").mkdir(parents=True)
    (old / "src/qrwkv_xla/__init__.py").write_text("", encoding="utf-8")
    (new / "src/radjax_tome/__init__.py").write_text("", encoding="utf-8")
    (old / "scripts").mkdir()
    (new / "scripts").mkdir()
    script = """
from __future__ import annotations

import argparse

TEACHER_TEXTBOOK_KIND = "TeacherTextbook"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.parse_args()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""".lstrip()
    (old / "scripts/build_teacher_textbook.py").write_text(script, encoding="utf-8")
    (new / "scripts/build_teacher_textbook.py").write_text(script, encoding="utf-8")
    return old, new


def test_closure_audit_allows_minimal_active_equivalence(tmp_path: Path) -> None:
    old, new = _write_repo_pair(tmp_path)
    ab_summary = tmp_path / "ab_summary.json"
    ab_summary.write_text(
        json.dumps(
            {"status": "pass", "cases": [{"case_id": "unit", "status": "pass"}]}
        ),
        encoding="utf-8",
    )

    audit = run_closure_audit(old, new, ab_summary=ab_summary)

    assert audit.spec3_gate["allowed"] is True
    assert audit.summary["file_closure_counts"]["active_behavior_equivalent"] == 1
    assert audit.summary["symbol_closure_counts"]["active_function_equivalent"] == 2


def test_closure_audit_blocks_missing_producer_surface(tmp_path: Path) -> None:
    old, new = _write_repo_pair(tmp_path)
    (new / "scripts/build_teacher_textbook.py").unlink()
    ab_summary = tmp_path / "ab_summary.json"
    ab_summary.write_text(json.dumps({"status": "pass", "cases": []}), encoding="utf-8")

    audit = run_closure_audit(old, new, ab_summary=ab_summary)

    assert audit.spec3_gate["allowed"] is False
    assert any(blocker["kind"] == "file" for blocker in audit.blockers)


def test_closure_script_writes_json_and_markdown(tmp_path: Path) -> None:
    old, new = _write_repo_pair(tmp_path)
    ab_summary = tmp_path / "ab_summary.json"
    output_json = tmp_path / "closure.json"
    output_md = tmp_path / "closure.md"
    ab_summary.write_text(json.dumps({"status": "pass", "cases": []}), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_tome_generator_closure.py",
            "--old-repo",
            str(old),
            "--new-repo",
            str(new),
            "--ab-summary",
            str(ab_summary),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--overwrite",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "status=complete" in result.stdout
    assert json.loads(output_json.read_text(encoding="utf-8"))["summary"]
    assert "Spec 3 allowed" in output_md.read_text(encoding="utf-8")


def test_write_closure_audit_refuses_overwrite(tmp_path: Path) -> None:
    old, new = _write_repo_pair(tmp_path)
    ab_summary = tmp_path / "ab_summary.json"
    ab_summary.write_text(json.dumps({"status": "pass", "cases": []}), encoding="utf-8")
    audit = run_closure_audit(old, new, ab_summary=ab_summary)
    output_json = tmp_path / "closure.json"
    output_md = tmp_path / "closure.md"
    write_closure_audit(
        audit,
        output_json=output_json,
        output_md=output_md,
        overwrite=True,
    )

    try:
        write_closure_audit(
            audit,
            output_json=output_json,
            output_md=output_md,
            overwrite=False,
        )
    except ValueError as exc:
        assert "refusing to overwrite" in str(exc)
    else:
        raise AssertionError("write_closure_audit should refuse overwrite")
