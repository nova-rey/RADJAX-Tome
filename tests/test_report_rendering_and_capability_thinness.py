from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

from radjax_tome.capabilities import (
    REQUIRED_CAPABILITY_IDS,
    prove_tome_generation_capabilities,
)
from radjax_tome.reports.rendering import markdown_table, status_line
from radjax_tome.reports.writers import write_json_report, write_markdown_report

ROOT = Path(__file__).resolve().parents[1]
SUBPROCESS_ENV = {
    **os.environ,
    "PYTHONPATH": str(ROOT / "src"),
}


def test_write_json_report_is_deterministic_with_trailing_newline(
    tmp_path: Path,
) -> None:
    output = tmp_path / "nested" / "report.json"

    write_json_report(output, {"z": 1, "a": {"b": 2}})

    assert output.read_text(encoding="utf-8") == (
        '{\n  "a": {\n    "b": 2\n  },\n  "z": 1\n}\n'
    )


def test_write_markdown_report_creates_parent_directory(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "report.md"

    write_markdown_report(output, "# Report\n")

    assert output.read_text(encoding="utf-8") == "# Report\n"


def test_markdown_table_and_status_line_are_stable() -> None:
    assert markdown_table(
        ("Name", "Status"),
        (("dense", "pass"), ("hf", "schema")),
    ) == (
        "| Name  | Status |\n| ----- | ------ |\n| dense | pass   |\n| hf    | schema |"
    )
    assert status_line(status="complete", capabilities=8, blockers=0) == (
        "status=complete capabilities=8 blockers=0"
    )


def test_capability_proof_function_runs_from_src(tmp_path: Path) -> None:
    result = prove_tome_generation_capabilities(
        work_dir=tmp_path / "capabilities",
        matrix_json=tmp_path / "capabilities" / "matrix.json",
        report_md=tmp_path / "capabilities" / "report.md",
        overwrite=True,
    )

    assert result.exit_code == 0
    assert len(result.capabilities) == 8
    assert result.blockers == ()
    assert result.matrix_json.is_file()
    assert result.report_md.is_file()
    assert _capability_ids(result.matrix) == REQUIRED_CAPABILITY_IDS
    assert "Spec 3 may proceed" in result.report_md.read_text(encoding="utf-8")


def test_capability_proof_script_still_runs(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/prove_tome_generation_capabilities.py",
            "--work-dir",
            str(tmp_path / "script-capabilities"),
            "--matrix-json",
            str(tmp_path / "script-capabilities" / "matrix.json"),
            "--report-md",
            str(tmp_path / "script-capabilities" / "report.md"),
            "--overwrite",
        ],
        cwd=ROOT,
        env=SUBPROCESS_ENV,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "status=complete" in result.stdout
    assert "capabilities=8" in result.stdout
    assert "blockers=0" in result.stdout
    matrix = json.loads(
        (tmp_path / "script-capabilities" / "matrix.json").read_text(encoding="utf-8")
    )
    assert _capability_ids(matrix) == REQUIRED_CAPABILITY_IDS
    assert not _blockers(matrix)
    assert (tmp_path / "script-capabilities" / "report.md").is_file()


def test_public_cli_prove_capabilities_uses_src_proof(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "radjax_tome.cli.main",
            "prove-capabilities",
            "--work-dir",
            str(tmp_path / "cli-capabilities"),
            "--overwrite",
        ],
        cwd=ROOT,
        env=SUBPROCESS_ENV,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "status=complete" in result.stdout
    assert "blockers=0" in result.stdout
    matrix = json.loads(
        (tmp_path / "cli-capabilities" / "matrix.json").read_text(encoding="utf-8")
    )
    assert _capability_ids(matrix) == REQUIRED_CAPABILITY_IDS
    assert not _blockers(matrix)
    assert (tmp_path / "cli-capabilities" / "report.md").is_file()


def test_capability_script_is_thin_wrapper() -> None:
    script = ROOT / "scripts" / "prove_tome_generation_capabilities.py"
    lines = script.read_text(encoding="utf-8").splitlines()
    tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]

    assert len(lines) <= 120
    assert functions == ["main"]


def test_help_does_not_import_heavy_optional_dependencies_after_refactor() -> None:
    script = (
        "import sys; "
        "from radjax_tome.cli.main import main; "
        "code=main(['--help']); "
        "bad=[name for name in ('torch','transformers','jax') if name in sys.modules]; "
        "raise SystemExit(1 if code or bad else 0)"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=SUBPROCESS_ENV,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def _capability_ids(matrix: dict[str, object]) -> frozenset[str]:
    return frozenset(
        str(entry["capability_id"])
        for entry in matrix["capabilities"]  # type: ignore[index]
    )


def _blockers(matrix: dict[str, object]) -> list[dict[str, object]]:
    return [
        dict(entry)
        for entry in matrix["capabilities"]  # type: ignore[index]
        if entry["blocks_spec3"]
    ]
