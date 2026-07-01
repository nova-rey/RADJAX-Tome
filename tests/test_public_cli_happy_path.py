from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBPROCESS_ENV = {
    **os.environ,
    "PYTHONPATH": str(ROOT / "src"),
}


def test_public_cli_top_level_help() -> None:
    result = _run_cli("--help")

    assert result.returncode == 0
    assert "RADJAX-Tome produces teacher-side distillation artifacts." in result.stdout
    assert "Recommended commands:" in result.stdout
    for command in ("build", "validate", "inspect", "prove-capabilities"):
        assert command in result.stdout


def test_public_cli_command_help() -> None:
    for command in ("build", "validate", "inspect"):
        result = _run_cli(command, "--help")

        assert result.returncode == 0
        assert "usage:" in result.stdout


def test_public_cli_fake_build_validate_and_inspect(tmp_path: Path) -> None:
    output = tmp_path / "fake_tome"

    build = _run_cli(
        "build",
        "--output",
        str(output),
        "--teacher-mode",
        "fake",
        "--max-examples",
        "2",
        "--sequence-length",
        "8",
        "--overwrite",
    )
    assert build.returncode == 0, build.stderr
    assert "status=pass" in build.stdout

    validate = _run_cli("validate", "--path", str(output))
    assert validate.returncode == 0, validate.stderr
    assert "status=pass" in validate.stdout

    inspect = _run_cli("inspect", "--path", str(output))
    assert inspect.returncode == 0, inspect.stderr
    assert f"path={output}" in inspect.stdout
    assert "artifact_type=teacher_textbook" in inspect.stdout
    assert "target_type=dense_logits" in inspect.stdout
    assert "num_examples=2" in inspect.stdout
    assert "shard_count=1" in inspect.stdout


def test_public_cli_prove_capabilities_smoke(tmp_path: Path) -> None:
    work_dir = tmp_path / "capabilities"

    result = _run_cli(
        "prove-capabilities",
        "--work-dir",
        str(work_dir),
        "--overwrite",
    )

    assert result.returncode == 0, result.stderr
    assert "status=complete" in result.stdout
    assert "blockers=0" in result.stdout
    assert (work_dir / "matrix.json").is_file()
    assert (work_dir / "report.md").is_file()


def test_public_cli_help_does_not_import_heavy_optional_dependencies() -> None:
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


def test_doctor_does_not_require_optional_hf_dependencies() -> None:
    result = _run_cli("doctor")

    assert result.returncode == 0
    assert "python=" in result.stdout
    assert "radjax_tome=ok" in result.stdout
    assert "optional_dependency.torch=" in result.stdout
    assert "optional_dependency.transformers=" in result.stdout
    assert "recommended=radjax-tome build --teacher-mode fake" in result.stdout


def test_cli_guide_and_readme_document_happy_path() -> None:
    guide = ROOT / "docs" / "CLI_GUIDE.md"
    readme = ROOT / "README.md"

    guide_text = guide.read_text(encoding="utf-8")
    readme_text = readme.read_text(encoding="utf-8")

    assert guide.is_file()
    assert "| Script | Classification | Use when |" in guide_text
    for classification in (
        "recommended wrapper / legacy-compatible",
        "advanced diagnostic",
        "internal/development",
        "archive-only",
    ):
        assert classification in guide_text
    assert "## Recommended CLI" in readme_text
    assert "python -m radjax_tome.cli.main build" in readme_text
    assert "For advanced/dev scripts, see `docs/CLI_GUIDE.md`." in readme_text


def test_legacy_scripts_still_expose_help() -> None:
    scripts = (
        "scripts/build_teacher_textbook.py",
        "scripts/build_teacher_tome.py",
        "scripts/validate_teacher_textbook.py",
        "scripts/inspect_targets.py",
        "scripts/validate_fingerprint_artifact.py",
        "scripts/inspect_fingerprint_artifact.py",
        "scripts/prove_tome_generation_capabilities.py",
    )
    for script in scripts:
        result = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=ROOT,
            env=SUBPROCESS_ENV,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, f"{script}\n{result.stderr}"
        assert "usage:" in result.stdout


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "radjax_tome.cli.main", *args],
        cwd=ROOT,
        env=SUBPROCESS_ENV,
        text=True,
        capture_output=True,
        check=False,
    )
