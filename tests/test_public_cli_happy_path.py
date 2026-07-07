from __future__ import annotations

import json
from pathlib import Path

from tests.helpers.subprocess import run_cli, run_repo_python, run_script

ROOT = Path(__file__).resolve().parents[1]


def test_public_cli_top_level_help() -> None:
    result = run_cli(ROOT, "--help")

    assert result.returncode == 0
    assert "RADJAX-Tome produces teacher-side distillation artifacts." in result.stdout
    assert "Recommended commands:" in result.stdout
    for command in (
        "build",
        "validate",
        "inspect",
        "pack",
        "unpack",
        "prove-capabilities",
    ):
        assert command in result.stdout


def test_public_cli_command_help() -> None:
    for command in ("build", "validate", "inspect", "pack", "unpack"):
        result = run_cli(ROOT, command, "--help")

        assert result.returncode == 0
        assert "usage:" in result.stdout


def test_public_cli_fake_build_validate_and_inspect(tmp_path: Path) -> None:
    output = tmp_path / "fake_tome"

    build = run_cli(
        ROOT,
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

    validate = run_cli(ROOT, "validate", "--path", str(output))
    assert validate.returncode == 0, validate.stderr
    assert "status=pass" in validate.stdout
    assert "cover_page_status=pass" in validate.stdout

    inspect = run_cli(ROOT, "inspect", "--path", str(output))
    assert inspect.returncode == 0, inspect.stderr
    assert f"path={output}" in inspect.stdout
    assert "artifact_type=teacher_textbook" in inspect.stdout
    assert "tome_artifact_kind=radjax_tome" in inspect.stdout
    assert "cover_page_version=1" in inspect.stdout
    assert "tome_version=1" in inspect.stdout
    assert "layout=unpacked_directory" in inspect.stdout
    assert "target_type=dense_logits" in inspect.stdout
    assert "num_examples=2" in inspect.stdout
    assert "shard_count=1" in inspect.stdout


def test_public_cli_backend_contract_build_smoke(tmp_path: Path) -> None:
    output = tmp_path / "backend_tome"

    build = run_cli(
        ROOT,
        "build",
        "--output",
        str(output),
        "--teacher-backend",
        "cpu_reference",
        "--runtime-mode",
        "cpu",
        "--target-policy",
        "dynamic",
        "--max-examples",
        "2",
        "--sequence-length",
        "8",
        "--overwrite",
    )

    assert build.returncode == 0, build.stderr
    assert "status=pass" in build.stdout

    inspect = run_cli(ROOT, "inspect", "--path", str(output))
    assert inspect.returncode == 0, inspect.stderr
    assert "target_type=dynamic_cascaded_soft_labels_v1" in inspect.stdout


def test_public_cli_backend_selection_manifest_smoke(tmp_path: Path) -> None:
    output = tmp_path / "selected_backend_tome"

    build = run_cli(
        ROOT,
        "build",
        "--output",
        str(output),
        "--teacher-backend",
        "cpu_reference",
        "--runtime-mode",
        "cpu",
        "--target-policy",
        "corridor",
        "--exemplar-selection-enabled",
        "--exemplar-selection-board-capacity",
        "2",
        "--max-examples",
        "2",
        "--sequence-length",
        "8",
        "--overwrite",
    )

    assert build.returncode == 0, build.stderr
    manifest = json.loads(
        (output / "exemplar_selection_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["selection_policy"] == "multi_leaderboard_exemplar_selector_v1"
    assert manifest["fulfillment_policy"] == "select_from_existing_capture"


def test_public_cli_prove_capabilities_smoke(tmp_path: Path) -> None:
    work_dir = tmp_path / "capabilities"

    result = run_cli(
        ROOT,
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

    result = run_repo_python(ROOT, "-c", script)

    assert result.returncode == 0, result.stderr


def test_doctor_does_not_require_optional_hf_dependencies() -> None:
    result = run_cli(ROOT, "doctor")

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
        result = run_script(ROOT, script, "--help")
        assert result.returncode == 0, f"{script}\n{result.stderr}"
        assert "usage:" in result.stdout
