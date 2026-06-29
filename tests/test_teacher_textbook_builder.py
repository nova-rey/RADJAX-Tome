from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from radjax_tome.builder import (
    TeacherTextbookBuildConfig,
    build_teacher_textbook,
    validate_teacher_textbook,
)

ROOT = Path(__file__).resolve().parents[1]
SUBPROCESS_ENV = {
    **os.environ,
    "PYTHONPATH": str(ROOT / "src"),
}


def test_builder_import_smoke() -> None:
    import radjax_tome.builder

    assert radjax_tome.builder.TeacherTextbookBuildConfig is TeacherTextbookBuildConfig


def test_fake_builder_creates_valid_legacy_teacher_textbook(tmp_path: Path) -> None:
    output = tmp_path / "teacher_textbook"

    report = build_teacher_textbook(_config(output))

    assert report.status == "pass"
    assert (output / "metadata.json").is_file()
    assert (output / "vocab_contract.json").is_file()
    assert (output / "teacher_manifest.json").is_file()
    assert (output / "emission_config.json").is_file()
    assert (output / "validation_report.json").is_file()
    assert (output / "shards" / "shard-00000.npz").is_file()
    assert not (output / "cover_page.json").exists()
    assert validate_teacher_textbook(output).status == "pass"


def test_fake_builder_reads_jsonl_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "texts.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"example_id": "a", "text": "alpha"},
                {"example_id": "b", "text": "beta"},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "teacher_textbook"

    report = build_teacher_textbook(_config(output, dataset_path=dataset))

    assert report.status == "pass"
    assert validate_teacher_textbook(output).status == "pass"


def test_teacher_textbook_cli_smoke(tmp_path: Path) -> None:
    output = tmp_path / "teacher_textbook"
    build_script = ROOT / "scripts" / "build_teacher_textbook.py"
    validate_script = ROOT / "scripts" / "validate_teacher_textbook.py"

    build = subprocess.run(
        [
            sys.executable,
            str(build_script),
            "--output",
            str(output),
            "--teacher-mode",
            "fake",
            "--max-examples",
            "2",
            "--sequence-length",
            "8",
            "--vocab-size",
            "16",
            "--overwrite",
        ],
        check=False,
        cwd=ROOT,
        env=SUBPROCESS_ENV,
        text=True,
        capture_output=True,
    )
    assert build.returncode == 0, build.stderr
    assert "status=pass" in build.stdout

    validate = subprocess.run(
        [
            sys.executable,
            str(validate_script),
            "--path",
            str(output),
            "--write-report",
        ],
        check=False,
        cwd=ROOT,
        env=SUBPROCESS_ENV,
        text=True,
        capture_output=True,
    )
    assert validate.returncode == 0, validate.stderr
    assert "status=pass" in validate.stdout


def test_default_import_does_not_load_heavy_optional_dependencies() -> None:
    script = (
        "import sys; import radjax_tome; "
        "bad=[name for name in ('torch','transformers','jax') if name in sys.modules]; "
        "raise SystemExit(1 if bad else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        cwd=ROOT,
        env=SUBPROCESS_ENV,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def _config(
    output: Path, dataset_path: Path | None = None
) -> TeacherTextbookBuildConfig:
    return TeacherTextbookBuildConfig(
        output_dir=output,
        dataset_path=dataset_path,
        teacher_mode="fake",
        sequence_length=8,
        batch_size=2,
        max_examples=2,
        vocab_size=16,
        top_k=4,
        overwrite=True,
    )
