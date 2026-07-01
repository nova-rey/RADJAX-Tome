from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from radjax_tome.builder import validate_teacher_textbook
from radjax_tome.fingerprint import validate_fingerprint_artifact
from radjax_tome.targets import inspect_target_store
from tests.helpers.fixtures import (
    build_fake_teacher_textbook_artifact,
    build_minimal_fingerprint_artifact,
    build_minimal_target_store,
    write_json,
    write_jsonl,
    write_prompt_corpus_fixture,
)
from tests.helpers.subprocess import run_repo_python

ROOT = Path(__file__).resolve().parents[1]


def test_write_json_is_deterministic(tmp_path: Path) -> None:
    output = write_json(tmp_path / "nested" / "payload.json", {"z": 1, "a": 2})

    assert output.read_text(encoding="utf-8") == '{\n  "a": 2,\n  "z": 1\n}\n'


def test_write_jsonl_writes_expected_records(tmp_path: Path) -> None:
    output = write_jsonl(
        tmp_path / "records.jsonl",
        ({"z": 1, "a": 2}, {"b": 3}),
    )

    assert output.read_text(encoding="utf-8") == '{"a": 2, "z": 1}\n{"b": 3}\n'


def test_run_repo_python_sets_repo_pythonpath() -> None:
    script = "import radjax_tome; print(radjax_tome.__name__)"

    result = run_repo_python(ROOT, "-c", script)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "radjax_tome"


def test_fake_teacher_textbook_fixture_builds_and_validates(tmp_path: Path) -> None:
    artifact = build_fake_teacher_textbook_artifact(tmp_path)

    report = validate_teacher_textbook(artifact)

    assert report.status == "pass"
    assert report.target_type == "dense_logits"


def test_target_store_fixture_inspects(tmp_path: Path) -> None:
    store = build_minimal_target_store(tmp_path)

    summary = inspect_target_store(store.root)

    assert summary["target_type"] == "synthetic"
    assert summary["num_examples"] == 2


def test_fingerprint_fixture_validates(tmp_path: Path) -> None:
    artifact = build_minimal_fingerprint_artifact(tmp_path)

    result = validate_fingerprint_artifact(artifact)

    assert result.ok
    assert result.metadata["records"] == 2


def test_prompt_corpus_fixture_writes_manifest_and_records(tmp_path: Path) -> None:
    path = write_prompt_corpus_fixture(tmp_path)

    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]

    assert [record["id"] for record in records] == ["p0", "p1"]
    assert (tmp_path / "prompt_manifest.json").is_file()


def test_helper_imports_do_not_import_heavy_optional_dependencies() -> None:
    script = (
        "import sys; "
        "import tests.helpers.fixtures; "
        "import tests.helpers.subprocess; "
        "bad=[name for name in ('torch','transformers','jax') if name in sys.modules]; "
        "raise SystemExit(1 if bad else 0)"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
