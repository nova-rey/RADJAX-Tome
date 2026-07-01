from __future__ import annotations

import hashlib
import json
from pathlib import Path

from radjax_tome.builder import validate_teacher_textbook
from radjax_tome.tome import validate_tome_cover_page
from tests.helpers.fixtures import build_fake_teacher_textbook_artifact
from tests.helpers.subprocess import run_cli, run_repo_python

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_TOP_LEVEL_FIELDS = {
    "artifact_kind",
    "cover_page_version",
    "tome_version",
    "layout",
    "created_by",
    "created_at",
    "source_artifact_type",
    "teacher",
    "tokenizer",
    "targets",
    "contents",
    "validation",
    "claims_not_made",
}
REQUIRED_CONTENTS = {
    "metadata.json",
    "vocab_contract.json",
    "teacher_manifest.json",
    "emission_config.json",
    "validation_report.json",
    "shards/shard-00000.npz",
}


def test_fake_build_writes_deterministic_cover_page(tmp_path: Path) -> None:
    artifact = build_fake_teacher_textbook_artifact(tmp_path)
    cover_page_path = artifact / "cover_page.json"
    raw = cover_page_path.read_text(encoding="utf-8")
    cover_page = json.loads(raw)

    assert cover_page_path.is_file()
    assert raw.endswith("\n")
    assert raw == json.dumps(cover_page, indent=2, sort_keys=True) + "\n"
    assert REQUIRED_TOP_LEVEL_FIELDS <= set(cover_page)
    assert cover_page["artifact_kind"] == "radjax_tome"
    assert cover_page["cover_page_version"] == 1
    assert cover_page["tome_version"] == 1
    assert cover_page["layout"] == "unpacked_directory"
    assert cover_page["source_artifact_type"] == "teacher_textbook"
    assert cover_page["teacher"]["model_id"] == "fake-deterministic-teacher"
    assert cover_page["teacher"]["backend_type"] == "fake"
    assert cover_page["tokenizer"]["vocab_contract_path"] == "vocab_contract.json"
    assert cover_page["targets"]["target_type"] == "dense_logits"
    assert cover_page["targets"]["num_examples"] == 2


def test_cover_page_contents_include_sidecars_and_matching_hashes(
    tmp_path: Path,
) -> None:
    artifact = build_fake_teacher_textbook_artifact(tmp_path)
    cover_page = json.loads((artifact / "cover_page.json").read_text(encoding="utf-8"))
    contents = {entry["path"]: entry for entry in cover_page["contents"]}

    assert REQUIRED_CONTENTS <= set(contents)
    assert "cover_page.json" not in contents
    for relative_path, entry in contents.items():
        path = artifact / relative_path
        assert entry["size_bytes"] == path.stat().st_size
        assert entry["sha256"] == _sha256(path)


def test_cover_page_validation_passes_and_fails_on_tampering(
    tmp_path: Path,
) -> None:
    artifact = build_fake_teacher_textbook_artifact(tmp_path)

    report = validate_tome_cover_page(artifact)

    assert report.status == "pass"
    assert report.artifact_kind_ok
    assert report.version_ok
    assert report.layout_ok
    assert report.contents_ok
    assert report.hashes_ok
    assert report.required_fields_ok

    validation_report = artifact / "validation_report.json"
    validation_report.write_text(
        validation_report.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    tampered = validate_tome_cover_page(artifact)
    assert tampered.status == "fail"
    assert any("sha256 mismatch" in blocker for blocker in tampered.blockers)


def test_legacy_teacher_textbook_without_cover_page_still_validates(
    tmp_path: Path,
) -> None:
    artifact = build_fake_teacher_textbook_artifact(tmp_path)
    (artifact / "cover_page.json").unlink()

    report = validate_teacher_textbook(artifact)
    cli = run_cli(ROOT, "validate", "--path", str(artifact))

    assert report.status == "pass"
    assert cli.returncode == 0, cli.stderr
    assert "status=pass" in cli.stdout
    assert "cover_page_status" not in cli.stdout


def test_public_cli_validate_and_inspect_are_cover_page_aware(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "cli_tome"
    build = run_cli(
        ROOT,
        "build",
        "--output",
        str(artifact),
        "--teacher-mode",
        "fake",
        "--max-examples",
        "2",
        "--sequence-length",
        "8",
        "--overwrite",
    )
    assert build.returncode == 0, build.stderr

    validate = run_cli(ROOT, "validate", "--path", str(artifact))
    inspect = run_cli(ROOT, "inspect", "--path", str(artifact))

    assert validate.returncode == 0, validate.stderr
    assert "status=pass" in validate.stdout
    assert "cover_page_status=pass" in validate.stdout
    assert inspect.returncode == 0, inspect.stderr
    assert "artifact_type=teacher_textbook" in inspect.stdout
    assert "tome_artifact_kind=radjax_tome" in inspect.stdout
    assert "cover_page_version=1" in inspect.stdout
    assert "tome_version=1" in inspect.stdout
    assert "layout=unpacked_directory" in inspect.stdout
    assert "target_type=dense_logits" in inspect.stdout
    assert "num_examples=2" in inspect.stdout
    assert "shard_count=1" in inspect.stdout


def test_cover_page_module_does_not_import_heavy_optional_dependencies() -> None:
    script = (
        "import sys; "
        "import radjax_tome.tome.cover_page; "
        "bad=[name for name in ('torch','transformers','jax') if name in sys.modules]; "
        "raise SystemExit(1 if bad else 0)"
    )

    result = run_repo_python(ROOT, "-c", script)

    assert result.returncode == 0, result.stderr


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()
