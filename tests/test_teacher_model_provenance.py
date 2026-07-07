from __future__ import annotations

import hashlib
import json
from pathlib import Path

from radjax_tome.builder import TeacherTextbookBuildConfig, build_teacher_textbook
from radjax_tome.provenance import (
    inspect_teacher_model,
    validate_teacher_model_provenance,
    write_teacher_model_provenance,
)
from tests.helpers.subprocess import run_cli

ROOT = Path(__file__).resolve().parents[1]


def _fake_model_dir(root: Path) -> Path:
    model = root / "model"
    model.mkdir(parents=True)
    (model / "config.json").write_text(
        json.dumps({"_name_or_path": "config/model", "model_type": "tiny"}),
        encoding="utf-8",
    )
    (model / "generation_config.json").write_text("{}", encoding="utf-8")
    (model / "tokenizer.json").write_text('{"version": "1.0"}', encoding="utf-8")
    (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"safe")
    (model / "pytorch_model-00001-of-00002.bin").write_bytes(b"bin shard")
    return model


def _write_provenance(
    tmp_path: Path,
    model: Path,
    **kwargs: str,
) -> tuple[Path, dict[str, object]]:
    payload = inspect_teacher_model(model, **kwargs)
    path = tmp_path / "teacher_model_provenance.json"
    write_teacher_model_provenance(payload, path)
    return path, payload


def test_inspect_local_directory_hashes_and_validates(tmp_path: Path) -> None:
    model = _fake_model_dir(tmp_path)
    path, payload = _write_provenance(tmp_path, model)
    report = validate_teacher_model_provenance(path)

    assert report.status in {"pass", "warn"}
    assert payload["schema_version"] == "teacher_model_provenance_v1"
    assert payload["model_source_kind"] == "local_directory"
    assert payload["model_identity_confidence"] == "verified"
    assert payload["model_name"] == "config/model"
    assert payload["model_name_source"] == "config"
    assert payload["network_used"] is False
    assert payload["allow_downloads"] is False
    assert payload["local_files_only"] is True
    assert payload["downloaded_by_radjax_tome"] is False
    assert payload["load_check"]["mode"] == "metadata_only"
    assert str(payload["config_hash"]).startswith("sha256:")
    assert str(payload["tokenizer_hash"]).startswith("sha256:")
    assert str(payload["weights_hash"]).startswith("sha256:")
    assert str(payload["model_directory_hash"]).startswith("sha256:")

    config_record = {
        record["relative_path"]: record for record in payload["config_files"]
    }["config.json"]
    assert config_record["size_bytes"] == (model / "config.json").stat().st_size
    assert config_record["sha256"] == (
        "sha256:" + hashlib.sha256((model / "config.json").read_bytes()).hexdigest()
    )
    assert {record["relative_path"] for record in payload["weight_files"]} == {
        "model.safetensors",
        "pytorch_model-00001-of-00002.bin",
    }


def test_hf_cache_snapshot_identity_is_inferred_locally(tmp_path: Path) -> None:
    snapshot = (
        tmp_path / "hub" / "models--nova-rey--tiny-teacher" / "snapshots" / "abc123"
    )
    _fake_model_dir(snapshot)

    payload = inspect_teacher_model(snapshot / "model")

    assert payload["model_source_kind"] == "local_hf_snapshot"
    assert payload["hf_repo_id"] == "nova-rey/tiny-teacher"
    assert payload["hf_revision"] == "abc123"
    assert payload["model_name"] == "nova-rey/tiny-teacher"
    assert payload["model_name_source"] == "hf_cache_path"
    assert payload["model_revision"] == "abc123"
    assert payload["model_revision_source"] == "hf_cache_path"
    assert payload["model_identity_confidence"] == "inferred"
    assert payload["hf_identity_source"] == "inferred_from_local_cache_path"
    assert payload["network_used"] is False


def test_user_declared_identity_and_unknown_identity_are_honest(
    tmp_path: Path,
) -> None:
    declared_model = tmp_path / "declared"
    declared_model.mkdir()
    (declared_model / "tokenizer.model").write_bytes(b"tok")
    (declared_model / "pytorch_model.bin").write_bytes(b"weights")

    declared = inspect_teacher_model(
        declared_model,
        model_name="my-custom-model",
        model_revision="manual-2026-07-07",
    )

    assert declared["model_identity_confidence"] == "declared"
    assert declared["model_provenance_mode"] == "user_declared"
    assert declared["model_name_source"] == "user_declared"
    assert declared["model_revision_source"] == "user_declared"
    assert declared["weights_hash"] is not None

    unknown_model = tmp_path / "unknown"
    unknown_model.mkdir()
    (unknown_model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (unknown_model / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    unknown = inspect_teacher_model(unknown_model)

    assert unknown["model_identity_confidence"] == "unknown"
    assert unknown["model_name"] is None
    assert unknown["model_name_source"] == "unknown"
    assert unknown["weights_hash"] is not None


def test_model_cli_inspect_validate_and_discover(tmp_path: Path) -> None:
    model = _fake_model_dir(tmp_path)
    output = tmp_path / "teacher_model_provenance.json"

    inspect_result = run_cli(
        ROOT,
        "model",
        "inspect",
        "--model-path",
        str(model),
        "--model-name",
        "declared/name",
        "--model-revision",
        "rev-a",
        "--output",
        str(output),
    )
    validate_result = run_cli(
        ROOT,
        "model",
        "validate",
        "--provenance",
        str(output),
    )
    discover_result = run_cli(
        ROOT,
        "model",
        "discover",
        "--search-path",
        str(tmp_path),
    )

    assert inspect_result.returncode == 0, inspect_result.stderr
    assert "status=" in inspect_result.stdout
    assert output.is_file()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["model_identity_confidence"] == "declared"
    assert payload["model_name_source"] == "user_declared"
    assert validate_result.returncode == 0, validate_result.stderr
    assert "model_source_kind=local_directory" in validate_result.stdout
    assert discover_result.returncode == 0, discover_result.stderr
    assert f"candidate_path={model}" in discover_result.stdout
    assert "has_weights=true" in discover_result.stdout


def test_tome_build_records_teacher_model_provenance(tmp_path: Path) -> None:
    model = _fake_model_dir(tmp_path)
    provenance_path, provenance = _write_provenance(tmp_path, model)
    output = tmp_path / "tome"

    report = build_teacher_textbook(
        TeacherTextbookBuildConfig(
            output_dir=output,
            max_examples=1,
            sequence_length=8,
            overwrite=True,
            teacher_model_provenance_path=provenance_path,
        )
    )

    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    teacher_manifest = json.loads(
        (output / "teacher_manifest.json").read_text(encoding="utf-8")
    )
    emission_config = json.loads(
        (output / "emission_config.json").read_text(encoding="utf-8")
    )
    cover_page = json.loads((output / "cover_page.json").read_text(encoding="utf-8"))

    assert report.status == "pass"
    assert metadata["target_params"]["teacher_model_provenance_schema"] == (
        "teacher_model_provenance_v1"
    )
    assert (
        metadata["target_params"]["teacher_model_weights_hash"]
        == (provenance["weights_hash"])
    )
    assert (
        teacher_manifest["teacher_model_provenance"]["weights_hash"]
        == (provenance["weights_hash"])
    )
    assert emission_config["teacher_model_provenance"]["network_used"] is False
    assert (
        cover_page["teacher_model_provenance"]["model_directory_hash"]
        == (provenance["model_directory_hash"])
    )
    assert "weight_files" not in cover_page["teacher_model_provenance"]


def test_tampered_teacher_model_file_fails_validation(tmp_path: Path) -> None:
    model = _fake_model_dir(tmp_path)
    provenance_path, _ = _write_provenance(tmp_path, model)
    (model / "model.safetensors").write_bytes(b"tampered")

    report = validate_teacher_model_provenance(provenance_path)

    assert report.status == "fail"
    assert any("hash mismatch" in blocker for blocker in report.blockers)


def test_corpus_validator_uses_current_manifest_hash_policy_wording(
    tmp_path: Path,
) -> None:
    from radjax_tome.corpora import (
        CORPUS_MANIFEST_FILENAME,
        CorpusBuildConfig,
        build_corpus_artifact,
        validate_corpus_artifact,
    )

    source = tmp_path / "source.txt"
    source.write_text("alpha", encoding="utf-8")
    corpus = tmp_path / "corpus"
    build_corpus_artifact(
        CorpusBuildConfig(inputs=(source,), output_dir=corpus, overwrite=True)
    )
    manifest_path = corpus / CORPUS_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["manifest_hash"] = "sha256:bad"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = validate_corpus_artifact(corpus)

    assert report.status == "fail"
    assert any(
        "exclude_self_hash_and_created_at_v1" in blocker for blocker in report.blockers
    )


def test_docs_and_bible_mention_spec_4_2() -> None:
    bible = (ROOT / "bible.md").read_text(encoding="utf-8")
    docs = (ROOT / "docs" / "TEACHER_MODEL_PROVENANCE.md").read_text(encoding="utf-8")
    cli = (ROOT / "docs" / "CLI_GUIDE.md").read_text(encoding="utf-8")

    assert "Spec 4.2" in bible
    assert "teacher model provenance" in bible
    assert "does not silently download teacher models" in docs
    assert "verified vs inferred vs declared" in docs
    assert "radjax-tome model inspect" in docs
    assert "--teacher-model-provenance" in cli
