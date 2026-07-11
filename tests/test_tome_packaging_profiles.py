from __future__ import annotations

import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

from radjax_tome.audit import audit_selected_linkage
from radjax_tome.builder import ProductionBuildConfig, build_production_gpu_tome
from radjax_tome.corpora import CorpusBuildConfig, build_corpus_artifact
from radjax_tome.provenance import (
    inspect_teacher_model,
    write_teacher_model_provenance,
)
from radjax_tome.tome import (
    FULL_DEBUG_PROVENANCE,
    STUDENT,
    open_student_tome,
    package_tome_artifact,
    validate_tome_package,
)
from tests.helpers.subprocess import run_cli

ROOT = Path(__file__).resolve().parents[1]


def _artifact(tmp_path: Path) -> Path:
    sources = []
    for index in range(6):
        path = tmp_path / f"source-{index}.txt"
        path.write_text(f"package artifact example {index}", encoding="utf-8")
        sources.append(path)
    corpus = tmp_path / "corpus"
    build_corpus_artifact(
        CorpusBuildConfig(inputs=tuple(sources), output_dir=corpus, overwrite=True)
    )
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"tiny"}', encoding="utf-8")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")
    provenance = tmp_path / "teacher_provenance.json"
    write_teacher_model_provenance(
        inspect_teacher_model(model, model_name="radjax/package-test"),
        provenance,
    )
    output = tmp_path / "artifact"
    report = build_production_gpu_tome(
        ProductionBuildConfig(
            teacher_model=str(model),
            tokenizer_id=str(model),
            dataset_path=corpus / "corpus.jsonl",
            corpus_manifest_path=corpus / "corpus_manifest.json",
            teacher_model_provenance_path=provenance,
            output_dir=output,
            teacher_backend="cpu_reference",
            runtime_mode="cpu",
            target_policy="corridor_exemplar_v1",
            sequence_length=5,
            vocab_size=64,
            top_k=4,
            num_buckets=3,
            gpu_batch_size_mode="preset",
            gpu_batch_size_preset=2,
            shard_size_examples=2,
            max_examples=6,
            exemplar_selection_enabled=True,
            exemplar_delivery_path="two_pass_rerun_selected",
            selected_exemplar_budget=2,
            retain_unselected_exemplar_payloads=False,
        )
    )
    assert report["status"] == "pass"
    return output


@pytest.fixture
def artifact(tmp_path: Path) -> Path:
    return _artifact(tmp_path)


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_full_package_uses_manifest_references_not_inline_shards(
    artifact: Path,
    tmp_path: Path,
) -> None:
    package = tmp_path / "full"
    package_tome_artifact(
        artifact,
        package,
        profile=FULL_DEBUG_PROVENANCE,
        overwrite=True,
    )

    cover = _json(package / "cover_page.json")
    content = _json(package / "manifests" / "content_manifest.json")
    shard_manifest = _json(package / "manifests" / "shard_manifest.json")

    assert cover["package_profile"] == FULL_DEBUG_PROVENANCE
    assert cover["shard_manifest"]["path"] == "manifests/shard_manifest.json"
    assert "contents" not in cover
    assert len(cover["shard_manifest"]) == 3
    assert content["entry_count"] == len(content["entries"])
    assert len(shard_manifest["shards"]) == 3
    assert validate_tome_package(package, profile=FULL_DEBUG_PROVENANCE).ok
    assert audit_selected_linkage(package, strict=True).status == "pass"


def test_student_package_is_self_contained_training_contract(
    artifact: Path,
    tmp_path: Path,
) -> None:
    package = tmp_path / "student"
    package_tome_artifact(artifact, package, profile=STUDENT, overwrite=True)

    assert not (package / "shards").exists()
    assert (
        package / "corridors" / "mode_assignments" / "examples_input_ids.npy"
    ).is_file()
    reader = open_student_tome(package)
    corridor = reader.corridor_batch(0)
    exemplar = reader.exemplar_batch(0)
    cover = _json(package / "cover_page.json")
    emission = _json(package / "emission_config.json")
    audit = audit_selected_linkage(package, strict=True, profile=STUDENT)
    audit_cli = run_cli(
        ROOT,
        "audit-selected-linkage",
        "--artifact",
        str(package),
        "--profile",
        STUDENT,
        "--strict",
    )

    assert set(corridor) == {
        "input_ids",
        "position",
        "mode_id",
        "weight",
        "mode_bounds",
    }
    assert exemplar["selected_position"] >= 0
    assert exemplar["input_ids"].shape == corridor["input_ids"].shape
    assert audit.status == "pass"
    assert audit.producer_shard_authority == "not_available_in_student_profile"
    assert audit_cli.returncode == 0, audit_cli.stdout + audit_cli.stderr
    assert "not_available_in_student_profile" in audit_cli.stdout
    assert cover["claims_made"]["student_batches_constructible"] is True
    assert cover["claims_not_made"]["does_not_include_full_producer_shards"] is True
    assert "non_portable_source_path" in emission["dataset_source"]
    assert validate_tome_package(package, profile=STUDENT).ok


def test_student_inputs_match_full_source_shards(
    artifact: Path, tmp_path: Path
) -> None:
    package = tmp_path / "student"
    package_tome_artifact(artifact, package, profile=STUDENT, overwrite=True)
    manifest = _json(package / "corridors" / "mode_assignments.json")
    metadata_path = package / manifest["examples_metadata"]["path"]
    metadata = [json.loads(line) for line in metadata_path.read_text().splitlines()]
    source_rows = []
    for shard in sorted((artifact / "shards").glob("shard-*.npz")):
        with np.load(shard, allow_pickle=False) as arrays:
            source_rows.extend(np.asarray(arrays["input_ids"]))
    student_rows = np.load(
        package / "corridors" / "mode_assignments" / "examples_input_ids.npy",
        allow_pickle=False,
    )
    for item in metadata:
        assert np.array_equal(
            student_rows[item["example_index"]],
            source_rows[item["example_index"]],
        )


@pytest.mark.parametrize("profile", (STUDENT, FULL_DEBUG_PROVENANCE))
def test_package_hash_validation_fails_on_mutation(
    artifact: Path,
    tmp_path: Path,
    profile: str,
) -> None:
    package = tmp_path / profile
    package_tome_artifact(artifact, package, profile=profile, overwrite=True)
    path = (
        sorted((package / "selected_exemplars").glob("selected-exemplars-*.json"))[0]
        if profile == STUDENT
        else sorted((package / "shards").glob("shard-*.npz"))[0]
    )
    path.write_bytes(path.read_bytes() + b"mutation")

    report = validate_tome_package(package, profile=profile)

    assert report.status == "fail"
    assert any(path.relative_to(package).as_posix() in item for item in report.blockers)


def test_student_tgz_round_trip_and_cli(artifact: Path, tmp_path: Path) -> None:
    archive = tmp_path / "student.tgz"
    result = package_tome_artifact(
        artifact,
        archive,
        profile=STUDENT,
        archive="tgz",
        overwrite=True,
    )
    extracted = tmp_path / "extracted"
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(extracted, filter="data")
    package_root = extracted / "student"
    package_cli_output = tmp_path / "student-cli"
    package_cli = run_cli(
        ROOT,
        "package-artifact",
        "--input",
        str(artifact),
        "--output",
        str(package_cli_output),
        "--profile",
        STUDENT,
        "--overwrite",
    )
    cli = run_cli(
        ROOT,
        "validate-package",
        "--artifact",
        str(package_cli_output),
        "--profile",
        STUDENT,
    )

    assert result.output_path == archive
    assert (package_root / "cover_page.json").is_file()
    assert package_cli.returncode == 0, package_cli.stdout + package_cli.stderr
    assert cli.returncode == 0, cli.stdout + cli.stderr
