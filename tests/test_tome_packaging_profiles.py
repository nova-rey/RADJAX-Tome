from __future__ import annotations

import json
import tarfile
from dataclasses import replace
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


def _artifact(
    tmp_path: Path,
    **production_overrides: object,
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
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
    config = ProductionBuildConfig(
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
    report = build_production_gpu_tome(replace(config, **production_overrides))
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
    payload_manifest = _json(package / "manifests" / "selected_payload_manifest.json")
    selected_payload = _json(
        sorted((package / "selected_exemplars").glob("selected-exemplars-*.json"))[0]
    )["selected_exemplars"][0]
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
    assert payload_manifest["long_tail_summary"]["count"] == 2
    assert (
        cover["diagnostics"]["long_tail_summary"]
        == payload_manifest["long_tail_summary"]
    )
    assert {
        "dynamic_mass_threshold",
        "dynamic_top_k_max",
        "top_k_saturated",
        "long_tail_class",
        "long_tail_warnings",
        "effective_top_k_fraction_of_vocab",
    } <= set(selected_payload)
    assert validate_tome_package(package, profile=STUDENT).ok


@pytest.mark.parametrize("include_perverse", (False, True))
def test_student_package_filters_perverse_tail_board_by_producer_opt_in(
    tmp_path: Path,
    include_perverse: bool,
) -> None:
    artifact = _artifact(
        tmp_path / ("included" if include_perverse else "excluded"),
        long_tail_warning_k=1,
        very_long_tail_warning_k=2,
        perverse_tail_warning_k=3,
        include_perverse_tail_in_student=include_perverse,
    )
    package_name = "student-included" if include_perverse else "student-excluded"
    package = tmp_path / package_name

    package_tome_artifact(artifact, package, profile=STUDENT, overwrite=True)

    selected = _json(package / "leaderboards" / "selected_exemplars.json")
    payload_manifest = _json(package / "manifests" / "selected_payload_manifest.json")
    cover = _json(package / "cover_page.json")
    audit = _json(package / "selected_linkage_audit.json")
    boards = selected["selected_exemplar_boards"]
    retained_count = len(selected["selected_exemplars"])
    assert selected["long_tail_summary"] == payload_manifest["long_tail_summary"]
    assert selected["long_tail_summary"] == cover["diagnostics"]["long_tail_summary"]
    assert (
        selected["selected_board_summary"] == payload_manifest["selected_board_summary"]
    )
    assert (
        selected["selected_board_summary"]
        == cover["diagnostics"]["selected_board_summary"]
    )
    assert selected["selected_board_summary"]["total_selected_count"] == retained_count
    assert audit["selected_count"] == retained_count
    assert audit["status"] == "pass"
    if include_perverse:
        assert (package / "leaderboards" / "perverse_tail_diagnostic.json").is_file()
        assert boards["perverse_tail_diagnostic"]
        assert (
            payload_manifest["selected_board_summary"]["perverse_tail_diagnostic_count"]
            > 0
        )
    else:
        assert not (package / "leaderboards" / "perverse_tail_diagnostic.json").exists()
        assert "perverse_tail_diagnostic" not in boards
        assert (
            payload_manifest["selected_board_summary"]["perverse_tail_diagnostic_count"]
            == 0
        )
    assert validate_tome_package(package, profile=STUDENT).ok


def test_full_debug_package_retains_perverse_tail_board(tmp_path: Path) -> None:
    artifact = _artifact(
        tmp_path / "full-debug",
        long_tail_warning_k=1,
        very_long_tail_warning_k=2,
        perverse_tail_warning_k=3,
    )
    package = tmp_path / "full-debug-package"

    package_tome_artifact(
        artifact,
        package,
        profile=FULL_DEBUG_PROVENANCE,
        overwrite=True,
    )

    cover = _json(package / "cover_page.json")
    selected = _json(package / "leaderboards" / "selected_exemplars.json")
    assert (package / "leaderboards" / "perverse_tail_diagnostic.json").is_file()
    assert selected["selected_exemplar_boards"]["perverse_tail_diagnostic"]
    assert (
        cover["diagnostics"]["selected_board_summary"]["perverse_tail_diagnostic_count"]
        > 0
    )
    assert validate_tome_package(package, profile=FULL_DEBUG_PROVENANCE).ok


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


@pytest.mark.parametrize("profile", (FULL_DEBUG_PROVENANCE, STUDENT))
def test_packaged_cover_page_summary_uses_package_local_truth(
    artifact: Path,
    tmp_path: Path,
    profile: str,
) -> None:
    package = tmp_path / profile
    package_tome_artifact(artifact, package, profile=profile, overwrite=True)

    cover = _json(package / "cover_page.json")
    top = cover["top_level_summary"]
    corridor = _json(package / "corridors" / "corridor_summary.json")
    assignments = _json(package / "manifests" / "corridor_assignment_manifest.json")
    selected = _json(package / "manifests" / "selected_payload_manifest.json")
    audit = _json(package / "selected_linkage_audit.json")
    validation = _json(package / "validation_report.json")

    assert all(
        top[key] is not None
        for key in (
            "num_examples_scored",
            "num_positions_scored",
            "num_selected_exemplars",
            "corridor_mode_count",
            "corridor_assignment_count",
            "delivery_path",
            "selected_linkage_audit_status",
            "validation_status",
            "package_profile",
            "producer_shard_authority",
        )
    )
    assert top["num_examples_scored"] == corridor["num_examples_scored"]
    assert top["num_positions_scored"] == corridor["num_positions_scored"]
    assert top["num_selected_exemplars"] == selected["selected_count"]
    assert top["corridor_mode_count"] == corridor["mode_count"]
    assert top["corridor_assignment_count"] == assignments["assignment_count"]
    assert top["delivery_path"] == corridor["delivery_path"]
    assert top["selected_linkage_audit_status"] == audit["status"]
    assert top["validation_status"] == validation["status"]
    assert top["package_profile"] == profile
    expected_authority = (
        "available"
        if profile == FULL_DEBUG_PROVENANCE
        else "not_available_in_student_profile"
    )
    assert top["producer_shard_authority"] == expected_authority
    if profile == FULL_DEBUG_PROVENANCE:
        production = _json(package / "production_build_report.json")
        assert top["num_examples_scored"] == production["num_examples_scored"]
        assert top["num_selected_exemplars"] == production["num_selected_exemplars"]
    else:
        assert not (package / "production_build_report.json").exists()


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
