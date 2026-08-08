"""Published ordinary-production authority for the P6.U1 reduced-burn fixture."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from radjax_contract.tome import (
    open_verified_student_jsonl_records_v6,
    open_verified_student_m7_payload_v6,
    open_verified_student_resource,
    open_verified_student_resource_component_v6,
    open_verified_student_resource_v6,
    validate_and_resolve_student_consumption,
)

from radjax_tome.builder import ProductionBuildConfig, build_production_gpu_tome
from radjax_tome.corpora import inspect_corpus_artifact, validate_corpus_artifact
from radjax_tome.io.json import write_json
from radjax_tome.tome.bundle import validate_tome_bundle
from radjax_tome.tome.packaging import (
    STUDENT,
    package_tome_artifact,
    validate_tome_package,
)
from radjax_tome.tome.v5_fixture import raw_tree_digest, sha256_file, tree_digest

FIXTURE_ID = "native_v3_student_v6_reduced_burn_v1"
FIXTURE_SCHEMA_VERSION = "radjax_tome_p6_u1_reduced_burn_fixture_v1"
DECLARED_INPUT_SCHEMA = "radjax_tome_p6_u1_declared_inputs_v1"
CONTRACT_RELEASE = "0.9.0"
CONTRACT_COMMIT = "1fa43e1aea2e198511db86dafb0aeefa525d48c7"
PROFILE_ID = "native_v3_student_v6"
EXAMPLE_COUNT = 64
SEQUENCE_LENGTH = 64
VALID_TOKEN_COUNT = 4096
SELECTED_ID_COUNT = 60
PUBLIC_BUILD_COMMAND = (
    "python3 scripts/build_v6_reduced_burn_fixture.py --spec "
    "tests/fixtures/p6_u1_reduced_burn_inputs.json "
    "--output <OUTPUT_DIR>"
)

_PRODUCTION_CONFIG = {
    "teacher_backend": "smoke_tokenizer",
    "runtime_mode": "cpu",
    "target_policy": "corridor_exemplar_v1",
    "sequence_length": SEQUENCE_LENGTH,
    "vocab_size": 512,
    "top_k": 4,
    "num_buckets": 3,
    "gpu_batch_size_mode": "preset",
    "gpu_batch_size_preset": 2,
    "shard_size_examples": 8,
    "max_examples": EXAMPLE_COUNT,
    "exemplar_selection_enabled": True,
    "exemplar_delivery_path": "two_pass_rerun_selected",
    "selected_exemplar_budget": 64,
    "total_selected_exemplar_budget": 64,
    "selection_integration_policy": "corridor_first_global_backfill_v1",
    "retain_unselected_exemplar_payloads": False,
}


def canonical_declared_input_bytes(spec: dict[str, Any]) -> bytes:
    return (json.dumps(spec, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def declared_input_digest(spec: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_declared_input_bytes(spec)).hexdigest()


def load_declared_inputs(spec_path: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(spec_path).resolve()
    spec = json.loads(path.read_text(encoding="utf-8"))
    _validate_declared_inputs(spec)
    if spec["schema_version"] != DECLARED_INPUT_SCHEMA:
        raise ValueError("unsupported P6.U1 declared-input schema")
    return path, spec


def build_v6_reduced_burn_fixture(
    output_dir: str | Path,
    *,
    spec_path: str | Path,
    overwrite: bool = False,
    command: str | None = None,
) -> Path:
    """Produce the reduced-burn artifact via canonical production and packaging."""

    spec_file, spec = load_declared_inputs(spec_path)
    root = Path(output_dir).resolve()
    if root.exists() and not overwrite:
        raise ValueError(f"fixture output already exists: {root}")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    inputs = spec_file.parent / spec["input_root"]
    corpus = inputs / spec["corpus"]["dataset_path"]
    manifest = inputs / spec["corpus"]["manifest_path"]
    provenance = inputs / spec["teacher"]["provenance_path"]
    model = inputs / spec["teacher"]["model_path"]
    _validate_input_files(spec, inputs, corpus, manifest, provenance, model)
    production_config = _production_config(spec)
    report = build_production_gpu_tome(
        ProductionBuildConfig(
            teacher_model=str(model),
            tokenizer_id=spec["teacher"]["tokenizer_id"],
            dataset_path=corpus,
            corpus_manifest_path=manifest,
            teacher_model_provenance_path=provenance,
            output_dir=root / "producer_artifact",
            **production_config,
        )
    )
    if report["status"] != "pass":
        raise ValueError(
            "reduced-burn production failed: " + "; ".join(report["blockers"])
        )
    producer = root / "producer_artifact"
    directory = package_tome_artifact(
        producer, root / "student", profile=STUDENT, student_contract_profile="v6"
    ).output_path
    archive = package_tome_artifact(
        producer,
        root / "student.tgz",
        profile=STUDENT,
        archive="tgz",
        student_contract_profile="v6",
    ).output_path
    directory_result = _exercise(directory)
    archive_result = _exercise(archive)
    _validate_vocabulary_identity(directory, spec)
    counts = _qualified_counts(directory)
    if counts != {
        "stable_examples": EXAMPLE_COUNT,
        "valid_tokens": VALID_TOKEN_COUNT,
        "selected_example_ids": SELECTED_ID_COUNT,
    }:
        raise ValueError(f"unexpected reduced-burn qualification: {counts}")
    if counts != _qualified_counts(archive):
        raise ValueError("directory and archive qualification counts differ")
    native_m7 = producer.with_name(f"{producer.name}.v4.tgz")
    packaged_m7 = directory / "student_consumption/v6/selected_exemplar_payload.m7.tgz"
    if native_m7.read_bytes() != packaged_m7.read_bytes():
        raise ValueError("packaged M7 differs from native M7 sibling")
    receipt = {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "fixture_id": FIXTURE_ID,
        "tome_source_commit": _git_commit(),
        "tome_version": "0.1.0",
        "contract": {
            "package": "radjax-contract",
            "release": CONTRACT_RELEASE,
            "commit": CONTRACT_COMMIT,
        },
        "profile_id": PROFILE_ID,
        "declared_inputs": {
            "path": _relative_to_repo(spec_file),
            "schema_version": spec["schema_version"],
            "canonical_sha256": declared_input_digest(spec),
            "record": spec,
        },
        "build_command": command or PUBLIC_BUILD_COMMAND,
        "production_config": production_config,
        "corpus": {
            **inspect_corpus_artifact(manifest.parent),
            "dataset_sha256": sha256_file(corpus),
            "manifest_sha256": sha256_file(manifest),
            "source_records_digest": spec["corpus"]["source_records_digest"],
        },
        "teacher": {
            "tokenizer_id": spec["teacher"]["tokenizer_id"],
            "model_name": spec["teacher"]["model_name"],
            "vocabulary_identity": spec["teacher"]["vocabulary_identity"],
            "model_directory_hash": spec["teacher"]["model_directory_hash"],
            "provenance_sha256": spec["teacher"]["provenance_sha256"],
            "config_sha256": spec["teacher"]["config_sha256"],
            "tokenizer_file_sha256": spec["teacher"]["tokenizer_file_sha256"],
            "tokenizer_identity": spec["teacher"]["tokenizer_identity"],
            "weights_sha256": spec["teacher"]["weights_sha256"],
        },
        "qualification": counts,
        "authority_identity": directory_result["behavioral_authority_digest"],
        "resource_identity": directory_result["composition_digest"],
        "behavioral_source_identity": directory_result["behavioral_source_identity"],
        "package_semantic_identity": directory_result["package_semantic_identity"],
        "directory_identity": directory_result["tree_digest"],
        "directory": directory_result,
        "archive": {
            **archive_result,
            "raw_sha256": sha256_file(archive),
            "size_bytes": archive.stat().st_size,
        },
        "native_m7": {
            "sha256": sha256_file(native_m7),
            "size_bytes": native_m7.stat().st_size,
        },
        "contract_validation": {
            "directory_strict_v6": "pass",
            "archive_strict_v6": "pass",
            "directory_archive_equivalent": True,
        },
        "directory_archive_equivalent": True,
        "reproducibility_pair": {
            "required": True,
            "input_digest": declared_input_digest(spec),
            "verified_by_test": False,
            "verification_receipt": None,
        },
        "claims_not_made": [
            "no_student_training_claim",
            "no_accelerator_execution_claim",
            "no_model_quality_claim",
            "no_cross_environment_path_identity_claim",
        ],
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }
    write_json(root / "FIXTURE_RECEIPT.json", receipt)
    return directory


def build_v6_reduced_burn_pair(
    primary_output: str | Path,
    comparison_output: str | Path,
    *,
    spec_path: str | Path,
) -> Path:
    """Build two fresh outputs and attach machine-readable pair evidence."""

    spec_file, spec = load_declared_inputs(spec_path)
    first_input_bytes = canonical_declared_input_bytes(spec)
    primary = build_v6_reduced_burn_fixture(primary_output, spec_path=spec_file)
    second_spec_file, second_spec = load_declared_inputs(spec_path)
    second_input_bytes = canonical_declared_input_bytes(second_spec)
    if first_input_bytes != second_input_bytes:
        raise ValueError("declared inputs changed between reproducibility builds")
    comparison = build_v6_reduced_burn_fixture(
        comparison_output, spec_path=second_spec_file
    )
    primary_root = primary.parent
    comparison_root = comparison.parent
    primary_receipt = json.loads((primary_root / "FIXTURE_RECEIPT.json").read_text())
    comparison_receipt = json.loads(
        (comparison_root / "FIXTURE_RECEIPT.json").read_text()
    )
    primary_native = primary_root / "producer_artifact.v4.tgz"
    comparison_native = comparison_root / "producer_artifact.v4.tgz"
    primary_archive = primary_root / "student.tgz"
    comparison_archive = comparison_root / "student.tgz"
    evidence = {
        "schema_version": "radjax_tome_p6_u1_reproducibility_pair_v1",
        "fixture_id": FIXTURE_ID,
        "tome_source_commit": primary_receipt["tome_source_commit"],
        "contract": primary_receipt["contract"],
        "declared_inputs": {
            "path": _relative_to_repo(spec_file),
            "canonical_sha256": "sha256:"
            + hashlib.sha256(first_input_bytes).hexdigest(),
            "byte_identical": first_input_bytes == second_input_bytes,
            "primary_record_sha256": "sha256:"
            + hashlib.sha256(first_input_bytes).hexdigest(),
            "comparison_record_sha256": "sha256:"
            + hashlib.sha256(second_input_bytes).hexdigest(),
            "primary_record_bytes": len(first_input_bytes),
            "comparison_record_bytes": len(second_input_bytes),
        },
        "primary": _pair_snapshot(primary_root),
        "comparison": _pair_snapshot(comparison_root),
        "byte_equal": {
            "native_m7": primary_native.read_bytes() == comparison_native.read_bytes(),
            "student_directory": _file_bytes(primary) == _file_bytes(comparison),
            "student_archive": primary_archive.read_bytes()
            == comparison_archive.read_bytes(),
            "receipts_before_pair_annotation": primary_receipt == comparison_receipt,
        },
    }
    if not all(evidence["byte_equal"].values()):
        raise ValueError("fresh reduced-burn builds are not byte-identical")
    pair_path = primary_root / "REPRODUCIBILITY_PAIR.json"
    write_json(pair_path, evidence)
    primary_receipt["reproducibility_pair"] = {
        "required": True,
        "input_digest": "sha256:" + hashlib.sha256(first_input_bytes).hexdigest(),
        "verified_by_test": True,
        "verification_receipt": "REPRODUCIBILITY_PAIR.json",
        "verification_receipt_sha256": sha256_file(pair_path),
    }
    write_json(primary_root / "FIXTURE_RECEIPT.json", primary_receipt)
    return primary


def _validate_declared_inputs(spec: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "fixture_id",
        "contract",
        "input_root",
        "corpus",
        "teacher",
        "behavior",
        "selection",
        "corridor",
        "execution",
        "package",
    }
    if set(spec) != required:
        raise ValueError(
            f"declared-input fields differ: {sorted(set(spec) ^ required)}"
        )
    if (
        spec["contract"]["package"] != "radjax-contract"
        or spec["input_root"] != "p6_u1_reduced_burn_inputs"
        or spec["teacher"]["tokenizer_id"] != "radjax/p6-reduced-burn-fixture"
    ):
        raise ValueError("declared-input authority or input-root mismatch")
    if (
        spec["fixture_id"] != FIXTURE_ID
        or spec["contract"]["commit"] != CONTRACT_COMMIT
    ):
        raise ValueError("declared-input fixture or Contract identity mismatch")
    if (
        spec["contract"]["release"] != CONTRACT_RELEASE
        or spec["contract"]["profile_id"] != PROFILE_ID
    ):
        raise ValueError("declared-input Contract profile mismatch")
    if (
        spec["corpus"]["num_examples"] != EXAMPLE_COUNT
        or spec["selection"]["max_examples"] != EXAMPLE_COUNT
    ):
        raise ValueError("declared-input example count mismatch")
    if spec["behavior"]["sequence_length"] != SEQUENCE_LENGTH:
        raise ValueError("declared-input sequence length mismatch")
    if (
        spec["teacher"]["backend"] != "smoke_tokenizer"
        or spec["teacher"]["runtime_mode"] != "cpu"
    ):
        raise ValueError("declared-input teacher execution mismatch")
    if spec["teacher"]["tokenizer_id"] != "radjax/p6-reduced-burn-fixture":
        raise ValueError("declared-input tokenizer mismatch")
    if spec["behavior"] != {
        "dynamic_mass_threshold": 0.95,
        "dynamic_top_k_max": 32,
        "dynamic_top_k_min": 1,
        "exemplar_delivery_path": "two_pass_rerun_selected",
        "num_buckets": 3,
        "selection_integration_policy": "corridor_first_global_backfill_v1",
        "sequence_length": 64,
        "target_policy": "corridor_exemplar_v1",
        "top_k": 4,
        "vocab_size": 512,
    }:
        raise ValueError("declared-input behavioral policy mismatch")
    if spec["selection"] != {
        "exemplar_leaderboard_capacity": 16,
        "exemplar_score_policy": "entropy_top_n_v1",
        "exemplar_selection_enabled": True,
        "fingerprint_corridor_budget_fraction": "0.50",
        "fingerprint_corridor_budget_max": None,
        "fingerprint_corridor_candidate_pool_cap": 4,
        "fingerprint_corridor_mode_cap": 10,
        "max_examples": 64,
        "retain_unselected_exemplar_payloads": False,
        "require_full_selected_budget": True,
        "selected_exemplar_budget": 64,
        "selected_exemplar_fraction": None,
        "selected_rerun_batch_size": None,
        "stable_example_policy": "all_valid_attention_mask_rows_v1",
        "track_delivery_timing": False,
        "total_selected_exemplar_budget": 64,
    }:
        raise ValueError("declared-input selection policy mismatch")
    if spec["corridor"] != {
        "include_long_tail_in_primary": False,
        "include_perverse_tail_in_primary": False,
        "include_perverse_tail_in_student": False,
        "long_tail_side_board_cap": 128,
        "long_tail_warning_k": 8192,
        "num_buckets": 3,
        "perverse_tail_side_board_cap": 32,
        "perverse_tail_warning_k": 65536,
        "policy_id": "corridor_exemplar_v1",
        "primary_selected_exemplar_budget": None,
        "reject_perverse_exemplars": False,
        "very_long_tail_warning_k": 32768,
    }:
        raise ValueError("declared-input corridor policy mismatch")
    if spec["execution"] != {
        "artifact_contract_version": "v2",
        "concurrency_policy": "serial_cpu_smoke_v1",
        "gpu_batch_size_mode": "preset",
        "gpu_batch_size_preset": 2,
        "gpu_batch_size_auto_max": 64,
        "gpu_batch_size_auto_min": 1,
        "gpu_batch_size_custom": None,
        "fail_on_plan_warnings": False,
        "max_artifact_bytes": None,
        "no_build_if_plan_warn": False,
        "overwrite": False,
        "payload_records_per_shard": 128,
        "progress": False,
        "resume": False,
        "seed": None,
        "shard_size_examples": 8,
        "strict_provenance": True,
        "temporary_paths_are_nonsemantic": True,
        "worker_count": 1,
    }:
        raise ValueError("declared-input execution policy mismatch")
    if spec["package"] != {
        "profile": "student",
        "transport_pair": ["directory", "tgz"],
        "student_contract_profile": "v6",
    }:
        raise ValueError("declared-input package profile mismatch")


def _validate_input_files(
    spec: dict[str, Any],
    inputs: Path,
    corpus: Path,
    manifest: Path,
    provenance: Path,
    model: Path,
) -> None:
    validation = validate_corpus_artifact(manifest.parent)
    if validation.status != "pass":
        raise ValueError(
            "declared corpus failed validation: " + "; ".join(validation.blockers)
        )
    expected = spec["corpus"]
    if (
        validation.corpus_hash != expected["corpus_hash"]
        or validation.manifest_hash != expected["manifest_hash"]
    ):
        raise ValueError("declared corpus identity changed")
    if sha256_file(manifest) != expected["manifest_file_sha256"]:
        raise ValueError("declared corpus manifest bytes changed")
    if (
        sha256_file(
            manifest.parent / spec["corpus"]["build_report_path"].split("/", 1)[1]
        )
        != expected["build_report_sha256"]
    ):
        raise ValueError("declared corpus build report bytes changed")
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    for key in (
        "source_discovery_policy",
        "normalization_policy",
        "chunking_policy",
        "deduplication_policy",
        "num_examples",
        "num_sources",
    ):
        if manifest_payload.get(key) != expected[key]:
            raise ValueError(f"declared corpus {key} changed")
    if manifest_payload.get("source_roots") != [expected["source_root"]]:
        raise ValueError("declared corpus source root changed")
    source_digest = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                manifest_payload["source_records"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    )
    if source_digest != expected["source_records_digest"]:
        raise ValueError("declared source-passport identity changed")
    if not corpus.is_file() or not provenance.is_file() or not model.is_dir():
        raise ValueError("declared input file is missing")
    if sha256_file(provenance) != spec["teacher"]["provenance_sha256"]:
        raise ValueError("declared teacher provenance bytes changed")
    provenance_payload = json.loads(provenance.read_text(encoding="utf-8"))
    if provenance_payload.get("model_name") != spec["teacher"]["model_name"]:
        raise ValueError("declared teacher model identity changed")
    for key, relative in (
        ("config_sha256", "config.json"),
        ("tokenizer_file_sha256", "tokenizer.json"),
        ("weights_sha256", "model.safetensors"),
    ):
        if sha256_file(model / relative) != spec["teacher"][key]:
            raise ValueError(f"declared teacher {relative} bytes changed")
    if (
        provenance_payload.get("model_directory_hash")
        != spec["teacher"]["model_directory_hash"]
    ):
        raise ValueError("declared teacher model identity changed")


def _production_config(spec: dict[str, Any]) -> dict[str, Any]:
    teacher = spec["teacher"]
    behavior = spec["behavior"]
    selection = spec["selection"]
    execution = spec["execution"]
    corridor = spec["corridor"]
    return {
        "teacher_backend": teacher["backend"],
        "artifact_contract_version": execution["artifact_contract_version"],
        "runtime_mode": teacher["runtime_mode"],
        "target_policy": behavior["target_policy"],
        "sequence_length": behavior["sequence_length"],
        "vocab_size": behavior["vocab_size"],
        "top_k": behavior["top_k"],
        "num_buckets": behavior["num_buckets"],
        "dynamic_top_k_min": behavior["dynamic_top_k_min"],
        "dynamic_top_k_max": behavior["dynamic_top_k_max"],
        "dynamic_mass_threshold": behavior["dynamic_mass_threshold"],
        "long_tail_warning_k": corridor["long_tail_warning_k"],
        "very_long_tail_warning_k": corridor["very_long_tail_warning_k"],
        "perverse_tail_warning_k": corridor["perverse_tail_warning_k"],
        "reject_perverse_exemplars": corridor["reject_perverse_exemplars"],
        "primary_selected_exemplar_budget": corridor[
            "primary_selected_exemplar_budget"
        ],
        "long_tail_side_board_cap": corridor["long_tail_side_board_cap"],
        "perverse_tail_side_board_cap": corridor["perverse_tail_side_board_cap"],
        "include_long_tail_in_primary": corridor["include_long_tail_in_primary"],
        "include_perverse_tail_in_primary": corridor[
            "include_perverse_tail_in_primary"
        ],
        "gpu_batch_size_mode": execution["gpu_batch_size_mode"],
        "gpu_batch_size_preset": execution["gpu_batch_size_preset"],
        "gpu_batch_size_custom": execution["gpu_batch_size_custom"],
        "gpu_batch_size_auto_min": execution["gpu_batch_size_auto_min"],
        "gpu_batch_size_auto_max": execution["gpu_batch_size_auto_max"],
        "shard_size_examples": execution["shard_size_examples"],
        "payload_records_per_shard": execution["payload_records_per_shard"],
        "resume": execution["resume"],
        "overwrite": execution["overwrite"],
        "strict_provenance": execution["strict_provenance"],
        "fail_on_plan_warnings": execution["fail_on_plan_warnings"],
        "no_build_if_plan_warn": execution["no_build_if_plan_warn"],
        "max_artifact_bytes": execution["max_artifact_bytes"],
        "progress": execution["progress"],
        "max_examples": selection["max_examples"],
        "exemplar_selection_enabled": selection["exemplar_selection_enabled"],
        "exemplar_leaderboard_capacity": selection["exemplar_leaderboard_capacity"],
        "selected_exemplar_fraction": selection["selected_exemplar_fraction"],
        "exemplar_score_policy": selection["exemplar_score_policy"],
        "selected_rerun_batch_size": selection["selected_rerun_batch_size"],
        "track_delivery_timing": selection["track_delivery_timing"],
        "exemplar_delivery_path": behavior["exemplar_delivery_path"],
        "selected_exemplar_budget": selection["selected_exemplar_budget"],
        "total_selected_exemplar_budget": selection["total_selected_exemplar_budget"],
        "selection_integration_policy": behavior["selection_integration_policy"],
        "retain_unselected_exemplar_payloads": selection[
            "retain_unselected_exemplar_payloads"
        ],
        "fingerprint_corridor_budget_fraction": selection[
            "fingerprint_corridor_budget_fraction"
        ],
        "fingerprint_corridor_budget_max": selection["fingerprint_corridor_budget_max"],
        "fingerprint_corridor_mode_cap": selection["fingerprint_corridor_mode_cap"],
        "fingerprint_corridor_candidate_pool_cap": selection[
            "fingerprint_corridor_candidate_pool_cap"
        ],
        "require_full_selected_budget": selection["require_full_selected_budget"],
        "include_perverse_tail_in_student": corridor[
            "include_perverse_tail_in_student"
        ],
    }


def _validate_vocabulary_identity(directory: Path, spec: dict[str, Any]) -> None:
    binding = json.loads(
        (directory / "manifests/language_tokenizer_binding_v1.json").read_text(
            encoding="utf-8"
        )
    )
    actual = binding["vocabulary"]["vocabulary_identity"]
    if actual != spec["teacher"]["vocabulary_identity"]:
        raise ValueError("declared vocabulary identity changed")
    teacher_manifest = directory / "teacher_manifest.json"
    if teacher_manifest.is_file():
        manifest = json.loads(teacher_manifest.read_text(encoding="utf-8"))
        provenance = manifest.get("teacher_model_provenance", {})
        if manifest.get("tokenizer_id") != spec["teacher"]["tokenizer_id"]:
            raise ValueError("declared tokenizer identifier changed")
        if provenance.get("tokenizer_hash") != spec["teacher"]["tokenizer_identity"]:
            raise ValueError("declared tokenizer identity changed")


def _pair_snapshot(root: Path) -> dict[str, Any]:
    receipt = json.loads((root / "FIXTURE_RECEIPT.json").read_text())
    return {
        "native_m7_sha256": receipt["native_m7"]["sha256"],
        "native_m7_size_bytes": receipt["native_m7"]["size_bytes"],
        "student_archive_sha256": receipt["archive"]["raw_sha256"],
        "student_archive_size_bytes": receipt["archive"]["size_bytes"],
        "student_directory_identity": receipt["directory"]["tree_digest"],
        "package_semantic_identity": receipt["directory"]["package_semantic_identity"],
        "resource_identity": receipt["directory"]["composition_digest"],
        "authority_identity": receipt["directory"]["behavioral_authority_digest"],
        "behavioral_source_identity": receipt["directory"][
            "behavioral_source_identity"
        ],
        "qualification": receipt["qualification"],
    }


def _file_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _exercise(artifact: Path) -> dict[str, Any]:
    if artifact.is_dir() and not validate_tome_package(artifact, profile=STUDENT).ok:
        raise ValueError("v6 package validation failed")
    if artifact.is_file() and not validate_tome_bundle(artifact).ok:
        raise ValueError("v6 archive validation failed")
    result = validate_and_resolve_student_consumption(
        artifact, profile_id=PROFILE_ID, strict=True
    )
    if not result.ok or result.descriptor is None:
        raise ValueError(
            "Contract v6 admission failed: " + ",".join(i.code for i in result.issues)
        )
    descriptor = result.descriptor
    for resource in (
        *descriptor.authority_resources,
        *descriptor.non_authority_resources,
    ):
        if resource.encoding == "m7_tome_archive":
            with open_verified_student_m7_payload_v6(
                artifact, resource.resource_id, strict=True
            ) as reader:
                list(reader)
                if reader.verification_state != "fully_verified":
                    raise ValueError("M7 stream was not fully verified")
        elif resource.encoding == "jsonl":
            with open_verified_student_jsonl_records_v6(
                artifact, resource.resource_id, strict=True
            ) as records:
                list(records)
            with open_verified_student_resource(
                artifact, resource.resource_id, profile_id=PROFILE_ID, strict=True
            ) as handle:
                handle.read()
            with open_verified_student_resource_v6(
                artifact, resource.resource_id, strict=True
            ) as handle:
                handle.read()
        else:
            with open_verified_student_resource(
                artifact, resource.resource_id, profile_id=PROFILE_ID, strict=True
            ) as handle:
                handle.read()
            with open_verified_student_resource_v6(
                artifact, resource.resource_id, strict=True
            ) as handle:
                handle.read()
    return {
        "behavioral_authority_digest": descriptor.behavioral_authority_digest,
        "composition_digest": descriptor.composition_digest,
        "package_semantic_identity": descriptor.package_semantic_identity,
        "language_binding_digest": descriptor.language_binding_digest,
        "behavioral_source_identity": descriptor.behavioral_source_identity,
        "authority_resources": [
            item.to_dict() for item in descriptor.authority_resources
        ],
        "non_authority_resources": [
            item.to_dict() for item in descriptor.non_authority_resources
        ],
        "required_joins": list(descriptor.required_joins),
        "tree_digest": tree_digest(artifact) if artifact.is_dir() else None,
        "raw_tree_digest": raw_tree_digest(artifact) if artifact.is_dir() else None,
    }


def _qualified_counts(artifact: Path) -> dict[str, int]:
    with open_verified_student_resource_component_v6(
        artifact, "target_shard/default", "attention_mask", strict=True
    ) as component:
        attention_mask = np.load(component.content, allow_pickle=False)
    with open_verified_student_resource_component_v6(
        artifact, "target_shard/default", "input_ids", strict=True
    ) as component:
        input_ids = np.load(component.content, allow_pickle=False)
    with open_verified_student_m7_payload_v6(
        artifact, "selected_exemplar_payload/default", strict=True
    ) as records:
        selected_ids = {str(row["selected_example_id"]) for row in records}
    return {
        "stable_examples": int(input_ids.shape[0]),
        "valid_tokens": int(np.asarray(attention_mask).sum()),
        "selected_example_ids": len(selected_ids),
    }


def _relative_to_repo(path: Path) -> str:
    try:
        return path.relative_to(_repo_root()).as_posix()
    except ValueError:
        return str(path)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_repo_root(), text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


__all__ = [
    "CONTRACT_COMMIT",
    "CONTRACT_RELEASE",
    "DECLARED_INPUT_SCHEMA",
    "EXAMPLE_COUNT",
    "FIXTURE_ID",
    "PUBLIC_BUILD_COMMAND",
    "PROFILE_ID",
    "SELECTED_ID_COUNT",
    "SEQUENCE_LENGTH",
    "VALID_TOKEN_COUNT",
    "build_v6_reduced_burn_fixture",
    "build_v6_reduced_burn_pair",
    "canonical_declared_input_bytes",
    "declared_input_digest",
    "load_declared_inputs",
]
