"""Ordinary-production evidence fixture for explicit native-v3 Student v6."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

from radjax_contract.tome import (
    open_verified_student_jsonl_records_v6,
    open_verified_student_m7_payload_v6,
    open_verified_student_resource,
    open_verified_student_resource_v6,
    validate_and_resolve_student_consumption,
)

from radjax_tome.builder import ProductionBuildConfig, build_production_gpu_tome
from radjax_tome.io.json import write_json
from radjax_tome.tome.bundle import validate_tome_bundle
from radjax_tome.tome.packaging import (
    STUDENT,
    package_tome_artifact,
    validate_tome_package,
)
from radjax_tome.tome.v5_fixture import (
    _PRODUCTION_CONFIG as _V5_PRODUCTION_CONFIG,
)
from radjax_tome.tome.v5_fixture import (
    CONTRACT_COMMIT as V5_HISTORICAL_CONTRACT_COMMIT,
)
from radjax_tome.tome.v5_fixture import (
    _prepare_inputs,
    raw_tree_digest,
    sha256_file,
    tree_digest,
)

FIXTURE_ID = "native_v3_student_v6_smoke_v1"
FIXTURE_SCHEMA_VERSION = "radjax_tome_v6_fixture_v1"
CONTRACT_VERSION = "v0.8.0"
CONTRACT_COMMIT = "b3275b8769c36b6261f4f241c47f0066c651e869"
PROFILE_ID = "native_v3_student_v6"

# Keep the v5 smoke fixture immutable while making this ordinary v6 fixture
# large enough to provide leakage-free exemplar train/held-out evidence.
_PRODUCTION_CONFIG = {
    **_V5_PRODUCTION_CONFIG,
    "selected_exemplar_budget": 3,
    "total_selected_exemplar_budget": 3,
}


def build_v6_behavioral_fixture(
    output_dir: str | Path, *, overwrite: bool = False
) -> Path:
    """Produce directory and archive v6 fixtures via the canonical writer."""

    root = Path(output_dir)
    if root.exists() and not overwrite:
        raise ValueError(f"fixture output already exists: {root}")
    if root.exists():
        import shutil

        shutil.rmtree(root)
    root.mkdir(parents=True)
    corpus, model, provenance = _prepare_inputs(root)
    producer = root / "producer_artifact"
    report = build_production_gpu_tome(
        ProductionBuildConfig(
            teacher_model=str(model),
            tokenizer_id="radjax/smoke-v6-fixture",
            dataset_path=corpus / "corpus.jsonl",
            corpus_manifest_path=corpus / "corpus_manifest.json",
            teacher_model_provenance_path=provenance,
            output_dir=producer,
            **_PRODUCTION_CONFIG,
        )
    )
    if report["status"] != "pass":
        raise ValueError(
            "v6 fixture production failed: " + "; ".join(report["blockers"])
        )
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
    source_m7 = producer.with_name(f"{producer.name}.v4.tgz")
    packaged_m7 = directory / "student_consumption/v6/selected_exemplar_payload.m7.tgz"
    if source_m7.read_bytes() != packaged_m7.read_bytes():
        raise ValueError("fixture packaged M7 member differs from native sibling")
    write_json(
        root / "FIXTURE_RECEIPT.json",
        {
            "schema_version": FIXTURE_SCHEMA_VERSION,
            "fixture_id": FIXTURE_ID,
            "contract": {"version": CONTRACT_VERSION, "commit": CONTRACT_COMMIT},
            "profile_id": PROFILE_ID,
            "historical_v5_mirror_commit": V5_HISTORICAL_CONTRACT_COMMIT,
            "directory": directory_result,
            "archive": {**archive_result, "raw_sha256": sha256_file(archive)},
            "native_m7_sibling": {
                "locator": f"{producer.name}.v4.tgz",
                "sha256": sha256_file(source_m7),
                "size_bytes": source_m7.stat().st_size,
                "packaged_locator": (
                    "student_consumption/v6/selected_exemplar_payload.m7.tgz"
                ),
            },
            "production_config": _PRODUCTION_CONFIG,
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
            "claims_not_made": [
                "no_student_training_claim",
                "no_accelerator_execution_claim",
                "no_default_profile_migration",
            ],
        },
    )
    return directory


def _exercise(artifact: Path) -> dict[str, object]:
    if artifact.is_dir() and not validate_tome_package(artifact, profile=STUDENT).ok:
        raise ValueError("fixture Tome package validation failed")
    if artifact.is_file() and not validate_tome_bundle(artifact).ok:
        raise ValueError("fixture Tome archive validation failed")
    result = validate_and_resolve_student_consumption(
        artifact, profile_id=PROFILE_ID, strict=True
    )
    if not result.ok or result.descriptor is None:
        raise ValueError(
            "fixture Contract admission failed: "
            + ",".join(item.code for item in result.issues)
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
                    raise ValueError("fixture M7 opener did not fully verify")
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
            # Exercise the stable public dispatcher as well as the v6-specific
            # byte opener. M7 intentionally has its dedicated stream protocol.
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
