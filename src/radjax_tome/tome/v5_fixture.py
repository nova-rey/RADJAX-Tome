"""Deterministic evidence fixture for the native v5 Student package path."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from radjax_contract.tome import validate_and_resolve_student_consumption

from radjax_tome.builder import ProductionBuildConfig, build_production_gpu_tome
from radjax_tome.corpora import CorpusBuildConfig, build_corpus_artifact
from radjax_tome.io.json import read_json_object, write_json
from radjax_tome.provenance import (
    inspect_teacher_model,
    write_teacher_model_provenance,
)
from radjax_tome.tome.packaging import (
    STUDENT,
    package_tome_artifact,
    validate_tome_package,
)

FIXTURE_ID = "native_v3_student_v5_smoke_v1"
FIXTURE_SCHEMA_VERSION = "radjax_tome_v5_fixture_v1"
FIXTURE_TOME_COMMIT = "3861c23"
CONTRACT_VERSION = "v0.7.0"
CONTRACT_COMMIT = "cac3dd21e0d56df5a9e6fd50b20267e0b8960995"
PROFILE_ID = "native_v3_student_v5"

_SOURCE_TEXTS = ("alpha", "beta", "gamma", "delta")
_PRODUCTION_CONFIG = {
    "teacher_backend": "smoke_tokenizer",
    "runtime_mode": "cpu",
    "target_policy": "corridor_exemplar_v1",
    "sequence_length": 8,
    "vocab_size": 512,
    "top_k": 4,
    "num_buckets": 3,
    "gpu_batch_size_mode": "preset",
    "gpu_batch_size_preset": 2,
    "shard_size_examples": 2,
    "max_examples": 4,
    "exemplar_selection_enabled": True,
    "exemplar_delivery_path": "two_pass_rerun_selected",
    "selected_exemplar_budget": 2,
    "total_selected_exemplar_budget": 2,
    "selection_integration_policy": "corridor_first_global_backfill_v1",
    "retain_unselected_exemplar_payloads": False,
}


def build_v5_language_tokenizer_fixture(
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    tome_commit: str = FIXTURE_TOME_COMMIT,
) -> Path:
    """Build one ordinary smoke-tokenizer v5 Student package fixture.

    The source artifact is made by the regular production builder and the
    Student package is made by its regular transactional package writer.  The
    checked-in receipt records physical bytes for this one output; repeated
    builds prove the binding, vocabulary, and semantic identity instead of
    treating path-bearing operational reports as semantic data.
    """

    root = Path(output_dir)
    if root.exists():
        if not overwrite:
            raise ValueError(f"fixture output already exists: {root}")
        shutil.rmtree(root)
    root.mkdir(parents=True)

    corpus, model, provenance = _prepare_inputs(root)
    producer = root / "producer_artifact"
    config = ProductionBuildConfig(
        teacher_model=str(model),
        tokenizer_id="radjax/smoke-v5-fixture",
        dataset_path=corpus / "corpus.jsonl",
        corpus_manifest_path=corpus / "corpus_manifest.json",
        teacher_model_provenance_path=provenance,
        output_dir=producer,
        **_PRODUCTION_CONFIG,
    )
    report = build_production_gpu_tome(config)
    if report["status"] != "pass":
        raise ValueError(
            "v5 fixture production failed: " + "; ".join(report["blockers"])
        )

    student = package_tome_artifact(producer, root / "student", profile=STUDENT)
    package_root = student.output_path
    tome_report = validate_tome_package(package_root, profile=STUDENT)
    if not tome_report.ok:
        raise ValueError(
            "v5 fixture Tome validation failed: " + "; ".join(tome_report.blockers)
        )
    contract_result = validate_and_resolve_student_consumption(
        package_root,
        profile_id=PROFILE_ID,
        strict=True,
    )
    if not contract_result.ok:
        codes = ",".join(issue.code for issue in contract_result.issues)
        raise ValueError(f"v5 fixture Contract validation failed: {codes}")

    source_binding = producer / "language_tokenizer_binding_v1.json"
    package_binding = package_root / "manifests/language_tokenizer_binding_v1.json"
    source_vocabulary = producer / "resources/tokenizer_vocabulary.jsonl"
    package_vocabulary = package_root / "resources/tokenizer_vocabulary.jsonl"
    if source_binding.read_bytes() != package_binding.read_bytes():
        raise ValueError("v5 fixture package binding differs from the producer capture")
    if source_vocabulary.read_bytes() != package_vocabulary.read_bytes():
        raise ValueError(
            "v5 fixture package vocabulary differs from the producer capture"
        )

    binding = read_json_object(package_binding)
    cover = read_json_object(package_root / "cover_page.json")
    write_json(
        root / "FIXTURE_RECEIPT.json",
        {
            "schema_version": FIXTURE_SCHEMA_VERSION,
            "fixture_id": FIXTURE_ID,
            "contract": {"version": CONTRACT_VERSION, "commit": CONTRACT_COMMIT},
            "tome_commit_at_production": tome_commit,
            "profile_id": PROFILE_ID,
            "generic_binding_digest": binding["canonical_binding_digest"],
            "fixture_semantic_digest": cover["identity"]["semantic_digest"],
            "fixture_raw_digest": raw_tree_digest(package_root),
            "fixture_tree_digest": tree_digest(package_root),
            "binding_sha256": sha256_file(package_binding),
            "vocabulary_sha256": sha256_file(package_vocabulary),
            "validator": {
                "entry_point": (
                    "radjax_contract.tome.validate_and_resolve_student_consumption"
                ),
                "arguments": {"profile_id": PROFILE_ID, "strict": True},
            },
            "deterministic_producer": {
                "entry_point": (
                    "radjax_tome.tome.v5_fixture.build_v5_language_tokenizer_fixture"
                ),
                "command": (
                    "python3 scripts/build_v5_language_tokenizer_fixture.py "
                    "--output tests/fixtures/native_v3_student_v5_smoke"
                ),
                "source_texts": list(_SOURCE_TEXTS),
                "production_config": _PRODUCTION_CONFIG,
            },
            "claims_not_made": [
                "no_model_quality_claim",
                "no_network_verification_claim",
                "no_student_training_claim",
                "no_accelerator_execution_claim",
            ],
        },
    )
    return package_root


def sha256_file(path: str | Path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw_tree_digest(root: str | Path) -> str:
    """Digest exact file bytes and canonical relative names in a package tree."""

    digest = hashlib.sha256()
    for path in _files(root):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def tree_digest(root: str | Path) -> str:
    """Digest canonical file names and their raw SHA-256 digests."""

    digest = hashlib.sha256()
    for path in _files(root):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
        digest.update(b"\n")
    return "sha256:" + digest.hexdigest()


def _prepare_inputs(root: Path) -> tuple[Path, Path, Path]:
    inputs = root / "inputs"
    inputs.mkdir()
    source_paths = []
    for index, text in enumerate(_SOURCE_TEXTS):
        path = inputs / f"source-{index}.txt"
        path.write_text(text, encoding="utf-8")
        source_paths.append(path)
    corpus = inputs / "corpus"
    build_corpus_artifact(
        CorpusBuildConfig(inputs=tuple(source_paths), output_dir=corpus, overwrite=True)
    )
    model = inputs / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"tiny"}', encoding="utf-8")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")
    provenance = inputs / "teacher_provenance.json"
    write_teacher_model_provenance(
        inspect_teacher_model(model, model_name="radjax/v5-fixture"), provenance
    )
    return corpus, model, provenance


def _files(root: str | Path) -> list[Path]:
    path = Path(root)
    return sorted(item for item in path.rglob("*") if item.is_file())


__all__ = [
    "CONTRACT_COMMIT",
    "CONTRACT_VERSION",
    "FIXTURE_ID",
    "FIXTURE_SCHEMA_VERSION",
    "FIXTURE_TOME_COMMIT",
    "PROFILE_ID",
    "build_v5_language_tokenizer_fixture",
    "raw_tree_digest",
    "sha256_file",
    "tree_digest",
]
