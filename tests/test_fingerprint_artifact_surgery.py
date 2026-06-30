from __future__ import annotations

import json
from pathlib import Path

from radjax_tome.fingerprint import (
    BEHAVIORAL_FINGERPRINT_ARTIFACT_TYPE,
    BEHAVIORAL_FINGERPRINT_VERSION,
    PACKED_TARGET_ARRAYS,
    PROBABILITY_LIKE_STATS,
    TARGET_PAYLOAD_LEGACY_JSONL,
    TARGET_PAYLOAD_PACKED_CORRIDOR_V1,
    FingerprintByteAccounting,
    FingerprintManifest,
    FingerprintValidationResult,
    build_artifact_source_lineage,
    inspect_fingerprint_artifact,
    read_fingerprint_manifest,
    stable_hash,
    summarize_fingerprint_artifact,
    validate_fingerprint_artifact,
    validate_fingerprint_byte_accounting,
    write_fingerprint_manifest,
)


def _minimal_artifact(root: Path) -> Path:
    artifact = root / "fingerprint"
    artifact.mkdir()
    (artifact / "modes.json").write_text(
        json.dumps({"modes": [{"mode_id": 0, "top1_margin": 0.2}]}) + "\n",
        encoding="utf-8",
    )
    (artifact / "targets-00000.jsonl").write_text(
        json.dumps({"example_id": "ex-1", "input_ids": [1, 2, 3], "mode_id": 0}) + "\n",
        encoding="utf-8",
    )
    write_fingerprint_manifest(
        artifact,
        FingerprintManifest(
            artifact_type=BEHAVIORAL_FINGERPRINT_ARTIFACT_TYPE,
            artifact_version=BEHAVIORAL_FINGERPRINT_VERSION,
            created_by="surgery-test",
            teacher={
                "model_name": "teacher",
                "tokenizer_name": "tok",
                "vocab_size": 8,
            },
            sequence={"max_seq_len": 3, "target_positions": 1},
            stats={"tracked": ["top1_margin"]},
            modes_file="modes.json",
            target_shards=({"path": "targets-00000.jsonl", "num_records": 1},),
        ),
    )
    return artifact


def test_manifest_write_read_validate_and_inspect(tmp_path: Path) -> None:
    artifact = _minimal_artifact(tmp_path)

    manifest = read_fingerprint_manifest(artifact)
    assert manifest.artifact_type == BEHAVIORAL_FINGERPRINT_ARTIFACT_TYPE

    validation = validate_fingerprint_artifact(artifact)
    assert validation.ok
    assert validation.metadata["records"] == 1

    summary = summarize_fingerprint_artifact(artifact)
    assert summary.teacher_model_name == "teacher"
    assert summary.num_corridor_records == 1

    inspected = inspect_fingerprint_artifact(artifact)
    assert inspected["artifact_version"] == BEHAVIORAL_FINGERPRINT_VERSION
    assert inspected["num_modes"] == 1


def test_missing_manifest_is_rejected(tmp_path: Path) -> None:
    artifact = tmp_path / "empty-artifact"
    artifact.mkdir()

    validation = validate_fingerprint_artifact(artifact)
    assert not validation.ok
    assert "missing manifest.json" in validation.blockers[0]


def test_byte_accounting_and_source_lineage(tmp_path: Path) -> None:
    artifact = _minimal_artifact(tmp_path)
    accounting = FingerprintByteAccounting(
        declared_byte_budget=100,
        physical_subset_bytes=80,
        logical_payload_bytes_selected=20,
        arm_charged_bytes=20,
        corridor_charged_bytes=12,
        exemplar_charged_bytes=8,
        unused_budget_bytes=80,
    )
    assert validate_fingerprint_byte_accounting(accounting).ok

    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps({"example_id": "ex-1", "text": "hello"}) + "\n",
        encoding="utf-8",
    )
    lineage = build_artifact_source_lineage(artifact, source)
    assert lineage.source_join_kind == "example_id"
    assert lineage.publication_grade_lineage
    assert lineage.artifact_manifest_sha256 is not None
    assert stable_hash({"a": 1}).startswith("sha256:")


def test_archived_fingerprint_schema_symbols_are_active() -> None:
    assert PROBABILITY_LIKE_STATS == frozenset(
        {"top1_margin", "top8_mass", "top32_mass", "tail_mass"}
    )
    assert TARGET_PAYLOAD_LEGACY_JSONL == "legacy_jsonl"
    assert TARGET_PAYLOAD_PACKED_CORRIDOR_V1 == "packed_corridor_v1"
    assert PACKED_TARGET_ARRAYS == {
        "examples_input_ids": 2,
        "position_example_index": 1,
        "position": 1,
        "mode_id": 1,
        "weight": 1,
    }

    result = FingerprintValidationResult(
        ok=False,
        blockers=("missing manifest.json",),
        warnings=("unit warning",),
    )
    assert result.status == "fail"
    assert result.to_dict()["blockers"] == ("missing manifest.json",)
