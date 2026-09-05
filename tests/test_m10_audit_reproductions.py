from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from radjax_tome.builder.production import ProductionBuildConfig
from radjax_tome.builder.production_stages.preflight import validate_required_inputs
from radjax_tome.corpora import build_corpus_artifact_v2, load_corpus_build_intent
from radjax_tome.corpora.dedup import deduplicate_records
from radjax_tome.corpora.records import SourceRecord
from radjax_tome.corpora.validation import validate_corpus_artifact_v2


def _intent(root: Path, *, overwrite: bool = False, capacity: int = 128) -> Path:
    source = root / "src"
    source.mkdir(parents=True)
    (source / "rows.jsonl").write_text(
        "".join(json.dumps({"text": f"row-{i}"}) + "\n" for i in range(300)),
        encoding="utf-8",
    )
    payload = {
        "schema_version": "radjax_tome_corpus_build_intent_v1",
        "artifact": {"schema_version": "radjax_tome_corpus_artifact_v2"},
        "sources": [
            {
                "source_id": "rows",
                "adapter": "local_jsonl_text_v1",
                "path": "src/rows.jsonl",
            }
        ],
        "policy": {
            "normalization": "text_normalize_lf_strip_trailing_ws_v1",
            "filtering": {"min_chars": 1},
            "chunking": {"name": "char_window_v1", "max_chars": 64},
            "deduplication": {"enabled": True},
            "ordering": "declared_source_ordinal_logical_locator_chunk_index_v1",
            "tokenizer": "smoke",
        },
        "layout": {"shard_capacity": capacity},
        "resources": {
            "memory_limit": "64MB",
            "duckdb_memory_limit": "64MB",
            "worker_count": 1,
            "max_open_files": 8,
        },
        "output": {"artifact_path": "artifact"},
        "execution": {"resume": False, "overwrite": overwrite},
        "reporting": {"progress": False},
    }
    path = root / "intent.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_overwrite_rejects_unrelated_nonempty_destination(tmp_path: Path) -> None:
    intent_path = _intent(tmp_path, overwrite=True)
    destination = tmp_path / "artifact"
    destination.mkdir()
    sentinel = destination / "unrelated.txt"
    sentinel.write_text("do not delete", encoding="utf-8")
    with pytest.raises(ValueError):
        build_corpus_artifact_v2(load_corpus_build_intent(intent_path))
    assert sentinel.exists()


def test_dedup_more_than_one_fetch_batch_reports_all_winners() -> None:
    records = [
        SourceRecord(
            source_id="s",
            source_ordinal=0,
            logical_locator=f"r-{index:04d}",
            chunk_index=0,
            chunk_count=1,
            text=f"value-{index}",
            normalized_text_digest=f"sha256:{index:064x}",
            source_digest="sha256:" + "a" * 64,
        )
        for index in range(300)
    ]
    records.extend(
        SourceRecord(
            source_id="dup",
            source_ordinal=1,
            logical_locator=f"d-{index:04d}",
            chunk_index=0,
            chunk_count=1,
            text="value-0",
            normalized_text_digest="sha256:" + "0" * 64,
            source_digest="sha256:" + "b" * 64,
        )
        for index in range(10)
    )
    winners, counts = deduplicate_records(records)
    emitted = list(winners)
    assert counts["output_records"] == len(emitted) == 300
    assert emitted[-1].logical_locator == "r-0299"
    assert emitted[0].duplicate_provenance == tuple(
        f"dup:d-{index:04d}:0" for index in range(10)
    )


def test_dedup_large_group_persists_complete_provenance(tmp_path: Path) -> None:
    records = [
        SourceRecord(
            source_id="s",
            source_ordinal=0,
            logical_locator=f"r-{index:04d}",
            chunk_index=0,
            chunk_count=1,
            text="same",
            normalized_text_digest="sha256:" + "1" * 64,
            source_digest="sha256:" + "a" * 64,
        )
        for index in range(101)
    ]
    sidecar = tmp_path / "duplicate_provenance.jsonl"
    winners, counts = deduplicate_records(records, provenance_path=sidecar)
    emitted = list(winners)
    assert counts["output_records"] == len(emitted) == 1
    provenance = sidecar.read_text(encoding="utf-8").splitlines()
    assert len(provenance) == 101
    assert emitted[0].duplicate_count == 101


def test_v2_validation_rejects_undeclared_public_member(tmp_path: Path) -> None:
    build_corpus_artifact_v2(load_corpus_build_intent(_intent(tmp_path)))
    (tmp_path / "artifact" / "UNDECLARED.txt").write_text("x")
    assert not validate_corpus_artifact_v2(tmp_path / "artifact").ok


def test_v2_validation_rejects_binding_substitution(tmp_path: Path) -> None:
    build_corpus_artifact_v2(load_corpus_build_intent(_intent(tmp_path)))
    binding = tmp_path / "artifact" / "language_tokenizer_binding_v1.json"
    payload = json.loads(binding.read_text())
    payload["tokenizer"] = {"forged": True}
    binding.write_text(json.dumps(payload), encoding="utf-8")
    assert not validate_corpus_artifact_v2(tmp_path / "artifact").ok


def test_v2_validation_rejects_duplicate_provenance_sidecar_substitution(
    tmp_path: Path,
) -> None:
    intent_path = _intent(tmp_path)
    source = tmp_path / "src" / "rows.jsonl"
    source.write_text(
        json.dumps({"text": "same"}) + "\n" + json.dumps({"text": "same"}) + "\n",
        encoding="utf-8",
    )
    build_corpus_artifact_v2(load_corpus_build_intent(intent_path))
    root = tmp_path / "artifact"
    sidecar = root / "duplicate_provenance.jsonl"
    sidecar.write_text(
        json.dumps({"winner": 1, "locator": "rows:forged:0"}) + "\n",
        encoding="utf-8",
    )
    cover_path = root / "corpus_cover.json"
    cover = json.loads(cover_path.read_text(encoding="utf-8"))
    for member in cover["members"]:
        if member["path"] == "duplicate_provenance.jsonl":
            member["size_bytes"] = sidecar.stat().st_size
            member["sha256"] = (
                "sha256:" + hashlib.sha256(sidecar.read_bytes()).hexdigest()
            )
    cover_path.write_text(json.dumps(cover), encoding="utf-8")
    result = validate_corpus_artifact_v2(root)
    assert not result.ok
    assert any(issue.code == "PROVENANCE_MISMATCH" for issue in result.issues)


def test_v2_preflight_accepts_matching_smoke_tokenizer_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent_path = _intent(tmp_path)
    report = build_corpus_artifact_v2(load_corpus_build_intent(intent_path))
    config = ProductionBuildConfig(
        teacher_model="teacher",
        tokenizer_id="smoke",
        dataset_path=tmp_path / "unused-dataset.jsonl",
        corpus_manifest_path=tmp_path / "artifact",
        teacher_model_provenance_path=tmp_path / "provenance.json",
        output_dir=tmp_path / "output",
        expected_corpus_semantic_identity=str(report["semantic_identity"]),
    )
    (tmp_path / "provenance.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "radjax_tome.builder.production_stages.preflight.validate_teacher_model_provenance",
        lambda path: type("TeacherReport", (), {"blockers": ()})(),
    )
    blockers: list[str] = []
    validate_required_inputs(config, blockers)
    assert blockers == []


def test_v2_preflight_resolves_missing_tokenizer_id_like_production(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent_path = _intent(tmp_path)
    report = build_corpus_artifact_v2(load_corpus_build_intent(intent_path))
    config = ProductionBuildConfig(
        teacher_model="smoke",
        tokenizer_id=None,
        dataset_path=tmp_path / "unused-dataset.jsonl",
        corpus_manifest_path=tmp_path / "artifact",
        teacher_model_provenance_path=tmp_path / "provenance.json",
        output_dir=tmp_path / "output",
        expected_corpus_semantic_identity=str(report["semantic_identity"]),
    )
    (tmp_path / "provenance.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "radjax_tome.builder.production_stages.preflight.validate_teacher_model_provenance",
        lambda path: type("TeacherReport", (), {"blockers": ()})(),
    )
    blockers: list[str] = []
    validate_required_inputs(config, blockers)
    assert blockers == []


def test_v2_preflight_rejects_different_tokenizer_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent_path = _intent(tmp_path)
    report = build_corpus_artifact_v2(load_corpus_build_intent(intent_path))
    config = ProductionBuildConfig(
        teacher_model="teacher",
        tokenizer_id="other",
        dataset_path=tmp_path / "unused-dataset.jsonl",
        corpus_manifest_path=tmp_path / "artifact",
        teacher_model_provenance_path=tmp_path / "provenance.json",
        output_dir=tmp_path / "output",
        expected_corpus_semantic_identity=str(report["semantic_identity"]),
    )
    (tmp_path / "provenance.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "radjax_tome.builder.production_stages.preflight.validate_teacher_model_provenance",
        lambda path: type("TeacherReport", (), {"blockers": ()})(),
    )
    from radjax_tome.corpora.tokenizer import SmokeTokenizer

    monkeypatch.setattr(
        "radjax_tome.corpora.tokenizer.create_tokenizer",
        lambda value: SmokeTokenizer(vocab_size=513),
    )
    blockers: list[str] = []
    validate_required_inputs(config, blockers)
    assert any("CORPUS_TOKENIZER_BINDING_MISMATCH" in item for item in blockers)
