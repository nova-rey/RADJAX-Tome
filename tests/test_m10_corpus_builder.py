from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from radjax_tome.builder.corpus_input import resolve_corpus_input
from radjax_tome.corpora import (
    build_corpus_artifact_v2,
    inspect_corpus_artifact_v2,
    load_corpus_build_intent,
    open_verified_corpus,
    validate_corpus_artifact_v2,
)


def _config(root: Path, output: str = "artifact") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    source = root / "sources"
    source.mkdir(exist_ok=True)
    (source / "b.txt").write_text("duplicate  \r\nsecond", encoding="utf-8")
    (source / "a.md").write_text("duplicate", encoding="utf-8")
    payload = {
        "schema_version": "radjax_tome_corpus_build_intent_v1",
        "artifact": {"schema_version": "radjax_tome_corpus_artifact_v2"},
        "sources": [
            {
                "source_id": "text",
                "adapter": "local_text_tree_v1",
                "path": "sources",
            }
        ],
        "policy": {
            "normalization": "text_normalize_lf_strip_trailing_ws_v1",
            "filtering": {"min_chars": 1},
            "chunking": {"name": "char_window_v1", "max_chars": 100},
            "deduplication": {"enabled": True},
            "ordering": "declared_source_ordinal_logical_locator_chunk_index_v1",
            "tokenizer": "smoke",
        },
        "layout": {"shard_capacity": 1},
        "resources": {"memory_limit": "128MB", "worker_count": 1},
        "output": {"artifact_path": output},
        "execution": {"resume": False, "overwrite": False},
        "reporting": {"progress": False},
    }
    path = root / "intent.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_v2_build_validates_and_exposes_verified_rows(tmp_path: Path) -> None:
    intent = load_corpus_build_intent(_config(tmp_path))
    report = build_corpus_artifact_v2(intent)

    assert report["status"] == "pass"
    artifact = tmp_path / "artifact"
    result = validate_corpus_artifact_v2(artifact)
    assert result.ok
    rows = list(open_verified_corpus(artifact))
    assert [row["example_id"] for row in rows] == [
        "corpus_000000001",
        "corpus_000000002",
    ]
    assert inspect_corpus_artifact_v2(artifact).shard_count == 2


def test_v2_semantic_identity_is_relocation_invariant(tmp_path: Path) -> None:
    first = load_corpus_build_intent(_config(tmp_path / "one"))
    first_report = build_corpus_artifact_v2(first)
    second_root = tmp_path / "two"
    second = load_corpus_build_intent(_config(second_root))
    second_report = build_corpus_artifact_v2(second)

    assert first_report["semantic_identity"] == second_report["semantic_identity"]
    assert (
        first_report["semantic_identity"]
        == inspect_corpus_artifact_v2(tmp_path / "one" / "artifact").semantic_identity
    )


def test_shard_capacity_changes_physical_layout_not_identity(tmp_path: Path) -> None:
    first = load_corpus_build_intent(_config(tmp_path / "one"))
    first_report = build_corpus_artifact_v2(first)
    second_config = _config(tmp_path / "two")
    payload = json.loads(second_config.read_text())
    payload["layout"]["shard_capacity"] = 2
    second_config.write_text(json.dumps(payload))
    second_report = build_corpus_artifact_v2(load_corpus_build_intent(second_config))

    assert first_report["semantic_identity"] == second_report["semantic_identity"]
    assert second_report["shard_count"] == 1


@pytest.mark.parametrize(
    "bad_line",
    [
        '{"text":"a", "text": "b"}',
        '{"text": 3}',
        "not-json",
    ],
)
def test_jsonl_source_rejects_malformed_or_non_string_records(
    tmp_path: Path, bad_line: str
) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text(bad_line + "\n", encoding="utf-8")
    payload = json.loads(_config(tmp_path).read_text())
    payload["sources"] = [
        {
            "source_id": "rows",
            "adapter": "local_jsonl_text_v1",
            "path": "rows.jsonl",
        }
    ]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError):
        build_corpus_artifact_v2(load_corpus_build_intent(path))


def test_corrupt_shard_fails_before_yielding_that_shard(tmp_path: Path) -> None:
    build_corpus_artifact_v2(load_corpus_build_intent(_config(tmp_path)))
    shard = next((tmp_path / "artifact" / "shards").glob("*.jsonl"))
    shard.write_text(shard.read_text() + "{}\n", encoding="utf-8")

    result = validate_corpus_artifact_v2(tmp_path / "artifact")
    assert not result.ok
    with pytest.raises(ValueError):
        list(open_verified_corpus(tmp_path / "artifact"))


def test_v2_production_adapter_checks_identity_and_ignores_source_path(
    tmp_path: Path,
) -> None:
    report = build_corpus_artifact_v2(load_corpus_build_intent(_config(tmp_path)))
    artifact = tmp_path / "artifact"
    moved = tmp_path / "moved"
    shutil.copytree(artifact, moved)
    reference = resolve_corpus_input(
        moved,
        expected_semantic_identity=str(report["semantic_identity"]),
    )
    assert reference.kind == "v2"
    with pytest.raises(ValueError, match="semantic identity"):
        resolve_corpus_input(moved, expected_semantic_identity="sha256:" + "0" * 64)


def test_preflight_rejects_resume_and_overwrite_without_writing(tmp_path: Path) -> None:
    path = _config(tmp_path)
    payload = json.loads(path.read_text())
    payload["execution"] = {"resume": True, "overwrite": True}
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="mutually exclusive"):
        load_corpus_build_intent(path)
    assert not (tmp_path / "artifact").exists()


def test_v2_authority_projection_excludes_paths() -> None:
    from radjax_tome.corpora.config import selection_authority_payload_v2

    payload = selection_authority_payload_v2(
        corpus_semantic_identity="sha256:" + "1" * 64,
        tokenizer_binding_digest="sha256:" + "2" * 64,
        policy={"ordering": "stable"},
        max_examples=4,
    )
    encoded = json.dumps(payload)
    assert "artifact_path" not in encoded
    assert "dataset_path" not in encoded
    assert "shard" not in encoded


def test_v2_duplicate_provenance_is_retained(tmp_path: Path) -> None:
    intent_path = _config(tmp_path)
    (tmp_path / "sources" / "b.txt").write_text("duplicate", encoding="utf-8")
    report = build_corpus_artifact_v2(load_corpus_build_intent(intent_path))
    rows = list(open_verified_corpus(tmp_path / "artifact"))
    assert report["num_examples"] == 1
    assert rows[0]["duplicate_provenance"] == ["text:b.txt:0"]


def test_v2_interruption_can_resume_owned_staging(tmp_path: Path) -> None:
    intent_path = _config(tmp_path)
    payload = json.loads(intent_path.read_text())
    initial_intent = load_corpus_build_intent(intent_path)
    with pytest.raises(InterruptedError):
        build_corpus_artifact_v2(initial_intent, fault_after="validated")
    assert not (tmp_path / "artifact").exists()
    payload["execution"]["resume"] = True
    intent_path.write_text(json.dumps(payload))
    resumed = build_corpus_artifact_v2(load_corpus_build_intent(intent_path))
    assert resumed["status"] == "resumed"
    assert validate_corpus_artifact_v2(tmp_path / "artifact").ok


def test_v2_binding_tampering_is_rejected(tmp_path: Path) -> None:
    build_corpus_artifact_v2(load_corpus_build_intent(_config(tmp_path)))
    binding = tmp_path / "artifact" / "language_tokenizer_binding_v1.json"
    payload = json.loads(binding.read_text())
    payload["canonical_binding_digest"] = "sha256:" + "0" * 64
    binding.write_text(json.dumps(payload))
    assert not validate_corpus_artifact_v2(tmp_path / "artifact").ok


def test_recover_absolute_destination_after_quarantine(tmp_path: Path) -> None:
    from radjax_tome.corpora.lifecycle import CorpusJournal, recover_publication

    parent = tmp_path / "publish"
    parent.mkdir()
    destination = parent / "artifact"
    destination.mkdir()
    (destination / ".radjax_corpus_staging").write_text("owned\n")
    sibling = parent / "unrelated"
    sibling.mkdir()
    quarantine = parent / ".artifact.quarantine-test"
    destination.rename(quarantine)
    journal_path = tmp_path / "journal.jsonl"
    journal = CorpusJournal(journal_path, "tx", "cfg")
    journal.append(
        "OLD_QUARANTINED", destination=str(destination), quarantine=quarantine.name
    )
    assert recover_publication(journal_path, parent) == "restored"
    assert destination.is_dir() and not quarantine.exists() and sibling.is_dir()


def test_recover_rejects_destination_outside_parent(tmp_path: Path) -> None:
    from radjax_tome.corpora.lifecycle import CorpusJournal, recover_publication

    parent = tmp_path / "publish"
    parent.mkdir()
    journal_path = tmp_path / "journal.jsonl"
    journal = CorpusJournal(journal_path, "tx", "cfg")
    journal.append(
        "OLD_QUARANTINED",
        destination=str(tmp_path / "elsewhere" / "artifact"),
        quarantine=".artifact.q",
    )
    assert recover_publication(journal_path, parent) == "no_safe_action"
