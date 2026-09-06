from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from radjax_tome.builder.status import read_corpus_journal
from radjax_tome.corpora.config import (
    apply_corpus_operational_overrides,
    load_corpus_build_intent,
    parse_corpus_build_intent_document,
    serialize_corpus_build_intent,
)
from radjax_tome.corpora.feasibility import assess_corpus_feasibility
from radjax_tome.corpora.lifecycle import CorpusJournal
from radjax_tome.io.config_export import save_as_config


def _corpus_config(root: Path) -> Path:
    source = root / "sources"
    source.mkdir(parents=True)
    (source / "one.txt").write_text("one\ntwo\n", encoding="utf-8")
    path = root / "corpus.json"
    path.write_text(
        json.dumps(
            {
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
                    "ordering": (
                        "declared_source_ordinal_logical_locator_chunk_index_v1"
                    ),
                    "tokenizer": "smoke",
                },
                "layout": {"shard_capacity": 2},
                "resources": {"memory_limit": "64MB", "worker_count": 1},
                "output": {"artifact_path": "artifact"},
                "execution": {"resume": False, "overwrite": False},
                "reporting": {"progress": False},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_corpus_config_round_trip_and_operational_override(tmp_path: Path) -> None:
    intent = load_corpus_build_intent(_corpus_config(tmp_path))
    document = json.loads(serialize_corpus_build_intent(intent))
    reloaded = parse_corpus_build_intent_document(
        document, source_path=tmp_path / "roundtrip.json"
    )

    assert serialize_corpus_build_intent(reloaded) == serialize_corpus_build_intent(
        intent
    )
    assert apply_corpus_operational_overrides(intent, resume=True).resume
    with pytest.raises(ValueError, match="mutually exclusive"):
        apply_corpus_operational_overrides(intent, resume=True, overwrite=True)


def test_save_as_reparses_and_never_clobbers(tmp_path: Path) -> None:
    existing = tmp_path / "existing.json"
    existing.write_text("sentinel\n", encoding="utf-8")
    destination = tmp_path / "new.json"
    text = '{"schema_version": "example"}\n'

    saved = save_as_config(destination, text, parse=json.loads)

    assert saved == destination.resolve()
    assert destination.read_text(encoding="utf-8") == text
    with pytest.raises(FileExistsError):
        save_as_config(existing, text, parse=json.loads)
    assert existing.read_text(encoding="utf-8") == "sentinel\n"
    assert not list(tmp_path.glob(".*.m11-*.tmp"))


def test_feasibility_parses_jsonl_without_staging(tmp_path: Path) -> None:
    config = _corpus_config(tmp_path)
    payload = json.loads(config.read_text(encoding="utf-8"))
    source = tmp_path / "rows.jsonl"
    source.write_text("not-json\n", encoding="utf-8")
    payload["sources"] = [
        {
            "source_id": "rows",
            "adapter": "local_jsonl_text_v1",
            "path": "rows.jsonl",
        }
    ]
    config.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid at line 1"):
        assess_corpus_feasibility(load_corpus_build_intent(config))
    assert not (tmp_path / "artifact").exists()


def test_journal_status_reader_verifies_hash_chain(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.jsonl"
    journal = CorpusJournal(journal_path, "tx", "config")
    journal.append("START")
    journal.append("COMPLETE", count=2)

    report = read_corpus_journal(journal_path)

    assert report["status"] == "present"
    assert report["event_count"] == 2
    assert report["latest"]["event_type"] == "COMPLETE"

    journal_path.write_text(
        journal_path.read_text(encoding="utf-8").replace('"count":2', '"count":3'),
        encoding="utf-8",
    )
    assert read_corpus_journal(journal_path)["status"] == "invalid"


def test_base_import_does_not_load_optional_textual() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, radjax_tome; assert 'textual' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
