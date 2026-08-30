"""Strict canonical configuration boundary tests."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from radjax_tome.builder.config import canonical_production_build_intent
from radjax_tome.builder.config_io import load_tome_build_intent


def _payload(tmp_path: Path) -> dict:
    value = asdict(
        canonical_production_build_intent(
            teacher_model="teacher",
            dataset_path=Path("data.jsonl"),
            corpus_manifest_path=Path("manifest.json"),
            teacher_model_provenance_path=Path("provenance.json"),
            output_dir=Path("out"),
        )
    )

    def paths(item: object) -> object:
        if isinstance(item, dict):
            return {key: paths(value) for key, value in item.items()}
        if isinstance(item, Path):
            return str(item)
        return item

    return paths(value)  # type: ignore[return-value]


def test_json_and_yaml_load_same_intent(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    json_path = tmp_path / "intent.json"
    yaml_path = tmp_path / "intent.yaml"
    json_path.write_text(json.dumps(payload))
    yaml_path.write_text(json.dumps(payload))
    assert load_tome_build_intent(json_path) == load_tome_build_intent(yaml_path)
    assert (
        load_tome_build_intent(json_path).outputs.output_dir
        == (tmp_path / "out").resolve()
    )


def test_duplicate_and_unknown_fields_are_rejected(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    path = tmp_path / "intent.json"
    path.write_text(
        '{"schema_version":"radjax_tome_build_intent_v1","schema_version":"x"}'
    )
    with pytest.raises(ValueError, match="duplicate key"):
        load_tome_build_intent(path)
    payload["unexpected"] = 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unknown build intent"):
        load_tome_build_intent(path)


def test_missing_section_is_rejected(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    del payload["selection"]
    path = tmp_path / "intent.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="missing build intent fields"):
        load_tome_build_intent(path)


def test_v2_complete_production_intent_loads(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    payload["schema_version"] = "radjax_tome_build_intent_v2"
    corpus = payload["corpus"]
    corpus["artifact_path"] = str(tmp_path / "corpus")
    corpus["expected_semantic_identity"] = "sha256:" + "1" * 64
    corpus.pop("dataset_path")
    corpus.pop("corpus_manifest_path")
    path = tmp_path / "intent-v2.json"
    path.write_text(json.dumps(payload))
    intent = load_tome_build_intent(path)
    assert intent.schema_version == "radjax_tome_build_intent_v2"
    assert intent.corpus.dataset_path == (tmp_path / "corpus").resolve()
    assert intent.corpus.corpus_manifest_path == (tmp_path / "corpus").resolve()
