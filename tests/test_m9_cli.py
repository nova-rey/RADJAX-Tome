import json

import pytest

from radjax_tome.cli.m9 import (
    classify_destination,
    load_build_intent,
    preflight,
    status,
)


def _intent(tmp_path):
    payload = {
        "schema_version": "radjax_tome_build_intent_v1",
        "teacher": {
            "model": "m",
            "model_provenance_path": str(tmp_path / "prov.json"),
            "backend": "cpu_reference",
            "runtime_mode": "cpu",
        },
        "corpus": {
            "dataset_path": str(tmp_path / "data.jsonl"),
            "corpus_manifest_path": str(tmp_path / "manifest.json"),
            "max_examples": 1,
        },
        "outputs": {"output_dir": str(tmp_path / "out")},
    }
    path = tmp_path / "intent.json"
    path.write_text(json.dumps(payload))
    return path


def test_m9_loads_canonical_intent_and_preflight_is_mutation_free(tmp_path):
    intent = load_build_intent(_intent(tmp_path))
    out = intent.outputs.output_dir
    assert classify_destination(out) == "missing"
    assert preflight(intent)["status"] == "ready"
    assert not out.exists()


def test_m9_rejects_unknown_fields(tmp_path):
    path = _intent(tmp_path)
    payload = json.loads(path.read_text())
    payload["unexpected"] = 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unknown build intent"):
        load_build_intent(path)


def test_m9_rejects_nonempty_destination_without_override(tmp_path):
    intent = load_build_intent(_intent(tmp_path))
    intent.outputs.output_dir.mkdir()
    (intent.outputs.output_dir / "keep").write_text("x")
    with pytest.raises(ValueError, match="nonempty directory"):
        preflight(intent)
    assert (intent.outputs.output_dir / "keep").read_text() == "x"


def test_m9_status_missing_is_new(tmp_path):
    assert status(tmp_path / "missing")["status"] == "new"
