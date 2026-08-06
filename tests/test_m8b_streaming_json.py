"""Grammar and transaction checks for the bounded M8B.2 staging writer."""

from __future__ import annotations

import hashlib
import json

import pytest

from radjax_tome.builder.delivery import staging
from radjax_tome.builder.delivery.streaming_json import (
    CanonicalJSONObjectError,
    _ObjectTerminus,
    stream_canonical_object_with_hash,
)


def _consume(chunks: list[str]) -> tuple[str, str]:
    emitted: list[str] = []
    closure: list[str] = []
    parser = _ObjectTerminus()
    for chunk in chunks:
        parser.consume(chunk, emitted.append, closure.append)
    parser.finish()
    return "".join(emitted), "".join(closure)


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"a": 1},
        {"a": {"nested": ["", "brace }", 'quote \\"', "\\u2603"]}},
    ],
)
def test_outer_object_grammar_accepts_every_character_split(
    value: dict[str, object],
) -> None:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    for point in range(len(encoded) + 1):
        prefix, closure = _consume([encoded[:point], encoded[point:]])
        assert prefix + closure == encoded


def test_outer_object_grammar_handles_one_character_chunks_and_trailing_space() -> None:
    encoded = '{"a":"\\\\\\"\\u2603}"}'
    prefix, closure = _consume([*encoded, " \n\t"])
    assert prefix + closure == encoded


@pytest.mark.parametrize(
    "encoded", ["[]", '"x"', "0", "true", "null", "{", "{}{}", "{} x"]
)
def test_outer_object_grammar_rejects_non_object_or_invalid_termination(
    encoded: str,
) -> None:
    with pytest.raises(CanonicalJSONObjectError):
        _consume([encoded])


def test_streams_once_and_hashes_exact_canonical_body() -> None:
    payload = {"z": [1, 2], "a": {"text": "brace } and \\u2603"}}
    chunks: list[str] = []
    actual = stream_canonical_object_with_hash(payload, write=chunks.append)
    expected = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    assert "".join(chunks) == expected[:-1]
    assert actual == hashlib.sha256(expected.encode("utf-8")).hexdigest()


def test_streaming_staging_preserves_existing_target_on_replace_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "selected-exemplars-00000.json"
    path.write_bytes(b"prior-valid")
    original_replace = staging.os.replace

    def fail_replace(source, target) -> None:
        if source.name.startswith(".selected-exemplars"):
            raise OSError("injected replacement failure")
        original_replace(source, target)

    monkeypatch.setattr(staging.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replacement"):
        staging._write_native_payload_shard_streaming(
            path, {"schema_version": "selected_exemplar_payload_shard_v1"}
        )
    assert path.read_bytes() == b"prior-valid"
    assert not list(tmp_path.glob(".selected-exemplars-*.tmp"))


def test_prepare_cleans_orphaned_private_temporary(tmp_path) -> None:
    stage = tmp_path / ".staging-native-c6" / "authority"
    stage.mkdir(parents=True)
    temporary = stage / ".selected-exemplars-00000.json.tmp"
    temporary.write_text("partial", encoding="utf-8")
    config = type(
        "Config",
        (),
        {
            "artifact_dir": tmp_path,
            "delivery_authority_hash": "authority",
            "rerun_metrics": {},
        },
    )()
    staging._prepare_native_payload_staging(config, selected_records=[])
    assert not temporary.exists()
