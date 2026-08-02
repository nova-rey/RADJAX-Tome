from __future__ import annotations

import hashlib
import json
import os
import tomllib
from pathlib import Path

import pytest

from radjax_tome.corpora import (
    SmokeTokenizer,
    capture_language_tokenizer_binding,
    tokenize_jsonl_corpus,
)

ROOT = Path(__file__).resolve().parents[1]
MIRROR = ROOT / "contracts/radjax_tome/student_consumption/v5"
PINNED_COMMIT = "cac3dd21e0d56df5a9e6fd50b20267e0b8960995"


def _entries(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_t1_v5_mirror_is_checksum_pinned_to_contract_v070() -> None:
    sums = {
        relative: digest
        for digest, relative in (
            line.split("  ", maxsplit=1)
            for line in (MIRROR / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
        )
    }
    observed = _entries(MIRROR)
    observed.pop("SHA256SUMS")
    assert observed == sums
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert any(PINNED_COMMIT in item for item in project["project"]["dependencies"])


def test_t1_v5_source_assets_match_offline_mirror_when_configured() -> None:
    configured = os.environ.get("RADJAX_CONTRACT_STUDENT_CONSUMPTION_V5_ROOT")
    if configured is not None:
        assert _entries(Path(configured)) == _entries(MIRROR)


def test_t1_smoke_capture_has_complete_deterministic_vocabulary(tmp_path: Path) -> None:
    captured = capture_language_tokenizer_binding(SmokeTokenizer(vocab_size=512))
    binding = captured.binding

    assert binding["profile_id"] == "native_v3_student_v5"
    assert binding["tokenizer"]["revision"]["kind"] == "content_digest"
    assert binding["vocabulary"]["token_domain"] == {"start": 0, "end_exclusive": 512}
    assert binding["vocabulary"]["reserved_token_ids"] == [0]
    rows = [json.loads(line) for line in captured.vocabulary_jsonl.splitlines()]
    assert [row["token_id"] for row in rows] == list(range(512))
    captured.write_to(tmp_path)
    assert (tmp_path / "language_tokenizer_binding_v1.json").is_file()
    assert (tmp_path / "resources/tokenizer_vocabulary.jsonl").read_bytes() == (
        captured.vocabulary_jsonl
    )


def test_t1_tokenization_captures_the_instantiated_encoder_or_fails_closed(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"example_id":"a","text":"alpha"}\n', encoding="utf-8")

    manifest = tokenize_jsonl_corpus(
        corpus,
        tmp_path / "tokens",
        tokenizer=SmokeTokenizer(vocab_size=512),
        sequence_length=8,
    )

    assert manifest.num_tokens == 5
    assert (tmp_path / "tokens/language_tokenizer_binding_v1.json").is_file()

    class UnsupportedTokenizer:
        metadata = SmokeTokenizer(vocab_size=512).metadata

        def encode(self, text: str, *, max_length: int | None = None) -> list[int]:
            return [1]

        def decode(self, token_ids: list[int] | tuple[int, ...]) -> str:
            return "x"

    with pytest.raises(ValueError, match="supports only instantiated"):
        tokenize_jsonl_corpus(
            corpus,
            tmp_path / "unsupported",
            tokenizer=UnsupportedTokenizer(),  # type: ignore[arg-type]
            sequence_length=8,
        )
