from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from radjax_tome.corpora import (
    TokenizerConfig,
    TokenizerMetadata,
    available_tokenizer_backends,
    create_tokenizer,
    normalize_tokenizer_config,
    tokenize_jsonl_corpus,
)

ROOT = Path(__file__).resolve().parents[1]
SUBPROCESS_ENV = {"PYTHONPATH": str(ROOT / "src")}


def test_smoke_tokenizer_round_trips_utf8_bytes() -> None:
    tokenizer = create_tokenizer()

    assert tokenizer.metadata.backend == "smoke"
    assert "smoke" in available_tokenizer_backends()
    assert tokenizer.encode("A") == [66]
    assert tokenizer.decode(tokenizer.encode("hello")) == "hello"


def test_tokenizer_config_normalizes_qwen_alias() -> None:
    config = normalize_tokenizer_config(
        {
            "backend": "qwen",
            "tokenizer_id": "Qwen/Qwen2.5-0.5B",
            "local_files_only": True,
            "use_fast": False,
        }
    )

    assert config.backend == "hf"
    assert config.local_files_only is True
    assert config.use_fast is False


def test_hf_tokenizer_backend_raises_install_hint_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "transformers", None)

    with pytest.raises(RuntimeError, match=r"\.\[teacher-hf\]"):
        create_tokenizer({"backend": "hf", "tokenizer_id": "local/tokenizer"})


def test_hf_tokenizer_backend_can_be_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_tokenizer = _FakeHFTokenizer()
    calls: list[dict[str, Any]] = []

    def _from_pretrained(*_args: object, **kwargs: object) -> _FakeHFTokenizer:
        calls.append(dict(kwargs))
        return fake_tokenizer

    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=_from_pretrained)
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    tokenizer = create_tokenizer(
        TokenizerConfig(
            backend="hf",
            tokenizer_id="local/tokenizer",
            vocab_size=1024,
            eos_token_id=9,
            pad_token_id=8,
            revision="main",
            local_files_only=True,
            use_fast=False,
        )
    )

    assert calls == [
        {
            "revision": "main",
            "trust_remote_code": False,
            "local_files_only": True,
            "use_fast": False,
        }
    ]
    assert tokenizer.metadata == TokenizerMetadata(
        backend="hf",
        tokenizer_id="local/tokenizer",
        vocab_size=1024,
        eos_token_id=9,
        pad_token_id=8,
        revision="main",
        unk_token_id=7,
    )
    assert tokenizer.encode("abc", max_length=2) == [1, 2]
    assert tokenizer.decode([1, 2, 3]) == "decoded:1,2,3"


def test_tokenize_jsonl_corpus_preserves_source_identity(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps({"example_id": "a", "text": "alpha"})
        + "\n"
        + json.dumps({"example_id": "b", "text": "beta"})
        + "\n",
        encoding="utf-8",
    )

    manifest = tokenize_jsonl_corpus(
        corpus,
        tmp_path / "tokens",
        tokenizer=create_tokenizer(),
        sequence_length=8,
    )

    assert manifest.num_examples == 2
    rows = [
        json.loads(line)
        for line in (tmp_path / "tokens" / "tokens-00000.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows[0]["example_id"] == "a"
    assert rows[0]["token_ids"] == [98, 109, 113, 105, 98]
    assert (tmp_path / "tokens" / "manifest.json").is_file()


def test_tokenize_corpus_cli(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps({"example_id": "a", "text": "alpha"}) + "\n",
        encoding="utf-8",
    )
    script = ROOT / "scripts" / "tokenize_corpus.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            str(corpus),
            "--out",
            str(tmp_path / "tokens"),
            "--sequence-length",
            "8",
        ],
        cwd=ROOT,
        env=SUBPROCESS_ENV,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Wrote tokenized corpus" in result.stdout
    assert (
        json.loads((tmp_path / "tokens" / "manifest.json").read_text())["num_examples"]
        == 1
    )


class _FakeHFTokenizer:
    vocab_size = 321
    eos_token_id = 0
    pad_token_id = None
    unk_token_id = 7
    eos_token = "<eos>"
    pad_token = None

    def encode(self, text: str, **kwargs: object) -> list[int]:
        token_ids = [index + 1 for index, _char in enumerate(text)]
        max_length = kwargs.get("max_length")
        if max_length is not None:
            token_ids = token_ids[: int(max_length)]
        return token_ids

    def decode(self, token_ids: list[int], **_kwargs: object) -> str:
        return "decoded:" + ",".join(str(token_id) for token_id in token_ids)
