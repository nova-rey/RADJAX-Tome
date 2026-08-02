"""Fail-closed capture of the Contract v5 tokenizer binding inputs.

This module deliberately knows only the concrete tokenizer adapters that
produce Tome token IDs.  It never reconstructs a vocabulary from metadata or
guesses an immutable revision after tokenization has happened.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from radjax_tome.corpora.tokenizer import HFTokenizer, SmokeTokenizer, Tokenizer

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_IMMUTABLE_RELEASE = re.compile(r"v?\d+(?:\.\d+)+(?:[-+][0-9A-Za-z.-]+)?\Z")
_BINDING_FILENAME = "language_tokenizer_binding_v1.json"
_VOCABULARY_RESOURCE = "resources/tokenizer_vocabulary.jsonl"


@dataclass(frozen=True)
class CapturedLanguageTokenizerBinding:
    """Binding payload and its canonical vocabulary resource bytes."""

    binding: dict[str, Any]
    vocabulary_jsonl: bytes

    def write_to(self, root: Path) -> Path:
        """Write the capture beside the token IDs it was derived from."""

        resource = root / _VOCABULARY_RESOURCE
        resource.parent.mkdir(parents=True, exist_ok=True)
        resource.write_bytes(self.vocabulary_jsonl)
        path = root / _BINDING_FILENAME
        path.write_bytes(canonical_json_bytes(self.binding) + b"\n")
        return path


def capture_language_tokenizer_binding(
    tokenizer: Tokenizer,
) -> CapturedLanguageTokenizerBinding:
    """Capture only immutable, complete evidence from an instantiated adapter.

    The adapter is the same object used by the caller's encode loop.  Metadata
    alone is intentionally insufficient: vocabulary, normalization, and
    immutable revision evidence must all be available at capture time.
    """

    if isinstance(tokenizer, SmokeTokenizer):
        return _capture_smoke(tokenizer)
    if isinstance(tokenizer, HFTokenizer):
        return _capture_hf(tokenizer)
    raise ValueError(
        "language/tokenizer binding capture supports only instantiated "
        "SmokeTokenizer and HFTokenizer adapters"
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Match Contract v5's canonical JSON encoding without importing Contract."""

    _reject_negative_zero(value)
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _capture_smoke(tokenizer: SmokeTokenizer) -> CapturedLanguageTokenizerBinding:
    records = [(0, ""), *((token + 1, chr(token)) for token in range(256))]
    records.extend(
        (token, f"<unused_{token}>") for token in range(257, tokenizer.vocab_size)
    )
    return _build_capture(
        tokenizer={
            "family": "radjax_smoke_byte",
            "implementation_id": "radjax_tome.corpora.tokenizer.SmokeTokenizer",
            "revision": _content_revision(
                {
                    "implementation": "radjax_smoke_byte_v1",
                    "vocabulary_size": tokenizer.vocab_size,
                    "eos_token_id": tokenizer.eos_token_id,
                    "pad_token_id": tokenizer.pad_token_id,
                }
            ),
            "configuration_identity": _digest_json(
                {
                    "vocabulary_size": tokenizer.vocab_size,
                    "eos_token_id": tokenizer.eos_token_id,
                    "pad_token_id": tokenizer.pad_token_id,
                }
            ),
            "normalization_identity": _digest_json(
                {"normalization": "utf8_bytes_to_latin1_codepoints_v1"}
            ),
        },
        records=records,
        added_tokens=[],
        reserved_token_ids=[0],
        special_tokens=[
            {"name": "eos", "token_id": tokenizer.eos_token_id},
            {"name": "pad", "token_id": tokenizer.pad_token_id},
        ],
    )


def _capture_hf(tokenizer: HFTokenizer) -> CapturedLanguageTokenizerBinding:
    raw = tokenizer._tokenizer
    metadata = tokenizer.metadata
    revision = _immutable_revision(metadata.revision)
    if not hasattr(raw, "get_vocab") or not callable(raw.get_vocab):
        raise ValueError("HF tokenizer has no complete get_vocab() evidence")
    if not hasattr(raw, "backend_tokenizer"):
        raise ValueError("HF tokenizer has no fast normalization evidence")
    vocabulary = raw.get_vocab()
    if not isinstance(vocabulary, dict) or not vocabulary:
        raise ValueError(
            "HF tokenizer get_vocab() did not return a complete vocabulary"
        )
    by_id: dict[int, str] = {}
    for token, token_id in vocabulary.items():
        if (
            not isinstance(token, str)
            or not isinstance(token_id, int)
            or isinstance(token_id, bool)
        ):
            raise ValueError(
                "HF tokenizer vocabulary contains a non-string token or ID"
            )
        if token_id in by_id:
            raise ValueError("HF tokenizer vocabulary maps multiple tokens to one ID")
        by_id[token_id] = token
    if list(sorted(by_id)) != list(range(len(by_id))):
        raise ValueError(
            "HF tokenizer vocabulary does not cover a contiguous token domain"
        )
    if len(raw) != len(by_id):
        raise ValueError(
            "HF tokenizer vocabulary evidence disagrees with tokenizer length"
        )
    backend = raw.backend_tokenizer
    normalizer = getattr(backend, "normalizer", None)
    if normalizer is None:
        raise ValueError("HF tokenizer has no normalization identity evidence")
    init_kwargs = getattr(raw, "init_kwargs", None)
    if not isinstance(init_kwargs, dict):
        raise ValueError("HF tokenizer has no configuration identity evidence")
    records = [(token_id, by_id[token_id]) for token_id in range(len(by_id))]
    added = _hf_added_tokens(raw, by_id)
    special = _hf_special_tokens(raw, by_id)
    return _build_capture(
        tokenizer={
            "family": "huggingface",
            "implementation_id": f"{type(raw).__module__}.{type(raw).__qualname__}",
            "revision": revision,
            "configuration_identity": _digest_json(init_kwargs),
            "normalization_identity": _digest_json(
                {"backend_normalizer": str(normalizer)}
            ),
        },
        records=records,
        added_tokens=added,
        reserved_token_ids=[],
        special_tokens=special,
    )


def _hf_added_tokens(raw: Any, by_id: dict[int, str]) -> list[dict[str, Any]]:
    getter = getattr(raw, "get_added_vocab", None)
    if getter is None or not callable(getter):
        raise ValueError("HF tokenizer has no added-token declaration evidence")
    added = getter()
    if not isinstance(added, dict):
        raise ValueError("HF tokenizer added-token evidence is invalid")
    entries: list[dict[str, Any]] = []
    for token, token_id in added.items():
        if by_id.get(token_id) != token:
            raise ValueError(
                "HF tokenizer added-token evidence disagrees with vocabulary"
            )
        entries.append(_token_record(token_id, token))
    return sorted(entries, key=lambda entry: entry["token_id"])


def _hf_special_tokens(raw: Any, by_id: dict[int, str]) -> list[dict[str, Any]]:
    mapping = getattr(raw, "special_tokens_map", None)
    if not isinstance(mapping, dict):
        raise ValueError("HF tokenizer has no special-token declaration evidence")
    entries: list[dict[str, Any]] = []
    for name, value in mapping.items():
        values = value if isinstance(value, list) else [value]
        for index, token in enumerate(values):
            if not isinstance(token, str):
                raise ValueError("HF tokenizer special-token declaration is invalid")
            token_id = next((key for key, item in by_id.items() if item == token), None)
            if token_id is None:
                raise ValueError("HF tokenizer special token is absent from vocabulary")
            entry_name = str(name) if len(values) == 1 else f"{name}_{index}"
            entries.append({"name": entry_name, "token_id": token_id})
    if len({entry["name"] for entry in entries}) != len(entries):
        raise ValueError("HF tokenizer special-token names are ambiguous")
    return sorted(entries, key=lambda entry: entry["name"])


def _build_capture(
    *,
    tokenizer: dict[str, Any],
    records: list[tuple[int, str]],
    added_tokens: list[dict[str, Any]],
    reserved_token_ids: list[int],
    special_tokens: list[dict[str, Any]],
) -> CapturedLanguageTokenizerBinding:
    if [token_id for token_id, _ in records] != list(range(len(records))):
        raise ValueError("tokenizer vocabulary must have an exact [0, size) domain")
    vocabulary = b"".join(
        canonical_json_bytes(_token_record(token_id, token)) + b"\n"
        for token_id, token in records
    )
    digest = _sha256(vocabulary)
    inventory = [
        {
            "resource_id": "tokenizer_vocabulary",
            "role": "vocabulary",
            "content_digest": digest,
            "inventory_binding": _VOCABULARY_RESOURCE,
            "raw_sha256": digest,
            "raw_size_bytes": len(vocabulary),
        }
    ]
    binding: dict[str, Any] = {
        "schema_version": "radjax_language_tokenizer_binding_v1",
        "profile_id": "native_v3_student_v5",
        "tokenizer": tokenizer,
        "behavior_content_inventory": inventory,
        "canonical_inventory_digest": _sha256(
            canonical_json_bytes(
                [
                    {
                        "resource_id": row["resource_id"],
                        "role": row["role"],
                        "content_digest": row["content_digest"],
                    }
                    for row in inventory
                ]
            )
        ),
        "vocabulary": {
            "resource_id": "tokenizer_vocabulary",
            "vocabulary_map_digest": digest,
            "vocabulary_identity": digest,
            "vocabulary_size": len(records),
            "token_domain": {"start": 0, "end_exclusive": len(records)},
            "added_tokens": added_tokens,
            "reserved_token_ids": reserved_token_ids,
            "special_tokens": special_tokens,
        },
    }
    binding["canonical_binding_digest"] = _sha256(
        canonical_json_bytes(
            {
                "tokenizer": binding["tokenizer"],
                "canonical_inventory_digest": binding["canonical_inventory_digest"],
                "vocabulary": binding["vocabulary"],
            }
        )
    )
    return CapturedLanguageTokenizerBinding(binding, vocabulary)


def _token_record(token_id: int, token: str) -> dict[str, Any]:
    return {
        "token_id": token_id,
        "token_utf8_b64": base64.b64encode(token.encode("utf-8")).decode("ascii"),
    }


def _immutable_revision(value: str | None) -> dict[str, str]:
    if value is None:
        raise ValueError("HF tokenizer revision must be immutable for v5 capture")
    if _SHA256.fullmatch(value):
        return {"kind": "content_digest", "value": value}
    if _GIT_COMMIT.fullmatch(value):
        return {"kind": "git_commit", "value": value}
    if _IMMUTABLE_RELEASE.fullmatch(value):
        return {"kind": "immutable_release", "value": value}
    raise ValueError("HF tokenizer revision is not immutable v5 evidence")


def _content_revision(value: Any) -> dict[str, str]:
    return {"kind": "content_digest", "value": _digest_json(value)}


def _digest_json(value: Any) -> str:
    return _sha256(canonical_json_bytes(value))


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _reject_negative_zero(value: Any) -> None:
    if isinstance(value, float):
        if value == 0 and str(value).startswith("-"):
            raise ValueError("negative zero is not canonical")
    elif isinstance(value, dict):
        for item in value.values():
            _reject_negative_zero(item)
    elif isinstance(value, list):
        for item in value:
            _reject_negative_zero(item)


__all__ = [
    "CapturedLanguageTokenizerBinding",
    "capture_language_tokenizer_binding",
    "canonical_json_bytes",
]
