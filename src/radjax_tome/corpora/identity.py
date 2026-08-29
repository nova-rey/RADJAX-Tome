"""Path-independent semantic identity primitives for corpus v2."""

from __future__ import annotations

import hashlib
from typing import Any

from radjax_tome.corpora.config import canonical_bytes, sha256


def normalized_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def text_digest(text: str) -> str:
    return sha256(text.encode("utf-8"))


def example_identity(
    *, source_id: str, logical_locator: str, chunk_index: int, text_digest: str
) -> str:
    return sha256(
        canonical_bytes(
            {
                "source_id": source_id,
                "logical_locator": logical_locator,
                "chunk_index": chunk_index,
                "text_digest": text_digest,
            }
        )
    )


def corpus_semantic_identity(
    *,
    policy: dict[str, Any],
    tokenizer_binding_digest: str,
    records: Any,
    source_declarations: list[dict[str, Any]] | None = None,
) -> str:
    digest = hashlib.sha256()
    digest.update(
        canonical_bytes(
            {
                "schema_version": "radjax_tome_corpus_semantic_identity_v2",
                "policy": policy,
                "tokenizer_binding_digest": tokenizer_binding_digest,
                "sources": source_declarations or [],
                "records": "stream_v1",
            }
        )
    )
    for record in records:
        digest.update(
            canonical_bytes(
                {
                    "example_id": record.example_id,
                    "source_id": record.source_id,
                    "logical_locator": record.logical_locator,
                    "chunk_index": record.chunk_index,
                    "text_digest": record.text_digest,
                }
            )
        )
        digest.update(b"\n")
    return "sha256:" + digest.hexdigest()


def digest_stream(chunks: Any) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return "sha256:" + digest.hexdigest()


__all__ = [
    "corpus_semantic_identity",
    "digest_stream",
    "example_identity",
    "normalized_text",
    "text_digest",
]
