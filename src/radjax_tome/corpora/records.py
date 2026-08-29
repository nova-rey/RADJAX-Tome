"""Immutable records shared by corpus ingestion, storage, and production."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    source_ordinal: int
    logical_locator: str
    chunk_index: int
    chunk_count: int
    text: str
    normalized_text_digest: str
    source_digest: str
    declared_record_id: str | None = None
    duplicate_of: str | None = None


@dataclass(frozen=True)
class CanonicalCorpusRecord:
    example_id: str
    source_id: str
    source_ordinal: int
    logical_locator: str
    chunk_index: int
    chunk_count: int
    text: str
    text_digest: str
    source_digest: str
    declared_record_id: str | None = None
    duplicate_provenance: tuple[str, ...] = ()
    duplicate_count: int = 1
    identity_digest: str | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "example_id": self.example_id,
            "source_id": self.source_id,
            "source_ordinal": self.source_ordinal,
            "logical_locator": self.logical_locator,
            "chunk_index": self.chunk_index,
            "chunk_count": self.chunk_count,
            "text": self.text,
            "text_digest": self.text_digest,
            "source_digest": self.source_digest,
        }
        if self.identity_digest is not None:
            result["example_identity"] = self.identity_digest
        if self.declared_record_id is not None:
            result["declared_record_id"] = self.declared_record_id
        if self.duplicate_provenance:
            result["duplicate_provenance"] = list(self.duplicate_provenance)
        if self.duplicate_count > 1:
            result["duplicate_count"] = self.duplicate_count
            result["duplicate_provenance_truncated"] = len(
                self.duplicate_provenance
            ) < self.duplicate_count - 1
        return result


__all__ = ["CanonicalCorpusRecord", "SourceRecord"]
