"""External exact deduplication and canonical ordering for corpus v2."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path

from radjax_tome.corpora.records import CanonicalCorpusRecord, SourceRecord


def deduplicate_records(
    records: Iterable[SourceRecord],
    *,
    enabled: bool = True,
    database_path: Path | None = None,
) -> tuple[Iterator[CanonicalCorpusRecord], dict[str, int]]:
    """Spill source rows to private DuckDB and yield winners in stable order."""

    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover
        raise ValueError("M10 corpus construction requires DuckDB 1.4.5") from exc
    if database_path is None:
        fd, name = tempfile.mkstemp(prefix="radjax-corpus-", suffix=".duckdb")
        os.close(fd)
        os.unlink(name)
        database_path = Path(name)
    connection = duckdb.connect(str(database_path))
    connection.execute(
        """CREATE TABLE rows (
            source_id VARCHAR, source_ordinal INTEGER, logical_locator VARCHAR,
            chunk_index INTEGER, chunk_count INTEGER, text VARCHAR,
            text_digest VARCHAR, source_digest VARCHAR, declared_record_id VARCHAR,
            text_bytes BLOB
        )"""
    )
    total = 0
    for record in records:
        connection.execute(
            "INSERT INTO rows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                record.source_id,
                record.source_ordinal,
                record.logical_locator,
                record.chunk_index,
                record.chunk_count,
                record.text,
                record.normalized_text_digest,
                record.source_digest,
                record.declared_record_id,
                record.text.encode("utf-8"),
            ],
        )
        total += 1

    collisions = connection.execute(
        "SELECT text_digest FROM rows GROUP BY text_digest "
        "HAVING COUNT(DISTINCT text_bytes) > 1"
    ).fetchall()
    if collisions:
        connection.close()
        raise ValueError(
            "CORPUS_HASH_COLLISION: digest maps to different normalized bytes"
        )
    if enabled:
        winner_sql = """
            SELECT r.*,
              (SELECT COUNT(*) FROM rows d WHERE d.text_digest = r.text_digest
               AND d.text_bytes = r.text_bytes) AS duplicate_count
            FROM rows r
            WHERE NOT EXISTS (
              SELECT 1 FROM rows earlier
              WHERE earlier.text_digest = r.text_digest
                AND earlier.text_bytes = r.text_bytes
                AND (earlier.source_ordinal, earlier.logical_locator,
                     earlier.chunk_index)
                    < (r.source_ordinal, r.logical_locator, r.chunk_index)
            )
            ORDER BY r.source_ordinal, r.logical_locator, r.chunk_index
        """
    else:
        winner_sql = """
            SELECT r.*, 1 AS duplicate_count FROM rows r
            ORDER BY r.source_ordinal, r.logical_locator, r.chunk_index
        """

    def output() -> Iterator[CanonicalCorpusRecord]:
        index = 0
        try:
            result = connection.execute(winner_sql)
            while batch := result.fetchmany(256):
                for row in batch:
                    (
                        source_id,
                        source_ordinal,
                        locator,
                        chunk_index,
                        chunk_count,
                        text,
                        digest,
                        source_digest,
                        record_id,
                        _text_bytes,
                        duplicate_count,
                    ) = row
                    provenance = ()
                    if enabled and int(duplicate_count) > 1:
                        matches = connection.execute(
                            "SELECT source_id || ':' || logical_locator || ':' || "
                            "CAST(chunk_index AS VARCHAR) FROM rows WHERE "
                            "text_digest = ? AND text_bytes = ? ORDER BY "
                            "source_ordinal, logical_locator, chunk_index",
                            [digest, _text_bytes],
                        ).fetchall()
                        provenance = tuple(str(match[0]) for match in matches[1:])
                    index += 1
                    yield CanonicalCorpusRecord(
                        example_id=f"corpus_{index:09d}",
                        source_id=str(source_id),
                        source_ordinal=int(source_ordinal),
                        logical_locator=str(locator),
                        chunk_index=int(chunk_index),
                        chunk_count=int(chunk_count),
                        text=str(text),
                        text_digest=str(digest),
                        source_digest=str(source_digest),
                        declared_record_id=(
                            None if record_id is None else str(record_id)
                        ),
                        duplicate_provenance=provenance,
                    )
        finally:
            connection.close()

    winners = int(
        connection.execute(f"SELECT COUNT(*) FROM ({winner_sql})").fetchone()[0]
    )
    return output(), {
        "input_records": total,
        "output_records": winners,
        "duplicates_removed": total - winners,
    }


__all__ = ["deduplicate_records"]
