"""External exact deduplication and canonical ordering for corpus v2."""

from __future__ import annotations

import json
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
    memory_limit: str | None = None,
    worker_count: int = 1,
    provenance_path: Path | None = None,
) -> tuple[Iterator[CanonicalCorpusRecord], dict[str, int]]:
    """Spill source rows to private DuckDB and yield winners in stable order."""

    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover
        raise ValueError("M10 corpus construction requires DuckDB 1.4.5") from exc
    temporary_database = database_path is None
    if database_path is None:
        fd, name = tempfile.mkstemp(prefix="radjax-corpus-", suffix=".duckdb")
        os.close(fd)
        os.unlink(name)
        database_path = Path(name)
    connection = duckdb.connect(str(database_path))
    if memory_limit is not None:
        connection.execute("SET memory_limit = ?", [memory_limit])
    if worker_count != 1:
        raise ValueError("corpus v2 requires resources.worker_count=1")
    connection.execute("SET threads = 1")
    connection.execute(
        """CREATE TABLE rows (
            source_id VARCHAR, source_ordinal INTEGER, logical_locator VARCHAR,
            chunk_index INTEGER, chunk_count INTEGER, text VARCHAR,
            text_digest VARCHAR, source_digest VARCHAR, declared_record_id VARCHAR,
            text_bytes BLOB
        )"""
    )
    total = 0
    pending: list[list[object]] = []
    for record in records:
        pending.append(
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
        if len(pending) == 512:
            connection.executemany(
                "INSERT INTO rows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", pending
            )
            pending.clear()
    if pending:
        connection.executemany(
            "INSERT INTO rows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", pending
        )

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
            SELECT source_id, source_ordinal, logical_locator, chunk_index,
                   chunk_count, text, text_digest, source_digest,
                   declared_record_id, text_bytes, duplicate_count
            FROM (
              SELECT r.*,
                COUNT(*) OVER (PARTITION BY text_digest, text_bytes)
                  AS duplicate_count,
                ROW_NUMBER() OVER (
                  PARTITION BY text_digest, text_bytes
                  ORDER BY source_ordinal, logical_locator, chunk_index
                ) AS winner_number
              FROM rows r
            ) ranked
            WHERE winner_number = 1
            ORDER BY source_ordinal, logical_locator, chunk_index
        """
    else:
        winner_sql = """
            SELECT r.*, 1 AS duplicate_count FROM rows r
            ORDER BY r.source_ordinal, r.logical_locator, r.chunk_index
        """

    def output() -> Iterator[CanonicalCorpusRecord]:
        index = 0
        provenance_cursor = connection.cursor()
        provenance_handle = (
            provenance_path.open("w", encoding="utf-8")
            if provenance_path is not None
            else None
        )
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
                        provenance_cursor.execute(
                            "SELECT source_id || ':' || logical_locator || ':' || "
                            "CAST(chunk_index AS VARCHAR) FROM rows WHERE "
                            "text_digest = ? AND text_bytes = ? ORDER BY "
                            "source_ordinal, logical_locator, chunk_index",
                            [digest, _text_bytes],
                        )
                        preview: list[str] = []
                        match_index = 0
                        while matches := provenance_cursor.fetchmany(256):
                            for match in matches:
                                locator = str(match[0])
                                if match_index > 0 and len(preview) < 32:
                                    preview.append(locator)
                                if provenance_handle is not None:
                                    provenance_handle.write(
                                        json.dumps(
                                            {"winner": index + 1, "locator": locator},
                                            sort_keys=True,
                                        )
                                        + "\n"
                                    )
                                match_index += 1
                        provenance = tuple(preview)
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
                        duplicate_count=int(duplicate_count),
                    )
        finally:
            provenance_cursor.close()
            if provenance_handle is not None:
                provenance_handle.close()
            connection.close()
            if temporary_database:
                try:
                    Path(database_path).unlink()
                except FileNotFoundError:
                    pass

    winners = int(
        connection.execute(f"SELECT COUNT(*) FROM ({winner_sql})").fetchone()[0]
    )
    return output(), {
        "input_records": total,
        "output_records": winners,
        "duplicates_removed": total - winners,
    }


__all__ = ["deduplicate_records"]
