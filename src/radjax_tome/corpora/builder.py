from __future__ import annotations

import fnmatch
import hashlib
import json
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from radjax_tome.corpora.config import CORPUS_ARTIFACT_SCHEMA_V2
from radjax_tome.io.json import read_json_object, write_json


def _m10_imports() -> tuple[Any, ...]:
    """Keep v1 imports lazy and make the v2 boundary explicit."""
    from radjax_tome.corpora.config import load_corpus_build_intent
    from radjax_tome.corpora.dedup import deduplicate_records
    from radjax_tome.corpora.identity import (
        corpus_semantic_identity,
        example_identity,
        normalized_text,
        text_digest,
    )
    from radjax_tome.corpora.lifecycle import (
        CorpusJournal,
        preflight_corpus_build,
        publish_staging,
    )
    from radjax_tome.corpora.records import SourceRecord
    from radjax_tome.corpora.sources import iter_source_records
    from radjax_tome.corpora.storage import write_json, write_shards

    return (
        load_corpus_build_intent,
        deduplicate_records,
        corpus_semantic_identity,
        example_identity,
        normalized_text,
        text_digest,
        CorpusJournal,
        preflight_corpus_build,
        publish_staging,
        SourceRecord,
        iter_source_records,
        write_json,
        write_shards,
    )


CORPUS_JSONL_FILENAME = "corpus.jsonl"
CORPUS_MANIFEST_FILENAME = "corpus_manifest.json"
CORPUS_BUILD_REPORT_FILENAME = "corpus_build_report.json"
CORPUS_MANIFEST_SCHEMA = "corpus_manifest_v1"
CORPUS_BUILD_REPORT_SCHEMA = "corpus_build_report_v1"
CORPUS_ARTIFACT_TYPE = "radjax_tome_corpus"
NORMALIZATION_POLICY = "text_normalize_lf_strip_trailing_ws_v1"
CHUNKING_POLICY = "char_window_v1"
DEDUPLICATION_POLICY = "exact_normalized_text_sha256_v1"
SOURCE_DISCOVERY_POLICY = "local_files_sorted_globs_v1"
MANIFEST_HASH_POLICY = "exclude_self_hash_and_created_at_v1"
SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".py", ".jsonl"}
DEFAULT_EXCLUDE_GLOBS = (
    "**/.git/**",
    "**/__pycache__/**",
    "**/.venv/**",
    "**/venv/**",
    "**/node_modules/**",
    "**/.mypy_cache/**",
    "**/.pytest_cache/**",
    "**/dist/**",
    "**/build/**",
)
_DEFAULT_EXCLUDE_PARTS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
}


@dataclass(frozen=True)
class CorpusBuildConfig:
    inputs: tuple[Path, ...]
    output_dir: Path
    include_globs: tuple[str, ...] = ()
    exclude_globs: tuple[str, ...] = DEFAULT_EXCLUDE_GLOBS
    min_chars: int = 1
    max_chars: int = 12_000
    overwrite: bool = False


@dataclass(frozen=True)
class CorpusValidationReport:
    status: str
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    corpus_hash: str | None = None
    manifest_hash: str | None = None
    num_examples: int = 0
    num_sources: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_corpus_artifact(config: CorpusBuildConfig) -> dict[str, Any]:
    _validate_build_config(config)
    if config.output_dir.exists():
        if not config.overwrite:
            raise ValueError(f"corpus output already exists: {config.output_dir}")
        shutil.rmtree(config.output_dir)
    config.output_dir.mkdir(parents=True)

    discovered = _discover_sources(config)
    rows, source_records, counters = _build_rows(config, discovered)
    if not rows:
        blockers = ["no usable corpus examples were produced"]
        report = _build_report(config, blockers=blockers, counters=counters)
        write_json(config.output_dir / CORPUS_BUILD_REPORT_FILENAME, report)
        raise ValueError("; ".join(blockers))

    corpus_bytes = _corpus_jsonl_bytes(rows)
    corpus_path = config.output_dir / CORPUS_JSONL_FILENAME
    corpus_path.write_bytes(corpus_bytes)
    corpus_hash = _sha256_bytes(corpus_bytes)
    manifest = _build_manifest(
        config,
        rows=rows,
        source_records=source_records,
        corpus_hash=corpus_hash,
    )
    write_json(config.output_dir / CORPUS_MANIFEST_FILENAME, manifest)
    report = _build_report(
        config,
        blockers=[],
        counters=counters,
        corpus_hash=corpus_hash,
        manifest_hash=str(manifest["manifest_hash"]),
        rows=rows,
        source_records=source_records,
    )
    write_json(config.output_dir / CORPUS_BUILD_REPORT_FILENAME, report)
    validation = validate_corpus_artifact(config.output_dir)
    if validation.status != "pass":
        raise ValueError(
            "built corpus artifact failed validation: " + "; ".join(validation.blockers)
        )
    return report


def inspect_corpus_artifact(path: Path) -> dict[str, Any]:
    manifest = read_corpus_manifest(Path(path) / CORPUS_MANIFEST_FILENAME)
    return {
        "corpus_schema": manifest["schema_version"],
        "num_examples": manifest["num_examples"],
        "num_sources": manifest["num_sources"],
        "num_chars": manifest["num_chars"],
        "corpus_hash": manifest["corpus_hash"],
        "manifest_hash": manifest["manifest_hash"],
        "manifest_hash_policy": manifest["manifest_hash_policy"],
        "normalization_policy": manifest["normalization_policy"],
        "chunking_policy": manifest["chunking_policy"],
        "deduplication_policy": manifest["deduplication_policy"],
    }


def validate_corpus_artifact(path: str | Path) -> CorpusValidationReport:
    root = Path(path)
    blockers: list[str] = []
    warnings: list[str] = []
    corpus_path = root / CORPUS_JSONL_FILENAME
    manifest_path = root / CORPUS_MANIFEST_FILENAME
    report_path = root / CORPUS_BUILD_REPORT_FILENAME
    for required in (corpus_path, manifest_path, report_path):
        if not required.is_file():
            blockers.append(f"missing required corpus file: {required.name}")
    if blockers:
        return _validation_report(blockers=blockers, warnings=warnings)

    rows = _read_corpus_rows(corpus_path, blockers)
    manifest: dict[str, Any] = {}
    try:
        manifest = read_corpus_manifest(manifest_path)
    except ValueError as exc:
        blockers.append(str(exc))

    _validate_rows(rows, blockers)
    actual_corpus_hash = _sha256_bytes(corpus_path.read_bytes())
    actual_manifest_hash = (
        _compute_manifest_hash(manifest) if isinstance(manifest, dict) else None
    )
    if manifest:
        if manifest.get("manifest_hash_policy") != MANIFEST_HASH_POLICY:
            blockers.append("corpus_manifest.json manifest_hash_policy is unsupported")
        if manifest.get("corpus_hash") != actual_corpus_hash:
            blockers.append(
                "corpus_manifest.json corpus_hash does not match corpus.jsonl"
            )
        if manifest.get("manifest_hash") != actual_manifest_hash:
            blockers.append(
                "corpus_manifest.json manifest_hash does not validate under "
                "exclude_self_hash_and_created_at_v1"
            )
        if int(manifest.get("num_examples", -1)) != len(rows):
            blockers.append("manifest num_examples does not match corpus rows")
        row_chars = sum(len(str(row.get("text", ""))) for row in rows)
        if int(manifest.get("num_chars", -1)) != row_chars:
            blockers.append("manifest num_chars does not match corpus rows")
        source_ids = {str(row.get("source_id")) for row in rows}
        if int(manifest.get("num_sources", -1)) != len(source_ids):
            blockers.append("manifest num_sources does not match corpus rows")

    try:
        build_report = read_json_object(report_path)
    except ValueError as exc:
        blockers.append(str(exc))
    else:
        if build_report.get("schema_version") != CORPUS_BUILD_REPORT_SCHEMA:
            blockers.append("corpus_build_report.json schema_version is unsupported")

    return _validation_report(
        blockers=blockers,
        warnings=warnings,
        corpus_hash=actual_corpus_hash if corpus_path.is_file() else None,
        manifest_hash=str(manifest.get("manifest_hash"))
        if isinstance(manifest, dict) and manifest.get("manifest_hash")
        else None,
        num_examples=len(rows),
        num_sources=len({str(row.get("source_id")) for row in rows}),
    )


def read_corpus_manifest(path: str | Path) -> dict[str, Any]:
    manifest = read_json_object(Path(path))
    if manifest.get("schema_version") != CORPUS_MANIFEST_SCHEMA:
        raise ValueError("corpus_manifest.json schema_version is unsupported")
    if manifest.get("artifact_type") != CORPUS_ARTIFACT_TYPE:
        raise ValueError("corpus_manifest.json artifact_type is unsupported")
    return manifest


def corpus_provenance_from_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    manifest = read_corpus_manifest(manifest_path)
    if manifest.get("manifest_hash_policy") != MANIFEST_HASH_POLICY:
        raise ValueError("corpus manifest hash policy is unsupported")
    expected_hash = _compute_manifest_hash(manifest)
    if manifest.get("manifest_hash") != expected_hash:
        raise ValueError("corpus manifest hash does not validate")
    return {
        "source_corpus_hash": manifest["corpus_hash"],
        "source_corpus_manifest_hash": manifest["manifest_hash"],
        "source_corpus_schema_version": manifest["schema_version"],
        "source_corpus_num_examples": manifest["num_examples"],
        "source_corpus_num_sources": manifest["num_sources"],
        "source_corpus_normalization_policy": manifest["normalization_policy"],
        "source_corpus_chunking_policy": manifest["chunking_policy"],
        "source_corpus_deduplication_policy": manifest["deduplication_policy"],
        "source_corpus_manifest_path": str(manifest_path),
    }


def stringify_corpus_provenance(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    return {
        key: str(value) for key, value in corpus_provenance_from_manifest(path).items()
    }


def normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    return normalized.strip()


def canonical_corpus_row(row: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(row),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _validate_build_config(config: CorpusBuildConfig) -> None:
    if not config.inputs:
        raise ValueError("at least one corpus input is required")
    if config.min_chars < 1:
        raise ValueError("min_chars must be >= 1")
    if config.max_chars < config.min_chars:
        raise ValueError("max_chars must be >= min_chars")


def _discover_sources(config: CorpusBuildConfig) -> list[dict[str, Any]]:
    discovered: list[dict[str, Any]] = []
    for input_path in config.inputs:
        root = input_path if input_path.is_dir() else input_path.parent
        if input_path.is_file():
            candidates = (input_path,)
        elif input_path.is_dir():
            candidates = tuple(path for path in input_path.rglob("*") if path.is_file())
        else:
            discovered.append(
                {
                    "path": input_path,
                    "root": root,
                    "included": False,
                    "excluded_reason": "input_missing",
                }
            )
            continue
        for candidate in candidates:
            reason = _excluded_reason(
                candidate,
                root=root,
                include_globs=config.include_globs,
                exclude_globs=config.exclude_globs,
            )
            discovered.append(
                {
                    "path": candidate,
                    "root": root,
                    "included": reason is None,
                    "excluded_reason": reason,
                }
            )
    return sorted(
        discovered,
        key=lambda item: (
            str(Path(item["root"]).resolve()),
            _relative_path(Path(item["path"]), Path(item["root"])),
        ),
    )


def _excluded_reason(
    path: Path,
    *,
    root: Path,
    include_globs: tuple[str, ...],
    exclude_globs: tuple[str, ...],
) -> str | None:
    relative = _relative_path(path, root)
    if any(part in _DEFAULT_EXCLUDE_PARTS for part in PurePosixPath(relative).parts):
        return "default_exclude"
    if _matches_any(relative, exclude_globs):
        return "exclude_glob"
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        return "unsupported_file_type"
    if include_globs and not _matches_any(relative, include_globs):
        return "include_glob_miss"
    if _looks_binary(path):
        return "binary_file"
    return None


def _matches_any(relative: str, globs: Iterable[str]) -> bool:
    posix = PurePosixPath(relative)
    for pattern in globs:
        root_pattern = pattern[3:] if pattern.startswith("**/") else pattern
        if (
            posix.match(pattern)
            or fnmatch.fnmatch(relative, pattern)
            or posix.match(root_pattern)
            or fnmatch.fnmatch(relative, root_pattern)
        ):
            return True
    return False


def _looks_binary(path: Path) -> bool:
    try:
        return b"\x00" in path.read_bytes()[:4096]
    except OSError:
        return True


def _build_rows(
    config: CorpusBuildConfig,
    discovered: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    candidate_rows: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    counters = {
        "sources_seen": len(discovered),
        "sources_included": 0,
        "sources_excluded": 0,
        "duplicates_removed": 0,
        "empty_removed": 0,
        "unsupported_removed": 0,
    }
    for item in discovered:
        path = Path(item["path"])
        root = Path(item["root"])
        source_hash = _sha256_file(path) if path.is_file() else None
        if not item["included"]:
            counters["sources_excluded"] += 1
            if item["excluded_reason"] in {"unsupported_file_type", "binary_file"}:
                counters["unsupported_removed"] += 1
            source_records.append(
                _source_record(
                    path,
                    root=root,
                    source_hash=source_hash,
                    included=False,
                    excluded_reason=str(item["excluded_reason"]),
                )
            )
            continue
        file_rows, empty_removed = _rows_for_file(path, root, source_hash, config)
        counters["empty_removed"] += empty_removed
        if not file_rows:
            counters["sources_excluded"] += 1
            source_records.append(
                _source_record(
                    path,
                    root=root,
                    source_hash=source_hash,
                    included=False,
                    excluded_reason="empty_after_normalization",
                )
            )
            continue
        counters["sources_included"] += 1
        candidate_rows.extend(file_rows)
        source_records.append(
            _source_record(
                path,
                root=root,
                source_hash=source_hash,
                included=True,
                excluded_reason=None,
                example_count=len(file_rows),
                char_count=sum(int(row["char_count"]) for row in file_rows),
            )
        )

    deduped: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for row in candidate_rows:
        content_hash = str(row["content_hash"])
        if content_hash in seen_hashes:
            counters["duplicates_removed"] += 1
            continue
        seen_hashes.add(content_hash)
        deduped.append(row)
    for index, row in enumerate(deduped, start=1):
        row["example_id"] = f"corpus_{index:09d}"
    return deduped, source_records, counters


def _rows_for_file(
    path: Path,
    root: Path,
    source_hash: str | None,
    config: CorpusBuildConfig,
) -> tuple[list[dict[str, Any]], int]:
    texts: list[str] = []
    empty_removed = 0
    if path.suffix.lower() == ".jsonl":
        for text in _jsonl_text_rows(path):
            normalized = normalize_text(text)
            if len(normalized) < config.min_chars:
                empty_removed += 1
                continue
            texts.append(normalized)
    else:
        try:
            normalized = normalize_text(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            return [], 0
        if len(normalized) < config.min_chars:
            return [], 1
        texts.append(normalized)

    rows: list[dict[str, Any]] = []
    for normalized in texts:
        chunks = _chunk_text(normalized, max_chars=config.max_chars)
        chunk_count = len(chunks)
        for chunk_index, chunk in enumerate(chunks):
            if len(chunk) < config.min_chars:
                empty_removed += 1
                continue
            rows.append(
                {
                    "char_count": len(chunk),
                    "chunk_count": chunk_count,
                    "chunk_index": chunk_index,
                    "chunking_policy": CHUNKING_POLICY,
                    "content_hash": _sha256_text(chunk),
                    "deduplication_policy": DEDUPLICATION_POLICY,
                    "example_id": "__pending__",
                    "normalization_policy": NORMALIZATION_POLICY,
                    "source_hash": source_hash,
                    "source_id": f"local:{path.resolve()}",
                    "source_kind": "local_file",
                    "source_path": str(path.resolve()),
                    "source_relative_path": _relative_path(path, root),
                    "source_root": str(root.resolve()),
                    "text": chunk,
                }
            )
    return rows, empty_removed


def _jsonl_text_rows(path: Path) -> list[str]:
    texts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            text = payload.get("text")
            if isinstance(text, str):
                texts.append(text)
    return texts


def _chunk_text(text: str, *, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end].strip())
        start = end
    return [chunk for chunk in chunks if chunk]


def _source_record(
    path: Path,
    *,
    root: Path,
    source_hash: str | None,
    included: bool,
    excluded_reason: str | None,
    example_count: int = 0,
    char_count: int = 0,
) -> dict[str, Any]:
    return {
        "char_count": char_count,
        "example_count": example_count,
        "excluded_reason": excluded_reason,
        "included": included,
        "source_hash": source_hash,
        "source_id": f"local:{path.resolve()}",
        "source_kind": "local_file",
        "source_path": str(path.resolve()),
        "source_relative_path": _relative_path(path, root),
    }


def _build_manifest(
    config: CorpusBuildConfig,
    *,
    rows: list[dict[str, Any]],
    source_records: list[dict[str, Any]],
    corpus_hash: str,
) -> dict[str, Any]:
    retained_source_ids = {row["source_id"] for row in rows}
    included_sources = [
        record
        for record in source_records
        if record["included"] and record["source_id"] in retained_source_ids
    ]
    manifest = {
        "artifact_type": CORPUS_ARTIFACT_TYPE,
        "builder": {"name": "radjax-tome corpus builder", "phase": "4.1"},
        "chunking_policy": CHUNKING_POLICY,
        "corpus_hash": corpus_hash,
        "corpus_jsonl_path": CORPUS_JSONL_FILENAME,
        "created_at": _utc_created_at(),
        "deduplication_policy": DEDUPLICATION_POLICY,
        "exclude_globs": list(config.exclude_globs),
        "include_globs": list(config.include_globs),
        "manifest_hash": None,
        "manifest_hash_policy": MANIFEST_HASH_POLICY,
        "normalization_policy": NORMALIZATION_POLICY,
        "num_chars": sum(int(row["char_count"]) for row in rows),
        "num_examples": len(rows),
        "num_sources": len(included_sources),
        "schema_version": CORPUS_MANIFEST_SCHEMA,
        "source_discovery_policy": SOURCE_DISCOVERY_POLICY,
        "source_records": included_sources,
        "source_roots": sorted(
            {
                str((path if path.is_dir() else path.parent).resolve())
                for path in config.inputs
            }
        ),
    }
    manifest["manifest_hash"] = _compute_manifest_hash(manifest)
    return manifest


def _build_report(
    config: CorpusBuildConfig,
    *,
    blockers: list[str],
    counters: Mapping[str, int],
    corpus_hash: str | None = None,
    manifest_hash: str | None = None,
    rows: list[dict[str, Any]] | None = None,
    source_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rows = rows or []
    source_records = source_records or []
    warnings: list[str] = []
    excluded = [record for record in source_records if not record["included"]]
    if excluded:
        warnings.append(f"excluded {len(excluded)} source file(s)")
    status = "fail" if blockers else "warn" if warnings else "pass"
    return {
        "blockers": blockers,
        "created_at": _utc_created_at(),
        "corpus_hash": corpus_hash,
        "corpus_jsonl_path": str(config.output_dir / CORPUS_JSONL_FILENAME),
        "corpus_manifest_path": str(config.output_dir / CORPUS_MANIFEST_FILENAME),
        "excluded_sources": excluded,
        "manifest_hash": manifest_hash,
        "manifest_hash_policy": MANIFEST_HASH_POLICY,
        "num_chars": sum(int(row["char_count"]) for row in rows),
        "num_duplicates_removed": counters.get("duplicates_removed", 0),
        "num_empty_removed": counters.get("empty_removed", 0),
        "num_examples": len(rows),
        "num_sources_excluded": counters.get("sources_excluded", 0),
        "num_sources_included": counters.get("sources_included", 0),
        "num_sources_seen": counters.get("sources_seen", 0),
        "num_unsupported_removed": counters.get("unsupported_removed", 0),
        "output_dir": str(config.output_dir),
        "schema_version": CORPUS_BUILD_REPORT_SCHEMA,
        "status": status,
        "warnings": warnings,
    }


def _read_corpus_rows(path: Path, blockers: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                blockers.append(
                    f"corpus.jsonl line {line_number} invalid JSON: {exc.msg}"
                )
                continue
            if not isinstance(payload, dict):
                blockers.append(f"corpus.jsonl line {line_number} must be an object")
                continue
            rows.append(payload)
    return rows


def _validate_rows(rows: list[dict[str, Any]], blockers: list[str]) -> None:
    required = {
        "example_id",
        "text",
        "source_id",
        "source_path",
        "source_kind",
        "source_hash",
        "content_hash",
        "chunk_index",
        "chunk_count",
        "char_count",
    }
    seen_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        missing = sorted(required.difference(row))
        blockers.extend(
            f"corpus.jsonl line {index} missing {field}" for field in missing
        )
        example_id = str(row.get("example_id", ""))
        if example_id in seen_ids:
            blockers.append(f"duplicate example_id: {example_id}")
        seen_ids.add(example_id)
        text = row.get("text")
        if not isinstance(text, str) or not text:
            blockers.append(f"corpus.jsonl line {index} text must be non-empty")
            continue
        if row.get("content_hash") != _sha256_text(text):
            blockers.append(f"corpus.jsonl line {index} content_hash mismatch")
        if int(row.get("char_count", -1)) != len(text):
            blockers.append(f"corpus.jsonl line {index} char_count mismatch")
        source_path = Path(str(row.get("source_path", "")))
        if source_path.is_file() and row.get("source_hash") != _sha256_file(
            source_path
        ):
            blockers.append(f"corpus.jsonl line {index} source_hash mismatch")


def _validation_report(
    *,
    blockers: list[str],
    warnings: list[str],
    corpus_hash: str | None = None,
    manifest_hash: str | None = None,
    num_examples: int = 0,
    num_sources: int = 0,
) -> CorpusValidationReport:
    return CorpusValidationReport(
        status="fail" if blockers else "pass",
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        corpus_hash=corpus_hash,
        manifest_hash=manifest_hash,
        num_examples=num_examples,
        num_sources=num_sources,
    )


def _corpus_jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return ("".join(canonical_corpus_row(row) + "\n" for row in rows)).encode("utf-8")


def _compute_manifest_hash(manifest: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in manifest.items()
        if key not in {"manifest_hash", "created_at"}
    }
    return _sha256_bytes(_canonical_json_bytes(payload))


def _utc_created_at() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# M10 v2 boundary.  The v1 functions above are deliberately retained verbatim
# for historical configs and fixtures.


def build_corpus_artifact_v2(
    intent: Any,
    *,
    fault_after: str | None = None,
) -> dict[str, Any]:
    """Build and publish a deterministic, sharded corpus v2 artifact."""

    (
        load_intent,
        deduplicate_records,
        semantic_identity,
        example_id_hash,
        normalize,
        digest_text,
        Journal,
        preflight,
        publish,
        SourceRecord,
        iter_sources,
        write_member,
        write_shards,
    ) = _m10_imports()
    if isinstance(intent, (str, Path)):
        intent = load_intent(intent)
    preflight(intent)
    destination = intent.output_path
    config_identity = _sha256_bytes(
        json.dumps(
            _transaction_intent(intent), sort_keys=True, separators=(",", ":")
        ).encode()
    )
    if intent.resume and destination.is_dir():
        from radjax_tome.corpora.validation import validate_corpus_artifact_v2

        existing = validate_corpus_artifact_v2(destination)
        report_path = destination / "build_report.json"
        existing_identity = None
        if report_path.is_file():
            existing_identity = json.loads(report_path.read_text()).get(
                "config_identity"
            )
        if existing.ok and existing_identity == config_identity:
            return {
                "status": "resumed",
                "artifact_path": str(destination),
                "semantic_identity": existing.semantic_identity,
                "num_examples": existing.num_examples,
            }
        raise ValueError("resume refused: existing corpus artifact is invalid")

    if intent.resume and not destination.exists():
        staging_candidates = sorted(
            destination.parent.glob(f".{destination.name}.m10-*"),
            key=lambda path: path.stat().st_mtime,
        )
        for candidate in reversed(staging_candidates):
            try:
                _require_valid(candidate)
                staged_report = json.loads(
                    (candidate / "build_report.json").read_text()
                )
                if staged_report.get("config_identity") != config_identity:
                    continue
            except (OSError, TypeError, ValueError):
                continue
            journal_path = candidate / "journal" / "corpus_build_journal_v1.jsonl"
            journal = Journal.reopen(journal_path, config_identity)
            publish(
                candidate,
                destination,
                overwrite=False,
                validate=lambda path: _require_valid(path),
                journal=journal,
                ownership_check=_is_owned_v2_artifact,
            )
            report = json.loads((destination / "build_report.json").read_text())
            report["status"] = "resumed"
            return report
        raise ValueError("resume refused: no valid owned staging state")

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.m10-", dir=str(destination.parent)
        )
    )
    (staging / ".radjax_corpus_staging").write_text(
        config_identity + "\n", encoding="utf-8"
    )
    transaction_id = staging.name
    policy = dict(intent.policy)
    tokenizer_value = policy.get("tokenizer", "smoke")
    from radjax_tome.corpora.language_tokenizer_binding import (
        capture_language_tokenizer_binding,
    )
    from radjax_tome.corpora.tokenizer import create_tokenizer

    journal = Journal(
        staging / "journal" / "corpus_build_journal_v1.jsonl",
        transaction_id,
        config_identity,
    )
    journal.append("PREFLIGHTED")
    tokenizer = create_tokenizer(tokenizer_value)
    binding = capture_language_tokenizer_binding(tokenizer)
    binding_path = staging / "language_tokenizer_binding_v1.json"
    binding_path.parent.mkdir(parents=True, exist_ok=True)
    binding_path.write_bytes(
        binding.binding and _canonical_json(binding.binding) + b"\n"
    )
    resource_path = staging / "resources" / "tokenizer_vocabulary.jsonl"
    resource_path.parent.mkdir(parents=True, exist_ok=True)
    resource_path.write_bytes(binding.vocabulary_jsonl)
    binding_digest = str(binding.binding["canonical_binding_digest"])
    journal.append("SOURCES_RESOLVED")

    minimum = int(
        policy.get("filtering", {}).get("min_chars", policy.get("min_chars", 1))
    )
    maximum = int(
        policy.get("chunking", {}).get("max_chars", policy.get("max_chars", 12_000))
    )
    if minimum < 1 or maximum < minimum:
        raise ValueError("invalid corpus filtering/chunking limits")
    source_inventory: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []
    arrival = 0

    def candidate_records() -> Iterable[Any]:
        nonlocal arrival
        for source_ordinal, spec in enumerate(intent.sources):
            candidate_count = retained_count = 0
            for item in iter_sources(spec):
                raw_text = item.text
                normalized_text_value = normalize(raw_text)
                if not normalized_text_value:
                    filtered.append(
                        {
                            "source_id": spec.source_id,
                            "reason": "CORPUS_FILTER_EMPTY"
                            if not normalized_text_value
                            else "CORPUS_FILTER_BELOW_MINIMUM",
                        }
                    )
                    continue
                chunk_count = (len(normalized_text_value) + maximum - 1) // maximum
                for chunk_index in range(chunk_count):
                    chunk = normalized_text_value[
                        chunk_index * maximum : (chunk_index + 1) * maximum
                    ]
                    if len(chunk) < minimum:
                        filtered.append(
                            {
                                "source_id": spec.source_id,
                                "reason": "CORPUS_FILTER_BELOW_MINIMUM",
                            }
                        )
                        continue
                    candidate_count += 1
                    retained_count += 1
                    arrival += 1
                    yield SourceRecord(
                        source_id=spec.source_id,
                        source_ordinal=source_ordinal,
                        logical_locator=item.logical_locator,
                        chunk_index=chunk_index,
                        chunk_count=chunk_count,
                        text=chunk,
                        normalized_text_digest=digest_text(chunk),
                        source_digest=item.source_digest,
                        declared_record_id=item.declared_record_id,
                    )
            source_inventory.append(
                {
                    "source_id": spec.source_id,
                    "source_ordinal": source_ordinal,
                    "adapter": spec.adapter,
                    "path": str(spec.path),
                    "candidate_count": candidate_count,
                    "retained_count": retained_count,
                }
            )

    dedup_enabled = bool(
        policy.get("deduplication", {}).get(
            "enabled", policy.get("deduplication_enabled", True)
        )
    )
    worker_count = int(intent.resources.get("worker_count", 1))
    records, counts = deduplicate_records(
        candidate_records(),
        enabled=dedup_enabled,
        memory_limit=intent.resources.get(
            "duckdb_memory_limit", intent.resources.get("memory_limit")
        ),
        worker_count=worker_count,
        provenance_path=staging / "duplicate_provenance.jsonl",
    )
    journal.append("INGEST_COMPLETE", candidate_count=arrival)

    def identified_records() -> Iterable[Any]:
        for record in records:
            yield replace(
                record,
                identity_digest=example_id_hash(
                    source_id=record.source_id,
                    logical_locator=record.logical_locator,
                    chunk_index=record.chunk_index,
                    text_digest=record.text_digest,
                ),
            )

    inventory = write_shards(
        staging,
        identified_records(),
        shard_capacity=int(
            intent.layout.get(
                "shard_capacity", intent.layout.get("shard_size_examples", 128)
            )
        ),
        max_shard_bytes=int(intent.layout["max_shard_bytes"])
        if intent.layout.get("max_shard_bytes") is not None
        else None,
    )
    write_member(staging / "shard_inventory.json", inventory)
    write_member(staging / "source_manifest.json", source_inventory)
    write_member(
        staging / "filter_report.json",
        {"schema_version": "corpus_filter_report_v2", "records": filtered},
    )
    write_member(
        staging / "dedup_report.json",
        {"schema_version": "corpus_dedup_report_v2", **counts},
    )
    write_member(staging / "normalized_intent.json", _jsonable_intent(intent))
    from radjax_tome.corpora.storage import VerifiedCorpusReader

    source_declarations = [
        {
            "source_id": source.source_id,
            "adapter": source.adapter,
            "include": list(source.include),
            "exclude": list(source.exclude),
            "text_field": source.text_field,
            "record_id_field": source.record_id_field,
        }
        for source in intent.sources
    ]
    semantic = semantic_identity(
        policy=policy,
        tokenizer_binding_digest=binding_digest,
        records=(_record_from_dict(row) for row in VerifiedCorpusReader(staging)),
        source_declarations=source_declarations,
    )
    manifest = {
        "schema_version": CORPUS_ARTIFACT_SCHEMA_V2,
        "artifact_type": CORPUS_ARTIFACT_SCHEMA_V2,
        "semantic_identity": semantic,
        "tokenizer_binding_digest": binding_digest,
        "policy": policy,
        "source_declarations": source_declarations,
        "num_examples": counts["output_records"],
        "num_sources": len(intent.sources),
        "shard_count": len(inventory),
        "storage_schema": "uncompressed_canonical_jsonl_with_offsets_v1",
    }
    write_member(staging / "corpus_manifest.json", manifest)
    write_member(
        staging / "build_report.json",
        {
            "schema_version": "corpus_build_report_v2",
            "status": "pass",
            "semantic_identity": semantic,
            "counts": counts,
            "filtered_count": len(filtered),
            "shard_count": len(inventory),
            "atomic_overwrite": False,
            "config_identity": config_identity,
            "resources": {
                "memory_limit": intent.resources.get("memory_limit"),
                "duckdb_memory_limit": intent.resources.get("duckdb_memory_limit"),
                "worker_count": worker_count,
                "max_open_files": intent.resources.get("max_open_files"),
                "max_artifact_bytes": intent.resources.get("max_artifact_bytes"),
            },
        },
    )
    write_member(
        staging / "corpus_cover.json",
        {
            "schema_version": CORPUS_ARTIFACT_SCHEMA_V2,
            "artifact_type": CORPUS_ARTIFACT_SCHEMA_V2,
            "semantic_members": ["corpus_manifest.json", "normalized_intent.json"],
            "provenance_members": [
                "source_manifest.json",
                "language_tokenizer_binding_v1.json",
            ],
            "physical_members": ["shard_inventory.json", "shards", "indexes"],
            "diagnostic_members": [
                "filter_report.json",
                "dedup_report.json",
                "build_report.json",
            ],
            "member_inventory_policy": (
                "sha256_size_role_schema_v1_excludes_cover_self_hash"
            ),
            "members": _public_member_inventory(staging),
            "atomic_overwrite": False,
        },
    )
    journal.append("SHARDS_SEALED", shard_count=len(inventory))
    from radjax_tome.corpora.validation import validate_corpus_artifact_v2

    validation = validate_corpus_artifact_v2(staging)
    if not validation.ok:
        raise ValueError(
            "built corpus artifact failed validation: " + "; ".join(validation.blockers)
        )
    max_bytes = intent.resources.get("max_artifact_bytes")
    if max_bytes is not None:
        total_bytes = sum(
            path.stat().st_size for path in staging.rglob("*") if path.is_file()
        )
        if total_bytes > int(max_bytes):
            raise ValueError("artifact exceeds resources.max_artifact_bytes")
    journal.append("VALIDATED", semantic_identity=semantic)
    if fault_after == "validated":
        raise InterruptedError("fault injection after validation")
    publish(
        staging,
        destination,
        overwrite=intent.overwrite,
        validate=lambda path: _require_valid(path),
        journal=journal,
        ownership_check=_is_owned_v2_artifact,
    )
    return {
        "status": "pass",
        "artifact_path": str(destination),
        "semantic_identity": semantic,
        "num_examples": counts["output_records"],
        "num_sources": len(intent.sources),
        "shard_count": len(inventory),
    }


def _require_valid(path: Path) -> None:
    from radjax_tome.corpora.validation import validate_corpus_artifact_v2

    result = validate_corpus_artifact_v2(path)
    if not result.ok:
        raise ValueError("corpus validation failed: " + "; ".join(result.blockers))


def _is_owned_v2_artifact(path: Path) -> bool:
    from radjax_tome.corpora.validation import validate_corpus_artifact_v2

    return path.is_dir() and validate_corpus_artifact_v2(path).ok


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _jsonable_intent(intent: Any) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        if hasattr(value, "__dict__"):
            return convert(vars(value))
        return value

    return convert(
        {
            "schema_version": intent.schema_version,
            "artifact": dict(intent.artifact),
            "sources": [
                {
                    "source_id": item.source_id,
                    "adapter": item.adapter,
                    "include": list(item.include),
                    "exclude": list(item.exclude),
                    "text_field": item.text_field,
                    "record_id_field": item.record_id_field,
                    "path": str(item.path),
                }
                for item in intent.sources
            ],
            "policy": dict(intent.policy),
            "layout": dict(intent.layout),
            "resources": dict(intent.resources),
            "output": dict(intent.output),
            "execution": dict(intent.execution),
            "reporting": dict(intent.reporting),
        }
    )


def _transaction_intent(intent: Any) -> dict[str, Any]:
    value = _jsonable_intent(intent)
    execution = dict(value["execution"])
    execution["resume"] = False
    value["execution"] = execution
    return value


def _public_member_inventory(root: Path) -> list[dict[str, Any]]:
    members = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == "corpus_cover.json" or relative.startswith("journal/"):
            continue
        if relative.startswith("."):
            continue
        role = "physical"
        if relative in {"corpus_manifest.json", "normalized_intent.json"}:
            role = "semantic"
        elif relative in {
            "source_manifest.json",
            "language_tokenizer_binding_v1.json",
        } or relative.startswith("resources/"):
            role = "provenance"
        elif relative.endswith("_report.json"):
            role = "diagnostic"
        members.append(
            {
                "path": relative,
                "role": role,
                "media_type": "application/json"
                if path.suffix == ".json"
                else "application/jsonl",
                "schema": _member_schema(relative),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return members


def _member_schema(relative: str) -> str:
    if relative.startswith("shards/"):
        return "corpus_record_v2"
    if relative.startswith("indexes/"):
        return "corpus_index_v1"
    if relative == "resources/tokenizer_vocabulary.jsonl":
        return "tokenizer_vocabulary_v1"
    return "json_document_v1"


def _record_from_dict(row: Mapping[str, Any]) -> Any:
    from radjax_tome.corpora.records import CanonicalCorpusRecord

    return CanonicalCorpusRecord(
        example_id=str(row["example_id"]),
        source_id=str(row["source_id"]),
        source_ordinal=int(row["source_ordinal"]),
        logical_locator=str(row["logical_locator"]),
        chunk_index=int(row["chunk_index"]),
        chunk_count=int(row.get("chunk_count", 1)),
        text=str(row["text"]),
        text_digest=str(row["text_digest"]),
        source_digest=str(row.get("source_digest", "")),
    )


__all__ = [
    "build_corpus_artifact_v2",
    "build_corpus_artifact",
    "validate_corpus_artifact",
]
