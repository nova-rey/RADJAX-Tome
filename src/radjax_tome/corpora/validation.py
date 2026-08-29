"""Validation and inspection for self-describing corpus v2 artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from radjax_tome.corpora.config import CORPUS_ARTIFACT_SCHEMA_V2, sha256
from radjax_tome.corpora.storage import VerifiedCorpusReader


@dataclass(frozen=True)
class CorpusIssue:
    code: str
    message: str
    member: str | None = None


@dataclass(frozen=True)
class CorpusValidationResult:
    status: str
    issues: tuple[CorpusIssue, ...] = ()
    semantic_identity: str | None = None
    num_examples: int = 0
    num_sources: int = 0

    @property
    def ok(self) -> bool:
        return self.status == "pass"

    @property
    def blockers(self) -> tuple[str, ...]:
        return tuple(issue.message for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "ok": self.ok,
            "issues": [issue.__dict__ for issue in self.issues],
            "semantic_identity": self.semantic_identity,
            "num_examples": self.num_examples,
            "num_sources": self.num_sources,
        }


@dataclass(frozen=True)
class CorpusInspection:
    path: Path
    schema_version: str
    semantic_identity: str
    num_examples: int
    num_sources: int
    shard_count: int
    validation: CorpusValidationResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "schema_version": self.schema_version,
            "semantic_identity": self.semantic_identity,
            "num_examples": self.num_examples,
            "num_sources": self.num_sources,
            "shard_count": self.shard_count,
            "validation": self.validation.to_dict(),
        }


def validate_corpus_artifact_v2(path: str | Path) -> CorpusValidationResult:
    root = Path(path).resolve()
    issues: list[CorpusIssue] = []
    required = (
        "corpus_cover.json",
        "corpus_manifest.json",
        "normalized_intent.json",
        "source_manifest.json",
        "language_tokenizer_binding_v1.json",
        "shard_inventory.json",
        "filter_report.json",
        "dedup_report.json",
        "build_report.json",
    )
    for name in required:
        if not (root / name).is_file():
            issues.append(
                CorpusIssue("MISSING_MEMBER", f"missing corpus member: {name}", name)
            )
    if issues:
        return CorpusValidationResult("fail", tuple(issues))
    try:
        cover = _json(root / "corpus_cover.json")
        manifest = _json(root / "corpus_manifest.json")
        inventory = _json(root / "shard_inventory.json")
    except ValueError as exc:
        return CorpusValidationResult(
            "fail", (CorpusIssue("MALFORMED_JSON", str(exc)),)
        )
    if cover.get("schema_version") != CORPUS_ARTIFACT_SCHEMA_V2:
        issues.append(
            CorpusIssue("SCHEMA_UNSUPPORTED", "corpus cover schema is unsupported")
        )
    if manifest.get("schema_version") != CORPUS_ARTIFACT_SCHEMA_V2:
        issues.append(
            CorpusIssue("SCHEMA_UNSUPPORTED", "corpus manifest schema is unsupported")
        )
    if not isinstance(inventory, list):
        issues.append(
            CorpusIssue("INVENTORY_INVALID", "shard inventory must be an array")
        )
        return CorpusValidationResult("fail", tuple(issues))
    try:
        rows = list(VerifiedCorpusReader(root))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        issues.append(CorpusIssue("SHARD_INVALID", str(exc)))
        return CorpusValidationResult("fail", tuple(issues))
    expected_count = int(manifest.get("num_examples", -1))
    if expected_count != len(rows):
        issues.append(
            CorpusIssue(
                "COUNT_MISMATCH", "manifest example count does not match shards"
            )
        )
    expected_ids = [f"corpus_{index:09d}" for index in range(1, len(rows) + 1)]
    if [row.get("example_id") for row in rows] != expected_ids:
        issues.append(
            CorpusIssue("ORDER_INVALID", "example IDs are not contiguous and ordered")
        )
    previous: tuple[int, str, int] | None = None
    for row in rows:
        text = row.get("text")
        if not isinstance(text, str):
            issues.append(CorpusIssue("ROW_INVALID", "corpus row text is not a string"))
            continue
        if sha256(text.encode("utf-8")) != row.get("text_digest"):
            issues.append(
                CorpusIssue("DIGEST_MISMATCH", "corpus row text digest mismatch")
            )
        order = (
            int(row.get("source_ordinal", -1)),
            str(row.get("logical_locator", "")),
            int(row.get("chunk_index", -1)),
        )
        if previous is not None and order < previous:
            issues.append(
                CorpusIssue("ORDER_INVALID", "corpus rows are not canonically ordered")
            )
        previous = order
    semantic = manifest.get("semantic_identity")
    if not isinstance(semantic, str) or not semantic.startswith("sha256:"):
        issues.append(
            CorpusIssue("IDENTITY_INVALID", "manifest semantic identity is invalid")
        )
    else:
        calculated = _semantic_from_rows(root, rows, manifest)
        if calculated != semantic:
            issues.append(
                CorpusIssue(
                    "IDENTITY_MISMATCH", "manifest semantic identity does not validate"
                )
            )
    if sum(int(item.get("record_count", -1)) for item in inventory) != len(rows):
        issues.append(
            CorpusIssue(
                "INVENTORY_MISMATCH", "shard inventory count does not match rows"
            )
        )
    return CorpusValidationResult(
        "pass" if not issues else "fail",
        tuple(issues),
        semantic_identity=semantic if isinstance(semantic, str) else None,
        num_examples=len(rows),
        num_sources=len({row.get("source_id") for row in rows}),
    )


def inspect_corpus_artifact_v2(path: str | Path) -> CorpusInspection:
    root = Path(path).resolve()
    result = validate_corpus_artifact_v2(root)
    if not result.ok:
        raise ValueError(
            "corpus artifact validation failed: " + "; ".join(result.blockers)
        )
    manifest = _json(root / "corpus_manifest.json")
    return CorpusInspection(
        root,
        str(manifest["schema_version"]),
        str(manifest["semantic_identity"]),
        result.num_examples,
        result.num_sources,
        len(_json(root / "shard_inventory.json")),
        result,
    )


def open_verified_corpus(path: str | Path) -> VerifiedCorpusReader:
    result = validate_corpus_artifact_v2(path)
    if not result.ok:
        raise ValueError(
            "corpus artifact validation failed: " + "; ".join(result.blockers)
        )
    return VerifiedCorpusReader(path)


def _semantic_from_rows(
    root: Path, rows: list[dict[str, Any]], manifest: dict[str, Any]
) -> str:
    from radjax_tome.corpora.identity import corpus_semantic_identity
    from radjax_tome.corpora.records import CanonicalCorpusRecord

    records = (
        CanonicalCorpusRecord(
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
        for row in rows
    )
    return corpus_semantic_identity(
        policy=dict(manifest.get("policy", {})),
        tokenizer_binding_digest=str(manifest.get("tokenizer_binding_digest")),
        records=records,
        source_declarations=list(manifest.get("source_declarations", [])),
    )


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "CorpusInspection",
    "CorpusIssue",
    "CorpusValidationResult",
    "VerifiedCorpusReader",
    "inspect_corpus_artifact_v2",
    "open_verified_corpus",
    "validate_corpus_artifact_v2",
]
