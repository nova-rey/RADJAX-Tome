"""Validation and inspection for self-describing corpus v2 artifacts."""

from __future__ import annotations

import hashlib
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
        "duplicate_provenance.jsonl",
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
        binding = _json(root / "language_tokenizer_binding_v1.json")
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
    issues.extend(_validate_member_inventory(root, cover))
    if not isinstance(inventory, list):
        issues.append(
            CorpusIssue("INVENTORY_INVALID", "shard inventory must be an array")
        )
        return CorpusValidationResult("fail", tuple(issues))
    if not isinstance(binding, dict):
        issues.append(
            CorpusIssue("BINDING_INVALID", "tokenizer binding must be an object")
        )
    else:
        binding_payload = {
            "tokenizer": binding.get("tokenizer"),
            "canonical_inventory_digest": binding.get("canonical_inventory_digest"),
            "vocabulary": binding.get("vocabulary"),
        }
        expected_binding = sha256(
            json.dumps(
                binding_payload,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        )
        if binding.get("canonical_binding_digest") != expected_binding:
            issues.append(
                CorpusIssue("BINDING_INVALID", "tokenizer binding digest mismatch")
            )
        if manifest.get("tokenizer_binding_digest") != binding.get(
            "canonical_binding_digest"
        ):
            issues.append(
                CorpusIssue("BINDING_MISMATCH", "manifest tokenizer binding mismatch")
            )
    row_count = 0
    source_ids: set[str] = set()
    previous: tuple[int, str, int] | None = None

    def checked_records():
        nonlocal row_count, previous
        from radjax_tome.corpora.records import CanonicalCorpusRecord

        for row in VerifiedCorpusReader(root):
            row_count += 1
            source_ids.add(str(row.get("source_id")))
            expected_id = f"corpus_{row_count:09d}"
            if row.get("example_id") != expected_id:
                issues.append(
                    CorpusIssue(
                        "ORDER_INVALID", "example IDs are not contiguous and ordered"
                    )
                )
            text = row.get("text")
            if not isinstance(text, str):
                issues.append(
                    CorpusIssue("ROW_INVALID", "corpus row text is not a string")
                )
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
                    CorpusIssue(
                        "ORDER_INVALID", "corpus rows are not canonically ordered"
                    )
                )
            previous = order
            yield CanonicalCorpusRecord(
                example_id=str(row["example_id"]),
                source_id=str(row["source_id"]),
                source_ordinal=int(row["source_ordinal"]),
                logical_locator=str(row["logical_locator"]),
                chunk_index=int(row["chunk_index"]),
                chunk_count=int(row.get("chunk_count", 1)),
                text=text,
                text_digest=str(row["text_digest"]),
                source_digest=str(row.get("source_digest", "")),
            )

    try:
        calculated = _semantic_from_records(checked_records(), manifest)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        issues.append(CorpusIssue("SHARD_INVALID", str(exc)))
        return CorpusValidationResult("fail", tuple(issues))
    issues.extend(_validate_duplicate_provenance(root))
    expected_count = int(manifest.get("num_examples", -1))
    if expected_count != row_count:
        issues.append(
            CorpusIssue(
                "COUNT_MISMATCH", "manifest example count does not match shards"
            )
        )
    semantic = manifest.get("semantic_identity")
    if not isinstance(semantic, str) or not semantic.startswith("sha256:"):
        issues.append(
            CorpusIssue("IDENTITY_INVALID", "manifest semantic identity is invalid")
        )
    else:
        if calculated != semantic:
            issues.append(
                CorpusIssue(
                    "IDENTITY_MISMATCH", "manifest semantic identity does not validate"
                )
            )
    if sum(int(item.get("record_count", -1)) for item in inventory) != row_count:
        issues.append(
            CorpusIssue(
                "INVENTORY_MISMATCH", "shard inventory count does not match rows"
            )
        )
    return CorpusValidationResult(
        "pass" if not issues else "fail",
        tuple(issues),
        semantic_identity=semantic if isinstance(semantic, str) else None,
        num_examples=row_count,
        num_sources=len(source_ids),
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


def _semantic_from_records(records: Any, manifest: dict[str, Any]) -> str:
    from radjax_tome.corpora.identity import corpus_semantic_identity

    return corpus_semantic_identity(
        policy=dict(manifest.get("policy", {})),
        tokenizer_binding_digest=str(manifest.get("tokenizer_binding_digest")),
        records=records,
        source_declarations=list(manifest.get("source_declarations", [])),
    )


def _validate_duplicate_provenance(root: Path) -> list[CorpusIssue]:
    """Cross-check the complete duplicate sidecar against canonical winners."""
    issues: list[CorpusIssue] = []
    sidecar = root / "duplicate_provenance.jsonl"
    try:
        handle = sidecar.open(encoding="utf-8")
    except OSError as exc:
        return [CorpusIssue("PROVENANCE_INVALID", str(exc), sidecar.name)]

    pending: dict[str, Any] | None = None
    line_number = 0

    def next_entry() -> dict[str, Any] | None:
        nonlocal line_number
        line = handle.readline()
        if not line:
            return None
        line_number += 1
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(
                CorpusIssue(
                    "PROVENANCE_INVALID",
                    f"duplicate provenance line {line_number} is malformed: {exc}",
                    sidecar.name,
                )
            )
            return None
        if (
            not isinstance(value, dict)
            or set(value) != {"winner", "locator"}
            or isinstance(value.get("winner"), bool)
            or not isinstance(value.get("winner"), int)
            or value["winner"] < 1
            or not isinstance(value.get("locator"), str)
            or not value["locator"]
        ):
            issues.append(
                CorpusIssue(
                    "PROVENANCE_INVALID",
                    f"duplicate provenance line {line_number} has invalid fields",
                    sidecar.name,
                )
            )
            return None
        return value

    try:
        pending = next_entry()
        for row_number, row in enumerate(VerifiedCorpusReader(root), 1):
            duplicate_count = row.get("duplicate_count", 1)
            if (
                isinstance(duplicate_count, bool)
                or not isinstance(duplicate_count, int)
                or duplicate_count < 1
            ):
                issues.append(
                    CorpusIssue(
                        "PROVENANCE_INVALID",
                        f"winner {row_number} has invalid duplicate_count",
                    )
                )
                continue
            preview = row.get("duplicate_provenance", [])
            if not isinstance(preview, list) or any(
                not isinstance(item, str) or not item for item in preview
            ):
                issues.append(
                    CorpusIssue(
                        "PROVENANCE_INVALID",
                        f"winner {row_number} has invalid duplicate preview",
                    )
                )
                preview = []
            if duplicate_count == 1:
                if pending is not None:
                    issues.append(
                        CorpusIssue(
                            "PROVENANCE_MISMATCH",
                            "unexpected duplicate provenance for winner "
                            + str(pending.get("winner")),
                            sidecar.name,
                        )
                    )
                    pending = next_entry()
                continue
            entries: list[str] = []
            for _ in range(duplicate_count):
                if pending is None:
                    issues.append(
                        CorpusIssue(
                            "PROVENANCE_MISMATCH",
                            "duplicate provenance is truncated for winner "
                            + str(row_number),
                            sidecar.name,
                        )
                    )
                    break
                if pending["winner"] != row_number:
                    issues.append(
                        CorpusIssue(
                            "PROVENANCE_MISMATCH",
                            "duplicate provenance winner is out of order at line "
                            + str(line_number),
                            sidecar.name,
                        )
                    )
                    break
                entries.append(pending["locator"])
                pending = next_entry()
            owner = (
                f"{row.get('source_id')}:{row.get('logical_locator')}"
                f":{row.get('chunk_index')}"
            )
            if entries and entries[0] != owner:
                issues.append(
                    CorpusIssue(
                        "PROVENANCE_MISMATCH",
                        "duplicate provenance winner "
                        + str(row_number)
                        + " does not identify the canonical row",
                        sidecar.name,
                    )
                )
            if entries[1 : 1 + len(preview)] != preview:
                issues.append(
                    CorpusIssue(
                        "PROVENANCE_MISMATCH",
                        "duplicate provenance preview mismatch for winner "
                        + str(row_number),
                        sidecar.name,
                    )
                )
            truncated = len(preview) < duplicate_count - 1
            if row.get("duplicate_provenance_truncated", False) is not truncated:
                issues.append(
                    CorpusIssue(
                        "PROVENANCE_MISMATCH",
                        "duplicate provenance truncation flag mismatch for winner "
                        + str(row_number),
                        sidecar.name,
                    )
                )
        if pending is not None:
            issues.append(
                CorpusIssue(
                    "PROVENANCE_MISMATCH",
                    f"trailing duplicate provenance for winner {pending.get('winner')}",
                    sidecar.name,
                )
            )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        issues.append(CorpusIssue("PROVENANCE_INVALID", str(exc), sidecar.name))
    finally:
        handle.close()
    return issues


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_member_inventory(root: Path, cover: dict[str, Any]) -> list[CorpusIssue]:
    issues: list[CorpusIssue] = []
    declared = cover.get("members")
    if (
        cover.get("member_inventory_policy")
        != "sha256_size_role_schema_v1_excludes_cover_self_hash"
    ):
        issues.append(
            CorpusIssue("INVENTORY_INVALID", "unsupported cover inventory policy")
        )
    if not isinstance(declared, list):
        return issues + [
            CorpusIssue("INVENTORY_INVALID", "cover members must be an array")
        ]
    actual: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            issues.append(
                CorpusIssue("SYMLINK_MEMBER", f"symlinked corpus member: {path.name}")
            )
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if (
                relative != "corpus_cover.json"
                and not relative.startswith("journal/")
                and not relative.startswith(".")
            ):
                actual[relative] = path
    seen: set[str] = set()
    for entry in declared:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            issues.append(
                CorpusIssue("INVENTORY_INVALID", "cover member entry is invalid")
            )
            continue
        relative = entry["path"]
        if relative in seen or relative not in actual:
            issues.append(
                CorpusIssue(
                    "INVENTORY_INVALID", f"invalid or duplicate member: {relative}"
                )
            )
            continue
        seen.add(relative)
        path = actual[relative]
        if (
            not isinstance(entry.get("role"), str)
            or not isinstance(entry.get("media_type"), str)
            or not isinstance(entry.get("schema"), str)
        ):
            issues.append(
                CorpusIssue(
                    "INVENTORY_INVALID", f"incomplete member metadata: {relative}"
                )
            )
        if entry.get("size_bytes") != path.stat().st_size:
            issues.append(
                CorpusIssue("MEMBER_SIZE_MISMATCH", f"member size mismatch: {relative}")
            )
        if entry.get("sha256") != _file_digest(path):
            issues.append(
                CorpusIssue(
                    "MEMBER_DIGEST_MISMATCH", f"member digest mismatch: {relative}"
                )
            )
    for relative in sorted(set(actual) - seen):
        issues.append(
            CorpusIssue("UNDECLARED_MEMBER", f"undeclared corpus member: {relative}")
        )
    return issues


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


__all__ = [
    "CorpusInspection",
    "CorpusIssue",
    "CorpusValidationResult",
    "VerifiedCorpusReader",
    "inspect_corpus_artifact_v2",
    "open_verified_corpus",
    "validate_corpus_artifact_v2",
]
