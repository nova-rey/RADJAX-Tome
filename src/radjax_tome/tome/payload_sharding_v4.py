"""Additive writer for the proposed streaming Tome v4 payload layout.

This module deliberately does not participate in the legacy v3 packaging path.
It accepts already-selected semantic records at the artifact boundary, emits the
portable v4 layout transactionally, and leaves M4 delivery staging untouched.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import sqlite3
import tarfile
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from radjax_tome.io.json import read_json_object
from radjax_tome.tome.canonical_artifact import derive_tome_semantic_identity

PREFIX = "sha256:"
_SEQUENCE_PREFIX = b'{"records":['
_SEQUENCE_SUFFIX = b'],"schema_version":"selected_exemplar_payload_sequence_v1"}'
_SEMANTIC_FIELDS = frozenset(
    {
        "selected_example_id",
        "selected_position",
        "selected_score",
        "score_selected_position_entropy",
        "score_top_token_id",
        "source_shard_id",
        "source_row",
        "source_position",
        "source_score",
        "source_top_token_id",
        "source_score_policy",
        "payload_ref",
        "selected_policy",
        "source_delivery_path",
        "top_token_ids",
        "top_log_probs",
        "top_probs",
        "top_selection_mask",
        "effective_top_k",
        "top_mass",
        "tail_mass",
        "bucket_masses",
        "teacher_entropy",
        "sequence_length",
        "vocab_size",
        "num_buckets",
        "dynamic_top_k",
        "dynamic_mass_threshold",
        "dynamic_top_k_max",
        "top_k_saturated",
        "long_tail_class",
        "long_tail_warnings",
        "effective_top_k_fraction_of_vocab",
        "semantic_tail_tag",
        "selected_board",
        "corridor_mode_id",
        "corridor_fingerprint_id",
        "corridor_assignment_status",
    }
)

# The legacy Path-B tree deliberately retains runtime and diagnostic evidence
# for resume/debugging.  A ``student`` v4 payload is the ordinary distributable
# M7 surface, however, and must not bind that machine-local evidence into its
# governed bytes.  Keep the full evidence for the explicit debug profile while
# projecting only the stable semantic/core profile here.
_V4_CANONICAL_RUNTIME_TIMESTAMP = "1970-01-01T00:00:00+00:00"
_V4_STUDENT_NONSEMANTIC_MEMBERS = (
    "c6/",
    "reports/",
    "delivery_report.json",
    "progress_log.jsonl",
    "run_manifest.json",
    "run_plan.json",
    "production_build_report.json",
    "selected_linkage_audit.json",
    "leaderboards/leaderboard_report.json",
    "leaderboards/long_tail_uncertainty.json",
    "leaderboards/perverse_tail_diagnostic.json",
)
_V4_RUNTIME_KEYS = frozenset(
    {
        "created_at",
        "generated_at",
        "completed_at",
        "updated_at",
        "started_at",
        "finished_at",
    }
)
_V4_REQUIRED_TIMESTAMP_MEMBERS = frozenset({"metadata.json", "teacher_manifest.json"})


@dataclass(frozen=True)
class ShardedTomeV4Result:
    """A completed v4 package root and its layout-independent identity."""

    root: Path
    semantic_identity_digest: str
    selected_count: int
    shard_count: int


def write_sharded_tome_v4_from_legacy_artifact(
    source: Path,
    output: Path,
    *,
    profile: str = "student",
    payload_records_per_shard: int = 128,
    overwrite: bool = False,
) -> ShardedTomeV4Result:
    """Adapt a complete native artifact without mutating its v3 contents."""
    identity = derive_tome_semantic_identity(source)
    return write_sharded_tome_v4(
        _legacy_selected_records(source),
        output,
        training_contract=identity.training_contract,
        authority=identity.authority,
        nonselected_training_payload=_nonselected_payload(identity.to_dict()),
        profile=profile,
        payload_records_per_shard=payload_records_per_shard,
        overwrite=overwrite,
        allow_compact_without_mask=True,
    )


def package_legacy_artifact_as_sharded_tome_v4(
    source: Path,
    output: Path,
    *,
    profile: str = "student",
    payload_records_per_shard: int = 128,
    overwrite: bool = False,
) -> ShardedTomeV4Result:
    """Package one complete legacy artifact behind the additive v4 boundary.

    Legacy selected wrappers and manifests are intentionally not copied: v4
    emits their replacement payload grammar and its own acyclic manifest graph.
    The full-debug profile retains the remaining source evidence as raw
    inventory members.  The student profile retains only stable semantic/core
    members and deterministically projects runtime/path-only metadata; private
    diagnostics remain in the legacy producer tree rather than changing M7
    bytes.
    """
    if output.exists() and not overwrite:
        raise ValueError(f"output already exists: {output}")
    identity = derive_tome_semantic_identity(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".radjax-v4-package-", dir=output.parent
    ) as tmp:
        stage = Path(tmp) / output.name
        stage.mkdir()
        _copy_legacy_profile_members(source, stage, profile=profile)
        result = _write_directory(
            _legacy_selected_records(source),
            stage,
            training_contract=identity.training_contract,
            authority=identity.authority,
            nonselected_training_payload=_nonselected_payload(identity.to_dict()),
            profile=profile,
            capacity=payload_records_per_shard,
            allow_compact_without_mask=True,
        )
        _validate_publishable(stage)
        if output.exists():
            shutil.rmtree(output)
        os.replace(stage, output)
    return ShardedTomeV4Result(
        output,
        result.semantic_identity_digest,
        result.selected_count,
        result.shard_count,
    )


def pack_sharded_tome_v4(
    root: Path,
    output: Path,
    *,
    compression: str = "gz",
    overwrite: bool = False,
) -> Path:
    """Create the deterministic v4 transport without changing legacy bundles."""
    if compression not in {"none", "gz"}:
        raise ValueError("v4 compression must be one of: none, gz")
    if output.exists() and not overwrite:
        raise ValueError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(output)
    try:
        _write_archive(root, temporary, compression=compression)
        _validate_publishable(temporary)
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return output


def write_sharded_tome_v4(
    records: Iterable[dict[str, Any]],
    output: Path,
    *,
    training_contract: dict[str, Any],
    authority: dict[str, Any],
    nonselected_training_payload: tuple[dict[str, str], ...] = (),
    profile: str = "student",
    payload_records_per_shard: int = 128,
    overwrite: bool = False,
    allow_compact_without_mask: bool = False,
) -> ShardedTomeV4Result:
    """Write a compact, count-sharded v4 directory without buffering a shard.

    Record order is the caller's selected order.  A shard boundary is *only*
    ``payload_records_per_shard``; one encoded record plus fixed I/O state is
    retained at a time.  The target is atomically replaced only after every
    raw and semantic reference has been written.
    """
    if profile not in {"student", "full_debug_provenance", "unpacked"}:
        raise ValueError("unsupported v4 package profile")
    if (
        not isinstance(payload_records_per_shard, int)
        or isinstance(payload_records_per_shard, bool)
        or payload_records_per_shard < 1
    ):
        raise ValueError("payload_records_per_shard must be a positive integer")
    if output.exists() and not overwrite:
        raise ValueError(f"output already exists: {output}")
    if not isinstance(training_contract, dict) or not isinstance(authority, dict):
        raise ValueError("training_contract and authority must be objects")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".radjax-v4-", dir=output.parent) as tmp:
        stage = Path(tmp) / output.name
        stage.mkdir()
        result = _write_directory(
            records,
            stage,
            training_contract=training_contract,
            authority=authority,
            nonselected_training_payload=nonselected_training_payload,
            profile=profile,
            capacity=payload_records_per_shard,
            allow_compact_without_mask=allow_compact_without_mask,
        )
        _validate_publishable(stage)
        if output.exists():
            shutil.rmtree(output)
        os.replace(stage, output)
    return ShardedTomeV4Result(
        root=output,
        semantic_identity_digest=result.semantic_identity_digest,
        selected_count=result.selected_count,
        shard_count=result.shard_count,
    )


def _temporary_sibling(output: Path) -> Path:
    """Reserve an unpublished archive path on the final filesystem.

    ``os.replace`` is only atomic when the staged archive is a sibling of its
    final path.  Keeping the temporary name private also ensures a failed
    pack cannot leave a truncated final archive visible to consumers.
    """
    descriptor, name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(descriptor)
    return Path(name)


def _write_archive(root: Path, output: Path, *, compression: str) -> None:
    """Write a deterministic archive to an unpublished staging file."""
    with output.open("wb") as raw:
        stream: Any = (
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
            if compression == "gz"
            else raw
        )
        try:
            with tarfile.open(
                fileobj=stream, mode="w", format=tarfile.USTAR_FORMAT
            ) as archive:
                for source in sorted(
                    (path for path in root.rglob("*") if path.is_file()),
                    key=lambda path: _archive_member_order(
                        path.relative_to(root).as_posix()
                    ),
                ):
                    relative = source.relative_to(root).as_posix()
                    info = tarfile.TarInfo(relative)
                    cover = (
                        _archive_cover_bytes(source, compression=compression)
                        if relative == "cover_page.json"
                        else None
                    )
                    info.size = (
                        len(cover) if cover is not None else source.stat().st_size
                    )
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mode = 0o644
                    if cover is not None:
                        import io

                        archive.addfile(info, io.BytesIO(cover))
                    else:
                        with source.open("rb") as handle:
                            archive.addfile(info, handle)
        finally:
            if compression == "gz":
                stream.close()


def _validate_publishable(path: Path) -> None:
    """Fail closed at the publication boundary using Contract's authority.

    This is deliberately a narrow validation-boundary import rather than a
    Tome packaging dependency: Contract owns the portable v4 semantics while
    Tome owns staging and deterministic bytes.  Nothing is promoted before
    the authoritative portable validator has accepted the staged artifact.
    """
    from radjax_contract.tome import validate_streaming_tome

    result = validate_streaming_tome(path, strict=True)
    if not result.ok:
        errors = ", ".join(result.errors) or "portable validation failed"
        raise ValueError(f"staged v4 Tome failed Contract validation: {errors}")


def _write_directory(
    records: Iterable[dict[str, Any]],
    root: Path,
    *,
    training_contract: dict[str, Any],
    authority: dict[str, Any],
    nonselected_training_payload: tuple[dict[str, str], ...],
    profile: str,
    capacity: int,
    allow_compact_without_mask: bool = False,
) -> ShardedTomeV4Result:
    selected_dir = root / "selected_exemplars"
    shard_dir = selected_dir / "shards"
    shard_dir.mkdir(parents=True)
    index_path = selected_dir / "payload-index.jsonl"
    shard_index_path = selected_dir / "payload-shards.jsonl"
    sequence = _SequenceHasher()
    shard_index_handle = shard_index_path.open("wb")
    shard_count = 0
    seen_path = root / ".selected-logical-ids.sqlite3"
    seen = sqlite3.connect(seen_path)
    seen.execute("CREATE TABLE ids (logical_id TEXT PRIMARY KEY)")
    selected_count = 0
    shard_handle = None
    shard_hasher: _SequenceHasher | None = None
    shard_start = 0
    shard_rows = 0
    shard_index = -1
    index_handle = index_path.open("wb")
    try:
        for record in records:
            _validate_semantic_record_for_write(
                record, allow_compact_without_mask=allow_compact_without_mask
            )
            _assert_finite(record)
            encoded = _canonical_bytes(record)
            semantic_digest = _digest_bytes(encoded)
            logical_id = _digest_json(
                {
                    "selected_example_id": record.get("selected_example_id"),
                    "selected_position": record.get("selected_position"),
                }
            )
            try:
                seen.execute("INSERT INTO ids VALUES (?)", (logical_id,))
            except sqlite3.IntegrityError as exc:
                raise ValueError("duplicate selected logical identifier") from exc
            if shard_handle is None or shard_rows == capacity:
                if shard_handle is not None:
                    shard_handle.close()
                    shard_index_handle.write(
                        _canonical_bytes(
                            _shard_entry(
                                root,
                                shard_index,
                                shard_start,
                                shard_rows,
                                shard_hasher.finish()
                                if shard_hasher is not None
                                else "",
                            )
                        )
                        + b"\n"
                    )
                    shard_count += 1
                shard_index += 1
                shard_start = selected_count
                shard_rows = 0
                shard_hasher = _SequenceHasher()
                relative = f"selected_exemplars/shards/shard-{shard_index:05d}.jsonl"
                shard_handle = (root / relative).open("wb")
            assert shard_handle is not None and shard_hasher is not None
            shard_handle.write(encoded + b"\n")
            sequence_item = {
                "logical_id": logical_id,
                "payload_semantic_digest": semantic_digest,
            }
            sequence.add(sequence_item)
            shard_hasher.add(sequence_item)
            raw_digest = _digest_bytes(encoded)
            index_handle.write(
                _canonical_bytes(
                    {
                        "logical_id": logical_id,
                        "selected_example_id": record.get("selected_example_id"),
                        "selected_position": record.get("selected_position"),
                        "selection_index": selected_count,
                        "shard_id": shard_index,
                        "row": shard_rows,
                        "payload_sha256": raw_digest,
                        "payload_semantic_digest": semantic_digest,
                        # Bound after the shard closes; index is rewritten below.
                        "shard_sha256": "",
                    }
                )
                + b"\n"
            )
            selected_count += 1
            shard_rows += 1
        if shard_handle is not None:
            shard_handle.close()
            shard_index_handle.write(
                _canonical_bytes(
                    _shard_entry(
                        root,
                        shard_index,
                        shard_start,
                        shard_rows,
                        shard_hasher.finish() if shard_hasher is not None else "",
                    )
                )
                + b"\n"
            )
            shard_count += 1
    finally:
        index_handle.close()
        shard_index_handle.close()
        if shard_handle is not None and not shard_handle.closed:
            shard_handle.close()
        seen.close()
        seen_path.unlink(missing_ok=True)

    # Bind raw shard hashes into JSONL without retaining payload records.
    _rewrite_index_shard_hashes(index_path, shard_index_path)
    sequence_digest = sequence.finish()
    identity = {
        "schema_version": "radjax_tome_semantic_identity_v2",
        "payload_sequence_digest": sequence_digest,
        "selected_count": selected_count,
        "nonselected_training_payload": list(nonselected_training_payload),
        "training_contract": training_contract,
        "authority": authority,
    }
    identity["semantic_digest"] = _digest_json(identity)
    layout = {
        "schema_version": "radjax_tome_payload_layout_v1",
        "layout_version": "selected_payload_shards_v1",
        "payload_index": {
            "path": "selected_exemplars/payload-index.jsonl",
            "sha256": _digest_file(index_path),
            "size_bytes": index_path.stat().st_size,
            "record_count": selected_count,
            "schema_version": "radjax_tome_payload_index_v2",
        },
        "shard_index": {
            "path": "selected_exemplars/payload-shards.jsonl",
            "sha256": _digest_file(shard_index_path),
            "size_bytes": shard_index_path.stat().st_size,
            "record_count": shard_count,
            "schema_version": "radjax_tome_payload_shard_index_v1",
        },
        "sequence_digest": sequence_digest,
        "selected_count": selected_count,
        "payload_records_per_shard": capacity,
    }
    _write_json(root / "selected_exemplars" / "payload-layout.json", layout)
    if profile == "full_debug_provenance":
        _write_json(
            root / "provenance" / "full-debug-receipt.json",
            {
                "schema_version": "radjax_tome_full_debug_receipt_v1",
                "profile": profile,
                "claim": "additional_nontraining_provenance",
            },
        )
    _write_manifest_graph(root, profile=profile, identity=identity)
    return ShardedTomeV4Result(
        root, identity["semantic_digest"], selected_count, shard_count
    )


def _validate_semantic_record_for_write(
    record: Any, *, allow_compact_without_mask: bool = False
) -> None:
    """Reject a malformed semantic payload while it is still staging-only.

    Full package validation remains Contract-owned at final publication.  This
    local boundary only prevents an obviously incomplete record from becoming
    a successfully returned artifact before that authoritative validation can
    run.
    """
    required_fields = _SEMANTIC_FIELDS
    if (
        allow_compact_without_mask
        and isinstance(record, dict)
        and ("top_selection_mask" not in record)
    ):
        required_fields = required_fields - {"top_selection_mask"}
    elif isinstance(record, dict) and record.get("storage_flavor") in {
        "compact_k_monolithic",
        "compact_k_immutable_body",
    }:
        required_fields = required_fields - {"top_selection_mask"}
    if not isinstance(record, dict) or not required_fields <= set(record):
        raise ValueError("record is missing a required semantic field")
    allowed = _SEMANTIC_FIELDS | {"opaque_extensions"}
    if set(record) - allowed:
        raise ValueError("record contains an undeclared semantic field")


def _archive_cover_bytes(source: Path, *, compression: str) -> bytes:
    """Return the transport-specific cover without changing the directory.

    The cover is deliberately excluded from the inventory graph, so replacing
    only this transport declaration does not alter package-member integrity or
    logical identity.  ``none`` is the deterministic ``rtome`` transport.
    """
    cover = read_json_object(source)
    package = cover.get("package")
    if not isinstance(package, dict):
        raise ValueError("v4 cover package section is invalid")
    package = dict(package)
    package["transport"] = "tgz" if compression == "gz" else "rtome"
    cover = dict(cover)
    cover["package"] = package
    return _canonical_bytes(cover)


def _archive_member_order(relative: str) -> tuple[int, str]:
    """Put the acyclic manifest prelude before inventory-governed members."""
    prelude = {
        "cover_page.json": 0,
        "manifests/content-manifest-header.json": 1,
        "manifests/content-manifest-inventory.jsonl": 2,
    }
    return prelude.get(relative, 3), relative


def _legacy_selected_records(source: Path) -> Iterable[dict[str, Any]]:
    """Yield legacy selected records in native shard order at the v4 boundary."""
    allowed = _SEMANTIC_FIELDS | {"opaque_extensions"}
    for path in sorted(
        (source / "selected_exemplars").glob("selected-exemplars-*.json")
    ):
        document = read_json_object(path)
        records = document.get("selected_exemplars")
        if not isinstance(records, list):
            raise ValueError(f"legacy selected payload shard is invalid: {path.name}")
        for record in records:
            required_fields = _SEMANTIC_FIELDS
            if isinstance(record, dict) and record.get("storage_flavor") in {
                "compact_k_monolithic",
                "compact_k_immutable_body",
            }:
                required_fields = required_fields - {"top_selection_mask"}
            if not isinstance(record, dict) or not required_fields <= set(record):
                raise ValueError(f"legacy selected payload is incomplete: {path.name}")
            # Native v3 delivery receipts may contain nonsemantic staging and
            # linkage details such as ``payload_hash``.  The versioned v4
            # adapter projects the closed public semantic surface explicitly;
            # direct v4 writers remain strict about undeclared fields.
            projected = {key: record[key] for key in record if key in allowed}
            if "top_selection_mask" not in projected:
                projected["top_selection_mask"] = [True] * len(
                    projected["top_token_ids"]
                )
            yield projected


def _nonselected_payload(identity: dict[str, Any]) -> tuple[dict[str, str], ...]:
    """Carry v3 semantic material except the physical selected-wrapper files."""
    entries = identity.get("training_payload")
    if not isinstance(entries, list):
        raise ValueError("legacy semantic identity is missing training payload")
    selected = tuple(
        {
            "logical_id": str(entry["logical_id"]),
            "semantic_digest": str(entry["semantic_digest"]),
        }
        for entry in entries
        if isinstance(entry, dict)
        and not str(entry.get("logical_id", "")).startswith("selected_exemplars/")
    )
    if [entry["logical_id"] for entry in selected] != sorted(
        entry["logical_id"] for entry in selected
    ):
        raise ValueError("legacy semantic identity payload order is invalid")
    return selected


def _copy_legacy_profile_members(
    source: Path, destination: Path, *, profile: str
) -> None:
    if profile not in {"student", "full_debug_provenance", "unpacked"}:
        raise ValueError("unsupported v4 package profile")
    for path in sorted(member for member in source.rglob("*") if member.is_file()):
        relative = path.relative_to(source).as_posix()
        if relative == "cover_page.json" or relative.startswith(
            ("manifests/", "selected_exemplars/")
        ):
            continue
        if relative.startswith(".staging-") or relative.startswith(".selected-"):
            continue
        if profile == "student" and relative.startswith("shards/"):
            continue
        if profile == "student" and _is_v4_student_nonsemantic_member(relative):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if profile == "student":
            target.write_bytes(
                _v4_profile_member_bytes(
                    path,
                    relative=relative,
                    source=source,
                )
            )
        else:
            shutil.copyfile(path, target)


def _is_v4_student_nonsemantic_member(relative: str) -> bool:
    """Return whether a legacy member is private runtime/diagnostic evidence.

    These files remain available in the legacy producer artifact and in the
    explicit ``full_debug_provenance`` profile.  They are not part of the
    selected Student M7 surface: their paths, wall-clock fields, and host
    measurements are not governed behavior and would make the raw v4 payload
    depend on the output directory or process schedule.
    """

    return any(
        relative == prefix or relative.startswith(prefix)
        for prefix in _V4_STUDENT_NONSEMANTIC_MEMBERS
    )


def _v4_profile_member_bytes(path: Path, *, relative: str, source: Path) -> bytes:
    """Write one v4 profile member with deterministic nonsemantic metadata.

    The transformation is part of the normal v4 writer, not a post-production
    fixture repair.  JSON key order and separators use the same canonical
    encoder as the generated v4 graph.  Runtime timestamps are removed from
    nested diagnostic objects; the two historical core manifests that require
    a timestamp retain a fixed epoch marker.  Absolute machine-local paths are
    reduced to source-relative paths (or a stable external basename) so they
    cannot enter a governed package identity.
    """

    if path.suffix not in {".json", ".jsonl"}:
        return path.read_bytes()
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        normalized = _v4_profile_value(payload, source=source)
        if relative in _V4_REQUIRED_TIMESTAMP_MEMBERS:
            if not isinstance(normalized, dict):
                raise ValueError(f"v4 profile member is not an object: {relative}")
            normalized["created_at"] = _V4_CANONICAL_RUNTIME_TIMESTAMP
        return _canonical_bytes(normalized)

    lines: list[bytes] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        value = json.loads(line)
        lines.append(_canonical_bytes(_v4_profile_value(value, source=source)))
    return b"\n".join(lines) + (b"\n" if lines else b"")


def _v4_profile_value(value: Any, *, source: Path) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _v4_profile_value(item, source=source)
            for key, item in value.items()
            if str(key) not in _V4_RUNTIME_KEYS
        }
    if isinstance(value, list):
        return [_v4_profile_value(item, source=source) for item in value]
    if isinstance(value, str) and value.startswith("/"):
        path = Path(value)
        try:
            return (
                path.resolve(strict=False)
                .relative_to(source.resolve(strict=False))
                .as_posix()
            )
        except ValueError:
            return f"external/{path.name}"
    return value


def _rewrite_index_shard_hashes(path: Path, shard_index_path: Path) -> None:
    """Bind index rows from an on-disk shard index without a shard map in RAM."""
    database = path.with_suffix(".shards.sqlite3")
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE hashes (shard_id INTEGER PRIMARY KEY, digest TEXT)"
    )
    with shard_index_path.open(encoding="utf-8") as source:
        for line in source:
            entry = json.loads(line)
            connection.execute(
                "INSERT INTO hashes VALUES (?, ?)", (entry["shard_id"], entry["sha256"])
            )
    temporary = path.with_suffix(".tmp")
    with path.open(encoding="utf-8") as source, temporary.open("wb") as target:
        for line in source:
            row = json.loads(line)
            digest = connection.execute(
                "SELECT digest FROM hashes WHERE shard_id = ?", (row["shard_id"],)
            ).fetchone()
            if digest is None:
                raise ValueError("payload index references an absent shard")
            row["shard_sha256"] = digest[0]
            target.write(_canonical_bytes(row) + b"\n")
    connection.close()
    database.unlink(missing_ok=True)
    os.replace(temporary, path)


def _shard_entry(
    root: Path, shard_id: int, first: int, count: int, semantic_digest: str
) -> dict[str, Any]:
    relative = f"selected_exemplars/shards/shard-{shard_id:05d}.jsonl"
    path = root / relative
    return {
        "shard_id": shard_id,
        "path": relative,
        "sha256": _digest_file(path),
        "size_bytes": path.stat().st_size,
        "first_selection_index": first,
        "last_selection_index": first + count - 1,
        "record_count": count,
        "semantic_digest": semantic_digest,
    }


def _write_manifest_graph(
    root: Path, *, profile: str, identity: dict[str, Any]
) -> None:
    ignored = {
        "cover_page.json",
        "manifests/content-manifest-header.json",
        "manifests/content-manifest-inventory.jsonl",
    }
    members = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() not in ignored
    )
    inventory_path = root / "manifests" / "content-manifest-inventory.jsonl"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    with inventory_path.open("wb") as handle:
        for relative in members:
            path = root / relative
            handle.write(
                _canonical_bytes(
                    {
                        "path": relative,
                        "sha256": _digest_file(path),
                        "size_bytes": path.stat().st_size,
                        "classification": (
                            "diagnostic"
                            if relative.startswith("provenance/")
                            else "training_critical"
                        ),
                        "training_authoritative": not relative.startswith(
                            "provenance/"
                        ),
                    }
                )
                + b"\n"
            )
    header = {
        "schema_version": "tome_content_manifest_header_v3",
        "profile": profile,
        "semantic_identity_digest": identity["semantic_digest"],
        "inventory_path": "manifests/content-manifest-inventory.jsonl",
        "inventory_sha256": _digest_file(inventory_path),
        "inventory_size_bytes": inventory_path.stat().st_size,
        "entry_count": len(members),
    }
    header_path = root / "manifests" / "content-manifest-header.json"
    _write_json(header_path, header)
    cover = {
        "schema_version": "radjax_tome_cover_v4",
        "identity": identity,
        "training": identity["training_contract"],
        "package": {"profile": profile, "transport": "directory"},
        "manifests": {
            "header": {
                "path": "manifests/content-manifest-header.json",
                "sha256": _digest_file(header_path),
                "size_bytes": header_path.stat().st_size,
                "schema_version": "tome_content_manifest_header_v3",
            }
        },
        "authority": identity["authority"],
        "provenance": {},
        "validation": {},
    }
    _write_json(root / "cover_page.json", cover)


class _SequenceHasher:
    def __init__(self) -> None:
        self._hash = hashlib.sha256()
        self._hash.update(_SEQUENCE_PREFIX)
        self._first = True

    def add(self, value: dict[str, str]) -> None:
        if not self._first:
            self._hash.update(b",")
        self._hash.update(_canonical_bytes(value))
        self._first = False

    def finish(self) -> str:
        self._hash.update(_SEQUENCE_SUFFIX)
        return PREFIX + self._hash.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _digest_bytes(value: bytes) -> str:
    return PREFIX + hashlib.sha256(value).hexdigest()


def _digest_json(value: Any) -> str:
    return _digest_bytes(_canonical_bytes(value))


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 16):
            digest.update(block)
    return PREFIX + digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value))


def _assert_finite(value: Any) -> None:
    if isinstance(value, float) and (
        value != value or value in {float("inf"), float("-inf")}
    ):
        raise ValueError("payload contains a non-finite number")
    if isinstance(value, dict):
        for item in value.values():
            _assert_finite(item)
    elif isinstance(value, list):
        for item in value:
            _assert_finite(item)
