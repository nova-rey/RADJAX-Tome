"""Tome-side adapter for the released RADJAX artifact Contract v3.

The publisher accepts only the finalized in-memory delivery handoff.  It never
parses the legacy selected-record files to reconstruct v3 semantics.
"""

from __future__ import annotations

import errno
import gzip
import hashlib
import json
import os
import shutil
import stat
import tarfile
import tempfile
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from radjax_contract.tome.v3.codec import (
    DOMAIN_LABELS,
    digest,
    logical_record_id,
    record_sequence_digest,
    semantic_root,
)
from radjax_contract.tome.v3.journal import (
    journal_restart_disposition_v3,
    validate_journal_state_v3,
)
from radjax_contract.tome.v3.models import JournalStateV3
from radjax_contract.tome.v3.schema import (
    CONTRACT_VERSION,
    RECORD_FIELDS,
    SEMANTIC_PROFILE_ID,
    normalize_semantic_record,
)
from radjax_contract.tome.v3.validation import validate_tome_artifact_v3

V3_SCHEMA = CONTRACT_VERSION
V3_CONTRACT_ID = "radjax_tome_artifact_contract"
V3_COVER_SCHEMA = "radjax_tome_cover_v5"
V3_SHARD_CAP = 128
PRIVATE_BINDING_SCHEMA = "radjax_tome_private_publication_binding_v2"
PRIVATE_TOPOLOGIES = frozenset({"canonical", "archive_only"})


@dataclass(frozen=True)
class FinalizedV3Handoff:
    """Immutable Tome-facing snapshot taken after late corridor finalization."""

    records: tuple[dict[str, Any], ...]
    authority: dict[str, Any]
    policy: dict[str, Any]
    selection_indexes: tuple[int, ...]
    shard_capacity: int


@dataclass(frozen=True)
class V3Publication:
    directory: Path
    archive: Path
    semantic_root: str
    authority_identity: str
    policy_identity: str
    record_count: int
    shard_count: int
    layout: str = SEMANTIC_PROFILE_ID


class V3ArchivePublicationError(RuntimeError):
    """Directory promotion succeeded but archive promotion did not."""

    def __init__(self, message: str, *, directory: Path, archive: Path):
        super().__init__(message)
        self.directory = directory
        self.archive = archive
        self.directory_promoted = directory.exists()
        self.archive_promoted = archive.exists()


class V3PublicationCrash(RuntimeError):
    """Test-only interruption which deliberately preserves private state.

    A real process crash does not execute Python ``finally`` cleanup.  The
    publisher exposes this narrow fault-injection exception so conformance
    tests can inspect the same durable staging/journal boundary without
    terminating the test runner.
    """


PublicationHook = Callable[[str], None]


def _raw(value: Any, *, jsonl: bool = False) -> bytes:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return encoded + (b"\n" if jsonl else b"")


def _digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _fsync_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_private_bytes(path: Path, raw: bytes) -> None:
    """Write a private receipt without following a replaced symlink."""

    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        remaining = memoryview(raw)
        while remaining:
            remaining = remaining[os.write(descriptor, remaining) :]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    """Durably publish a directory entry on filesystem-backed transports."""

    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _json_no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"v3_duplicate_journal_key:{key}")
        result[key] = value
    return result


def _rename_noreplace(source: Path, target: Path) -> None:
    """Publish on one filesystem without intentionally replacing a target.

    Directory rename has no portable no-replace primitive in Python.  The
    explicit existence check plus same-filesystem rename is therefore a
    required capability boundary: callers fail closed when the destination is
    already visible rather than replacing it.  Regular archive files use a
    hard-link promotion, which is atomic and no-replace on POSIX filesystems.
    """

    lock = target.with_name(f".{target.name}.v3-lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError("v3_publication_lock_present") from exc
    try:
        if os.path.lexists(target):
            raise FileExistsError(target)
        if source.is_file():
            try:
                os.link(source, target)
            except (FileExistsError, OSError) as exc:
                if isinstance(exc, FileExistsError) or os.path.lexists(target):
                    raise FileExistsError(target) from exc
                raise RuntimeError("v3_no_replace_file_promotion_unavailable") from exc
            source.unlink()
            return
        os.rename(source, target)
    finally:
        os.close(descriptor)
        lock.unlink(missing_ok=True)


def _private_transaction_prefixes(output_base: Path) -> tuple[str, str]:
    return (
        f".{output_base.name}.v3",
        f".{output_base.name}.v3-journal",
    )


def _private_lstat(path: Path, *, expected: str | None = None) -> os.stat_result:
    """Inspect a Tome-private path without following symlinks."""

    try:
        info = os.lstat(path)
    except FileNotFoundError as exc:
        raise ValueError(f"v3_private_path_missing:{path.name}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise ValueError(f"v3_private_symlink_rejected:{path.name}")
    if expected == "directory" and not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"v3_private_directory_required:{path.name}")
    if expected == "file" and not stat.S_ISREG(info.st_mode):
        raise ValueError(f"v3_private_file_required:{path.name}")
    return info


def _assert_private_tree_no_symlinks(root: Path) -> None:
    """Reject symlinks in a private transaction before any public mutation."""

    _private_lstat(root, expected="directory")
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in (*directories, *files):
            path = current_path / name
            info = _private_lstat(path)
            if stat.S_ISDIR(info.st_mode):
                continue


def _private_child(root: Path, name: str, *, expected: str) -> Path:
    """Return a direct private child after enforcing no-follow ownership."""

    if Path(name).name != name or name in {".", ".."}:
        raise ValueError("v3_private_path_shape_invalid")
    child = root / name
    _private_lstat(child, expected=expected)
    return child


def _reject_stale_private_transactions(output_base: Path) -> None:
    """Refuse to start over while an earlier interrupted transaction remains.

    The private state is intentionally not guessed at or deleted.  A caller
    must inspect/quarantine it or use the explicit archive-resume helper.  This
    prevents a new run from mixing authorities or accepting an unreceipted
    shard left by an earlier process.
    """

    prefixes = _private_transaction_prefixes(output_base)
    stale: list[Path] = []
    for path in output_base.parent.iterdir():
        if path.name.startswith(prefixes):
            info = os.lstat(path)
            if stat.S_ISLNK(info.st_mode):
                raise ValueError(f"v3_private_symlink_rejected:{path.name}")
            stale.append(path)
    stale.sort()
    if stale:
        names = ",".join(path.name for path in stale)
        raise ValueError("v3_stale_private_transaction_present:" + names)


def _publication_event(hook: PublicationHook | None, event: str) -> None:
    if hook is not None:
        hook(event)


def _tar_add_deterministic(tar: tarfile.TarFile, root: Path) -> None:
    """Add regular members with stable transport metadata and ordering."""

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        info = tar.gettarinfo(str(path), arcname=relative)
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        info.mtime = 0
        info.mode = 0o644
        with path.open("rb") as handle:
            tar.addfile(info, handle)


def _ref(
    path: str, raw: bytes, schema: str, *, record_count: int | None = None
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": path,
        "sha256": _digest(raw),
        "size_bytes": len(raw),
        "schema_version": schema,
    }
    if record_count is not None:
        result["record_count"] = record_count
    return result


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item") and callable(value.item):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _closed_record(
    record: Mapping[str, Any], payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Project the two finalized delivery views into Contract's closed record."""

    merged = dict(_jsonable(record))
    merged.update(_jsonable(payload))
    if "long_tail_warnings" not in merged:
        merged["long_tail_warnings"] = []
    merged.setdefault("semantic_tail_tag", "unknown_open_class_tail")
    merged.setdefault("long_tail_class", "normal")
    merged.setdefault(
        "effective_top_k_fraction_of_vocab",
        float(merged["effective_top_k"]) / float(merged["vocab_size"]),
    )
    merged.setdefault("top_k_saturated", False)
    merged.setdefault("dynamic_mass_threshold", 0.0)
    merged.setdefault("dynamic_top_k_max", int(merged["effective_top_k"]))
    dynamic_threshold = float(merged["dynamic_mass_threshold"])
    if dynamic_threshold > 0:
        merged["dynamic_top_k"] = {
            "kind": "mass_threshold",
            "threshold": dynamic_threshold,
            "max_k": int(merged["dynamic_top_k_max"]),
        }
    else:
        merged["dynamic_top_k"] = {"kind": "disabled_v1"}
    payload_ref = merged.get("payload_ref")
    if not isinstance(payload_ref, Mapping):
        raise ValueError("v3_record_payload_ref_missing")
    merged["payload_ref"] = {
        "kind": "source_coordinate",
        "source_shard_id": int(payload_ref["source_shard_id"]),
        "source_row": int(payload_ref["source_row"]),
        "source_position": int(payload_ref["source_position"]),
    }
    if "corridor_mode_id" in merged and not isinstance(merged["corridor_mode_id"], str):
        merged["corridor_mode_id"] = str(merged["corridor_mode_id"])
    missing = sorted(RECORD_FIELDS - set(merged))
    if missing:
        raise ValueError("v3_record_fields_missing:" + ",".join(missing))
    closed = {name: merged[name] for name in RECORD_FIELDS}
    return dict(normalize_semantic_record(closed))


def _authority_and_policy(
    config: Any, context: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    authorities = dict(context.get("authorities") or {})
    provenance = json.loads(
        Path(config.teacher_model_provenance_path).read_text(encoding="utf-8")
    )

    def required_digest(value: Any, name: str) -> str:
        if isinstance(value, str) and value.startswith("sha256:") and len(value) == 71:
            return value
        raise ValueError(f"v3_authority_missing:{name}")

    corpus = required_digest(_manifest_hash(config.corpus_manifest_path), "corpus")
    teacher = required_digest(provenance.get("weights_hash"), "teacher")
    tokenizer = required_digest(
        provenance.get("tokenizer_hash"), "tokenizer_vocabulary"
    )
    score = required_digest(authorities.get("score_pass_authority_hash"), "score_pass")
    selection = required_digest(
        authorities.get("selection_integration_config_hash"), "selection"
    )
    delivery = required_digest(authorities.get("score_pass_authority_hash"), "delivery")
    authority = {
        "schema_version": "radjax_tome_semantic_authority_v1",
        "contract_version": V3_SCHEMA,
        "semantic_profile_id": SEMANTIC_PROFILE_ID,
        "entries": [
            {
                "role": role,
                "schema_id": f"radjax.tome.authority.{role}.v1",
                "identity": identity,
            }
            for role, identity in (
                ("corpus", corpus),
                ("delivery", delivery),
                ("score_pass", score),
                ("selection", selection),
                ("teacher", teacher),
                ("tokenizer_vocabulary", tokenizer),
            )
        ],
    }
    policy = {
        "schema_version": "radjax_tome_behavioral_policy_v1",
        "contract_version": V3_SCHEMA,
        "semantic_profile_id": SEMANTIC_PROFILE_ID,
        "selection_policy": str(config.exemplar_score_policy),
        "dynamic_top_k_policy": "mass_threshold_v1",
        "corridor_link_policy": "c5_authoritative_linkage_v1",
    }
    return authority, policy


def _manifest_hash(path: Path) -> str:
    raw = path.read_bytes()
    return _digest(raw)


def snapshot_finalized_handoff(
    prepared: Any,
    context: Mapping[str, Any],
    *,
    source_config: Any | None = None,
) -> FinalizedV3Handoff:
    if prepared is None or not getattr(prepared, "selected_records", None):
        raise ValueError("v3_requires_finalized_selected_records")
    payloads = list(
        getattr(prepared, "publication_payloads", None)
        or getattr(prepared, "selected_payloads", ())
    )
    records = list(getattr(prepared, "selected_records", ()))
    if len(records) != len(payloads):
        raise ValueError("v3_record_payload_count_mismatch")
    projected = tuple(
        _closed_record(record, payload)
        for record, payload in zip(records, payloads, strict=True)
    )
    indexes = tuple(
        int(record.get("selection_index", index))
        for index, record in enumerate(records)
    )
    if indexes != tuple(range(len(records))):
        raise ValueError("v3_selection_order_not_contiguous")
    authority, policy = _authority_and_policy(
        source_config or prepared.config,
        context,
    )
    return FinalizedV3Handoff(
        projected,
        authority,
        policy,
        indexes,
        int(prepared.config.payload_records_per_shard),
    )


def _write_package(
    root: Path, handoff: FinalizedV3Handoff, transport: str
) -> tuple[str, str, int, int]:
    records = list(handoff.records)
    authority_identity = digest(DOMAIN_LABELS["semantic_authority"], handoff.authority)
    policy_identity = digest(DOMAIN_LABELS["behavioral_policy"], handoff.policy)
    sequence = record_sequence_digest(
        records, selection_indexes=handoff.selection_indexes
    )
    identity_without_root = {
        "schema_version": "radjax_tome_semantic_identity_v3",
        "contract_version": V3_SCHEMA,
        "semantic_profile_id": SEMANTIC_PROFILE_ID,
        "semantic_authority_identity": authority_identity,
        "behavioral_policy_identity": policy_identity,
        "record_count": len(records),
        "ordered_record_sequence_digest": sequence,
    }
    identity = {
        **identity_without_root,
        "semantic_root": semantic_root(identity_without_root),
    }
    identity_path = "provenance/semantic-identity.json"
    authority_path = "provenance/semantic-authority.json"
    policy_path = "provenance/behavioral-policy.json"
    caps_path = "provenance/capabilities.json"
    layout_path = "selected_exemplars/layout.json"
    payload_index_path = "selected_exemplars/payload-index.jsonl"
    shard_index_path = "selected_exemplars/payload-shards.jsonl"
    identity_raw, authority_raw, policy_raw = (
        _raw(identity),
        _raw(handoff.authority),
        _raw(handoff.policy),
    )
    caps_raw = _raw(
        {
            "schema_version": "radjax_tome_capabilities_v1",
            "required": ["standard_integrity_v3", "streaming_shard_receipts_v3"],
            "optional": [],
        }
    )
    payload_rows: list[dict[str, Any]] = []
    shard_rows: list[dict[str, Any]] = []
    chunks = [
        records[i : i + handoff.shard_capacity]
        for i in range(0, len(records), handoff.shard_capacity)
    ]
    if not chunks:
        raise ValueError("v3_requires_records")
    for shard_id, chunk in enumerate(chunks):
        path = f"selected_exemplars/shards/shard-{shard_id:05d}.jsonl"
        raw = b"".join(_raw(record, jsonl=True) for record in chunk)
        _fsync_bytes(root / path, raw)
        shard_rows.append(
            {
                "shard_id": shard_id,
                "path": path,
                "sha256": _digest(raw),
                "size_bytes": len(raw),
                "first_selection_index": shard_id * handoff.shard_capacity,
                "record_count": len(chunk),
            }
        )
        payload_rows.extend(
            {
                "logical_record_id": logical_record_id(record),
                "selection_index": shard_id * handoff.shard_capacity + row,
                "shard_id": shard_id,
                "row": row,
            }
            for row, record in enumerate(chunk)
        )
    payload_index_raw = b"".join(_raw(row, jsonl=True) for row in payload_rows)
    shard_index_raw = b"".join(_raw(row, jsonl=True) for row in shard_rows)
    layout = {
        "schema_version": "radjax_tome_payload_layout_v2",
        "semantic_identity_ref": _ref(
            identity_path, identity_raw, "radjax_tome_semantic_identity_v3"
        ),
        "payload_index_ref": _ref(
            payload_index_path,
            payload_index_raw,
            "radjax_tome_payload_index_v3",
            record_count=len(records),
        ),
        "shard_index_ref": _ref(
            shard_index_path,
            shard_index_raw,
            "radjax_tome_payload_shard_index_v2",
            record_count=len(shard_rows),
        ),
        "record_count": len(records),
        "shard_capacity": handoff.shard_capacity,
    }
    layout_raw = _raw(layout)
    members: dict[str, tuple[bytes, str]] = {
        identity_path: (identity_raw, "semantic_identity"),
        authority_path: (authority_raw, "semantic_authority"),
        policy_path: (policy_raw, "behavioral_policy"),
        caps_path: (caps_raw, "capabilities"),
        layout_path: (layout_raw, "payload_layout"),
        payload_index_path: (payload_index_raw, "payload_index"),
        shard_index_path: (shard_index_raw, "payload_shard_index"),
    }
    for row in shard_rows:
        members[row["path"]] = ((root / row["path"]).read_bytes(), "payload_shard")
    inventory_rows = [
        {
            "path": path,
            "sha256": _digest(raw),
            "size_bytes": len(raw),
            "member_role": role,
            "classification": "integrity_or_provenance",
            "required_for_standard_validation": True,
        }
        for path, (raw, role) in members.items()
    ]
    inventory_path = "manifests/content-manifest-inventory.jsonl"
    inventory_raw = b"".join(_raw(row, jsonl=True) for row in inventory_rows)
    header_path = "manifests/content-manifest-header.json"
    header = {
        "schema_version": "tome_content_manifest_header_v4",
        "contract_version": V3_SCHEMA,
        "profile_id": SEMANTIC_PROFILE_ID,
        "capabilities_ref": _ref(caps_path, caps_raw, "radjax_tome_capabilities_v1"),
        "semantic_identity_ref": _ref(
            identity_path, identity_raw, "radjax_tome_semantic_identity_v3"
        ),
        "layout_ref": _ref(layout_path, layout_raw, "radjax_tome_payload_layout_v2"),
        "inventory_ref": _ref(
            inventory_path, inventory_raw, "tome_content_manifest_inventory_v4"
        ),
        "entry_count": len(inventory_rows),
    }
    header_raw = _raw(header)
    cover = {
        "schema_version": V3_COVER_SCHEMA,
        "contract_version": V3_SCHEMA,
        "package": {"profile_id": SEMANTIC_PROFILE_ID, "transport": transport},
        "capabilities_ref": _ref(caps_path, caps_raw, "radjax_tome_capabilities_v1"),
        "semantic_identity_ref": _ref(
            identity_path, identity_raw, "radjax_tome_semantic_identity_v3"
        ),
        "semantic_authority_ref": _ref(
            authority_path, authority_raw, "radjax_tome_semantic_authority_v1"
        ),
        "behavioral_policy_ref": _ref(
            policy_path, policy_raw, "radjax_tome_behavioral_policy_v1"
        ),
        "manifest_header_ref": _ref(
            header_path, header_raw, "tome_content_manifest_header_v4"
        ),
        "record_count": len(records),
        "shard_count": len(shard_rows),
    }
    _fsync_bytes(root / "cover_page.json", _raw(cover))
    _fsync_bytes(root / header_path, header_raw)
    _fsync_bytes(root / inventory_path, inventory_raw)
    for path, (raw, _) in members.items():
        if not (root / path).exists():
            _fsync_bytes(root / path, raw)
    return (
        identity["semantic_root"],
        authority_identity,
        policy_identity,
        len(shard_rows),
    )


def _journal_path(root: Path, transaction: str = "directory") -> Path:
    """Return a private journal path for one independent publication."""

    return root / f"{transaction}-journal.json"


def _publication_configuration_identity(
    *, record_count: int, shard_capacity: int
) -> str:
    """Bind private recovery to the public v3 layout configuration.

    This is producer-private transaction metadata.  The Contract journal API
    still owns state-machine validation; Tome adds only the filesystem/layout
    binding needed to compare a journal with the promoted package.
    """

    raw = _raw(
        {
            "contract_version": V3_SCHEMA,
            "record_count": int(record_count),
            "schema_version": PRIVATE_BINDING_SCHEMA,
            "semantic_profile_id": SEMANTIC_PROFILE_ID,
            "shard_capacity": int(shard_capacity),
        }
    )
    return _digest(raw)


def _journal_binding(
    *,
    topology: str,
    transaction_kind: str,
    transaction_id: str,
    archive_transaction_id: str,
    output_base_name: str,
    directory_name: str,
    archive_name: str,
    staging_name: str,
    journal_root_name: str,
    configuration_identity: str,
    semantic_root: str | None = None,
    semantic_authority_identity: str | None = None,
    behavioral_policy_identity: str | None = None,
    ordered_record_sequence_digest: str | None = None,
    record_count: int | None = None,
    shard_count: int | None = None,
    shard_capacity: int | None = None,
    transport: str | None = None,
) -> dict[str, Any]:
    if topology not in PRIVATE_TOPOLOGIES:
        raise ValueError("v3_private_topology_invalid")
    return {
        "archive_transaction_id": archive_transaction_id,
        "archive_name": archive_name,
        "behavioral_policy_identity": behavioral_policy_identity,
        "configuration_identity": configuration_identity,
        "directory_name": directory_name,
        "journal_root_name": journal_root_name,
        "output_base_name": output_base_name,
        "ordered_record_sequence_digest": ordered_record_sequence_digest,
        "record_count": record_count,
        "schema_version": PRIVATE_BINDING_SCHEMA,
        "semantic_authority_identity": semantic_authority_identity,
        "semantic_root": semantic_root,
        "shard_capacity": shard_capacity,
        "shard_count": shard_count,
        "staging_name": staging_name,
        "transaction_id": transaction_id,
        "transaction_kind": transaction_kind,
        "topology": topology,
        "transport": transport,
    }


def _write_journal(
    path: Path,
    state: JournalStateV3,
    *,
    binding: Mapping[str, Any],
) -> None:
    validate_journal_state_v3(state)
    raw = _raw(
        {
            "binding": dict(binding),
            "committed_next_selection_index": state.committed_next_selection_index,
            "completion_intent": state.completion_intent,
            "configuration_identity": state.configuration_identity,
            "promotion_marker": state.promotion_marker,
            "sealed_shards": list(state.sealed_shards),
            "semantic_authority_identity": state.semantic_authority_identity,
            "state": state.state,
            "transaction_id": state.transaction_id,
        }
    )
    _fsync_private_bytes(path, raw)


def _read_journal(path: Path) -> tuple[JournalStateV3, dict[str, Any]]:
    try:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        try:
            with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
                descriptor = -1
                raw = json.loads(
                    handle.read(), object_pairs_hook=_json_no_duplicate_keys
                )
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        if isinstance(exc, OSError) and getattr(exc, "errno", None) == errno.ELOOP:
            raise ValueError(f"v3_private_symlink_rejected:{path.name}") from exc
        raise ValueError(f"v3_private_journal_malformed:{path.name}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"v3_private_journal_malformed:{path.name}")
    required = {
        "binding",
        "committed_next_selection_index",
        "completion_intent",
        "configuration_identity",
        "promotion_marker",
        "sealed_shards",
        "semantic_authority_identity",
        "state",
        "transaction_id",
    }
    if set(raw) != required or not isinstance(raw["binding"], dict):
        raise ValueError(f"v3_private_journal_shape:{path.name}")
    state = JournalStateV3(
        transaction_id=str(raw["transaction_id"]),
        configuration_identity=str(raw["configuration_identity"]),
        semantic_authority_identity=str(raw["semantic_authority_identity"]),
        state=str(raw["state"]),
        sealed_shards=tuple(raw["sealed_shards"]),
        committed_next_selection_index=int(raw["committed_next_selection_index"]),
        completion_intent=bool(raw["completion_intent"]),
        promotion_marker=bool(raw["promotion_marker"]),
    )
    binding = dict(raw["binding"])
    if binding.get("schema_version") != PRIVATE_BINDING_SCHEMA:
        raise ValueError(f"v3_private_binding_schema:{path.name}")
    expected_binding_fields = {
        "archive_transaction_id",
        "archive_name",
        "behavioral_policy_identity",
        "configuration_identity",
        "directory_name",
        "journal_root_name",
        "output_base_name",
        "ordered_record_sequence_digest",
        "record_count",
        "schema_version",
        "semantic_authority_identity",
        "semantic_root",
        "shard_capacity",
        "shard_count",
        "staging_name",
        "transaction_id",
        "transaction_kind",
        "topology",
        "transport",
    }
    if set(binding) != expected_binding_fields:
        raise ValueError(f"v3_private_binding_shape:{path.name}")
    if state.configuration_identity != binding["configuration_identity"]:
        raise ValueError(f"v3_private_configuration_mismatch:{path.name}")
    if state.semantic_authority_identity != binding.get("semantic_authority_identity"):
        raise ValueError(f"v3_private_authority_mismatch:{path.name}")
    if state.transaction_id != binding["transaction_id"]:
        raise ValueError(f"v3_private_transaction_mismatch:{path.name}")
    if binding["topology"] not in PRIVATE_TOPOLOGIES:
        raise ValueError(f"v3_private_topology_invalid:{path.name}")
    validate_journal_state_v3(
        state,
        expected_configuration_identity=state.configuration_identity,
        expected_semantic_authority_identity=state.semantic_authority_identity,
    )
    return state, binding


def _sealed_receipts(root: Path) -> tuple[dict[str, Any], ...]:
    rows = [
        json.loads(line)
        for line in (root / "selected_exemplars/payload-shards.jsonl")
        .read_text()
        .splitlines()
    ]
    return tuple(
        {
            "shard_id": int(row["shard_id"]),
            "member_path": row["path"],
            "sha256": row["sha256"],
            "size_bytes": int(row["size_bytes"]),
            "first_selection_index": int(row["first_selection_index"]),
            "record_count": int(row["record_count"]),
        }
        for row in rows
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_json_no_duplicate_keys
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"v3_public_json_malformed:{path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"v3_public_json_object_required:{path}")
    return value


def _artifact_public_metadata(artifact: Path) -> dict[str, Any]:
    """Read only the public identity/layout objects after Contract validation."""

    report = validate_tome_artifact_v3(artifact)
    artifact = Path(artifact)
    temporary: Path | None = None
    root = artifact
    archive: tarfile.TarFile | None = None
    if artifact.is_file():
        temporary = Path(tempfile.mkdtemp(prefix=".v3-identity-", dir=artifact.parent))
        archive = tarfile.open(artifact, mode="r:*")
        try:
            seen_members: set[str] = set()
            for member in archive.getmembers():
                member_path = Path(member.name)
                if (
                    not member.isfile()
                    or member.name in seen_members
                    or member_path.is_absolute()
                    or ".." in member_path.parts
                ):
                    continue
                seen_members.add(member.name)
                destination = temporary / member.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError("v3_archive_member_unreadable")
                _fsync_bytes(destination, source.read())
        finally:
            archive.close()
        root = temporary
    try:
        identity = _read_json_object(root / "provenance/semantic-identity.json")
        authority = _read_json_object(root / "provenance/semantic-authority.json")
        policy = _read_json_object(root / "provenance/behavioral-policy.json")
        layout = _read_json_object(root / "selected_exemplars/layout.json")
        authority_identity = digest(DOMAIN_LABELS["semantic_authority"], authority)
        policy_identity = digest(DOMAIN_LABELS["behavioral_policy"], policy)
        if identity.get("semantic_authority_identity") != authority_identity:
            raise ValueError("v3_public_authority_identity_mismatch")
        if identity.get("behavioral_policy_identity") != policy_identity:
            raise ValueError("v3_public_policy_identity_mismatch")
        if identity.get("contract_version") != V3_SCHEMA:
            raise ValueError("v3_public_contract_identity_mismatch")
        if identity.get("semantic_profile_id") != SEMANTIC_PROFILE_ID:
            raise ValueError("v3_public_profile_identity_mismatch")
        if not isinstance(layout.get("record_count"), int) or not isinstance(
            layout.get("shard_capacity"), int
        ):
            raise ValueError("v3_public_layout_configuration_missing")
        if layout["record_count"] != report.record_count or layout.get(
            "shard_count"
        ) not in (None, report.shard_count):
            raise ValueError("v3_public_layout_count_mismatch")
        return {
            "report": report,
            "semantic_root": identity["semantic_root"],
            "semantic_authority_identity": authority_identity,
            "behavioral_policy_identity": policy_identity,
            "ordered_record_sequence_digest": identity[
                "ordered_record_sequence_digest"
            ],
            "record_count": report.record_count,
            "shard_count": report.shard_count,
            "shard_capacity": layout["shard_capacity"],
            "configuration_identity": _publication_configuration_identity(
                record_count=report.record_count,
                shard_capacity=layout["shard_capacity"],
            ),
            "receipts": _sealed_receipts(root),
        }
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


def _expected_binding(
    metadata: Mapping[str, Any],
    *,
    topology: str,
    transaction_kind: str,
    transaction_id: str,
    archive_transaction_id: str,
    output_base_name: str,
    directory_name: str,
    archive_name: str,
    staging_name: str,
    journal_root_name: str,
    transport: str,
) -> dict[str, Any]:
    return _journal_binding(
        topology=topology,
        transaction_kind=transaction_kind,
        transaction_id=transaction_id,
        archive_transaction_id=archive_transaction_id,
        output_base_name=output_base_name,
        directory_name=directory_name,
        archive_name=archive_name,
        staging_name=staging_name,
        journal_root_name=journal_root_name,
        configuration_identity=metadata["configuration_identity"],
        semantic_root=metadata["semantic_root"],
        semantic_authority_identity=metadata["semantic_authority_identity"],
        behavioral_policy_identity=metadata["behavioral_policy_identity"],
        ordered_record_sequence_digest=metadata["ordered_record_sequence_digest"],
        record_count=metadata["record_count"],
        shard_count=metadata["shard_count"],
        shard_capacity=metadata["shard_capacity"],
        transport=transport,
    )


def _require_binding_match(
    actual: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
) -> None:
    if dict(actual) != dict(expected):
        mismatched = sorted(
            key
            for key in set(actual) | set(expected)
            if actual.get(key) != expected.get(key)
        )
        raise ValueError(f"v3_private_binding_mismatch:{label}:{','.join(mismatched)}")


def _cleanup_private_state(
    journal_root: Path,
    staging_paths: Path | tuple[Path, ...],
    *,
    publication_hook: PublicationHook | None = None,
) -> None:
    paths = (staging_paths,) if isinstance(staging_paths, Path) else staging_paths
    seen: set[Path] = set()
    for staging in paths:
        if staging in seen:
            continue
        seen.add(staging)
        if os.path.lexists(staging):
            info = _private_lstat(staging)
            if stat.S_ISDIR(info.st_mode):
                _assert_private_tree_no_symlinks(staging)
                shutil.rmtree(staging)
            else:
                staging.unlink()
    _fsync_directory(journal_root.parent)
    _publication_event(publication_hook, "CLEANUP_after_staging_removed")
    if os.path.lexists(journal_root):
        info = _private_lstat(journal_root)
        if stat.S_ISDIR(info.st_mode):
            _assert_private_tree_no_symlinks(journal_root)
            shutil.rmtree(journal_root)
        else:
            journal_root.unlink()
    _fsync_directory(journal_root.parent)
    _publication_event(publication_hook, "CLEANUP_after_journal_removed")


def _journal_state(
    *,
    transaction_id: str,
    authority: str,
    configuration_identity: str,
    state: str,
    receipts: tuple[dict[str, Any], ...],
    completion: bool = False,
    promoted: bool = False,
) -> JournalStateV3:
    return JournalStateV3(
        transaction_id=transaction_id,
        configuration_identity=configuration_identity,
        semantic_authority_identity=authority,
        state=state,
        sealed_shards=receipts,
        committed_next_selection_index=sum(
            int(item["record_count"]) for item in receipts
        ),
        completion_intent=completion,
        promotion_marker=promoted,
    )


def publish_v3_from_handoff(
    handoff: FinalizedV3Handoff,
    output_base: Path,
    *,
    publication_hook: PublicationHook | None = None,
) -> V3Publication:
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    directory = output_base.with_name(output_base.name + ".v3")
    archive = output_base.with_name(output_base.name + ".v3.tgz")
    _reject_stale_private_transactions(output_base)
    if directory.exists() or archive.exists():
        raise ValueError("v3_publication_target_exists")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_base.name}.v3-", dir=output_base.parent)
    )
    journal_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_base.name}.v3-journal-", dir=output_base.parent
        )
    )
    transaction_id = str(uuid.uuid4())
    archive_transaction_id = transaction_id + ":archive"
    authority_identity = digest(DOMAIN_LABELS["semantic_authority"], handoff.authority)
    policy_identity = digest(DOMAIN_LABELS["behavioral_policy"], handoff.policy)
    configuration_identity = _publication_configuration_identity(
        record_count=len(handoff.records), shard_capacity=handoff.shard_capacity
    )
    directory_binding = _journal_binding(
        topology="canonical",
        transaction_kind="directory",
        transaction_id=transaction_id,
        archive_transaction_id=archive_transaction_id,
        output_base_name=output_base.name,
        directory_name=directory.name,
        archive_name=archive.name,
        staging_name=staging.name,
        journal_root_name=journal_root.name,
        configuration_identity=configuration_identity,
        semantic_authority_identity=authority_identity,
        behavioral_policy_identity=policy_identity,
        record_count=len(handoff.records),
        shard_capacity=handoff.shard_capacity,
        transport="directory",
    )
    preserve_private_state = False
    private_state_cleaned = False
    archive_transaction_started = False
    try:
        root = staging / "package"
        _write_journal(
            _journal_path(journal_root),
            _journal_state(
                transaction_id=transaction_id,
                authority=authority_identity,
                configuration_identity=configuration_identity,
                state="OPEN",
                receipts=(),
            ),
            binding=directory_binding,
        )
        _publication_event(publication_hook, "PC39_before_shard_sealing")
        semantic, authority, policy, shards = _write_package(root, handoff, "directory")
        directory_binding.update(
            {
                "behavioral_policy_identity": policy,
                "ordered_record_sequence_digest": record_sequence_digest(
                    handoff.records, selection_indexes=handoff.selection_indexes
                ),
                "semantic_authority_identity": authority,
                "semantic_root": semantic,
                "shard_count": shards,
            }
        )
        _publication_event(publication_hook, "PC40_after_shard_bytes_durable")
        receipts = _sealed_receipts(root)
        _write_journal(
            _journal_path(journal_root),
            _journal_state(
                transaction_id=transaction_id,
                authority=authority,
                configuration_identity=configuration_identity,
                state="SEALING",
                receipts=receipts,
            ),
            binding=directory_binding,
        )
        _publication_event(publication_hook, "PC41_after_receipt_durable")
        _write_journal(
            _journal_path(journal_root),
            _journal_state(
                transaction_id=transaction_id,
                authority=authority,
                configuration_identity=configuration_identity,
                state="OPEN",
                receipts=receipts,
            ),
            binding=directory_binding,
        )
        _publication_event(publication_hook, "PC42_after_range_commit")
        _write_journal(
            _journal_path(journal_root),
            _journal_state(
                transaction_id=transaction_id,
                authority=authority,
                configuration_identity=configuration_identity,
                state="COMPLETE_INTENT",
                receipts=receipts,
                completion=True,
            ),
            binding=directory_binding,
        )
        _publication_event(publication_hook, "PC43_after_completion_intent")
        _write_journal(
            _journal_path(journal_root),
            _journal_state(
                transaction_id=transaction_id,
                authority=authority,
                configuration_identity=configuration_identity,
                state="PROMOTING",
                receipts=receipts,
                completion=True,
            ),
            binding=directory_binding,
        )
        _publication_event(publication_hook, "PC44_after_promotion_intent")
        validate_tome_artifact_v3(root)
        _rename_noreplace(root, directory)
        _publication_event(publication_hook, "PC45_after_target_visible")
        _fsync_directory(output_base.parent)
        validate_tome_artifact_v3(directory)
        _publication_event(publication_hook, "PC46_after_atomic_rename")
        _write_journal(
            _journal_path(journal_root),
            _journal_state(
                transaction_id=transaction_id,
                authority=authority,
                configuration_identity=configuration_identity,
                state="PROMOTED",
                receipts=receipts,
                completion=True,
                promoted=True,
            ),
            binding=directory_binding,
        )
        _publication_event(publication_hook, "PC47_after_completion_marker")
        archive_receipts = receipts
        archive_transaction_started = True
        archive_binding = dict(directory_binding)
        archive_binding.update(
            {
                "transaction_id": archive_transaction_id,
                "transaction_kind": "archive",
                "topology": "canonical",
                "transport": "tgz",
            }
        )
        _write_journal(
            _journal_path(journal_root, "archive"),
            _journal_state(
                transaction_id=archive_transaction_id,
                authority=authority,
                configuration_identity=configuration_identity,
                state="OPEN",
                receipts=(),
            ),
            binding=archive_binding,
        )
        _write_journal(
            _journal_path(journal_root, "archive"),
            _journal_state(
                transaction_id=archive_transaction_id,
                authority=authority,
                configuration_identity=configuration_identity,
                state="COMPLETE_INTENT",
                receipts=archive_receipts,
                completion=True,
            ),
            binding=archive_binding,
        )
        _publication_event(publication_hook, "ARCHIVE_after_completion_intent")
        _write_journal(
            _journal_path(journal_root, "archive"),
            _journal_state(
                transaction_id=archive_transaction_id,
                authority=authority,
                configuration_identity=configuration_identity,
                state="PROMOTING",
                receipts=archive_receipts,
                completion=True,
            ),
            binding=archive_binding,
        )
        _publication_event(publication_hook, "ARCHIVE_after_promotion_intent")
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_base.name}.v3-",
            suffix=".tgz",
            dir=output_base.parent,
            delete=False,
        ) as handle:
            archive_tmp = Path(handle.name)
        try:
            archive_root = staging / "archive"
            shutil.copytree(directory, archive_root)
            cover = json.loads((archive_root / "cover_page.json").read_text())
            cover["package"]["transport"] = "tgz"
            _fsync_bytes(archive_root / "cover_page.json", _raw(cover))
            with archive_tmp.open("wb") as raw_archive:
                with gzip.GzipFile(
                    filename="",
                    fileobj=raw_archive,
                    mode="wb",
                    compresslevel=9,
                    mtime=0,
                ) as compressed:
                    with tarfile.open(fileobj=compressed, mode="w") as tar:
                        _tar_add_deterministic(tar, archive_root)
            with archive_tmp.open("rb") as handle:
                os.fsync(handle.fileno())
            # Validate the complete archive while it is still private.  A
            # failed validator must never leave an invalid public archive.
            validate_tome_artifact_v3(archive_tmp)
            _rename_noreplace(archive_tmp, archive)
            _publication_event(publication_hook, "ARCHIVE_after_target_visible")
            _fsync_directory(output_base.parent)
            validate_tome_artifact_v3(archive)
            _publication_event(publication_hook, "ARCHIVE_after_atomic_rename")
        finally:
            archive_tmp.unlink(missing_ok=True)
        _write_journal(
            _journal_path(journal_root, "archive"),
            _journal_state(
                transaction_id=archive_transaction_id,
                authority=authority,
                configuration_identity=configuration_identity,
                state="PROMOTED",
                receipts=archive_receipts,
                completion=True,
                promoted=True,
            ),
            binding=archive_binding,
        )
        _publication_event(publication_hook, "ARCHIVE_after_completion_marker")
        _cleanup_private_state(
            journal_root,
            staging,
            publication_hook=publication_hook,
        )
        private_state_cleaned = True
        return V3Publication(
            directory,
            archive,
            semantic,
            authority,
            policy,
            len(handoff.records),
            shards,
        )
    except V3PublicationCrash:
        preserve_private_state = True
        raise
    except Exception as exc:
        if archive_transaction_started and directory.exists() and not archive.exists():
            preserve_private_state = True
            raise V3ArchivePublicationError(
                "v3_archive_publication_failed_after_directory_promotion",
                directory=directory,
                archive=archive,
            ) from exc
        raise
    finally:
        if not preserve_private_state and not private_state_cleaned:
            _cleanup_private_state(journal_root, staging)


def pack_v3_rtome(directory: Path, output: Path) -> Path:
    """Pack an already-promoted v3 directory into the accepted `.rtome` transport.

    This is a transport wrapper only: it does not construct a second semantic
    package, emit a canonical sibling, or alter any inventoried member bytes.
    The cover transport declaration is the sole nonsemantic packaging change.
    """

    directory = Path(directory)
    output = Path(output)
    if output.suffix != ".rtome" or not directory.is_dir() or output.exists():
        raise ValueError("v3_rtome_transport_target_invalid")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".radjax-tome-v3-rtome-", dir=output.parent))
    archive_tmp: Path | None = None
    try:
        root = staging / "package"
        shutil.copytree(directory, root)
        cover_path = root / "cover_page.json"
        cover = json.loads(cover_path.read_text(encoding="utf-8"))
        cover["package"]["transport"] = "rtome"
        _fsync_bytes(cover_path, _raw(cover))
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.name}-", suffix=".rtome", dir=output.parent, delete=False
        ) as handle:
            archive_tmp = Path(handle.name)
        with archive_tmp.open("wb") as raw_archive:
            with tarfile.open(fileobj=raw_archive, mode="w") as archive:
                _tar_add_deterministic(archive, root)
            raw_archive.flush()
            os.fsync(raw_archive.fileno())
        validate_tome_artifact_v3(archive_tmp)
        _rename_noreplace(archive_tmp, output)
        _fsync_directory(output.parent)
        validate_tome_artifact_v3(output)
        if output.name.endswith(".v3.tgz"):
            base_name = output.name[: -len(".v3.tgz")]
            for private in output.parent.iterdir():
                if private.name.startswith(f".{base_name}.v3"):
                    if private.is_dir():
                        shutil.rmtree(private)
                    else:
                        private.unlink()
        return output
    finally:
        if archive_tmp is not None:
            archive_tmp.unlink(missing_ok=True)
        shutil.rmtree(staging, ignore_errors=True)


def resume_v3_archive_from_directory(
    directory: Path,
    output: Path,
    *,
    publication_hook: PublicationHook | None = None,
) -> Path:
    """Recover a canonical or explicitly archive-only private topology."""

    directory = Path(directory)
    output = Path(output)
    if not directory.is_dir():
        raise ValueError("v3_archive_resume_target_invalid")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = _artifact_public_metadata(directory)
    base_name = (
        output.name[: -len(".v3.tgz")]
        if output.name.endswith(".v3.tgz")
        else output.stem
    )
    journal_prefix = f".{base_name}.v3-journal-"
    staging_prefix = f".{base_name}.v3-"
    candidates: list[Path] = []
    residual_staging: list[Path] = []
    for path in sorted(output.parent.iterdir()):
        if path.name.startswith(journal_prefix):
            if stat.S_ISLNK(os.lstat(path).st_mode):
                raise ValueError(f"v3_private_symlink_rejected:{path.name}")
            candidates.append(path)
        elif path.name.startswith(staging_prefix) and not path.name.startswith(
            f".{base_name}.v3-lock"
        ):
            if stat.S_ISLNK(os.lstat(path).st_mode):
                raise ValueError(f"v3_private_symlink_rejected:{path.name}")
            residual_staging.append(path)
    if len(candidates) > 1:
        raise ValueError("v3_archive_resume_multiple_journals")
    if not candidates and residual_staging:
        raise ValueError("v3_private_staging_without_journal")
    if os.path.lexists(output) and stat.S_ISLNK(os.lstat(output).st_mode):
        raise ValueError("v3_public_archive_symlink")

    journal_root = candidates[0] if candidates else None
    directory_state: JournalStateV3 | None = None
    archive_state: JournalStateV3 | None = None
    directory_binding: dict[str, Any] | None = None
    archive_binding: dict[str, Any] | None = None
    original_staging: Path | None = None
    archive_only = False

    def validate_staging(binding: Mapping[str, Any]) -> Path:
        name = binding["staging_name"]
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not name.startswith(f".{base_name}.v3-")
        ):
            raise ValueError("v3_private_staging_binding_invalid")
        path = output.parent / name
        if os.path.lexists(path):
            _assert_private_tree_no_symlinks(path)
        return path

    def check_common(binding: Mapping[str, Any]) -> None:
        if binding["journal_root_name"] != journal_root.name:  # type: ignore[union-attr]
            raise ValueError("v3_private_journal_root_binding_mismatch")
        if binding["directory_name"] != directory.name:
            raise ValueError("v3_private_directory_binding_mismatch")
        if binding["archive_name"] != output.name:
            raise ValueError("v3_private_archive_binding_mismatch")
        if binding["output_base_name"] != base_name:
            raise ValueError("v3_private_output_base_binding_mismatch")

    if journal_root is not None:
        _assert_private_tree_no_symlinks(journal_root)
        entries = {path.name for path in journal_root.iterdir()}
        if entries not in (
            {"directory-journal.json"},
            {"directory-journal.json", "archive-journal.json"},
            {"archive-journal.json"},
        ):
            raise ValueError("v3_private_journal_objects_incomplete")
        if "directory-journal.json" in entries:
            directory_state, directory_binding = _read_journal(
                _private_child(journal_root, "directory-journal.json", expected="file")
            )
            check_common(directory_binding)
            if directory_binding["topology"] != "canonical":
                raise ValueError("v3_private_directory_topology_invalid")
            expected_directory = _expected_binding(
                metadata,
                topology="canonical",
                transaction_kind="directory",
                transaction_id=directory_binding["transaction_id"],
                archive_transaction_id=directory_binding["archive_transaction_id"],
                output_base_name=base_name,
                directory_name=directory.name,
                archive_name=output.name,
                staging_name=directory_binding["staging_name"],
                journal_root_name=journal_root.name,
                transport="directory",
            )
            _require_binding_match(
                directory_binding, expected_directory, label="directory"
            )
            if tuple(directory_state.sealed_shards) != metadata["receipts"]:
                raise ValueError("v3_private_directory_receipts_mismatch")
            disposition = journal_restart_disposition_v3(
                directory_state,
                public_location_present=True,
                expected_configuration_identity=metadata["configuration_identity"],
                expected_semantic_authority_identity=metadata[
                    "semantic_authority_identity"
                ],
            )
            if disposition.action not in {
                "validate_public_then_mark",
                "open_completed_public_package",
            }:
                raise ValueError("v3_private_directory_restart_not_permitted")
            original_staging = validate_staging(directory_binding)
            if directory_state.state == "PROMOTING":
                directory_state = _journal_state(
                    transaction_id=directory_state.transaction_id,
                    authority=metadata["semantic_authority_identity"],
                    configuration_identity=metadata["configuration_identity"],
                    state="PROMOTED",
                    receipts=metadata["receipts"],
                    completion=True,
                    promoted=True,
                )
                _write_journal(
                    _private_child(
                        journal_root, "directory-journal.json", expected="file"
                    ),
                    directory_state,
                    binding=directory_binding,
                )
            if entries == {"directory-journal.json"}:
                if os.path.lexists(output):
                    raise ValueError("v3_private_archive_visible_without_journal")
                archive_binding = dict(directory_binding)
                archive_binding.update(
                    {
                        "transaction_id": directory_binding["archive_transaction_id"],
                        "transaction_kind": "archive",
                        "topology": "canonical",
                        "transport": "tgz",
                    }
                )
                archive_state = _journal_state(
                    transaction_id=archive_binding["transaction_id"],
                    authority=metadata["semantic_authority_identity"],
                    configuration_identity=metadata["configuration_identity"],
                    state="OPEN",
                    receipts=(),
                )
                _write_journal(
                    _journal_path(journal_root, "archive"),
                    archive_state,
                    binding=archive_binding,
                )
                entries.add("archive-journal.json")
            else:
                archive_state, archive_binding = _read_journal(
                    _private_child(
                        journal_root, "archive-journal.json", expected="file"
                    )
                )
        else:
            archive_only = True
            archive_state, archive_binding = _read_journal(
                _private_child(journal_root, "archive-journal.json", expected="file")
            )
            check_common(archive_binding)
            if archive_binding["topology"] != "archive_only":
                raise ValueError("v3_private_archive_only_topology_invalid")
            if (
                archive_binding["transaction_kind"] != "archive"
                or archive_binding["transaction_id"]
                != archive_binding["archive_transaction_id"]
            ):
                raise ValueError("v3_private_archive_only_transaction_invalid")
            expected_archive = _expected_binding(
                metadata,
                topology="archive_only",
                transaction_kind="archive",
                transaction_id=archive_binding["transaction_id"],
                archive_transaction_id=archive_binding["archive_transaction_id"],
                output_base_name=base_name,
                directory_name=directory.name,
                archive_name=output.name,
                staging_name=archive_binding["staging_name"],
                journal_root_name=journal_root.name,
                transport="tgz",
            )
            _require_binding_match(
                archive_binding, expected_archive, label="archive_only"
            )
            original_staging = validate_staging(archive_binding)

        if archive_binding is None:
            archive_state, archive_binding = _read_journal(
                _private_child(journal_root, "archive-journal.json", expected="file")
            )
        if not archive_only:
            if archive_binding["topology"] != "canonical":
                raise ValueError("v3_private_archive_topology_invalid")
            if archive_binding["transaction_kind"] != "archive":
                raise ValueError("v3_private_archive_transaction_kind")
            if (
                archive_binding["transaction_id"]
                != directory_binding["archive_transaction_id"]
            ):  # type: ignore[index]
                raise ValueError("v3_private_archive_transaction_binding_mismatch")
            expected_archive = _expected_binding(
                metadata,
                topology="canonical",
                transaction_kind="archive",
                transaction_id=archive_binding["transaction_id"],
                archive_transaction_id=directory_binding["archive_transaction_id"],  # type: ignore[index]
                output_base_name=base_name,
                directory_name=directory.name,
                archive_name=output.name,
                staging_name=directory_binding["staging_name"],  # type: ignore[index]
                journal_root_name=journal_root.name,
                transport="tgz",
            )
            _require_binding_match(archive_binding, expected_archive, label="archive")
            if archive_state.state == "OPEN":
                if archive_state.sealed_shards:
                    raise ValueError("v3_private_archive_receipts_mismatch")
            elif tuple(archive_state.sealed_shards) != metadata["receipts"]:
                raise ValueError("v3_private_archive_receipts_mismatch")
            if archive_state.state == "PROMOTED" and not os.path.lexists(output):
                raise ValueError("v3_private_archive_marker_without_archive")
            if archive_state.state == "OPEN" and os.path.lexists(output):
                raise ValueError("v3_private_archive_visible_before_promotion")
            if archive_state.state == "OPEN" and archive_state.sealed_shards:
                raise ValueError("v3_private_archive_receipts_mismatch")
            validate_staging(archive_binding)

    else:
        transaction_id = f"directory-promoted:{metadata['semantic_root']}"
        archive_transaction_id = transaction_id + ":archive"
        archive_binding = None

    if os.path.lexists(output):
        archive_metadata = _artifact_public_metadata(output)
        for key in (
            "semantic_root",
            "semantic_authority_identity",
            "behavioral_policy_identity",
            "ordered_record_sequence_digest",
            "record_count",
            "shard_count",
        ):
            if archive_metadata[key] != metadata[key]:
                raise ValueError("v3_conflicting_existing_archive")
        if journal_root is not None:
            assert archive_state is not None and archive_binding is not None
            disposition = journal_restart_disposition_v3(
                archive_state,
                public_location_present=True,
                expected_configuration_identity=metadata["configuration_identity"],
                expected_semantic_authority_identity=metadata[
                    "semantic_authority_identity"
                ],
            )
            if disposition.action not in {
                "validate_public_then_mark",
                "open_completed_public_package",
            }:
                raise ValueError("v3_private_archive_restart_not_permitted")
            _write_journal(
                _private_child(journal_root, "archive-journal.json", expected="file"),
                _journal_state(
                    transaction_id=archive_binding["transaction_id"],
                    authority=metadata["semantic_authority_identity"],
                    configuration_identity=metadata["configuration_identity"],
                    state="PROMOTED",
                    receipts=metadata["receipts"],
                    completion=True,
                    promoted=True,
                ),
                binding=archive_binding,
            )
            _cleanup_private_state(
                journal_root, original_staging or (), publication_hook=publication_hook
            )
        return output

    if journal_root is None:
        journal_root = Path(
            tempfile.mkdtemp(prefix=f".{base_name}.v3-journal-", dir=output.parent)
        )
        _assert_private_tree_no_symlinks(journal_root)
        build_staging = Path(
            tempfile.mkdtemp(prefix=f".{base_name}.v3-archive-", dir=output.parent)
        )
        _assert_private_tree_no_symlinks(build_staging)
        archive_binding = _expected_binding(
            metadata,
            topology="archive_only",
            transaction_kind="archive",
            transaction_id=archive_transaction_id,
            archive_transaction_id=archive_transaction_id,
            output_base_name=base_name,
            directory_name=directory.name,
            archive_name=output.name,
            staging_name=build_staging.name,
            journal_root_name=journal_root.name,
            transport="tgz",
        )
        _write_journal(
            _private_child(journal_root, "archive-journal.json", expected="file")
            if os.path.lexists(_journal_path(journal_root, "archive"))
            else _journal_path(journal_root, "archive"),
            _journal_state(
                transaction_id=archive_transaction_id,
                authority=metadata["semantic_authority_identity"],
                configuration_identity=metadata["configuration_identity"],
                state="OPEN",
                receipts=(),
            ),
            binding=archive_binding,
        )
        archive_state = _journal_state(
            transaction_id=archive_transaction_id,
            authority=metadata["semantic_authority_identity"],
            configuration_identity=metadata["configuration_identity"],
            state="OPEN",
            receipts=(),
        )
    else:
        assert archive_binding is not None and archive_state is not None
        if archive_state.state not in {"OPEN", "COMPLETE_INTENT", "PROMOTING"}:
            raise ValueError("v3_private_archive_state_not_resumable")
        build_staging = Path(
            tempfile.mkdtemp(
                prefix=f".{base_name}.v3-archive-resume-", dir=output.parent
            )
        )
        _assert_private_tree_no_symlinks(build_staging)

    assert (
        archive_binding is not None
        and journal_root is not None
        and archive_state is not None
    )
    receipts = metadata["receipts"]
    disposition = journal_restart_disposition_v3(
        archive_state,
        public_location_present=False,
        expected_configuration_identity=metadata["configuration_identity"],
        expected_semantic_authority_identity=metadata["semantic_authority_identity"],
    )
    if disposition.action not in {
        "resume_committed_prefix",
        "derive_public_evidence",
        "retry_promotion",
    }:
        raise ValueError("v3_private_archive_restart_not_permitted")
    archive_journal = _private_child(
        journal_root, "archive-journal.json", expected="file"
    )
    if archive_state.state == "OPEN":
        _write_journal(
            archive_journal,
            _journal_state(
                transaction_id=archive_binding["transaction_id"],
                authority=metadata["semantic_authority_identity"],
                configuration_identity=metadata["configuration_identity"],
                state="COMPLETE_INTENT",
                receipts=receipts,
                completion=True,
            ),
            binding=archive_binding,
        )
        _publication_event(publication_hook, "RESUME_ARCHIVE_after_completion_intent")
    _write_journal(
        archive_journal,
        _journal_state(
            transaction_id=archive_binding["transaction_id"],
            authority=metadata["semantic_authority_identity"],
            configuration_identity=metadata["configuration_identity"],
            state="PROMOTING",
            receipts=receipts,
            completion=True,
        ),
        binding=archive_binding,
    )
    _publication_event(publication_hook, "RESUME_ARCHIVE_after_promotion_intent")
    archive_tmp: Path | None = None
    try:
        _assert_private_tree_no_symlinks(build_staging)
        root = build_staging / "package"
        shutil.copytree(directory, root)
        cover_path = root / "cover_page.json"
        cover = _read_json_object(cover_path)
        cover["package"]["transport"] = "tgz"
        _fsync_bytes(cover_path, _raw(cover))
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.name}-", suffix=".tgz", dir=output.parent, delete=False
        ) as handle:
            archive_tmp = Path(handle.name)
        with archive_tmp.open("wb") as raw_archive:
            with gzip.GzipFile(
                filename="", fileobj=raw_archive, mode="wb", compresslevel=9, mtime=0
            ) as compressed:
                with tarfile.open(fileobj=compressed, mode="w") as tar:
                    _tar_add_deterministic(tar, root)
        with archive_tmp.open("rb") as handle:
            os.fsync(handle.fileno())
        validate_tome_artifact_v3(archive_tmp)
        _rename_noreplace(archive_tmp, output)
        _publication_event(publication_hook, "RESUME_ARCHIVE_after_target_visible")
        _fsync_directory(output.parent)
        _publication_event(publication_hook, "RESUME_ARCHIVE_after_atomic_rename")
        archive_metadata = _artifact_public_metadata(output)
        if archive_metadata["semantic_root"] != metadata["semantic_root"]:
            raise ValueError("v3_archive_semantic_root_mismatch")
        _write_journal(
            archive_journal,
            _journal_state(
                transaction_id=archive_binding["transaction_id"],
                authority=metadata["semantic_authority_identity"],
                configuration_identity=metadata["configuration_identity"],
                state="PROMOTED",
                receipts=receipts,
                completion=True,
                promoted=True,
            ),
            binding=archive_binding,
        )
        _publication_event(publication_hook, "RESUME_ARCHIVE_after_completion_marker")
        _cleanup_private_state(
            journal_root,
            tuple(
                path for path in (original_staging, build_staging) if path is not None
            ),
            publication_hook=publication_hook,
        )
        return output
    finally:
        if archive_tmp is not None:
            archive_tmp.unlink(missing_ok=True)
        if os.path.lexists(build_staging):
            _assert_private_tree_no_symlinks(build_staging)
            shutil.rmtree(build_staging)


__all__ = [
    "FinalizedV3Handoff",
    "V3Publication",
    "V3ArchivePublicationError",
    "V3PublicationCrash",
    "snapshot_finalized_handoff",
    "publish_v3_from_handoff",
    "pack_v3_rtome",
    "resume_v3_archive_from_directory",
]
