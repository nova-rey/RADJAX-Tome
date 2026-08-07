"""Tome-side adapter for the released RADJAX artifact Contract v3.

The publisher accepts only the finalized in-memory delivery handoff.  It never
parses the legacy selected-record files to reconstruct v3 semantics.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
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
from radjax_contract.tome.v3.journal import validate_journal_state_v3
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


def _reject_stale_private_transactions(output_base: Path) -> None:
    """Refuse to start over while an earlier interrupted transaction remains.

    The private state is intentionally not guessed at or deleted.  A caller
    must inspect/quarantine it or use the explicit archive-resume helper.  This
    prevents a new run from mixing authorities or accepting an unreceipted
    shard left by an earlier process.
    """

    prefixes = _private_transaction_prefixes(output_base)
    stale = sorted(
        path for path in output_base.parent.iterdir() if path.name.startswith(prefixes)
    )
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


def _write_journal(path: Path, state: JournalStateV3) -> None:
    validate_journal_state_v3(state)
    raw = _raw(
        {
            "transaction_id": state.transaction_id,
            "configuration_identity": state.configuration_identity,
            "semantic_authority_identity": state.semantic_authority_identity,
            "state": state.state,
            "sealed_shards": list(state.sealed_shards),
            "committed_next_selection_index": state.committed_next_selection_index,
            "completion_intent": state.completion_intent,
            "promotion_marker": state.promotion_marker,
        }
    )
    _fsync_bytes(path, raw)


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


def _journal_state(
    *,
    transaction_id: str,
    authority: str,
    state: str,
    receipts: tuple[dict[str, Any], ...],
    completion: bool = False,
    promoted: bool = False,
) -> JournalStateV3:
    return JournalStateV3(
        transaction_id=transaction_id,
        configuration_identity=V3_SCHEMA,
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
    preserve_private_state = False
    archive_transaction_started = False
    try:
        root = staging / "package"
        _write_journal(
            _journal_path(journal_root),
            _journal_state(
                transaction_id=transaction_id,
                authority=digest(
                    DOMAIN_LABELS["semantic_authority"], handoff.authority
                ),
                state="OPEN",
                receipts=(),
            ),
        )
        _publication_event(publication_hook, "PC39_before_shard_sealing")
        semantic, authority, policy, shards = _write_package(root, handoff, "directory")
        _publication_event(publication_hook, "PC40_after_shard_bytes_durable")
        receipts = _sealed_receipts(root)
        _write_journal(
            _journal_path(journal_root),
            _journal_state(
                transaction_id=transaction_id,
                authority=authority,
                state="SEALING",
                receipts=receipts,
            ),
        )
        _publication_event(publication_hook, "PC41_after_receipt_durable")
        _write_journal(
            _journal_path(journal_root),
            _journal_state(
                transaction_id=transaction_id,
                authority=authority,
                state="OPEN",
                receipts=receipts,
            ),
        )
        _publication_event(publication_hook, "PC42_after_range_commit")
        _write_journal(
            _journal_path(journal_root),
            _journal_state(
                transaction_id=transaction_id,
                authority=authority,
                state="COMPLETE_INTENT",
                receipts=receipts,
                completion=True,
            ),
        )
        _publication_event(publication_hook, "PC43_after_completion_intent")
        _write_journal(
            _journal_path(journal_root),
            _journal_state(
                transaction_id=transaction_id,
                authority=authority,
                state="PROMOTING",
                receipts=receipts,
                completion=True,
            ),
        )
        _publication_event(publication_hook, "PC44_after_promotion_intent")
        validate_tome_artifact_v3(root)
        _rename_noreplace(root, directory)
        _publication_event(publication_hook, "PC45_after_target_visible")
        _fsync_directory(output_base.parent)
        _publication_event(publication_hook, "PC46_after_atomic_rename")
        _write_journal(
            _journal_path(journal_root),
            _journal_state(
                transaction_id=transaction_id,
                authority=authority,
                state="PROMOTED",
                receipts=receipts,
                completion=True,
                promoted=True,
            ),
        )
        _publication_event(publication_hook, "PC47_after_completion_marker")
        archive_receipts = receipts
        archive_transaction_id = transaction_id + ":archive"
        archive_transaction_started = True
        _write_journal(
            _journal_path(journal_root, "archive"),
            _journal_state(
                transaction_id=archive_transaction_id,
                authority=authority,
                state="OPEN",
                receipts=(),
            ),
        )
        _write_journal(
            _journal_path(journal_root, "archive"),
            _journal_state(
                transaction_id=archive_transaction_id,
                authority=authority,
                state="COMPLETE_INTENT",
                receipts=archive_receipts,
                completion=True,
            ),
        )
        _publication_event(publication_hook, "ARCHIVE_after_completion_intent")
        _write_journal(
            _journal_path(journal_root, "archive"),
            _journal_state(
                transaction_id=archive_transaction_id,
                authority=authority,
                state="PROMOTING",
                receipts=archive_receipts,
                completion=True,
            ),
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
            _publication_event(publication_hook, "ARCHIVE_after_atomic_rename")
        finally:
            archive_tmp.unlink(missing_ok=True)
        _write_journal(
            _journal_path(journal_root, "archive"),
            _journal_state(
                transaction_id=archive_transaction_id,
                authority=authority,
                state="PROMOTED",
                receipts=archive_receipts,
                completion=True,
                promoted=True,
            ),
        )
        _publication_event(publication_hook, "ARCHIVE_after_completion_marker")
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
        if not preserve_private_state:
            shutil.rmtree(staging, ignore_errors=True)
            shutil.rmtree(journal_root, ignore_errors=True)


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


def resume_v3_archive_from_directory(directory: Path, output: Path) -> Path:
    """Retry only the archive transaction after a promoted directory exists.

    The directory is standard-validated before it is copied.  The retry never
    reads legacy output files or private journals and never replaces an
    existing archive.  A caller can therefore recover a directory/archive
    partial publication without re-running score, selection, or assembly.
    """

    directory = Path(directory)
    output = Path(output)
    if not directory.is_dir() or output.exists():
        raise ValueError("v3_archive_resume_target_invalid")
    validate_tome_artifact_v3(directory)
    semantic_identity = json.loads(
        (directory / "provenance/semantic-identity.json").read_text(encoding="utf-8")
    )
    authority_identity = semantic_identity["semantic_authority_identity"]
    output.parent.mkdir(parents=True, exist_ok=True)
    base_name = (
        output.name[: -len(".v3.tgz")]
        if output.name.endswith(".v3.tgz")
        else output.stem
    )
    existing_journals = sorted(
        path
        for path in output.parent.iterdir()
        if path.name.startswith(f".{base_name}.v3-journal-")
    )
    if len(existing_journals) > 1:
        raise ValueError("v3_archive_resume_multiple_journals")
    journal_root = (
        existing_journals[0]
        if existing_journals
        else Path(
            tempfile.mkdtemp(prefix=f".{base_name}.v3-journal-", dir=output.parent)
        )
    )
    transaction_id = str(uuid.uuid4()) + ":archive-resume"
    receipts = _sealed_receipts(directory)
    for state, completion in (
        ("OPEN", False),
        ("COMPLETE_INTENT", True),
        ("PROMOTING", True),
    ):
        _write_journal(
            _journal_path(journal_root, "archive"),
            _journal_state(
                transaction_id=transaction_id,
                authority=authority_identity,
                state=state,
                receipts=receipts if completion else (),
                completion=completion,
            ),
        )
    completed = False
    staging = Path(
        tempfile.mkdtemp(prefix=".radjax-tome-v3-archive-resume-", dir=output.parent)
    )
    archive_tmp: Path | None = None
    try:
        root = staging / "package"
        shutil.copytree(directory, root)
        cover_path = root / "cover_page.json"
        cover = json.loads(cover_path.read_text(encoding="utf-8"))
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
        _fsync_directory(output.parent)
        validate_tome_artifact_v3(output)
        _write_journal(
            _journal_path(journal_root, "archive"),
            _journal_state(
                transaction_id=transaction_id,
                authority=authority_identity,
                state="PROMOTED",
                receipts=receipts,
                completion=True,
                promoted=True,
            ),
        )
        completed = True
        return output
    finally:
        if archive_tmp is not None:
            archive_tmp.unlink(missing_ok=True)
        shutil.rmtree(staging, ignore_errors=True)
        if completed:
            shutil.rmtree(journal_root, ignore_errors=True)


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
