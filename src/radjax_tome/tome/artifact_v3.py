"""Tome-side adapter for the released RADJAX artifact Contract v3.

The publisher accepts only the finalized in-memory delivery handoff.  It never
parses the legacy selected-record files to reconstruct v3 semantics.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import uuid
from collections.abc import Mapping
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


def _journal_path(root: Path) -> Path:
    return root / "journal.json"


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
    handoff: FinalizedV3Handoff, output_base: Path
) -> V3Publication:
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    directory = output_base.with_name(output_base.name + ".v3")
    archive = output_base.with_name(output_base.name + ".v3.tgz")
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
    try:
        root = staging / "package"
        semantic, authority, policy, shards = _write_package(root, handoff, "directory")
        receipts = _sealed_receipts(root)
        _write_journal(
            _journal_path(journal_root),
            _journal_state(
                transaction_id=transaction_id,
                authority=authority,
                state="OPEN",
                receipts=(),
            ),
        )
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
        validate_tome_artifact_v3(root)
        os.rename(root, directory)
        _fsync_directory(output_base.parent)
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
        archive_receipts = receipts
        archive_transaction_id = transaction_id + ":archive"
        _write_journal(
            _journal_path(journal_root),
            _journal_state(
                transaction_id=archive_transaction_id,
                authority=authority,
                state="OPEN",
                receipts=(),
            ),
        )
        _write_journal(
            _journal_path(journal_root),
            _journal_state(
                transaction_id=archive_transaction_id,
                authority=authority,
                state="COMPLETE_INTENT",
                receipts=archive_receipts,
                completion=True,
            ),
        )
        _write_journal(
            _journal_path(journal_root),
            _journal_state(
                transaction_id=archive_transaction_id,
                authority=authority,
                state="PROMOTING",
                receipts=archive_receipts,
                completion=True,
            ),
        )
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
            with tarfile.open(archive_tmp, "w:gz") as tar:
                for path in sorted(archive_root.rglob("*")):
                    if path.is_file():
                        tar.add(
                            path,
                            arcname=path.relative_to(archive_root).as_posix(),
                            recursive=False,
                        )
            if archive.exists():
                raise ValueError("v3_archive_target_exists")
            os.rename(archive_tmp, archive)
            _fsync_directory(output_base.parent)
        finally:
            archive_tmp.unlink(missing_ok=True)
        validate_tome_artifact_v3(archive)
        _write_journal(
            _journal_path(journal_root),
            _journal_state(
                transaction_id=archive_transaction_id,
                authority=authority,
                state="PROMOTED",
                receipts=archive_receipts,
                completion=True,
                promoted=True,
            ),
        )
        return V3Publication(
            directory,
            archive,
            semantic,
            authority,
            policy,
            len(handoff.records),
            shards,
        )
    finally:
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
    staging = Path(tempfile.mkdtemp(prefix=".radjax-tome-v3-rtome-"))
    try:
        root = staging / "package"
        shutil.copytree(directory, root)
        cover_path = root / "cover_page.json"
        cover = json.loads(cover_path.read_text(encoding="utf-8"))
        cover["package"]["transport"] = "rtome"
        _fsync_bytes(cover_path, _raw(cover))
        output.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(output, "w") as archive:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    archive.add(
                        path,
                        arcname=path.relative_to(root).as_posix(),
                        recursive=False,
                    )
        validate_tome_artifact_v3(output)
        return output
    finally:
        shutil.rmtree(staging, ignore_errors=True)


__all__ = [
    "FinalizedV3Handoff",
    "V3Publication",
    "snapshot_finalized_handoff",
    "publish_v3_from_handoff",
    "pack_v3_rtome",
]
