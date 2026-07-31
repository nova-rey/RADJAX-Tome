#!/usr/bin/env python3
"""Stdlib-only streaming validator for the proposed RADJAX-Tome v2 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

PREFIX = "sha256:"
PROFILES = {"unpacked", "student", "full_debug_provenance"}
TRANSPORTS = {"directory", "rtome", "tgz"}
CLASSIFICATIONS = {
    "training_critical",
    "integrity_or_provenance",
    "diagnostic",
    "human_readable",
    "operational",
}
CHUNK = 1 << 16
SEMANTIC_FIELDS = {
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
INTEGER_FIELDS = {
    "selected_position",
    "score_top_token_id",
    "source_shard_id",
    "source_row",
    "source_position",
    "source_top_token_id",
    "effective_top_k",
    "sequence_length",
    "vocab_size",
    "num_buckets",
    "dynamic_top_k_max",
}
NUMBER_FIELDS = {
    "selected_score",
    "score_selected_position_entropy",
    "source_score",
    "top_mass",
    "tail_mass",
    "teacher_entropy",
    "dynamic_mass_threshold",
    "effective_top_k_fraction_of_vocab",
}
STRING_FIELDS = {
    "selected_example_id",
    "source_score_policy",
    "selected_policy",
    "source_delivery_path",
    "long_tail_class",
    "semantic_tail_tag",
    "selected_board",
    "corridor_assignment_status",
}
EXTENSION_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class Result:
    ok: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ContractError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("shape_invalid") from exc
    if not isinstance(value, dict):
        raise ContractError("shape_invalid")
    return value


def _digest_path(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while block := handle.read(CHUNK):
                digest.update(block)
                size += len(block)
    except OSError as exc:
        raise ContractError("transport_corrupt") from exc
    return PREFIX + digest.hexdigest(), size


def _sha(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 71 or not value.startswith(PREFIX):
        raise ContractError("digest_syntax_invalid")
    if any(char not in "0123456789abcdef" for char in value[len(PREFIX) :]):
        raise ContractError("digest_syntax_invalid")
    return value


def _path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ContractError("path_unsafe")
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or value != parsed.as_posix()
        or any(part in {".", ".."} for part in parsed.parts)
    ):
        raise ContractError("path_unsafe")
    return value


def _canonical(value: Any) -> str:
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    except (TypeError, ValueError) as exc:
        raise ContractError("shape_invalid") from exc
    return PREFIX + hashlib.sha256(encoded).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    except (TypeError, ValueError) as exc:
        raise ContractError("shape_invalid") from exc


def _nonnegative_int(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ContractError("shape_invalid")
    return value


def _finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ContractError("payload_nonfinite_number")
    if isinstance(value, dict):
        for child in value.values():
            _finite(child)
    elif isinstance(value, list):
        for child in value:
            _finite(child)


class _SequenceDigest:
    """Constant-space hasher for the canonical sequence-digest JSON value."""

    def __init__(self) -> None:
        self._digest = hashlib.sha256()
        self._digest.update(b'{"records":[')
        self._first = True

    def add(self, record: dict[str, str]) -> None:
        if not self._first:
            self._digest.update(b",")
        self._digest.update(_canonical_bytes(record))
        self._first = False

    def finish(self) -> str:
        self._digest.update(
            b'],"schema_version":"selected_exemplar_payload_sequence_v1"}'
        )
        return PREFIX + self._digest.hexdigest()


def _semantic_record(record: Any) -> tuple[str, str]:
    if not isinstance(record, dict) or not SEMANTIC_FIELDS <= set(record):
        raise ContractError("payload_semantic_projection_invalid")
    allowed = SEMANTIC_FIELDS | {"opaque_extensions"}
    if set(record) - allowed:
        raise ContractError("payload_semantic_projection_invalid")
    _finite(record)
    for field in INTEGER_FIELDS:
        if not isinstance(record[field], int) or isinstance(record[field], bool):
            raise ContractError("payload_semantic_projection_invalid")
    for field in NUMBER_FIELDS:
        if not isinstance(record[field], (int, float)) or isinstance(
            record[field], bool
        ):
            raise ContractError("payload_semantic_projection_invalid")
    for field in STRING_FIELDS:
        if not isinstance(record[field], str) or not record[field]:
            raise ContractError("payload_semantic_projection_invalid")
    if record["sequence_length"] < 1 or record["vocab_size"] < 1:
        raise ContractError("payload_semantic_projection_invalid")
    if record["num_buckets"] < 0 or record["dynamic_top_k_max"] < 1:
        raise ContractError("payload_semantic_projection_invalid")
    if not isinstance(record["payload_ref"], dict):
        raise ContractError("payload_semantic_projection_invalid")
    for name in ("top_token_ids",):
        if not isinstance(record[name], list) or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in record[name]
        ):
            raise ContractError("payload_semantic_projection_invalid")
    for name in ("top_log_probs", "top_probs", "bucket_masses"):
        if not isinstance(record[name], list) or any(
            not isinstance(value, (int, float)) or isinstance(value, bool)
            for value in record[name]
        ):
            raise ContractError("payload_semantic_projection_invalid")
    if not isinstance(record["top_selection_mask"], list) or any(
        not isinstance(value, bool) for value in record["top_selection_mask"]
    ):
        raise ContractError("payload_semantic_projection_invalid")
    if not isinstance(record["dynamic_top_k"], (bool, dict)):
        raise ContractError("payload_semantic_projection_invalid")
    if not isinstance(record["top_k_saturated"], bool):
        raise ContractError("payload_semantic_projection_invalid")
    if not isinstance(record["long_tail_warnings"], list) or any(
        not isinstance(value, str) for value in record["long_tail_warnings"]
    ):
        raise ContractError("payload_semantic_projection_invalid")
    for name in ("corridor_mode_id", "corridor_fingerprint_id"):
        if not isinstance(record[name], (str, int)) or isinstance(record[name], bool):
            raise ContractError("payload_semantic_projection_invalid")
    extensions = record.get("opaque_extensions", {})
    if not isinstance(extensions, dict):
        raise ContractError("opaque_extension_undocumented")
    for name, extension in extensions.items():
        if not EXTENSION_NAME.fullmatch(name) or not isinstance(extension, dict):
            raise ContractError("opaque_extension_undocumented")
        _require(extension, {"schema_id", "value", "semantic_digest"})
        if not isinstance(extension["schema_id"], str) or not extension["schema_id"]:
            raise ContractError("opaque_extension_undocumented")
        _finite(extension["value"])
        if _canonical(extension["value"]) != _sha(extension["semantic_digest"]):
            raise ContractError("opaque_extension_undocumented")
    logical_id = _canonical(
        {
            "selected_example_id": record["selected_example_id"],
            "selected_position": record["selected_position"],
        }
    )
    return logical_id, _canonical(record)


def _inside(root: Path, relative: str) -> Path:
    candidate = root / relative
    if candidate.is_symlink():
        raise ContractError("path_unsafe")
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ContractError("path_unsafe") from exc
    return candidate


def _lines(path: Path) -> Iterator[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    raise ContractError("shape_invalid")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ContractError("shape_invalid")
                yield value
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("shape_invalid") from exc


def _require(value: dict[str, Any], keys: set[str]) -> None:
    if set(value) != keys:
        raise ContractError("shape_invalid")


def _validate_identity(identity: Any) -> dict[str, Any]:
    if not isinstance(identity, dict):
        raise ContractError("payload_semantic_projection_invalid")
    _require(
        identity,
        {
            "schema_version",
            "payload_sequence_digest",
            "selected_count",
            "training_contract",
            "authority",
            "semantic_digest",
        },
    )
    if identity["schema_version"] != "radjax_tome_semantic_identity_v2":
        raise ContractError("schema_version_unsupported")
    _sha(identity["payload_sequence_digest"])
    _nonnegative_int(identity["selected_count"])
    if not isinstance(identity["training_contract"], dict) or not isinstance(
        identity["authority"], dict
    ):
        raise ContractError("payload_semantic_projection_invalid")
    _finite(identity["training_contract"])
    _finite(identity["authority"])
    expected = _canonical(
        {key: value for key, value in identity.items() if key != "semantic_digest"}
    )
    if expected != _sha(identity["semantic_digest"]):
        raise ContractError("payload_semantic_projection_invalid")
    return identity


def _validate_selected_payloads(
    root: Path, listed: set[str], identity: dict[str, Any]
) -> None:
    layout_path = "selected_exemplars/payload-layout.json"
    if layout_path not in listed:
        raise ContractError("profile_inventory_mismatch")
    layout = _json(_inside(root, layout_path))
    _require(
        layout,
        {
            "schema_version",
            "layout_version",
            "payload_index",
            "sequence_digest",
            "selected_count",
            "payload_records_per_shard",
            "shards",
        },
    )
    if (
        layout["schema_version"] != "radjax_tome_payload_layout_v1"
        or layout["layout_version"] != "selected_payload_shards_v1"
    ):
        raise ContractError("schema_version_unsupported")
    selected_count = _nonnegative_int(layout["selected_count"])
    capacity = _nonnegative_int(layout["payload_records_per_shard"])
    if capacity < 1:
        raise ContractError("shape_invalid")
    _sha(layout["sequence_digest"])
    index_ref = layout["payload_index"]
    if not isinstance(index_ref, dict):
        raise ContractError("shape_invalid")
    _require(
        index_ref, {"path", "sha256", "size_bytes", "record_count", "schema_version"}
    )
    if index_ref["schema_version"] != "radjax_tome_payload_index_v2":
        raise ContractError("schema_version_unsupported")
    index_path = _path(index_ref["path"])
    if (
        index_path != "selected_exemplars/payload-index.jsonl"
        or index_path not in listed
    ):
        raise ContractError("payload_index_address_invalid")
    if _digest_path(_inside(root, index_path)) != (
        _sha(index_ref["sha256"]),
        _nonnegative_int(index_ref["size_bytes"]),
    ):
        raise ContractError("digest_mismatch")
    index_count = _nonnegative_int(index_ref["record_count"])
    shards = layout["shards"]
    if not isinstance(shards, list):
        raise ContractError("shape_invalid")
    index_iter = iter(_lines(_inside(root, index_path)))
    overall_digest = _SequenceDigest()
    # The list only contains fixed two-digest references.  It deliberately never
    # contains payload bytes; M7D will replace it with a digest sink for unbounded
    # packages before writer integration.
    expected_selection = 0
    index_seen = 0
    seen_logical_ids: set[str] = set()
    for shard_position, shard in enumerate(shards):
        if not isinstance(shard, dict):
            raise ContractError("shape_invalid")
        _require(
            shard,
            {
                "shard_id",
                "path",
                "sha256",
                "size_bytes",
                "first_selection_index",
                "last_selection_index",
                "record_count",
                "semantic_digest",
            },
        )
        shard_id = _nonnegative_int(shard["shard_id"])
        record_count = _nonnegative_int(shard["record_count"])
        first = _nonnegative_int(shard["first_selection_index"])
        last = _nonnegative_int(shard["last_selection_index"])
        if shard_id != shard_position or record_count < 1 or record_count > capacity:
            raise ContractError("payload_index_address_invalid")
        if first != expected_selection or last != first + record_count - 1:
            raise ContractError("payload_index_address_invalid")
        if shard_position < len(shards) - 1 and record_count != capacity:
            raise ContractError("payload_index_address_invalid")
        shard_path = _path(shard["path"])
        if shard_path not in listed:
            raise ContractError("profile_inventory_mismatch")
        if _digest_path(_inside(root, shard_path)) != (
            _sha(shard["sha256"]),
            _nonnegative_int(shard["size_bytes"]),
        ):
            raise ContractError("digest_mismatch")
        shard_digest = _SequenceDigest()
        shard_seen = 0
        for row, payload in enumerate(_lines(_inside(root, shard_path))):
            if row >= record_count:
                raise ContractError("manifest_record_count_mismatch")
            try:
                index = next(index_iter)
            except StopIteration as exc:
                raise ContractError("manifest_record_count_mismatch") from exc
            index_seen += 1
            _require(
                index,
                {
                    "logical_id",
                    "selected_example_id",
                    "selected_position",
                    "selection_index",
                    "shard_id",
                    "row",
                    "payload_sha256",
                    "payload_semantic_digest",
                    "shard_sha256",
                },
            )
            logical_id, semantic_digest = _semantic_record(payload)
            if (
                index.get("logical_id") != logical_id
                or index.get("selected_example_id") != payload["selected_example_id"]
                or index.get("selected_position") != payload["selected_position"]
                or _nonnegative_int(index.get("selection_index")) != expected_selection
                or _nonnegative_int(index.get("shard_id")) != shard_id
                or _nonnegative_int(index.get("row")) != row
                or _sha(index.get("payload_sha256")) != _canonical(payload)
                or _sha(index.get("shard_sha256")) != shard["sha256"]
                or logical_id in seen_logical_ids
            ):
                raise ContractError("payload_index_address_invalid")
            if _sha(index.get("payload_semantic_digest")) != semantic_digest:
                raise ContractError("payload_semantic_projection_invalid")
            seen_logical_ids.add(logical_id)
            record = {
                "logical_id": logical_id,
                "payload_semantic_digest": semantic_digest,
            }
            shard_digest.add(record)
            overall_digest.add(record)
            shard_seen += 1
            expected_selection += 1
        if shard_seen != record_count:
            raise ContractError("manifest_record_count_mismatch")
        if shard_digest.finish() != _sha(shard["semantic_digest"]):
            raise ContractError("payload_sequence_digest_mismatch")
    try:
        next(index_iter)
    except StopIteration:
        pass
    else:
        raise ContractError("manifest_record_count_mismatch")
    if index_seen != index_count or expected_selection != selected_count:
        raise ContractError("manifest_record_count_mismatch")
    sequence_digest = overall_digest.finish()
    if sequence_digest != _sha(layout["sequence_digest"]):
        raise ContractError("payload_sequence_digest_mismatch")
    if (
        selected_count != identity["selected_count"]
        or sequence_digest != identity["payload_sequence_digest"]
    ):
        raise ContractError("payload_semantic_projection_invalid")


def validate_directory(root: Path) -> Result:
    try:
        cover = _json(root / "cover_page.json")
        _require(
            cover,
            {
                "schema_version",
                "identity",
                "training",
                "package",
                "manifests",
                "authority",
                "provenance",
                "validation",
            },
        )
        if cover["schema_version"] != "radjax_tome_cover_v4":
            raise ContractError("schema_version_unsupported")
        package = cover["package"]
        manifests = cover["manifests"]
        if (
            not isinstance(package, dict)
            or set(package) != {"profile", "transport"}
            or package.get("profile") not in PROFILES
            or package.get("transport") not in TRANSPORTS
        ):
            raise ContractError("profile_inventory_mismatch")
        if not isinstance(manifests, dict) or set(manifests) != {"header"}:
            raise ContractError("shape_invalid")
        ref = manifests["header"]
        if not isinstance(ref, dict):
            raise ContractError("shape_invalid")
        if (
            set(ref) != {"path", "sha256", "size_bytes", "schema_version"}
            or ref.get("schema_version") != "tome_content_manifest_header_v3"
        ):
            raise ContractError("shape_invalid")
        header_path = _path(ref.get("path"))
        if header_path != "manifests/content-manifest-header.json":
            raise ContractError("shape_invalid")
        if _digest_path(_inside(root, header_path)) != (
            _sha(ref.get("sha256")),
            ref.get("size_bytes"),
        ):
            raise ContractError("digest_mismatch")
        header = _json(_inside(root, header_path))
        _require(
            header,
            {
                "schema_version",
                "profile",
                "semantic_identity_digest",
                "inventory_path",
                "inventory_sha256",
                "inventory_size_bytes",
                "entry_count",
            },
        )
        if (
            header["schema_version"] != "tome_content_manifest_header_v3"
            or header["profile"] != package["profile"]
        ):
            raise ContractError("profile_inventory_mismatch")
        inventory_path = _path(header["inventory_path"])
        if inventory_path != "manifests/content-manifest-inventory.jsonl":
            raise ContractError("shape_invalid")
        _sha(header["semantic_identity_digest"])
        if _digest_path(_inside(root, inventory_path)) != (
            _sha(header["inventory_sha256"]),
            header["inventory_size_bytes"],
        ):
            raise ContractError("digest_mismatch")
        previous = ""
        count = 0
        listed: set[str] = set()
        for entry in _lines(_inside(root, inventory_path)):
            _require(
                entry,
                {
                    "path",
                    "sha256",
                    "size_bytes",
                    "classification",
                    "training_authoritative",
                },
            )
            path = _path(entry["path"])
            if path <= previous or path in {
                "cover_page.json",
                header_path,
                inventory_path,
            }:
                raise ContractError("ordering_invalid")
            previous, count = path, count + 1
            listed.add(path)
            if not isinstance(entry["size_bytes"], int) or isinstance(
                entry["size_bytes"], bool
            ):
                raise ContractError("shape_invalid")
            if entry["classification"] not in CLASSIFICATIONS or not isinstance(
                entry["training_authoritative"], bool
            ):
                raise ContractError("shape_invalid")
            if _digest_path(_inside(root, path)) != (
                _sha(entry["sha256"]),
                entry["size_bytes"],
            ):
                raise ContractError("digest_mismatch")
        if count != header["entry_count"]:
            raise ContractError("profile_inventory_mismatch")
        observed = {
            p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
        }
        if observed != listed | {"cover_page.json", header_path, inventory_path}:
            raise ContractError("profile_inventory_mismatch")
        identity = _validate_identity(cover["identity"])
        if (
            header["semantic_identity_digest"] != identity["semantic_digest"]
            or cover["training"] != identity["training_contract"]
            or cover["authority"] != identity["authority"]
        ):
            raise ContractError("payload_semantic_projection_invalid")
        _validate_selected_payloads(root, listed, identity)
        return Result(True)
    except ContractError as exc:
        return Result(False, (exc.code,))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    result = (
        validate_directory(args.path)
        if args.path.is_dir()
        else Result(False, ("schema_version_unsupported",))
    )
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
