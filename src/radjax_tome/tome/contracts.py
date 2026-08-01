"""Pure M5 canonical Tome identity, manifest, and cover contracts.

The v3 cover is intentionally not a writer yet.  Historical v2 covers and
package-cover v1 remain their native validation contracts until the M5C--M5E
work is explicitly authorized.  These functions define the profile-neutral
semantic boundary that those later adapters must share.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

CANONICAL_TOME_COVER_SCHEMA = "radjax_tome_cover_v3"
CANONICAL_TOME_STUDENT_CONSUMPTION_V2_COVER_SCHEMA = (
    "radjax_tome_cover_v3_student_consumption_v2"
)
CANONICAL_TOME_STUDENT_CONSUMPTION_V3_COVER_SCHEMA = (
    "radjax_tome_cover_v3_student_consumption_v3"
)
CANONICAL_TOME_STUDENT_CONSUMPTION_V4_COVER_SCHEMA = (
    "radjax_tome_cover_v3_student_consumption_v4"
)
CANONICAL_CONTENT_MANIFEST_SCHEMA = "tome_content_manifest_v2"
TOME_SEMANTIC_IDENTITY_SCHEMA = "radjax_tome_semantic_identity_v1"
HISTORICAL_PACKAGE_COVER_SCHEMA = "radjax_tome_package_cover_v1"

_PACKAGE_PROFILES = frozenset({"unpacked", "student", "full_debug_provenance"})
_CONTENT_CLASSIFICATIONS = frozenset(
    {
        "training_critical",
        "integrity_or_provenance",
        "diagnostic",
        "human_readable",
        "operational",
    }
)
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IDENTITY_KEYS = frozenset(
    {
        "schema_version",
        "training_payload",
        "training_contract",
        "authority",
        "semantic_digest",
    }
)
_TRAINING_PAYLOAD_ENTRY_KEYS = frozenset({"logical_id", "semantic_digest"})
_CONTENT_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "profile",
        "semantic_identity_digest",
        "inventory",
        "manifest_digest",
    }
)
_INVENTORY_ENTRY_KEYS = frozenset(
    {
        "path",
        "sha256",
        "size_bytes",
        "classification",
        "training_authoritative",
    }
)
_COVER_KEYS = frozenset(
    {
        "schema_version",
        "identity",
        "training",
        "package",
        "manifests",
        "authority",
        "provenance",
        "validation",
    }
)
_STUDENT_CONSUMPTION_V2_COVER_KEYS = _COVER_KEYS | frozenset({"student_consumption"})
_STUDENT_CONSUMPTION_V3_COVER_KEYS = _STUDENT_CONSUMPTION_V2_COVER_KEYS
_PACKAGE_KEYS = frozenset({"profile", "transport"})
_MANIFEST_WRAPPER_KEYS = frozenset({"content"})
_STUDENT_CONSUMPTION_V2_KEYS = frozenset(
    {
        "profile_id",
        "manifest_path",
        "manifest_sha256",
        "semantic_digest",
        "digest_method",
        "required_capabilities",
    }
)
_STUDENT_CONSUMPTION_V2_REQUIRED_KEYS = frozenset(
    {"profile_id", "manifest_path", "manifest_sha256", "semantic_digest"}
)
_RUNTIME_ONLY_KEYS = frozenset(
    {
        "created_at",
        "generated_at",
        "package_profile",
        "transport",
        "archive",
        "manifest_digest",
        "raw_artifact_digest",
    }
)


@dataclass(frozen=True, order=True)
class TrainingPayloadEntry:
    """One logical, training-authoritative payload digest."""

    logical_id: str
    semantic_digest: str

    def to_dict(self) -> dict[str, str]:
        return {"logical_id": self.logical_id, "semantic_digest": self.semantic_digest}


@dataclass(frozen=True)
class TomeSemanticIdentity:
    """Profile- and transport-neutral identity of training-authoritative data."""

    training_payload: tuple[TrainingPayloadEntry, ...]
    training_contract: dict[str, Any]
    authority: dict[str, Any]
    semantic_digest: str
    schema_version: str = TOME_SEMANTIC_IDENTITY_SCHEMA

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "training_payload": [entry.to_dict() for entry in self.training_payload],
            "training_contract": self.training_contract,
            "authority": self.authority,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "semantic_digest": self.semantic_digest}


@dataclass(frozen=True, order=True)
class PackageInventoryEntry:
    """A physical package member with a raw-byte integrity digest."""

    path: str
    sha256: str
    size_bytes: int
    classification: str
    training_authoritative: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "classification": self.classification,
            "training_authoritative": self.training_authoritative,
        }


@dataclass(frozen=True)
class CanonicalContentManifest:
    """Profile-specific inventory that references, but never defines, identity."""

    profile: str
    semantic_identity_digest: str
    inventory: tuple[PackageInventoryEntry, ...]
    manifest_digest: str
    schema_version: str = CANONICAL_CONTENT_MANIFEST_SCHEMA

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile": self.profile,
            "semantic_identity_digest": self.semantic_identity_digest,
            "inventory": [entry.to_dict() for entry in self.inventory],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "manifest_digest": self.manifest_digest}


def canonical_json_digest(payload: Mapping[str, Any]) -> str:
    """Return the contract-wide compact sorted JSON SHA-256 digest."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def build_tome_semantic_identity(
    *,
    training_payload: Sequence[TrainingPayloadEntry],
    training_contract: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> TomeSemanticIdentity:
    """Build identity from semantic training material only.

    Runtime creation data, package profiles, transport wrapping, manifest
    digests, and raw artifact digests are rejected here rather than normalized
    away.  This makes the boundary inspectable and prevents debug receipts
    from silently becoming training identity.
    """

    entries = _sorted_training_payload(training_payload)
    contract = _require_json_object(training_contract, "training_contract")
    authority_payload = _require_json_object(authority, "authority")
    _reject_runtime_only_keys(contract, "training_contract")
    _reject_runtime_only_keys(authority_payload, "authority")
    identity_payload = {
        "schema_version": TOME_SEMANTIC_IDENTITY_SCHEMA,
        "training_payload": [entry.to_dict() for entry in entries],
        "training_contract": contract,
        "authority": authority_payload,
    }
    return TomeSemanticIdentity(
        training_payload=entries,
        training_contract=contract,
        authority=authority_payload,
        semantic_digest=canonical_json_digest(identity_payload),
    )


def validate_canonical_tome_semantic_identity(identity: Mapping[str, Any]) -> None:
    """Validate the complete v1 identity structure and recompute its digest.

    The identity contract is closed at this version.  In particular, callers
    must not compare a merely plausible digest string without first proving it
    still binds the exact, ordered semantic payload carried beside it.
    """

    payload = _require_exact_object(identity, "identity", _IDENTITY_KEYS)
    if payload["schema_version"] != TOME_SEMANTIC_IDENTITY_SCHEMA:
        raise ValueError("semantic identity contract is unsupported")
    entries = payload["training_payload"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("semantic identity training_payload must be a non-empty list")
    previous_logical_id: str | None = None
    validated_entries: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        entry_payload = _require_exact_object(
            entry,
            f"identity.training_payload[{index}]",
            _TRAINING_PAYLOAD_ENTRY_KEYS,
        )
        logical_id = entry_payload["logical_id"]
        if not isinstance(logical_id, str) or not logical_id:
            raise ValueError(
                f"identity.training_payload[{index}].logical_id must be non-empty"
            )
        if previous_logical_id is not None and logical_id <= previous_logical_id:
            raise ValueError(
                "semantic identity training_payload logical_id values must be "
                "strictly sorted and unique"
            )
        previous_logical_id = logical_id
        _require_sha256(
            entry_payload["semantic_digest"],
            f"identity.training_payload[{index}].semantic_digest",
        )
        validated_entries.append(entry_payload)
    training_contract = _require_json_object(
        payload["training_contract"], "identity.training_contract"
    )
    authority = _require_json_object(payload["authority"], "identity.authority")
    _reject_runtime_only_keys(training_contract, "identity.training_contract")
    _reject_runtime_only_keys(authority, "identity.authority")
    _require_sha256(payload["semantic_digest"], "identity.semantic_digest")
    expected = canonical_json_digest(
        {
            "schema_version": payload["schema_version"],
            "training_payload": validated_entries,
            "training_contract": training_contract,
            "authority": authority,
        }
    )
    if payload["semantic_digest"] != expected:
        raise ValueError("semantic identity semantic_digest mismatch")


def build_canonical_content_manifest(
    *,
    profile: str,
    semantic_identity: TomeSemanticIdentity,
    inventory: Sequence[PackageInventoryEntry],
) -> CanonicalContentManifest:
    """Build a profile inventory, excluding the cover to avoid circular hashing."""

    validate_canonical_tome_semantic_identity(semantic_identity.to_dict())
    if profile not in _PACKAGE_PROFILES:
        raise ValueError(f"unsupported canonical package profile: {profile}")
    entries = _sorted_inventory(inventory)
    payload = {
        "schema_version": CANONICAL_CONTENT_MANIFEST_SCHEMA,
        "profile": profile,
        "semantic_identity_digest": semantic_identity.semantic_digest,
        "inventory": [entry.to_dict() for entry in entries],
    }
    manifest = CanonicalContentManifest(
        profile=profile,
        semantic_identity_digest=semantic_identity.semantic_digest,
        inventory=entries,
        manifest_digest=canonical_json_digest(payload),
    )
    validate_canonical_content_manifest(manifest.to_dict())
    return manifest


def validate_canonical_content_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate the complete closed v2 profile-inventory contract."""

    payload = _require_exact_object(
        manifest, "content manifest", _CONTENT_MANIFEST_KEYS
    )
    if payload["schema_version"] != CANONICAL_CONTENT_MANIFEST_SCHEMA:
        raise ValueError("canonical content manifest schema_version mismatch")
    if payload["profile"] not in _PACKAGE_PROFILES:
        raise ValueError("canonical content manifest profile is invalid")
    _require_sha256(
        payload["semantic_identity_digest"],
        "canonical content manifest semantic_identity_digest",
    )
    inventory = payload["inventory"]
    if not isinstance(inventory, list):
        raise ValueError("canonical content manifest inventory must be a list")
    previous_path: str | None = None
    validated_inventory: list[dict[str, Any]] = []
    for index, entry in enumerate(inventory):
        entry_payload = _require_exact_object(
            entry,
            f"canonical content manifest inventory[{index}]",
            _INVENTORY_ENTRY_KEYS,
        )
        path = entry_payload["path"]
        _require_normalized_relative_path(
            path, f"canonical content manifest inventory[{index}].path"
        )
        if path == "cover_page.json":
            raise ValueError("canonical content manifest must exclude cover_page.json")
        if previous_path is not None and path <= previous_path:
            raise ValueError(
                "canonical content manifest inventory paths must be strictly "
                "sorted and unique"
            )
        previous_path = path
        _require_sha256(
            entry_payload["sha256"],
            f"canonical content manifest inventory[{index}].sha256",
        )
        size_bytes = entry_payload["size_bytes"]
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
        ):
            raise ValueError(
                f"canonical content manifest inventory[{index}].size_bytes "
                "must be a non-negative integer"
            )
        if entry_payload["classification"] not in _CONTENT_CLASSIFICATIONS:
            raise ValueError(
                f"canonical content manifest inventory[{index}].classification "
                "is invalid"
            )
        if not isinstance(entry_payload["training_authoritative"], bool):
            raise ValueError(
                "canonical content manifest inventory["
                f"{index}].training_authoritative must be boolean"
            )
        validated_inventory.append(entry_payload)
    _require_sha256(payload["manifest_digest"], "canonical content manifest digest")
    expected = canonical_json_digest(
        {
            "schema_version": payload["schema_version"],
            "profile": payload["profile"],
            "semantic_identity_digest": payload["semantic_identity_digest"],
            "inventory": validated_inventory,
        }
    )
    if payload["manifest_digest"] != expected:
        raise ValueError("canonical content manifest digest mismatch")


def build_canonical_tome_cover(
    *,
    semantic_identity: TomeSemanticIdentity,
    content_manifest: CanonicalContentManifest,
    provenance: Mapping[str, Any],
    validation: Mapping[str, Any],
    transport: str = "directory",
) -> dict[str, Any]:
    """Build the nested public v3 cover without materializing any artifact."""

    validate_canonical_tome_semantic_identity(semantic_identity.to_dict())
    validate_canonical_content_manifest(content_manifest.to_dict())
    if transport not in {"directory", "rtome", "tgz"}:
        raise ValueError(f"unsupported canonical transport: {transport}")
    if content_manifest.semantic_identity_digest != semantic_identity.semantic_digest:
        raise ValueError("content manifest does not reference the supplied identity")
    provenance_payload = _require_json_object(provenance, "provenance")
    validation_payload = _require_json_object(validation, "validation")
    cover = {
        "schema_version": CANONICAL_TOME_COVER_SCHEMA,
        "identity": semantic_identity.to_dict(),
        "training": dict(semantic_identity.training_contract),
        "package": {
            "profile": content_manifest.profile,
            "transport": transport,
        },
        "manifests": {"content": content_manifest.to_dict()},
        "authority": dict(semantic_identity.authority),
        "provenance": provenance_payload,
        "validation": validation_payload,
    }
    validate_canonical_tome_cover(cover)
    return cover


def validate_canonical_tome_cover(cover: Mapping[str, Any]) -> None:
    """Fail closed on a base-v3 or closed Student-consumption-v2 cover.

    The additive v2 declaration is deliberately outside the base v3 semantic
    identity and the profile inventory's identity reference.  It identifies
    independently-digested derived sidecars; it does not make those sidecars
    retroactively appear in the historical base training payload.
    """

    declared_schema = (
        cover.get("schema_version") if isinstance(cover, Mapping) else None
    )
    if declared_schema == CANONICAL_TOME_COVER_SCHEMA:
        payload = _require_exact_object(cover, "canonical cover", _COVER_KEYS)
    elif declared_schema in {
        CANONICAL_TOME_STUDENT_CONSUMPTION_V2_COVER_SCHEMA,
        CANONICAL_TOME_STUDENT_CONSUMPTION_V3_COVER_SCHEMA,
        CANONICAL_TOME_STUDENT_CONSUMPTION_V4_COVER_SCHEMA,
    }:
        payload = _require_exact_object(
            cover,
            "canonical Student-consumption cover",
            _STUDENT_CONSUMPTION_V3_COVER_KEYS,
        )
    else:
        raise ValueError("canonical cover schema_version mismatch")
    identity = _require_exact_object(payload["identity"], "identity", _IDENTITY_KEYS)
    validate_canonical_tome_semantic_identity(identity)
    training = _require_json_object(payload["training"], "canonical cover training")
    authority = _require_json_object(payload["authority"], "canonical cover authority")
    if training != identity["training_contract"]:
        raise ValueError("canonical cover training does not match identity")
    if authority != identity["authority"]:
        raise ValueError("canonical cover authority does not match identity")
    package = _require_exact_object(
        payload["package"], "canonical cover package", _PACKAGE_KEYS
    )
    if package["profile"] not in _PACKAGE_PROFILES:
        raise ValueError("canonical cover package profile is invalid")
    if package["transport"] not in {"directory", "rtome", "tgz"}:
        raise ValueError("canonical cover package transport is invalid")
    manifest_wrapper = _require_exact_object(
        payload["manifests"], "canonical cover manifests", _MANIFEST_WRAPPER_KEYS
    )
    manifest = _require_exact_object(
        manifest_wrapper["content"],
        "canonical cover manifests.content",
        _CONTENT_MANIFEST_KEYS,
    )
    validate_canonical_content_manifest(manifest)
    if manifest["semantic_identity_digest"] != identity["semantic_digest"]:
        raise ValueError("canonical content manifest identity reference mismatch")
    if manifest["profile"] != package["profile"]:
        raise ValueError("canonical content manifest profile mismatch")
    _require_json_object(payload["provenance"], "canonical cover provenance")
    _require_json_object(payload["validation"], "canonical cover validation")
    if declared_schema == CANONICAL_TOME_STUDENT_CONSUMPTION_V2_COVER_SCHEMA:
        _validate_student_consumption_v2_declaration(payload["student_consumption"])
    if declared_schema == CANONICAL_TOME_STUDENT_CONSUMPTION_V3_COVER_SCHEMA:
        _validate_student_consumption_v3_declaration(payload["student_consumption"])
    if declared_schema == CANONICAL_TOME_STUDENT_CONSUMPTION_V4_COVER_SCHEMA:
        _validate_student_consumption_v4_declaration(payload["student_consumption"])


def _validate_student_consumption_v2_declaration(value: Any) -> None:
    """Validate the closed additive declaration without changing v3 identity."""

    payload = _require_allowed_object(
        value,
        "canonical cover student_consumption",
        required_keys=_STUDENT_CONSUMPTION_V2_REQUIRED_KEYS,
        allowed_keys=_STUDENT_CONSUMPTION_V2_KEYS,
    )
    if payload["profile_id"] != "native_v3_student_v2":
        raise ValueError("canonical cover Student-consumption profile is invalid")
    if payload["manifest_path"] != "manifests/student_consumption_v2.json":
        raise ValueError("canonical cover Student-consumption manifest path is invalid")
    _require_sha256(
        payload["manifest_sha256"],
        "canonical cover Student-consumption manifest_sha256",
    )
    _require_sha256(
        payload["semantic_digest"],
        "canonical cover Student-consumption semantic_digest",
    )
    if payload.get("digest_method", "sha256") != "sha256":
        raise ValueError("canonical cover Student-consumption digest_method is invalid")
    capabilities = payload.get("required_capabilities", [])
    if not isinstance(capabilities, list) or capabilities:
        raise ValueError(
            "canonical cover Student-consumption required_capabilities must be empty"
        )


def _validate_student_consumption_v3_declaration(value: Any) -> None:
    """Validate the explicit v3 successor without accepting a v2 fallback."""

    payload = _require_allowed_object(
        value,
        "canonical cover Student-consumption",
        required_keys=_STUDENT_CONSUMPTION_V2_REQUIRED_KEYS,
        allowed_keys=_STUDENT_CONSUMPTION_V2_KEYS,
    )
    if payload["profile_id"] != "native_v3_student_v3":
        raise ValueError("canonical cover Student-consumption profile is invalid")
    if payload["manifest_path"] != "manifests/student_consumption_v3.json":
        raise ValueError("canonical cover Student-consumption manifest path is invalid")
    _require_sha256(
        payload["manifest_sha256"],
        "canonical cover Student-consumption manifest_sha256",
    )
    _require_sha256(
        payload["semantic_digest"],
        "canonical cover Student-consumption semantic_digest",
    )
    if payload.get("digest_method", "sha256") != "sha256":
        raise ValueError("canonical cover Student-consumption digest_method is invalid")
    capabilities = payload.get("required_capabilities", [])
    if not isinstance(capabilities, list) or capabilities:
        raise ValueError(
            "canonical cover Student-consumption required_capabilities must be empty"
        )


def _validate_student_consumption_v4_declaration(value: Any) -> None:
    """Validate the explicit v4 successor without a historical fallback."""

    payload = _require_allowed_object(
        value,
        "canonical cover student_consumption",
        required_keys=_STUDENT_CONSUMPTION_V2_REQUIRED_KEYS,
        allowed_keys=_STUDENT_CONSUMPTION_V2_KEYS,
    )
    if payload["profile_id"] != "native_v3_student_v4":
        raise ValueError("canonical cover Student-consumption profile is invalid")
    if payload["manifest_path"] != "manifests/student_consumption_v4.json":
        raise ValueError("canonical cover Student-consumption manifest path is invalid")
    _require_sha256(
        payload["manifest_sha256"],
        "canonical cover Student-consumption manifest_sha256",
    )
    _require_sha256(
        payload["semantic_digest"],
        "canonical cover Student-consumption semantic_digest",
    )
    if payload.get("digest_method", "sha256") != "sha256":
        raise ValueError("canonical cover Student-consumption digest_method is invalid")
    capabilities = payload.get("required_capabilities", [])
    if not isinstance(capabilities, list) or capabilities:
        raise ValueError(
            "canonical cover Student-consumption required_capabilities must be empty"
        )


def compare_canonical_tome_identities(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    """Compare identities only when their explicit contract versions match."""

    validate_canonical_tome_semantic_identity(left)
    validate_canonical_tome_semantic_identity(right)
    return left["semantic_digest"] == right["semantic_digest"]


def _sorted_training_payload(
    entries: Sequence[TrainingPayloadEntry],
) -> tuple[TrainingPayloadEntry, ...]:
    if not entries:
        raise ValueError("semantic identity requires training payload entries")
    if any(not isinstance(entry, TrainingPayloadEntry) for entry in entries):
        raise ValueError("training payload entries must be TrainingPayloadEntry values")
    if any(
        not isinstance(entry.logical_id, str) or not entry.logical_id
        for entry in entries
    ):
        raise ValueError(
            "training payload logical_id values must be unique and non-empty"
        )
    normalized = tuple(sorted(entries, key=lambda entry: entry.logical_id))
    identifiers = [entry.logical_id for entry in normalized]
    if any(not identifier for identifier in identifiers) or len(
        set(identifiers)
    ) != len(identifiers):
        raise ValueError(
            "training payload logical_id values must be unique and non-empty"
        )
    for entry in normalized:
        _require_sha256(entry.semantic_digest, "training payload semantic_digest")
    return normalized


def _sorted_inventory(
    entries: Sequence[PackageInventoryEntry],
) -> tuple[PackageInventoryEntry, ...]:
    if any(not isinstance(entry, PackageInventoryEntry) for entry in entries):
        raise ValueError(
            "package inventory entries must be PackageInventoryEntry values"
        )
    if any(not isinstance(entry.path, str) for entry in entries):
        raise ValueError("package inventory paths must be strings")
    normalized = tuple(sorted(entries, key=lambda entry: entry.path))
    paths = [entry.path for entry in normalized]
    for path in paths:
        _require_normalized_relative_path(path, "package inventory path")
    if len(set(paths)) != len(paths):
        raise ValueError("package inventory paths must be unique")
    for entry in normalized:
        _require_sha256(entry.sha256, "package inventory sha256")
        if (
            isinstance(entry.size_bytes, bool)
            or not isinstance(entry.size_bytes, int)
            or entry.size_bytes < 0
        ):
            raise ValueError("package inventory sizes must be non-negative integers")
        if entry.classification not in _CONTENT_CLASSIFICATIONS:
            raise ValueError("package inventory classification is invalid")
        if not isinstance(entry.training_authoritative, bool):
            raise ValueError("package inventory training_authoritative must be boolean")
    return normalized


def _require_exact_object(
    value: Any,
    location: str,
    expected_keys: frozenset[str],
) -> dict[str, Any]:
    payload = _require_json_object(value, location)
    keys = frozenset(payload)
    missing = sorted(expected_keys - keys)
    unexpected = sorted(keys - expected_keys)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise ValueError(f"{location} must contain exact keys: " + "; ".join(details))
    return payload


def _require_allowed_object(
    value: Any,
    location: str,
    *,
    required_keys: frozenset[str],
    allowed_keys: frozenset[str],
) -> dict[str, Any]:
    """Validate a closed object with explicitly documented optional fields."""

    payload = _require_json_object(value, location)
    keys = frozenset(payload)
    missing = sorted(required_keys - keys)
    unexpected = sorted(keys - allowed_keys)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise ValueError(f"{location} must contain allowed keys: " + "; ".join(details))
    return payload


def _require_json_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    _validate_json_value(value, name)
    return dict(value)


def _validate_json_value(value: Any, location: str) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if math.isfinite(value):
            return
        raise ValueError(f"{location} contains a non-finite number")
    if isinstance(value, list):
        for index, nested in enumerate(value):
            _validate_json_value(nested, f"{location}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{location} contains a non-string object key")
            _validate_json_value(nested, f"{location}.{key}")
        return
    raise ValueError(f"{location} must contain only JSON values")


def _require_sha256(value: Any, location: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{location} must be sha256: followed by 64 lowercase hex")


def _require_normalized_relative_path(value: Any, location: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{location} must be a non-empty normalized relative path")
    if "\\" in value:
        raise ValueError(f"{location} must use normalized POSIX separators")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value == "."
        or value != path.as_posix()
        or any(part in {".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{location} must be a normalized relative path")


def _reject_runtime_only_keys(value: Any, location: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key) in _RUNTIME_ONLY_KEYS:
                raise ValueError(f"{location} contains runtime-only key {key}")
            _reject_runtime_only_keys(nested, f"{location}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_runtime_only_keys(nested, f"{location}[{index}]")


__all__ = [
    "CANONICAL_CONTENT_MANIFEST_SCHEMA",
    "CANONICAL_TOME_COVER_SCHEMA",
    "CANONICAL_TOME_STUDENT_CONSUMPTION_V2_COVER_SCHEMA",
    "CANONICAL_TOME_STUDENT_CONSUMPTION_V3_COVER_SCHEMA",
    "CANONICAL_TOME_STUDENT_CONSUMPTION_V4_COVER_SCHEMA",
    "HISTORICAL_PACKAGE_COVER_SCHEMA",
    "TOME_SEMANTIC_IDENTITY_SCHEMA",
    "CanonicalContentManifest",
    "PackageInventoryEntry",
    "TomeSemanticIdentity",
    "TrainingPayloadEntry",
    "build_canonical_content_manifest",
    "build_canonical_tome_cover",
    "build_tome_semantic_identity",
    "canonical_json_digest",
    "compare_canonical_tome_identities",
    "validate_canonical_content_manifest",
    "validate_canonical_tome_semantic_identity",
    "validate_canonical_tome_cover",
]
