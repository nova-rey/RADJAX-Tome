"""Pure M5 canonical Tome identity, manifest, and cover contracts.

The v3 cover is intentionally not a writer yet.  Historical v2 covers and
package-cover v1 remain their native validation contracts until the M5C--M5E
work is explicitly authorized.  These functions define the profile-neutral
semantic boundary that those later adapters must share.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

CANONICAL_TOME_COVER_SCHEMA = "radjax_tome_cover_v3"
CANONICAL_CONTENT_MANIFEST_SCHEMA = "tome_content_manifest_v2"
TOME_SEMANTIC_IDENTITY_SCHEMA = "radjax_tome_semantic_identity_v1"

_PACKAGE_PROFILES = frozenset({"unpacked", "student", "full_debug_provenance"})
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
_COVER_SECTIONS = (
    "identity",
    "training",
    "package",
    "manifests",
    "authority",
    "provenance",
    "validation",
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
    contract = _json_object(training_contract, "training_contract")
    authority_payload = _json_object(authority, "authority")
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


def build_canonical_content_manifest(
    *,
    profile: str,
    semantic_identity: TomeSemanticIdentity,
    inventory: Sequence[PackageInventoryEntry],
) -> CanonicalContentManifest:
    """Build a profile inventory, excluding the cover to avoid circular hashing."""

    if profile not in _PACKAGE_PROFILES:
        raise ValueError(f"unsupported canonical package profile: {profile}")
    entries = _sorted_inventory(inventory)
    if any(entry.path == "cover_page.json" for entry in entries):
        raise ValueError("canonical content manifest must exclude cover_page.json")
    payload = {
        "schema_version": CANONICAL_CONTENT_MANIFEST_SCHEMA,
        "profile": profile,
        "semantic_identity_digest": semantic_identity.semantic_digest,
        "inventory": [entry.to_dict() for entry in entries],
    }
    return CanonicalContentManifest(
        profile=profile,
        semantic_identity_digest=semantic_identity.semantic_digest,
        inventory=entries,
        manifest_digest=canonical_json_digest(payload),
    )


def build_canonical_tome_cover(
    *,
    semantic_identity: TomeSemanticIdentity,
    content_manifest: CanonicalContentManifest,
    provenance: Mapping[str, Any],
    validation: Mapping[str, Any],
    transport: str = "directory",
) -> dict[str, Any]:
    """Build the nested public v3 cover without materializing any artifact."""

    if transport not in {"directory", "rtome", "tgz"}:
        raise ValueError(f"unsupported canonical transport: {transport}")
    if content_manifest.semantic_identity_digest != semantic_identity.semantic_digest:
        raise ValueError("content manifest does not reference the supplied identity")
    provenance_payload = _json_object(provenance, "provenance")
    validation_payload = _json_object(validation, "validation")
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
    """Fail closed on a malformed v3 cover or mismatched nested contracts."""

    if cover.get("schema_version") != CANONICAL_TOME_COVER_SCHEMA:
        raise ValueError("canonical cover schema_version mismatch")
    missing = [section for section in _COVER_SECTIONS if section not in cover]
    if missing:
        raise ValueError(
            "canonical cover missing nested sections: " + ", ".join(missing)
        )
    identity = _json_object(cover["identity"], "identity")
    manifest_wrapper = _json_object(cover["manifests"], "manifests")
    package = _json_object(cover["package"], "package")
    manifest = _json_object(manifest_wrapper.get("content"), "manifests.content")
    if identity.get("schema_version") != TOME_SEMANTIC_IDENTITY_SCHEMA:
        raise ValueError("canonical cover identity schema_version mismatch")
    _reject_runtime_only_keys(identity, "identity")
    identity_payload = {
        key: identity.get(key)
        for key in (
            "schema_version",
            "training_payload",
            "training_contract",
            "authority",
        )
    }
    if identity.get("semantic_digest") != canonical_json_digest(identity_payload):
        raise ValueError("canonical cover identity semantic_digest mismatch")
    if cover["training"] != identity.get("training_contract"):
        raise ValueError("canonical cover training does not match identity")
    if cover["authority"] != identity.get("authority"):
        raise ValueError("canonical cover authority does not match identity")
    if package.get("profile") not in _PACKAGE_PROFILES:
        raise ValueError("canonical cover package profile is invalid")
    if package.get("transport") not in {"directory", "rtome", "tgz"}:
        raise ValueError("canonical cover package transport is invalid")
    if manifest.get("schema_version") != CANONICAL_CONTENT_MANIFEST_SCHEMA:
        raise ValueError("canonical content manifest schema_version mismatch")
    manifest_payload = {
        key: manifest.get(key)
        for key in (
            "schema_version",
            "profile",
            "semantic_identity_digest",
            "inventory",
        )
    }
    if manifest.get("manifest_digest") != canonical_json_digest(manifest_payload):
        raise ValueError("canonical content manifest digest mismatch")
    if manifest.get("semantic_identity_digest") != identity.get("semantic_digest"):
        raise ValueError("canonical content manifest identity reference mismatch")
    if manifest.get("profile") != package.get("profile"):
        raise ValueError("canonical content manifest profile mismatch")
    inventory = manifest.get("inventory")
    if not isinstance(inventory, list):
        raise ValueError("canonical content manifest inventory must be a list")
    if any(
        not isinstance(item, dict) or item.get("path") == "cover_page.json"
        for item in inventory
    ):
        raise ValueError("canonical content manifest must exclude cover_page.json")


def compare_canonical_tome_identities(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    """Compare identities only when their explicit contract versions match."""

    if left.get("schema_version") != TOME_SEMANTIC_IDENTITY_SCHEMA:
        raise ValueError("left semantic identity contract is unsupported")
    if right.get("schema_version") != TOME_SEMANTIC_IDENTITY_SCHEMA:
        raise ValueError("right semantic identity contract is unsupported")
    return left.get("semantic_digest") == right.get("semantic_digest")


def _sorted_training_payload(
    entries: Sequence[TrainingPayloadEntry],
) -> tuple[TrainingPayloadEntry, ...]:
    if not entries:
        raise ValueError("semantic identity requires training payload entries")
    normalized = tuple(sorted(entries, key=lambda entry: entry.logical_id))
    identifiers = [entry.logical_id for entry in normalized]
    if any(not identifier for identifier in identifiers) or len(
        set(identifiers)
    ) != len(identifiers):
        raise ValueError(
            "training payload logical_id values must be unique and non-empty"
        )
    if any(not entry.semantic_digest.startswith("sha256:") for entry in normalized):
        raise ValueError(
            "training payload semantic_digest values must be sha256 digests"
        )
    return normalized


def _sorted_inventory(
    entries: Sequence[PackageInventoryEntry],
) -> tuple[PackageInventoryEntry, ...]:
    normalized = tuple(sorted(entries, key=lambda entry: entry.path))
    paths = [entry.path for entry in normalized]
    if any(
        not path or path.startswith("/") or ".." in path.split("/") for path in paths
    ):
        raise ValueError("package inventory paths must be relative and normalized")
    if len(set(paths)) != len(paths):
        raise ValueError("package inventory paths must be unique")
    if any(entry.size_bytes < 0 for entry in normalized):
        raise ValueError("package inventory sizes must be non-negative")
    if any(not entry.sha256.startswith("sha256:") for entry in normalized):
        raise ValueError("package inventory sha256 values must be sha256 digests")
    return normalized


def _json_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be JSON serializable") from exc
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise ValueError(f"{name} must be a JSON object")
    return decoded


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
    "validate_canonical_tome_cover",
]
