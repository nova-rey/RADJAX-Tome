#!/usr/bin/env python3
"""Independent portable validator for the published RADJAX-Tome v1 contract.

This tool intentionally imports only the Python standard library. It is a
conformance implementation, not a writer or a production runtime dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

SHA256_PREFIX = "sha256:"
SHA256_LENGTH = len(SHA256_PREFIX) + 64
PROFILES = {"unpacked", "student", "full_debug_provenance"}
TRANSPORTS = {"directory", "rtome", "tgz"}
CLASSIFICATIONS = {
    "training_critical",
    "integrity_or_provenance",
    "diagnostic",
    "human_readable",
    "operational",
}
RUNTIME_KEYS = {
    "created_at",
    "generated_at",
    "package_profile",
    "transport",
    "archive",
    "manifest_digest",
    "raw_artifact_digest",
}


@dataclass(frozen=True)
class Result:
    ok: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ContractError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(detail)


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError("shape_invalid", f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> dict[str, Any]:
    try:
        return parse_json(path.read_text(encoding="utf-8"), str(path))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("shape_invalid", f"invalid JSON {path}: {exc}") from exc


def parse_json(text: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(text, object_pairs_hook=_pairs_no_duplicates)
    except json.JSONDecodeError as exc:
        raise ContractError("shape_invalid", f"invalid JSON {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError("shape_invalid", f"JSON root must be object: {label}")
    return value


def canonical_digest(value: dict[str, Any]) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    except (TypeError, ValueError) as exc:
        raise ContractError(
            "shape_invalid", f"value is not canonical JSON: {exc}"
        ) from exc
    return SHA256_PREFIX + hashlib.sha256(encoded).hexdigest()


def _exact_object(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError("shape_invalid", f"{label} must be object")
    actual = set(value)
    if actual != keys:
        detail = f"{label} keys differ; missing={sorted(keys - actual)}"
        detail += f", extra={sorted(actual - keys)}"
        raise ContractError(
            "shape_invalid",
            detail,
        )
    return value


def _sha256(value: Any, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != SHA256_LENGTH
        or not value.startswith(SHA256_PREFIX)
        or any(char not in "0123456789abcdef" for char in value[len(SHA256_PREFIX) :])
    ):
        raise ContractError(
            "digest_syntax_invalid", f"{label} is not sha256: plus 64 lowercase hex"
        )


def _safe_relative_path(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ContractError(
            "path_unsafe", f"{label} is not normalized relative POSIX path"
        )
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value == "."
        or value != path.as_posix()
        or any(part in {".", ".."} for part in path.parts)
    ):
        raise ContractError(
            "path_unsafe", f"{label} is not normalized relative POSIX path"
        )


def _reject_runtime(value: Any, label: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in RUNTIME_KEYS:
                raise ContractError(
                    "shape_invalid", f"{label} contains runtime-only key {key}"
                )
            _reject_runtime(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_runtime(child, f"{label}[{index}]")


def validate_identity(identity: Any) -> None:
    value = _exact_object(
        identity,
        {
            "schema_version",
            "training_payload",
            "training_contract",
            "authority",
            "semantic_digest",
        },
        "identity",
    )
    if value["schema_version"] != "radjax_tome_semantic_identity_v1":
        raise ContractError(
            "schema_version_unsupported", "unsupported semantic identity schema"
        )
    entries = value["training_payload"]
    if not isinstance(entries, list) or not entries:
        raise ContractError("shape_invalid", "training_payload must be non-empty list")
    previous: str | None = None
    for index, entry in enumerate(entries):
        entry = _exact_object(
            entry, {"logical_id", "semantic_digest"}, f"training_payload[{index}]"
        )
        logical_id = entry["logical_id"]
        if not isinstance(logical_id, str) or not logical_id:
            raise ContractError("shape_invalid", "logical_id must be non-empty string")
        if previous is not None and logical_id <= previous:
            raise ContractError(
                "ordering_invalid", "logical IDs must be sorted and unique"
            )
        previous = logical_id
        _sha256(entry["semantic_digest"], "training payload digest")
    if not isinstance(value["training_contract"], dict) or not isinstance(
        value["authority"], dict
    ):
        raise ContractError(
            "shape_invalid", "training_contract and authority must be objects"
        )
    _reject_runtime(value["training_contract"], "training_contract")
    _reject_runtime(value["authority"], "authority")
    _sha256(value["semantic_digest"], "semantic digest")
    projection = {
        key: value[key]
        for key in (
            "schema_version",
            "training_payload",
            "training_contract",
            "authority",
        )
    }
    if value["semantic_digest"] != canonical_digest(projection):
        raise ContractError("digest_mismatch", "semantic identity digest mismatch")


def validate_manifest(manifest: Any) -> None:
    value = _exact_object(
        manifest,
        {
            "schema_version",
            "profile",
            "semantic_identity_digest",
            "inventory",
            "manifest_digest",
        },
        "content manifest",
    )
    if value["schema_version"] != "tome_content_manifest_v2":
        raise ContractError(
            "schema_version_unsupported", "unsupported content manifest schema"
        )
    if value["profile"] not in PROFILES:
        raise ContractError("profile_inventory_mismatch", "unsupported profile")
    _sha256(value["semantic_identity_digest"], "manifest semantic identity digest")
    inventory = value["inventory"]
    if not isinstance(inventory, list):
        raise ContractError("shape_invalid", "inventory must be list")
    previous: str | None = None
    for index, entry in enumerate(inventory):
        entry = _exact_object(
            entry,
            {
                "path",
                "sha256",
                "size_bytes",
                "classification",
                "training_authoritative",
            },
            f"inventory[{index}]",
        )
        path = entry["path"]
        _safe_relative_path(path, f"inventory[{index}].path")
        if path == "cover_page.json":
            raise ContractError(
                "profile_inventory_mismatch",
                "cover_page.json must be excluded from inventory",
            )
        if previous is not None and path <= previous:
            raise ContractError(
                "ordering_invalid", "inventory paths must be sorted and unique"
            )
        previous = path
        _sha256(entry["sha256"], "inventory raw digest")
        if (
            isinstance(entry["size_bytes"], bool)
            or not isinstance(entry["size_bytes"], int)
            or entry["size_bytes"] < 0
        ):
            raise ContractError(
                "shape_invalid", "inventory size_bytes must be non-negative integer"
            )
        if entry["classification"] not in CLASSIFICATIONS or not isinstance(
            entry["training_authoritative"], bool
        ):
            raise ContractError(
                "shape_invalid", "inventory classification or authority flag invalid"
            )
    _sha256(value["manifest_digest"], "manifest digest")
    projection = {
        key: value[key]
        for key in (
            "schema_version",
            "profile",
            "semantic_identity_digest",
            "inventory",
        )
    }
    if value["manifest_digest"] != canonical_digest(projection):
        raise ContractError("digest_mismatch", "content manifest digest mismatch")


def validate_cover(cover: Any) -> None:
    value = _exact_object(
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
        "cover",
    )
    if value["schema_version"] != "radjax_tome_cover_v3":
        raise ContractError("schema_version_unsupported", "unsupported cover schema")
    validate_identity(value["identity"])
    if (
        not isinstance(value["training"], dict)
        or value["training"] != value["identity"]["training_contract"]
    ):
        raise ContractError(
            "profile_inventory_mismatch", "cover training does not match identity"
        )
    if (
        not isinstance(value["authority"], dict)
        or value["authority"] != value["identity"]["authority"]
    ):
        raise ContractError(
            "profile_inventory_mismatch", "cover authority does not match identity"
        )
    package = _exact_object(value["package"], {"profile", "transport"}, "package")
    if package["profile"] not in PROFILES or package["transport"] not in TRANSPORTS:
        raise ContractError("profile_inventory_mismatch", "cover package invalid")
    manifests = _exact_object(value["manifests"], {"content"}, "manifests")
    validate_manifest(manifests["content"])
    if (
        manifests["content"]["profile"] != package["profile"]
        or manifests["content"]["semantic_identity_digest"]
        != value["identity"]["semantic_digest"]
    ):
        raise ContractError(
            "profile_inventory_mismatch", "cover manifest references disagree"
        )
    if not isinstance(value["provenance"], dict) or not isinstance(
        value["validation"], dict
    ):
        raise ContractError(
            "shape_invalid", "provenance and validation must be objects"
        )


def validate_directory(root: Path) -> Result:
    try:
        cover = read_json(root / "cover_page.json")
        validate_cover(cover)
        expected = cover["manifests"]["content"]["inventory"]
        listed = {entry["path"] for entry in expected}
        observed = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.relative_to(root).as_posix() != "cover_page.json"
        }
        if observed != listed:
            raise ContractError(
                "profile_inventory_mismatch", "directory members do not match inventory"
            )
        for entry in expected:
            path = root / entry["path"]
            if (
                path.stat().st_size != entry["size_bytes"]
                or SHA256_PREFIX + hashlib.sha256(path.read_bytes()).hexdigest()
                != entry["sha256"]
            ):
                raise ContractError(
                    "digest_mismatch", f"raw inventory mismatch: {entry['path']}"
                )
        return Result(True)
    except ContractError as exc:
        return Result(False, (exc.code,))


def validate_archive(path: Path, *, strict_canonicality: bool) -> Result:
    try:
        with tarfile.open(path, mode="r:*") as archive:
            members = archive.getmembers()
            names: list[str] = []
            warnings: list[str] = []
            payloads: dict[str, bytes] = {}
            for member in members:
                _safe_relative_path(member.name, "archive member")
                if member.name in names:
                    raise ContractError("transport_unsafe", "duplicate archive member")
                names.append(member.name)
                if not member.isfile():
                    raise ContractError(
                        "transport_unsafe", "archive member is not regular file"
                    )
                handle = archive.extractfile(member)
                if handle is None:
                    raise ContractError(
                        "transport_corrupt", "archive member cannot be read"
                    )
                payloads[member.name] = handle.read()
                if (
                    member.mtime != 0
                    or member.uid != 0
                    or member.gid != 0
                    or member.mode != 0o644
                    or member.uname
                    or member.gname
                ):
                    warnings.append("transport_noncanonical")
            if "cover_page.json" not in payloads:
                raise ContractError(
                    "transport_corrupt", "archive has no cover_page.json"
                )
            cover = parse_json(
                payloads["cover_page.json"].decode("utf-8"), "cover_page.json"
            )
            validate_cover(cover)
            expected = cover["manifests"]["content"]["inventory"]
            expected_names = {"cover_page.json", *(entry["path"] for entry in expected)}
            if set(names) != expected_names:
                raise ContractError(
                    "profile_inventory_mismatch",
                    "archive members do not match inventory",
                )
            for entry in expected:
                raw = payloads[entry["path"]]
                if (
                    len(raw) != entry["size_bytes"]
                    or SHA256_PREFIX + hashlib.sha256(raw).hexdigest()
                    != entry["sha256"]
                ):
                    raise ContractError(
                        "digest_mismatch", f"raw inventory mismatch: {entry['path']}"
                    )
        if strict_canonicality and warnings:
            return Result(False, tuple(sorted(set(warnings))))
        return Result(True, warnings=tuple(sorted(set(warnings))))
    except (OSError, tarfile.TarError, UnicodeDecodeError):
        return Result(False, ("transport_corrupt",))
    except ContractError as exc:
        return Result(False, (exc.code,))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--strict-canonicality", action="store_true")
    args = parser.parse_args()
    result = (
        validate_directory(args.path)
        if args.path.is_dir()
        else validate_archive(args.path, strict_canonicality=args.strict_canonicality)
    )
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
