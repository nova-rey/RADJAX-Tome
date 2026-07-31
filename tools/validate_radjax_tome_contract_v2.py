#!/usr/bin/env python3
"""Stdlib-only streaming validator for the proposed RADJAX-Tome v2 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

PREFIX = "sha256:"
PROFILES = {"unpacked", "student", "full_debug_provenance"}
CHUNK = 1 << 16


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
        if not isinstance(package, dict) or package.get("profile") not in PROFILES:
            raise ContractError("profile_inventory_mismatch")
        if not isinstance(manifests, dict) or set(manifests) != {"header"}:
            raise ContractError("shape_invalid")
        ref = manifests["header"]
        if not isinstance(ref, dict):
            raise ContractError("shape_invalid")
        header_path = _path(ref.get("path"))
        if _digest_path(root / header_path) != (
            _sha(ref.get("sha256")),
            ref.get("size_bytes"),
        ):
            raise ContractError("digest_mismatch")
        header = _json(root / header_path)
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
        if _digest_path(root / inventory_path) != (
            _sha(header["inventory_sha256"]),
            header["inventory_size_bytes"],
        ):
            raise ContractError("digest_mismatch")
        previous = ""
        count = 0
        listed: set[str] = set()
        for entry in _lines(root / inventory_path):
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
            if _digest_path(root / path) != (
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
