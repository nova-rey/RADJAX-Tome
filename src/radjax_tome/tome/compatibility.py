"""Explicit, non-inferential readers for historical Tome cover contracts."""

from __future__ import annotations

import json
import tarfile
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from radjax_tome.io.json import read_json_object
from radjax_tome.tome.contracts import HISTORICAL_PACKAGE_COVER_SCHEMA

LEGACY_COVER_PAGE_V2 = "cover_page_v2"
_LEGACY_LAYOUT_TO_TRANSPORT = {"unpacked_directory": "directory"}
_LEGACY_PACKAGE_PROFILES = frozenset({"student", "full_debug_provenance"})


@dataclass(frozen=True)
class HistoricalTomeDescriptor:
    """Known historical facts expressed in the v3 section vocabulary.

    This is deliberately not a v3 cover.  Missing identity, authority, or
    package facts remain ``None`` rather than receiving defaults or inferred
    values from sibling artifact files.
    """

    source_schema: str
    identity: None
    training: dict[str, Any] | None
    package: dict[str, str] | None
    manifests: dict[str, Any] | None
    authority: None
    provenance: dict[str, Any] | None
    validation: dict[str, Any] | None
    unavailable_sections: tuple[str, ...]
    migration_diagnostic: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def adapt_historical_tome_cover(cover: Mapping[str, Any]) -> HistoricalTomeDescriptor:
    """Map one supported historical cover without creating unknown v3 facts."""

    payload = _object(cover, "historical cover")
    if payload.get("cover_page_version") == 2:
        return _adapt_cover_page_v2(payload)
    if payload.get("schema_version") == HISTORICAL_PACKAGE_COVER_SCHEMA:
        return _adapt_package_cover_v1(payload)
    raise ValueError(
        "unsupported historical Tome cover; expected cover_page v2 or "
        "radjax_tome_package_cover_v1"
    )


def read_historical_tome_descriptor(source: str | Path) -> HistoricalTomeDescriptor:
    """Natively validate then adapt a historical directory or tar archive."""

    path = Path(source)
    if path.is_dir():
        cover = read_json_object(path / "cover_page.json")
        _validate_native_historical_directory(path, cover)
        return adapt_historical_tome_cover(cover)
    if not path.is_file():
        raise ValueError(f"historical Tome source does not exist: {path}")
    if path.suffix == ".json":
        raise ValueError(
            "a standalone historical cover cannot establish native artifact "
            "validity; pass its parsed object to adapt_historical_tome_cover"
        )
    return adapt_historical_tome_cover(_read_and_validate_archive(path))


def _adapt_cover_page_v2(payload: dict[str, Any]) -> HistoricalTomeDescriptor:
    if payload.get("artifact_kind") != "radjax_tome":
        raise ValueError("cover_page v2 artifact_kind must be radjax_tome")
    if payload.get("layout") != "unpacked_directory":
        raise ValueError("cover_page v2 layout is unsupported for migration")
    targets = _optional_object(payload, "targets")
    tokenizer = _optional_object(payload, "tokenizer")
    training: dict[str, Any] = {}
    if targets is not None:
        for key in ("target_type", "sequence_length"):
            if key in targets:
                training[key] = targets[key]
    if tokenizer is not None and "vocab_size" in tokenizer:
        training["vocab_size"] = tokenizer["vocab_size"]
    if "tome_version" in payload:
        training["tome_version"] = payload["tome_version"]
    contents = payload.get("contents")
    if contents is not None and not isinstance(contents, list):
        raise ValueError("cover_page v2 contents must be a list when present")
    if contents is not None and any(
        not isinstance(entry, Mapping) for entry in contents
    ):
        raise ValueError("cover_page v2 contents entries must be objects")
    manifests = (
        {
            "legacy_content_inventory": {
                "entries": [dict(entry) for entry in contents],
                "source_contract": LEGACY_COVER_PAGE_V2,
                "profile_complete": False,
            }
        }
        if contents is not None
        else None
    )
    provenance = _known_fields(
        payload,
        (
            "created_at",
            "created_by",
            "source_artifact_type",
            "teacher",
            "tokenizer",
            "corpus",
            "teacher_model_provenance",
        ),
    )
    validation = _known_fields(
        payload,
        (
            "validation",
            "claims_not_made",
            "behavioral_surfaces",
            "recommended_training_plan",
        ),
    )
    return HistoricalTomeDescriptor(
        source_schema=LEGACY_COVER_PAGE_V2,
        identity=None,
        training=training or None,
        package=None,
        manifests=manifests,
        authority=None,
        provenance=provenance or None,
        validation=validation or None,
        unavailable_sections=(
            "identity",
            "authority",
            "package.profile",
            "package.transport",
        ),
        migration_diagnostic=(
            "cover_page v2 has no canonical semantic identity, authority binding, "
            "or profile-complete inventory; rebuild or repackage with a current "
            "canonical writer to emit radjax_tome_cover_v3"
        ),
    )


def _adapt_package_cover_v1(payload: dict[str, Any]) -> HistoricalTomeDescriptor:
    profile = payload.get("package_profile")
    layout = payload.get("layout")
    if profile not in _LEGACY_PACKAGE_PROFILES:
        raise ValueError(
            "package-cover v1 package_profile is unsupported for migration"
        )
    if not isinstance(layout, str) or layout not in _LEGACY_LAYOUT_TO_TRANSPORT:
        raise ValueError("package-cover v1 layout is unsupported for migration")
    references = _known_fields(
        payload,
        (
            "content_manifest",
            "corridor_assignment_manifest",
            "selected_payload_manifest",
            "shard_manifest",
        ),
    )
    return HistoricalTomeDescriptor(
        source_schema=HISTORICAL_PACKAGE_COVER_SCHEMA,
        identity=None,
        training=None,
        package={
            "profile": profile,
            "transport": _LEGACY_LAYOUT_TO_TRANSPORT[layout],
        },
        manifests={"legacy_manifest_references": references} if references else None,
        authority=None,
        provenance=_known_fields(
            payload,
            (
                "created_at",
                "created_by",
                "diagnostics",
                "claims_made",
                "claims_not_made",
            ),
        )
        or None,
        validation=_known_fields(payload, ("top_level_summary",)) or None,
        unavailable_sections=("identity", "training", "authority"),
        migration_diagnostic=(
            "package-cover v1 establishes its declared profile and manifest "
            "references only; it has no canonical identity/training/authority "
            "proof. Repackage with a current canonical writer to emit "
            "radjax_tome_cover_v3"
        ),
    )


def _read_and_validate_archive(path: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="radjax-historical-tome-") as tmp:
        root, cover = _extract_historical_archive(path, Path(tmp))
        _validate_native_historical_directory(root, cover)
        return cover


def _extract_historical_archive(
    path: Path, destination: Path
) -> tuple[Path, dict[str, Any]]:
    try:
        with tarfile.open(path, mode="r:*") as archive:
            candidates = [
                member
                for member in archive.getmembers()
                if member.isfile()
                and _safe_member_path(member.name)
                and PurePosixPath(member.name).name == "cover_page.json"
            ]
            if len(candidates) != 1:
                raise ValueError(
                    "historical archive must contain exactly one safe cover_page.json"
                )
            prefix = PurePosixPath(candidates[0].name).parent.parts
            root = destination / "artifact"
            root.mkdir()
            for member in archive.getmembers():
                if not member.isfile() or not _safe_member_path(member.name):
                    raise ValueError("historical archive contains an unsafe member")
                parts = PurePosixPath(member.name).parts
                if parts[: len(prefix)] != prefix:
                    raise ValueError(
                        "historical archive members do not share the cover directory"
                    )
                relative = parts[len(prefix) :]
                if not relative:
                    raise ValueError("historical archive member has an empty path")
                target = root.joinpath(*relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                handle = archive.extractfile(member)
                if handle is None:
                    raise ValueError(
                        f"historical archive member is unreadable: {member.name}"
                    )
                target.write_bytes(handle.read())
            payload = read_json_object(root / "cover_page.json")
    except (OSError, tarfile.TarError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"historical Tome archive is unreadable: {exc}") from exc
    return root, _object(payload, "historical archive cover_page.json")


def _validate_native_historical_directory(root: Path, cover: Mapping[str, Any]) -> None:
    if cover.get("cover_page_version") == 2:
        from radjax_tome.tome.cover_page import validate_tome_cover_page

        report = validate_tome_cover_page(root)
    elif cover.get("schema_version") == HISTORICAL_PACKAGE_COVER_SCHEMA:
        from radjax_tome.tome.packaging import validate_tome_package

        report = validate_tome_package(root)
    else:
        raise ValueError(
            "unsupported historical Tome cover; expected cover_page v2 or "
            "radjax_tome_package_cover_v1"
        )
    if not report.ok:
        raise ValueError(
            "historical Tome failed native validation: " + "; ".join(report.blockers)
        )


def _safe_member_path(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts


def _known_fields(payload: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: payload[key] for key in keys if key in payload}


def _optional_object(payload: Mapping[str, Any], key: str) -> dict[str, Any] | None:
    value = payload.get(key)
    if value is None:
        return None
    return _object(value, key)


def _object(value: Mapping[str, Any] | Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return dict(value)
