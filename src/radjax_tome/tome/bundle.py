from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from radjax_tome.io.json import read_json_object
from radjax_tome.tome.cover_page import (
    ARTIFACT_KIND,
    COVER_PAGE_FILENAME,
    COVER_PAGE_VERSION,
    REQUIRED_TOP_LEVEL_FIELDS,
    TOME_VERSION,
    validate_tome_cover_page,
)

SUPPORTED_COMPRESSION = {"none"}


@dataclass(frozen=True)
class TomeBundleValidationReport:
    status: str
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    format_ok: bool = False
    cover_page_ok: bool = False
    contents_ok: bool = False
    deterministic_layout_ok: bool = False

    @property
    def ok(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def pack_tome_bundle(
    tome_root: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
    compression: str = "none",
) -> Path:
    root = Path(tome_root)
    output = Path(output_path)
    _require_supported_compression(compression)
    if output.exists() and not overwrite:
        raise ValueError(f"bundle already exists: {output}")
    cover_report = validate_tome_cover_page(root)
    if not cover_report.ok:
        raise ValueError(
            "cannot pack invalid Tome cover page: " + "; ".join(cover_report.blockers)
        )

    cover_page = read_json_object(root / COVER_PAGE_FILENAME)
    entry_paths = _bundle_entry_paths(cover_page)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, mode=_tar_write_mode(compression)) as archive:
        for relative_path in entry_paths:
            source = _safe_source_path(root, relative_path)
            if source is None or not source.is_file():
                raise ValueError(f"bundle content path is invalid: {relative_path}")
            _add_file(archive, source, relative_path)
    return output


def inspect_tome_bundle(bundle_path: str | Path) -> dict[str, object]:
    path = Path(bundle_path)
    with tarfile.open(path, mode="r:*") as archive:
        cover_page = _read_cover_page_from_archive(archive)
    targets = cover_page.get("targets", {})
    return {
        "artifact_kind": cover_page.get("artifact_kind"),
        "bundle_path": str(path),
        "compression": _compression_label(path),
        "content_count": len(cover_page.get("contents", [])),
        "cover_page_version": cover_page.get("cover_page_version"),
        "layout": cover_page.get("layout"),
        "num_examples": targets.get("num_examples")
        if isinstance(targets, dict)
        else None,
        "shard_count": targets.get("shard_count")
        if isinstance(targets, dict)
        else None,
        "target_type": targets.get("target_type")
        if isinstance(targets, dict)
        else None,
        "tome_version": cover_page.get("tome_version"),
    }


def validate_tome_bundle(bundle_path: str | Path) -> TomeBundleValidationReport:
    path = Path(bundle_path)
    blockers: list[str] = []
    warnings: list[str] = []
    format_ok = False
    cover_page_ok = False
    contents_ok = False
    deterministic_layout_ok = False
    try:
        with tarfile.open(path, mode="r:*") as archive:
            members = archive.getmembers()
            format_ok = True
            member_blockers, duplicate_names = _validate_member_names(members)
            blockers.extend(member_blockers)
            member_names = [member.name for member in members]
            deterministic_layout_ok = not member_blockers and member_names == sorted(
                member_names
            )
            if COVER_PAGE_FILENAME not in member_names:
                blockers.append("bundle missing root cover_page.json")
                return _report(
                    blockers=blockers,
                    warnings=warnings,
                    format_ok=format_ok,
                    cover_page_ok=cover_page_ok,
                    contents_ok=contents_ok,
                    deterministic_layout_ok=deterministic_layout_ok,
                )
            cover_page = _read_cover_page_from_archive(archive)
            cover_page_blockers = _validate_embedded_cover_page(cover_page)
            blockers.extend(cover_page_blockers)
            cover_page_ok = not cover_page_blockers
            content_blockers, content_warnings = _validate_archive_contents(
                archive,
                members,
                cover_page,
                duplicate_names=duplicate_names,
            )
            blockers.extend(content_blockers)
            warnings.extend(content_warnings)
            contents_ok = not content_blockers
    except (tarfile.TarError, OSError, ValueError) as exc:
        blockers.append(f"bundle is not readable as a supported tar archive: {exc}")
    return _report(
        blockers=blockers,
        warnings=warnings,
        format_ok=format_ok,
        cover_page_ok=cover_page_ok,
        contents_ok=contents_ok,
        deterministic_layout_ok=deterministic_layout_ok,
    )


def unpack_tome_bundle(
    bundle_path: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    output = Path(output_dir)
    report = validate_tome_bundle(bundle_path)
    if not report.ok:
        raise ValueError(
            "cannot unpack invalid Tome bundle: " + "; ".join(report.blockers)
        )
    if output.exists():
        if any(output.iterdir()) and not overwrite:
            raise ValueError(f"output directory is not empty: {output}")
        if overwrite:
            shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(Path(bundle_path), mode="r:*") as archive:
        for member in archive.getmembers():
            target = _safe_output_path(output, member.name)
            if target is None:
                raise ValueError(f"unsafe bundle member path: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"could not extract bundle member: {member.name}")
                target.write_bytes(extracted.read())
            else:
                raise ValueError(f"unsupported bundle member type: {member.name}")
    cover_report = validate_tome_cover_page(output)
    if not cover_report.ok:
        raise ValueError(
            "unpacked Tome cover page failed validation: "
            + "; ".join(cover_report.blockers)
        )
    return output


def _add_file(archive: tarfile.TarFile, source: Path, relative_path: str) -> None:
    info = tarfile.TarInfo(relative_path)
    stat = source.stat()
    info.size = stat.st_size
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = 0o644
    with source.open("rb") as handle:
        archive.addfile(info, handle)


def _bundle_entry_paths(cover_page: dict[str, Any]) -> tuple[str, ...]:
    paths = {COVER_PAGE_FILENAME}
    contents = cover_page.get("contents", [])
    if not isinstance(contents, list):
        raise ValueError("cover_page.json contents must be a list")
    for entry in contents:
        if not isinstance(entry, dict):
            raise ValueError("cover_page.json content entry must be an object")
        relative_path = str(entry.get("path", ""))
        if _safe_relative_path(relative_path) is None:
            raise ValueError(f"unsafe cover page content path: {relative_path}")
        paths.add(relative_path)
    return tuple(sorted(paths))


def _read_cover_page_from_archive(archive: tarfile.TarFile) -> dict[str, Any]:
    try:
        member = archive.getmember(COVER_PAGE_FILENAME)
    except KeyError as exc:
        raise ValueError("bundle missing root cover_page.json") from exc
    extracted = archive.extractfile(member)
    if extracted is None:
        raise ValueError("bundle cover_page.json is not a file")
    payload = json.loads(extracted.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("bundle cover_page.json must be a JSON object")
    return payload


def _validate_member_names(
    members: list[tarfile.TarInfo],
) -> tuple[list[str], set[str]]:
    blockers: list[str] = []
    seen: set[str] = set()
    duplicates: set[str] = set()
    for member in members:
        if member.name in seen:
            duplicates.add(member.name)
            blockers.append(f"duplicate bundle member: {member.name}")
        seen.add(member.name)
        if _safe_relative_path(member.name) is None:
            blockers.append(f"unsafe bundle member path: {member.name}")
        if not (member.isfile() or member.isdir()):
            blockers.append(f"unsupported bundle member type: {member.name}")
        if member.isfile() and (
            member.mtime != 0
            or member.uid != 0
            or member.gid != 0
            or member.uname != ""
            or member.gname != ""
            or member.mode != 0o644
        ):
            blockers.append(f"non-deterministic tar metadata: {member.name}")
    return blockers, duplicates


def _validate_embedded_cover_page(cover_page: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    for field in REQUIRED_TOP_LEVEL_FIELDS:
        if field not in cover_page:
            blockers.append(f"cover_page.json missing required field: {field}")
    if cover_page.get("artifact_kind") != ARTIFACT_KIND:
        blockers.append("cover_page.json artifact_kind must be radjax_tome")
    if cover_page.get("cover_page_version") != COVER_PAGE_VERSION:
        blockers.append(
            f"cover_page.json cover_page_version must be {COVER_PAGE_VERSION}"
        )
    if cover_page.get("tome_version") != TOME_VERSION:
        blockers.append("cover_page.json tome_version must be 1")
    return blockers


def _validate_archive_contents(
    archive: tarfile.TarFile,
    members: list[tarfile.TarInfo],
    cover_page: dict[str, Any],
    *,
    duplicate_names: set[str],
) -> tuple[list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    member_by_name = {member.name: member for member in members}
    contents = cover_page.get("contents", [])
    if not isinstance(contents, list):
        return ["cover_page.json contents must be a list"], warnings
    listed_paths = {COVER_PAGE_FILENAME}
    for index, entry in enumerate(contents):
        if not isinstance(entry, dict):
            blockers.append(f"contents[{index}] must be an object")
            continue
        missing = [
            field
            for field in ("path", "role", "sha256", "size_bytes")
            if field not in entry
        ]
        blockers.extend(
            f"contents[{index}] missing required field: {field}" for field in missing
        )
        if missing:
            continue
        relative_path = str(entry["path"])
        listed_paths.add(relative_path)
        if _safe_relative_path(relative_path) is None:
            blockers.append(
                f"contents[{index}] path escapes artifact root: {relative_path}"
            )
            continue
        if relative_path in duplicate_names:
            blockers.append(f"contents[{index}] duplicate member: {relative_path}")
            continue
        member = member_by_name.get(relative_path)
        if member is None:
            blockers.append(f"contents[{index}] missing bundle member: {relative_path}")
            continue
        if not member.isfile():
            blockers.append(f"contents[{index}] member is not a file: {relative_path}")
            continue
        data = _read_member_bytes(archive, member)
        try:
            size_bytes = int(entry["size_bytes"])
        except (TypeError, ValueError):
            blockers.append(
                f"contents[{index}] size_bytes must be an integer for {relative_path}"
            )
            continue
        if size_bytes != len(data):
            blockers.append(f"contents[{index}] size mismatch for {relative_path}")
        if str(entry["sha256"]) != hashlib.sha256(data).hexdigest():
            blockers.append(f"contents[{index}] sha256 mismatch for {relative_path}")
    for member in members:
        if member.name not in listed_paths:
            warnings.append(
                f"extra bundle member not listed in cover page: {member.name}"
            )
    return blockers, warnings


def _read_member_bytes(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    extracted = archive.extractfile(member)
    if extracted is None:
        raise ValueError(f"bundle member is not readable: {member.name}")
    return extracted.read()


def _safe_source_path(root: Path, relative_path: str) -> Path | None:
    posix = _safe_relative_path(relative_path)
    if posix is None:
        return None
    target = root.joinpath(*posix.parts)
    try:
        target.resolve().relative_to(root.resolve())
    except ValueError:
        return None
    return target


def _safe_output_path(root: Path, relative_path: str) -> Path | None:
    posix = _safe_relative_path(relative_path)
    if posix is None:
        return None
    return root.joinpath(*posix.parts)


def _safe_relative_path(relative_path: str) -> PurePosixPath | None:
    posix = PurePosixPath(relative_path)
    if posix.is_absolute() or ".." in posix.parts or not posix.parts:
        return None
    return posix


def _tar_write_mode(compression: str) -> str:
    _require_supported_compression(compression)
    return "w"


def _compression_label(path: Path) -> str:
    suffixes = path.suffixes
    if suffixes[-2:] == [".tar", ".gz"] or suffixes[-1:] == [".tgz"]:
        return "gz"
    if suffixes[-2:] == [".tar", ".bz2"]:
        return "bz2"
    if suffixes[-2:] == [".tar", ".xz"]:
        return "xz"
    return "none"


def _require_supported_compression(compression: str) -> None:
    if compression not in SUPPORTED_COMPRESSION:
        raise ValueError(
            f"unsupported bundle compression: {compression!r}; supported: none"
        )


def _report(
    *,
    blockers: list[str],
    warnings: list[str] | None = None,
    format_ok: bool = False,
    cover_page_ok: bool = False,
    contents_ok: bool = False,
    deterministic_layout_ok: bool = False,
) -> TomeBundleValidationReport:
    blocker_tuple = tuple(blockers)
    return TomeBundleValidationReport(
        status="fail" if blocker_tuple else "pass",
        blockers=blocker_tuple,
        warnings=tuple(warnings or ()),
        format_ok=format_ok,
        cover_page_ok=cover_page_ok,
        contents_ok=contents_ok,
        deterministic_layout_ok=deterministic_layout_ok,
    )
