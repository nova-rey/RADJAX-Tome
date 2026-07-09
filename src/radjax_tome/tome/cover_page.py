from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from radjax_tome.io.json import read_json_object, write_json

COVER_PAGE_FILENAME = "cover_page.json"
COVER_PAGE_VERSION = 1
TOME_VERSION = 1
ARTIFACT_KIND = "radjax_tome"
UNPACKED_LAYOUT = "unpacked_directory"
REQUIRED_TOP_LEVEL_FIELDS = (
    "artifact_kind",
    "cover_page_version",
    "tome_version",
    "layout",
    "created_by",
    "created_at",
    "source_artifact_type",
    "teacher",
    "tokenizer",
    "targets",
    "contents",
    "validation",
    "claims_not_made",
)
REQUIRED_CONTENTS = (
    "metadata.json",
    "vocab_contract.json",
    "teacher_manifest.json",
    "emission_config.json",
    "validation_report.json",
)


@dataclass(frozen=True)
class CoverPageValidationReport:
    status: str
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    artifact_kind_ok: bool = False
    version_ok: bool = False
    layout_ok: bool = False
    contents_ok: bool = False
    hashes_ok: bool = False
    required_fields_ok: bool = False

    @property
    def ok(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_cover_page(tome_root: str | Path) -> dict[str, Any]:
    root = Path(tome_root)
    metadata = read_json_object(root / "metadata.json")
    vocab_contract = read_json_object(root / "vocab_contract.json")
    teacher_manifest = read_json_object(root / "teacher_manifest.json")
    validation_report = read_json_object(root / "validation_report.json")
    contents = _content_entries(root)
    cover_page = {
        "artifact_kind": ARTIFACT_KIND,
        "claims_not_made": _claims_not_made(teacher_manifest, validation_report),
        "contents": contents,
        "cover_page_version": COVER_PAGE_VERSION,
        "created_at": str(teacher_manifest.get("created_at") or metadata["created_at"]),
        "created_by": "radjax-tome.radjax_tome.tome.cover_page",
        "layout": UNPACKED_LAYOUT,
        "source_artifact_type": str(
            teacher_manifest.get("artifact_type", "teacher_textbook")
        ),
        "targets": {
            "dtype": metadata["dtype"],
            "num_examples": metadata["num_examples"],
            "sequence_length": metadata["sequence_length"],
            "shard_count": metadata["shard_count"],
            "target_params": dict(metadata.get("target_params", {})),
            "target_type": metadata["target_type"],
        },
        "teacher": {
            "allow_downloads": teacher_manifest.get("allow_downloads"),
            "backend_type": teacher_manifest.get("teacher_backend_type"),
            "local_files_only": teacher_manifest.get("local_files_only"),
            "model_family": metadata.get("model_family"),
            "model_id": metadata["model_id"],
        },
        "tokenizer": {
            "tokenizer_hash": vocab_contract.get("tokenizer_hash"),
            "tokenizer_id": vocab_contract.get("tokenizer_id"),
            "vocab_contract_path": teacher_manifest.get(
                "vocab_contract_path", "vocab_contract.json"
            ),
            "vocab_size": vocab_contract.get("vocab_size"),
        },
        "tome_version": TOME_VERSION,
        "validation": {
            "status": validation_report.get("status"),
            "validated_by": "radjax_tome.builder.validate_teacher_textbook",
            "validation_report_path": "validation_report.json",
        },
    }
    corpus_provenance = teacher_manifest.get("corpus_provenance")
    if isinstance(corpus_provenance, dict):
        cover_page["corpus"] = dict(corpus_provenance)
    teacher_model_provenance = teacher_manifest.get("teacher_model_provenance")
    if isinstance(teacher_model_provenance, dict):
        cover_page["teacher_model_provenance"] = {
            "allow_downloads": teacher_model_provenance.get("allow_downloads"),
            "config_hash": teacher_model_provenance.get("config_hash"),
            "local_files_only": teacher_model_provenance.get("local_files_only"),
            "model_directory_hash": teacher_model_provenance.get(
                "model_directory_hash"
            ),
            "model_identity_confidence": teacher_model_provenance.get(
                "model_identity_confidence"
            ),
            "model_name": teacher_model_provenance.get("model_name"),
            "model_name_source": teacher_model_provenance.get("model_name_source"),
            "model_provenance_mode": teacher_model_provenance.get(
                "model_provenance_mode"
            ),
            "model_revision": teacher_model_provenance.get("model_revision"),
            "model_revision_source": teacher_model_provenance.get(
                "model_revision_source"
            ),
            "model_source_kind": teacher_model_provenance.get("model_source_kind"),
            "network_used": teacher_model_provenance.get("network_used"),
            "schema_version": teacher_model_provenance.get("schema_version"),
            "teacher_model_provenance_path": teacher_model_provenance.get(
                "teacher_model_provenance_path"
            ),
            "tokenizer_hash": teacher_model_provenance.get("tokenizer_hash"),
            "weights_hash": teacher_model_provenance.get("weights_hash"),
        }
    target_params = metadata.get("target_params", {})
    if (
        isinstance(target_params, dict)
        and target_params.get("streaming_build") == "true"
    ):
        cover_page["streaming"] = {
            "streaming_build": True,
            "resume_supported": target_params.get("resume_supported") == "true",
            "run_manifest_path": target_params.get("run_manifest_path"),
            "progress_log_path": target_params.get("progress_log_path"),
            "shard_size_examples": target_params.get("shard_size_examples"),
            "resume_config_hash": target_params.get("resume_config_hash"),
            "atomic_shard_write_policy": target_params.get("atomic_shard_write_policy"),
        }
    return cover_page


def write_cover_page(tome_root: str | Path) -> Path:
    root = Path(tome_root)
    path = root / COVER_PAGE_FILENAME
    write_json(path, build_cover_page(root))
    report = validate_tome_cover_page(root)
    if not report.ok:
        raise ValueError(
            "generated cover_page.json failed validation: " + "; ".join(report.blockers)
        )
    return path


def validate_tome_cover_page(tome_root: str | Path) -> CoverPageValidationReport:
    root = Path(tome_root)
    blockers: list[str] = []
    warnings: list[str] = []
    path = root / COVER_PAGE_FILENAME
    if not path.is_file():
        blockers.append("missing cover_page.json")
        return _report(blockers=blockers)

    try:
        cover_page = read_json_object(path)
    except ValueError as exc:
        blockers.append(str(exc))
        return _report(blockers=blockers)

    missing_fields = [
        field for field in REQUIRED_TOP_LEVEL_FIELDS if field not in cover_page
    ]
    blockers.extend(
        f"cover_page.json missing required field: {field}" for field in missing_fields
    )
    required_fields_ok = not missing_fields

    artifact_kind_ok = cover_page.get("artifact_kind") == ARTIFACT_KIND
    if not artifact_kind_ok:
        blockers.append("cover_page.json artifact_kind must be radjax_tome")

    version_ok = (
        cover_page.get("cover_page_version") == COVER_PAGE_VERSION
        and cover_page.get("tome_version") == TOME_VERSION
    )
    if not version_ok:
        blockers.append("cover_page.json version fields must both be 1")

    layout_ok = cover_page.get("layout") == UNPACKED_LAYOUT
    if not layout_ok:
        blockers.append("cover_page.json layout must be unpacked_directory")

    contents = cover_page.get("contents")
    if not isinstance(contents, list):
        blockers.append("cover_page.json contents must be a list")
        contents = []

    listed_paths = {
        str(entry.get("path"))
        for entry in contents
        if isinstance(entry, dict) and "path" in entry
    }
    for required in REQUIRED_CONTENTS:
        if required not in listed_paths:
            blockers.append(f"cover_page.json missing content entry: {required}")
    shard_paths = sorted(
        path.relative_to(root).as_posix() for path in _shard_files(root)
    )
    for shard_path in shard_paths:
        if shard_path not in listed_paths:
            blockers.append(
                f"cover_page.json missing shard content entry: {shard_path}"
            )

    content_blockers, hash_blockers = _validate_contents(root, contents)
    blockers.extend(content_blockers)
    blockers.extend(hash_blockers)
    required_paths = set(REQUIRED_CONTENTS) | set(shard_paths)
    contents_ok = not content_blockers and all(
        required in listed_paths for required in required_paths
    )
    hashes_ok = not hash_blockers

    validation = cover_page.get("validation")
    if not isinstance(validation, dict):
        blockers.append("cover_page.json validation must be an object")
    else:
        report_path = validation.get("validation_report_path")
        if report_path != "validation_report.json":
            blockers.append("cover_page.json validation_report_path is unsupported")
        try:
            validation_report = read_json_object(root / "validation_report.json")
        except ValueError as exc:
            blockers.append(f"validation_report.json invalid: {exc}")
        else:
            if validation.get("status") != validation_report.get("status"):
                blockers.append(
                    "cover_page.json validation status does not match "
                    "validation_report.json"
                )
            if validation_report.get("status") != "pass":
                warnings.append("validation_report.json status is not pass")

    return _report(
        blockers=blockers,
        warnings=warnings,
        artifact_kind_ok=artifact_kind_ok,
        version_ok=version_ok,
        layout_ok=layout_ok,
        contents_ok=contents_ok,
        hashes_ok=hashes_ok,
        required_fields_ok=required_fields_ok,
    )


def _content_entries(root: Path) -> list[dict[str, Any]]:
    entries = [
        _content_entry(root, "metadata.json", "target_store_metadata"),
        _content_entry(root, "vocab_contract.json", "vocab_contract"),
        _content_entry(root, "teacher_manifest.json", "teacher_manifest"),
        _content_entry(root, "emission_config.json", "emission_config"),
        _content_entry(root, "validation_report.json", "validation_report"),
    ]
    for relative_path, role in (
        ("corridors/corridor_summary.json", "corridor_summary"),
        ("corridors/corridor_fingerprints.json", "corridor_fingerprints"),
        ("corridors/corridor_modes.json", "corridor_modes"),
        ("corridors/mode_assignments.json", "corridor_mode_assignments"),
        ("corridors/corridor_summary.txt", "corridor_human_summary"),
    ):
        if (root / relative_path).is_file():
            entries.append(_content_entry(root, relative_path, role))
    entries.extend(
        _content_entry(root, path.relative_to(root).as_posix(), "target_shard")
        for path in _shard_files(root)
    )
    return entries


def _content_entry(root: Path, relative_path: str, role: str) -> dict[str, Any]:
    path = root / relative_path
    return {
        "path": relative_path,
        "role": role,
        "sha256": _file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _validate_contents(
    root: Path,
    contents: list[Any],
) -> tuple[list[str], list[str]]:
    content_blockers: list[str] = []
    hash_blockers: list[str] = []
    for index, entry in enumerate(contents):
        if not isinstance(entry, dict):
            content_blockers.append(f"contents[{index}] must be an object")
            continue
        missing = [
            field
            for field in ("path", "role", "sha256", "size_bytes")
            if field not in entry
        ]
        content_blockers.extend(
            f"contents[{index}] missing required field: {field}" for field in missing
        )
        if missing:
            continue
        relative_path = str(entry["path"])
        target = _safe_content_path(root, relative_path)
        if target is None:
            content_blockers.append(
                f"contents[{index}] path escapes artifact root: {relative_path}"
            )
            continue
        if not target.is_file():
            content_blockers.append(
                f"contents[{index}] path does not exist: {relative_path}"
            )
            continue
        try:
            size_bytes = int(entry["size_bytes"])
        except (TypeError, ValueError):
            content_blockers.append(
                f"contents[{index}] size_bytes must be an integer for {relative_path}"
            )
            continue
        if size_bytes != target.stat().st_size:
            content_blockers.append(
                f"contents[{index}] size mismatch for {relative_path}"
            )
        expected_hash = str(entry["sha256"])
        actual_hash = _file_sha256(target)
        if expected_hash != actual_hash:
            hash_blockers.append(
                f"contents[{index}] sha256 mismatch for {relative_path}"
            )
    return content_blockers, hash_blockers


def _safe_content_path(root: Path, relative_path: str) -> Path | None:
    posix = PurePosixPath(relative_path)
    if posix.is_absolute() or ".." in posix.parts:
        return None
    target = root.joinpath(*posix.parts)
    try:
        target.resolve().relative_to(root.resolve())
    except ValueError:
        return None
    return target


def _shard_files(root: Path) -> tuple[Path, ...]:
    shards = root / "shards"
    if not shards.is_dir():
        return ()
    return tuple(sorted(path for path in shards.rglob("*") if path.is_file()))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _claims_not_made(
    teacher_manifest: dict[str, Any],
    validation_report: dict[str, Any],
) -> list[str]:
    claims: list[str] = []
    for payload in (teacher_manifest, validation_report):
        value = payload.get("claims_not_made", ())
        if isinstance(value, list):
            claims.extend(str(item) for item in value)
        elif isinstance(value, tuple):
            claims.extend(str(item) for item in value)
    return sorted(set(claims))


def _report(
    *,
    blockers: list[str],
    warnings: list[str] | None = None,
    artifact_kind_ok: bool = False,
    version_ok: bool = False,
    layout_ok: bool = False,
    contents_ok: bool = False,
    hashes_ok: bool = False,
    required_fields_ok: bool = False,
) -> CoverPageValidationReport:
    blocker_tuple = tuple(blockers)
    return CoverPageValidationReport(
        status="fail" if blocker_tuple else "pass",
        blockers=blocker_tuple,
        warnings=tuple(warnings or ()),
        artifact_kind_ok=artifact_kind_ok,
        version_ok=version_ok,
        layout_ok=layout_ok,
        contents_ok=contents_ok,
        hashes_ok=hashes_ok,
        required_fields_ok=required_fields_ok,
    )
