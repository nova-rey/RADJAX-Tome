from __future__ import annotations

import json
import os
import shutil
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from radjax_tome.audit import audit_selected_linkage, write_selected_linkage_audit
from radjax_tome.io.json import read_json_object, write_json
from radjax_tome.provenance.hashes import sha256_file

FULL_DEBUG_PROVENANCE = "full_debug_provenance"
STUDENT = "student"
PACKAGE_PROFILES = frozenset({FULL_DEBUG_PROVENANCE, STUDENT})
PACKAGE_COVER_SCHEMA = "radjax_tome_package_cover_v1"
CONTENT_MANIFEST_SCHEMA = "tome_content_manifest_v1"
SHARD_MANIFEST_SCHEMA = "tome_shard_manifest_v1"
CORRIDOR_ASSIGNMENT_MANIFEST_SCHEMA = "corridor_assignment_manifest_v1"
SELECTED_PAYLOAD_MANIFEST_SCHEMA = "selected_payload_manifest_v1"

_CORE_FILES = (
    "metadata.json",
    "vocab_contract.json",
    "teacher_manifest.json",
    "emission_config.json",
    "validation_report.json",
)
_STUDENT_CORRIDOR_FILES = (
    "corridors/corridor_summary.json",
    "corridors/corridor_modes.json",
    "corridors/mode_assignments.json",
)
_MANIFEST_FILES = {
    "content_manifest": "manifests/content_manifest.json",
    "corridor_assignment_manifest": "manifests/corridor_assignment_manifest.json",
    "selected_payload_manifest": "manifests/selected_payload_manifest.json",
    "shard_manifest": "manifests/shard_manifest.json",
}


@dataclass(frozen=True)
class TomePackageResult:
    output_path: Path
    profile: str
    archive: str
    package_root: Path | None = None


@dataclass(frozen=True)
class TomePackageValidationReport:
    status: str
    profile: str | None
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    content_manifest_ok: bool = False
    corridor_assignment_ok: bool = False
    selected_payloads_ok: bool = False
    shard_manifest_ok: bool | None = None
    selected_linkage_audit_ok: bool | None = None

    @property
    def ok(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StudentTomeReader:
    """Minimal raw-shard-free reader for the student training contract."""

    root: Path
    examples_input_ids: np.ndarray
    position_example_index: np.ndarray
    positions: np.ndarray
    mode_ids: np.ndarray
    weights: np.ndarray
    modes_by_id: dict[int, dict[str, Any]]
    example_index_by_id: dict[str, int]
    selected_payloads: tuple[dict[str, Any], ...]

    def corridor_batch(self, assignment_index: int) -> dict[str, Any]:
        example_index = int(self.position_example_index[assignment_index])
        mode_id = int(self.mode_ids[assignment_index])
        return {
            "input_ids": self.examples_input_ids[example_index],
            "position": int(self.positions[assignment_index]),
            "mode_id": mode_id,
            "weight": float(self.weights[assignment_index]),
            "mode_bounds": self.modes_by_id[mode_id]["bounds"],
        }

    def exemplar_batch(self, selected_index: int) -> dict[str, Any]:
        payload = dict(self.selected_payloads[selected_index])
        example_id = str(payload.get("selected_example_id"))
        example_index = self.example_index_by_id.get(example_id)
        if example_index is not None:
            payload["input_ids"] = self.examples_input_ids[example_index]
        return payload


def package_tome_artifact(
    artifact_dir: Path,
    output: Path,
    *,
    profile: str,
    archive: str = "none",
    overwrite: bool = False,
) -> TomePackageResult:
    _require_profile(profile)
    if archive not in {"none", "tgz"}:
        raise ValueError("archive must be one of: none, tgz")
    source = artifact_dir.resolve()
    if not source.is_dir():
        raise ValueError(f"artifact directory does not exist: {source}")
    if output.exists() and not overwrite:
        raise ValueError(f"package output already exists: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".radjax-package-",
        dir=output.parent,
    ) as tmp:
        temporary_root = Path(tmp) / _package_root_name(output, archive)
        _materialize_package(source, temporary_root, profile=profile)
        _write_package_audit(temporary_root, profile=profile)
        _write_package_manifests(temporary_root, profile=profile)
        _write_package_cover_page(temporary_root, profile=profile)
        report = validate_tome_package(temporary_root, profile=profile)
        if not report.ok:
            raise ValueError(
                "packaged Tome failed validation: " + "; ".join(report.blockers)
            )
        if archive == "none":
            _replace_output_path(temporary_root, output, overwrite=overwrite)
            return TomePackageResult(
                output_path=output,
                profile=profile,
                archive=archive,
                package_root=output,
            )

        temporary_archive = Path(tmp) / output.name
        _write_tgz(temporary_root, temporary_archive)
        _replace_output_path(temporary_archive, output, overwrite=overwrite)
    return TomePackageResult(output_path=output, profile=profile, archive=archive)


def validate_tome_package(
    artifact_dir: Path,
    *,
    profile: str | None = None,
) -> TomePackageValidationReport:
    root = artifact_dir.resolve()
    blockers: list[str] = []
    warnings: list[str] = []
    cover = _read_object(root / "cover_page.json", blockers, "cover_page.json")
    actual_profile = cover.get("package_profile") if cover else None
    if actual_profile not in PACKAGE_PROFILES:
        blockers.append("cover_page.json package_profile is invalid")
        actual_profile = None
    if profile is not None:
        try:
            _require_profile(profile)
        except ValueError as exc:
            blockers.append(str(exc))
        if actual_profile is not None and actual_profile != profile:
            blockers.append("package profile does not match requested profile")
    if not cover or cover.get("schema_version") != PACKAGE_COVER_SCHEMA:
        blockers.append("cover_page.json schema_version mismatch")

    content_ok = _validate_content_manifest(root, cover, actual_profile, blockers)
    assignment_ok = _validate_corridor_assignment_manifest(root, cover, blockers)
    selected_ok = _validate_selected_payload_manifest(root, cover, blockers)
    shard_ok: bool | None = None
    audit_ok: bool | None = None
    if actual_profile == FULL_DEBUG_PROVENANCE:
        shard_ok = _validate_shard_manifest(root, cover, blockers)
        if not (root / "shards").is_dir():
            blockers.append("full_debug_provenance package missing shards/")
        else:
            from radjax_tome.builder import validate_teacher_textbook

            producer_report = validate_teacher_textbook(root)
            if producer_report.status == "fail":
                blockers.append(
                    "full_debug_provenance producer validation failed: "
                    + "; ".join(producer_report.blockers)
                )
    elif actual_profile == STUDENT:
        shard_ok = None
        if (root / "shards").exists():
            blockers.append("student package must not retain raw producer shards/")
        _validate_student_contract(root, blockers)

    if _has_selected_payloads(root):
        if actual_profile is not None:
            audit = audit_selected_linkage(
                root,
                strict=True,
                profile=actual_profile,
            )
            audit_ok = audit.status == "pass"
            if not audit_ok:
                blockers.append(
                    "selected linkage audit failed: " + _audit_summary(audit)
                )
    return TomePackageValidationReport(
        status="pass" if not blockers else "fail",
        profile=actual_profile,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        content_manifest_ok=content_ok,
        corridor_assignment_ok=assignment_ok,
        selected_payloads_ok=selected_ok,
        shard_manifest_ok=shard_ok,
        selected_linkage_audit_ok=audit_ok,
    )


def open_student_tome(artifact_dir: Path) -> StudentTomeReader:
    root = artifact_dir.resolve()
    report = validate_tome_package(root, profile=STUDENT)
    if not report.ok:
        raise ValueError(
            "cannot open invalid student package: " + "; ".join(report.blockers)
        )
    assignment = read_json_object(root / "corridors" / "mode_assignments.json")
    arrays = assignment["arrays"]
    example_ids = _read_examples_metadata(
        root / str(assignment["examples_metadata"]["path"])
    )
    modes = read_json_object(root / "corridors" / "corridor_modes.json")["modes"]
    payloads = _read_selected_payloads(root)
    return StudentTomeReader(
        root=root,
        examples_input_ids=np.load(
            root / "corridors" / "mode_assignments" / "examples_input_ids.npy",
            allow_pickle=False,
            mmap_mode="r",
        ),
        position_example_index=np.load(
            root / str(arrays["position_example_index"]["path"]),
            allow_pickle=False,
            mmap_mode="r",
        ),
        positions=np.load(
            root / str(arrays["position"]["path"]),
            allow_pickle=False,
            mmap_mode="r",
        ),
        mode_ids=np.load(
            root / str(arrays["mode_id"]["path"]),
            allow_pickle=False,
            mmap_mode="r",
        ),
        weights=np.load(
            root / str(arrays["weight"]["path"]),
            allow_pickle=False,
            mmap_mode="r",
        ),
        modes_by_id={int(item["mode_id"]): item for item in modes},
        example_index_by_id={
            example_id: index for index, example_id in enumerate(example_ids)
        },
        selected_payloads=tuple(payloads),
    )


def _materialize_package(source: Path, destination: Path, *, profile: str) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    if profile == FULL_DEBUG_PROVENANCE:
        for child in sorted(source.iterdir(), key=lambda item: item.name):
            if child.name in {"cover_page.json", "manifests"}:
                continue
            _copy_path(child, destination / child.name)
        return
    for relative_path in (*_CORE_FILES, *_STUDENT_CORRIDOR_FILES):
        _copy_required(source, destination, relative_path)
    _copy_required(
        source,
        destination,
        "corridors/mode_assignments",
        directory=True,
    )
    _copy_optional(source, destination, "selected_exemplars", directory=True)
    _copy_optional(source, destination, "leaderboards/selected_exemplars.json")
    for optional in ("selected_linkage_audit.json", "student_validation_report.json"):
        _copy_optional(source, destination, optional)
    _export_student_inputs(source, destination)
    _sanitize_student_portability(destination)


def _copy_required(
    source: Path,
    destination: Path,
    relative_path: str,
    *,
    directory: bool = False,
) -> None:
    path = source / relative_path
    if (
        not path.exists()
        or (directory and not path.is_dir())
        or (not directory and not path.is_file())
    ):
        raise ValueError(f"artifact missing required package file: {relative_path}")
    _copy_path(path, destination / relative_path)


def _copy_optional(
    source: Path,
    destination: Path,
    relative_path: str,
    *,
    directory: bool = False,
) -> None:
    path = source / relative_path
    if path.exists() and (path.is_dir() if directory else path.is_file()):
        _copy_path(path, destination / relative_path)


def _copy_path(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination, symlinks=False)
    else:
        shutil.copy2(source, destination)


def _export_student_inputs(source: Path, destination: Path) -> None:
    assignments_dir = destination / "corridors" / "mode_assignments"
    assignment = read_json_object(destination / "corridors" / "mode_assignments.json")
    metadata = _read_examples_metadata(
        destination / str(assignment["examples_metadata"]["path"])
    )
    input_rows, mask_rows = _source_input_rows(source, expected_example_ids=metadata)
    if len(input_rows) != len(metadata):
        raise ValueError("could not resolve every corridor example to source input_ids")
    input_ids = np.stack(input_rows, axis=0)
    fits_int32 = input_ids.max(initial=0) <= np.iinfo(np.int32).max
    if np.issubdtype(input_ids.dtype, np.integer) and fits_int32:
        input_ids = input_ids.astype(np.int32, copy=False)
    else:
        input_ids = input_ids.astype(np.int64, copy=False)
    np.save(assignments_dir / "examples_input_ids.npy", input_ids)
    if mask_rows:
        np.save(
            assignments_dir / "examples_attention_mask.npy",
            np.stack(mask_rows, axis=0).astype(np.int8, copy=False),
        )


def _sanitize_student_portability(root: Path) -> None:
    """Keep source provenance while making machine-local paths explicit."""
    for path in sorted(root.rglob("*.json")):
        if path.relative_to(root).as_posix() == "corridors/mode_assignments.json":
            continue
        try:
            payload = read_json_object(path)
        except (OSError, ValueError):
            continue
        write_json(path, _portable_json_value(payload))


def _portable_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _portable_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_json_value(item) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        return {
            "non_portable_source_path": value,
            "portable_basename": Path(value).name,
        }
    return value


def _source_input_rows(
    source: Path,
    *,
    expected_example_ids: list[str],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    keyed: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
    ordered: list[tuple[np.ndarray, np.ndarray | None]] = []
    for shard_path in sorted((source / "shards").glob("shard-*.npz")):
        with np.load(shard_path, allow_pickle=False) as shard:
            input_ids = np.asarray(shard["input_ids"])
            attention = (
                np.asarray(shard["attention_mask"])
                if "attention_mask" in shard.files
                else None
            )
            score_ids = (
                np.asarray(shard["score_example_ids"])
                if "score_example_ids" in shard.files
                else None
            )
            for row in range(input_ids.shape[0]):
                attention_row = (
                    None if attention is None else np.asarray(attention[row])
                )
                item = (np.asarray(input_ids[row]), attention_row)
                ordered.append(item)
                if score_ids is not None:
                    keyed[_decode_example_id(score_ids[row])] = item
    if all(example_id in keyed for example_id in expected_example_ids):
        selected = [keyed[example_id] for example_id in expected_example_ids]
    elif len(ordered) == len(expected_example_ids):
        selected = ordered
    else:
        raise ValueError("source shards cannot resolve corridor example identities")
    inputs = [item[0] for item in selected]
    masks = [item[1] for item in selected if item[1] is not None]
    if masks and len(masks) != len(inputs):
        raise ValueError("source shards inconsistently retain attention_mask")
    return inputs, [np.asarray(mask) for mask in masks]


def _write_package_audit(root: Path, *, profile: str) -> None:
    if not _has_selected_payloads(root):
        return
    audit = audit_selected_linkage(root, strict=True, profile=profile)
    write_selected_linkage_audit(audit, root / "selected_linkage_audit.json")
    if audit.status != "pass":
        raise ValueError("selected linkage audit failed: " + _audit_summary(audit))


def _write_package_manifests(root: Path, *, profile: str) -> None:
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    write_json(
        manifests / "corridor_assignment_manifest.json",
        _corridor_assignment_manifest(root),
    )
    if _has_selected_payloads(root):
        write_json(
            manifests / "selected_payload_manifest.json",
            _selected_payload_manifest(root),
        )
    if profile == FULL_DEBUG_PROVENANCE:
        write_json(manifests / "shard_manifest.json", _shard_manifest(root))
    write_json(manifests / "content_manifest.json", _content_manifest(root, profile))


def _write_package_cover_page(root: Path, *, profile: str) -> None:
    metadata = read_json_object(root / "metadata.json")
    corridor_summary = read_json_object(root / "corridors" / "corridor_summary.json")
    delivery = _optional_object(root / "delivery_report.json")
    production = _optional_object(root / "production_build_report.json")
    assignments = read_json_object(
        root / _MANIFEST_FILES["corridor_assignment_manifest"]
    )
    selected = _optional_object(root / "manifests" / "selected_payload_manifest.json")
    manifest_refs = {
        "content_manifest": _manifest_reference(
            root,
            _MANIFEST_FILES["content_manifest"],
        ),
        "corridor_assignment_manifest": _manifest_reference(
            root,
            _MANIFEST_FILES["corridor_assignment_manifest"],
        ),
    }
    if selected is not None:
        manifest_refs["selected_payload_manifest"] = _manifest_reference(
            root,
            _MANIFEST_FILES["selected_payload_manifest"],
        )
    if profile == FULL_DEBUG_PROVENANCE:
        manifest_refs["shard_manifest"] = _manifest_reference(
            root,
            _MANIFEST_FILES["shard_manifest"],
        )
    top_level_summary = _package_top_level_summary(
        profile=profile,
        metadata=metadata,
        corridor_summary=corridor_summary,
        corridor_assignment_manifest=assignments,
        selected_payload_manifest=selected,
        delivery_report=delivery,
        production_build_report=production,
        selected_linkage_audit=_optional_object(root / "selected_linkage_audit.json"),
        validation_report=_optional_object(root / "validation_report.json"),
    )
    claims_made, claims_not_made = _profile_claims(profile)
    cover = {
        "schema_version": PACKAGE_COVER_SCHEMA,
        "artifact_kind": "radjax_tome",
        "layout": "unpacked_directory",
        "package_profile": profile,
        "created_at": str(metadata.get("created_at") or _utc_now()),
        "created_by": "radjax_tome.tome.packaging",
        **manifest_refs,
        "top_level_summary": top_level_summary,
        "claims_made": claims_made,
        "claims_not_made": claims_not_made,
    }
    write_json(root / "cover_page.json", cover)


def _package_top_level_summary(
    *,
    profile: str,
    metadata: dict[str, Any],
    corridor_summary: dict[str, Any],
    corridor_assignment_manifest: dict[str, Any],
    selected_payload_manifest: dict[str, Any] | None,
    delivery_report: dict[str, Any] | None,
    production_build_report: dict[str, Any] | None,
    selected_linkage_audit: dict[str, Any] | None,
    validation_report: dict[str, Any] | None,
) -> dict[str, Any]:
    delivery = delivery_report or {}
    production = production_build_report or {}
    selected = selected_payload_manifest or {}
    audit = selected_linkage_audit or {}
    validation = validation_report or {}
    return {
        "num_examples_scored": _first_int(
            delivery.get("num_examples_scored"),
            production.get("num_examples_scored"),
            corridor_summary.get("num_examples_scored"),
            metadata.get("num_examples"),
        ),
        "num_positions_scored": _first_int(
            delivery.get("num_positions_scored"),
            production.get("num_positions_scored"),
            corridor_summary.get("num_positions_scored"),
        ),
        "num_selected_exemplars": _first_int(
            selected.get("selected_count"),
            delivery.get("num_selected_exemplars"),
            production.get("num_selected_exemplars"),
            corridor_summary.get("selected_exemplar_count"),
        ),
        "corridor_mode_count": _first_int(
            corridor_summary.get("mode_count"),
            delivery.get("corridor_mode_count"),
            production.get("corridor_mode_count"),
        ),
        "corridor_assignment_count": _first_int(
            corridor_assignment_manifest.get("assignment_count"),
            corridor_summary.get("corridor_assignment_count"),
            delivery.get("corridor_assignment_count"),
            production.get("corridor_assignment_count"),
        ),
        "delivery_path": _first_string(
            delivery.get("delivery_path"),
            production.get("delivery_path"),
            corridor_summary.get("delivery_path"),
            default="not_recorded",
        ),
        "selected_linkage_audit_status": _first_string(
            audit.get("status"),
            default="not_applicable",
        ),
        "validation_status": _first_string(
            validation.get("status"),
            default="not_recorded",
        ),
        "package_profile": profile,
        "producer_shard_authority": _first_string(
            audit.get("producer_shard_authority"),
            default=(
                "available"
                if profile == FULL_DEBUG_PROVENANCE
                else "not_available_in_student_profile"
            ),
        ),
    }


def _first_int(*values: Any) -> int:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def _first_string(*values: Any, default: str) -> str:
    for value in values:
        if value is not None and str(value):
            return str(value)
    return default


def _content_manifest(root: Path, profile: str) -> dict[str, Any]:
    entries = []
    for path in _package_content_files(root):
        relative = path.relative_to(root).as_posix()
        entries.append(
            {
                "path": relative,
                "role": _content_role(relative),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
                "required": True,
            }
        )
    return {
        "schema_version": CONTENT_MANIFEST_SCHEMA,
        "package_profile": profile,
        "artifact_root": ".",
        "created_at": _utc_now(),
        "entries": entries,
        "entry_count": len(entries),
        "total_size_bytes": sum(int(item["size_bytes"]) for item in entries),
    }


def _corridor_assignment_manifest(root: Path) -> dict[str, Any]:
    assignment = read_json_object(root / "corridors" / "mode_assignments.json")
    arrays = assignment.get("arrays")
    if not isinstance(arrays, dict):
        raise ValueError("mode_assignments.json arrays must be an object")
    paths: dict[str, str] = {}
    files: list[dict[str, Any]] = []
    for name in ("position_example_index", "position", "mode_id", "weight"):
        spec = arrays.get(name)
        if not isinstance(spec, dict):
            raise ValueError(f"mode_assignments missing {name}")
        path = str(spec.get("path") or "")
        paths[name] = path
        files.append(_numpy_file_entry(root, path))
    examples_metadata = assignment.get("examples_metadata")
    if not isinstance(examples_metadata, dict):
        raise ValueError("mode_assignments missing examples_metadata")
    metadata_path = str(examples_metadata.get("path") or "")
    paths["examples_metadata"] = metadata_path
    files.append(_plain_file_entry(root, metadata_path))
    inputs_path = "corridors/mode_assignments/examples_input_ids.npy"
    if (root / inputs_path).is_file():
        paths["examples_input_ids"] = inputs_path
        files.append(_numpy_file_entry(root, inputs_path))
    attention_path = "corridors/mode_assignments/examples_attention_mask.npy"
    if (root / attention_path).is_file():
        paths["examples_attention_mask"] = attention_path
        files.append(_numpy_file_entry(root, attention_path))
    metadata = read_json_object(root / "metadata.json")
    return {
        "schema_version": CORRIDOR_ASSIGNMENT_MANIFEST_SCHEMA,
        "storage_kind": assignment.get("storage_kind"),
        "paths": paths,
        "files": sorted(files, key=lambda item: str(item["path"])),
        "assignment_count": int(assignment.get("num_assignments") or 0),
        "example_count": int(assignment.get("num_examples") or 0),
        "sequence_length": int(metadata.get("sequence_length") or 0),
    }


def _selected_payload_manifest(root: Path) -> dict[str, Any]:
    shards = []
    selected_count = 0
    for path in sorted((root / "selected_exemplars").glob("selected-exemplars-*.json")):
        payload = read_json_object(path)
        records = payload.get("selected_exemplars")
        if not isinstance(records, list):
            raise ValueError(f"selected payload shard is invalid: {path}")
        selected_count += len(records)
        effective = [int(item.get("effective_top_k") or 0) for item in records]
        mass = [float(item.get("top_mass") or 0.0) for item in records]
        shards.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
                "record_count": len(records),
                "effective_top_k_min": min(effective, default=0),
                "effective_top_k_max": max(effective, default=0),
                "top_mass_min": min(mass, default=0.0),
                "top_mass_max": max(mass, default=0.0),
            }
        )
    return {
        "schema_version": SELECTED_PAYLOAD_MANIFEST_SCHEMA,
        "selected_count": selected_count,
        "shard_count": len(shards),
        "payload_shards": shards,
    }


def _shard_manifest(root: Path) -> dict[str, Any]:
    shards: list[dict[str, Any]] = []
    offset = 0
    for shard_id, path in enumerate(sorted((root / "shards").glob("shard-*.npz"))):
        with np.load(path, allow_pickle=False) as shard:
            fields = [
                {
                    "name": name,
                    "shape": list(shard[name].shape),
                    "dtype": str(shard[name].dtype),
                }
                for name in sorted(shard.files)
            ]
            count = int(np.asarray(shard["input_ids"]).shape[0])
        shards.append(
            {
                "shard_id": shard_id,
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
                "num_examples": count,
                "example_start_index": offset,
                "example_end_index_exclusive": offset + count,
                "fields": fields,
            }
        )
        offset += count
    return {
        "schema_version": SHARD_MANIFEST_SCHEMA,
        "package_profile": FULL_DEBUG_PROVENANCE,
        "shard_count": len(shards),
        "shards": shards,
    }


def _validate_content_manifest(
    root: Path,
    cover: dict[str, Any],
    profile: str | None,
    blockers: list[str],
) -> bool:
    manifest = _read_manifest_from_cover(root, cover, "content_manifest", blockers)
    if manifest is None:
        return False
    if manifest.get("schema_version") != CONTENT_MANIFEST_SCHEMA:
        blockers.append("content manifest schema_version mismatch")
    if manifest.get("package_profile") != profile:
        blockers.append("content manifest package_profile mismatch")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        blockers.append("content manifest entries must be a list")
        return False
    paths: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            blockers.append("content manifest contains invalid entry")
            continue
        relative = str(entry.get("path") or "")
        paths.append(relative)
        _validate_hashed_file(root, relative, entry, blockers, label="content")
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        blockers.append("content manifest entries must be uniquely path-sorted")
    actual = [
        path.relative_to(root).as_posix() for path in _package_content_files(root)
    ]
    if set(paths) != set(actual):
        blockers.append("content manifest does not cover every package content file")
    if int(manifest.get("entry_count") or -1) != len(entries):
        blockers.append("content manifest entry_count mismatch")
    return not any(
        "content manifest" in item or item.startswith("content ") for item in blockers
    )


def _validate_corridor_assignment_manifest(
    root: Path,
    cover: dict[str, Any],
    blockers: list[str],
) -> bool:
    manifest = _read_manifest_from_cover(
        root,
        cover,
        "corridor_assignment_manifest",
        blockers,
    )
    if manifest is None:
        return False
    if manifest.get("schema_version") != CORRIDOR_ASSIGNMENT_MANIFEST_SCHEMA:
        blockers.append("corridor assignment manifest schema_version mismatch")
    paths = manifest.get("paths")
    if not isinstance(paths, dict):
        blockers.append("corridor assignment manifest paths must be an object")
        return False
    files = manifest.get("files")
    if not isinstance(files, list):
        blockers.append("corridor assignment manifest files must be a list")
        return False
    for entry in files:
        if isinstance(entry, dict):
            _validate_hashed_file(
                root,
                str(entry.get("path") or ""),
                entry,
                blockers,
                label="corridor assignment",
            )
    required = (
        "position_example_index",
        "position",
        "mode_id",
        "weight",
        "examples_metadata",
    )
    if any(not isinstance(paths.get(name), str) for name in required):
        blockers.append("corridor assignment manifest missing required paths")
        return False
    try:
        example_index = np.load(
            root / str(paths["position_example_index"]),
            allow_pickle=False,
        )
        position = np.load(root / str(paths["position"]), allow_pickle=False)
        mode_id = np.load(root / str(paths["mode_id"]), allow_pickle=False)
        weight = np.load(root / str(paths["weight"]), allow_pickle=False)
        examples = _read_examples_metadata(root / str(paths["examples_metadata"]))
        modes = read_json_object(root / "corridors" / "corridor_modes.json")["modes"]
    except (IndexError, KeyError, OSError, TypeError, ValueError) as exc:
        blockers.append(f"corridor assignment inputs are unreadable: {exc}")
        return False
    count = int(manifest.get("assignment_count") or -1)
    if any(
        array.shape != (count,) for array in (example_index, position, mode_id, weight)
    ):
        blockers.append("corridor assignment array shape mismatch")
    if len(examples) != int(manifest.get("example_count") or -1):
        blockers.append("corridor assignment example_count mismatch")
    if np.any(example_index < 0) or np.any(example_index >= len(examples)):
        blockers.append("corridor assignment example index out of range")
    sequence_length = int(manifest.get("sequence_length") or 0)
    if np.any(position < 0) or np.any(position >= sequence_length):
        blockers.append("corridor assignment position out of range")
    valid_modes = {int(item["mode_id"]) for item in modes if isinstance(item, dict)}
    if not np.isin(mode_id, np.asarray(sorted(valid_modes), dtype=mode_id.dtype)).all():
        blockers.append("corridor assignment mode_id references missing mode")
    if np.any(~np.isfinite(weight)) or np.any(weight < 0.0):
        blockers.append("corridor assignment weights must be finite and nonnegative")
    return not any("corridor assignment" in item for item in blockers)


def _validate_selected_payload_manifest(
    root: Path,
    cover: dict[str, Any],
    blockers: list[str],
) -> bool:
    if "selected_payload_manifest" not in cover:
        return not _has_selected_payloads(root)
    manifest = _read_manifest_from_cover(
        root,
        cover,
        "selected_payload_manifest",
        blockers,
    )
    if manifest is None:
        return False
    if manifest.get("schema_version") != SELECTED_PAYLOAD_MANIFEST_SCHEMA:
        blockers.append("selected payload manifest schema_version mismatch")
    shards = manifest.get("payload_shards")
    if not isinstance(shards, list):
        blockers.append("selected payload manifest payload_shards must be a list")
        return False
    count = 0
    for entry in shards:
        if not isinstance(entry, dict):
            blockers.append("selected payload manifest contains invalid shard")
            continue
        relative = str(entry.get("path") or "")
        _validate_hashed_file(root, relative, entry, blockers, label="selected payload")
        try:
            records = read_json_object(root / relative).get("selected_exemplars")
        except (OSError, ValueError):
            records = None
        if not isinstance(records, list):
            blockers.append(f"selected payload shard invalid: {relative}")
            continue
        count += len(records)
        if int(entry.get("record_count") or -1) != len(records):
            blockers.append(f"selected payload record_count mismatch: {relative}")
    if int(manifest.get("selected_count") or -1) != count:
        blockers.append("selected payload manifest selected_count mismatch")
    return not any("selected payload" in item for item in blockers)


def _validate_shard_manifest(
    root: Path,
    cover: dict[str, Any],
    blockers: list[str],
) -> bool:
    manifest = _read_manifest_from_cover(root, cover, "shard_manifest", blockers)
    if manifest is None:
        return False
    if manifest.get("schema_version") != SHARD_MANIFEST_SCHEMA:
        blockers.append("shard manifest schema_version mismatch")
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        blockers.append("shard manifest shards must be a list")
        return False
    actual_paths = [
        path.relative_to(root).as_posix()
        for path in sorted((root / "shards").glob("shard-*.npz"))
    ]
    manifest_paths = [
        str(item.get("path") or "") for item in shards if isinstance(item, dict)
    ]
    if manifest_paths != actual_paths:
        blockers.append("shard manifest does not cover every producer shard")
    if int(manifest.get("shard_count") or -1) != len(shards):
        blockers.append("shard manifest shard_count mismatch")
    for entry in shards:
        if not isinstance(entry, dict):
            blockers.append("shard manifest contains invalid shard")
            continue
        relative = str(entry.get("path") or "")
        _validate_hashed_file(root, relative, entry, blockers, label="shard")
        try:
            with np.load(root / relative, allow_pickle=False) as shard:
                actual = [
                    {
                        "name": name,
                        "shape": list(shard[name].shape),
                        "dtype": str(shard[name].dtype),
                    }
                    for name in sorted(shard.files)
                ]
        except (OSError, ValueError) as exc:
            blockers.append(f"shard unreadable: {relative}: {exc}")
            continue
        if entry.get("fields") != actual:
            blockers.append(f"shard fields mismatch: {relative}")
    return not any("shard " in item or item.startswith("shard") for item in blockers)


def _validate_student_contract(root: Path, blockers: list[str]) -> None:
    path = root / "corridors" / "mode_assignments" / "examples_input_ids.npy"
    if not path.is_file():
        blockers.append("student package missing examples_input_ids.npy")
        return
    try:
        input_ids = np.load(path, allow_pickle=False)
        assignment = read_json_object(root / "corridors" / "mode_assignments.json")
        examples = _read_examples_metadata(
            root / str(assignment["examples_metadata"]["path"])
        )
    except (IndexError, KeyError, OSError, TypeError, ValueError) as exc:
        blockers.append(f"student training contract inputs unreadable: {exc}")
        return
    if input_ids.ndim != 2 or input_ids.shape[0] != len(examples):
        blockers.append("student examples_input_ids shape does not match metadata")
    if not np.issubdtype(input_ids.dtype, np.integer):
        blockers.append("student examples_input_ids must have integer dtype")


def _read_manifest_from_cover(
    root: Path,
    cover: dict[str, Any],
    key: str,
    blockers: list[str],
) -> dict[str, Any] | None:
    reference = cover.get(key)
    if not isinstance(reference, dict):
        blockers.append(f"cover_page.json missing {key}")
        return None
    relative = str(reference.get("path") or "")
    path = root / relative
    _validate_hashed_file(root, relative, reference, blockers, label=f"cover {key}")
    try:
        return read_json_object(path)
    except (OSError, ValueError) as exc:
        blockers.append(f"{key} is unreadable: {exc}")
        return None


def _validate_hashed_file(
    root: Path,
    relative: str,
    entry: dict[str, Any],
    blockers: list[str],
    *,
    label: str,
) -> None:
    path = _safe_package_path(root, relative)
    if path is None or not path.is_file():
        blockers.append(f"{label} file missing: {relative}")
        return
    if entry.get("sha256") != _sha256(path):
        blockers.append(f"{label} hash mismatch: {relative}")
    if int(entry.get("size_bytes") or -1) != path.stat().st_size:
        blockers.append(f"{label} size mismatch: {relative}")


def _manifest_reference(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    return {
        "path": relative,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _numpy_file_entry(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    array = np.load(path, allow_pickle=False, mmap_mode="r")
    return {
        **_plain_file_entry(root, relative),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def _plain_file_entry(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    if not path.is_file():
        raise ValueError(f"package file missing: {relative}")
    return {
        "path": relative,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _package_content_files(root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.relative_to(root).as_posix() != "cover_page.json"
            and path.relative_to(root).as_posix() != "manifests/content_manifest.json"
        ),
        key=lambda item: item.relative_to(root).as_posix(),
    )


def _content_role(relative: str) -> str:
    if relative.startswith("shards/"):
        return "producer_shard"
    if relative.startswith("selected_exemplars/"):
        return "selected_exemplar_payload"
    if relative.startswith("corridors/mode_assignments/"):
        return "corridor_assignment_input"
    if relative.startswith("corridors/"):
        return "corridor_artifact"
    if relative.startswith("leaderboards/"):
        return "selected_exemplar_index"
    return Path(relative).stem


def _profile_claims(profile: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if profile == STUDENT:
        return (
            {
                "self_consistent_training_contract": True,
                "content_hashes_verified": True,
                "selected_payloads_linked": True,
                "corridor_assignments_loadable": True,
                "student_batches_constructible": True,
                "derived_from_full_tome": True,
            },
            {
                "does_not_include_full_producer_shards": True,
                "does_not_include_raw_teacher_logits": True,
                "does_not_include_non_selected_exemplar_payloads": True,
                "does_not_prove_recomputability_without_full_debug_tome": True,
                "producer_shard_authority": "not_available_in_student_profile",
            },
        )
    return (
        {
            "contains_full_retained_producer_shards": True,
            "contains_score_pass_fields": True,
            "contains_corridor_debug_surfaces": True,
            "can_audit_selected_records_against_source_shards": True,
            "can_audit_student_package_derivation": True,
            "content_hashes_verified": True,
        },
        {
            "does_not_include_raw_teacher_logits": True,
            "does_not_include_non_selected_exemplar_payloads": True,
        },
    )


def _read_examples_metadata(path: Path) -> list[str]:
    rows: list[tuple[int, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        rows.append((int(item["example_index"]), str(item["example_id"])))
    rows.sort()
    if [index for index, _ in rows] != list(range(len(rows))):
        raise ValueError("examples_metadata example_index must be contiguous")
    return [example_id for _, example_id in rows]


def _read_selected_payloads(root: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for path in sorted((root / "selected_exemplars").glob("selected-exemplars-*.json")):
        records = read_json_object(path).get("selected_exemplars", [])
        if isinstance(records, list):
            payloads.extend(item for item in records if isinstance(item, dict))
    return payloads


def _has_selected_payloads(root: Path) -> bool:
    return bool(list((root / "selected_exemplars").glob("selected-exemplars-*.json")))


def _optional_object(path: Path) -> dict[str, Any] | None:
    try:
        return read_json_object(path)
    except (OSError, ValueError):
        return None


def _read_object(path: Path, blockers: list[str], label: str) -> dict[str, Any]:
    try:
        return read_json_object(path)
    except (OSError, ValueError) as exc:
        blockers.append(f"{label} is missing or invalid: {exc}")
        return {}


def _audit_status(root: Path) -> str | None:
    audit = _optional_object(root / "selected_linkage_audit.json")
    return None if audit is None else str(audit.get("status") or "") or None


def _validation_status(root: Path) -> str | None:
    report = _optional_object(root / "validation_report.json")
    return None if report is None else str(report.get("status") or "") or None


def _audit_summary(audit: Any) -> str:
    errors = getattr(audit, "errors", ())
    return str(errors[0].get("mismatch_fields") if errors else "unknown mismatch")


def _decode_example_id(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _sha256(path: Path) -> str:
    return "sha256:" + sha256_file(path)


def _safe_package_path(root: Path, relative: str) -> Path | None:
    pure = PurePosixPath(relative)
    if not relative or pure.is_absolute() or ".." in pure.parts:
        return None
    return root.joinpath(*pure.parts)


def _package_root_name(output: Path, archive: str) -> str:
    return output.stem if archive == "tgz" else output.name


def _replace_output_path(source: Path, output: Path, *, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise ValueError(f"package output already exists: {output}")
        if output.is_dir():
            shutil.rmtree(output)
        else:
            output.unlink()
    os.replace(source, output)


def _write_tgz(root: Path, output: Path) -> None:
    with tarfile.open(output, "w:gz") as archive:
        for path in sorted(
            root.rglob("*"),
            key=lambda item: item.relative_to(root).as_posix(),
        ):
            if not path.is_file():
                continue
            arcname = (Path(root.name) / path.relative_to(root)).as_posix()
            info = archive.gettarinfo(str(path), arcname=arcname)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            with path.open("rb") as handle:
                archive.addfile(info, handle)


def _require_profile(profile: str) -> None:
    if profile not in PACKAGE_PROFILES:
        raise ValueError(
            "profile must be one of: " + ", ".join(sorted(PACKAGE_PROFILES))
        )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
