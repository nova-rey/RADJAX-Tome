from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from radjax_tome.io.json import read_json_object, write_json

COVER_PAGE_FILENAME = "cover_page.json"
COVER_PAGE_VERSION = 2
TOME_VERSION = 1
ARTIFACT_KIND = "radjax_tome"
UNPACKED_LAYOUT = "unpacked_directory"
SURFACE_SCHEMA_VERSION = "behavioral_surface_v1"
PASS_PLAN_SCHEMA_VERSION = "recommended_training_plan_v1"
CONTENT_CLASSIFICATIONS = frozenset(
    {
        "training_critical",
        "integrity_or_provenance",
        "diagnostic",
        "human_readable",
        "operational",
    }
)
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
    "behavioral_surfaces",
    "recommended_training_plan",
    "validation",
    "claims_not_made",
)
_CORE_CONTENT_SPECS = (
    ("metadata.json", "target_store_metadata", True, "integrity_or_provenance"),
    ("vocab_contract.json", "vocab_contract", True, "training_critical"),
    ("teacher_manifest.json", "teacher_manifest", True, "integrity_or_provenance"),
    ("emission_config.json", "emission_config", True, "integrity_or_provenance"),
    ("validation_report.json", "validation_report", True, "integrity_or_provenance"),
)
_CORRIDOR_FILE_SPECS = (
    (
        "corridors/corridor_summary.json",
        "corridor_summary",
        True,
        "integrity_or_provenance",
    ),
    (
        "corridors/corridor_fingerprints.json",
        "corridor_fingerprints",
        False,
        "diagnostic",
    ),
    (
        "corridors/corridor_modes.json",
        "corridor_mode_table",
        True,
        "training_critical",
    ),
    (
        "corridors/mode_assignments.json",
        "corridor_assignment_manifest",
        True,
        "training_critical",
    ),
    (
        "corridors/corridor_summary.txt",
        "corridor_human_summary",
        False,
        "human_readable",
    ),
)
_INTEGRATION_FILE_SPECS = (
    (
        "reports/fingerprint_corridor_coverage.json",
        "fingerprint_corridor_coverage",
        False,
        "diagnostic",
    ),
    (
        "reports/c6_integrated_selection_validation.json",
        "c6_integrated_selection_validation",
        False,
        "integrity_or_provenance",
    ),
)
_ASSIGNMENT_ARRAY_ROLES = {
    "position_example_index": "corridor_assignment_position_example_index",
    "position": "corridor_assignment_position",
    "mode_id": "corridor_assignment_mode_id",
    "weight": "corridor_assignment_weight",
    "fingerprint_index": "corridor_assignment_fingerprint_index",
}
_CORRIDOR_REQUIRED_ROLES = (
    "target_shard",
    "corridor_summary",
    "corridor_mode_table",
    "corridor_assignment_manifest",
    "corridor_assignment_position_example_index",
    "corridor_assignment_position",
    "corridor_assignment_mode_id",
    "corridor_assignment_weight",
    "corridor_assignment_examples_metadata",
)
_CORRIDOR_OPTIONAL_ROLES = (
    "corridor_assignment_fingerprint_index",
    "corridor_fingerprints",
    "corridor_human_summary",
)
_EXEMPLAR_REQUIRED_ROLES = (
    "target_shard",
    "selected_exemplar_index",
    "selected_exemplar_payload_shard",
)
_EXEMPLAR_OPTIONAL_ROLES = (
    "exemplar_delivery_report",
    "exemplar_leaderboard_report",
)
_SINGLETON_ROLES = frozenset(
    {
        role
        for _, role, _, _ in (
            *_CORE_CONTENT_SPECS,
            *_CORRIDOR_FILE_SPECS,
            *_INTEGRATION_FILE_SPECS,
        )
    }
    | {
        "corridor_assignment_position_example_index",
        "corridor_assignment_position",
        "corridor_assignment_mode_id",
        "corridor_assignment_weight",
        "corridor_assignment_fingerprint_index",
        "corridor_assignment_examples_metadata",
        "selected_exemplar_index",
        "exemplar_delivery_report",
        "exemplar_leaderboard_report",
    }
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
    surfaces = _behavioral_surfaces(root, contents)
    cover_page = {
        "artifact_kind": ARTIFACT_KIND,
        "claims_not_made": _claims_not_made(teacher_manifest, validation_report),
        "contents": contents,
        "behavioral_surfaces": surfaces,
        "cover_page_version": COVER_PAGE_VERSION,
        "created_at": str(teacher_manifest.get("created_at") or metadata["created_at"]),
        "created_by": "radjax-tome.radjax_tome.tome.cover_page",
        "layout": UNPACKED_LAYOUT,
        "recommended_training_plan": _recommended_training_plan(surfaces),
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
    coverage_path = root / "reports" / "fingerprint_corridor_coverage.json"
    if coverage_path.is_file():
        coverage = read_json_object(coverage_path)
        validation_path = root / "reports" / "c6_integrated_selection_validation.json"
        integrated_validation = (
            read_json_object(validation_path) if validation_path.is_file() else {}
        )
        cover_page["selection_integration"] = {
            "selection_integration_policy": coverage.get(
                "selection_integration_policy"
            ),
            "selected_unique_count": coverage.get("selected_unique_count"),
            "selected_obligation_count": coverage.get("selected_obligation_count"),
            "selected_multi_role_count": coverage.get("multi_role_coordinate_count"),
            "corridor_budget_actual": coverage.get("corridor_budget_actual"),
            "corridor_modes_covered": coverage.get("corridor_modes_fulfilled"),
            "global_backfill_count": coverage.get("global_backfill_count"),
            "multi_role_schema_version": coverage.get("c5_schema_version"),
            "coverage_report_path": "reports/fingerprint_corridor_coverage.json",
            "coverage_report_sha256": _file_sha256(coverage_path),
            "integrated_validation_status": integrated_validation.get("status"),
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
        blockers.append(
            "cover_page.json versions unsupported: "
            f"cover_page_version={cover_page.get('cover_page_version')} "
            f"tome_version={cover_page.get('tome_version')}"
        )

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
    blockers.extend(_validate_surfaces_and_plan(cover_page, contents))

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
    entries = [_content_entry(root, *spec) for spec in _CORE_CONTENT_SPECS]
    for relative_path, role, required, classification in _CORRIDOR_FILE_SPECS:
        if (root / relative_path).is_file():
            entries.append(
                _content_entry(root, relative_path, role, required, classification)
            )
    for relative_path, role, required, classification in _INTEGRATION_FILE_SPECS:
        if (root / relative_path).is_file():
            entries.append(
                _content_entry(root, relative_path, role, required, classification)
            )
    entries.extend(_assignment_content_entries(root))
    entries.extend(
        _content_entry(
            root,
            path.relative_to(root).as_posix(),
            "target_shard",
            True,
            "training_critical",
        )
        for path in _shard_files(root)
    )
    exemplar_specs = (
        (
            "leaderboards/selected_exemplars.json",
            "selected_exemplar_index",
            True,
            "training_critical",
        ),
        (
            "leaderboards/leaderboard_report.json",
            "exemplar_leaderboard_report",
            False,
            "diagnostic",
        ),
        (
            "delivery_report.json",
            "exemplar_delivery_report",
            False,
            "diagnostic",
        ),
    )
    for spec in exemplar_specs:
        if (root / spec[0]).is_file():
            entries.append(_content_entry(root, *spec))
    selected_dir = root / "selected_exemplars"
    if selected_dir.is_dir():
        entries.extend(
            _content_entry(
                root,
                path.relative_to(root).as_posix(),
                "selected_exemplar_payload_shard",
                True,
                "training_critical",
            )
            for path in sorted(selected_dir.glob("*.json"))
            if path.is_file()
        )
    return sorted(entries, key=lambda item: str(item["path"]))


def _assignment_content_entries(root: Path) -> list[dict[str, Any]]:
    manifest_path = root / "corridors" / "mode_assignments.json"
    if not manifest_path.is_file():
        return []
    try:
        manifest = read_json_object(manifest_path)
    except ValueError:
        return []
    entries: list[dict[str, Any]] = []
    arrays = manifest.get("arrays", {})
    if isinstance(arrays, dict):
        for name, role in _ASSIGNMENT_ARRAY_ROLES.items():
            descriptor = arrays.get(name)
            if not isinstance(descriptor, dict) or descriptor.get("path") is None:
                continue
            relative_path = str(descriptor["path"])
            if not (root / relative_path).is_file():
                continue
            diagnostic = name == "fingerprint_index"
            entries.append(
                _content_entry(
                    root,
                    relative_path,
                    role,
                    not diagnostic,
                    "diagnostic" if diagnostic else "training_critical",
                )
            )
    metadata = manifest.get("examples_metadata")
    if isinstance(metadata, dict) and metadata.get("path") is not None:
        relative_path = str(metadata["path"])
        if (root / relative_path).is_file():
            entries.append(
                _content_entry(
                    root,
                    relative_path,
                    "corridor_assignment_examples_metadata",
                    True,
                    "training_critical",
                )
            )
    return entries


def _behavioral_surfaces(
    root: Path,
    contents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    roles = {str(entry["role"]) for entry in contents}
    surfaces: list[dict[str, Any]] = []
    if roles & {
        "corridor_summary",
        "corridor_mode_table",
        "corridor_assignment_manifest",
    }:
        summary = _read_optional_json(root / "corridors" / "corridor_summary.json")
        surfaces.append(
            {
                "surface_id": "corridor",
                "surface_kind": "fingerprint_corridor",
                "schema_version": SURFACE_SCHEMA_VERSION,
                "required_content_roles": list(_CORRIDOR_REQUIRED_ROLES),
                "optional_content_roles": list(_CORRIDOR_OPTIONAL_ROLES),
                "required_capabilities": [
                    "radjax.corridor.packed_assignments.v1",
                    "radjax.corridor.stat_bands.v1",
                ],
                "prerequisites": [],
                "target_scope": {"kind": "whole_model"},
                "semantics": {
                    "mode_policy": summary.get("corridor_mode_policy"),
                    "position_semantics": "zero_based_token_position_v1",
                    "storage_kind": summary.get("corridor_assignment_storage_kind"),
                },
            }
        )
    if roles & {"selected_exemplar_index", "selected_exemplar_payload_shard"}:
        surfaces.append(
            {
                "surface_id": "exemplar",
                "surface_kind": "selected_exemplar",
                "schema_version": SURFACE_SCHEMA_VERSION,
                "required_content_roles": list(_EXEMPLAR_REQUIRED_ROLES),
                "optional_content_roles": list(_EXEMPLAR_OPTIONAL_ROLES),
                "required_capabilities": ["radjax.exemplar.selected_dynamic_topk.v1"],
                "prerequisites": ["corridor"]
                if "corridor" in {str(item["surface_id"]) for item in surfaces}
                else [],
                "target_scope": {"kind": "whole_model"},
                "semantics": {
                    "delivery_path_is_provenance": True,
                    "position_semantics": "zero_based_token_position_v1",
                },
            }
        )
    return surfaces


def _recommended_training_plan(
    surfaces: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": PASS_PLAN_SCHEMA_VERSION,
        "passes": [
            {
                "pass_id": f"{surface['surface_id']}_pass",
                "surface_id": surface["surface_id"],
                "required_capabilities": list(surface["required_capabilities"]),
                "prerequisites": list(surface["prerequisites"]),
                "checkpoint_after": True,
            }
            for surface in surfaces
        ],
    }


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return read_json_object(path)
    except ValueError:
        return {}


def _validate_surfaces_and_plan(
    cover_page: dict[str, Any],
    contents: list[Any],
) -> tuple[str, ...]:
    blockers: list[str] = []
    indexed_roles = {
        str(entry.get("role")) for entry in contents if isinstance(entry, dict)
    }
    surfaces = cover_page.get("behavioral_surfaces")
    if not isinstance(surfaces, list):
        return ("cover_page.json behavioral_surfaces must be a list",)
    surface_ids: set[str] = set()
    prerequisites_by_id: dict[str, tuple[str, ...]] = {}
    for index, surface in enumerate(surfaces):
        if not isinstance(surface, dict):
            blockers.append(f"behavioral_surfaces[{index}] must be an object")
            continue
        missing = [
            field
            for field in (
                "surface_id",
                "surface_kind",
                "schema_version",
                "required_content_roles",
                "optional_content_roles",
                "required_capabilities",
                "prerequisites",
                "target_scope",
                "semantics",
            )
            if field not in surface
        ]
        blockers.extend(
            f"behavioral_surfaces[{index}] missing field: {field}" for field in missing
        )
        if missing:
            continue
        surface_id = str(surface["surface_id"])
        if not surface_id or surface_id in surface_ids:
            blockers.append(f"behavioral surface ID invalid or duplicate: {surface_id}")
            continue
        surface_ids.add(surface_id)
        required_roles = surface["required_content_roles"]
        if not isinstance(required_roles, list):
            blockers.append(f"surface {surface_id} required roles must be a list")
        else:
            for role in required_roles:
                if str(role) not in indexed_roles:
                    blockers.append(
                        f"surface {surface_id} missing required content role: {role}"
                    )
        capabilities = surface["required_capabilities"]
        if not isinstance(capabilities, list) or not all(
            isinstance(item, str) and item for item in capabilities
        ):
            blockers.append(
                f"surface {surface_id} required capabilities must be strings"
            )
        prerequisites = surface["prerequisites"]
        if not isinstance(prerequisites, list):
            blockers.append(f"surface {surface_id} prerequisites must be a list")
            prerequisites_by_id[surface_id] = ()
        else:
            prerequisites_by_id[surface_id] = tuple(str(item) for item in prerequisites)
    for surface_id, prerequisites in prerequisites_by_id.items():
        for prerequisite in prerequisites:
            if prerequisite not in surface_ids:
                blockers.append(
                    f"surface {surface_id} references unknown prerequisite: "
                    f"{prerequisite}"
                )
    blockers.extend(_surface_cycle_blockers(prerequisites_by_id))

    plan = cover_page.get("recommended_training_plan")
    if not isinstance(plan, dict):
        return tuple(blockers + ["recommended_training_plan must be an object"])
    if plan.get("schema_version") != PASS_PLAN_SCHEMA_VERSION:
        blockers.append("recommended_training_plan schema_version mismatch")
    passes = plan.get("passes")
    if not isinstance(passes, list):
        return tuple(blockers + ["recommended_training_plan passes must be a list"])
    completed: set[str] = set()
    seen_pass_ids: set[str] = set()
    for index, item in enumerate(passes):
        if not isinstance(item, dict):
            blockers.append(f"recommended pass[{index}] must be an object")
            continue
        pass_id = str(item.get("pass_id", ""))
        surface_id = str(item.get("surface_id", ""))
        if not pass_id or pass_id in seen_pass_ids:
            blockers.append(f"recommended pass ID invalid or duplicate: {pass_id}")
        seen_pass_ids.add(pass_id)
        if surface_id not in surface_ids:
            blockers.append(
                f"recommended pass references unknown surface: {surface_id}"
            )
        prerequisites = item.get("prerequisites", [])
        if not isinstance(prerequisites, list) or any(
            str(value) not in completed for value in prerequisites
        ):
            blockers.append(f"recommended pass prerequisites not satisfied: {pass_id}")
        if not isinstance(item.get("checkpoint_after"), bool):
            blockers.append(f"recommended pass checkpoint flag invalid: {pass_id}")
        completed.add(surface_id)
    return tuple(blockers)


def _surface_cycle_blockers(
    prerequisites_by_id: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(surface_id: str) -> bool:
        if surface_id in visiting:
            return True
        if surface_id in visited:
            return False
        visiting.add(surface_id)
        for prerequisite in prerequisites_by_id.get(surface_id, ()):
            if prerequisite in prerequisites_by_id and visit(prerequisite):
                return True
        visiting.remove(surface_id)
        visited.add(surface_id)
        return False

    for surface_id in prerequisites_by_id:
        if visit(surface_id):
            return ("behavioral surface prerequisites contain a cycle",)
    return ()


def _content_entry(
    root: Path,
    relative_path: str,
    role: str,
    required: bool,
    classification: str,
) -> dict[str, Any]:
    path = root / relative_path
    return {
        "path": relative_path,
        "role": role,
        "required": required,
        "classification": classification,
        "sha256": _file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _validate_contents(
    root: Path,
    contents: list[Any],
) -> tuple[list[str], list[str]]:
    content_blockers: list[str] = []
    hash_blockers: list[str] = []
    seen_paths: set[str] = set()
    role_counts: dict[str, int] = {}
    for index, entry in enumerate(contents):
        if not isinstance(entry, dict):
            content_blockers.append(f"contents[{index}] must be an object")
            continue
        missing = [
            field
            for field in (
                "path",
                "role",
                "sha256",
                "size_bytes",
                "required",
                "classification",
            )
            if field not in entry
        ]
        content_blockers.extend(
            f"contents[{index}] missing required field: {field}" for field in missing
        )
        if missing:
            continue
        relative_path = str(entry["path"])
        role = str(entry["role"])
        classification = str(entry["classification"])
        if relative_path in seen_paths:
            content_blockers.append(f"contents duplicate path: {relative_path}")
            continue
        seen_paths.add(relative_path)
        role_counts[role] = role_counts.get(role, 0) + 1
        posix = PurePosixPath(relative_path)
        if not relative_path or posix.as_posix() != relative_path:
            content_blockers.append(
                f"contents[{index}] path is not normalized POSIX: {relative_path}"
            )
        if not role:
            content_blockers.append(f"contents[{index}] role must be nonempty")
        if classification not in CONTENT_CLASSIFICATIONS:
            content_blockers.append(
                f"contents[{index}] classification unsupported: {classification}"
            )
        if not isinstance(entry["required"], bool):
            content_blockers.append(f"contents[{index}] required must be boolean")
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
    for role in sorted(_SINGLETON_ROLES):
        if role_counts.get(role, 0) > 1:
            content_blockers.append(f"contents role cardinality exceeded: {role}")
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
