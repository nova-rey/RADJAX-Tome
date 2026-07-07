from __future__ import annotations

import fnmatch
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from radjax_tome.io.json import read_json_object, write_json
from radjax_tome.provenance.hashes import sha256_file

TEACHER_MODEL_PROVENANCE_SCHEMA = "teacher_model_provenance_v1"
TEACHER_MODEL_PROVENANCE_FILENAME = "teacher_model_provenance.json"

CONFIG_FILENAMES = frozenset(
    {
        "config.json",
        "generation_config.json",
        "adapter_config.json",
        "preprocessor_config.json",
    }
)
TOKENIZER_FILENAMES = frozenset(
    {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "spiece.model",
        "tokenizer.model",
    }
)
WEIGHT_PATTERNS = (
    "model.safetensors",
    "model.safetensors.index.json",
    "*.safetensors",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "pytorch_model-*.bin",
)

SUMMARY_FIELDS = (
    "schema_version",
    "model_source_kind",
    "model_identity_confidence",
    "model_provenance_mode",
    "model_name",
    "model_name_source",
    "model_revision",
    "model_revision_source",
    "config_hash",
    "tokenizer_hash",
    "weights_hash",
    "model_directory_hash",
    "network_used",
    "local_files_only",
    "allow_downloads",
)


@dataclass(frozen=True)
class TeacherModelProvenanceValidationReport:
    status: str
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    model_identity_confidence: str | None = None
    model_source_kind: str | None = None
    model_directory_hash: str | None = None

    @property
    def ok(self) -> bool:
        return self.status in {"pass", "warn"}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_teacher_model(
    model_path: str | Path,
    *,
    model_name: str | None = None,
    model_revision: str | None = None,
    check: str = "metadata_only",
) -> dict[str, Any]:
    if check != "metadata_only":
        raise ValueError("Spec 4.2 supports only check='metadata_only'")

    root = Path(model_path)
    warnings: list[str] = []
    blockers: list[str] = []
    if not root.is_dir():
        blockers.append(f"model_path is not a directory: {root}")

    config_files = _file_records(root, "config") if root.is_dir() else []
    tokenizer_files = _file_records(root, "tokenizer") if root.is_dir() else []
    weight_files = _file_records(root, "weight") if root.is_dir() else []

    if not config_files:
        warnings.append("no recognized config files found")
    if not tokenizer_files:
        warnings.append("no recognized tokenizer files found")
    if not weight_files:
        warnings.append("no recognized weight files found")
    if root.is_dir() and not (config_files or tokenizer_files or weight_files):
        blockers.append("no recognized teacher model files found")

    hf_identity = _infer_hf_cache_identity(root)
    config_identity = _infer_config_identity(root)
    identity = _identity_fields(
        model_name=model_name,
        model_revision=model_revision,
        hf_identity=hf_identity,
        config_identity=config_identity,
    )
    if hf_identity is not None:
        source_kind = "local_hf_snapshot"
    elif root.is_dir() and (config_files or tokenizer_files or weight_files):
        source_kind = "local_directory"
    elif root.is_dir():
        source_kind = "unknown_local_path"
    else:
        source_kind = "unavailable"

    return {
        "allow_downloads": False,
        "blockers": blockers,
        "config_files": config_files,
        "config_hash": _category_hash(config_files),
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "downloaded_by_radjax_tome": False,
        "hf_identity_source": _hf_identity_source(hf_identity),
        "hf_repo_id": hf_identity["hf_repo_id"] if hf_identity else None,
        "hf_revision": hf_identity["hf_revision"] if hf_identity else None,
        "load_check": {
            "blockers": [],
            "config_loadable": None,
            "mode": check,
            "model_loadable": None,
            "status": "pass",
            "tiny_forward_ran": False,
            "tokenizer_loadable": None,
            "warnings": ["metadata_only does not import transformers or load weights"],
        },
        "local_files_only": True,
        "model_directory_hash": _directory_hash(
            config_files=config_files,
            tokenizer_files=tokenizer_files,
            weight_files=weight_files,
        ),
        "model_path": str(root),
        "model_source_kind": source_kind,
        "network_used": False,
        "schema_version": TEACHER_MODEL_PROVENANCE_SCHEMA,
        "tokenizer_files": tokenizer_files,
        "tokenizer_hash": _category_hash(tokenizer_files),
        "total_config_bytes": _total_bytes(config_files),
        "total_model_bytes": _total_bytes(
            config_files + tokenizer_files + weight_files
        ),
        "total_tokenizer_bytes": _total_bytes(tokenizer_files),
        "total_weight_bytes": _total_bytes(weight_files),
        "warnings": warnings,
        "weight_files": weight_files,
        "weights_hash": _category_hash(weight_files),
        **identity,
    }


def write_teacher_model_provenance(
    provenance: dict[str, Any],
    path: str | Path,
) -> Path:
    output = Path(path)
    write_json(output, provenance)
    return output


def validate_teacher_model_provenance(
    path: str | Path,
) -> TeacherModelProvenanceValidationReport:
    provenance_path = Path(path)
    blockers: list[str] = []
    warnings: list[str] = []
    payload: dict[str, Any] = {}
    try:
        payload = read_json_object(provenance_path)
    except ValueError as exc:
        return _report(blockers=[str(exc)])

    if payload.get("schema_version") != TEACHER_MODEL_PROVENANCE_SCHEMA:
        blockers.append("teacher_model_provenance.json schema_version is unsupported")
    model_path_value = payload.get("model_path")
    if not isinstance(model_path_value, str) or not model_path_value:
        blockers.append("teacher_model_provenance.json model_path is required")
        model_root = None
    else:
        model_root = Path(model_path_value)
        if not model_root.is_dir():
            blockers.append(
                "teacher_model_provenance.json model_path is not a directory"
            )

    if payload.get("network_used") is not False:
        blockers.append("teacher_model_provenance.json network_used must be false")
    if payload.get("allow_downloads") is not False:
        blockers.append("teacher_model_provenance.json allow_downloads must be false")
    if payload.get("local_files_only") is not True:
        blockers.append("teacher_model_provenance.json local_files_only must be true")
    if payload.get("downloaded_by_radjax_tome") is True:
        blockers.append(
            "teacher_model_provenance.json must not claim RADJAX downloaded"
        )
    if payload.get("load_check") is None:
        blockers.append("teacher_model_provenance.json load_check field is required")

    if model_root is not None and model_root.is_dir():
        config_files = _records(payload, "config_files", blockers)
        tokenizer_files = _records(payload, "tokenizer_files", blockers)
        weight_files = _records(payload, "weight_files", blockers)
        _validate_file_records(model_root, config_files, blockers)
        _validate_file_records(model_root, tokenizer_files, blockers)
        _validate_file_records(model_root, weight_files, blockers)
        _validate_hash_field(payload, "config_hash", config_files, blockers)
        _validate_hash_field(payload, "tokenizer_hash", tokenizer_files, blockers)
        _validate_hash_field(payload, "weights_hash", weight_files, blockers)
        expected_directory_hash = _directory_hash(
            config_files=config_files,
            tokenizer_files=tokenizer_files,
            weight_files=weight_files,
        )
        if payload.get("model_directory_hash") != expected_directory_hash:
            blockers.append(
                "teacher_model_provenance.json model_directory_hash does not recompute"
            )

    _validate_identity_fields(payload, blockers, warnings)
    warnings.extend(str(item) for item in payload.get("warnings", []) if item)
    blockers.extend(str(item) for item in payload.get("blockers", []) if item)
    return _report(
        blockers=blockers,
        warnings=warnings,
        model_identity_confidence=_optional_str(
            payload.get("model_identity_confidence")
        ),
        model_source_kind=_optional_str(payload.get("model_source_kind")),
        model_directory_hash=_optional_str(payload.get("model_directory_hash")),
    )


def teacher_model_provenance_summary(path: str | Path) -> dict[str, Any]:
    report = validate_teacher_model_provenance(path)
    if report.status == "fail":
        raise ValueError(
            "teacher model provenance failed validation: " + "; ".join(report.blockers)
        )
    payload = read_json_object(Path(path))
    summary = {key: payload.get(key) for key in SUMMARY_FIELDS}
    summary["teacher_model_provenance_path"] = str(Path(path))
    summary["downloaded_by_radjax_tome"] = payload.get("downloaded_by_radjax_tome")
    return summary


def teacher_model_target_params(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    summary = teacher_model_provenance_summary(path)
    return {
        "teacher_model_provenance_schema": str(summary["schema_version"]),
        "teacher_model_source_kind": str(summary["model_source_kind"]),
        "teacher_model_identity_confidence": str(summary["model_identity_confidence"]),
        "teacher_model_provenance_mode": str(summary["model_provenance_mode"]),
        "teacher_model_name": _param_value(summary["model_name"]),
        "teacher_model_name_source": str(summary["model_name_source"]),
        "teacher_model_revision": _param_value(summary["model_revision"]),
        "teacher_model_revision_source": str(summary["model_revision_source"]),
        "teacher_model_config_hash": _param_value(summary["config_hash"]),
        "teacher_model_tokenizer_hash": _param_value(summary["tokenizer_hash"]),
        "teacher_model_weights_hash": _param_value(summary["weights_hash"]),
        "teacher_model_directory_hash": _param_value(summary["model_directory_hash"]),
        "teacher_model_provenance_path": str(summary["teacher_model_provenance_path"]),
        "teacher_model_network_used": _param_value(summary["network_used"]),
    }


def discover_teacher_model_candidates(
    search_path: str | Path,
) -> tuple[dict[str, Any], ...]:
    root = Path(search_path)
    if not root.is_dir():
        raise ValueError(f"search_path is not a directory: {root}")
    candidates: list[dict[str, Any]] = []
    for path in sorted((root, *root.rglob("*")), key=lambda item: item.as_posix()):
        if not path.is_dir():
            continue
        has_config = any((path / name).is_file() for name in CONFIG_FILENAMES)
        has_tokenizer = any((path / name).is_file() for name in TOKENIZER_FILENAMES)
        has_weights = any(
            any(fnmatch.fnmatch(child.name, pattern) for pattern in WEIGHT_PATTERNS)
            for child in path.iterdir()
            if child.is_file()
        )
        hf_identity = _infer_hf_cache_identity(path)
        if not (has_config or has_tokenizer or has_weights or hf_identity):
            continue
        candidates.append(
            {
                "candidate_path": str(path),
                "has_config": has_config,
                "has_tokenizer": has_tokenizer,
                "has_weights": has_weights,
                "source_kind": (
                    "local_hf_snapshot" if hf_identity else "local_directory"
                ),
            }
        )
    return tuple(candidates)


def _file_records(root: Path, category: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(
        root.rglob("*"),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        if not path.is_file():
            continue
        if not _matches_category(path.name, category):
            continue
        records.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": "sha256:" + sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return records


def _matches_category(filename: str, category: str) -> bool:
    if category == "config":
        return filename in CONFIG_FILENAMES
    if category == "tokenizer":
        return filename in TOKENIZER_FILENAMES
    if category == "weight":
        return any(fnmatch.fnmatch(filename, pattern) for pattern in WEIGHT_PATTERNS)
    raise ValueError(f"unknown teacher model file category: {category!r}")


def _category_hash(records: list[dict[str, Any]]) -> str | None:
    if not records:
        return None
    return _sha256_json(records)


def _directory_hash(
    *,
    config_files: list[dict[str, Any]],
    tokenizer_files: list[dict[str, Any]],
    weight_files: list[dict[str, Any]],
) -> str | None:
    inventory: list[dict[str, Any]] = []
    for category, records in (
        ("config", config_files),
        ("tokenizer", tokenizer_files),
        ("weight", weight_files),
    ):
        inventory.extend({"category": category, **record} for record in records)
    if not inventory:
        return None
    return _sha256_json(sorted(inventory, key=lambda item: item["relative_path"]))


def _sha256_json(value: Any) -> str:
    import hashlib

    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _total_bytes(records: list[dict[str, Any]]) -> int:
    return sum(int(record["size_bytes"]) for record in records)


def _infer_hf_cache_identity(path: Path) -> dict[str, str] | None:
    parts = path.parts
    for index, part in enumerate(parts):
        if part != "snapshots" or index == 0 or index + 1 >= len(parts):
            continue
        repo_part = parts[index - 1]
        if not repo_part.startswith("models--"):
            continue
        repo_id = repo_part.removeprefix("models--").replace("--", "/")
        revision = parts[index + 1]
        if repo_id and revision:
            return {"hf_repo_id": repo_id, "hf_revision": revision}
    return None


def _hf_identity_source(hf_identity: dict[str, str] | None) -> str:
    if hf_identity is None:
        return "unavailable"
    return "inferred_from_local_cache_path"


def _infer_config_identity(root: Path) -> str | None:
    config_path = root / "config.json"
    if not config_path.is_file():
        return None
    try:
        config = read_json_object(config_path)
    except ValueError:
        return None
    value = config.get("_name_or_path") or config.get("model_name")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _identity_fields(
    *,
    model_name: str | None,
    model_revision: str | None,
    hf_identity: dict[str, str] | None,
    config_identity: str | None,
) -> dict[str, Any]:
    declared_name = _clean_optional(model_name)
    declared_revision = _clean_optional(model_revision)
    if declared_name is not None or declared_revision is not None:
        return {
            "hf_identity_source": (
                "inferred_from_local_cache_path"
                if hf_identity is not None
                else "unavailable"
            ),
            "model_identity_confidence": "declared",
            "model_name": declared_name,
            "model_name_source": "user_declared" if declared_name else "unknown",
            "model_provenance_mode": "user_declared",
            "model_revision": declared_revision,
            "model_revision_source": (
                "user_declared" if declared_revision else "unknown"
            ),
        }
    if hf_identity is not None:
        return {
            "model_identity_confidence": "inferred",
            "model_name": hf_identity["hf_repo_id"],
            "model_name_source": "hf_cache_path",
            "model_provenance_mode": "inferred_from_hf_cache_path",
            "model_revision": hf_identity["hf_revision"],
            "model_revision_source": "hf_cache_path",
        }
    if config_identity is not None:
        return {
            "model_identity_confidence": "verified",
            "model_name": config_identity,
            "model_name_source": "config",
            "model_provenance_mode": "inspected_local_files",
            "model_revision": None,
            "model_revision_source": "unknown",
        }
    return {
        "model_identity_confidence": "unknown",
        "model_name": None,
        "model_name_source": "unknown",
        "model_provenance_mode": "inspected_local_files",
        "model_revision": None,
        "model_revision_source": "unknown",
    }


def _records(
    payload: dict[str, Any],
    key: str,
    blockers: list[str],
) -> list[dict[str, Any]]:
    value = payload.get(key)
    if not isinstance(value, list):
        blockers.append(f"teacher_model_provenance.json {key} must be a list")
        return []
    records: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            blockers.append(
                f"teacher_model_provenance.json {key}[{index}] must be an object"
            )
            continue
        records.append(dict(item))
    return records


def _validate_file_records(
    root: Path,
    records: list[dict[str, Any]],
    blockers: list[str],
) -> None:
    for record in records:
        rel = record.get("relative_path")
        if not isinstance(rel, str) or not rel:
            blockers.append("teacher model file record relative_path is required")
            continue
        path = root / rel
        if not path.is_file():
            blockers.append(f"teacher model file missing: {rel}")
            continue
        expected_size = path.stat().st_size
        if record.get("size_bytes") != expected_size:
            blockers.append(f"teacher model file size mismatch: {rel}")
        expected_hash = "sha256:" + sha256_file(path)
        if record.get("sha256") != expected_hash:
            blockers.append(f"teacher model file hash mismatch: {rel}")


def _validate_hash_field(
    payload: dict[str, Any],
    key: str,
    records: list[dict[str, Any]],
    blockers: list[str],
) -> None:
    expected = _category_hash(records)
    if payload.get(key) != expected:
        blockers.append(f"teacher_model_provenance.json {key} does not recompute")


def _validate_identity_fields(
    payload: dict[str, Any],
    blockers: list[str],
    warnings: list[str],
) -> None:
    confidence = payload.get("model_identity_confidence")
    name_source = payload.get("model_name_source")
    revision_source = payload.get("model_revision_source")
    if confidence not in {"verified", "inferred", "declared", "unknown"}:
        blockers.append("teacher_model_provenance.json identity confidence invalid")
    if confidence == "declared":
        if payload.get("model_name") is not None and name_source != "user_declared":
            blockers.append("declared model_name must use user_declared source")
        if (
            payload.get("model_revision") is not None
            and revision_source != "user_declared"
        ):
            blockers.append("declared model_revision must use user_declared source")
    if confidence == "unknown" and payload.get("model_name"):
        blockers.append("unknown identity must not fabricate model_name")
    if confidence == "inferred" and name_source != "hf_cache_path":
        warnings.append("inferred identity should come from hf_cache_path")


def _report(
    *,
    blockers: list[str],
    warnings: list[str] | None = None,
    model_identity_confidence: str | None = None,
    model_source_kind: str | None = None,
    model_directory_hash: str | None = None,
) -> TeacherModelProvenanceValidationReport:
    warning_tuple = tuple(warnings or ())
    return TeacherModelProvenanceValidationReport(
        status="fail" if blockers else ("warn" if warning_tuple else "pass"),
        blockers=tuple(blockers),
        warnings=warning_tuple,
        model_identity_confidence=model_identity_confidence,
        model_source_kind=model_source_kind,
        model_directory_hash=model_directory_hash,
    )


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _param_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
