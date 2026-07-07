from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from radjax_tome.io.json import read_json_object, write_json
from radjax_tome.reports.metadata_sanity import (
    METADATA_SANITY_REPORT_FILENAME,
    build_artifact_metadata_sanity_report,
)
from radjax_tome.targets.store import TeacherTargetStore

TOME_PARITY_REPORT_SCHEMA = "tome_parity_report_v1"
TOME_PARITY_REPORT_FILENAME = "parity_report.json"

REQUIRED_SIDECARS = (
    "metadata.json",
    "vocab_contract.json",
    "teacher_manifest.json",
    "emission_config.json",
    "validation_report.json",
    "cover_page.json",
)
OPTIONAL_SIDECARS = (
    "exemplar_selection_manifest.json",
    METADATA_SANITY_REPORT_FILENAME,
    "teacher_model_provenance.json",
)
STRICT_METADATA_FIELDS = (
    "target_type",
    "sequence_length",
    "num_examples",
    "vocab_size",
    "dtype",
)
INTEGER_VALUE_ARRAYS = {
    "input_ids",
    "attention_mask",
    "top_token_ids",
    "corridor_top_token_ids",
    "exemplar_positions",
    "exemplar_selection_mask",
    "source_policy_ids",
    "exemplar_source_policy_ids",
    "score_source_policy_ids",
}
FORBIDDEN_TRUE_FIELDS = (
    "production_global_selector",
    "semantic_diversity_used",
    "utility_calibrated",
    "silent_cpu_fallback",
    "real_auto_batch_probing",
    "multidevice_scheduling",
    "tpu_jax_support",
    "network_verified",
    "upstream_hf_verified",
    "downloaded_by_radjax_tome",
)


@dataclass(frozen=True)
class TomeParityConfig:
    rtol: float = 1e-4
    atol: float = 1e-5
    compare_values: bool = True
    max_examples: int | None = None
    strict_schema: bool = True
    strict_metadata_truth: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TomeParityReport:
    report_schema: str = TOME_PARITY_REPORT_SCHEMA
    status: str = "fail"
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    left_path: str = ""
    right_path: str = ""
    left_label: str = "left"
    right_label: str = "right"
    created_at: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    schema_comparison: dict[str, Any] = field(default_factory=dict)
    sidecar_comparison: dict[str, Any] = field(default_factory=dict)
    cover_page_comparison: dict[str, Any] = field(default_factory=dict)
    metadata_truth_comparison: dict[str, Any] = field(default_factory=dict)
    target_store_comparison: dict[str, Any] = field(default_factory=dict)
    shard_comparisons: tuple[dict[str, Any], ...] = ()
    array_comparisons: tuple[dict[str, Any], ...] = ()
    selector_manifest_comparison: dict[str, Any] = field(default_factory=dict)
    corpus_provenance_comparison: dict[str, Any] = field(default_factory=dict)
    teacher_model_provenance_comparison: dict[str, Any] = field(default_factory=dict)
    claims_not_made_comparison: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_tome_artifacts(
    left_path: str | Path,
    right_path: str | Path,
    config: TomeParityConfig | None = None,
    *,
    left_label: str = "left",
    right_label: str = "right",
) -> TomeParityReport:
    parity_config = config or TomeParityConfig()
    left = Path(left_path)
    right = Path(right_path)
    blockers: list[str] = []
    warnings: list[str] = []

    left_artifact = _load_artifact(left, "left", blockers)
    right_artifact = _load_artifact(right, "right", blockers)
    sidecars = _compare_sidecars(
        left,
        right,
        left_artifact,
        right_artifact,
        blockers,
        warnings,
    )
    schema = _compare_schema(left_artifact, right_artifact, blockers)
    target_store = _compare_target_store(
        left_artifact,
        right_artifact,
        blockers,
        warnings,
    )
    shard_comparisons, array_comparisons = _compare_shards(
        left_artifact,
        right_artifact,
        parity_config,
        blockers,
        warnings,
    )
    selector = _compare_selector_manifests(left_artifact, right_artifact, blockers)
    corpus = _compare_provenance(
        "corpus",
        left_artifact.teacher_manifest.get("corpus_provenance"),
        right_artifact.teacher_manifest.get("corpus_provenance"),
        blockers,
        warnings,
        hash_fields=("source_corpus_hash", "source_corpus_manifest_hash"),
    )
    teacher_model = _compare_provenance(
        "teacher_model_provenance",
        left_artifact.teacher_manifest.get("teacher_model_provenance"),
        right_artifact.teacher_manifest.get("teacher_model_provenance"),
        blockers,
        warnings,
        hash_fields=(
            "config_hash",
            "tokenizer_hash",
            "weights_hash",
            "model_directory_hash",
        ),
    )
    metadata_truth = _compare_metadata_truth(
        left_artifact,
        right_artifact,
        blockers,
        warnings,
        strict=parity_config.strict_metadata_truth,
    )
    claims = _compare_claims(left_artifact, right_artifact, blockers)
    cover_page = _compare_cover_page(left_artifact, right_artifact, blockers)

    status = _status(blockers, warnings)
    return TomeParityReport(
        status=status,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        left_path=str(left),
        right_path=str(right),
        left_label=left_label,
        right_label=right_label,
        created_at=datetime.now(UTC).replace(microsecond=0).isoformat(),
        config=parity_config.to_dict(),
        summary={
            "schema_parity": schema["status"],
            "array_parity": _section_status(array_comparisons),
            "metadata_truth": metadata_truth["status"],
            "numeric_parity": _numeric_status(array_comparisons),
            "selector_manifest": selector["status"],
            "corpus_provenance": corpus["status"],
            "teacher_model_provenance": teacher_model["status"],
        },
        schema_comparison=schema,
        sidecar_comparison=sidecars,
        cover_page_comparison=cover_page,
        metadata_truth_comparison=metadata_truth,
        target_store_comparison=target_store,
        shard_comparisons=tuple(shard_comparisons),
        array_comparisons=tuple(array_comparisons),
        selector_manifest_comparison=selector,
        corpus_provenance_comparison=corpus,
        teacher_model_provenance_comparison=teacher_model,
        claims_not_made_comparison=claims,
    )


def write_tome_parity_report(report: TomeParityReport, path: str | Path) -> Path:
    output = Path(path)
    write_json(output, report.to_dict())
    return output


@dataclass(frozen=True)
class _Artifact:
    root: Path
    label: str
    sidecars: dict[str, dict[str, Any]]
    store: TeacherTargetStore | None
    metadata_sanity: dict[str, Any]

    @property
    def metadata(self) -> dict[str, Any]:
        return self.sidecars.get("metadata.json", {})

    @property
    def teacher_manifest(self) -> dict[str, Any]:
        return self.sidecars.get("teacher_manifest.json", {})

    @property
    def cover_page(self) -> dict[str, Any]:
        return self.sidecars.get("cover_page.json", {})


def _load_artifact(root: Path, label: str, blockers: list[str]) -> _Artifact:
    sidecars: dict[str, dict[str, Any]] = {}
    for name in (*REQUIRED_SIDECARS, *OPTIONAL_SIDECARS):
        path = root / name
        if not path.is_file():
            continue
        try:
            sidecars[name] = read_json_object(path)
        except ValueError as exc:
            blockers.append(f"{label} {name} invalid: {exc}")
    store = None
    try:
        store = TeacherTargetStore.open(root)
    except ValueError as exc:
        blockers.append(f"{label} target store invalid: {exc}")
    if METADATA_SANITY_REPORT_FILENAME in sidecars:
        metadata_sanity = sidecars[METADATA_SANITY_REPORT_FILENAME]
    else:
        metadata_sanity = build_artifact_metadata_sanity_report(root)
    return _Artifact(
        root=root,
        label=label,
        sidecars=sidecars,
        store=store,
        metadata_sanity=metadata_sanity,
    )


def _compare_sidecars(
    left: Path,
    right: Path,
    left_artifact: _Artifact,
    right_artifact: _Artifact,
    blockers: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_SIDECARS:
        left_present = (left / name).is_file()
        right_present = (right / name).is_file()
        files[name] = {
            "required": True,
            "left_present": left_present,
            "right_present": right_present,
            "status": "pass" if left_present and right_present else "fail",
        }
        if not left_present:
            blockers.append(f"left missing required sidecar: {name}")
        if not right_present:
            blockers.append(f"right missing required sidecar: {name}")
    for name in OPTIONAL_SIDECARS:
        left_present = (left / name).is_file()
        right_present = (right / name).is_file()
        status = "pass" if left_present == right_present else "warn"
        files[name] = {
            "required": False,
            "left_present": left_present,
            "right_present": right_present,
            "status": status,
        }
        if left_present != right_present:
            warnings.append(f"optional sidecar presence differs: {name}")
    _check_claimed_sidecars(left_artifact, blockers, "left")
    _check_claimed_sidecars(right_artifact, blockers, "right")
    sidecar_warnings = [
        name for name, item in files.items() if item["status"] == "warn"
    ]
    return {"status": _status([], sidecar_warnings), "files": files}


def _check_claimed_sidecars(
    artifact: _Artifact,
    blockers: list[str],
    label: str,
) -> None:
    params = _target_params(artifact)
    manifest_name = _string_or_none(params.get("exemplar_selection_manifest_path"))
    if manifest_name and not (artifact.root / manifest_name).is_file():
        blockers.append(f"{label} claims missing selector manifest: {manifest_name}")
    if (
        artifact.teacher_manifest.get("corpus_provenance")
        and "corpus" not in artifact.cover_page
    ):
        blockers.append(f"{label} cover_page.json missing corpus section")
    if (
        artifact.teacher_manifest.get("teacher_model_provenance")
        and "teacher_model_provenance" not in artifact.cover_page
    ):
        blockers.append(
            f"{label} cover_page.json missing teacher_model_provenance section"
        )


def _compare_schema(
    left: _Artifact,
    right: _Artifact,
    blockers: list[str],
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for file_name, key in (
        ("metadata.json", "schema_version"),
        ("metadata.json", "target_store_version"),
        ("teacher_manifest.json", "artifact_type"),
        ("teacher_manifest.json", "artifact_version"),
        ("validation_report.json", "artifact_type"),
        ("validation_report.json", "artifact_version"),
        ("cover_page.json", "cover_page_version"),
        ("cover_page.json", "tome_version"),
    ):
        left_value = left.sidecars.get(file_name, {}).get(key)
        right_value = right.sidecars.get(file_name, {}).get(key)
        ok = left_value == right_value
        comparisons[f"{file_name}:{key}"] = {
            "left": left_value,
            "right": right_value,
            "status": "pass" if ok else "fail",
        }
        if not ok:
            blockers.append(
                f"schema mismatch {file_name}:{key}: "
                f"left={left_value!r} right={right_value!r}"
            )
    return {"status": _section_status(comparisons.values()), "fields": comparisons}


def _compare_target_store(
    left: _Artifact,
    right: _Artifact,
    blockers: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    if left.store is None or right.store is None:
        return {"status": "fail"}
    left_meta = asdict(left.store.metadata)
    right_meta = asdict(right.store.metadata)
    fields: dict[str, Any] = {}
    for field_name in STRICT_METADATA_FIELDS:
        left_value = left_meta[field_name]
        right_value = right_meta[field_name]
        ok = left_value == right_value
        fields[field_name] = {
            "left": left_value,
            "right": right_value,
            "status": "pass" if ok else "fail",
        }
        if not ok:
            blockers.append(
                f"target store {field_name} mismatch: "
                f"left={left_value!r} right={right_value!r}"
            )
    shard_ok = left_meta["shard_count"] == right_meta["shard_count"]
    fields["shard_count"] = {
        "left": left_meta["shard_count"],
        "right": right_meta["shard_count"],
        "status": "pass" if shard_ok else "warn",
    }
    if not shard_ok:
        warnings.append("target store shard_count differs")
    return {"status": _section_status(fields.values()), "fields": fields}


def _compare_shards(
    left: _Artifact,
    right: _Artifact,
    config: TomeParityConfig,
    blockers: list[str],
    warnings: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if left.store is None or right.store is None:
        return [], []
    left_shards = left.store.list_shards()
    right_shards = right.store.list_shards()
    shard_reports: list[dict[str, Any]] = []
    array_reports: list[dict[str, Any]] = []
    if len(left_shards) != len(right_shards):
        blockers.append(
            f"shard file count mismatch: left={len(left_shards)} "
            f"right={len(right_shards)}"
        )
    for shard_id, (left_shard, right_shard) in enumerate(
        zip(left_shards, right_shards, strict=False)
    ):
        before = len(blockers)
        left_arrays = _load_npz(left_shard)
        right_arrays = _load_npz(right_shard)
        left_names = set(left_arrays)
        right_names = set(right_arrays)
        missing_right = sorted(left_names - right_names)
        missing_left = sorted(right_names - left_names)
        for name in missing_right:
            blockers.append(f"right shard {shard_id} missing array: {name}")
        for name in missing_left:
            blockers.append(f"left shard {shard_id} missing array: {name}")
        for name in sorted(left_names & right_names):
            array_reports.append(
                _compare_array(
                    shard_id,
                    name,
                    left_arrays[name],
                    right_arrays[name],
                    config,
                    blockers,
                    warnings,
                )
            )
        shard_reports.append(
            {
                "shard_id": shard_id,
                "left_path": left_shard.relative_to(left.root).as_posix(),
                "right_path": right_shard.relative_to(right.root).as_posix(),
                "left_arrays": sorted(left_names),
                "right_arrays": sorted(right_names),
                "status": "fail" if len(blockers) != before else "pass",
            }
        )
    return shard_reports, array_reports


def _compare_array(
    shard_id: int,
    name: str,
    left: np.ndarray,
    right: np.ndarray,
    config: TomeParityConfig,
    blockers: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    left_shape = tuple(int(part) for part in left.shape)
    right_shape = tuple(int(part) for part in right.shape)
    report: dict[str, Any] = {
        "shard_id": shard_id,
        "array": name,
        "left_shape": left_shape,
        "right_shape": right_shape,
        "left_dtype": str(left.dtype),
        "right_dtype": str(right.dtype),
        "finite": True,
        "value_comparison": "skipped",
        "status": "pass",
    }
    if left_shape != right_shape:
        blockers.append(
            f"array shape mismatch shard={shard_id} {name}: "
            f"left={left_shape} right={right_shape}"
        )
        report["status"] = "fail"
        return report
    if str(left.dtype) != str(right.dtype):
        blockers.append(
            f"array dtype mismatch shard={shard_id} {name}: "
            f"left={left.dtype} right={right.dtype}"
        )
        report["status"] = "fail"
        return report
    if np.issubdtype(left.dtype, np.floating):
        finite = bool(np.all(np.isfinite(left)) and np.all(np.isfinite(right)))
        report["finite"] = finite
        if not finite:
            blockers.append(f"non-finite floating values shard={shard_id} {name}")
            report["status"] = "fail"
            return report
        if config.max_examples is not None:
            left_values = left[: config.max_examples]
            right_values = right[: config.max_examples]
        else:
            left_values = left
            right_values = right
        diff = np.abs(left_values.astype(np.float64) - right_values.astype(np.float64))
        denom = np.maximum(np.abs(left_values.astype(np.float64)), 1e-12)
        tolerance = config.atol + config.rtol * np.abs(left_values.astype(np.float64))
        within = diff <= tolerance
        report.update(
            {
                "max_abs_diff": float(np.max(diff)) if diff.size else 0.0,
                "mean_abs_diff": float(np.mean(diff)) if diff.size else 0.0,
                "max_rel_diff": float(np.max(diff / denom)) if diff.size else 0.0,
                "within_tolerance_fraction": (
                    float(np.mean(within)) if within.size else 1.0
                ),
            }
        )
        if config.compare_values:
            report["value_comparison"] = "checked"
            if not bool(np.all(within)):
                blockers.append(
                    f"numeric tolerance exceeded shard={shard_id} {name}: "
                    f"max_abs_diff={report['max_abs_diff']}"
                )
                report["status"] = "fail"
        return report
    if (
        config.compare_values
        and name in INTEGER_VALUE_ARRAYS
        and not np.array_equal(left, right)
    ):
        blockers.append(f"integer value mismatch shard={shard_id} {name}")
        report["value_comparison"] = "checked"
        report["status"] = "fail"
    elif config.compare_values and np.issubdtype(left.dtype, np.integer):
        report["value_comparison"] = "checked"
        if not np.array_equal(left, right):
            warnings.append(f"integer array differs shard={shard_id} {name}")
            report["status"] = "warn"
    return report


def _compare_selector_manifests(
    left: _Artifact,
    right: _Artifact,
    blockers: list[str],
) -> dict[str, Any]:
    left_manifest = _optional_json(left.root / "exemplar_selection_manifest.json")
    right_manifest = _optional_json(right.root / "exemplar_selection_manifest.json")
    if left_manifest is None and right_manifest is None:
        return {"status": "pass", "present": False}
    if left_manifest is None or right_manifest is None:
        return {"status": "warn", "present": "one_sided"}
    fields: dict[str, Any] = {}
    for key in (
        "schema_version",
        "selection_policy",
        "deduplication_policy",
        "fulfillment_policy",
        "capture_mode",
        "production_global_selector",
        "semantic_diversity_used",
        "utility_calibrated",
    ):
        left_value = left_manifest.get(key)
        right_value = right_manifest.get(key)
        ok = left_value == right_value
        fields[key] = {
            "left": left_value,
            "right": right_value,
            "status": "pass" if ok else "fail",
        }
        if not ok and key in {
            "schema_version",
            "selection_policy",
            "deduplication_policy",
            "production_global_selector",
            "semantic_diversity_used",
            "utility_calibrated",
        }:
            blockers.append(f"selector manifest {key} mismatch")
    _check_forbidden_mapping(left_manifest, "left selector manifest", blockers)
    _check_forbidden_mapping(right_manifest, "right selector manifest", blockers)
    left_examples = _selected_examples(left_manifest)
    right_examples = _selected_examples(right_manifest)
    left_positions = _selected_positions(left_manifest)
    right_positions = _selected_positions(right_manifest)
    return {
        "status": _section_status(fields.values()),
        "present": True,
        "fields": fields,
        "selected_example_jaccard": _jaccard(left_examples, right_examples),
        "selected_position_jaccard": _jaccard(left_positions, right_positions),
        "left_selected_example_count": len(left_examples),
        "right_selected_example_count": len(right_examples),
    }


def _compare_provenance(
    name: str,
    left_value: Any,
    right_value: Any,
    blockers: list[str],
    warnings: list[str],
    *,
    hash_fields: tuple[str, ...],
) -> dict[str, Any]:
    left_present = isinstance(left_value, dict)
    right_present = isinstance(right_value, dict)
    if not left_present and not right_present:
        return {"status": "pass", "left_present": False, "right_present": False}
    if left_present != right_present:
        warnings.append(f"one-sided {name} provenance")
        return {
            "status": "warn",
            "left_present": left_present,
            "right_present": right_present,
        }
    fields: dict[str, Any] = {}
    for field_name in hash_fields:
        left_field = left_value.get(field_name)
        right_field = right_value.get(field_name)
        ok = left_field == right_field
        fields[field_name] = {
            "left": left_field,
            "right": right_field,
            "status": "pass" if ok else "fail",
        }
        if not ok:
            blockers.append(f"{name} {field_name} mismatch")
    return {
        "status": _section_status(fields.values()),
        "left_present": True,
        "right_present": True,
        "fields": fields,
    }


def _compare_metadata_truth(
    left: _Artifact,
    right: _Artifact,
    blockers: list[str],
    warnings: list[str],
    *,
    strict: bool,
) -> dict[str, Any]:
    left_status = str(left.metadata_sanity.get("status"))
    right_status = str(right.metadata_sanity.get("status"))
    if strict and left_status == "fail":
        blockers.append("left metadata sanity failed")
    if strict and right_status == "fail":
        blockers.append("right metadata sanity failed")
    if not strict and "fail" in {left_status, right_status}:
        warnings.append("metadata sanity failed under non-strict mode")
    for artifact in (left, right):
        params = _target_params(artifact)
        _check_forbidden_mapping(params, f"{artifact.label} target_params", blockers)
        _check_forbidden_mapping(
            artifact.teacher_manifest,
            f"{artifact.label} teacher_manifest.json",
            blockers,
        )
        teacher_provenance = artifact.teacher_manifest.get("teacher_model_provenance")
        if isinstance(teacher_provenance, dict):
            _check_forbidden_mapping(
                teacher_provenance,
                f"{artifact.label} teacher_model_provenance",
                blockers,
            )
            if teacher_provenance.get("network_used") is not False:
                blockers.append(
                    f"{artifact.label} teacher model provenance used network"
                )
    status = "fail" if "fail" in {left_status, right_status} and strict else "pass"
    return {
        "status": status,
        "left_metadata_sanity_status": left_status,
        "right_metadata_sanity_status": right_status,
        "left_metadata_sanity_blockers": left.metadata_sanity.get("blockers", []),
        "right_metadata_sanity_blockers": right.metadata_sanity.get("blockers", []),
    }


def _compare_claims(
    left: _Artifact,
    right: _Artifact,
    blockers: list[str],
) -> dict[str, Any]:
    left_claims = _claim_set(left)
    right_claims = _claim_set(right)
    for label, claims in (("left", left_claims), ("right", right_claims)):
        for claim in claims:
            if claim in {
                "production_global_selector_implemented",
                "semantic_embedding_diversity_implemented",
                "utility_calibrated_selector_implemented",
                "silent_cpu_fallback",
                "real_auto_batch_probing",
                "multidevice_scheduling",
                "tpu_jax_support",
                "network_model_verification",
                "model_downloaded_by_radjax_tome",
            }:
                blockers.append(f"{label} forbidden claim present: {claim}")
    return {
        "status": "pass",
        "left_claims_not_made": sorted(left_claims),
        "right_claims_not_made": sorted(right_claims),
        "shared_claims_not_made": sorted(left_claims & right_claims),
    }


def _compare_cover_page(
    left: _Artifact,
    right: _Artifact,
    blockers: list[str],
) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key in ("artifact_kind", "cover_page_version", "tome_version", "layout"):
        left_value = left.cover_page.get(key)
        right_value = right.cover_page.get(key)
        ok = left_value == right_value
        fields[key] = {
            "left": left_value,
            "right": right_value,
            "status": "pass" if ok else "fail",
        }
        if not ok:
            blockers.append(f"cover_page.json {key} mismatch")
    return {"status": _section_status(fields.values()), "fields": fields}


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


def _optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return read_json_object(path)


def _target_params(artifact: _Artifact) -> dict[str, Any]:
    value = artifact.metadata.get("target_params", {})
    if not isinstance(value, dict):
        return {}
    return {str(key): _normalise_value(item) for key, item in value.items()}


def _normalise_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    return value


def _check_forbidden_mapping(
    mapping: dict[str, Any],
    source: str,
    blockers: list[str],
) -> None:
    for field_name in FORBIDDEN_TRUE_FIELDS:
        if _bool_value(mapping.get(field_name)) is True:
            blockers.append(f"{source} forbidden truth claim: {field_name}=true")


def _claim_set(artifact: _Artifact) -> set[str]:
    claims: set[str] = set()
    for source in (artifact.teacher_manifest, artifact.cover_page):
        value = source.get("claims_not_made")
        if isinstance(value, list):
            claims.update(str(item) for item in value)
    params = _target_params(artifact)
    claims.update(
        key
        for key, value in params.items()
        if key.startswith("no_") and _bool_value(value) is True
    )
    return claims


def _selected_examples(manifest: dict[str, Any]) -> set[str]:
    examples: set[str] = set()
    for board in manifest.get("boards", []):
        if not isinstance(board, dict):
            continue
        for winner in board.get("winners", []):
            if isinstance(winner, dict) and winner.get("example_id") is not None:
                examples.add(str(winner["example_id"]))
    return examples


def _selected_positions(manifest: dict[str, Any]) -> set[str]:
    positions: set[str] = set()
    for board in manifest.get("boards", []):
        if not isinstance(board, dict):
            continue
        for winner in board.get("winners", []):
            if not isinstance(winner, dict):
                continue
            if (
                winner.get("example_id") is None
                or winner.get("selected_position") is None
            ):
                continue
            positions.add(f"{winner['example_id']}:{winner['selected_position']}")
    return positions


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def _section_status(items: Any) -> str:
    statuses = [str(item.get("status")) for item in items if isinstance(item, dict)]
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    return "pass"


def _numeric_status(items: Any) -> str:
    statuses = [
        str(item.get("status"))
        for item in items
        if isinstance(item, dict) and item.get("max_abs_diff") is not None
    ]
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    return "pass"


def _status(blockers: list[str], warnings: list[str]) -> str:
    if blockers:
        return "fail"
    if warnings:
        return "warn"
    return "pass"


def _bool_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
