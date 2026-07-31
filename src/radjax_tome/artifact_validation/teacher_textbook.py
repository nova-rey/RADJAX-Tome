"""Native TeacherTextbook artifact validation independent of builder ownership."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from radjax_tome.artifact_validation.corridors import validate_corridor_artifacts
from radjax_tome.io.json import int_value, read_json_object, require_fields, write_json
from radjax_tome.targets.store import TeacherTargetStore

TEACHER_TEXTBOOK_VERSION = 0
TEACHER_MANIFEST_FIELDS = (
    "artifact_type",
    "artifact_version",
    "teacher_model_id",
    "teacher_backend_type",
    "tokenizer_id",
    "vocab_size",
    "vocab_contract_path",
    "target_type",
    "dtype",
    "sequence_length",
    "num_examples",
    "shard_count",
    "created_at",
    "local_files_only",
    "allow_downloads",
    "claims_not_made",
)
EMISSION_CONFIG_FIELDS = (
    "dataset_source",
    "max_examples",
    "batch_size",
    "sequence_length",
    "logits_dtype",
    "include_hidden_states",
    "sampling_used",
    "temperature",
    "top_p",
    "top_k",
    "seed",
    "teacher_mode",
)


@dataclass(frozen=True)
class TeacherTextbookValidationReport:
    artifact_type: str = "teacher_textbook"
    artifact_version: int = TEACHER_TEXTBOOK_VERSION
    status: str = "fail"
    checks: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata_ok: bool = False
    vocab_contract_ok: bool = False
    manifest_ok: bool = False
    emission_config_ok: bool = False
    validation_report_ok: bool = False
    shards_ok: bool = False
    shape_ok: bool = False
    dtype_ok: bool = False
    count_ok: bool = False
    target_type: str | None = None
    top_k: int | None = None
    bucket_count: int | None = None
    compressed_target_ok: bool | None = None
    bucket_target_ok: bool | None = None
    mass_ok: bool | None = None
    bucket_mass_ok: bool | None = None
    bucket_count_ok: bool | None = None
    sort_ok: bool | None = None
    duplicate_ok: bool | None = None
    corridor_artifact_ok: bool | None = None
    corridor_fingerprints_ok: bool | None = None
    corridor_modes_ok: bool | None = None
    corridor_mode_count: int | None = None
    corridor_fingerprint_count: int | None = None
    corridor_observation_basis: str | None = None
    degraded_corridor_export: bool | None = None
    corridor_positions_available: int | None = None
    corridor_positions_used: int | None = None
    corridor_mode_policy: str | None = None
    corridor_stat_top_k: int | None = None
    corridor_assignment_storage_kind: str | None = None
    corridor_assignment_count: int | None = None
    claims_not_made: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_teacher_textbook(path: str | Path) -> TeacherTextbookValidationReport:
    """Validate existing artifact bytes and reports; never invoke teacher execution."""
    root = Path(path)
    checks: list[str] = []
    blockers: list[str] = []
    warnings: list[str] = []
    status: dict[str, Any] = {}
    if not root.is_dir():
        return _report(blockers=[f"teacher textbook path is not a directory: {root}"])
    for name in (
        "metadata.json",
        "vocab_contract.json",
        "teacher_manifest.json",
        "emission_config.json",
    ):
        if (root / name).is_file():
            checks.append(f"{name}: present")
        else:
            blockers.append(f"missing required file: {name}")
    if not (root / "validation_report.json").is_file():
        warnings.append(
            "validation_report.json missing; this validation can generate it"
        )
    else:
        status["validation_report_ok"] = True
        checks.append("validation_report.json: present")
    if not (root / "shards").is_dir():
        blockers.append("missing required directory: shards")
    metadata = None
    try:
        store = TeacherTargetStore.open(root)
        metadata = store.metadata
        status["target_type"] = metadata.target_type
        if metadata.target_type in {"topk_with_tail_v0", "cascaded_soft_labels_v1"}:
            status.update(
                top_k=int(metadata.target_params.get("top_k", "0")),
                compressed_target_ok=True,
                mass_ok=True,
                sort_ok=True,
                duplicate_ok=True,
            )
        if metadata.target_type == "cascaded_soft_labels_v1":
            status.update(
                bucket_count=int(metadata.target_params.get("bucket_count", "0")),
                bucket_target_ok=True,
                bucket_mass_ok=True,
                bucket_count_ok=True,
            )
        store.validate()
        status.update(
            metadata_ok=True,
            shards_ok=True,
            shape_ok=True,
            dtype_ok=True,
            count_ok=True,
        )
        checks.append("TeacherTargetStore: valid")
    except ValueError as exc:
        if metadata is not None and metadata.target_type in {
            "topk_with_tail_v0",
            "cascaded_soft_labels_v1",
        }:
            status.update(
                compressed_target_ok=False,
                mass_ok=False,
                sort_ok=False,
                duplicate_ok=False,
            )
        if metadata is not None and metadata.target_type == "cascaded_soft_labels_v1":
            status.update(
                bucket_target_ok=False, bucket_mass_ok=False, bucket_count_ok=False
            )
        blockers.append(f"TeacherTargetStore validation failed: {exc}")
    vocab: dict[str, Any] | None = None
    try:
        vocab = read_json_object(root / "vocab_contract.json")
        status["vocab_contract_ok"] = True
        checks.append("vocab_contract.json: valid JSON object")
    except (OSError, ValueError) as exc:
        blockers.append(f"vocab_contract.json invalid: {exc}")
    try:
        manifest = read_json_object(root / "teacher_manifest.json")
        manifest_blockers = require_fields(
            manifest, TEACHER_MANIFEST_FIELDS, source="teacher_manifest.json"
        )
        blockers.extend(manifest_blockers)
        status["manifest_ok"] = not manifest_blockers
        claims = manifest.get("claims_not_made", ())
        if isinstance(claims, list):
            status["claims_not_made"] = tuple(str(item) for item in claims)
        _validate_manifest_matches_metadata(manifest, metadata, blockers)
        _validate_manifest_matches_vocab(manifest, vocab, blockers)
        if not manifest_blockers:
            checks.append("teacher_manifest.json: required fields present")
    except (OSError, ValueError) as exc:
        blockers.append(f"teacher_manifest.json invalid: {exc}")
    try:
        emission = read_json_object(root / "emission_config.json")
        emission_blockers = require_fields(
            emission, EMISSION_CONFIG_FIELDS, source="emission_config.json"
        )
        blockers.extend(emission_blockers)
        status["emission_config_ok"] = not emission_blockers
        if not emission_blockers:
            checks.append("emission_config.json: required fields present")
    except (OSError, ValueError) as exc:
        blockers.append(f"emission_config.json invalid: {exc}")
    # The delivery leaf is installed by the delivery extraction; import lazily
    # so TeacherTextbook construction remains independent of optional package paths.
    from radjax_tome.artifact_validation.delivery import (
        validate_selected_exemplar_delivery,
    )

    delivery_blockers, delivery_warnings = validate_selected_exemplar_delivery(root)
    blockers.extend(delivery_blockers)
    warnings.extend(delivery_warnings)
    if delivery_blockers:
        checks.append("selected exemplar delivery: invalid")
    elif (root / "delivery_report.json").is_file():
        checks.append("selected exemplar delivery: valid")
    if (root / "delivery_report.json").is_file():
        corridor = validate_corridor_artifacts(root)
        status.update(
            corridor_artifact_ok=corridor.corridor_artifact_ok,
            corridor_fingerprints_ok=corridor.corridor_fingerprints_ok,
            corridor_modes_ok=corridor.corridor_modes_ok,
            corridor_mode_count=corridor.corridor_mode_count,
            corridor_fingerprint_count=corridor.corridor_fingerprint_count,
            corridor_observation_basis=corridor.corridor_observation_basis,
            degraded_corridor_export=corridor.degraded_corridor_export,
            corridor_positions_available=corridor.corridor_positions_available,
            corridor_positions_used=corridor.corridor_positions_used,
            corridor_mode_policy=corridor.corridor_mode_policy,
            corridor_stat_top_k=corridor.corridor_stat_top_k,
            corridor_assignment_storage_kind=corridor.corridor_assignment_storage_kind,
            corridor_assignment_count=corridor.corridor_assignment_count,
        )
        if corridor.ok:
            checks.append("corridor artifacts: valid")
    return _report(checks=checks, blockers=blockers, warnings=warnings, **status)


def write_teacher_textbook_validation_report(
    report: TeacherTextbookValidationReport, path: str | Path
) -> None:
    write_json(Path(path), report.to_dict())


def _validate_manifest_matches_metadata(
    manifest: dict[str, Any], metadata: Any, blockers: list[str]
) -> None:
    if metadata is None:
        return
    expected = {
        "teacher_model_id": metadata.model_id,
        "tokenizer_id": metadata.tokenizer_id,
        "vocab_size": metadata.vocab_size,
        "target_type": metadata.target_type,
        "dtype": metadata.dtype,
        "sequence_length": metadata.sequence_length,
        "num_examples": metadata.num_examples,
        "shard_count": metadata.shard_count,
    }
    for key, value in expected.items():
        actual = (
            int_value(manifest, key) if isinstance(value, int) else manifest.get(key)
        )
        if actual != value:
            blockers.append(
                f"teacher_manifest.json {key} mismatch: "
                f"expected {value!r}, got {actual!r}"
            )


def _validate_manifest_matches_vocab(
    manifest: dict[str, Any], vocab: dict[str, Any] | None, blockers: list[str]
) -> None:
    if vocab is None:
        return
    for key in ("tokenizer_id", "vocab_size"):
        expected = int_value(vocab, key) if key == "vocab_size" else vocab.get(key)
        actual = int_value(manifest, key) if key == "vocab_size" else manifest.get(key)
        if expected is not None and actual != expected:
            blockers.append(
                f"teacher_manifest.json {key} does not match vocab_contract.json: "
                f"expected {expected!r}, got {actual!r}"
            )


def _report(**values: Any) -> TeacherTextbookValidationReport:
    blockers = tuple(values.pop("blockers", ()) or ())
    return TeacherTextbookValidationReport(
        status="fail" if blockers else "pass",
        blockers=blockers,
        checks=tuple(values.pop("checks", ()) or ()),
        warnings=tuple(values.pop("warnings", ()) or ()),
        **values,
    )
