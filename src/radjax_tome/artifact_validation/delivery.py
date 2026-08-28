"""Builder-independent selected-exemplar delivery validation.

This module validates persisted delivery artifacts and provides the exact
linkage diagnostics consumed by Builder delivery adapters.  It imports no
Builder package or production implementation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from radjax_tome.artifact_validation.corridors import validate_corridor_artifacts
from radjax_tome.io.json import read_json_object
from radjax_tome.quantization import (
    ENTROPY_PARITY_QUANTIZATION_STEP,
    entropy_absolute_delta,
    entropy_parity_close,
)
from radjax_tome.targets.store import TeacherTargetStore

EXEMPLAR_DELIVERY_REPORT_FILENAME = "delivery_report.json"
EXEMPLAR_DELIVERY_REPORT_SCHEMA = "selected_exemplar_delivery_report_v1"
EXEMPLAR_DELIVERY_PARITY_REPORT_SCHEMA = "exemplar_delivery_parity_report_v1"
LEADERBOARD_REPORT_FILENAME = "leaderboard_report.json"
SELECTED_EXEMPLARS_FILENAME = "selected_exemplars.json"
CURRICULUM_ROUTES_FILENAME = "selected_routes.json"
CURRICULUM_ROUTES_SCHEMA = "selected_exemplar_curriculum_routes_v1"
SELECTED_LINKAGE_MISMATCH = (
    "selected exemplar linkage mismatch: selected record/payload does not match "
    "source candidate coordinate"
)
ONE_PASS_PRUNED_CANDIDATE = "one_pass_pruned_candidate"
TWO_PASS_RERUN_SELECTED = "two_pass_rerun_selected"
NATIVE_C6_PATH_B_EXECUTION = "native_c6_path_b_v1"
EXEMPLAR_SCORE_POLICY = "entropy_top_n_v1"
_REQUIRED_SELECTED_PAYLOAD_FIELDS = (
    "selected_example_id",
    "selected_position",
    "selected_score",
    "score_selected_position_entropy",
    "score_top_token_id",
    "source_shard_id",
    "source_row",
    "source_position",
    "source_score",
    "source_top_token_id",
    "source_score_policy",
    "payload_ref",
    "selected_policy",
    "source_delivery_path",
    "top_token_ids",
    "top_log_probs",
    "top_probs",
    "top_selection_mask",
    "effective_top_k",
    "top_mass",
    "tail_mass",
    "bucket_masses",
    "teacher_entropy",
    "sequence_length",
    "vocab_size",
    "num_buckets",
    "dynamic_top_k",
    "dynamic_mass_threshold",
    "dynamic_top_k_max",
    "top_k_saturated",
    "long_tail_class",
    "long_tail_warnings",
    "effective_top_k_fraction_of_vocab",
    "semantic_tail_tag",
    "selected_board",
    "corridor_mode_id",
    "corridor_fingerprint_id",
    "corridor_assignment_status",
)


class SelectedExemplarDeliveryError(ValueError):
    """Machine-readable coordinate trace for delivery validation failures."""

    def __init__(self, diagnostic: dict[str, Any]) -> None:
        self.diagnostic = diagnostic
        super().__init__(
            f"{SELECTED_LINKAGE_MISMATCH}: {json.dumps(diagnostic, sort_keys=True)}"
        )


def _payload_scalar_summary(
    payload: dict[str, Any],
    *,
    record_index: int,
) -> dict[str, Any]:
    summary = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "top_token_ids",
            "top_log_probs",
            "top_probs",
            "top_selection_mask",
            "bucket_masses",
        }
    }
    top_token_ids = payload.get("top_token_ids")
    if isinstance(top_token_ids, list) and top_token_ids:
        summary["payload_top_token_id"] = top_token_ids[0]
    summary["_record_index"] = record_index
    return summary


@dataclass(frozen=True)
class _DatasetExample:
    example_id: str


def _load_examples(path: Path, *, max_examples: int) -> tuple[_DatasetExample, ...]:
    examples: list[_DatasetExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if len(examples) >= max_examples:
                break
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"dataset line {line_number} must be an object")
            examples.append(
                _DatasetExample(
                    str(payload.get("example_id") or f"row-{line_number:06d}")
                )
            )
    return tuple(examples)


def validate_selected_exemplar_delivery(
    artifact_dir: Path,
) -> tuple[list[str], list[str]]:
    report_path = artifact_dir / EXEMPLAR_DELIVERY_REPORT_FILENAME
    selected_dir = artifact_dir / "selected_exemplars"
    leaderboards_dir = artifact_dir / "leaderboards"
    if not report_path.exists() and not selected_dir.exists():
        return [], []
    blockers: list[str] = []
    warnings: list[str] = []
    if not report_path.is_file():
        return ["delivery_report.json missing"], warnings
    try:
        report = read_json_object(report_path)
    except (OSError, ValueError) as exc:
        return [f"delivery_report.json invalid: {exc}"], warnings
    if report.get("non_selected_exemplar_payload_retained") is True:
        blockers.append("non_selected_exemplar_payload_retained=true")
    if report.get("selection_enabled") is True:
        if int(report.get("num_selected_exemplars") or 0) <= 0:
            blockers.append("selected exemplar count is zero")
    selected_path = leaderboards_dir / SELECTED_EXEMPLARS_FILENAME
    selected = _read_selected_exemplars(selected_path, blockers)
    payloads = _read_selected_payload_summaries(selected_dir, blockers)
    expected_selected_count = int(report.get("num_selected_exemplars") or 0)
    if selected and len(selected) != expected_selected_count:
        blockers.append("selected_exemplars.json count does not match delivery report")
    if payloads and len(payloads) != expected_selected_count:
        blockers.append("selected payload count does not match delivery report")
    _validate_selected_record_payload_linkage(
        artifact_dir,
        selected_records=selected,
        selected_payloads=payloads,
        blockers=blockers,
    )
    sequence_length = _metadata_int(artifact_dir, "sequence_length")
    selected_ids = {str(item.get("selected_example_id")) for item in selected}
    payload_ids = {str(item.get("selected_example_id")) for item in payloads}
    _validate_selected_ids_against_dataset(report, selected_ids, blockers)
    missing_payload_ids = selected_ids.difference(payload_ids)
    if missing_payload_ids:
        blockers.append(
            "selected_exemplars.json references selections without payloads: "
            + ", ".join(sorted(missing_payload_ids))
        )
    for item in selected:
        position = int(item.get("selected_position", -1))
        if sequence_length is not None and not 0 <= position < sequence_length:
            blockers.append(f"selected position outside sequence length: {position}")
    for payload in payloads:
        required_fields = _REQUIRED_SELECTED_PAYLOAD_FIELDS
        if payload.get("storage_flavor") in {
            "compact_k_monolithic",
            "compact_k_immutable_body",
        }:
            required_fields = tuple(
                field
                for field in required_fields
                if field != "top_selection_mask"
            )
        missing = [
            field for field in required_fields if field not in payload
        ]
        if missing:
            blockers.append(
                "selected exemplar payload missing compressed teacher target fields: "
                + ", ".join(missing)
            )
    if report.get("delivery_path") == TWO_PASS_RERUN_SELECTED:
        if report.get("teacher_rerun_count") != report.get(
            "selected_rerun_example_count",
            report.get("selected_example_count"),
        ):
            blockers.append(
                "Path B teacher_rerun_count does not match selected rerun examples"
            )
    for name in ("unselected_candidate_payloads", "temporary_candidates"):
        if (artifact_dir / name).exists():
            blockers.append(
                "final artifact includes temporary unselected candidate payloads: "
                f"{name}"
            )
    if report.get("non_selected_exemplar_payload_retained") is False:
        retained_arrays = _retained_one_pass_candidate_payload_arrays(artifact_dir)
        if retained_arrays:
            blockers.append(
                "final artifact retains unselected one-pass candidate payload arrays: "
                + ", ".join(retained_arrays)
            )
    corridor_validation = validate_corridor_artifacts(
        artifact_dir,
        selected_records=selected,
        selected_payloads=payloads,
        expected_selected_count=expected_selected_count,
    )
    blockers.extend(corridor_validation.blockers)
    warnings.extend(corridor_validation.warnings)
    return blockers, warnings


def _validate_selected_ids_against_dataset(
    report: dict[str, Any],
    selected_ids: set[str],
    blockers: list[str],
) -> None:
    dataset_path_value = report.get("dataset_path")
    if not dataset_path_value:
        return
    dataset_path = Path(str(dataset_path_value))
    if not dataset_path.is_file():
        blockers.append("delivery_report.json dataset_path is missing")
        return
    try:
        dataset_ids = {
            example.example_id
            for example in _load_examples(
                dataset_path,
                max_examples=max(len(selected_ids), 1_000_000_000),
            )
        }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        blockers.append(f"delivery_report.json dataset_path invalid: {exc}")
        return
    missing = selected_ids.difference(dataset_ids)
    if missing:
        blockers.append(
            "selected_exemplars.json references examples not present in dataset: "
            + ", ".join(sorted(missing))
        )


def _validate_selected_record_payload_linkage(
    artifact_dir: Path,
    *,
    selected_records: list[dict[str, Any]],
    selected_payloads: list[dict[str, Any]],
    blockers: list[str],
) -> None:
    if not selected_records or not selected_payloads:
        return
    if len(selected_records) != len(selected_payloads):
        return
    shard_cache: dict[int, dict[str, np.ndarray]] = {}
    try:
        store = TeacherTargetStore.open(artifact_dir)
    except (OSError, ValueError):
        blockers.append(SELECTED_LINKAGE_MISMATCH)
        return
    for record, payload in zip(selected_records, selected_payloads, strict=True):
        if _record_payload_tuple_mismatch(record, payload):
            blockers.append(SELECTED_LINKAGE_MISMATCH)
            return
        try:
            source_shard_id = int(record["source_shard_id"])
            source_row = int(record["source_row"])
        except (KeyError, TypeError, ValueError):
            blockers.append(SELECTED_LINKAGE_MISMATCH)
            return
        try:
            shard = shard_cache.setdefault(
                source_shard_id,
                store.read_shard(source_shard_id),
            )
        except (OSError, ValueError, KeyError):
            blockers.append(SELECTED_LINKAGE_MISMATCH)
            return
        if _source_coordinate_linkage_mismatch(
            shard,
            row=source_row,
            record=record,
            payload=payload,
        ):
            blockers.append(SELECTED_LINKAGE_MISMATCH)
            return


def _record_payload_tuple_mismatch(
    record: dict[str, Any],
    payload: dict[str, Any],
) -> bool:
    if record.get("source_delivery_path") == ONE_PASS_PRUNED_CANDIDATE and (
        not isinstance(record.get("payload_ref"), dict)
        or not record.get("payload_ref")
        or not isinstance(payload.get("payload_ref"), dict)
        or not payload.get("payload_ref")
    ):
        return True
    fields = [
        "selected_example_id",
        "selected_position",
        "score_top_token_id",
        "source_shard_id",
        "source_row",
        "source_position",
        "source_top_token_id",
        "source_score_policy",
        "payload_ref",
    ]
    if not bool(record.get("c5_authoritative_coordinate")):
        fields.extend(("corridor_mode_id", "corridor_assignment_status"))
    if any(record.get(field) != payload.get(field) for field in fields):
        return True
    if "selected_board" in record and record.get("selected_board") != payload.get(
        "selected_board"
    ):
        return True
    return not (
        _close_float(record.get("selected_score"), payload.get("selected_score"))
        and _close_float(record.get("source_score"), payload.get("source_score"))
    )


def _source_coordinate_linkage_mismatch(
    shard: dict[str, np.ndarray],
    *,
    row: int,
    record: dict[str, Any],
    payload: dict[str, Any],
) -> bool:
    try:
        selected_position = int(record["selected_position"])
        source_position = int(record["source_position"])
        source_score = float(record["source_score"])
        source_top_token_id = int(record["source_top_token_id"])
    except (IndexError, KeyError, TypeError, ValueError):
        return True
    if selected_position != source_position:
        return True
    if not _close_float(record.get("selected_score"), source_score):
        return True
    if not _close_float(payload.get("selected_score"), source_score):
        return True
    if not _close_float(payload.get("source_score"), source_score):
        return True
    if not _entropy_parity_close(payload.get("teacher_entropy"), source_score):
        return True
    if int(payload.get("source_top_token_id", -1)) != source_top_token_id:
        return True
    payload_top_token_id = _first_payload_token_id(payload)
    if payload_top_token_id is None:
        return True
    if payload_top_token_id != source_top_token_id:
        return True
    source_delivery_path = record.get("source_delivery_path")
    if source_delivery_path == ONE_PASS_PRUNED_CANDIDATE:
        if _path_a_source_payload_token_mismatch(shard, row=row, record=record):
            return True
    elif source_delivery_path == TWO_PASS_RERUN_SELECTED:
        if not bool(
            record.get("c5_authoritative_coordinate")
        ) and not _path_b_score_pass_aliases_match(
            record,
            payload,
            shard,
            row=row,
        ):
            return True
    else:
        return True
    entropy_key = (
        "corridor_entropy"
        if "corridor_entropy" in shard
        else "corridor_teacher_entropy"
    )
    try:
        corridor_entropy = float(np.asarray(shard[entropy_key])[row, source_position])
    except (IndexError, KeyError, TypeError, ValueError):
        return True
    if not _close_float(corridor_entropy, source_score):
        return True
    if source_delivery_path == TWO_PASS_RERUN_SELECTED and not bool(
        record.get("c5_authoritative_coordinate")
    ):
        try:
            corridor_top_token_id = int(
                np.asarray(shard["corridor_top_token_ids"])[row, source_position]
            )
        except (IndexError, KeyError, TypeError, ValueError):
            return True
        return corridor_top_token_id != source_top_token_id
    return False


def _path_a_source_payload_token_mismatch(
    shard: dict[str, np.ndarray],
    *,
    row: int,
    record: dict[str, Any],
) -> bool:
    if "exemplar_source_top_token_ids" not in shard:
        # Candidate arrays are deliberately pruned after payload materialization.
        return False
    try:
        payload_position = _one_pass_payload_position_index(
            shard,
            row=row,
            record=record,
        )
        source_top_token_id = _source_top_token_at(
            np.asarray(shard["exemplar_source_top_token_ids"]),
            row=row,
            position=payload_position,
        )
        return source_top_token_id != int(record["source_top_token_id"])
    except (IndexError, KeyError, TypeError, ValueError):
        return True


def _path_b_score_pass_aliases_match(
    record: dict[str, Any],
    payload: dict[str, Any],
    shard: dict[str, np.ndarray],
    *,
    row: int,
) -> bool:
    payload_ref = record.get("payload_ref", {})
    if not isinstance(payload_ref, dict):
        payload_ref = {}
    if payload_ref.get("kind") != "corridor_exemplar_score_pass_v1":
        return True
    try:
        shard_score = float(np.asarray(shard["score_selected_position_entropy"])[row])
        shard_top_token_id = int(np.asarray(shard["score_top_token_id"])[row])
    except (IndexError, KeyError, TypeError, ValueError):
        return False
    return (
        _path_b_score_pass_record_matches(record, shard, row=row)
        and _close_float(payload.get("score_selected_position_entropy"), shard_score)
        and int(payload.get("score_top_token_id", -1)) == shard_top_token_id
    )


def _path_b_score_pass_record_matches(
    record: dict[str, Any],
    shard: dict[str, np.ndarray],
    *,
    row: int,
) -> bool:
    payload_ref = record.get("payload_ref", {})
    if not isinstance(payload_ref, dict):
        return False
    corridor_record = "fingerprint_corridor_representative" in (
        record.get("selection_roles") or []
    )
    try:
        if corridor_record:
            shard_position = int(record.get("source_position", -1))
            shard_score = float(
                np.asarray(
                    shard.get("corridor_entropy", shard["corridor_teacher_entropy"])
                )[row, shard_position]
            )
            shard_top_token_id = int(
                np.asarray(shard["corridor_top_token_ids"])[row, shard_position]
            )
        else:
            shard_position = int(np.asarray(shard["score_selected_position"])[row])
            shard_score = float(
                np.asarray(shard["score_selected_position_entropy"])[row]
            )
            shard_top_token_id = int(np.asarray(shard["score_top_token_id"])[row])
    except (IndexError, KeyError, TypeError, ValueError):
        return False
    return (
        payload_ref.get("kind") == "corridor_exemplar_score_pass_v1"
        and int(record.get("selected_position", -1)) == shard_position
        and int(record.get("source_position", -1)) == shard_position
        and _close_float(record.get("selected_score"), shard_score)
        and _close_float(record.get("source_score"), shard_score)
        and int(record.get("source_top_token_id", -1)) == shard_top_token_id
        and _close_float(record.get("score_selected_position_entropy"), shard_score)
        and int(record.get("score_top_token_id", -1)) == shard_top_token_id
        and _int_or_none(payload_ref.get("source_shard_id"))
        == _int_or_none(record.get("source_shard_id"))
        and _int_or_none(payload_ref.get("source_row"))
        == _int_or_none(record.get("source_row"))
        and _int_or_none(payload_ref.get("source_position")) == shard_position
        and _close_float(payload_ref.get("source_score"), shard_score)
        and _int_or_none(payload_ref.get("source_top_token_id")) == shard_top_token_id
    )


def _validate_path_b_score_pass_records(
    selected_records: list[dict[str, Any]],
    *,
    store: TeacherTargetStore,
    require_score_pass_tuple: bool = True,
) -> None:
    selected_record_order = [
        str(record.get("selected_example_id", "")) for record in selected_records
    ]
    shard_cache: dict[int, dict[str, np.ndarray]] = {}
    for record in selected_records:
        try:
            source_shard_id = int(record["source_shard_id"])
            source_row = int(record["source_row"])
            shard = shard_cache.setdefault(
                source_shard_id,
                store.read_shard(source_shard_id),
            )
        except (IndexError, KeyError, OSError, TypeError, ValueError) as exc:
            raise _path_b_delivery_error(
                record,
                store=store,
                failure_reason=(
                    f"selected record cannot be resolved to a score-pass shard: {exc}"
                ),
                selected_record_order=selected_record_order,
            ) from exc
        if require_score_pass_tuple and not _path_b_score_pass_record_matches(
            record,
            shard,
            row=source_row,
        ):
            raise _path_b_delivery_error(
                record,
                store=store,
                failure_reason=(
                    "selected record does not match its score-pass shard tuple"
                ),
                selected_record_order=selected_record_order,
            )


def _path_b_rerun_payload_mismatch(
    record: dict[str, Any],
    payload: dict[str, Any],
) -> list[str]:
    mismatch_fields: list[str] = []
    payload_top_token_id = _first_payload_token_id(payload)
    if payload_top_token_id is None:
        mismatch_fields.append("top_token_ids")
    elif payload_top_token_id != int(record["source_top_token_id"]):
        mismatch_fields.append("top_token_ids[0]")
    if not _entropy_parity_close(
        payload.get("teacher_entropy"), record.get("source_score")
    ):
        mismatch_fields.append("teacher_entropy")
    if _record_payload_tuple_mismatch(record, payload):
        mismatch_fields.append("record_payload_tuple")
    return mismatch_fields


def _path_b_delivery_error(
    record: dict[str, Any],
    *,
    store: TeacherTargetStore,
    failure_reason: str,
    selected_record_order: list[str],
    rerun_input_order: list[str] | None = None,
    rerun_row_index: int | None = None,
    rerun_payload: dict[str, Any] | None = None,
    mismatch_fields: list[str] | None = None,
) -> SelectedExemplarDeliveryError:
    source_shard_id = _int_or_none(record.get("source_shard_id"))
    source_row = _int_or_none(record.get("source_row"))
    source_position = _int_or_none(record.get("source_position"))
    shard: dict[str, np.ndarray] | None = None
    if source_shard_id is not None:
        try:
            shard = store.read_shard(source_shard_id)
        except (OSError, ValueError, KeyError):
            shard = None
    evidence_row = _resolve_score_pass_evidence_row(shard, record)
    score_fields = _path_b_shard_diagnostic_fields(
        shard,
        row=evidence_row if evidence_row is not None else source_row,
        position=source_position,
    )
    record_matches_score_pass = (
        shard is not None
        and source_row is not None
        and evidence_row == source_row
        and _path_b_score_pass_record_matches(record, shard, row=source_row)
    )
    diagnostic = {
        "failure_stage": "selected_exemplar_delivery",
        "delivery_path": TWO_PASS_RERUN_SELECTED,
        "failure_reason": failure_reason,
        "selected_example_id": record.get("selected_example_id"),
        "rank": record.get("rank"),
        "source_shard_id": source_shard_id,
        "source_row": source_row,
        "source_position": source_position,
        "source_score": record.get("source_score"),
        "source_top_token_id": record.get("source_top_token_id"),
        "payload_ref": record.get("payload_ref"),
        "selected_record": record,
        "record_matches_score_pass_tuple": record_matches_score_pass,
        "score_pass_evidence_row": evidence_row,
        "score_pass_evidence_coordinate_match": evidence_row == source_row,
        "selected_record_order": selected_record_order,
        "rerun_input_order": rerun_input_order,
        "rerun_row_index": rerun_row_index,
        "rerun_payload_top_token_id": _first_payload_token_id(rerun_payload or {}),
        "rerun_payload_teacher_entropy": (
            None if rerun_payload is None else rerun_payload.get("teacher_entropy")
        ),
        "entropy_absolute_delta": _entropy_absolute_delta(
            None if rerun_payload is None else rerun_payload.get("teacher_entropy"),
            record.get("source_score"),
        ),
        "entropy_allowed_tolerance": ENTROPY_PARITY_QUANTIZATION_STEP,
        "entropy_parity_status": (
            "pass"
            if rerun_payload is not None
            and _entropy_parity_close(
                rerun_payload.get("teacher_entropy"), record.get("source_score")
            )
            else "fail"
        ),
        "mismatch_fields": mismatch_fields or [],
        **score_fields,
    }
    return SelectedExemplarDeliveryError(diagnostic)


def _path_b_shard_diagnostic_fields(
    shard: dict[str, np.ndarray] | None,
    *,
    row: int | None,
    position: int | None,
) -> dict[str, Any]:
    if shard is None or row is None or position is None:
        return {
            "score_selected_position": None,
            "score_selected_position_entropy": None,
            "score_top_token_id": None,
            "corridor_entropy_at_source_position": None,
            "corridor_top_token_id_at_source_position": None,
        }
    return {
        "score_selected_position": _array_scalar_or_none(
            shard,
            "score_selected_position",
            row,
        ),
        "score_selected_position_entropy": _array_scalar_or_none(
            shard,
            "score_selected_position_entropy",
            row,
        ),
        "score_top_token_id": _array_scalar_or_none(
            shard,
            "score_top_token_id",
            row,
        ),
        "corridor_entropy_at_source_position": _array_scalar_or_none(
            shard,
            "corridor_entropy"
            if "corridor_entropy" in shard
            else "corridor_teacher_entropy",
            row,
            position,
        ),
        "corridor_top_token_id_at_source_position": _array_scalar_or_none(
            shard,
            "corridor_top_token_ids",
            row,
            position,
        ),
    }


def _resolve_score_pass_evidence_row(
    shard: dict[str, np.ndarray] | None,
    record: Mapping[str, Any],
) -> int | None:
    """Resolve score evidence by exact ID/position, then verify the passport row."""

    if shard is None:
        return None
    example_id = str(record.get("selected_example_id", ""))
    source_position = _int_or_none(record.get("source_position"))
    source_row = _int_or_none(record.get("source_row"))
    if source_position is None:
        return None
    try:
        example_ids = np.asarray(shard["score_example_ids"]).reshape(-1)
        positions = np.asarray(shard["score_selected_position"]).reshape(-1)
    except (KeyError, TypeError, ValueError):
        return source_row
    if np.issubdtype(example_ids.dtype, np.integer):
        matches = [
            index
            for index, candidate_position in enumerate(positions.tolist())
            if index == source_row and int(candidate_position) == source_position
        ]
    else:
        matches = [
            index
            for index, (candidate_id, candidate_position) in enumerate(
                zip(example_ids.tolist(), positions.tolist(), strict=False)
            )
            if _score_pass_example_id(candidate_id) == example_id
            and int(candidate_position) == source_position
        ]
    if len(matches) != 1:
        return source_row
    return matches[0]


def _score_pass_example_id(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _array_scalar_or_none(
    shard: dict[str, np.ndarray],
    key: str,
    row: int,
    position: int | None = None,
) -> int | float | None:
    try:
        value = np.asarray(shard[key])[row]
        if position is not None:
            value = value[position]
        scalar = value.item()
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    if isinstance(scalar, (int, np.integer)):
        return int(scalar)
    return float(scalar)


def _close_float(left: Any, right: Any, *, atol: float = 1e-4) -> bool:
    try:
        return bool(np.isclose(float(left), float(right), rtol=1e-5, atol=atol))
    except (TypeError, ValueError):
        return False


def _entropy_absolute_delta(left: Any, right: Any) -> float | None:
    return entropy_absolute_delta(left, right)


def _entropy_parity_close(left: Any, right: Any) -> bool:
    return entropy_parity_close(left, right)


def _path_a_selected_payload_mismatch(
    payload: dict[str, Any],
    record: dict[str, Any],
) -> list[str]:
    mismatch_fields: list[str] = []
    payload_top_token_id = _first_payload_token_id(payload)
    if payload_top_token_id is None:
        mismatch_fields.append("top_token_ids")
    elif payload_top_token_id != int(record["source_top_token_id"]):
        mismatch_fields.append("top_token_ids[0]")
    if not _close_float(payload.get("teacher_entropy"), record["source_score"]):
        mismatch_fields.append("teacher_entropy")
    if int(payload.get("selected_position", -1)) != int(record["source_position"]):
        mismatch_fields.append("selected_position")
    if not _close_float(payload.get("selected_score"), record["source_score"]):
        mismatch_fields.append("selected_score")
    return mismatch_fields


def _first_payload_token_id(payload: dict[str, Any]) -> int | None:
    top_token_ids = payload.get("top_token_ids")
    if top_token_ids is None:
        return None
    try:
        values = np.asarray(top_token_ids).reshape(-1)
        if values.size == 0:
            return None
        return int(values[0])
    except (TypeError, ValueError):
        return None


def _one_pass_linkage_error(
    *,
    record: dict[str, Any],
    shard: dict[str, np.ndarray],
    row: int,
    failure_reason: str,
    candidate_rank: int | None = None,
    full_sequence_top_token_id: int | None = None,
    payload_position: int | None = None,
    payload_top_token_id: int | None = None,
    payload_teacher_entropy: Any = None,
    mismatch_fields: list[str] | None = None,
) -> SelectedExemplarDeliveryError:
    positions = np.asarray(shard.get("exemplar_positions", ()))
    source_top_token_ids = np.asarray(shard.get("exemplar_source_top_token_ids", ()))
    storage_kind = _one_pass_payload_storage_kind(shard, source_top_token_ids)
    source_position = _int_or_none(record.get("source_position"))
    source_top_token_id = _int_or_none(record.get("source_top_token_id"))
    payload_ref = record.get("payload_ref")
    searched_ranks = (
        []
        if storage_kind == "full_sequence"
        else _candidate_rank_diagnostics(
            positions=positions,
            source_top_token_ids=source_top_token_ids,
            row=row,
            source_position=source_position,
            source_top_token_id=source_top_token_id,
        )
    )
    diagnostic = {
        "failure_stage": "selected_exemplar_delivery",
        "failure_reason": failure_reason,
        "delivery_path": ONE_PASS_PRUNED_CANDIDATE,
        "selected_example_id": record.get("selected_example_id"),
        "rank": record.get("rank"),
        "source_shard_id": record.get("source_shard_id"),
        "source_row": record.get("source_row"),
        "resolved_source_row": row,
        "source_position": source_position,
        "source_score": record.get("source_score"),
        "source_top_token_id": source_top_token_id,
        "payload_ref": payload_ref,
        "candidate_rank": candidate_rank,
        "exemplar_source_top_token_ids_shape": list(source_top_token_ids.shape),
        "exemplar_positions_shape": list(positions.shape),
        "payload_array_storage_kind": storage_kind,
        "candidate_ranks_searched": searched_ranks,
        "full_sequence_considered": storage_kind == "full_sequence",
        "full_sequence_source_position": source_position,
        "full_sequence_top_token_id": full_sequence_top_token_id,
        "full_sequence_top_match": (
            full_sequence_top_token_id == source_top_token_id
            if full_sequence_top_token_id is not None
            and source_top_token_id is not None
            else None
        ),
        "payload_position": payload_position,
        "payload_top_token_id": payload_top_token_id,
        "payload_teacher_entropy": payload_teacher_entropy,
        "mismatch_fields": mismatch_fields or [],
    }
    return SelectedExemplarDeliveryError(diagnostic)


def _candidate_rank_diagnostics(
    *,
    positions: np.ndarray,
    source_top_token_ids: np.ndarray,
    row: int,
    source_position: int | None,
    source_top_token_id: int | None,
) -> list[dict[str, Any]]:
    if positions.ndim != 2 or not 0 <= row < positions.shape[0]:
        return []
    diagnostics: list[dict[str, Any]] = []
    for candidate_rank, candidate_position in enumerate(positions[row].tolist()):
        candidate_top_token_id = _source_top_token_at(
            source_top_token_ids,
            row=row,
            position=candidate_rank,
        )
        diagnostics.append(
            {
                "candidate_rank": candidate_rank,
                "exemplar_position": int(candidate_position),
                "exemplar_source_top_token_id": candidate_top_token_id,
                "position_match": int(candidate_position) == source_position,
                "top_token_match": candidate_top_token_id == source_top_token_id,
            }
        )
    return diagnostics


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _one_pass_payload_ref_mismatch(
    record: dict[str, Any],
    payload_ref: dict[str, Any],
) -> list[str]:
    mismatch_fields: list[str] = []
    for field in (
        "source_shard_id",
        "source_row",
        "source_position",
        "source_top_token_id",
    ):
        if _int_or_none(record.get(field)) != _int_or_none(payload_ref.get(field)):
            mismatch_fields.append(f"payload_ref.{field}")
    if not _close_float(record.get("source_score"), payload_ref.get("source_score")):
        mismatch_fields.append("payload_ref.source_score")
    return mismatch_fields


def _retained_one_pass_candidate_payload_arrays(artifact_dir: Path) -> list[str]:
    retained: set[str] = set()
    shards_dir = artifact_dir / "shards"
    if not shards_dir.is_dir():
        return []
    for path in sorted(shards_dir.glob("shard-*.npz")):
        with np.load(path, allow_pickle=False) as loaded:
            retained.update(
                key for key in loaded.files if key.startswith("exemplar_source_")
            )
    return sorted(retained)


def _read_selected_exemplars(
    path: Path,
    blockers: list[str],
) -> list[dict[str, Any]]:
    try:
        payload = read_json_object(path)
    except (OSError, ValueError) as exc:
        blockers.append(f"selected_exemplars.json invalid: {exc}")
        return []
    selected = payload.get("selected_exemplars", [])
    if not isinstance(selected, list):
        blockers.append("selected_exemplars.json selected_exemplars must be a list")
        return []
    return [item for item in selected if isinstance(item, dict)]


def _read_selected_payload_summaries(
    selected_dir: Path,
    blockers: list[str],
) -> list[dict[str, Any]]:
    """Validate payload shards one at a time, retaining only scalar state."""

    if not selected_dir.is_dir():
        blockers.append("selected_exemplars directory missing")
        return []
    summaries: list[dict[str, Any]] = []
    for path in sorted(selected_dir.glob("selected-exemplars-*.json")):
        try:
            envelope = read_json_object(path)
        except (OSError, ValueError) as exc:
            blockers.append(f"{path.name} invalid: {exc}")
            continue
        records = envelope.get("selected_exemplars", [])
        if not isinstance(records, list):
            blockers.append(f"{path.name} selected_exemplars is invalid")
            continue
        for item in records:
            if not isinstance(item, dict):
                blockers.append(f"{path.name} selected exemplar is invalid")
                continue
            summary = _payload_scalar_summary(
                item,
                record_index=int(envelope.get("record_index", len(summaries))),
            )
            summary["payload_hash"] = envelope.get("payload_hash")
            top_token_id = item.get("top_token_ids", [None])
            if isinstance(top_token_id, list):
                summary["top_token_ids"] = top_token_id[:1]
            for key in (
                "top_log_probs",
                "top_probs",
                "top_selection_mask",
                "bucket_masses",
            ):
                if key in item:
                    value = item[key]
                    summary[key] = value[:1] if key != "bucket_masses" else value
            summaries.append(summary)
    if not summaries:
        blockers.append("selected exemplar payloads are missing")
    return summaries


def _metadata_int(artifact_dir: Path, key: str) -> int | None:
    try:
        metadata = read_json_object(artifact_dir / "metadata.json")
    except (OSError, ValueError):
        return None
    value = metadata.get(key)
    return int(value) if value is not None else None


def _one_pass_payload_position_index(
    shard: dict[str, np.ndarray],
    *,
    row: int,
    record: dict[str, Any],
) -> int:
    positions = np.asarray(shard.get("exemplar_positions", ()))
    source = np.asarray(shard["exemplar_source_top_token_ids"])
    source_position = int(record["source_position"])
    source_top_token_id = int(record["source_top_token_id"])
    payload_ref = record.get("payload_ref", {})
    candidate_rank = None
    if isinstance(payload_ref, dict):
        raw_rank = payload_ref.get("candidate_rank", payload_ref.get("position_index"))
        if raw_rank is not None:
            try:
                candidate_rank = int(raw_rank)
            except (TypeError, ValueError):
                candidate_rank = None
    storage_kind = _one_pass_payload_storage_kind(shard, source)
    if storage_kind == "full_sequence":
        full_sequence_top_token_id = _source_top_token_at(
            source,
            row=row,
            position=source_position,
        )
        if full_sequence_top_token_id == source_top_token_id:
            return source_position
        raise _one_pass_linkage_error(
            record=record,
            shard=shard,
            row=row,
            failure_reason=(
                "full-sequence source payload top token does not match record"
            ),
            candidate_rank=candidate_rank,
            full_sequence_top_token_id=full_sequence_top_token_id,
        )
    rank = _find_matching_candidate_rank(
        source,
        positions=positions,
        row=row,
        source_position=source_position,
        source_top_token_id=source_top_token_id,
    )
    if rank is None:
        raise _one_pass_linkage_error(
            record=record,
            shard=shard,
            row=row,
            failure_reason=(
                "no compact candidate payload slot matches source coordinate"
            ),
            candidate_rank=candidate_rank,
        )
    return rank


def _one_pass_payload_storage_kind(
    shard: dict[str, np.ndarray],
    source_top_token_ids: np.ndarray,
) -> str:
    entropy = np.asarray(shard.get("corridor_teacher_entropy", ()))
    if (
        source_top_token_ids.ndim >= 3
        and entropy.ndim >= 2
        and int(source_top_token_ids.shape[1]) == int(entropy.shape[1])
    ):
        return "full_sequence"
    return "compact_candidate_rank"


def _source_top_token_at(
    source_top_token_ids: np.ndarray,
    *,
    row: int,
    position: int,
) -> int | None:
    try:
        return int(source_top_token_ids[row, position, 0])
    except (IndexError, TypeError, ValueError):
        return None


def _candidate_payload_slot_matches(
    source_top_token_ids: np.ndarray,
    *,
    positions: np.ndarray,
    row: int,
    candidate_rank: int,
    source_position: int,
    source_top_token_id: int,
) -> bool:
    try:
        if (
            positions.ndim == 2
            and int(positions[row, candidate_rank]) != source_position
        ):
            return False
        return int(source_top_token_ids[row, candidate_rank, 0]) == source_top_token_id
    except (IndexError, TypeError, ValueError):
        return False


def _find_matching_candidate_rank(
    source_top_token_ids: np.ndarray,
    *,
    positions: np.ndarray,
    row: int,
    source_position: int,
    source_top_token_id: int,
) -> int | None:
    if positions.ndim != 2:
        return None
    for candidate_rank, candidate_position in enumerate(positions[row].tolist()):
        if int(candidate_position) != source_position:
            continue
        if _candidate_payload_slot_matches(
            source_top_token_ids,
            positions=positions,
            row=row,
            candidate_rank=candidate_rank,
            source_position=source_position,
            source_top_token_id=source_top_token_id,
        ):
            return candidate_rank
    return None


__all__ = [name for name in globals() if not name.startswith("__")]
