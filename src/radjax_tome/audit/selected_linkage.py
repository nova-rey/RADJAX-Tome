from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from radjax_tome.io.json import read_json_object, write_json
from radjax_tome.quantization import (
    ENTROPY_PARITY_QUANTIZATION_STEP,
    entropy_absolute_delta,
    entropy_parity_close,
)

AUDIT_SCHEMA_VERSION = "selected_exemplar_linkage_audit_v1"
PATH_A = "one_pass_pruned_candidate"
PATH_B = "two_pass_rerun_selected"
FULL_DEBUG_PROVENANCE = "full_debug_provenance"
STUDENT = "student"

_PASSPORT_FIELDS = (
    "selected_example_id",
    "source_shard_id",
    "source_row",
    "source_position",
    "source_score",
    "source_top_token_id",
    "source_score_policy",
    "payload_ref",
    "source_delivery_path",
    "corridor_mode_id",
    "corridor_assignment_status",
)
_COMMON_FIELDS = (
    "selected_example_id",
    "selected_position",
    "selected_score",
    "source_shard_id",
    "source_row",
    "source_position",
    "source_score",
    "source_top_token_id",
    "source_score_policy",
    "source_delivery_path",
    "payload_ref",
    "corridor_mode_id",
    "corridor_assignment_status",
    "selected_board",
)
_FLOAT_FIELDS = {"selected_score", "source_score"}


@dataclass(frozen=True)
class SelectedLinkageAuditReport:
    status: str
    selected_count: int
    checked_count: int
    bad_count: int
    delivery_path: str | None
    path_a_payload_authority: str | None
    path_b_score_pass_authority: str | None
    corridor_assignment_linkage: str
    errors: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...] = ()
    profile: str = FULL_DEBUG_PROVENANCE
    producer_shard_authority: str = "available"
    entropy_absolute_delta: float | None = None
    entropy_allowed_tolerance: float = ENTROPY_PARITY_QUANTIZATION_STEP
    entropy_parity_status: str = "pass"
    schema_version: str = AUDIT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["errors"] = list(self.errors)
        payload["warnings"] = list(self.warnings)
        return payload


def audit_selected_linkage(
    artifact_dir: Path,
    *,
    strict: bool = True,
    profile: str = FULL_DEBUG_PROVENANCE,
) -> SelectedLinkageAuditReport:
    if profile not in {FULL_DEBUG_PROVENANCE, STUDENT}:
        raise ValueError("profile must be full_debug_provenance or student")
    root = artifact_dir.resolve()
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    records = _read_selected_records(root, errors)
    delivery_report = _read_delivery_report(root)
    expected_authority = delivery_report.get("delivery_authority_hash")
    payload_index = _read_payload_index(root, errors)
    delivery_path = _delivery_path(root, records, [])
    selected_count = len(records)
    assignment_modes, example_ids = _load_selected_mode_assignments(
        root,
        records,
        errors=errors,
        warnings=warnings,
        strict=strict,
    )
    shard_cache: dict[int, dict[str, np.ndarray]] = {}
    shard_offsets = (
        _shard_offsets(root, shard_cache, errors)
        if profile == FULL_DEBUG_PROVENANCE
        else {}
    )
    student_inputs = (
        _load_student_inputs(root, errors=errors, strict=strict)
        if profile == STUDENT
        else {}
    )
    if not example_ids:
        example_ids = _dataset_example_ids(root, warnings)

    records_by_coordinate = {
        _coordinate_key(record): (index, record) for index, record in enumerate(records)
    }
    records_by_example_id: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, record in enumerate(records):
        records_by_example_id.setdefault(
            str(record.get("selected_example_id")), []
        ).append((index, record))
    seen_coordinates: set[tuple[str, int]] = set()
    payload_count = 0
    entropy_deltas: list[float] = []
    for payload, envelope in _iter_selected_payloads(root, errors):
        payload_count += 1
        coordinate = _coordinate_key(payload)
        record_entry = records_by_coordinate.get(coordinate)
        if record_entry is None:
            candidates = records_by_example_id.get(
                str(payload.get("selected_example_id")), []
            )
            if len(candidates) == 1:
                record_entry = candidates[0]
        record = None if record_entry is None else record_entry[1]
        if record is None:
            _append_record_error(
                errors,
                record=None,
                payload=payload,
                mismatch_fields=["record_missing"],
                source_values={"coordinate": list(coordinate)},
            )
            continue
        if coordinate in seen_coordinates:
            _append_record_error(
                errors,
                record=record,
                payload=payload,
                mismatch_fields=["duplicate_payload_coordinate"],
                source_values={"coordinate": list(coordinate)},
            )
            continue
        seen_coordinates.add(coordinate)
        mismatch_fields: list[str] = []
        source_values: dict[str, Any] = {"coordinate": list(coordinate)}
        expected_index = record_entry[0]
        envelope_record_index = envelope.get("record_index")
        if envelope_record_index is not None:
            if _int_or_none(envelope_record_index) != expected_index:
                mismatch_fields.append("selected_example_id")
        elif expected_index != payload_count - 1:
            mismatch_fields.append("selected_example_id")
        entropy_delta = _entropy_delta(
            payload.get("teacher_entropy"), record.get("source_score")
        )
        if entropy_delta is not None:
            entropy_deltas.append(entropy_delta)
        source_values.update(
            {
                "entropy_absolute_delta": entropy_delta,
                "entropy_allowed_tolerance": ENTROPY_PARITY_QUANTIZATION_STEP,
                "entropy_parity_status": (
                    "pass"
                    if entropy_delta is not None
                    and entropy_delta <= ENTROPY_PARITY_QUANTIZATION_STEP
                    else "fail"
                ),
                "payload_hash": envelope.get("payload_hash"),
            }
        )
        _check_payload_index_reconciliation(
            payload,
            payload_index=payload_index,
            envelope=envelope,
            mismatch_fields=mismatch_fields,
        )
        _check_payload_envelope(
            payload,
            envelope=envelope,
            expected_authority=expected_authority,
            mismatch_fields=mismatch_fields,
        )
        _check_common_record_payload(record, payload, mismatch_fields)
        row = _int_or_none(record.get("source_row"))
        position = _int_or_none(record.get("source_position"))
        shard = (
            _source_shard(root, record, shard_cache, mismatch_fields)
            if profile == FULL_DEBUG_PROVENANCE
            else None
        )
        if (
            profile == FULL_DEBUG_PROVENANCE
            and shard is not None
            and row is not None
            and position is not None
        ):
            _check_source_coordinate(
                record,
                payload,
                shard=shard,
                row=row,
                position=position,
                mismatch_fields=mismatch_fields,
                source_values=source_values,
            )
            _check_source_example_id(
                record,
                shard_id=_int_or_none(record.get("source_shard_id")),
                row=row,
                shard_offsets=shard_offsets,
                example_ids=example_ids,
                mismatch_fields=mismatch_fields,
                source_values=source_values,
            )
        elif profile == STUDENT:
            _check_student_coordinate(
                record,
                payload,
                student_inputs=student_inputs,
                mismatch_fields=mismatch_fields,
                source_values=source_values,
            )
        _check_mode_assignment(
            record,
            assignment_modes=assignment_modes,
            mismatch_fields=mismatch_fields,
            source_values=source_values,
        )
        if mismatch_fields:
            _append_record_error(
                errors,
                record=record,
                payload=payload,
                mismatch_fields=mismatch_fields,
                source_values=source_values,
            )

    missing_coordinates = set(records_by_coordinate).difference(seen_coordinates)
    if missing_coordinates:
        for coordinate in sorted(missing_coordinates):
            _append_record_error(
                errors,
                record=records_by_coordinate[coordinate][1],
                payload=None,
                mismatch_fields=["payload_missing"],
                source_values={"coordinate": list(coordinate)},
            )
    checked_count = min(len(records), payload_count)
    if payload_count != len(records):
        _append_global_error(
            errors,
            mismatch_fields=["selected_payload_count"],
            source_values={
                "record_count": len(records),
                "payload_count": payload_count,
            },
        )

    path_a_status = _authority_status(records, errors, PATH_A)
    path_b_status = _authority_status(records, errors, PATH_B)
    corridor_status = (
        "fail"
        if any(
            "corridor_mode_id" in error.get("mismatch_fields", ())
            or "corridor_assignment_status" in error.get("mismatch_fields", ())
            or "mode_assignment" in error.get("mismatch_fields", ())
            for error in errors
        )
        else "pass"
    )
    return SelectedLinkageAuditReport(
        status="fail" if errors else "pass",
        selected_count=selected_count,
        checked_count=checked_count,
        bad_count=len(errors),
        delivery_path=delivery_path,
        path_a_payload_authority=path_a_status,
        path_b_score_pass_authority=path_b_status,
        corridor_assignment_linkage=corridor_status,
        errors=tuple(errors),
        warnings=tuple(warnings),
        profile=profile,
        producer_shard_authority=(
            "not_available_in_student_profile" if profile == STUDENT else "available"
        ),
        entropy_absolute_delta=max(entropy_deltas, default=None),
        entropy_allowed_tolerance=ENTROPY_PARITY_QUANTIZATION_STEP,
        entropy_parity_status=(
            "pass"
            if not any(
                error.get("source_values", {}).get("entropy_parity_status") == "fail"
                for error in errors
            )
            else "fail"
        ),
    )


def write_selected_linkage_audit(
    report: SelectedLinkageAuditReport,
    path: Path,
) -> None:
    write_json(path, report.to_dict())


def _read_selected_records(
    root: Path,
    errors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    path = root / "leaderboards" / "selected_exemplars.json"
    try:
        payload = read_json_object(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _append_global_error(
            errors,
            mismatch_fields=["selected_records_missing_or_invalid"],
            source_values={"path": str(path), "error": str(exc)},
        )
        return []
    records = payload.get("selected_exemplars")
    if not isinstance(records, list) or any(
        not isinstance(item, dict) for item in records
    ):
        _append_global_error(
            errors,
            mismatch_fields=["selected_records_invalid"],
            source_values={"path": str(path)},
        )
        return []
    return records


def _read_selected_payloads(
    root: Path,
    errors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected_dir = root / "selected_exemplars"
    paths = sorted(selected_dir.glob("selected-exemplars-*.json"))
    if not paths:
        _append_global_error(
            errors,
            mismatch_fields=["selected_payload_shards_missing"],
            source_values={"path": str(selected_dir)},
        )
        return []
    payloads: list[dict[str, Any]] = []
    for shard_index, path in enumerate(paths):
        try:
            shard = read_json_object(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            _append_global_error(
                errors,
                mismatch_fields=["selected_payload_shard_invalid"],
                source_values={"path": str(path), "error": str(exc)},
            )
            continue
        items = shard.get("selected_exemplars")
        if not isinstance(items, list) or any(
            not isinstance(item, dict) for item in items
        ):
            _append_global_error(
                errors,
                mismatch_fields=["selected_payload_shard_records_invalid"],
                source_values={"path": str(path)},
            )
            continue
        for shard_row, item in enumerate(items):
            item = dict(item)
            item.setdefault("_audit_payload_shard_index", shard_index)
            item.setdefault("_audit_payload_shard_row", shard_row)
            payloads.append(item)
    return payloads


def _selected_payload_paths(root: Path, errors: list[dict[str, Any]]) -> list[Path]:
    selected_dir = root / "selected_exemplars"
    paths = sorted(selected_dir.glob("selected-exemplars-*.json"))
    if not paths:
        _append_global_error(
            errors,
            mismatch_fields=["selected_payload_shards_missing"],
            source_values={"path": str(selected_dir)},
        )
    return paths


def _iter_selected_payloads(
    root: Path,
    errors: list[dict[str, Any]],
):
    for shard_index, path in enumerate(_selected_payload_paths(root, errors)):
        try:
            shard = read_json_object(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            _append_global_error(
                errors,
                mismatch_fields=["selected_payload_shard_invalid"],
                source_values={"path": str(path), "error": str(exc)},
            )
            continue
        items = shard.get("selected_exemplars")
        if not isinstance(items, list) or any(
            not isinstance(item, dict) for item in items
        ):
            _append_global_error(
                errors,
                mismatch_fields=["selected_payload_shard_records_invalid"],
                source_values={"path": str(path)},
            )
            continue
        envelope = {
            "path": str(path),
            "shard_index": shard_index,
            "record_index": shard.get("record_index"),
            "delivery_authority_hash": shard.get("delivery_authority_hash"),
            "payload_hash": shard.get("payload_hash"),
            "payload_hash_valid": (
                shard.get("payload_hash") == _payload_hash(shard)
                if shard.get("payload_hash") is not None
                else None
            ),
            "native": shard.get("payload_hash") is not None,
        }
        for shard_row, item in enumerate(items):
            payload = dict(item)
            payload.setdefault("_audit_payload_shard_index", shard_index)
            payload.setdefault("_audit_payload_shard_row", shard_row)
            yield payload, envelope


def _read_payload_index(
    root: Path,
    errors: list[dict[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    path = root / "selected_exemplars" / "payload_index.json"
    if not path.is_file():
        return {}
    try:
        payload = read_json_object(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _append_global_error(
            errors,
            mismatch_fields=["payload_index_invalid"],
            source_values={"path": str(path), "error": str(exc)},
        )
        return {}
    items = payload.get("selected_exemplars")
    if not isinstance(items, list):
        _append_global_error(
            errors,
            mismatch_fields=["payload_index_records_invalid"],
            source_values={"path": str(path)},
        )
        return {}
    index: dict[tuple[str, int], dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            _append_global_error(
                errors,
                mismatch_fields=["payload_index_record_invalid"],
                source_values={"path": str(path)},
            )
            continue
        coordinate = _coordinate_key(item)
        if coordinate in index:
            _append_global_error(
                errors,
                mismatch_fields=["payload_index_duplicate_coordinate"],
                source_values={"coordinate": list(coordinate)},
            )
            continue
        index[coordinate] = item
    return index


def _coordinate_key(item: dict[str, Any]) -> tuple[str, int]:
    position = _int_or_none(item.get("selected_position"))
    return str(item.get("selected_example_id")), -1 if position is None else position


def _check_payload_index_reconciliation(
    payload: dict[str, Any],
    *,
    payload_index: dict[tuple[str, int], dict[str, Any]],
    envelope: dict[str, Any],
    mismatch_fields: list[str],
) -> None:
    if not payload_index:
        return
    indexed = payload_index.get(_coordinate_key(payload))
    if indexed is None:
        mismatch_fields.append("payload_index.coordinate")
        return
    if indexed.get("payload_ref") != payload.get("payload_ref"):
        mismatch_fields.append("payload_index.payload_ref")
    indexed_hash = indexed.get("payload_hash")
    if indexed_hash is not None and indexed_hash != envelope.get("payload_hash"):
        mismatch_fields.append("payload_index.payload_hash")


def _check_payload_envelope(
    payload: dict[str, Any],
    *,
    envelope: dict[str, Any],
    expected_authority: Any,
    mismatch_fields: list[str],
) -> None:
    if expected_authority is not None:
        # Native C6 shards carry the authority on the shard envelope.  The
        # legacy Path A JSON shard carries it on each selected payload record;
        # require whichever authority surface the artifact actually exposes.
        if (
            envelope.get("native")
            and envelope.get("delivery_authority_hash") != expected_authority
        ):
            mismatch_fields.append("delivery_authority_hash")
        if payload.get("delivery_authority_hash") != expected_authority:
            mismatch_fields.append("payload.delivery_authority_hash")
    if envelope.get("native") and envelope.get("payload_hash_valid") is not True:
        mismatch_fields.append("payload_hash")


def _payload_hash(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "payload_hash"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_delivery_report(root: Path) -> dict[str, Any]:
    try:
        return read_json_object(root / "delivery_report.json")
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _delivery_path(
    root: Path,
    records: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
) -> str | None:
    report_path = root / "delivery_report.json"
    if report_path.is_file():
        try:
            return str(read_json_object(report_path).get("delivery_path") or "") or None
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    values = {
        str(item.get("source_delivery_path"))
        for item in [*records, *payloads]
        if item.get("source_delivery_path") is not None
    }
    return next(iter(values)) if len(values) == 1 else None


def _check_common_record_payload(
    record: dict[str, Any],
    payload: dict[str, Any],
    mismatch_fields: list[str],
) -> None:
    for field in _PASSPORT_FIELDS:
        if field not in record:
            mismatch_fields.append(f"record.{field}")
        if field not in payload:
            mismatch_fields.append(f"payload.{field}")
    for field in _COMMON_FIELDS:
        left = record.get(field)
        right = payload.get(field)
        if field in _FLOAT_FIELDS:
            if not _close(left, right):
                mismatch_fields.append(field)
        elif left != right:
            mismatch_fields.append(field)
    if _int_or_none(record.get("selected_position")) != _int_or_none(
        record.get("source_position")
    ):
        mismatch_fields.append("selected_position")
    if not _close(record.get("selected_score"), record.get("source_score")):
        mismatch_fields.append("selected_score")
    if not _entropy_close(payload.get("teacher_entropy"), record.get("source_score")):
        mismatch_fields.append("teacher_entropy")


def _check_source_coordinate(
    record: dict[str, Any],
    payload: dict[str, Any],
    *,
    shard: dict[str, np.ndarray],
    row: int,
    position: int,
    mismatch_fields: list[str],
    source_values: dict[str, Any],
) -> None:
    entropy_key = (
        "corridor_entropy"
        if "corridor_entropy" in shard
        else "corridor_teacher_entropy"
    )
    corridor_entropy = _array_value(shard, entropy_key, row, position)
    corridor_top_token_id = _array_value(
        shard,
        "corridor_top_token_ids",
        row,
        position,
    )
    source_values.update(
        {
            "corridor_entropy": corridor_entropy,
            "corridor_top_token_id": corridor_top_token_id,
        }
    )
    if corridor_entropy is None or not _entropy_close(
        corridor_entropy, record.get("source_score")
    ):
        mismatch_fields.append("source_score")
    delivery_path = record.get("source_delivery_path")
    if delivery_path == PATH_A:
        _check_path_a_authority(
            record,
            payload,
            shard=shard,
            row=row,
            position=position,
            mismatch_fields=mismatch_fields,
            source_values=source_values,
        )
    elif delivery_path == PATH_B:
        _check_path_b_authority(
            record,
            payload,
            shard=shard,
            row=row,
            position=position,
            corridor_top_token_id=corridor_top_token_id,
            mismatch_fields=mismatch_fields,
            source_values=source_values,
        )
    else:
        mismatch_fields.append("source_delivery_path")


def _check_student_coordinate(
    record: dict[str, Any],
    payload: dict[str, Any],
    *,
    student_inputs: dict[str, np.ndarray],
    mismatch_fields: list[str],
    source_values: dict[str, Any],
) -> None:
    example_id = str(record.get("selected_example_id"))
    input_ids = student_inputs.get(example_id)
    position = _int_or_none(record.get("source_position"))
    if input_ids is None:
        mismatch_fields.append("student_input_ids")
        return
    source_values["student_input_shape"] = list(input_ids.shape)
    if position is None or not 0 <= position < input_ids.shape[0]:
        mismatch_fields.append("source_position")
    if _first_token(payload) != _int_or_none(record.get("source_top_token_id")):
        mismatch_fields.append("top_token_ids[0]")


def _check_path_a_authority(
    record: dict[str, Any],
    payload: dict[str, Any],
    *,
    shard: dict[str, np.ndarray],
    row: int,
    position: int,
    mismatch_fields: list[str],
    source_values: dict[str, Any],
) -> None:
    source_top_token_id = _int_or_none(record.get("source_top_token_id"))
    payload_top_token_id = _first_token(payload)
    source_values["payload_top_token_id"] = payload_top_token_id
    if source_top_token_id != payload_top_token_id:
        mismatch_fields.append("source_top_token_id")
    payload_ref = record.get("payload_ref")
    if not isinstance(payload_ref, dict):
        mismatch_fields.append("payload_ref")
        return
    expected_ref = {
        "kind": "one_pass_candidate_v1",
        "source_shard_id": record.get("source_shard_id"),
        "source_row": record.get("source_row"),
        "source_position": position,
        "source_score": record.get("source_score"),
        "source_top_token_id": source_top_token_id,
    }
    for field, expected in expected_ref.items():
        actual = payload_ref.get(field)
        matches = (
            _close(actual, expected) if field == "source_score" else actual == expected
        )
        if not matches:
            mismatch_fields.append(f"payload_ref.{field}")
            if field == "source_position":
                mismatch_fields.append("source_position")
    if "exemplar_source_top_token_ids" not in shard:
        return
    source = np.asarray(shard["exemplar_source_top_token_ids"])
    positions = np.asarray(shard.get("exemplar_positions", ()))
    sequence_length = int(np.asarray(shard["corridor_teacher_entropy"]).shape[1])
    candidate_rank = (
        _int_or_none(payload_ref.get("candidate_rank"))
        if payload_ref is not None
        else None
    )
    if source.ndim >= 3 and int(source.shape[1]) == sequence_length:
        payload_position = position
    else:
        matches = [
            rank
            for rank, candidate_position in enumerate(positions[row].tolist())
            if int(candidate_position) == position
            and int(source[row, rank, 0]) == source_top_token_id
        ]
        payload_position = matches[0] if len(matches) == 1 else None
        if payload_position is None:
            mismatch_fields.append("payload_ref.candidate_rank")
        elif candidate_rank != payload_position:
            mismatch_fields.append("payload_ref.candidate_rank")
    source_payload_token_id = (
        None if payload_position is None else int(source[row, payload_position, 0])
    )
    source_values.update(
        {
            "payload_coordinate": payload_position,
            "exemplar_source_top_token_id": source_payload_token_id,
        }
    )
    if source_payload_token_id != source_top_token_id:
        mismatch_fields.append("source_top_token_id")
    score_top_token_id = _int_or_none(record.get("score_top_token_id"))
    corridor_top_token_id = _array_value(
        shard,
        "corridor_top_token_ids",
        row,
        position,
    )
    if score_top_token_id is not None and score_top_token_id != corridor_top_token_id:
        mismatch_fields.append("score_top_token_id")


def _check_path_b_authority(
    record: dict[str, Any],
    payload: dict[str, Any],
    *,
    shard: dict[str, np.ndarray],
    row: int,
    position: int,
    corridor_top_token_id: int | float | None,
    mismatch_fields: list[str],
    source_values: dict[str, Any],
) -> None:
    if bool(record.get("c5_authoritative_coordinate")):
        source_top_token_id = _int_or_none(record.get("source_top_token_id"))
        source_values["payload_top_token_id"] = _first_token(payload)
        if corridor_top_token_id != source_top_token_id:
            mismatch_fields.append("corridor_top_token_id")
        if _first_token(payload) != source_top_token_id:
            mismatch_fields.append("top_token_ids[0]")
        payload_ref = record.get("payload_ref")
        if not isinstance(payload_ref, dict):
            mismatch_fields.append("payload_ref")
            return
        for field in (
            "source_shard_id",
            "source_row",
            "source_position",
            "source_top_token_id",
        ):
            if payload_ref.get(field) != record.get(field):
                mismatch_fields.append(f"payload_ref.{field}")
        if not _close(payload_ref.get("source_score"), record.get("source_score")):
            mismatch_fields.append("payload_ref.source_score")
        return
    score_position = _array_value(shard, "score_selected_position", row)
    score = _array_value(shard, "score_selected_position_entropy", row)
    score_top_token_id = _array_value(shard, "score_top_token_id", row)
    source_values.update(
        {
            "score_selected_position": score_position,
            "score_selected_position_entropy": score,
            "score_top_token_id": score_top_token_id,
            "payload_top_token_id": _first_token(payload),
        }
    )
    if score_position != position:
        mismatch_fields.append("source_position")
    if not _close(score, record.get("source_score")):
        mismatch_fields.append("source_score")
    source_top_token_id = _int_or_none(record.get("source_top_token_id"))
    if score_top_token_id != source_top_token_id:
        mismatch_fields.append("source_top_token_id")
    if corridor_top_token_id != source_top_token_id:
        mismatch_fields.append("corridor_top_token_id")
    if _first_token(payload) != source_top_token_id:
        mismatch_fields.append("top_token_ids[0]")
    payload_ref = record.get("payload_ref")
    if not isinstance(payload_ref, dict):
        mismatch_fields.append("payload_ref")
        return
    expected_ref = {
        "kind": "corridor_exemplar_score_pass_v1",
        "source_shard_id": record.get("source_shard_id"),
        "source_row": record.get("source_row"),
        "source_position": score_position,
        "source_score": score,
        "source_top_token_id": score_top_token_id,
    }
    for field, expected in expected_ref.items():
        actual = payload_ref.get(field)
        if field == "source_score":
            matches = _close(actual, expected)
        else:
            matches = actual == expected
        if not matches:
            mismatch_fields.append(f"payload_ref.{field}")


def _source_shard(
    root: Path,
    record: dict[str, Any],
    cache: dict[int, dict[str, np.ndarray]],
    mismatch_fields: list[str],
) -> dict[str, np.ndarray] | None:
    shard_id = _int_or_none(record.get("source_shard_id"))
    row = _int_or_none(record.get("source_row"))
    if shard_id is None or row is None:
        mismatch_fields.append("source_shard_id" if shard_id is None else "source_row")
        return None
    try:
        shard = cache.setdefault(shard_id, _load_shard(root, shard_id))
        row_count = int(np.asarray(shard["input_ids"]).shape[0])
    except (IndexError, KeyError, OSError, TypeError, ValueError):
        mismatch_fields.append("source_shard_id")
        return None
    if not 0 <= row < row_count:
        mismatch_fields.append("source_row")
        return None
    return shard


def _load_shard(root: Path, shard_id: int) -> dict[str, np.ndarray]:
    path = root / "shards" / f"shard-{shard_id:05d}.npz"
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _shard_offsets(
    root: Path,
    cache: dict[int, dict[str, np.ndarray]],
    errors: list[dict[str, Any]],
) -> dict[int, int]:
    offsets: dict[int, int] = {}
    offset = 0
    for path in sorted((root / "shards").glob("shard-*.npz")):
        try:
            shard_id = int(path.stem.split("-")[-1])
            shard = cache.setdefault(shard_id, _load_shard(root, shard_id))
            offsets[shard_id] = offset
            offset += int(np.asarray(shard["input_ids"]).shape[0])
        except (IndexError, KeyError, OSError, TypeError, ValueError) as exc:
            _append_global_error(
                errors,
                mismatch_fields=["source_shard_invalid"],
                source_values={"path": str(path), "error": str(exc)},
            )
    return offsets


def _check_source_example_id(
    record: dict[str, Any],
    *,
    shard_id: int | None,
    row: int,
    shard_offsets: dict[int, int],
    example_ids: list[str],
    mismatch_fields: list[str],
    source_values: dict[str, Any],
) -> None:
    if shard_id is None or shard_id not in shard_offsets or not example_ids:
        return
    global_row = shard_offsets[shard_id] + row
    source_values["source_global_row"] = global_row
    if not 0 <= global_row < len(example_ids):
        mismatch_fields.append("source_row")
        return
    source_example_id = example_ids[global_row]
    source_values["source_example_id"] = source_example_id
    if source_example_id != str(record.get("selected_example_id")):
        mismatch_fields.append("selected_example_id")


def _load_selected_mode_assignments(
    root: Path,
    records: list[dict[str, Any]],
    *,
    errors: list[dict[str, Any]],
    warnings: list[str],
    strict: bool,
) -> tuple[dict[tuple[str, int], int], list[str]]:
    manifest_path = root / "corridors" / "mode_assignments.json"
    try:
        manifest = read_json_object(manifest_path)
        arrays = manifest["arrays"]
        metadata_spec = manifest["examples_metadata"]
        example_ids = _read_example_metadata(root / str(metadata_spec["path"]))
        example_index = np.load(
            root / str(arrays["position_example_index"]["path"]),
            mmap_mode="r",
        )
        positions = np.load(root / str(arrays["position"]["path"]), mmap_mode="r")
        mode_ids = np.load(root / str(arrays["mode_id"]["path"]), mmap_mode="r")
    except (IndexError, KeyError, OSError, TypeError, ValueError) as exc:
        message = f"mode assignments unavailable: {exc}"
        if strict:
            _append_global_error(
                errors,
                mismatch_fields=["mode_assignment"],
                source_values={"path": str(manifest_path), "error": str(exc)},
            )
        else:
            warnings.append(message)
        return {}, []
    selected_keys = {
        (str(record.get("selected_example_id")), int(record.get("source_position", -1)))
        for record in records
    }
    selected_example_ids = {key[0] for key in selected_keys}
    selected_indices = [
        index
        for index, example_id in enumerate(example_ids)
        if example_id in selected_example_ids
    ]
    if not selected_indices:
        return {}, example_ids
    indexes = np.flatnonzero(np.isin(example_index, np.asarray(selected_indices)))
    assignments: dict[tuple[str, int], int] = {}
    for index in indexes.tolist():
        metadata_index = int(example_index[index])
        key = (example_ids[metadata_index], int(positions[index]))
        if key in selected_keys:
            assignments[key] = int(mode_ids[index])
    return assignments, example_ids


def _load_student_inputs(
    root: Path,
    *,
    errors: list[dict[str, Any]],
    strict: bool,
) -> dict[str, np.ndarray]:
    manifest_path = root / "corridors" / "mode_assignments.json"
    try:
        manifest = read_json_object(manifest_path)
        metadata_spec = manifest["examples_metadata"]
        example_ids = _read_example_metadata(root / str(metadata_spec["path"]))
        input_path = root / "corridors" / "mode_assignments" / "examples_input_ids.npy"
        input_ids = np.load(input_path, allow_pickle=False, mmap_mode="r")
        if input_ids.ndim != 2 or input_ids.shape[0] != len(example_ids):
            raise ValueError(
                "examples_input_ids shape does not match examples_metadata"
            )
    except (IndexError, KeyError, OSError, TypeError, ValueError) as exc:
        if strict:
            _append_global_error(
                errors,
                mismatch_fields=["student_input_ids"],
                source_values={"path": str(manifest_path), "error": str(exc)},
            )
        return {}
    return {
        example_id: input_ids[index] for index, example_id in enumerate(example_ids)
    }


def _read_example_metadata(path: Path) -> list[str]:
    rows: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            rows[int(payload["example_index"])] = str(payload["example_id"])
    return [rows[index] for index in range(len(rows))]


def _dataset_example_ids(root: Path, warnings: list[str]) -> list[str]:
    report_path = root / "delivery_report.json"
    try:
        report = read_json_object(report_path)
        dataset_path = Path(str(report["dataset_path"]))
        with dataset_path.open("r", encoding="utf-8") as handle:
            return [
                str(json.loads(line)["example_id"]) for line in handle if line.strip()
            ]
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        warnings.append(f"source example IDs unavailable: {exc}")
        return []


def _check_mode_assignment(
    record: dict[str, Any],
    *,
    assignment_modes: dict[tuple[str, int], int],
    mismatch_fields: list[str],
    source_values: dict[str, Any],
) -> None:
    if record.get("corridor_assignment_status") != "linked":
        mismatch_fields.append("corridor_assignment_status")
    key = (
        str(record.get("selected_example_id")),
        int(record.get("source_position", -1)),
    )
    assigned_mode = assignment_modes.get(key)
    source_values["mode_assignment"] = assigned_mode
    if assigned_mode is None or assigned_mode != _int_or_none(
        record.get("corridor_mode_id")
    ):
        mismatch_fields.append("corridor_mode_id")


def _authority_status(
    records: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    delivery_path: str,
) -> str | None:
    if not any(
        record.get("source_delivery_path") == delivery_path for record in records
    ):
        return None
    return (
        "fail"
        if any(error.get("delivery_path") == delivery_path for error in errors)
        else "pass"
    )


def _array_value(
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


def _first_token(payload: dict[str, Any]) -> int | None:
    values = payload.get("top_token_ids")
    if not isinstance(values, list) or not values:
        return None
    return _int_or_none(values[0])


def _close(left: Any, right: Any, *, atol: float = 1e-4) -> bool:
    try:
        return bool(np.isclose(float(left), float(right), rtol=1e-5, atol=atol))
    except (TypeError, ValueError):
        return False


def _entropy_delta(left: Any, right: Any) -> float | None:
    return entropy_absolute_delta(left, right)


def _entropy_close(left: Any, right: Any) -> bool:
    return entropy_parity_close(left, right)


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _append_record_error(
    errors: list[dict[str, Any]],
    *,
    record: dict[str, Any] | None,
    payload: dict[str, Any] | None,
    mismatch_fields: list[str],
    source_values: dict[str, Any],
) -> None:
    source = record or payload or {}
    errors.append(
        {
            "failure_stage": "selected_linkage_audit",
            "delivery_path": source.get("source_delivery_path"),
            "selected_example_id": source.get("selected_example_id"),
            "rank": None if record is None else record.get("rank"),
            "source_shard_id": source.get("source_shard_id"),
            "source_row": source.get("source_row"),
            "source_position": source.get("source_position"),
            "mismatch_fields": sorted(set(mismatch_fields)),
            "record": record,
            "payload": _audit_payload_scalars(payload),
            "source_values": source_values,
        }
    )


def _audit_payload_scalars(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """Keep failure diagnostics bounded by dropping full probability arrays."""

    if payload is None:
        return None
    scalar = {
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
    if isinstance(payload.get("top_token_ids"), list):
        scalar["payload_top_token_id"] = (
            payload["top_token_ids"][0] if payload["top_token_ids"] else None
        )
    return scalar


def _append_global_error(
    errors: list[dict[str, Any]],
    *,
    mismatch_fields: list[str],
    source_values: dict[str, Any],
) -> None:
    _append_record_error(
        errors,
        record=None,
        payload=None,
        mismatch_fields=mismatch_fields,
        source_values=source_values,
    )
