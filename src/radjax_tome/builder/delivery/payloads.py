"""payloads ownership for selected-exemplar delivery."""

# ruff: noqa: F403, F405
from __future__ import annotations

# Shared constants and low-level, dependency-free helpers.
from ._shared import *  # noqa: F403
from .validation import (
    _one_pass_linkage_error,
    _one_pass_payload_ref_mismatch,
    _path_a_selected_payload_mismatch,
)


def _selected_payloads_from_one_pass_capture(
    selected_records: list[dict[str, Any]],
    *,
    store: TeacherTargetStore,
    config: ExemplarDeliveryConfig,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    shard_cache: dict[int, dict[str, np.ndarray]] = {}
    for record in selected_records:
        payload_ref = record.get("payload_ref", {})
        if not isinstance(payload_ref, dict) or not payload_ref:
            raise ValueError("selected record missing one-pass payload_ref")
        source_shard_id = int(record.get("source_shard_id", -1))
        source_row = int(record.get("source_row", -1))
        if source_shard_id < 0 or source_row < 0:
            raise ValueError("selected record has invalid one-pass payload_ref")
        shard = shard_cache.setdefault(
            source_shard_id, store.read_shard(source_shard_id)
        )
        payload_ref_mismatch = _one_pass_payload_ref_mismatch(record, payload_ref)
        if payload_ref_mismatch:
            raise _one_pass_linkage_error(
                record=record,
                shard=shard,
                row=source_row,
                failure_reason=(
                    "one-pass payload reference does not match selected record "
                    "source coordinate"
                ),
                mismatch_fields=payload_ref_mismatch,
            )
        payloads.append(
            _selected_payload_from_one_pass_shard(
                record,
                shard=shard,
                row=source_row,
                config=config,
            )
        )
    return payloads


def _selected_payload_from_one_pass_shard(
    record: dict[str, Any],
    *,
    shard: dict[str, np.ndarray],
    row: int,
    config: ExemplarDeliveryConfig,
) -> dict[str, Any]:
    position = int(record["source_position"])
    payload_position = _one_pass_payload_position_index(
        shard,
        row=row,
        record=record,
    )
    top_selection_mask = _payload_slice(
        shard,
        "exemplar_source_top_selection_mask",
        row,
        payload_position,
    )
    effective_top_k = int(
        _payload_scalar(shard, "exemplar_source_effective_top_k", row, payload_position)
    )
    payload = {
        "selected_example_id": record["selected_example_id"],
        "selected_position": position,
        "selected_score": record["source_score"],
        "score_selected_position_entropy": record["source_score"],
        "score_top_token_id": record["score_top_token_id"],
        "source_shard_id": record["source_shard_id"],
        "source_row": record["source_row"],
        "source_position": record["source_position"],
        "source_score": record["source_score"],
        "source_top_token_id": record["source_top_token_id"],
        "source_score_policy": record["source_score_policy"],
        "payload_ref": record["payload_ref"],
        "selected_policy": record["selected_policy"],
        "source_delivery_path": record["source_delivery_path"],
        "delivery_authority_hash": getattr(config, "delivery_authority_hash", None),
        "selected_board": record.get("selected_board", PRIMARY_SELECTED_BOARD),
        "mode_key": record.get("mode_key"),
        "rank_by_board": record.get("rank_by_board", {}),
        "scores_by_board": record.get("scores_by_board", {}),
        "top_token_ids": _payload_slice(
            shard,
            "exemplar_source_top_token_ids",
            row,
            payload_position,
        ),
        "top_log_probs": _payload_slice(
            shard,
            "exemplar_source_top_log_probs",
            row,
            payload_position,
        ),
        "top_probs": _payload_slice(
            shard,
            "exemplar_source_top_probs",
            row,
            payload_position,
        ),
        "top_selection_mask": top_selection_mask,
        "effective_top_k": effective_top_k,
        "top_mass": _payload_scalar(
            shard,
            "exemplar_source_top_mass",
            row,
            payload_position,
        ),
        "tail_mass": _payload_scalar(
            shard,
            "exemplar_source_tail_mass",
            row,
            payload_position,
        ),
        "bucket_masses": _payload_slice(
            shard,
            "exemplar_source_bucket_masses",
            row,
            payload_position,
        ),
        "teacher_entropy": _payload_scalar(
            shard,
            "corridor_teacher_entropy",
            row,
            position,
        ),
        "sequence_length": config.sequence_length,
        "vocab_size": config.vocab_size,
        "num_buckets": config.num_buckets,
        "dynamic_top_k": _dynamic_top_k_metadata(
            config,
            effective_top_k=effective_top_k,
            source_payload="one_pass_candidate_shard",
        ),
    }
    mismatch = _path_a_selected_payload_mismatch(payload, record)
    if mismatch:
        raise _one_pass_linkage_error(
            record=record,
            shard=shard,
            row=row,
            failure_reason="materialized payload does not match its source coordinate",
            payload_position=payload_position,
            payload_top_token_id=_first_payload_token_id(payload),
            payload_teacher_entropy=payload.get("teacher_entropy"),
            mismatch_fields=mismatch,
        )
    return payload


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


def _selected_payload_from_emission(
    record: dict[str, Any],
    *,
    payload: Any,
    row: int,
    config: ExemplarDeliveryConfig,
    position_index: int | None = None,
) -> dict[str, Any]:
    position = int(record["source_position"])
    payload_position = position if position_index is None else position_index
    if "top_selection_mask" in payload:
        top_selection_mask = _payload_slice(
            payload, "top_selection_mask", row, payload_position
        )
    else:
        top_selection_mask = [True] * len(
            _payload_slice(payload, "top_token_ids", row, payload_position)
        )
    effective_top_k = int(
        _payload_scalar(payload, "effective_top_k", row, payload_position)
    )
    top_token_ids = _payload_slice(payload, "top_token_ids", row, payload_position)
    return {
        "selected_example_id": record["selected_example_id"],
        "selected_position": position,
        "selected_score": record["source_score"],
        "score_selected_position_entropy": record["source_score"],
        "score_top_token_id": record["score_top_token_id"],
        "source_shard_id": record["source_shard_id"],
        "source_row": record["source_row"],
        "source_position": record["source_position"],
        "source_score": record["source_score"],
        "source_top_token_id": record["source_top_token_id"],
        "source_score_policy": record["source_score_policy"],
        "payload_ref": record["payload_ref"],
        "selected_policy": record["selected_policy"],
        "source_delivery_path": record["source_delivery_path"],
        "delivery_authority_hash": config.delivery_authority_hash,
        "selected_board": record.get("selected_board", PRIMARY_SELECTED_BOARD),
        "mode_key": record.get("mode_key"),
        "rank_by_board": record.get("rank_by_board", {}),
        "scores_by_board": record.get("scores_by_board", {}),
        "top_token_ids": top_token_ids,
        "top_log_probs": _payload_slice(
            payload, "top_log_probs", row, payload_position
        ),
        "top_probs": _payload_slice(payload, "top_probs", row, payload_position),
        "top_selection_mask": top_selection_mask,
        "effective_top_k": effective_top_k,
        "top_mass": _payload_scalar(payload, "top_mass", row, payload_position),
        "tail_mass": _payload_scalar(payload, "tail_mass", row, payload_position),
        "bucket_masses": _payload_slice(
            payload, "bucket_masses", row, payload_position
        ),
        "teacher_entropy": _payload_scalar(
            payload, "teacher_entropy", row, payload_position
        ),
        "sequence_length": config.sequence_length,
        "vocab_size": config.vocab_size,
        "num_buckets": config.num_buckets,
        "dynamic_top_k": _dynamic_top_k_metadata(
            config,
            effective_top_k=effective_top_k,
            source_payload="backend_emit_batch",
        ),
    }


def _dynamic_top_k_metadata(
    config: ExemplarDeliveryConfig,
    *,
    effective_top_k: int,
    source_payload: str,
) -> dict[str, Any]:
    return {
        "policy": config.backend_config.dynamic_top_k_policy
        if config.backend_config is not None
        else "mass_threshold_v1",
        "requested_top_k": config.top_k,
        "effective_top_k": effective_top_k,
        "dynamic_mass_threshold": _dynamic_mass_threshold(config),
        "dynamic_top_k_max": _dynamic_top_k_max(config),
        "score_policy": config.score_policy,
        "source_payload": source_payload,
    }


def _dynamic_top_k_metadata(
    config: ExemplarDeliveryConfig,
    *,
    effective_top_k: int,
    source_payload: str,
) -> dict[str, Any]:
    return {
        "policy": config.backend_config.dynamic_top_k_policy
        if config.backend_config is not None
        else "mass_threshold_v1",
        "requested_top_k": config.top_k,
        "effective_top_k": effective_top_k,
        "dynamic_mass_threshold": _dynamic_mass_threshold(config),
        "dynamic_top_k_max": _dynamic_top_k_max(config),
        "score_policy": config.score_policy,
        "source_payload": source_payload,
    }


def _dynamic_top_k_max(config: ExemplarDeliveryConfig) -> int:
    if config.backend_config is None:
        return config.top_k
    return int(config.backend_config.dynamic_top_k_max)


def _dynamic_mass_threshold(config: ExemplarDeliveryConfig) -> float:
    if config.backend_config is None:
        return 0.95
    return float(config.backend_config.dynamic_mass_threshold)


def _candidate_is_perverse(
    candidate: Any,
    *,
    config: ExemplarDeliveryConfig,
) -> bool:
    diagnostic = _candidate_long_tail_diagnostic(candidate, config=config)
    return is_perverse_long_tail(diagnostic)


def _candidate_long_tail_diagnostic(
    candidate: Any,
    *,
    config: ExemplarDeliveryConfig,
) -> dict[str, Any]:
    effective_top_k = int(
        candidate.score_fields.get(
            "diagnostic_effective_top_k",
            candidate.score_fields.get("effective_top_k", 1),
        )
        or 1
    )
    diagnostic = long_tail_diagnostics(
        effective_top_k=effective_top_k,
        top_mass=float(
            candidate.score_fields.get(
                "diagnostic_top_mass",
                candidate.score_fields.get("top_mass", 0.0),
            )
            or 0.0
        ),
        vocab_size=config.vocab_size,
        dynamic_mass_threshold=_dynamic_mass_threshold(config),
        dynamic_top_k_max=_dynamic_top_k_max(config),
        policy=_long_tail_policy(config),
    )
    return diagnostic


def _long_tail_policy(config: ExemplarDeliveryConfig) -> LongTailPolicy:
    return LongTailPolicy(
        long_tail_warning_k=config.long_tail_warning_k,
        very_long_tail_warning_k=config.very_long_tail_warning_k,
        perverse_tail_warning_k=config.perverse_tail_warning_k,
        reject_perverse_exemplars=config.reject_perverse_exemplars,
    )


def _primary_budget(config: ExemplarDeliveryConfig) -> int | None:
    return (
        config.primary_selected_exemplar_budget
        if config.primary_selected_exemplar_budget is not None
        else config.selected_exemplar_budget
    )


def _records_by_selected_board(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        board_id: [
            record
            for record in records
            if str(record.get("selected_board") or PRIMARY_SELECTED_BOARD) == board_id
        ]
        for board_id in (
            PRIMARY_SELECTED_BOARD,
            LONG_TAIL_UNCERTAINTY_BOARD,
            PERVERSE_TAIL_DIAGNOSTIC_BOARD,
        )
    }


def _route_records_for_delivery(
    records: list[dict[str, Any]],
    *,
    config: ExemplarDeliveryConfig,
) -> list[dict[str, Any]]:
    for record in records:
        diagnostic = long_tail_diagnostics(
            effective_top_k=max(1, int(record.get("diagnostic_effective_top_k") or 1)),
            top_mass=float(record.get("diagnostic_top_mass") or 0.0),
            vocab_size=config.vocab_size,
            dynamic_mass_threshold=_dynamic_mass_threshold(config),
            dynamic_top_k_max=_dynamic_top_k_max(config),
            policy=_long_tail_policy(config),
        )
        record["selected_board"] = selected_board_for_long_tail(
            str(diagnostic["long_tail_class"]),
            include_long_tail_in_primary=config.include_long_tail_in_primary,
            include_perverse_tail_in_primary=config.include_perverse_tail_in_primary,
        )
    return _cap_curriculum_records(records, config=config)


def _route_materialized_selected_exemplars(
    records: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
    *,
    config: ExemplarDeliveryConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    routed: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for record, payload in zip(records, payloads, strict=True):
        board = selected_board_for_long_tail(
            str(payload["long_tail_class"]),
            include_long_tail_in_primary=config.include_long_tail_in_primary,
            include_perverse_tail_in_primary=config.include_perverse_tail_in_primary,
        )
        record["selected_board"] = board
        payload["selected_board"] = board
        routed.append((record, payload))
    if config.authoritative_selection:
        selected_pairs = routed
    else:
        selected_records = _cap_curriculum_records(
            [record for record, _ in routed],
            config=config,
        )
        selected_ids = {id(record) for record in selected_records}
        selected_pairs = [pair for pair in routed if id(pair[0]) in selected_ids]
        selected_pairs.sort(
            key=lambda pair: (
                -float(pair[0]["selected_score"]),
                str(pair[0]["selected_example_id"]),
                int(pair[0]["selected_position"]),
            )
        )
    for rank, (record, _) in enumerate(selected_pairs, start=1):
        record["rank"] = rank
    return (
        [record for record, _ in selected_pairs],
        [payload for _, payload in selected_pairs],
    )


def _cap_curriculum_records(
    records: list[dict[str, Any]],
    *,
    config: ExemplarDeliveryConfig,
) -> list[dict[str, Any]]:
    grouped = _records_by_selected_board(records)
    primary_limit = _primary_budget(config)
    if config.selected_exemplar_fraction is not None:
        fraction_limit = max(
            1,
            int(
                np.ceil(
                    len(grouped[PRIMARY_SELECTED_BOARD])
                    * config.selected_exemplar_fraction
                )
            ),
        )
        primary_limit = (
            fraction_limit
            if primary_limit is None
            else min(primary_limit, fraction_limit)
        )
    limits = {
        PRIMARY_SELECTED_BOARD: primary_limit,
        LONG_TAIL_UNCERTAINTY_BOARD: config.long_tail_side_board_cap,
        PERVERSE_TAIL_DIAGNOSTIC_BOARD: config.perverse_tail_side_board_cap,
    }
    return [
        record
        for board_id in (
            PRIMARY_SELECTED_BOARD,
            LONG_TAIL_UNCERTAINTY_BOARD,
            PERVERSE_TAIL_DIAGNOSTIC_BOARD,
        )
        for record in grouped[board_id][: limits[board_id]]
    ]


def _attach_long_tail_diagnostics(
    selected_records: list[dict[str, Any]],
    selected_payloads: list[dict[str, Any]],
    *,
    config: ExemplarDeliveryConfig,
    policy: LongTailPolicy,
) -> None:
    if len(selected_records) != len(selected_payloads):
        raise ValueError("selected record/payload count mismatch")
    for record, payload in zip(selected_records, selected_payloads, strict=True):
        diagnostic = long_tail_diagnostics(
            effective_top_k=int(payload["effective_top_k"]),
            top_mass=float(payload["top_mass"]),
            vocab_size=int(payload["vocab_size"]),
            dynamic_mass_threshold=_dynamic_mass_threshold(config),
            dynamic_top_k_max=_dynamic_top_k_max(config),
            policy=policy,
        )
        payload.update(diagnostic)
        payload["dynamic_top_k"].update(
            {
                "dynamic_mass_threshold": diagnostic["dynamic_mass_threshold"],
                "dynamic_top_k_max": diagnostic["dynamic_top_k_max"],
                "top_k_saturated": diagnostic["top_k_saturated"],
            }
        )
        record.update(diagnostic)
        record["dynamic_top_k"] = dict(payload["dynamic_top_k"])
        selected_board = str(record.get("selected_board") or PRIMARY_SELECTED_BOARD)
        record["selected_board"] = selected_board
        payload["selected_board"] = selected_board
        tag = semantic_tail_tag(long_tail_class=str(diagnostic["long_tail_class"]))
        record["semantic_tail_tag"] = tag
        payload["semantic_tail_tag"] = tag


def _payload_slice(payload: Any, key: str, row: int, position: int) -> list[Any]:
    if key not in payload:
        raise ValueError(f"selected backend payload missing {key}")
    raw = payload[key]
    if isinstance(raw, np.ndarray) and raw.dtype == object:
        value = raw[row, position]
    elif isinstance(raw, list) and raw and isinstance(raw[0], list):
        value = raw[row][position]
    else:
        value = np.asarray(raw)[row, position]
    if value.dtype == np.bool_:
        return [bool(item) for item in value.tolist()]
    if np.issubdtype(value.dtype, np.integer):
        return [int(item) for item in value.tolist()]
    return [float(item) for item in value.tolist()]


def _payload_scalar(payload: Any, key: str, row: int, position: int) -> int | float:
    if key not in payload:
        raise ValueError(f"selected backend payload missing {key}")
    value = np.asarray(payload[key])[row, position].item()
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return float(value)


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


__all__ = [name for name in globals() if not name.startswith("__")]
