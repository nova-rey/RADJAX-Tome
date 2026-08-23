"""Canonical M8G selected-source materialization modes."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any, Final

from radjax_contract.tome.m8g import CompactBody

LEGACY_PADDED_MONOLITHIC: Final = "legacy_padded_monolithic"
COMPACT_K_MONOLITHIC: Final = "compact_k_monolithic"
COMPACT_K_IMMUTABLE_BODY: Final = "compact_k_immutable_body"
MATERIALIZATION_MODES: Final = (
    LEGACY_PADDED_MONOLITHIC,
    COMPACT_K_MONOLITHIC,
    COMPACT_K_IMMUTABLE_BODY,
)


def validate_materialization_mode(mode: str | None) -> str:
    resolved = LEGACY_PADDED_MONOLITHIC if mode is None else str(mode)
    if resolved not in MATERIALIZATION_MODES:
        raise ValueError(
            f"unsupported selected-source materialization mode: {resolved!r}"
        )
    return resolved


def compact_body_from_logical_payload(
    payload: dict[str, Any], *, profile: str
) -> CompactBody:
    """Build a Contract body directly from already compact logical arrays."""

    ids = tuple(int(value) for value in payload["top_token_ids"])
    probs = tuple(float(value) for value in payload["top_probs"])
    logs = tuple(float(value) for value in payload["top_log_probs"])
    k = len(ids)
    return CompactBody(
        profile=profile,
        vocab_size=int(payload["vocab_size"]),
        num_buckets=int(payload["num_buckets"]),
        top_offsets=(0, k),
        top_lengths=(k,),
        top_token_ids=ids,
        top_probs=probs,
        top_log_probs=logs,
        effective_top_k=(int(payload["effective_top_k"]),),
        top_mass=(float(payload["top_mass"]),),
        tail_mass=(float(payload["tail_mass"]),),
        bucket_masses=tuple(float(value) for value in payload["bucket_masses"]),
    )


def compact_payload_for_storage(payload: dict[str, Any]) -> dict[str, Any]:
    """Return one canonical K-length payload for new persistent storage."""

    compact = dict(payload)
    ids = _json_array_values(compact["top_token_ids"])
    probs = _json_array_values(compact["top_probs"])
    logs = _json_array_values(compact["top_log_probs"])
    k = int(compact["effective_top_k"])
    if k < 0 or k > len(ids) or len(ids) != len(probs) or len(ids) != len(logs):
        raise ValueError("effective_top_k and top arrays are inconsistent")
    mask = compact.pop("top_selection_mask", None)
    if mask is not None:
        active = [index for index, value in enumerate(mask) if bool(value)]
        if len(active) != k:
            raise ValueError(
                "historical selection mask does not describe effective K entries"
            )
        ids = [ids[index] for index in active]
        probs = [probs[index] for index in active]
        logs = [logs[index] for index in active]
    elif len(ids) != k:
        raise ValueError("compact payload contains padding without a selection mask")
    compact["top_token_ids"] = ids[:k]
    compact["top_probs"] = probs[:k]
    compact["top_log_probs"] = logs[:k]
    compact["storage_flavor"] = COMPACT_K_MONOLITHIC
    compact["physical_retained_entry_count"] = k
    compact["logical_k"] = k
    return {key: _json_value(value) for key, value in compact.items()}


def _json_array_values(value: Any) -> list[Any]:
    """Convert one governed contiguous array at the JSON persistence edge."""
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        converted = tolist()
        if isinstance(converted, list):
            return converted
        return [converted]
    return list(value)


def _json_value(value: Any) -> Any:
    """Normalize buffer/native scalar values only at the JSON edge."""
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_value(tolist())
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def collate_compact_logical_records(
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Temporarily pad a compact batch only to its largest effective K."""

    if not records:
        return {
            "width": 0,
            "top_token_ids": [],
            "top_probs": [],
            "top_log_probs": [],
            "mask": [],
        }
    compact = [compact_payload_for_storage(record) for record in records]
    width = max(int(record["effective_top_k"]) for record in compact)
    ids: list[list[int]] = []
    probs: list[list[float]] = []
    logs: list[list[float]] = []
    mask: list[list[bool]] = []
    for record in compact:
        k = int(record["effective_top_k"])
        ids.append(list(record["top_token_ids"]) + [0] * (width - k))
        probs.append(list(record["top_probs"]) + [0.0] * (width - k))
        logs.append(list(record["top_log_probs"]) + [0.0] * (width - k))
        mask.append([True] * k + [False] * (width - k))
    return {
        "width": width,
        "top_token_ids": ids,
        "top_probs": probs,
        "top_log_probs": logs,
        "mask": mask,
    }


def mode_configuration_identity(mode: str) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            ("RDX-M8G-MODE-1\x00" + validate_materialization_mode(mode)).encode()
        ).hexdigest()
    )
