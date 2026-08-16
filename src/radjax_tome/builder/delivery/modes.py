"""Canonical M8G selected-source materialization modes."""

from __future__ import annotations

import hashlib
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
    """Return the compact physical payload, without a padded selection mask."""

    compact = dict(payload)
    compact.pop("top_selection_mask", None)
    compact["storage_flavor"] = COMPACT_K_MONOLITHIC
    compact["physical_retained_entry_count"] = len(compact["top_token_ids"])
    compact["logical_k"] = int(compact["effective_top_k"])
    return compact


def mode_configuration_identity(mode: str) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            ("RDX-M8G-MODE-1\x00" + validate_materialization_mode(mode)).encode()
        ).hexdigest()
    )
