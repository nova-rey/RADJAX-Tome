"""Opt-in compact physical selected-exemplar payload adapter.

The legacy padded backend representation remains the default.  This module
converts one already-materialized padded payload to the reviewed M8G Contract
body resource without changing logical K, ordering, mass, or CSL evidence.
"""

from __future__ import annotations

from typing import Any

from radjax_contract.tome.m8g import (
    CompactBody,
    compact_from_padded,
    encode_compact_body,
)

COMPACT_STORAGE_FLAVOR = "selected_exemplar_body_v1"


def compact_body_from_payload(
    payload: dict[str, Any], *, profile: str = "producer_evidence"
) -> CompactBody:
    """Build a Contract compact body from one padded selected payload."""

    return compact_from_padded(payload, profile=profile)


def encode_compact_payload(
    payload: dict[str, Any], *, profile: str = "producer_evidence"
) -> bytes:
    """Encode one payload using the explicit opt-in compact storage flavor."""

    return encode_compact_body(compact_body_from_payload(payload, profile=profile))


__all__ = [
    "COMPACT_STORAGE_FLAVOR",
    "compact_body_from_payload",
    "encode_compact_payload",
]
