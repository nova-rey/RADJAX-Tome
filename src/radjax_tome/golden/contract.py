from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

GOLDEN_CONTRACT_SCHEMA_VERSION = "radjax_tome.golden_contract.v1"
GOLDEN_CONTRACT_NAME = "t4_gemma3_270m_fingerprint_corridor_path_b_1k"
COLLECTION_NAMES = ("selected_obligations", "source_passports", "payload_semantics")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _normalize(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def semantic_digest(domain: str, value: Any) -> str:
    prefix = f"radjax-tome:{domain}:v1\n".encode()
    return "sha256:" + hashlib.sha256(prefix + canonical_json_bytes(value)).hexdigest()


def ordered_collection_root(domain: str, rows: Sequence[Mapping[str, Any]]) -> str:
    digests = [semantic_digest(f"{domain}-row", row) for row in rows]
    return semantic_digest(f"{domain}-root", digests)


def build_contract(
    *,
    fixture_metadata: Mapping[str, Any],
    input_identity: Mapping[str, Any],
    semantic_policy: Mapping[str, Any],
    stage_summary: Sequence[Mapping[str, Any]],
    selected_obligations: Sequence[Mapping[str, Any]],
    source_passports: Sequence[Mapping[str, Any]],
    payload_semantics: Sequence[Mapping[str, Any]],
    board_summary: Mapping[str, Any],
    capture_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    collections = {
        "selected_obligations": list(selected_obligations),
        "source_passports": list(source_passports),
        "payload_semantics": list(payload_semantics),
    }
    roots = {
        name: ordered_collection_root(name, rows) for name, rows in collections.items()
    }
    contract = {
        "schema_version": GOLDEN_CONTRACT_SCHEMA_VERSION,
        "fixture_name": GOLDEN_CONTRACT_NAME,
        "fixture_metadata": dict(fixture_metadata),
        "input_identity": dict(input_identity),
        "semantic_policy": dict(semantic_policy),
        "stage_summary": list(stage_summary),
        "collection_roots": roots,
        "board_summary_digest": semantic_digest("board-summary", board_summary),
        "capture_metadata": dict(capture_metadata or {}),
    }
    contract["semantic_root"] = semantic_digest(
        "golden-contract",
        {
            key: value
            for key, value in contract.items()
            if key not in {"capture_metadata", "semantic_root"}
        },
    )
    validate_contract(contract, collections=collections)
    return contract


def validate_contract(
    contract: Mapping[str, Any],
    *,
    collections: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> None:
    if contract.get("schema_version") != GOLDEN_CONTRACT_SCHEMA_VERSION:
        raise ValueError("unsupported golden contract schema version")
    if contract.get("fixture_name") != GOLDEN_CONTRACT_NAME:
        raise ValueError("unexpected golden contract fixture name")
    roots = contract.get("collection_roots")
    if not isinstance(roots, Mapping) or set(roots) != set(COLLECTION_NAMES):
        raise ValueError("golden contract collection roots are incomplete")
    if not isinstance(contract.get("semantic_root"), str):
        raise ValueError("golden contract semantic_root is missing")
    expected_root = semantic_digest(
        "golden-contract",
        {
            key: value
            for key, value in contract.items()
            if key not in {"capture_metadata", "semantic_root"}
        },
    )
    if contract["semantic_root"] != expected_root:
        raise ValueError("golden contract semantic_root does not match content")
    if collections is None:
        return
    for name in COLLECTION_NAMES:
        rows = collections.get(name)
        if rows is None:
            raise ValueError(f"golden contract collection missing: {name}")
        _validate_collection(name, rows)
        observed = ordered_collection_root(name, rows)
        if observed != roots[name]:
            raise ValueError(f"golden contract {name} root does not match rows")
    coordinates = [_coordinate(row) for row in collections["selected_obligations"]]
    if len(coordinates) != len(set(coordinates)):
        raise ValueError("golden contract contains duplicate selected coordinates")
    passport_coordinates = {_coordinate(row) for row in collections["source_passports"]}
    payload_coordinates = {_coordinate(row) for row in collections["payload_semantics"]}
    selected_coordinates = set(coordinates)
    if selected_coordinates != passport_coordinates:
        raise ValueError("selected obligations and source passports do not join")
    if selected_coordinates != payload_coordinates:
        raise ValueError("selected obligations and payload semantics do not join")


def _validate_collection(name: str, rows: Sequence[Mapping[str, Any]]) -> None:
    previous_index = -1
    coordinates: set[tuple[str, int]] = set()
    for row in rows:
        coordinate = _coordinate(row)
        if coordinate in coordinates:
            raise ValueError(f"{name} contains duplicate selected coordinates")
        coordinates.add(coordinate)
        index = row.get("selection_index")
        if not isinstance(index, int) or index < 0 or index <= previous_index:
            raise ValueError(f"{name} selection_index is not strictly ordered")
        previous_index = index


def _coordinate(row: Mapping[str, Any]) -> tuple[str, int]:
    example_id = row.get("selected_example_id")
    position = row.get("selected_position")
    if not isinstance(example_id, str) or not isinstance(position, int):
        raise ValueError("golden contract row has invalid coordinate identity")
    return example_id, position


def _normalize(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Mapping):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("semantic digests require finite numeric values")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"unsupported semantic value: {type(value).__name__}")
