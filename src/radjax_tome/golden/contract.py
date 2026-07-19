from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

GOLDEN_CONTRACT_SCHEMA_VERSION = "radjax_tome.golden_contract.v1"
GOLDEN_CONTRACT_NAME = "t4_gemma3_270m_fingerprint_corridor_path_b_1k"
COLLECTION_NAMES = ("selected_obligations", "source_passports", "payload_semantics")
ACTIVE_PAYLOAD_DIGEST_VERSION = "golden_active_payload_v2"
ACTIVE_PAYLOAD_DIGEST_CHUNK_ENTRIES = 4096
ACTIVE_PAYLOAD_DIGEST_FIELDS = (
    "active_top_token_ids_digest",
    "active_top_probs_digest",
    "active_top_log_probs_digest",
    "active_payload_digest",
)
FORBIDDEN_DENSE_PAYLOAD_FIELDS = frozenset(
    {
        "top_token_ids",
        "top_probs",
        "top_log_probs",
        "top_selection_mask",
        "logits",
        "dense_logits",
        "dense_probabilities",
        "full_vocab_log_probs",
        "full_vocab_probs",
    }
)
_SHA256_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


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
    obligation_indices = {
        _coordinate(row): row["selection_index"]
        for row in collections["selected_obligations"]
    }
    for name in ("source_passports", "payload_semantics"):
        for row in collections[name]:
            if row["selection_index"] != obligation_indices[_coordinate(row)]:
                raise ValueError(f"{name} selection_index does not match obligations")


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
        if name == "payload_semantics":
            validate_sparse_payload_semantics_record(row)


def validate_sparse_payload_semantics_record(row: Mapping[str, Any]) -> None:
    forbidden = sorted(FORBIDDEN_DENSE_PAYLOAD_FIELDS & set(row))
    if forbidden:
        raise ValueError(
            "payload_semantics contains forbidden dense payload fields: "
            + ", ".join(forbidden)
        )
    effective_top_k = row.get("effective_top_k")
    if (
        not isinstance(effective_top_k, int)
        or isinstance(effective_top_k, bool)
        or effective_top_k < 1
    ):
        raise ValueError("payload_semantics effective_top_k must be a positive integer")
    if row.get("payload_digest_version") != ACTIVE_PAYLOAD_DIGEST_VERSION:
        raise ValueError("payload_semantics payload digest version is unsupported")
    for key in ACTIVE_PAYLOAD_DIGEST_FIELDS:
        value = row.get(key)
        if (
            not isinstance(value, str)
            or _SHA256_DIGEST_PATTERN.fullmatch(value) is None
        ):
            raise ValueError(f"payload_semantics {key} is required")
    for key in ("teacher_entropy", "top_mass", "tail_mass", "dynamic_mass_threshold"):
        value = row.get(key)
        if value is not None and (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"payload_semantics {key} must be finite")
    bucket_masses = row.get("bucket_masses")
    if bucket_masses is not None:
        if not isinstance(bucket_masses, (list, tuple)) or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or value < 0.0
            for value in bucket_masses
        ):
            raise ValueError(
                "payload_semantics bucket_masses must be finite probabilities"
            )


def digest_active_payload_storage(row: Mapping[str, Any]) -> dict[str, str]:
    effective_top_k = row.get("effective_top_k")
    arrays = {
        key: row.get(key)
        for key in ("top_token_ids", "top_probs", "top_log_probs", "top_selection_mask")
    }
    if (
        not isinstance(effective_top_k, int)
        or isinstance(effective_top_k, bool)
        or effective_top_k < 1
        or any(not isinstance(value, list) for value in arrays.values())
    ):
        raise ValueError("golden capture payload arrays or effective_top_k are invalid")
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("golden capture payload arrays have inconsistent lengths")
    mask = arrays["top_selection_mask"]
    if any(not isinstance(value, bool) for value in mask):
        raise ValueError("golden capture top_selection_mask must contain booleans")
    active_count = sum(mask)
    if active_count != effective_top_k:
        raise ValueError(
            "golden capture top_selection_mask active count does not equal "
            "effective_top_k"
        )
    token_hasher = _active_payload_hasher("token-ids", active_count)
    probability_hasher = _active_payload_hasher("probabilities", active_count)
    log_probability_hasher = _active_payload_hasher("log-probabilities", active_count)
    seen_token_ids: set[int] = set()
    token_buffer: list[int] = []
    probability_buffer: list[float] = []
    log_probability_buffer: list[float] = []
    for token_id, probability, log_probability, active in zip(
        arrays["top_token_ids"],
        arrays["top_probs"],
        arrays["top_log_probs"],
        mask,
        strict=True,
    ):
        if not active:
            continue
        _validate_active_payload_entry(token_id, probability, log_probability)
        if token_id in seen_token_ids:
            raise ValueError(
                "golden capture payload contains duplicate active token IDs"
            )
        seen_token_ids.add(token_id)
        token_buffer.append(token_id)
        probability_buffer.append(_canonical_float64(probability))
        log_probability_buffer.append(_canonical_float64(log_probability))
        if len(token_buffer) == ACTIVE_PAYLOAD_DIGEST_CHUNK_ENTRIES:
            _flush_active_payload_buffers(
                token_hasher,
                probability_hasher,
                log_probability_hasher,
                token_buffer,
                probability_buffer,
                log_probability_buffer,
            )
            token_buffer.clear()
            probability_buffer.clear()
            log_probability_buffer.clear()
    if token_buffer:
        _flush_active_payload_buffers(
            token_hasher,
            probability_hasher,
            log_probability_hasher,
            token_buffer,
            probability_buffer,
            log_probability_buffer,
        )
    combined_hasher = _active_payload_hasher("combined", active_count)
    for hasher in (token_hasher, probability_hasher, log_probability_hasher):
        combined_hasher.update(hasher.digest())
    return {
        "payload_digest_version": ACTIVE_PAYLOAD_DIGEST_VERSION,
        "active_top_token_ids_digest": "sha256:" + token_hasher.hexdigest(),
        "active_top_probs_digest": "sha256:" + probability_hasher.hexdigest(),
        "active_top_log_probs_digest": "sha256:" + log_probability_hasher.hexdigest(),
        "active_payload_digest": "sha256:" + combined_hasher.hexdigest(),
    }


def _active_payload_hasher(domain: str, active_count: int) -> Any:
    hasher = hashlib.sha256()
    hasher.update(f"radjax-tome:golden-active-{domain}:binary-v2\n".encode())
    hasher.update(struct.pack(">Q", active_count))
    return hasher


def _flush_active_payload_buffers(
    token_hasher: Any,
    probability_hasher: Any,
    log_probability_hasher: Any,
    token_ids: list[int],
    probabilities: list[float],
    log_probabilities: list[float],
) -> None:
    count = len(token_ids)
    token_hasher.update(struct.pack(f">{count}q", *token_ids))
    probability_hasher.update(struct.pack(f">{count}d", *probabilities))
    log_probability_hasher.update(struct.pack(f">{count}d", *log_probabilities))


def _canonical_float64(value: Any) -> float:
    """Normalize both signed-zero encodings to positive zero before hashing."""
    numeric = float(value)
    return 0.0 if numeric == 0.0 else numeric


def _validate_active_payload_entry(
    token_id: Any,
    probability: Any,
    log_probability: Any,
) -> None:
    if (
        not isinstance(token_id, int)
        or isinstance(token_id, bool)
        or token_id < 0
        or token_id > (2**63 - 1)
    ):
        raise ValueError(
            "golden capture active top_token_ids must be nonnegative integers"
        )
    for name, value in (
        ("top_probs", probability),
        ("top_log_probs", log_probability),
    ):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"golden capture active {name} contains nonfinite values")
    if probability < 0.0 or probability > 1.0:
        raise ValueError(
            "golden capture active top_probs contains invalid probabilities"
        )


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
