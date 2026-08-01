"""Black-box conformance checks for the M7B portable sharding validator."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools" / "validate_radjax_tome_contract_v2.py"
PREFIX = "sha256:"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _digest(value: object) -> str:
    return PREFIX + hashlib.sha256(_canonical(value)).hexdigest()


def _file_digest(path: Path) -> str:
    return PREFIX + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(root: Path, relative: str, value: object) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value))


def _write_jsonl(root: Path, relative: str, values: list[dict[str, object]]) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(_canonical(value) + b"\n" for value in values))


def _payload() -> dict[str, object]:
    return {
        "selected_example_id": "fixture-example",
        "selected_position": 7,
        "selected_score": 1.0,
        "score_selected_position_entropy": 1.0,
        "score_top_token_id": 3,
        "source_shard_id": 0,
        "source_row": 0,
        "source_position": 7,
        "source_score": 1.0,
        "source_top_token_id": 3,
        "source_score_policy": "fixture",
        "payload_ref": {"fixture": "payload"},
        "selected_policy": "fixture",
        "source_delivery_path": "fixture",
        "top_token_ids": [3],
        "top_log_probs": [-0.1],
        "top_probs": [0.9],
        "top_selection_mask": [True],
        "effective_top_k": 1,
        "top_mass": 0.9,
        "tail_mass": 0.1,
        "bucket_masses": [0.1],
        "teacher_entropy": 1.0,
        "sequence_length": 8,
        "vocab_size": 16,
        "num_buckets": 1,
        "dynamic_top_k": False,
        "dynamic_mass_threshold": 0.9,
        "dynamic_top_k_max": 1,
        "top_k_saturated": False,
        "long_tail_class": "fixture",
        "long_tail_warnings": [],
        "effective_top_k_fraction_of_vocab": 0.0625,
        "semantic_tail_tag": "fixture",
        "selected_board": "fixture",
        "corridor_mode_id": 0,
        "corridor_fingerprint_id": 0,
        "corridor_assignment_status": "fixture",
    }


Mutator = Callable[[dict[str, Any]], None]


def _package(root: Path, mutate: Mutator | None = None) -> Path:
    """Write a complete one-record v4 package, then inventory it last."""
    payload = _payload()
    payload_semantic_digest = _digest(payload)
    logical_id = _digest(
        {
            "selected_example_id": payload["selected_example_id"],
            "selected_position": payload["selected_position"],
        }
    )
    shard_path = "selected_exemplars/shards/fixture-00000.jsonl"
    index_path = "selected_exemplars/payload-index.jsonl"
    shard_index_path = "selected_exemplars/payload-shards.jsonl"
    layout_path = "selected_exemplars/payload-layout.json"
    _write_jsonl(root, shard_path, [payload])
    shard_sha256 = _file_digest(root / shard_path)
    sequence = {
        "schema_version": "selected_exemplar_payload_sequence_v1",
        "records": [
            {
                "logical_id": logical_id,
                "payload_semantic_digest": payload_semantic_digest,
            }
        ],
    }
    index = {
        "logical_id": logical_id,
        "selected_example_id": payload["selected_example_id"],
        "selected_position": payload["selected_position"],
        "selection_index": 0,
        "shard_id": 0,
        "row": 0,
        "payload_sha256": _digest(payload),
        "payload_semantic_digest": payload_semantic_digest,
        "shard_sha256": shard_sha256,
    }
    _write_jsonl(root, index_path, [index])
    shard_entry = {
        "shard_id": 0,
        "path": shard_path,
        "sha256": shard_sha256,
        "size_bytes": (root / shard_path).stat().st_size,
        "first_selection_index": 0,
        "last_selection_index": 0,
        "record_count": 1,
        "semantic_digest": _digest(sequence),
    }
    _write_jsonl(root, shard_index_path, [shard_entry])
    layout = {
        "schema_version": "radjax_tome_payload_layout_v1",
        "layout_version": "selected_payload_shards_v1",
        "payload_index": {
            "path": index_path,
            "sha256": _file_digest(root / index_path),
            "size_bytes": (root / index_path).stat().st_size,
            "record_count": 1,
            "schema_version": "radjax_tome_payload_index_v2",
        },
        "shard_index": {
            "path": shard_index_path,
            "sha256": _file_digest(root / shard_index_path),
            "size_bytes": (root / shard_index_path).stat().st_size,
            "record_count": 1,
            "schema_version": "radjax_tome_payload_shard_index_v1",
        },
        "sequence_digest": _digest(sequence),
        "selected_count": 1,
        "payload_records_per_shard": 1,
    }
    identity = {
        "schema_version": "radjax_tome_semantic_identity_v2",
        "payload_sequence_digest": _digest(sequence),
        "selected_count": 1,
        "nonselected_training_payload": [],
        "training_contract": {"target_type": "fixture"},
        "authority": {"selection": "fixture"},
    }
    identity["semantic_digest"] = _digest(identity)
    package: dict[str, Any] = {
        "payload": payload,
        "index": index,
        "layout": layout,
        "shard_entry": shard_entry,
        "identity": identity,
    }
    if mutate is not None:
        mutate(package)
    _write_jsonl(root, shard_path, [package["payload"]])
    shard_sha256 = _file_digest(root / shard_path)
    package["shard_entry"]["sha256"] = shard_sha256
    package["shard_entry"]["size_bytes"] = (root / shard_path).stat().st_size
    _write_jsonl(root, shard_index_path, [package["shard_entry"]])
    package["layout"]["shard_index"]["sha256"] = _file_digest(root / shard_index_path)
    package["layout"]["shard_index"]["size_bytes"] = (
        (root / shard_index_path).stat().st_size
    )
    package["index"]["shard_sha256"] = shard_sha256
    package["index"]["payload_sha256"] = _digest(package["payload"])
    _write_jsonl(root, index_path, [package["index"]])
    package["layout"]["payload_index"]["sha256"] = _file_digest(root / index_path)
    package["layout"]["payload_index"]["size_bytes"] = (
        (root / index_path).stat().st_size
    )
    _write_json(root, layout_path, package["layout"])

    inventory_path = "manifests/content-manifest-inventory.jsonl"
    header_path = "manifests/content-manifest-header.json"
    members = [layout_path, index_path, shard_index_path, shard_path]
    inventory = [
        {
            "path": relative,
            "sha256": _file_digest(root / relative),
            "size_bytes": (root / relative).stat().st_size,
            "classification": "training_critical",
            "training_authoritative": True,
        }
        for relative in sorted(members)
    ]
    _write_jsonl(root, inventory_path, inventory)
    header = {
        "schema_version": "tome_content_manifest_header_v3",
        "profile": "unpacked",
        "semantic_identity_digest": package["identity"]["semantic_digest"],
        "inventory_path": inventory_path,
        "inventory_sha256": _file_digest(root / inventory_path),
        "inventory_size_bytes": (root / inventory_path).stat().st_size,
        "entry_count": len(inventory),
    }
    _write_json(root, header_path, header)
    cover = {
        "schema_version": "radjax_tome_cover_v4",
        "identity": package["identity"],
        "training": package["identity"]["training_contract"],
        "package": {"profile": "unpacked", "transport": "directory"},
        "manifests": {
            "header": {
                "path": header_path,
                "sha256": _file_digest(root / header_path),
                "size_bytes": (root / header_path).stat().st_size,
                "schema_version": "tome_content_manifest_header_v3",
            }
        },
        "authority": package["identity"]["authority"],
        "provenance": {},
        "validation": {},
    }
    _write_json(root, "cover_page.json", cover)
    return root


def _validate(path: Path) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, str(VALIDATOR), str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    return {"returncode": result.returncode, **json.loads(result.stdout)}


def test_m7b_portable_validator_accepts_minimal_v4_sharded_package(
    tmp_path: Path,
) -> None:
    assert _validate(_package(tmp_path)) == {
        "returncode": 0,
        "errors": [],
        "ok": True,
        "warnings": [],
    }


def test_m7b_portable_validator_rejects_layout_count_mismatch(tmp_path: Path) -> None:
    package = _package(
        tmp_path,
        lambda package: package["layout"].update(selected_count=2),
    )

    assert _validate(package)["errors"] == ["manifest_record_count_mismatch"]


def test_m7b_portable_validator_rejects_index_address_mismatch(tmp_path: Path) -> None:
    package = _package(tmp_path, lambda package: package["index"].update(row=1))

    assert _validate(package)["errors"] == ["payload_index_address_invalid"]


def test_m7b_portable_validator_rejects_stale_shard_sequence_digest(
    tmp_path: Path,
) -> None:
    package = _package(
        tmp_path,
        lambda package: package["shard_entry"].update(
            semantic_digest=PREFIX + "0" * 64
        ),
    )

    assert _validate(package)["errors"] == ["payload_sequence_digest_mismatch"]


def test_m7b_portable_validator_rejects_tampered_shard_payload(tmp_path: Path) -> None:
    def mutate(package: dict[str, Any]) -> None:
        package["payload"]["selected_score"] = 2.0

    assert _validate(_package(tmp_path, mutate))["errors"] == [
        "payload_semantic_projection_invalid"
    ]


def test_m7b_portable_validator_rejects_identity_payload_projection_mismatch(
    tmp_path: Path,
) -> None:
    def mutate(package: dict[str, Any]) -> None:
        package["identity"]["payload_sequence_digest"] = PREFIX + "0" * 64
        semantic_identity = package["identity"]
        semantic_identity["semantic_digest"] = _digest(
            {
                key: value
                for key, value in semantic_identity.items()
                if key != "semantic_digest"
            }
        )

    assert _validate(_package(tmp_path, mutate))["errors"] == [
        "payload_semantic_projection_invalid"
    ]
