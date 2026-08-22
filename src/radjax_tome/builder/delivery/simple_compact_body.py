"""Simple compact-K body/metadata storage.

The body is immutable after its first write.  Linkage changes are confined to
the small JSON metadata record, so a consumer never needs to reread or hash a
large numeric payload merely to update provenance.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from radjax_contract.tome.m8g import body_raw_digest, encode_compact_body_packed

from .modes import compact_body_from_logical_payload, compact_payload_for_storage


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def write_compact_body_store(
    root: Path,
    payloads: Iterable[dict[str, Any]],
    *,
    profile: str = "compact_k_monolithic",
) -> dict[str, Any]:
    """Write each compact body once and return an authority-bound inventory."""

    root = Path(root)
    bodies = root / "bodies"
    bodies.mkdir(parents=True, exist_ok=True)
    metadata_path = root / "metadata.jsonl"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata: list[dict[str, Any]] = []
    for payload in payloads:
        compact = compact_payload_for_storage(payload)
        body = compact_body_from_logical_payload(compact, profile=profile)
        encoded = encode_compact_body_packed(body)
        digest = body_raw_digest(encoded).hex()
        body_path = bodies / f"{digest}.body"
        if body_path.exists():
            if body_path.read_bytes() != encoded:
                raise ValueError("compact body digest collision")
        else:
            temporary = body_path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, body_path)
        metadata.append(
            {
                "schema_version": "compact_exemplar_metadata_v1",
                "selected_example_id": str(payload["selected_example_id"]),
                "selected_position": int(payload["selected_position"]),
                "body_semantic_id": body.semantic_id.hex(),
                "body_raw_digest": digest,
                "body_size_bytes": len(encoded),
                "linkage": payload.get("linkage") or payload.get("mode_key"),
            }
        )
    data = b"".join(_json_bytes(item) + b"\n" for item in metadata)
    temporary = metadata_path.with_suffix(".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, metadata_path)
    return {
        "schema_version": "compact_body_store_v1",
        "body_count": len(metadata),
        "metadata_path": str(metadata_path),
        "metadata_sha256": hashlib.sha256(data).hexdigest(),
        "body_digests": [item["body_raw_digest"] for item in metadata],
    }


def update_compact_linkage(root: Path, updates: dict[tuple[str, int], Any]) -> int:
    """Update metadata only; body files are never opened or rewritten."""

    path = Path(root) / "metadata.jsonl"
    rows = [json.loads(line) for line in path.read_bytes().splitlines() if line]
    changed = 0
    for row in rows:
        key = (row["selected_example_id"], int(row["selected_position"]))
        if key in updates:
            row["linkage"] = updates[key]
            changed += 1
    data = b"".join(_json_bytes(row) + b"\n" for row in rows)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)
    return changed
