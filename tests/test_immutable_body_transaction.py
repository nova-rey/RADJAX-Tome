from __future__ import annotations

from pathlib import Path

from radjax_contract.tome.m8g import (
    _m8g_fv3,
    body_raw_digest,
    encode_compact_body,
    manifest_semantic_id,
)

from radjax_tome.builder.delivery.compact_k import compact_body_from_payload
from radjax_tome.builder.delivery.immutable_body import ImmutableBodyTransaction


def _body():
    return compact_body_from_payload(
        {
            "vocab_size": 8,
            "num_buckets": 2,
            "top_selection_mask": [[False, True, True, False]],
            "top_token_ids": [[7, 2, 1, 0]],
            "top_probs": [[0.0, 0.6, 0.2, 0.0]],
            "top_log_probs": [[0.0, -0.5108256, -1.609438, 0.0]],
            "effective_top_k": [2],
            "top_mass": [0.8],
            "tail_mass": [0.2],
            "bucket_masses": [[0.1, 0.1]],
        }
    )


def test_body_then_manifest_are_atomic(tmp_path: Path) -> None:
    body = _body()
    raw = body_raw_digest(encode_compact_body(body))
    manifest = {
        "schema_version": "selected_exemplar_manifest_v1",
        "profile": "producer_evidence",
        "selected_example_id": "example-1",
        "selected_position": 0,
        "source_passport_id": "passport-1",
        "corridor_mode_id": None,
        "corridor_fingerprint_id": None,
        "selection_obligation_count": 0,
        "selection_obligations": [],
        "body_semantic_id": body.semantic_id,
        "body_raw_digest": raw,
        "authority_id": b"a" * 32,
        "selection_authority_id": b"b" * 32,
        "package_role": "producer_evidence",
    }
    manifest["manifest_semantic_id"] = manifest_semantic_id(manifest)
    body_path, manifest_path = ImmutableBodyTransaction(tmp_path).commit(
        body, manifest, canonical_manifest_bytes=_m8g_fv3(manifest)
    )
    assert body_path.is_file()
    assert manifest_path.read_bytes() == _m8g_fv3(manifest)
    recovery = ImmutableBodyTransaction(tmp_path).recover()
    assert recovery and recovery[0]["states"] == list(range(1, 13))
    assert not list((tmp_path / ".transactions").rglob("*.tmp"))
    try:
        ImmutableBodyTransaction(tmp_path).commit(
            body, manifest, canonical_manifest_bytes=b"wrong"
        )
    except ValueError as exc:
        assert "manifest bytes" in str(exc)
    else:
        raise AssertionError("mismatched manifest bytes were accepted")
    manifest["selected_position"] = 1
    manifest["manifest_semantic_id"] = manifest_semantic_id(manifest)
    body_path_2, manifest_path_2 = ImmutableBodyTransaction(tmp_path).commit(
        body, manifest, canonical_manifest_bytes=_m8g_fv3(manifest)
    )
    assert body_path_2 == body_path
    assert manifest_path_2 != manifest_path
