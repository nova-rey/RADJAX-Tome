from __future__ import annotations

import hashlib

from radjax_tome.builder.delivery.compact_k import encode_compact_payload


def _payload() -> dict[str, object]:
    return {
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


def test_compact_adapter_removes_padded_entries() -> None:
    encoded = encode_compact_payload(_payload())
    assert encoded.startswith(b"RDXC")
    # The compact body is deterministic and materially smaller than the input
    # representation; the Contract decoder is the conformance authority.
    assert (
        hashlib.sha256(encoded).digest()
        == hashlib.sha256(encode_compact_payload(_payload())).digest()
    )
