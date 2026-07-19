from __future__ import annotations

import pytest

from radjax_tome.golden.contract import (
    GOLDEN_CONTRACT_SCHEMA_VERSION,
    build_contract,
    canonical_json_bytes,
    validate_contract,
)


def test_canonical_json_digest_ignores_mapping_order_and_capture_metadata() -> None:
    rows = _rows()
    left = _contract(rows, capture_metadata={"captured_at": "one", "path": "/rental/a"})
    right = _contract(
        rows, capture_metadata={"captured_at": "two", "path": "/rental/b"}
    )

    assert canonical_json_bytes({"b": 2, "a": 1}) == canonical_json_bytes(
        {"a": 1, "b": 2}
    )
    assert left["semantic_root"] == right["semantic_root"]
    assert left["schema_version"] == GOLDEN_CONTRACT_SCHEMA_VERSION


def test_contract_rejects_duplicate_coordinates_and_unknown_schema() -> None:
    rows = _rows()
    contract = _contract(rows)
    with pytest.raises(ValueError, match="duplicate"):
        validate_contract(
            contract, collections={name: rows[name] + [rows[name][0]] for name in rows}
        )
    contract["schema_version"] = "unknown"
    with pytest.raises(ValueError, match="unsupported"):
        validate_contract(contract, collections=rows)


def test_contract_rejects_selection_index_mismatch() -> None:
    rows = _rows()
    rows["payload_semantics"][0]["selection_index"] = 9
    with pytest.raises(ValueError, match="selection_index"):
        _contract(rows)


def _contract(
    rows: dict[str, list[dict[str, object]]], **kwargs: object
) -> dict[str, object]:
    return build_contract(
        fixture_metadata={"profile": "full_debug_provenance"},
        input_identity={"vocab_size": 262144},
        semantic_policy={"delivery_path": "two_pass_rerun_selected"},
        stage_summary=[{"stage": "delivery", "status": "pass"}],
        selected_obligations=rows["selected_obligations"],
        source_passports=rows["source_passports"],
        payload_semantics=rows["payload_semantics"],
        board_summary={"mode_count": 47},
        **kwargs,
    )


def _rows() -> dict[str, list[dict[str, object]]]:
    base = [
        {"selection_index": 1, "selected_example_id": "one", "selected_position": 3},
        {"selection_index": 2, "selected_example_id": "two", "selected_position": 4},
    ]
    rows = {
        name: [dict(row) for row in base]
        for name in ("selected_obligations", "source_passports", "payload_semantics")
    }
    for row in rows["payload_semantics"]:
        row.update(
            {
                "effective_top_k": 1,
                "top_token_ids": [7],
                "top_probs": [0.5],
                "top_log_probs": [-0.6931471805599453],
            }
        )
    return rows
