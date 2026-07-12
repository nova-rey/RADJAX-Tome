from __future__ import annotations

import json
from pathlib import Path

import pytest

from radjax_tome.golden.compare import compare_contracts
from radjax_tome.golden.contract import build_contract
from radjax_tome.golden.projection import capture_golden_contract
from radjax_tome.tome.golden_fixture import build_production_contract_fixture


def test_compare_allows_one_entropy_quantization_step_but_not_token_drift(
    tmp_path: Path,
) -> None:
    expected = _write_fixture(tmp_path / "expected")
    observed = _write_fixture(tmp_path / "observed", entropy=1.00390625)

    assert compare_contracts(expected, observed)["status"] == "pass"

    payload = observed / "payload_semantics.jsonl"
    row = json.loads(payload.read_text(encoding="utf-8"))
    row["top_token_ids"] = [999]
    payload.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert compare_contracts(expected, observed)["status"] == "fail"


def test_compare_reports_coordinate_and_selection_order_drift(tmp_path: Path) -> None:
    expected = _write_fixture(tmp_path / "expected")
    observed = _write_fixture(tmp_path / "observed", position=8)

    report = compare_contracts(expected, observed)
    assert report["status"] == "fail"
    assert report["differences"][0]["field"] == "coordinate"


def test_capture_refuses_nonterminal_artifact_without_modifying_it(
    tmp_path: Path,
) -> None:
    artifact = build_production_contract_fixture(tmp_path / "artifact")
    output = tmp_path / "capture"

    with pytest.raises(ValueError, match="production_build_report"):
        capture_golden_contract(artifact, output)

    assert not output.exists()


def _write_fixture(root: Path, *, entropy: float = 1.0, position: int = 3) -> Path:
    obligations = [
        {
            "selection_index": 1,
            "selected_example_id": "one",
            "selected_position": position,
            "primary_role": "corridor",
        }
    ]
    passports = [
        {
            "selection_index": 1,
            "selected_example_id": "one",
            "selected_position": position,
            "source_row": 7,
        }
    ]
    payloads = [
        {
            "selection_index": 1,
            "selected_example_id": "one",
            "selected_position": position,
            "teacher_entropy": entropy,
            "top_token_ids": [7],
        }
    ]
    contract = build_contract(
        fixture_metadata={},
        input_identity={},
        semantic_policy={},
        stage_summary=[],
        selected_obligations=obligations,
        source_passports=passports,
        payload_semantics=payloads,
        board_summary={},
    )
    root.mkdir()
    (root / "contract.json").write_text(json.dumps(contract), encoding="utf-8")
    for name, rows in (
        ("selected_obligations", obligations),
        ("source_passports", passports),
        ("payload_semantics", payloads),
    ):
        (root / f"{name}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    return root
