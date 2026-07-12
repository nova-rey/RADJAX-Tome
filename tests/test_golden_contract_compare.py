from __future__ import annotations

import json
from pathlib import Path

import pytest

from radjax_tome.golden.compare import compare_contracts
from radjax_tome.golden.contract import build_contract
from radjax_tome.golden.projection import _payload_index, capture_golden_contract
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


def test_payload_index_uses_native_selected_exemplars_field(tmp_path: Path) -> None:
    selected = tmp_path / "selected_exemplars"
    selected.mkdir()
    (selected / "payload_index.json").write_text(
        json.dumps(
            {
                "schema_version": "selected_exemplar_payload_index_v1",
                "selected_exemplars": [
                    {"selected_example_id": "one", "selected_position": 3}
                ],
            }
        ),
        encoding="utf-8",
    )

    assert _payload_index(tmp_path) == {
        ("one", 3): {"selected_example_id": "one", "selected_position": 3}
    }


def test_compare_rejects_c5_role_and_passport_drift(tmp_path: Path) -> None:
    expected = _write_fixture(tmp_path / "expected")
    observed = _write_fixture(tmp_path / "observed")

    obligations = observed / "selected_obligations.jsonl"
    row = json.loads(obligations.read_text(encoding="utf-8"))
    row["primary_claim"] = "global"
    obligations.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert compare_contracts(expected, observed)["status"] == "fail"


@pytest.mark.parametrize(
    ("collection", "field", "value"),
    [
        ("selected_obligations", "represented_fingerprint_corridor_ids", [99]),
        ("selected_obligations", "selection_obligations", []),
        ("selected_obligations", "selection_index", 8),
        ("payload_semantics", "effective_top_k", 99),
        ("payload_semantics", "bucket_masses", [0.9]),
        ("payload_semantics", "semantic_authority_hash", "sha256:changed"),
    ],
)
def test_compare_rejects_semantic_field_drift(
    tmp_path: Path, collection: str, field: str, value: object
) -> None:
    expected = _write_fixture(tmp_path / "expected")
    observed = _write_fixture(tmp_path / "observed")
    path = observed / f"{collection}.jsonl"
    row = json.loads(path.read_text(encoding="utf-8"))
    row[field] = value
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert compare_contracts(expected, observed)["status"] == "fail"

    observed = _write_fixture(tmp_path / "passport", position=3)
    passports = observed / "source_passports.jsonl"
    row = json.loads(passports.read_text(encoding="utf-8"))
    row["source_row"] = 99
    passports.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert compare_contracts(expected, observed)["status"] == "fail"


def _write_fixture(root: Path, *, entropy: float = 1.0, position: int = 3) -> Path:
    obligations = [
        {
            "selection_index": 1,
            "selected_example_id": "one",
            "selected_position": position,
            "primary_role": "corridor",
            "primary_claim": "corridor",
            "selection_roles": ["corridor"],
            "selection_obligations": [{"role": "corridor", "rank": 1}],
            "represented_fingerprint_corridor_ids": [7],
            "global_board_ids": ["entropy"],
            "source_passport": {"source_row": 7},
            "payload_identity": {"payload_key": "one:3"},
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
