from __future__ import annotations

import json
from pathlib import Path

from radjax_contract.tome import validate_student_tome_consumption

from radjax_tome.tome.golden_fixture import build_production_contract_fixture


def test_current_tome_fixture_uses_stable_roles_and_passes_contract(
    tmp_path: Path,
) -> None:
    artifact = build_production_contract_fixture(tmp_path / "fixture")
    cover = json.loads((artifact / "cover_page.json").read_text(encoding="utf-8"))
    logical_ids = {item["logical_id"] for item in cover["identity"]["training_payload"]}
    assert "metadata.json" in logical_ids
    assert "corridors/mode_assignments/weight.npy" in logical_ids
    assert "shards/shard-00000.npz" in logical_ids
    assert "selected_exemplars/selected-exemplars-00000.json" in logical_ids
    result = validate_student_tome_consumption(artifact)
    assert result.ok
    assert result.semantic_digest == (
        "sha256:c7eb093e3481504197018209e94eca41a5b31efc16588d54cc6b453ac1e91d72"
    )


def test_offline_consumption_assets_match_contract_publication() -> None:
    tome_root = Path(__file__).parents[1] / "contracts/radjax_tome/consumption/v1"
    contract_root = (
        Path(__file__).parents[2]
        / "RADJAX-Contract/src/radjax_contract/contracts/radjax_tome/consumption/v1"
    )
    tome_files = sorted(
        path.relative_to(tome_root) for path in tome_root.rglob("*") if path.is_file()
    )
    contract_files = sorted(
        path.relative_to(contract_root)
        for path in contract_root.rglob("*")
        if path.is_file()
    )
    assert tome_files == contract_files
    for relative in tome_files:
        assert (tome_root / relative).read_bytes() == (
            contract_root / relative
        ).read_bytes()
