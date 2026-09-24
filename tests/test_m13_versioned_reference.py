"""Owner-amended M13 reference separation and mutation checks."""
from __future__ import annotations
import json
from pathlib import Path
import pytest
from radjax_tome.golden.projection import validate_fixture
ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "evidence/m13_current_production_1k_v1"
HISTORICAL = ROOT / "tests/fixtures/golden_t4_1k"

def test_current_reference_is_valid_and_distinct_from_historical() -> None:
    current = validate_fixture(CURRENT)
    historical = json.loads((HISTORICAL / "contract.json").read_text())
    captured = json.loads((CURRENT / "contract.json").read_text())
    assert current["status"] == "pass"
    assert captured["schema_version"] == "radjax_tome.golden_contract.v2"
    assert captured["semantic_root"] != historical["semantic_root"]
    assert captured["semantic_root"] == "sha256:5e65c5388a7079f143421b5a35a72a586083cf47a3e37e82cac13eff4f608fdb"

def test_current_reference_governed_mutation_is_rejected(tmp_path: Path) -> None:
    mutated = tmp_path / "current"
    mutated.mkdir()
    for path in CURRENT.iterdir():
        (mutated / path.name).write_bytes(path.read_bytes())
    rows = mutated / "selected_obligations.jsonl"
    first, *rest = rows.read_text().splitlines()
    row = json.loads(first)
    row["selected_position"] = int(row["selected_position"]) + 1
    rows.write_text("\n".join([json.dumps(row, sort_keys=True), *rest]) + "\n")
    with pytest.raises(ValueError, match="root does not match rows"):
        validate_fixture(mutated)
