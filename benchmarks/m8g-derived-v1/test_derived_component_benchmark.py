import json
from pathlib import Path


def test_derived_manifest_and_samples():
    root = Path(__file__).parent
    manifest = json.loads((root / "derived_dataset_manifest.json").read_text())
    assert manifest["dataset"] == "M8G_DERIVED_VALID_EXEMPLAR_COMPONENT_BENCHMARK_V1"
    assert manifest["included_count"] == 169
    assert (
        len(
            {
                (r["coordinate"]["example_id"], r["coordinate"]["position"])
                for r in manifest["records"]
            }
        )
        == 169
    )
    raw = json.loads((root / "raw_nine_sample_report.json").read_text())
    assert len(raw["results"]) == 9
    assert {x["mode"] for x in raw["results"]} == {
        "legacy_padded_monolithic",
        "compact_k_monolithic",
        "compact_k_immutable_body",
    }
    assert all(x["corridor_valid"] for x in raw["results"])


def test_logical_equivalence():
    eq = json.loads((Path(__file__).parent / "logical_equivalence.json").read_text())
    assert eq["legacy_padded_monolithic"]["exact"]
    assert eq["compact_k_monolithic"]["exact"]
    assert eq["compact_k_immutable_body"]["exact_body_identity"]
