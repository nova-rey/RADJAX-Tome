"""M7C catalog and historical-boundary conformance checks."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "contracts" / "radjax_tome" / "v2" / "fixtures" / "catalog.json"


def test_m7c_catalog_covers_the_black_box_streaming_validator_cases() -> None:
    catalog = json.loads(CATALOG.read_text())
    cases = {case["id"]: case["expected"] for case in catalog["cases"]}
    assert {
        "minimal-v4-sharded-directory": "pass",
        "payload-layout-count-mismatch": "manifest_record_count_mismatch",
        "index-address-mismatch": "payload_index_address_invalid",
        "stale-shard-sequence-digest": "payload_sequence_digest_mismatch",
        "refreshed-envelope-stale-payload-projection": (
            "payload_semantic_projection_invalid"
        ),
        "rehashed-stale-identity-projection": "payload_semantic_projection_invalid",
    }.items() <= cases.items()


def test_m7c_catalog_preserves_historical_nonpromotion() -> None:
    catalog = json.loads(CATALOG.read_text())
    cases = {case["id"]: case["expected"] for case in catalog["cases"]}
    assert cases["historical-v3"] == "native-historical-only"
    assert cases["historical-cover-v2"] == "incomplete_descriptor"
    assert cases["historical-package-v1"] == "incomplete_descriptor"
