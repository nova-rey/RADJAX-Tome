from __future__ import annotations

from pathlib import Path

from radjax_tome.tome import contracts
from radjax_tome.tome.bundle import SUPPORTED_COMPRESSION
from radjax_tome.tome.compatibility import LEGACY_COVER_PAGE_V2
from radjax_tome.tome.packaging import PACKAGE_COVER_SCHEMA

ROOT = Path(__file__).resolve().parents[1]


def test_m6a_inventory_pins_current_public_contract_versions() -> None:
    assert contracts.CANONICAL_TOME_COVER_SCHEMA == "radjax_tome_cover_v3"
    assert contracts.TOME_SEMANTIC_IDENTITY_SCHEMA == "radjax_tome_semantic_identity_v1"
    assert contracts.CANONICAL_CONTENT_MANIFEST_SCHEMA == "tome_content_manifest_v2"
    assert PACKAGE_COVER_SCHEMA == "radjax_tome_package_cover_v1"
    assert LEGACY_COVER_PAGE_V2 == "cover_page_v2"


def test_m6a_inventory_pins_portable_profile_and_transport_vocabulary() -> None:
    assert contracts._PACKAGE_PROFILES == {
        "unpacked",
        "student",
        "full_debug_provenance",
    }
    assert contracts._CONTENT_CLASSIFICATIONS == {
        "training_critical",
        "integrity_or_provenance",
        "diagnostic",
        "human_readable",
        "operational",
    }
    assert SUPPORTED_COMPRESSION == {"none", "gz"}


def test_m6a_inventory_records_v3_and_validator_ownership() -> None:
    inventory = (ROOT / "docs" / "M6_PUBLICATION_INVENTORY.md").read_text(
        encoding="utf-8"
    )
    assert "`radjax_tome_cover_v3`" in inventory
    assert "historical v2 cover" in inventory
    assert "producer determinism is separate" in inventory
    assert "only portable validator" in inventory
