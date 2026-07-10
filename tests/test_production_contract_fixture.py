from __future__ import annotations

import json
from pathlib import Path

from radjax_tome.builder import validate_teacher_textbook
from radjax_tome.tome import validate_tome_cover_page
from radjax_tome.tome.golden_fixture import (
    artifact_tree_digest,
    build_production_contract_fixture,
)


def test_production_contract_fixture_is_deterministic_and_valid(
    tmp_path: Path,
) -> None:
    left = build_production_contract_fixture(tmp_path / "left")
    right = build_production_contract_fixture(tmp_path / "right")

    assert artifact_tree_digest(left) == artifact_tree_digest(right)
    assert validate_teacher_textbook(left).status == "pass"
    assert validate_tome_cover_page(left).status == "pass"
    cover = json.loads((left / "cover_page.json").read_text(encoding="utf-8"))
    assert cover["cover_page_version"] == 2
    assert [item["surface_id"] for item in cover["behavioral_surfaces"]] == [
        "corridor",
        "exemplar",
    ]
    assert [
        item["surface_id"] for item in cover["recommended_training_plan"]["passes"]
    ] == ["corridor", "exemplar"]
    roles = {item["role"] for item in cover["contents"]}
    assert "corridor_assignment_mode_id" in roles
    assert "selected_exemplar_payload_shard" in roles


def test_production_contract_fixture_varies_effective_top_k(tmp_path: Path) -> None:
    artifact = build_production_contract_fixture(tmp_path / "fixture")
    payload_path = artifact / "selected_exemplars" / "selected-exemplars-00000.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))

    assert [item["effective_top_k"] for item in payload["selected_exemplars"]] == [
        2,
        3,
        4,
        5,
    ]
