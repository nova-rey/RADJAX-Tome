from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from radjax_tome.golden.contract import FORBIDDEN_DENSE_PAYLOAD_FIELDS
from radjax_tome.golden.projection import (
    _assert_portable_fixture_value,
    validate_fixture,
)

FIXTURE = Path(__file__).parent / "fixtures" / "golden_t4_1k"
EXPECTED_SEMANTIC_ROOT = (
    "sha256:4dcc4baa6bfc1c065d2f45268289db504a511891b875c40315c5748825e261ba"
)
EXPECTED_PIPELINE = "native_two_pass_fingerprint_corridor_path_b"
FIXTURE_SURFACES = (
    "contract.json",
    "board_summary.json",
    "payload_semantics.jsonl",
    "selected_obligations.jsonl",
    "source_passports.jsonl",
)


def test_canonical_t4_1k_fixture_is_valid_and_self_contained() -> None:
    result = validate_fixture(FIXTURE)
    contract = _read_json(FIXTURE / "contract.json")

    assert result == {
        "status": "pass",
        "semantic_root": EXPECTED_SEMANTIC_ROOT,
        "count": 256,
    }
    assert contract["semantic_root"] == EXPECTED_SEMANTIC_ROOT
    assert contract["fixture_metadata"]["canonical_pipeline"] == EXPECTED_PIPELINE

    payload_rows = list(_read_jsonl(FIXTURE / "payload_semantics.jsonl"))
    assert len(payload_rows) == 256
    for row in payload_rows:
        assert not (FORBIDDEN_DENSE_PAYLOAD_FIELDS & row.keys())


def test_canonical_t4_1k_fixture_surfaces_are_portable() -> None:
    for name in FIXTURE_SURFACES:
        for value in _fixture_surface_values(FIXTURE / name):
            _assert_portable_fixture_value(value, context=name)


def _fixture_surface_values(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".json":
        return [_read_json(path)]
    return list(_read_jsonl(path))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        assert isinstance(value, dict)
        rows.append(value)
    return rows
