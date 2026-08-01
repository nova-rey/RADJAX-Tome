from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts" / "radjax_tome" / "v2"
VALIDATOR = ROOT / "tools" / "validate_radjax_tome_contract_v2.py"


def test_m7b_schemas_are_valid_draft_2020_12() -> None:
    for path in sorted((CONTRACT / "schemas").glob("*.json")):
        jsonschema.Draft202012Validator.check_schema(json.loads(path.read_text()))


def test_m7b_contract_checksum_inventory_is_complete() -> None:
    expected = {}
    for line in (CONTRACT / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", maxsplit=1)
        expected[relative] = digest
    observed = {
        path.relative_to(CONTRACT).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in CONTRACT.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    assert expected == observed


def test_m7b_contract_declares_the_acyclic_streaming_manifest_graph() -> None:
    contract = json.loads((CONTRACT / "contract.json").read_text())
    assert contract["supported_wire_schemas"] == {
        "cover": "radjax_tome_cover_v4",
        "content_manifest_header": "tome_content_manifest_header_v3",
        "content_manifest_inventory": "tome_content_manifest_inventory_v3",
        "payload_layout": "radjax_tome_payload_layout_v1",
        "payload_index": "radjax_tome_payload_index_v2",
        "payload_shard_index": "radjax_tome_payload_shard_index_v1",
        "semantic_identity": "radjax_tome_semantic_identity_v2",
    }
    assert contract["manifest_graph"] == "cover_to_header_to_inventory_acyclic"


def test_m7b_portable_validator_is_a_contract_owned_forwarder() -> None:
    tree = ast.parse(VALIDATOR.read_text())
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    assert "radjax_tome" not in imports
    assert "radjax_contract" in imports
