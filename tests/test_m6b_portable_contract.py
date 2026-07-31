from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import jsonschema

from radjax_tome.tome.contracts import (
    PackageInventoryEntry,
    TrainingPayloadEntry,
    build_canonical_content_manifest,
    build_canonical_tome_cover,
    build_tome_semantic_identity,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts" / "radjax_tome" / "v1"
VALIDATOR = ROOT / "tools" / "validate_radjax_tome_contract.py"


def _cover() -> dict[str, object]:
    identity = build_tome_semantic_identity(
        training_payload=(TrainingPayloadEntry("payload", "sha256:" + "1" * 64),),
        training_contract={"target_type": "fixture"},
        authority={"selection_integration_config_hash": "sha256:" + "2" * 64},
    )
    manifest = build_canonical_content_manifest(
        profile="unpacked",
        semantic_identity=identity,
        inventory=(
            PackageInventoryEntry(
                "payload.bin", "sha256:" + "3" * 64, 7, "training_critical", True
            ),
        ),
    )
    return build_canonical_tome_cover(
        semantic_identity=identity,
        content_manifest=manifest,
        provenance={},
        validation={},
    )


def test_m6b_contract_schemas_are_valid_draft_2020_12() -> None:
    for path in sorted((CONTRACT / "schemas").glob("*.json")):
        jsonschema.Draft202012Validator.check_schema(json.loads(path.read_text()))


def test_m6b_contract_checksum_inventory_is_complete() -> None:
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


def test_m6b_independent_validator_accepts_native_contract_shape(
    tmp_path: Path,
) -> None:
    cover = _cover()
    (tmp_path / "payload.bin").write_bytes(b"payload")
    cover["manifests"]["content"]["inventory"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(b"payload").hexdigest()
    )
    manifest = cover["manifests"]["content"]
    manifest["manifest_digest"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                {
                    key: manifest[key]
                    for key in (
                        "schema_version",
                        "profile",
                        "semantic_identity_digest",
                        "inventory",
                    )
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    )
    (tmp_path / "cover_page.json").write_text(json.dumps(cover), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(VALIDATOR), str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"errors": [], "ok": True, "warnings": []}


def test_m6b_portable_validator_is_standard_library_only() -> None:
    tree = ast.parse(VALIDATOR.read_text(encoding="utf-8"))
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
    assert "jsonschema" not in imports
