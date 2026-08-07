from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
CONTRACT_ROOT = ROOT / "contracts" / "radjax_tome" / "v3"
CONTRACT_COMMIT = "1fa43e1aea2e198511db86dafb0aeefa525d48c7"
ASSET_TREE_DIGEST = (
    "sha256:4f81cd901ad074cc24e279e37b7fbfbe25c22fb6b7fd77cbcc747b202995acf8"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_offline_v3_mirror_is_closed_and_self_verified() -> None:
    manifest = CONTRACT_ROOT / "SHA256SUMS"
    assert _sha256(manifest) == ASSET_TREE_DIGEST.removeprefix("sha256:")
    rows = [line.split(maxsplit=1) for line in manifest.read_text().splitlines()]
    assert rows
    for digest, relative in rows:
        path = CONTRACT_ROOT / relative
        assert path.is_file(), relative
        assert _sha256(path) == digest, relative


def test_v3_mirror_contains_released_identity_and_vectors() -> None:
    contract = json.loads((CONTRACT_ROOT / "contract.json").read_text())
    assert contract["contract_id"] == "radjax_tome_artifact_contract"
    assert contract["publication_version"] == "3.0.0"
    assert (CONTRACT_ROOT / "vectors" / "tome_provenance_v3_vectors.json").is_file()


def test_installed_contract_exposes_exact_v3_identity() -> None:
    from radjax_contract.tome.v3 import validate_tome_artifact_v3
    from radjax_contract.tome.v3.schema import CONTRACT_VERSION, SEMANTIC_PROFILE_ID

    assert CONTRACT_VERSION == "radjax_tome_artifact_contract@3.0.0"
    assert SEMANTIC_PROFILE_ID == "selected_exemplar_semantic_profile_v3"
    assert callable(validate_tome_artifact_v3)


def test_pin_is_commit_specific() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert f"RADJAX-Contract.git@{CONTRACT_COMMIT}" in pyproject
