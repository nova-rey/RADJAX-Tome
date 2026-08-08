from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
CONTRACT_ROOT = ROOT / "contracts" / "radjax_tome" / "v3"
RELEASE_FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "tome_artifact_v3_smoke"
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


def _release_metadata() -> tuple[dict[str, object], dict[str, object]]:
    receipt = json.loads(
        (RELEASE_FIXTURE_ROOT / "contract_release_receipt_v0.9.0.json").read_text()
    )
    assets = json.loads(
        (RELEASE_FIXTURE_ROOT / "contract_release_asset_hashes_v0.9.0.json").read_text()
    )
    return receipt, assets


def test_sanitized_release_receipt_matches_the_pin_and_asset_hash_metadata() -> None:
    receipt, assets = _release_metadata()
    assert receipt["package_name"] == "radjax-contract"
    assert receipt["package_version"] == "0.9.0"
    assert receipt["release_tag"] == "v0.9.0"
    assert receipt["release_commit"] == CONTRACT_COMMIT
    assert receipt["peeled_tag_commit"] == CONTRACT_COMMIT
    assert receipt["artifact_contract_id"] == "radjax_tome_artifact_contract"
    assert receipt["artifact_contract_version"] == "3.0.0"
    assert receipt["reviewed_semantic_commit"] == (
        "63ea3cfa6c7ae91e6a42f4929d59d9cdd6748836"
    )
    assert receipt["v3_asset_manifest_sha256"] == ASSET_TREE_DIGEST
    assert assets["release_tag"] == receipt["release_tag"]
    assert assets["release_commit"] == receipt["release_commit"]
    by_name = {item["filename"]: item for item in receipt["artifacts"]}
    assert (
        by_name["radjax_contract-0.9.0-py3-none-any.whl"]["sha256"]
        == assets["wheel_sha256"]
    )
    assert (
        by_name["radjax_contract-0.9.0-py3-none-any.whl"]["size_bytes"]
        == assets["wheel_size_bytes"]
    )
    assert by_name["radjax_contract-0.9.0.tar.gz"]["sha256"] == assets["sdist_sha256"]
    assert (
        by_name["radjax_contract-0.9.0.tar.gz"]["size_bytes"]
        == assets["sdist_size_bytes"]
    )


def test_source_wheel_sdist_release_and_offline_mirror_share_v3_asset_identity() -> (
    None
):
    receipt, assets = _release_metadata()
    mirror_digest = f"sha256:{_sha256(CONTRACT_ROOT / 'SHA256SUMS')}"
    assert mirror_digest == ASSET_TREE_DIGEST
    assert mirror_digest == receipt["v3_asset_manifest_sha256"]
    assert mirror_digest == assets["source_v3_asset_manifest_sha256"]
    assert mirror_digest == assets["wheel_v3_asset_manifest_sha256"]
    assert mirror_digest == assets["sdist_v3_asset_manifest_sha256"]
    assert mirror_digest == assets["offline_mirror_manifest_sha256"]


def test_release_asset_receipt_hashes_are_pinned_in_sanitized_metadata() -> None:
    _, assets = _release_metadata()
    receipt_asset = RELEASE_FIXTURE_ROOT / "contract_release_receipt_v0.9.0.json"
    sums_asset = RELEASE_FIXTURE_ROOT / "contract_release_SHA256SUMS_v0.9.0"
    assert f"sha256:{_sha256(receipt_asset)}" == assets["release_receipt_asset_sha256"]
    assert f"sha256:{_sha256(sums_asset)}" == assets["release_sha256s_asset_sha256"]
    assert assets["release_receipt_asset_sha256"] == (
        "sha256:3698d64bf4212b6290d69ca181a09fc454d176cf657a7bd73905ceefb5a96014"
    )
    assert assets["release_sha256s_asset_sha256"] == (
        "sha256:29c8ef2eac9cada4393dcaaf0fe553a8700cb55db12b7ae1c63b8e174293d071"
    )
