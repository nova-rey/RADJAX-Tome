"""M6D packaging consumes Tome descriptors and not builder internals."""

from __future__ import annotations

import ast
from pathlib import Path

from radjax_tome.tome.artifact_descriptor import ValidatedTomeArtifact
from radjax_tome.tome.packaging import FULL_DEBUG_PROVENANCE, package_tome_artifact
from tests.test_tome_packaging_profiles import _artifact

ROOT = Path(__file__).resolve().parents[1]
PACKAGING = ROOT / "src" / "radjax_tome" / "tome" / "packaging.py"


def test_packaging_has_no_direct_builder_or_audit_imports() -> None:
    tree = ast.parse(PACKAGING.read_text(encoding="utf-8"), filename=str(PACKAGING))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any(
        name == "radjax_tome.builder" or name.startswith("radjax_tome.builder.")
        for name in imported
    )
    assert "radjax_tome.audit" not in imported
    assert "radjax_tome.tome.artifact_descriptor" in imported
    assert "radjax_tome.tome.producer_validation" in imported


def test_packaging_consumes_a_complete_canonical_descriptor(tmp_path: Path) -> None:
    source = _artifact(tmp_path / "source")
    package = tmp_path / "package"
    package_tome_artifact(
        source, package, profile=FULL_DEBUG_PROVENANCE, overwrite=True
    )
    descriptor = ValidatedTomeArtifact.from_canonical_directory(package)

    assert descriptor.root == package.resolve()
    assert descriptor.cover["identity"]["semantic_digest"] == (
        descriptor.semantic_identity.semantic_digest
    )
    assert descriptor.profile == FULL_DEBUG_PROVENANCE
    assert descriptor.inventory == descriptor.content_manifest.inventory
    assert descriptor.validation_evidence
    assert "selection_integration_config_hash" in descriptor.authority_references
