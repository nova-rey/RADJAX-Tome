"""M6D packaging consumes Tome descriptors and not builder internals."""

from __future__ import annotations

import ast
from pathlib import Path

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
