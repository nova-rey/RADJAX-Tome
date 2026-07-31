"""M6A characterization for production-boundary extraction.

The checks deliberately pin the live M4/M5 public and orchestration seams.
They become stricter only when a later M6 checkpoint removes an edge.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "radjax_tome"
OWNERSHIP_RECORD = ROOT / "docs" / "M6_PRODUCTION_BOUNDARY_OWNERSHIP.md"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
    return names


def test_m6a_ownership_record_pins_required_boundaries() -> None:
    text = OWNERSHIP_RECORD.read_text(encoding="utf-8")
    for required in (
        "sole orchestrator",
        "intent -> resolved configuration -> execution-plan",
        "25-field selection",
        "Contract `v0.2.0`",
        "M7 owns",
    ):
        assert required in text


def test_m6a_native_state_machine_has_no_upward_or_presentation_imports() -> None:
    forbidden = (
        "radjax_tome.builder.production",
        "radjax_tome.cli",
        "radjax_tome.backends.hf_specimen",
        "radjax_tome.backends.hf_export",
        "radjax_tome.backends.qwen_policy",
        "torch",
        "transformers",
    )
    for path in sorted((SOURCE / "builder" / "native_path_b").glob("*.py")):
        imports = _imports(path)
        assert not any(
            imported == blocked or imported.startswith(f"{blocked}.")
            for imported in imports
            for blocked in forbidden
        ), path


def test_m6a_public_facades_and_m4_callback_seam_are_explicit() -> None:
    production = (SOURCE / "builder" / "production.py").read_text(encoding="utf-8")
    for symbol in (
        "class ProductionBuildConfig",
        "def build_production_gpu_tome",
        "def _run_native_path_b_post_score_stages",
        "run_preflight_then_score_pass",
        "NativePathBCallbacks",
        "run_post_score_path_b",
    ):
        assert symbol in production

    packaging = (SOURCE / "tome" / "packaging.py").read_text(encoding="utf-8")
    for symbol in (
        "def package_tome_artifact",
        "def validate_tome_package",
        "class StudentTomeReader",
    ):
        assert symbol in packaging


def test_m6a_production_callers_use_only_the_tome_facade() -> None:
    forbidden_tome_leaves = (
        "radjax_tome.tome.contracts",
        "radjax_tome.tome.canonical_artifact",
        "radjax_tome.tome.cover_page",
        "radjax_tome.tome.packaging",
        "radjax_tome.tome.bundle",
        "radjax_tome.tome.compatibility",
    )
    for relative in (
        "builder/backend_textbook.py",
        "builder/production.py",
        "builder/teacher_textbook.py",
    ):
        imports = _imports(SOURCE / relative)
        assert "radjax_tome.tome" in imports
        assert not any(
            imported == forbidden or imported.startswith(f"{forbidden}.")
            for imported in imports
            for forbidden in forbidden_tome_leaves
        ), relative
