"""M6 regression tests for Builder-independent artifact-validation leaves."""

from __future__ import annotations

import ast
from pathlib import Path

from radjax_tome.artifact_validation.long_tail import long_tail_summary

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "src" / "radjax_tome" / "artifact_validation"


def test_artifact_validation_leaves_have_no_builder_import_path() -> None:
    """A forwarding import would reintroduce the prohibited packaging edge."""
    for path in VALIDATION.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert not any(
            name == "radjax_tome.builder" or name.startswith("radjax_tome.builder.")
            for name in imported
        ), path


def test_long_tail_summary_is_a_pure_payload_validation_primitive() -> None:
    summary = long_tail_summary(
        [
            {
                "long_tail_class": "normal",
                "effective_top_k": 8,
                "effective_top_k_fraction_of_vocab": 0.25,
                "top_k_saturated": False,
            },
            {
                "long_tail_class": "suspicious_flat",
                "effective_top_k": 32,
                "effective_top_k_fraction_of_vocab": 1.0,
                "top_k_saturated": True,
            },
        ]
    )
    assert summary == {
        "count": 2,
        "normal_count": 1,
        "long_tail_count": 0,
        "very_long_tail_count": 0,
        "suspicious_flat_count": 1,
        "full_vocab_or_near_full_vocab_count": 0,
        "saturated_count": 1,
        "max_effective_top_k": 32,
        "mean_effective_top_k": 20.0,
        "max_effective_top_k_fraction_of_vocab": 1.0,
    }
