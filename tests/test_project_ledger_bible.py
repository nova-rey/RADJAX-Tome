from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_project_ledger_bible_exists_and_mentions_current_arc() -> None:
    text = (ROOT / "bible.md").read_text(encoding="utf-8")

    assert "2.14" in text
    assert "2.18" in text
    assert "3.0" in text
    assert "3.1" in text
    assert "cover_page.json" in text
    assert "3.2" in text
    assert ".rtome" in text
    assert "deterministic tar" in text
    assert "3.3A" in text
    assert "runtime mode" in text
    assert "capability matrix" in text
    assert "3.3B" in text
    assert "backend contract" in text
    assert "fake_numpy" in text
    assert "3.3C" in text
    assert "CPU reference backend" in text
    assert "cpu_reference" in text
    assert "3.3C.1" in text
    assert "corridor_exemplar_v1" in text
    assert "3.3D" in text
    assert "auto / serial / staged" in text
    assert "backend batch runner" in text
    assert "3.3E" in text
    assert "hf_torch" in text
    assert "lazy" in text
    assert "3.3F1" in text
    assert "gpu_torch" in text
    assert "CUDA-then-MPS" in text
    assert "3.3F2" in text
    assert "topk_with_tail_v0" in text
    assert "compact payload" in text
    assert "3.3F3" in text
    assert "cascaded_soft_labels_v1" in text
    assert "bucket_masses" in text
    assert "3.3F4" in text
    assert "vocab chunking" in text
    assert "estimated_reducer_workspace_bytes" in text
    assert "3.3F4.1" in text
    assert "exact_bucket_policy_requires_full_probability_workspace" in text
    assert "metadata truth" in text
    assert "3.3F5" in text
    assert "runtime diagnostics" in text
    assert "no silent CPU fallback" in text
    assert "3.3F6" in text
    assert "dynamic_cascaded_soft_labels_v1" in text
    assert "top_selection_mask" in text
    assert "3.3F7" in text
    assert "compact payload transfer" in text
    assert "3.3F7.1" in text
    assert "vectorized dynamic head selection" in text
    assert "bucketed tail is preserved" in text
