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
    assert "3.3F8" in text
    assert "production_corridor_schema" in text
    assert "exemplar_source_policy" in text
    assert "3.3F9" in text
    assert "compact_corridor_exemplar" in text
    assert "gpu_torch_production" in text
    assert "3.3F9.1" in text
    assert "one_pass_candidate" in text
    assert "batch_all_examples" in text
    assert "3.3F9.2" in text
    assert "two_pass_sparse_exemplar" in text
    assert "score_pass" in text
    assert "selected_exemplar_pass" in text
    assert "3.3F9.3" in text
    assert "exemplar_capture_mode=auto" in text
    assert "auto_policy_reason" in text
    assert "auto_policy_inputs_missing" in text
    assert "3.3F9.4" in text
    assert "gpu batch size policy" in text
    assert "exponential_probe_v1" in text
    assert "3.3F10" in text
    assert "GPU Builder Integration Gate" in text
    assert "metadata propagation" in text
    assert "no silent CPU fallback" in text
    assert "3.3F10.1" in text
    assert "multi_leaderboard_exemplar_selector_v1" in text
    assert "exemplar_selection_manifest.json" in text
    assert "3.3F10.1.1" in text
    assert "rank_aware_board_assignment_with_backfill_v1" in text
    assert "runner-up backfill" in text
    assert "3.3F11" in text
    assert "GPU Runtime Final Polish" in text
    assert "runtime doctor" in text
    assert "preflight report" in text
    assert "artifact metadata sanity report" in text
    assert "remediation hints" in text
    assert "backend availability summary" in text
    assert "selector metadata sanity" in text
    assert "batch-size metadata sanity" in text
    assert "no new reducer math" in text
    assert "no new selector policy" in text
    assert "no real auto batch probing" in text
    assert "no production global selector" in text
    assert "no multidevice" in text
    assert "no TPU/JAX" in text
    assert "Spec 4.1" in text
    assert "Corpus Builder and Provenance Contract" in text
    assert "corpus_hash" in text
    assert "manifest_hash" in text
    assert "radjax-tome corpus build" in text
    assert "source_corpus_hash" in text
    assert "Spec 4.1.1" in text
    assert "Corpus Format Truth Cleanup" in text
    assert "Structured `.json` import is intentionally not supported yet" in text
    assert "exclude_self_hash_and_created_at_v1" in text
