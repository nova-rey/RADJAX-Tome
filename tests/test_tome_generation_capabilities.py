from __future__ import annotations

import json
from pathlib import Path

from radjax_tome.fingerprint import (
    generate_corridor_measurement_report,
    generate_corridor_subset_receipt,
    generate_exemplar_reservoir,
    summarize_exemplar_reservoir,
    validate_fingerprint_artifact,
)
from radjax_tome.targets import inspect_target_store, write_compressed_target_store
from tests.helpers.fixtures import (
    build_minimal_fingerprint_artifact,
    build_minimal_target_store,
)
from tests.helpers.subprocess import run_script

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CAPABILITIES = {
    "dense_teacher_targets",
    "topk_tail_targets",
    "cascaded_bucket_targets",
    "fingerprint_artifact_generation",
    "corridor_subset_generation",
    "exemplar_reservoir_generation",
    "hf_local_teacher_export",
    "prompt_corpus_tokenization",
}


def test_dense_teacher_targets(tmp_path: Path) -> None:
    store = build_minimal_target_store(tmp_path)

    summary = inspect_target_store(store.root)

    assert summary["target_type"] == "synthetic"
    assert "logits" in summary["array_keys"]


def test_topk_tail_targets(tmp_path: Path) -> None:
    dense = build_minimal_target_store(tmp_path)
    topk = write_compressed_target_store(
        dense,
        tmp_path / "topk",
        target_type="topk_with_tail_v0",
        top_k=2,
    )

    summary = inspect_target_store(topk.root)

    assert summary["target_type"] == "topk_with_tail_v0"
    assert {
        "input_ids",
        "attention_mask",
        "top_token_ids",
        "top_log_probs",
        "top_mass",
        "tail_mass",
        "teacher_entropy",
    } <= set(summary["array_keys"])


def test_cascaded_bucket_targets(tmp_path: Path) -> None:
    dense = build_minimal_target_store(tmp_path)
    cascaded = write_compressed_target_store(
        dense,
        tmp_path / "cascaded",
        target_type="cascaded_soft_labels_v1",
        top_k=2,
        bucket_edges=(1.0, 0.1, 0.0),
    )

    summary = inspect_target_store(cascaded.root)

    assert summary["target_type"] == "cascaded_soft_labels_v1"
    assert {
        "bucket_mass",
        "bucket_count",
        "bucket_mean_logp",
    } <= set(summary["array_keys"])
    assert summary["target_params"]["bucket_edges"] == "1.0,0.1,0.0"


def test_fingerprint_artifact_generation(tmp_path: Path) -> None:
    artifact = build_minimal_fingerprint_artifact(tmp_path)
    validation = validate_fingerprint_artifact(artifact)

    assert validation.ok
    assert (artifact / "manifest.json").is_file()
    assert (artifact / "modes.json").is_file()
    assert (artifact / "targets-00000.jsonl").is_file()
    assert (artifact / "byte_accounting.json").is_file()


def test_corridor_subset_generation(tmp_path: Path) -> None:
    artifact = build_minimal_fingerprint_artifact(tmp_path)

    receipt = generate_corridor_subset_receipt(
        artifact,
        tmp_path / "receipt.json",
    )
    report = generate_corridor_measurement_report(
        artifact,
        tmp_path / "corridor.json",
    )

    assert receipt.subset_role == "corridor_subset"
    assert receipt.selected_record_count == 2
    assert report.status == "pass"


def test_exemplar_reservoir_generation(tmp_path: Path) -> None:
    artifact = build_minimal_fingerprint_artifact(tmp_path)

    manifest = generate_exemplar_reservoir(
        artifact,
        max_seq_len=3,
        vocab_size=8,
    )
    summary = summarize_exemplar_reservoir(artifact)

    assert manifest.num_records == 1
    assert summary.num_records == 1


def test_capability_harness_writes_matrix_and_report(tmp_path: Path) -> None:
    script = ROOT / "scripts" / "prove_tome_generation_capabilities.py"
    matrix_json = tmp_path / "matrix.json"
    report_md = tmp_path / "matrix.md"

    result = run_script(
        ROOT,
        str(script),
        "--work-dir",
        str(tmp_path / "artifacts"),
        "--matrix-json",
        str(matrix_json),
        "--report-md",
        str(report_md),
        "--overwrite",
    )

    assert result.returncode == 0, result.stderr
    matrix = json.loads(matrix_json.read_text(encoding="utf-8"))
    entries = {item["capability_id"]: item for item in matrix["capabilities"]}
    assert REQUIRED_CAPABILITIES == set(entries)
    assert entries["dense_teacher_targets"]["status"].endswith("synthetic_only")
    assert entries["topk_tail_targets"]["blocks_spec3"] is False
    assert entries["cascaded_bucket_targets"]["blocks_spec3"] is False
    assert (
        entries["hf_local_teacher_export"]["status"] == "schema_validate_inspect_only"
    )
    assert "torch" not in result.stdout.lower()
    assert "Spec 3 may proceed" in report_md.read_text(encoding="utf-8")


def test_committed_capability_matrix_is_compact_when_present() -> None:
    path = ROOT / "docs/TOME_GENERATION_CAPABILITY_MATRIX.json"
    if path.exists():
        matrix = json.loads(path.read_text(encoding="utf-8"))
        assert REQUIRED_CAPABILITIES == {
            item["capability_id"] for item in matrix["capabilities"]
        }
        assert path.stat().st_size < 100_000
