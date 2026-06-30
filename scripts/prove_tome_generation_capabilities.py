#!/usr/bin/env python
# ruff: noqa: E501
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from radjax_tome.backends.hf_export import (
    HFTeacherExportConfig,
    build_hf_export_metadata,
    validate_hf_export_config,
    write_hf_export_metadata,
)
from radjax_tome.corpora import (
    PromptCorpus,
    PromptRecord,
    build_prompt_corpus_manifest,
    create_tokenizer,
    tokenize_jsonl_corpus,
    write_prompt_corpus,
    write_prompt_corpus_manifest,
)
from radjax_tome.fingerprint import (
    build_minimal_fingerprint_artifact_from_target_store,
    generate_corridor_measurement_report,
    generate_corridor_subset_receipt,
    generate_exemplar_reservoir,
    inspect_fingerprint_artifact,
    summarize_exemplar_reservoir,
    validate_fingerprint_artifact,
)
from radjax_tome.targets import inspect_target_store, write_compressed_target_store
from radjax_tome.targets.export import export_synthetic_teacher_targets

STATUSES = {
    "active_generation_supported",
    "active_generation_supported_synthetic_only",
    "schema_validate_inspect_only",
    "deferred_with_reason",
    "missing_blocker",
    "out_of_scope_student_or_contract",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prove active RADJAX-Tome teacher-side generation capabilities."
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--matrix-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run-optional-hf-local", action="store_true")
    args = parser.parse_args(argv)

    if args.work_dir.exists():
        if not args.overwrite:
            raise ValueError(f"work dir already exists: {args.work_dir}")
        shutil.rmtree(args.work_dir)
    args.work_dir.mkdir(parents=True)

    entries = prove_capabilities(
        args.work_dir,
        run_optional_hf_local=args.run_optional_hf_local,
    )
    matrix = {
        "kind": "radjax_tome_generation_capability_matrix",
        "version": 1,
        "default_proof_mode": "ci_safe_no_network",
        "capabilities": entries,
    }
    _validate_matrix(matrix)
    args.matrix_json.parent.mkdir(parents=True, exist_ok=True)
    args.matrix_json.write_text(
        json.dumps(matrix, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(render_report(matrix), encoding="utf-8")
    blockers = [entry for entry in entries if entry["blocks_spec3"]]
    print(
        "status=complete "
        f"capabilities={len(entries)} blockers={len(blockers)} "
        f"matrix_json={args.matrix_json}"
    )
    return 1 if blockers else 0


def prove_capabilities(
    work_dir: Path, *, run_optional_hf_local: bool = False
) -> list[dict[str, Any]]:
    dense_dir = work_dir / "dense_teacher_targets"
    dense_store = export_synthetic_teacher_targets(
        dense_dir / "targets",
        num_examples=2,
        sequence_length=3,
        vocab_size=8,
        overwrite=True,
    )
    dense_report = inspect_target_store(dense_store.root)
    _write_json(dense_dir / "inspection.json", dense_report)

    topk_dir = work_dir / "topk_tail_targets"
    topk_store = write_compressed_target_store(
        dense_store,
        topk_dir / "targets",
        target_type="topk_with_tail_v0",
        top_k=2,
        overwrite=True,
    )
    topk_report = inspect_target_store(topk_store.root)
    _write_json(topk_dir / "inspection.json", topk_report)

    cascaded_dir = work_dir / "cascaded_bucket_targets"
    cascaded_store = write_compressed_target_store(
        dense_store,
        cascaded_dir / "targets",
        target_type="cascaded_soft_labels_v1",
        top_k=2,
        bucket_edges=(1.0, 0.1, 0.0),
        overwrite=True,
    )
    cascaded_report = inspect_target_store(cascaded_store.root)
    _write_json(cascaded_dir / "inspection.json", cascaded_report)

    fingerprint_dir = work_dir / "fingerprint_artifact_generation"
    fingerprint = build_minimal_fingerprint_artifact_from_target_store(
        dense_store,
        fingerprint_dir / "artifact",
        overwrite=True,
    )
    fingerprint_validation = validate_fingerprint_artifact(fingerprint)
    _write_json(
        fingerprint_dir / "inspection.json",
        inspect_fingerprint_artifact(fingerprint),
    )

    corridor_dir = work_dir / "corridor_subset_generation"
    receipt = generate_corridor_subset_receipt(
        fingerprint,
        corridor_dir / "budget_subset_receipt.json",
    )
    generate_corridor_measurement_report(
        fingerprint,
        corridor_dir / "corridor_measurement_report.json",
    )

    exemplar_dir = work_dir / "exemplar_reservoir_generation"
    exemplar_manifest = generate_exemplar_reservoir(
        fingerprint,
        max_seq_len=dense_store.metadata.sequence_length,
        vocab_size=dense_store.metadata.vocab_size,
    )
    exemplar_summary = summarize_exemplar_reservoir(fingerprint)
    _write_json(exemplar_dir / "summary.json", exemplar_summary.to_dict())

    hf_dir = work_dir / "hf_local_teacher_export"
    hf_config = HFTeacherExportConfig(
        resolved_model_id="local/tiny-hf-model",
        tokenizer_id="local/tiny-hf-tokenizer",
        local_files_only=True,
        allow_downloads=False,
        sequence_length=3,
        vocab_size=8,
        prompt_count=2,
    )
    ok, blockers = validate_hf_export_config(hf_config)
    if not ok:
        raise ValueError("; ".join(blockers))
    hf_metadata = build_hf_export_metadata(hf_config)
    write_hf_export_metadata(hf_dir / "hf_export_metadata.json", hf_metadata)

    prompt_dir = work_dir / "prompt_corpus_tokenization"
    prompt_corpus = PromptCorpus(
        corpus_id="capability-smoke",
        records=(
            PromptRecord(id="p0", text="alpha", split="train", tags=("smoke",)),
            PromptRecord(id="p1", text="beta", split="validation", tags=("smoke",)),
        ),
    )
    corpus_path = write_prompt_corpus(
        prompt_corpus,
        prompt_dir / "prompts.jsonl",
        overwrite=True,
    )
    manifest = build_prompt_corpus_manifest(
        prompt_corpus,
        description="Spec 2.11 capability proof prompt corpus.",
    )
    write_prompt_corpus_manifest(
        manifest,
        prompt_dir / "prompt_manifest.json",
        overwrite=True,
    )
    token_manifest = tokenize_jsonl_corpus(
        corpus_path,
        prompt_dir / "tokenized",
        tokenizer=create_tokenizer(),
        sequence_length=8,
        overwrite=True,
    )

    return [
        _entry(
            "dense_teacher_targets",
            "Generate dense/high-resolution teacher target store.",
            ["scripts/build_teacher_textbook.py", "src/qrwkv_xla/teacher_export/hf.py"],
            [
                "scripts/export_teacher_targets.py",
                "radjax_tome.targets.export.export_synthetic_teacher_targets",
            ],
            "active_generation_supported_synthetic_only",
            "python scripts/export_teacher_targets.py --backend synthetic --out artifacts/tome_generation_capabilities/dense_teacher_targets/targets --num-examples 2 --sequence-length 3 --overwrite",
            "python scripts/inspect_targets.py artifacts/tome_generation_capabilities/dense_teacher_targets/targets --json",
            [str(dense_store.root), str(dense_dir / "inspection.json")],
            ["tests/test_tome_generation_capabilities.py::test_dense_teacher_targets"],
            [
                "Default proof uses SyntheticTeacherBackend; real HF teacher export is classified separately."
            ],
        ),
        _entry(
            "topk_tail_targets",
            "Generate top-k/tail compressed teacher target store.",
            ["docs/TOPK_TAIL_TEXTBOOK.md", "tests/test_topk_tail_textbook.py"],
            ["radjax_tome.targets.compression.write_compressed_target_store"],
            "active_generation_supported_synthetic_only",
            "python scripts/prove_tome_generation_capabilities.py --work-dir artifacts/tome_generation_capabilities --matrix-json docs/TOME_GENERATION_CAPABILITY_MATRIX.json --report-md docs/TOME_GENERATION_CAPABILITY_MATRIX.md --overwrite",
            "python scripts/inspect_targets.py artifacts/tome_generation_capabilities/topk_tail_targets/targets --json",
            [str(topk_store.root), str(topk_dir / "inspection.json")],
            ["tests/test_tome_generation_capabilities.py::test_topk_tail_targets"],
            ["Compression proof starts from deterministic synthetic dense logits."],
        ),
        _entry(
            "cascaded_bucket_targets",
            "Generate cascaded bucket compressed teacher target store.",
            [
                "docs/CASCADED_SOFT_LABELS.md",
                "tests/test_cascaded_soft_labels_textbook.py",
            ],
            ["radjax_tome.targets.compression.write_compressed_target_store"],
            "active_generation_supported_synthetic_only",
            "python scripts/prove_tome_generation_capabilities.py --work-dir artifacts/tome_generation_capabilities --matrix-json docs/TOME_GENERATION_CAPABILITY_MATRIX.json --report-md docs/TOME_GENERATION_CAPABILITY_MATRIX.md --overwrite",
            "python scripts/inspect_targets.py artifacts/tome_generation_capabilities/cascaded_bucket_targets/targets --json",
            [str(cascaded_store.root), str(cascaded_dir / "inspection.json")],
            [
                "tests/test_tome_generation_capabilities.py::test_cascaded_bucket_targets"
            ],
            [
                "Bucket proof uses deterministic synthetic logits and probability bucket edges 1.0,0.1,0.0."
            ],
        ),
        _entry(
            "fingerprint_artifact_generation",
            "Generate a minimal behavioral fingerprint artifact from a target store.",
            [
                "scripts/build_fingerprint_artifact.py",
                "src/qrwkv_xla/artifacts/fingerprint.py",
            ],
            [
                "radjax_tome.fingerprint.generation.build_minimal_fingerprint_artifact_from_target_store"
            ],
            "active_generation_supported_synthetic_only",
            "python scripts/prove_tome_generation_capabilities.py --work-dir artifacts/tome_generation_capabilities --matrix-json docs/TOME_GENERATION_CAPABILITY_MATRIX.json --report-md docs/TOME_GENERATION_CAPABILITY_MATRIX.md --overwrite",
            "python scripts/validate_fingerprint_artifact.py artifacts/tome_generation_capabilities/fingerprint_artifact_generation/artifact",
            [str(fingerprint), str(fingerprint_dir / "inspection.json")],
            [
                "tests/test_tome_generation_capabilities.py::test_fingerprint_artifact_generation"
            ],
            [
                f"Validator status: {fingerprint_validation.status}; generated from synthetic target store."
            ],
        ),
        _entry(
            "corridor_subset_generation",
            "Generate producer-side corridor subset receipt and measurement report.",
            [
                "scripts/run_corridor_measurement.py",
                "src/qrwkv_xla/fingerprint/budgeted_artifact.py",
            ],
            [
                "radjax_tome.fingerprint.generation.generate_corridor_subset_receipt",
                "radjax_tome.fingerprint.generation.generate_corridor_measurement_report",
            ],
            "active_generation_supported",
            "python scripts/prove_tome_generation_capabilities.py --work-dir artifacts/tome_generation_capabilities --matrix-json docs/TOME_GENERATION_CAPABILITY_MATRIX.json --report-md docs/TOME_GENERATION_CAPABILITY_MATRIX.md --overwrite",
            "python -m pytest tests/test_tome_generation_capabilities.py::test_corridor_subset_generation",
            [
                str(corridor_dir / "budget_subset_receipt.json"),
                str(corridor_dir / "corridor_measurement_report.json"),
            ],
            [
                "tests/test_tome_generation_capabilities.py::test_corridor_subset_generation"
            ],
            [
                f"Producer artifact receipt role: {receipt.subset_role}; Student runtime corridor scoring is out of scope."
            ],
        ),
        _entry(
            "exemplar_reservoir_generation",
            "Generate exemplar reservoir manifest and JSONL records.",
            [
                "scripts/run_exemplar_pass.py",
                "src/qrwkv_xla/artifacts/fingerprint_exemplars.py",
            ],
            ["radjax_tome.fingerprint.generation.generate_exemplar_reservoir"],
            "active_generation_supported_synthetic_only",
            "python scripts/prove_tome_generation_capabilities.py --work-dir artifacts/tome_generation_capabilities --matrix-json docs/TOME_GENERATION_CAPABILITY_MATRIX.json --report-md docs/TOME_GENERATION_CAPABILITY_MATRIX.md --overwrite",
            "python -m pytest tests/test_tome_generation_capabilities.py::test_exemplar_reservoir_generation",
            [str(fingerprint / "exemplars"), str(exemplar_dir / "summary.json")],
            [
                "tests/test_tome_generation_capabilities.py::test_exemplar_reservoir_generation"
            ],
            [
                f"Generated {exemplar_manifest.num_records} deterministic exemplar record; summary records={exemplar_summary.num_records}."
            ],
        ),
        _entry(
            "hf_local_teacher_export",
            "Validate HF/local-files-only teacher export metadata boundary.",
            [
                "src/qrwkv_xla/teacher_export/hf.py",
                "tests/test_teacher_export_hf_backend.py",
            ],
            ["radjax_tome.backends.hf_export.build_hf_export_metadata"],
            "schema_validate_inspect_only",
            "python scripts/prove_tome_generation_capabilities.py --work-dir artifacts/tome_generation_capabilities --matrix-json docs/TOME_GENERATION_CAPABILITY_MATRIX.json --report-md docs/TOME_GENERATION_CAPABILITY_MATRIX.md --overwrite",
            "python -m pytest tests/test_tome_generation_capabilities.py::test_hf_local_teacher_export_metadata_only",
            [str(hf_dir / "hf_export_metadata.json")],
            [
                "tests/test_tome_generation_capabilities.py::test_hf_local_teacher_export_metadata_only"
            ],
            [
                "Default proof validates local-files-only HF metadata only; no optional torch/transformers run or model files are required.",
                f"Optional local HF proof requested: {run_optional_hf_local}.",
            ],
        ),
        _entry(
            "prompt_corpus_tokenization",
            "Generate prompt corpus, manifest, tokenized corpus, and source hashes.",
            ["corpora/smoke_prompts.jsonl", "scripts/create_prompt_manifest.py"],
            ["scripts/tokenize_corpus.py", "radjax_tome.corpora.tokenize_jsonl_corpus"],
            "active_generation_supported",
            "python scripts/tokenize_corpus.py artifacts/tome_generation_capabilities/prompt_corpus_tokenization/prompts.jsonl --out artifacts/tome_generation_capabilities/prompt_corpus_tokenization/tokenized --sequence-length 8 --overwrite",
            "python -m pytest tests/test_tome_generation_capabilities.py::test_prompt_corpus_tokenization_generation",
            [
                str(corpus_path),
                str(prompt_dir / "prompt_manifest.json"),
                str(prompt_dir / "tokenized"),
            ],
            [
                "tests/test_tome_generation_capabilities.py::test_prompt_corpus_tokenization_generation"
            ],
            [
                f"Smoke tokenizer proof generated {token_manifest.num_sequences} tokenized sequences."
            ],
        ),
    ]


def _entry(
    capability_id: str,
    description: str,
    old_repo_reference_paths: list[str],
    active_entrypoints: list[str],
    status: str,
    generation_command: str,
    validation_command: str,
    proof_artifacts: list[str],
    tested_by: list[str],
    limitations: list[str],
    *,
    blocks_spec3: bool = False,
) -> dict[str, Any]:
    return {
        "capability_id": capability_id,
        "description": description,
        "old_repo_reference_paths": old_repo_reference_paths,
        "active_entrypoints": active_entrypoints,
        "status": status,
        "generation_command": generation_command,
        "validation_command": validation_command,
        "proof_artifacts": proof_artifacts,
        "tested_by": tested_by,
        "limitations": limitations,
        "blocks_spec3": blocks_spec3,
    }


def render_report(matrix: dict[str, Any]) -> str:
    entries = matrix["capabilities"]
    blockers = [entry for entry in entries if entry["blocks_spec3"]]
    lines = [
        "# Tome Generation Capability Matrix",
        "",
        "## Executive Verdict",
        "",
        (
            "Spec 3 may proceed: no teacher-side Tome generation capability in "
            "this matrix blocks Contract-valid Tome emission."
            if not blockers
            else "Spec 3 is blocked by capability gaps listed below."
        ),
        "",
        "| Capability | Status | Proof | Limitation | Blocks Spec 3 |",
        "|---|---|---|---|---|",
    ]
    for entry in entries:
        lines.append(
            "| "
            + " | ".join(
                [
                    entry["capability_id"],
                    entry["status"],
                    entry["proof_artifacts"][0],
                    entry["limitations"][0],
                    "yes" if entry["blocks_spec3"] else "no",
                ]
            )
            + " |"
        )
    active = [
        entry["capability_id"]
        for entry in entries
        if entry["status"].startswith("active_generation_supported")
    ]
    schema_only = [
        entry["capability_id"]
        for entry in entries
        if entry["status"] == "schema_validate_inspect_only"
    ]
    lines.extend(
        [
            "",
            "## Active Generation",
            "",
            ", ".join(active) if active else "None.",
            "",
            "## Schema Or Metadata Only",
            "",
            ", ".join(schema_only) if schema_only else "None.",
            "",
            "## Synthetic Vs Real Teacher Proof",
            "",
            "Dense, top-k/tail, cascaded, fingerprint, and exemplar proofs use",
            "deterministic synthetic data by default. Real HF teacher generation",
            "is not claimed by the default proof; only local-files-only metadata",
            "is validated.",
            "",
            "## Exact Next Recommendation",
            "",
            (
                "Proceed to Spec 3 Contract-valid Tome emission, keeping real HF "
                "generation optional/local-files-only until an explicit optional "
                "proof is run."
                if not blockers
                else "Resolve blocking capability gaps before Spec 3."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _validate_matrix(matrix: dict[str, Any]) -> None:
    required = {
        "dense_teacher_targets",
        "topk_tail_targets",
        "cascaded_bucket_targets",
        "fingerprint_artifact_generation",
        "corridor_subset_generation",
        "exemplar_reservoir_generation",
        "hf_local_teacher_export",
        "prompt_corpus_tokenization",
    }
    entries = matrix.get("capabilities", [])
    seen = {entry.get("capability_id") for entry in entries}
    missing = sorted(required - seen)
    if missing:
        raise ValueError(f"capability matrix missing entries: {missing}")
    for entry in entries:
        status = entry.get("status")
        if status not in STATUSES:
            raise ValueError(f"unsupported capability status: {status!r}")
        if not entry.get("limitations"):
            raise ValueError(f"{entry.get('capability_id')} missing limitations")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"Capability proof failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
