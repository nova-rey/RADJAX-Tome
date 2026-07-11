from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path

HELP_DESCRIPTION = """RADJAX-Tome produces teacher-side distillation artifacts.

Recommended commands:
  build
  production-build
  validate
  inspect
  plan
  exemplar-delivery-parity
  audit-selected-linkage
  package-artifact
  validate-package
  corpus
  pack
  unpack
  prove-capabilities
"""


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="radjax-tome",
        description=HELP_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build",
        help="Build a teacher-side Tome artifact.",
        description=(
            "Build a teacher-side TeacherTextbook artifact. Fake mode is the "
            "recommended offline happy path."
        ),
    )
    build.add_argument("--output", type=Path, required=True)
    build.add_argument(
        "--teacher-mode",
        choices=("fake", "synthetic", "hf"),
        default="fake",
        help="Use fake/synthetic for offline deterministic output; hf needs extras.",
    )
    build.add_argument("--teacher-model", default="fake-deterministic-teacher")
    build.add_argument("--teacher-model-provenance", type=Path)
    build.add_argument("--dataset", type=Path)
    build.add_argument("--corpus-manifest", type=Path)
    build.add_argument("--max-examples", type=int, default=4)
    build.add_argument("--sequence-length", type=int, default=16)
    build.add_argument("--batch-size", type=int, default=2)
    build.add_argument("--vocab-size", type=int, default=32)
    build.add_argument(
        "--target-type",
        choices=("dense_logits", "topk_with_tail_v0", "cascaded_soft_labels_v1"),
        default="dense_logits",
    )
    build.add_argument("--top-k", type=int, default=256)
    build.add_argument(
        "--teacher-backend",
        choices=("fake_numpy", "cpu_reference", "hf_torch", "gpu_torch"),
        help="Route build through the TeacherEmissionBackend contract.",
    )
    build.add_argument("--runtime-mode", choices=("cpu", "cpu_gpu"), default="cpu")
    build.add_argument(
        "--target-policy",
        choices=(
            "dense",
            "dense_logits",
            "topk",
            "topk_with_tail_v0",
            "cascaded",
            "cascaded_soft_labels_v1",
            "dynamic",
            "dynamic_cascaded_soft_labels_v1",
            "corridor",
            "corridor_exemplar_v1",
        ),
        help="Backend contract target policy. Defaults to --target-type.",
    )
    build.add_argument(
        "--exemplar-source-policy",
        choices=(
            "dense_logits",
            "cascaded_soft_labels_v1",
            "dynamic_cascaded_soft_labels_v1",
        ),
        default="dynamic_cascaded_soft_labels_v1",
    )
    build.add_argument(
        "--exemplar-capture-mode",
        choices=("one_pass_candidate", "two_pass_sparse_exemplar", "auto"),
        default="one_pass_candidate",
    )
    build.add_argument(
        "--exemplar-second-pass-source-policy",
        choices=(
            "dense_logits",
            "cascaded_soft_labels_v1",
            "dynamic_cascaded_soft_labels_v1",
        ),
        default="dynamic_cascaded_soft_labels_v1",
    )
    build.add_argument(
        "--gpu-batch-size-mode",
        choices=("preset", "custom", "auto"),
        default="preset",
    )
    build.add_argument("--gpu-batch-size-preset", type=int, default=8)
    build.add_argument("--gpu-batch-size-custom", type=int)
    build.add_argument("--gpu-batch-size-auto-min", type=int, default=1)
    build.add_argument("--gpu-batch-size-auto-max", type=int, default=64)
    build.add_argument("--streaming", action="store_true")
    build.add_argument("--resume", action="store_true")
    build.add_argument("--shard-size-examples", type=int)
    build.add_argument("--progress-log", type=Path)
    build.add_argument("--run-manifest", type=Path)
    build.add_argument("--fallback-policy", choices=("error", "auto"), default="error")
    build.add_argument("--exemplar-selection-enabled", action="store_true")
    build.add_argument(
        "--exemplar-selector-policy",
        choices=("multi_leaderboard_exemplar_selector_v1",),
        default="multi_leaderboard_exemplar_selector_v1",
    )
    build.add_argument("--exemplar-selection-board-capacity", type=int, default=16)
    build.add_argument("--exemplar-selection-budget-examples", type=int)
    build.add_argument("--exemplar-selection-budget-fraction", type=float)
    build.add_argument(
        "--exemplar-fulfillment-policy",
        choices=(
            "auto",
            "select_from_existing_capture",
            "rerun_selected_capture",
        ),
        default="auto",
    )
    build.add_argument("--overwrite", action="store_true")
    build.set_defaults(func=_cmd_build)

    validate = subparsers.add_parser(
        "validate",
        help="Validate a generated artifact.",
        description="Validate a generated TeacherTextbook artifact.",
    )
    validate.add_argument("--path", type=Path, required=True)
    validate.add_argument("--write-report", action="store_true")
    validate.add_argument("--metadata-sanity", action="store_true")
    validate.set_defaults(func=_cmd_validate)

    inspect = subparsers.add_parser(
        "inspect",
        help="Inspect a generated artifact.",
        description="Print a concise summary for a TeacherTextbook artifact.",
    )
    inspect.add_argument("--path", type=Path, required=True)
    inspect.add_argument("--metadata-sanity", action="store_true")
    inspect.set_defaults(func=_cmd_inspect)

    pack = subparsers.add_parser(
        "pack",
        help="Pack an unpacked Tome directory into a deterministic .rtome bundle.",
        description=(
            "Pack an unpacked Tome directory into bundle v1: a deterministic "
            "stdlib tar archive with cover_page.json at archive root."
        ),
    )
    pack.add_argument("--input", type=Path, required=True)
    pack.add_argument("--output", type=Path, required=True)
    pack.add_argument("--overwrite", action="store_true")
    pack.set_defaults(func=_cmd_pack)

    unpack = subparsers.add_parser(
        "unpack",
        help="Unpack a deterministic .rtome bundle into a directory.",
        description="Safely unpack a Tome bundle v1 deterministic tar archive.",
    )
    unpack.add_argument("--input", type=Path, required=True)
    unpack.add_argument("--output", type=Path, required=True)
    unpack.add_argument("--overwrite", action="store_true")
    unpack.set_defaults(func=_cmd_unpack)

    parity = subparsers.add_parser(
        "parity",
        help="Compare two generated Tome artifact directories.",
        description="Build a post-build parity report for two Tome artifacts.",
    )
    parity.add_argument("--left", type=Path, required=True)
    parity.add_argument("--right", type=Path, required=True)
    parity.add_argument("--output", type=Path, required=True)
    parity.add_argument("--rtol", type=float, default=1e-4)
    parity.add_argument("--atol", type=float, default=1e-5)
    parity.add_argument("--max-examples", type=int)
    parity.add_argument(
        "--compare-values",
        dest="compare_values",
        action="store_true",
        default=True,
    )
    parity.add_argument(
        "--no-compare-values",
        dest="compare_values",
        action="store_false",
    )
    parity.add_argument("--left-label", default="left")
    parity.add_argument("--right-label", default="right")
    parity.set_defaults(func=_cmd_parity)

    plan = subparsers.add_parser(
        "plan",
        help="Write a GPU run preflight plan.",
        description=(
            "Write gpu_run_plan_v1 JSON without running a production build. "
            "Auto GPU batch mode performs bounded local probing."
        ),
    )
    plan.add_argument("--teacher-backend", choices=("gpu_torch",), default="gpu_torch")
    plan.add_argument("--runtime-mode", choices=("cpu_gpu",), default="cpu_gpu")
    plan.add_argument(
        "--target-policy",
        choices=(
            "dense",
            "dense_logits",
            "topk",
            "topk_with_tail_v0",
            "cascaded",
            "cascaded_soft_labels_v1",
            "dynamic",
            "dynamic_cascaded_soft_labels_v1",
            "corridor",
            "corridor_exemplar_v1",
        ),
        default="corridor_exemplar_v1",
    )
    plan.add_argument("--teacher-model", required=True)
    plan.add_argument("--tokenizer-id")
    plan.add_argument("--dataset", type=Path, required=True)
    plan.add_argument("--corpus-manifest", type=Path)
    plan.add_argument("--teacher-model-provenance", type=Path)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--max-examples", type=int)
    plan.add_argument("--sequence-length", type=int, default=16)
    plan.add_argument("--batch-size", type=int, default=2)
    plan.add_argument("--vocab-size", type=int, default=32)
    plan.add_argument("--top-k", type=int, default=8)
    plan.add_argument("--num-buckets", type=int, default=4)
    plan.add_argument(
        "--exemplar-capture-mode",
        choices=("one_pass_candidate", "two_pass_sparse_exemplar", "auto"),
        default="one_pass_candidate",
    )
    plan.add_argument(
        "--exemplar-source-policy",
        choices=(
            "dense_logits",
            "cascaded_soft_labels_v1",
            "dynamic_cascaded_soft_labels_v1",
        ),
        default="dynamic_cascaded_soft_labels_v1",
    )
    plan.add_argument(
        "--exemplar-second-pass-source-policy",
        choices=(
            "dense_logits",
            "cascaded_soft_labels_v1",
            "dynamic_cascaded_soft_labels_v1",
        ),
        default="dynamic_cascaded_soft_labels_v1",
    )
    plan.add_argument(
        "--gpu-batch-size-mode",
        choices=("preset", "custom", "auto"),
        default="preset",
    )
    plan.add_argument("--gpu-batch-size-preset", type=int, default=8)
    plan.add_argument("--gpu-batch-size-custom", type=int)
    plan.add_argument("--gpu-batch-size-auto-min", type=int, default=1)
    plan.add_argument("--gpu-batch-size-auto-max", type=int, default=64)
    plan.add_argument("--fallback-policy", choices=("error", "auto"), default="error")
    plan.add_argument("--exemplar-selection-enabled", action="store_true")
    plan.add_argument(
        "--exemplar-fulfillment-policy",
        choices=("auto", "select_from_existing_capture", "rerun_selected_capture"),
        default="auto",
    )
    plan.add_argument("--strict-provenance", action="store_true")
    plan.add_argument("--max-artifact-bytes", type=int)
    plan.add_argument("--fail-on-warnings", action="store_true")
    plan.set_defaults(func=_cmd_plan)

    production = subparsers.add_parser(
        "production-build",
        help="Run the one-command production GPU Tome build workflow.",
        description=(
            "Validate provenance, run doctor and planner preflights, build a "
            "streaming Tome artifact, validate it, write cover_page.json, and "
            "emit production_build_report.json."
        ),
    )
    production.add_argument("--teacher-model", required=True)
    production.add_argument("--tokenizer-id")
    production.add_argument("--dataset", type=Path, required=True)
    production.add_argument("--corpus-manifest", type=Path, required=True)
    production.add_argument("--teacher-model-provenance", type=Path, required=True)
    production.add_argument("--output", type=Path, required=True)
    production.add_argument(
        "--teacher-backend",
        choices=("gpu_torch", "cpu_reference"),
        default="gpu_torch",
    )
    production.add_argument(
        "--runtime-mode",
        choices=("cpu_gpu", "cpu"),
        default="cpu_gpu",
    )
    production.add_argument(
        "--target-policy",
        choices=(
            "dense",
            "dense_logits",
            "topk",
            "topk_with_tail_v0",
            "cascaded",
            "cascaded_soft_labels_v1",
            "dynamic",
            "dynamic_cascaded_soft_labels_v1",
            "corridor",
            "corridor_exemplar_v1",
        ),
        default="corridor_exemplar_v1",
    )
    production.add_argument("--sequence-length", type=int, default=16)
    production.add_argument("--vocab-size", type=int, default=32)
    production.add_argument("--top-k", type=int, default=8)
    production.add_argument("--num-buckets", type=int, default=4)
    production.add_argument("--dynamic-top-k-min", type=int, default=1)
    production.add_argument("--dynamic-top-k-max", type=int, default=32)
    production.add_argument("--dynamic-mass-threshold", type=float, default=0.95)
    production.add_argument(
        "--gpu-batch-size-mode",
        choices=("preset", "custom", "auto"),
        default="auto",
    )
    production.add_argument("--gpu-batch-size-preset", type=int, default=8)
    production.add_argument("--gpu-batch-size-custom", type=int)
    production.add_argument("--gpu-batch-size-auto-min", type=int, default=1)
    production.add_argument("--gpu-batch-size-auto-max", type=int, default=64)
    production.add_argument("--shard-size-examples", type=int, default=1024)
    production.add_argument("--max-examples", type=int)
    production.add_argument("--resume", action="store_true")
    production.add_argument("--overwrite", action="store_true")
    production.add_argument(
        "--strict-provenance",
        dest="strict_provenance",
        action="store_true",
        default=True,
    )
    production.add_argument(
        "--no-strict-provenance",
        dest="strict_provenance",
        action="store_false",
    )
    production.add_argument("--fail-on-plan-warnings", action="store_true")
    production.add_argument("--no-build-if-plan-warn", action="store_true")
    production.add_argument("--max-artifact-bytes", type=int)
    production.add_argument("--run-plan", type=Path)
    production.add_argument("--production-report", type=Path)
    production.add_argument("--parity-left", type=Path)
    production.add_argument("--parity-report", type=Path)
    production.add_argument("--run-manifest", type=Path)
    production.add_argument("--progress-log", type=Path)
    production.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        default=True,
        help="Emit production progress lines and production_progress.json.",
    )
    production.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Disable production progress stdout and sidecar updates.",
    )
    production.add_argument(
        "--exemplar-delivery-path",
        choices=("one_pass_pruned_candidate", "two_pass_rerun_selected"),
    )
    production.add_argument("--exemplar-selection-enabled", action="store_true")
    production.add_argument("--exemplar-leaderboard-capacity", type=int, default=16)
    production.add_argument("--selected-exemplar-budget", type=int)
    production.add_argument("--selected-exemplar-fraction", type=float)
    production.add_argument(
        "--retain-unselected-exemplar-payloads",
        dest="retain_unselected_exemplar_payloads",
        action="store_true",
        default=True,
    )
    production.add_argument(
        "--no-retain-unselected-exemplar-payloads",
        dest="retain_unselected_exemplar_payloads",
        action="store_false",
    )
    production.add_argument(
        "--exemplar-score-policy",
        choices=("entropy_top_n_v1",),
        default="entropy_top_n_v1",
    )
    production.add_argument("--track-delivery-timing", action="store_true")
    production.set_defaults(func=_cmd_production_build)

    exemplar_delivery_parity = subparsers.add_parser(
        "exemplar-delivery-parity",
        help="Compare selected-only exemplar delivery artifacts.",
        description=(
            "Compare Path A and Path B selected exemplar delivery reports, "
            "leaderboards, and compressed selected payload shapes."
        ),
    )
    exemplar_delivery_parity.add_argument("--path-a", type=Path, required=True)
    exemplar_delivery_parity.add_argument("--path-b", type=Path, required=True)
    exemplar_delivery_parity.add_argument("--output", type=Path, required=True)
    exemplar_delivery_parity.add_argument(
        "--require-selection-match",
        action="store_true",
        help="Require identical selected IDs, positions, ranks, scores, and modes.",
    )
    exemplar_delivery_parity.set_defaults(func=_cmd_exemplar_delivery_parity)

    selected_linkage_audit = subparsers.add_parser(
        "audit-selected-linkage",
        help="Audit selected exemplar source-coordinate passports.",
        description=(
            "Validate selected records and payload shards against source shards, "
            "delivery-path authority, and packed corridor mode assignments."
        ),
    )
    selected_linkage_audit.add_argument("--artifact", type=Path, required=True)
    selected_linkage_audit.add_argument("--strict", action="store_true")
    selected_linkage_audit.add_argument(
        "--profile",
        choices=("full_debug_provenance", "student"),
        default="full_debug_provenance",
    )
    selected_linkage_audit.add_argument("--output", type=Path)
    selected_linkage_audit.set_defaults(func=_cmd_audit_selected_linkage)

    package_artifact = subparsers.add_parser(
        "package-artifact",
        help="Export a full-debug or student-consumable Tome package.",
    )
    package_artifact.add_argument("--input", type=Path, required=True)
    package_artifact.add_argument("--output", type=Path, required=True)
    package_artifact.add_argument(
        "--profile",
        choices=("full_debug_provenance", "student"),
        required=True,
    )
    package_artifact.add_argument("--archive", choices=("none", "tgz"), default="none")
    package_artifact.add_argument("--overwrite", action="store_true")
    package_artifact.set_defaults(func=_cmd_package_artifact)

    validate_package = subparsers.add_parser(
        "validate-package",
        help="Validate a packaged full-debug or student Tome.",
    )
    validate_package.add_argument("--artifact", type=Path, required=True)
    validate_package.add_argument(
        "--profile",
        choices=("full_debug_provenance", "student"),
    )
    validate_package.set_defaults(func=_cmd_validate_package)

    multi_gpu = subparsers.add_parser(
        "multi-gpu-path-b",
        help="Run the experimental multi-GPU Path B candidate harness.",
        description=(
            "Experimental opt-in Path B candidate scheduling. Single-GPU "
            "production-build remains the recommended production path."
        ),
    )
    multi_gpu.add_argument("--teacher-model", required=True)
    multi_gpu.add_argument("--tokenizer-id")
    multi_gpu.add_argument("--dataset", type=Path, required=True)
    multi_gpu.add_argument("--corpus-manifest", type=Path, required=True)
    multi_gpu.add_argument("--teacher-model-provenance", type=Path, required=True)
    multi_gpu.add_argument("--output", type=Path, required=True)
    multi_gpu.add_argument("--devices", required=True)
    multi_gpu.add_argument(
        "--target-policy",
        choices=("corridor", "corridor_exemplar_v1"),
        default="corridor_exemplar_v1",
    )
    multi_gpu.add_argument("--sequence-length", type=int, default=128)
    multi_gpu.add_argument("--batch-size-per-device", type=int, required=True)
    multi_gpu.add_argument("--shard-size-examples", type=int, required=True)
    multi_gpu.add_argument("--max-examples", type=int)
    multi_gpu.add_argument("--top-k", type=int, default=8)
    multi_gpu.add_argument("--num-buckets", type=int, default=4)
    multi_gpu.add_argument("--resume", action="store_true")
    multi_gpu.add_argument("--overwrite", action="store_true")
    multi_gpu.add_argument("--dry-run", action="store_true")
    multi_gpu.add_argument("--fake-workers", action="store_true")
    multi_gpu.set_defaults(func=_cmd_multi_gpu_path_b)

    prove = subparsers.add_parser(
        "prove-capabilities",
        help="Run advanced local capability diagnostics.",
        description=(
            "Run the advanced capability proof harness. This is diagnostic; "
            "normal users should start with build, validate, and inspect."
        ),
    )
    prove.add_argument("--work-dir", type=Path, required=True)
    prove.add_argument("--matrix-json", type=Path)
    prove.add_argument("--report-md", type=Path)
    prove.add_argument("--overwrite", action="store_true")
    prove.set_defaults(func=_cmd_prove_capabilities)

    corpus = subparsers.add_parser(
        "corpus",
        help="Build, inspect, and validate deterministic local corpus artifacts.",
    )
    corpus_subparsers = corpus.add_subparsers(dest="corpus_command", required=True)
    corpus_build = corpus_subparsers.add_parser(
        "build",
        help="Build a deterministic corpus artifact from local files.",
    )
    corpus_build.add_argument("--input", type=Path, action="append", required=True)
    corpus_build.add_argument("--output", type=Path, required=True)
    corpus_build.add_argument("--include", action="append", default=[])
    corpus_build.add_argument("--exclude", action="append", default=[])
    corpus_build.add_argument("--min-chars", type=int, default=1)
    corpus_build.add_argument("--max-chars", type=int, default=12_000)
    corpus_build.add_argument("--overwrite", action="store_true")
    corpus_build.set_defaults(func=_cmd_corpus_build)

    corpus_inspect = corpus_subparsers.add_parser(
        "inspect",
        help="Inspect a deterministic corpus artifact.",
    )
    corpus_inspect.add_argument("--path", type=Path, required=True)
    corpus_inspect.set_defaults(func=_cmd_corpus_inspect)

    corpus_validate = corpus_subparsers.add_parser(
        "validate",
        help="Validate a deterministic corpus artifact.",
    )
    corpus_validate.add_argument("--path", type=Path, required=True)
    corpus_validate.set_defaults(func=_cmd_corpus_validate)

    model = subparsers.add_parser(
        "model",
        help="Inspect and validate local teacher model provenance.",
    )
    model_subparsers = model.add_subparsers(dest="model_command", required=True)
    model_inspect = model_subparsers.add_parser(
        "inspect",
        help="Inspect local teacher model files and write provenance JSON.",
    )
    model_inspect.add_argument("--model-path", type=Path, required=True)
    model_inspect.add_argument("--output", type=Path, required=True)
    model_inspect.add_argument("--model-name")
    model_inspect.add_argument("--model-revision")
    model_inspect.add_argument(
        "--check",
        choices=("metadata_only",),
        default="metadata_only",
    )
    model_inspect.set_defaults(func=_cmd_model_inspect)

    model_validate = model_subparsers.add_parser(
        "validate",
        help="Validate teacher model provenance JSON.",
    )
    model_validate.add_argument("--provenance", type=Path, required=True)
    model_validate.set_defaults(func=_cmd_model_validate)

    model_discover = model_subparsers.add_parser(
        "discover",
        help="Report candidate local teacher model directories.",
    )
    model_discover.add_argument("--search-path", type=Path, required=True)
    model_discover.set_defaults(func=_cmd_model_discover)

    doctor = subparsers.add_parser(
        "doctor",
        help="Show environment and command recommendations.",
        description="Show lightweight environment status without importing HF deps.",
    )
    doctor.add_argument(
        "--teacher-backend",
        choices=("fake_numpy", "cpu_reference", "hf_torch", "gpu_torch"),
        default="cpu_reference",
    )
    doctor.add_argument("--runtime-mode", choices=("cpu", "cpu_gpu"), default="cpu")
    doctor.add_argument(
        "--target-policy",
        choices=(
            "dense",
            "dense_logits",
            "topk",
            "topk_with_tail_v0",
            "cascaded",
            "cascaded_soft_labels_v1",
            "dynamic",
            "dynamic_cascaded_soft_labels_v1",
            "corridor",
            "corridor_exemplar_v1",
        ),
        default="dense_logits",
    )
    doctor.add_argument("--teacher-model", default="fake-deterministic-teacher")
    doctor.add_argument("--tokenizer-id", default="fake-deterministic-tokenizer")
    doctor.add_argument("--sequence-length", type=int, default=16)
    doctor.add_argument("--batch-size", type=int, default=2)
    doctor.add_argument("--vocab-size", type=int, default=32)
    doctor.add_argument("--top-k", type=int, default=8)
    doctor.add_argument(
        "--exemplar-capture-mode",
        choices=("one_pass_candidate", "two_pass_sparse_exemplar", "auto"),
        default="one_pass_candidate",
    )
    doctor.add_argument(
        "--gpu-batch-size-mode",
        choices=("preset", "custom", "auto"),
        default="preset",
    )
    doctor.add_argument("--gpu-batch-size-preset", type=int, default=8)
    doctor.add_argument("--gpu-batch-size-custom", type=int)
    doctor.add_argument("--allow-downloads", action="store_true")
    doctor.add_argument("--fallback-policy", choices=("error", "auto"), default="error")
    doctor.add_argument("--exemplar-selection-enabled", action="store_true")
    doctor.add_argument(
        "--exemplar-fulfillment-policy",
        choices=("auto", "select_from_existing_capture", "rerun_selected_capture"),
        default="auto",
    )
    doctor.add_argument("--write-report", type=Path)
    doctor.set_defaults(func=_cmd_doctor)

    return parser


def _cmd_build(args: argparse.Namespace) -> int:
    from radjax_tome.builder import (
        BackendTeacherTextbookBuildConfig,
        TeacherTextbookBuildConfig,
        build_backend_teacher_textbook,
        build_teacher_textbook,
    )

    if args.teacher_backend is not None:
        config = BackendTeacherTextbookBuildConfig(
            output_dir=args.output,
            dataset_path=args.dataset,
            teacher_backend=args.teacher_backend,
            runtime_mode=args.runtime_mode,
            target_policy=_normalize_target_policy(
                args.target_policy or args.target_type
            ),
            teacher_model_id=args.teacher_model,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            max_examples=args.max_examples,
            overwrite=args.overwrite,
            vocab_size=args.vocab_size,
            top_k=args.top_k,
            exemplar_source_policy=args.exemplar_source_policy,
            exemplar_capture_mode=args.exemplar_capture_mode,
            exemplar_second_pass_source_policy=args.exemplar_second_pass_source_policy,
            gpu_batch_size_mode=args.gpu_batch_size_mode,
            gpu_batch_size_preset=args.gpu_batch_size_preset,
            gpu_batch_size_custom=args.gpu_batch_size_custom,
            gpu_batch_size_auto_min=args.gpu_batch_size_auto_min,
            gpu_batch_size_auto_max=args.gpu_batch_size_auto_max,
            streaming=args.streaming,
            resume=args.resume,
            shard_size_examples=args.shard_size_examples,
            progress_log_path=args.progress_log,
            run_manifest_path=args.run_manifest,
            fallback_policy=args.fallback_policy,
            exemplar_selector_policy=args.exemplar_selector_policy,
            exemplar_selection_enabled=args.exemplar_selection_enabled,
            exemplar_selection_board_capacity=args.exemplar_selection_board_capacity,
            exemplar_selection_budget_examples=args.exemplar_selection_budget_examples,
            exemplar_selection_budget_fraction=args.exemplar_selection_budget_fraction,
            exemplar_fulfillment_policy=args.exemplar_fulfillment_policy,
            local_files_only=True,
            allow_downloads=False,
            corpus_manifest_path=args.corpus_manifest,
            teacher_model_provenance_path=args.teacher_model_provenance,
        )
        report = build_backend_teacher_textbook(config)
        print(
            f"status={report.status} blockers={len(report.blockers)} "
            f"warnings={len(report.warnings)} output={args.output}"
        )
        return 0 if report.status == "pass" else 1

    teacher_mode = "fake" if args.teacher_mode == "synthetic" else args.teacher_mode
    config = TeacherTextbookBuildConfig(
        output_dir=args.output,
        dataset_path=args.dataset,
        teacher_mode=teacher_mode,
        teacher_model_id=args.teacher_model,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        max_examples=args.max_examples,
        overwrite=args.overwrite,
        vocab_size=args.vocab_size,
        target_type=args.target_type,
        top_k=args.top_k,
        local_files_only=teacher_mode != "hf",
        allow_downloads=False,
        corpus_manifest_path=args.corpus_manifest,
        teacher_model_provenance_path=args.teacher_model_provenance,
    )
    try:
        report = build_teacher_textbook(config)
    except ImportError as exc:
        if args.teacher_mode == "hf":
            print(
                "HF teacher mode requires optional dependency group `teacher-hf`.\n"
                "Install with: pip install 'radjax-tome[teacher-hf]'",
                file=sys.stderr,
            )
            return 2
        raise exc
    except ModuleNotFoundError as exc:
        if args.teacher_mode == "hf":
            print(
                "HF teacher mode requires optional dependency group `teacher-hf`.\n"
                "Install with: pip install 'radjax-tome[teacher-hf]'",
                file=sys.stderr,
            )
            return 2
        raise exc
    print(
        f"status={report.status} blockers={len(report.blockers)} "
        f"warnings={len(report.warnings)} output={args.output}"
    )
    return 0 if report.status == "pass" else 1


def _normalize_target_policy(value: str) -> str:
    aliases = {
        "dense": "dense_logits",
        "topk": "topk_with_tail_v0",
        "cascaded": "cascaded_soft_labels_v1",
        "dynamic": "dynamic_cascaded_soft_labels_v1",
        "corridor": "corridor_exemplar_v1",
    }
    return aliases.get(value, value)


def _cmd_validate(args: argparse.Namespace) -> int:
    from radjax_tome.builder import (
        validate_teacher_textbook,
        write_teacher_textbook_validation_report,
    )
    from radjax_tome.tome import (
        COVER_PAGE_FILENAME,
        validate_tome_bundle,
        validate_tome_cover_page,
    )
    from radjax_tome.tome.cover_page import write_cover_page

    if args.path.is_file():
        report = validate_tome_bundle(args.path)
        print(
            f"status={report.status} blockers={len(report.blockers)} "
            f"warnings={len(report.warnings)} path={args.path}"
        )
        print(
            f"bundle_format_ok={report.format_ok} "
            f"bundle_cover_page_ok={report.cover_page_ok} "
            f"bundle_contents_ok={report.contents_ok}"
        )
        return 0 if report.status == "pass" else 1

    metadata_sanity_status = "pass"
    report = validate_teacher_textbook(args.path)
    if args.write_report:
        write_teacher_textbook_validation_report(
            report,
            args.path / "validation_report.json",
        )
        if (args.path / COVER_PAGE_FILENAME).is_file():
            write_cover_page(args.path)
    cover_report = None
    if (args.path / COVER_PAGE_FILENAME).is_file():
        cover_report = validate_tome_cover_page(args.path)
    print(
        f"status={report.status} blockers={len(report.blockers)} "
        f"warnings={len(report.warnings)} path={args.path}"
    )
    if cover_report is not None:
        print(
            f"cover_page_status={cover_report.status} "
            f"cover_page_blockers={len(cover_report.blockers)}"
        )
    if args.metadata_sanity:
        metadata_sanity_report = _artifact_metadata_sanity_report(args.path)
        metadata_sanity_status = str(metadata_sanity_report["status"])
        for line in _artifact_metadata_sanity_summary(metadata_sanity_report):
            print(line)
        if args.write_report:
            _write_artifact_metadata_sanity_report(
                metadata_sanity_report,
                args.path / "metadata_sanity_report.json",
            )
    if report.status != "pass":
        return 1
    if cover_report is not None and cover_report.status != "pass":
        return 1
    if metadata_sanity_status == "fail":
        return 1
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    from radjax_tome.io.json import read_json_object
    from radjax_tome.targets import inspect_target_store
    from radjax_tome.tome import COVER_PAGE_FILENAME, inspect_tome_bundle

    if args.path.is_file():
        summary = inspect_tome_bundle(args.path)
        print("RADJAX-Tome bundle summary")
        print(f"path={args.path}")
        print(f"bundle_path={summary['bundle_path']}")
        print(f"tome_artifact_kind={summary['artifact_kind']}")
        print(f"cover_page_version={summary['cover_page_version']}")
        print(f"tome_version={summary['tome_version']}")
        print(f"layout={summary['layout']}")
        print(f"target_type={summary['target_type']}")
        print(f"num_examples={summary['num_examples']}")
        print(f"shard_count={summary['shard_count']}")
        print(f"content_count={summary['content_count']}")
        print(f"compression={summary['compression']}")
        if args.metadata_sanity:
            print("metadata_sanity=unsupported_for_bundle")
        return 0

    summary = inspect_target_store(args.path)
    cover_page_path = args.path / COVER_PAGE_FILENAME
    cover_page = read_json_object(cover_page_path) if cover_page_path.is_file() else {}
    print("RADJAX-Tome artifact summary")
    print(f"path={args.path}")
    print("artifact_type=teacher_textbook")
    if cover_page:
        print(f"tome_artifact_kind={cover_page['artifact_kind']}")
        print(f"cover_page_version={cover_page['cover_page_version']}")
        print(f"tome_version={cover_page['tome_version']}")
        print(f"layout={cover_page['layout']}")
    print(f"target_type={summary['target_type']}")
    print(f"model_id={summary['model_id']}")
    print(f"vocab_size={summary['vocab_size']}")
    print(f"sequence_length={summary['sequence_length']}")
    print(f"num_examples={summary['num_examples']}")
    print(f"shard_count={summary['shard_count']}")
    if not args.metadata_sanity:
        return 0
    metadata_sanity_report = _artifact_metadata_sanity_report(args.path)
    for line in _artifact_metadata_sanity_summary(metadata_sanity_report):
        print(line)
    return 1 if metadata_sanity_report["status"] == "fail" else 0


def _cmd_pack(args: argparse.Namespace) -> int:
    from radjax_tome.tome import pack_tome_bundle

    output = pack_tome_bundle(
        args.input,
        args.output,
        overwrite=args.overwrite,
    )
    print(f"status=pass bundle={output}")
    return 0


def _cmd_unpack(args: argparse.Namespace) -> int:
    from radjax_tome.tome import unpack_tome_bundle

    output = unpack_tome_bundle(
        args.input,
        args.output,
        overwrite=args.overwrite,
    )
    print(f"status=pass output={output}")
    return 0


def _cmd_parity(args: argparse.Namespace) -> int:
    from radjax_tome.reports import (
        TomeParityConfig,
        compare_tome_artifacts,
        write_tome_parity_report,
    )

    report = compare_tome_artifacts(
        args.left,
        args.right,
        TomeParityConfig(
            rtol=args.rtol,
            atol=args.atol,
            compare_values=args.compare_values,
            max_examples=args.max_examples,
        ),
        left_label=args.left_label,
        right_label=args.right_label,
    )
    write_tome_parity_report(report, args.output)
    print(
        f"status={report.status} blockers={len(report.blockers)} "
        f"warnings={len(report.warnings)} output={args.output}"
    )
    print(f"schema_parity={report.summary['schema_parity']}")
    print(f"array_parity={report.summary['array_parity']}")
    print(f"metadata_truth={report.summary['metadata_truth']}")
    print(f"numeric_parity={report.summary['numeric_parity']}")
    return 0 if report.status in {"pass", "warn"} else 1


def _cmd_plan(args: argparse.Namespace) -> int:
    from radjax_tome.backends import TeacherBackendConfig
    from radjax_tome.reports import (
        GPURunPlanConfig,
        build_gpu_run_plan,
        render_gpu_run_plan_summary,
        write_gpu_run_plan,
    )

    teacher_model = str(args.teacher_model)
    config = TeacherBackendConfig(
        backend_id=args.teacher_backend,
        runtime_mode=args.runtime_mode,
        target_policy=_normalize_target_policy(args.target_policy),  # type: ignore[arg-type]
        model_id=teacher_model,
        tokenizer_id=args.tokenizer_id or teacher_model,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        top_k=args.top_k,
        num_buckets=args.num_buckets,
        exemplar_source_policy=args.exemplar_source_policy,
        exemplar_capture_mode=args.exemplar_capture_mode,
        exemplar_second_pass_source_policy=args.exemplar_second_pass_source_policy,
        gpu_batch_size_mode=args.gpu_batch_size_mode,
        gpu_batch_size_preset=args.gpu_batch_size_preset,
        gpu_batch_size_custom=args.gpu_batch_size_custom,
        gpu_batch_size_auto_min=args.gpu_batch_size_auto_min,
        gpu_batch_size_auto_max=args.gpu_batch_size_auto_max,
        fallback_policy=args.fallback_policy,
        local_files_only=True,
        allow_downloads=False,
    )
    plan_config = GPURunPlanConfig(
        backend_config=config,
        dataset_path=args.dataset,
        corpus_manifest_path=args.corpus_manifest,
        teacher_model_provenance_path=args.teacher_model_provenance,
        max_examples=args.max_examples,
        exemplar_selection_enabled=args.exemplar_selection_enabled,
        exemplar_fulfillment_policy=args.exemplar_fulfillment_policy,
        strict_provenance=args.strict_provenance,
        max_artifact_bytes=args.max_artifact_bytes,
        fail_on_warnings=args.fail_on_warnings,
    )
    plan = build_gpu_run_plan(plan_config)
    write_gpu_run_plan(plan, args.output)
    for line in render_gpu_run_plan_summary(plan, args.output):
        print(line)
    return 1 if plan["status"] == "fail" else 0


def _cmd_production_build(args: argparse.Namespace) -> int:
    from radjax_tome.builder import (
        ProductionBuildConfig,
        build_production_gpu_tome,
        render_production_build_summary,
    )

    report = build_production_gpu_tome(
        ProductionBuildConfig(
            teacher_model=str(args.teacher_model),
            tokenizer_id=args.tokenizer_id,
            dataset_path=args.dataset,
            corpus_manifest_path=args.corpus_manifest,
            teacher_model_provenance_path=args.teacher_model_provenance,
            output_dir=args.output,
            teacher_backend=args.teacher_backend,
            runtime_mode=args.runtime_mode,
            target_policy=_normalize_target_policy(args.target_policy),
            sequence_length=args.sequence_length,
            vocab_size=args.vocab_size,
            top_k=args.top_k,
            num_buckets=args.num_buckets,
            dynamic_top_k_min=args.dynamic_top_k_min,
            dynamic_top_k_max=args.dynamic_top_k_max,
            dynamic_mass_threshold=args.dynamic_mass_threshold,
            gpu_batch_size_mode=args.gpu_batch_size_mode,
            gpu_batch_size_preset=args.gpu_batch_size_preset,
            gpu_batch_size_custom=args.gpu_batch_size_custom,
            gpu_batch_size_auto_min=args.gpu_batch_size_auto_min,
            gpu_batch_size_auto_max=args.gpu_batch_size_auto_max,
            shard_size_examples=args.shard_size_examples,
            max_examples=args.max_examples,
            resume=args.resume,
            overwrite=args.overwrite,
            strict_provenance=args.strict_provenance,
            fail_on_plan_warnings=args.fail_on_plan_warnings,
            no_build_if_plan_warn=args.no_build_if_plan_warn,
            max_artifact_bytes=args.max_artifact_bytes,
            run_plan_path=args.run_plan,
            production_report_path=args.production_report,
            parity_left=args.parity_left,
            parity_report_path=args.parity_report,
            run_manifest_path=args.run_manifest,
            progress_log_path=args.progress_log,
            progress=args.progress,
            exemplar_delivery_path=args.exemplar_delivery_path,
            exemplar_selection_enabled=args.exemplar_selection_enabled,
            exemplar_leaderboard_capacity=args.exemplar_leaderboard_capacity,
            selected_exemplar_budget=args.selected_exemplar_budget,
            selected_exemplar_fraction=args.selected_exemplar_fraction,
            retain_unselected_exemplar_payloads=(
                args.retain_unselected_exemplar_payloads
            ),
            exemplar_score_policy=args.exemplar_score_policy,
            track_delivery_timing=args.track_delivery_timing,
        )
    )
    for line in render_production_build_summary(report):
        print(line)
    return 0 if report["status"] in {"pass", "warn"} else 1


def _cmd_exemplar_delivery_parity(args: argparse.Namespace) -> int:
    from radjax_tome.builder import compare_exemplar_delivery_artifacts

    report = compare_exemplar_delivery_artifacts(
        args.path_a,
        args.path_b,
        output=args.output,
        require_selection_match=args.require_selection_match,
    )
    print(f"status={report['status']} path_a={args.path_a} path_b={args.path_b}")
    print(
        "selected_example_ids_match="
        f"{str(report['selected_example_ids_match']).lower()}"
    )
    print(f"selected_positions_match={str(report['selected_positions_match']).lower()}")
    return 0 if report["status"] in {"pass", "warn"} else 1


def _cmd_audit_selected_linkage(args: argparse.Namespace) -> int:
    from radjax_tome.audit import (
        audit_selected_linkage,
        write_selected_linkage_audit,
    )

    report = audit_selected_linkage(
        args.artifact,
        strict=args.strict,
        profile=args.profile,
    )
    if args.output is not None:
        write_selected_linkage_audit(report, args.output)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.status == "pass" else 1


def _cmd_package_artifact(args: argparse.Namespace) -> int:
    from radjax_tome.tome import package_tome_artifact

    result = package_tome_artifact(
        args.input,
        args.output,
        profile=args.profile,
        archive=args.archive,
        overwrite=args.overwrite,
    )
    print(
        f"status=pass package={result.output_path} profile={result.profile} "
        f"archive={result.archive}"
    )
    return 0


def _cmd_validate_package(args: argparse.Namespace) -> int:
    from radjax_tome.tome import validate_tome_package

    report = validate_tome_package(args.artifact, profile=args.profile)
    print(
        f"status={report.status} profile={report.profile} "
        f"blockers={len(report.blockers)} warnings={len(report.warnings)}"
    )
    for blocker in report.blockers:
        print(f"blocker={blocker}")
    return 0 if report.status == "pass" else 1


def _cmd_multi_gpu_path_b(args: argparse.Namespace) -> int:
    from radjax_tome.builder import (
        MultiGPUPathBConfig,
        render_multi_gpu_path_b_summary,
        run_multi_gpu_path_b_candidate_harness,
    )

    report = run_multi_gpu_path_b_candidate_harness(
        MultiGPUPathBConfig(
            teacher_model=str(args.teacher_model),
            tokenizer_id=args.tokenizer_id,
            dataset_path=args.dataset,
            corpus_manifest_path=args.corpus_manifest,
            teacher_model_provenance_path=args.teacher_model_provenance,
            output_dir=args.output,
            devices=args.devices,
            target_policy=_normalize_target_policy(args.target_policy),
            sequence_length=args.sequence_length,
            batch_size_per_device=args.batch_size_per_device,
            shard_size_examples=args.shard_size_examples,
            max_examples=args.max_examples,
            top_k=args.top_k,
            num_buckets=args.num_buckets,
            resume=args.resume,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            fake_workers=args.fake_workers,
        )
    )
    for line in render_multi_gpu_path_b_summary(report):
        print(line)
    for warning in report.get("warnings", ()):
        print(f"warning: {warning}")
    return 0 if report["status"] in {"pass", "warn"} else 1


def _cmd_prove_capabilities(args: argparse.Namespace) -> int:
    from radjax_tome.capabilities import prove_tome_generation_capabilities

    matrix_json = args.matrix_json or args.work_dir / "matrix.json"
    report_md = args.report_md or args.work_dir / "report.md"
    result = prove_tome_generation_capabilities(
        work_dir=args.work_dir,
        matrix_json=matrix_json,
        report_md=report_md,
        overwrite=args.overwrite,
    )
    print(result.status_line())
    return result.exit_code


def _cmd_corpus_build(args: argparse.Namespace) -> int:
    from radjax_tome.corpora import (
        DEFAULT_EXCLUDE_GLOBS,
        CorpusBuildConfig,
        build_corpus_artifact,
    )

    exclude_globs = tuple(args.exclude) if args.exclude else DEFAULT_EXCLUDE_GLOBS
    report = build_corpus_artifact(
        CorpusBuildConfig(
            inputs=tuple(args.input),
            output_dir=args.output,
            include_globs=tuple(args.include),
            exclude_globs=exclude_globs,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            overwrite=args.overwrite,
        )
    )
    print(
        f"status={report['status']} output={args.output} "
        f"examples={report['num_examples']} sources={report['num_sources_included']} "
        f"corpus_hash={report['corpus_hash']}"
    )
    return 1 if report["status"] == "fail" else 0


def _cmd_corpus_inspect(args: argparse.Namespace) -> int:
    from radjax_tome.corpora import inspect_corpus_artifact

    summary = inspect_corpus_artifact(args.path)
    for key in (
        "corpus_schema",
        "num_examples",
        "num_sources",
        "num_chars",
        "corpus_hash",
        "manifest_hash",
        "manifest_hash_policy",
        "normalization_policy",
        "chunking_policy",
        "deduplication_policy",
    ):
        print(f"{key}={summary[key]}")
    return 0


def _cmd_corpus_validate(args: argparse.Namespace) -> int:
    from radjax_tome.corpora import validate_corpus_artifact

    report = validate_corpus_artifact(args.path)
    print(
        f"status={report.status} blockers={len(report.blockers)} "
        f"warnings={len(report.warnings)} corpus_hash={report.corpus_hash}"
    )
    return 0 if report.status == "pass" else 1


def _cmd_model_inspect(args: argparse.Namespace) -> int:
    from radjax_tome.provenance import (
        inspect_teacher_model,
        validate_teacher_model_provenance,
        write_teacher_model_provenance,
    )

    provenance = inspect_teacher_model(
        args.model_path,
        model_name=args.model_name,
        model_revision=args.model_revision,
        check=args.check,
    )
    write_teacher_model_provenance(provenance, args.output)
    report = validate_teacher_model_provenance(args.output)
    print(
        f"status={report.status} blockers={len(report.blockers)} "
        f"warnings={len(report.warnings)} output={args.output} "
        f"model_directory_hash={report.model_directory_hash}"
    )
    return 0 if report.status in {"pass", "warn"} else 1


def _cmd_model_validate(args: argparse.Namespace) -> int:
    from radjax_tome.provenance import validate_teacher_model_provenance

    report = validate_teacher_model_provenance(args.provenance)
    print(
        f"status={report.status} blockers={len(report.blockers)} "
        f"warnings={len(report.warnings)} "
        f"model_source_kind={report.model_source_kind} "
        f"model_directory_hash={report.model_directory_hash}"
    )
    return 0 if report.status in {"pass", "warn"} else 1


def _cmd_model_discover(args: argparse.Namespace) -> int:
    from radjax_tome.provenance import discover_teacher_model_candidates

    candidates = discover_teacher_model_candidates(args.search_path)
    for candidate in candidates:
        print(
            f"candidate_path={candidate['candidate_path']} "
            f"source_kind={candidate['source_kind']} "
            f"has_config={str(candidate['has_config']).lower()} "
            f"has_tokenizer={str(candidate['has_tokenizer']).lower()} "
            f"has_weights={str(candidate['has_weights']).lower()}"
        )
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    print(f"python={platform.python_version()}")
    try:
        import radjax_tome
    except Exception as exc:  # pragma: no cover - defensive only
        print(f"radjax_tome=unavailable error={exc}")
        return 1
    print(f"radjax_tome=ok module={radjax_tome.__name__}")
    for dep in ("torch", "transformers", "jax"):
        status = "available" if importlib.util.find_spec(dep) else "unavailable"
        print(f"optional_dependency.{dep}={status}")
    print("recommended=radjax-tome build --teacher-mode fake")
    print("recommended=radjax-tome validate --path ARTIFACT")
    print("recommended=radjax-tome inspect --path ARTIFACT")

    from radjax_tome.backends import TeacherBackendConfig
    from radjax_tome.reports import (
        build_runtime_doctor_report,
        render_runtime_doctor_summary,
        write_runtime_doctor_report,
    )

    config = TeacherBackendConfig(
        backend_id=args.teacher_backend,
        runtime_mode=args.runtime_mode,
        target_policy=_normalize_target_policy(args.target_policy),  # type: ignore[arg-type]
        model_id=args.teacher_model,
        tokenizer_id=args.tokenizer_id,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        top_k=args.top_k,
        exemplar_capture_mode=args.exemplar_capture_mode,
        gpu_batch_size_mode=args.gpu_batch_size_mode,
        gpu_batch_size_preset=args.gpu_batch_size_preset,
        gpu_batch_size_custom=args.gpu_batch_size_custom,
        fallback_policy=args.fallback_policy,
        local_files_only=not args.allow_downloads,
        allow_downloads=args.allow_downloads,
    )
    report = build_runtime_doctor_report(
        config,
        exemplar_selection_enabled=args.exemplar_selection_enabled,
        exemplar_fulfillment_policy=args.exemplar_fulfillment_policy,
    )
    for line in render_runtime_doctor_summary(report):
        print(line)
    if args.write_report is not None:
        write_runtime_doctor_report(report, args.write_report)
    return 0


def _artifact_metadata_sanity_report(path: Path) -> dict[str, object]:
    from radjax_tome.reports import build_artifact_metadata_sanity_report

    return build_artifact_metadata_sanity_report(path)


def _artifact_metadata_sanity_summary(report: dict[str, object]) -> list[str]:
    from radjax_tome.reports import render_artifact_metadata_sanity_summary

    return render_artifact_metadata_sanity_summary(report)


def _write_artifact_metadata_sanity_report(
    report: dict[str, object],
    path: Path,
) -> None:
    from radjax_tome.reports import write_artifact_metadata_sanity_report

    write_artifact_metadata_sanity_report(report, path)


if __name__ == "__main__":
    raise SystemExit(main())
