from __future__ import annotations

import argparse
import importlib.util
import platform
import sys
from pathlib import Path

HELP_DESCRIPTION = """RADJAX-Tome produces teacher-side distillation artifacts.

Recommended commands:
  build
  validate
  inspect
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
