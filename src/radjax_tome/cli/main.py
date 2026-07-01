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
    build.add_argument("--overwrite", action="store_true")
    build.set_defaults(func=_cmd_build)

    validate = subparsers.add_parser(
        "validate",
        help="Validate a generated artifact.",
        description="Validate a generated TeacherTextbook artifact.",
    )
    validate.add_argument("--path", type=Path, required=True)
    validate.add_argument("--write-report", action="store_true")
    validate.set_defaults(func=_cmd_validate)

    inspect = subparsers.add_parser(
        "inspect",
        help="Inspect a generated artifact.",
        description="Print a concise summary for a TeacherTextbook artifact.",
    )
    inspect.add_argument("--path", type=Path, required=True)
    inspect.set_defaults(func=_cmd_inspect)

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

    doctor = subparsers.add_parser(
        "doctor",
        help="Show environment and command recommendations.",
        description="Show lightweight environment status without importing HF deps.",
    )
    doctor.set_defaults(func=_cmd_doctor)

    return parser


def _cmd_build(args: argparse.Namespace) -> int:
    from radjax_tome.builder import TeacherTextbookBuildConfig, build_teacher_textbook

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


def _cmd_validate(args: argparse.Namespace) -> int:
    from radjax_tome.builder import (
        validate_teacher_textbook,
        write_teacher_textbook_validation_report,
    )

    report = validate_teacher_textbook(args.path)
    if args.write_report:
        write_teacher_textbook_validation_report(
            report,
            args.path / "validation_report.json",
        )
    print(
        f"status={report.status} blockers={len(report.blockers)} "
        f"warnings={len(report.warnings)} path={args.path}"
    )
    return 0 if report.status == "pass" else 1


def _cmd_inspect(args: argparse.Namespace) -> int:
    from radjax_tome.targets import inspect_target_store

    summary = inspect_target_store(args.path)
    print("RADJAX-Tome artifact summary")
    print(f"path={args.path}")
    print("artifact_type=teacher_textbook")
    print(f"target_type={summary['target_type']}")
    print(f"model_id={summary['model_id']}")
    print(f"vocab_size={summary['vocab_size']}")
    print(f"sequence_length={summary['sequence_length']}")
    print(f"num_examples={summary['num_examples']}")
    print(f"shard_count={summary['shard_count']}")
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


def _cmd_doctor(args: argparse.Namespace) -> int:
    del args
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
