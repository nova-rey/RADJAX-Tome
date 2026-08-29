"""Opinionated six-command public CLI."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

from radjax_tome.builder.config import (
    apply_production_advanced_overrides,
    production_build_config_from_resolved,
    resolve_tome_build_intent,
)
from radjax_tome.builder.config_io import load_tome_build_intent
from radjax_tome.builder.production_stages.preflight import assess_production_preflight
from radjax_tome.cli.models import CLIError, CLIResult
from radjax_tome.cli.rendering import emit


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="radjax-tome",
        description=(
            "RADJAX-Tome produces teacher-side distillation artifacts. "
            "Opinionated production lifecycle CLI.\n\n"
            "Recommended commands: build, validate, inspect, package, doctor, research"
            "\nLegacy-compatible research commands: "
            "build-fingerprint-corridor-leaderboards, "
            "allocate-fingerprint-corridor-coverage, "
            "claim-corridor-and-backfill-global, build-multi-role-selected-exemplars, "
            "pack, unpack, prove-capabilities"
        ),
    )
    root.add_argument("--json", action="store_true", dest="machine")
    root.add_argument("--quiet", action="store_true")
    root.add_argument("--debug", action="store_true")
    root.add_argument("--no-color", action="store_true")
    root.add_argument("--version", action="version", version="radjax-tome 0.1.0")
    commands = root.add_subparsers(dest="command", required=True)
    build = commands.add_parser(
        "build", help="Build from a complete canonical M5 config"
    )
    build.add_argument("--config", type=Path, required=True)
    build.add_argument(
        "--output", type=Path, help="Canonical output_dir operational override"
    )
    build.add_argument("--resume", action="store_true")
    build.add_argument("--overwrite", action="store_true")
    build.add_argument("--preflight-only", action="store_true")
    validate = commands.add_parser(
        "validate", help="Validate a promised production artifact"
    )
    validate.add_argument("artifact", type=Path)
    validate.add_argument(
        "--mode",
        choices=("standard", "governed", "external-attestation"),
        default="standard",
    )
    validate.add_argument("--expected", type=Path)
    validate.add_argument("--attestation", type=Path)
    validate.add_argument(
        "--attestation-policy", choices=("optional", "required"), default="optional"
    )
    validate.add_argument("--evaluation-time")
    inspect = commands.add_parser("inspect", help="Validate and inspect an artifact")
    inspect.add_argument("artifact", type=Path)
    package = commands.add_parser(
        "package", help="Project a producer workspace into a package"
    )
    package.add_argument("workspace", type=Path)
    package.add_argument("--output", type=Path, required=True)
    package.add_argument(
        "--profile", choices=("student", "full_debug_provenance"), required=True
    )
    package.add_argument("--transport", choices=("tgz", "directory"), default="tgz")
    package.add_argument(
        "--student-contract-profile", choices=("v5", "v6"), default="v5"
    )
    package.add_argument("--overwrite", action="store_true")
    doctor = commands.add_parser("doctor", help="Report runtime capability")
    doctor.add_argument("--config", type=Path)
    commands.add_parser(
        "research", help="Access retained engineering commands"
    ).add_argument("args", nargs=argparse.REMAINDER)
    return root


def _error(
    command: str, code: str, message: str, exit_code: int, *, repair: str | None = None
) -> CLIResult:
    return CLIResult(
        command,
        "fail",
        exit_code,
        error=CLIError(code, message, "preflight", repair=repair),
    )


def run(args: argparse.Namespace) -> CLIResult:
    try:
        if args.command == "corpus":
            from dataclasses import replace

            from radjax_tome.corpora import (
                build_corpus_artifact_v2,
                inspect_corpus_artifact_v2,
                load_corpus_build_intent,
                validate_corpus_artifact_v2,
            )

            if args.corpus_command == "build":
                intent = load_corpus_build_intent(args.config)
                if args.resume and args.overwrite:
                    return _error(
                        "corpus build",
                        "INVALID_CONFIGURATION",
                        "--resume and --overwrite are mutually exclusive",
                        2,
                    )
                execution = dict(intent.execution)
                execution["resume"] = args.resume or execution.get("resume", False)
                execution["overwrite"] = args.overwrite or execution.get(
                    "overwrite", False
                )
                if execution["resume"] and execution["overwrite"]:
                    return _error(
                        "corpus build",
                        "INVALID_CONFIGURATION",
                        "--resume and --overwrite are mutually exclusive",
                        2,
                    )
                if execution != dict(intent.execution):
                    intent = replace(intent, execution=execution)
                report = build_corpus_artifact_v2(intent)
                return CLIResult("corpus build", "pass", 0, reports=report)
            if args.corpus_command == "validate":
                result = validate_corpus_artifact_v2(args.artifact)
                return CLIResult(
                    "corpus validate",
                    result.status,
                    0 if result.ok else 4,
                    reports=result.to_dict(),
                )
            item = inspect_corpus_artifact_v2(args.artifact)
            return CLIResult("corpus inspect", "pass", 0, reports=item.to_dict())
        if args.command == "build":
            intent = load_tome_build_intent(args.config)
            overrides = {}
            if args.output:
                overrides["output_dir"] = args.output
            if args.resume:
                overrides["resume"] = True
            if args.overwrite:
                overrides["overwrite"] = True
            if overrides:
                intent = apply_production_advanced_overrides(intent, overrides)
            resolved = resolve_tome_build_intent(intent, source="m9_cli")
            production = production_build_config_from_resolved(resolved)
            if (
                intent.package.profile != "unpacked"
                or intent.package.transport != "directory"
            ):
                return _error(
                    "build",
                    "PACKAGE_PROJECTION_UNSUPPORTED",
                    "public build produces the canonical unpacked directory only; "
                    "package profile/transport must use the package command",
                    3,
                    repair=(
                        "set package.profile=unpacked and package.transport=directory, "
                        "or run 'radjax-tome package'"
                    ),
                )
            if not args.preflight_only:
                from radjax_tome.builder.production_stages.preflight import (
                    validate_required_inputs,
                )

                blockers: list[str] = []
                validate_required_inputs(production, blockers)
                if blockers:
                    return _error(
                        "build",
                        "M5_CONFIG_INVALID",
                        "; ".join(blockers),
                        2,
                        repair=(
                            "correct the canonical config inputs and rerun preflight"
                        ),
                    )
            assessment = assess_production_preflight(
                intent.outputs.output_dir,
                # A preflight-only invocation is also the documented config
                # projection/destination dry run; input artifacts are checked
                # by the production preflight before any real build.
                config=None if args.preflight_only else production,
                resume=args.resume,
                overwrite=args.overwrite,
            )
            if assessment.status != "pass":
                return _error(
                    "build",
                    "OUTPUT_CONFLICT",
                    "; ".join(assessment.blockers),
                    5,
                    repair="choose a safe destination or use --resume/--overwrite",
                )
            if args.preflight_only:
                return CLIResult(
                    "build",
                    "pass",
                    0,
                    artifact={
                        "workspace": str(assessment.destination),
                        "action": assessment.action,
                    },
                )
            from radjax_tome.builder.production import build_production_gpu_tome

            with contextlib.redirect_stdout(sys.stderr):
                report = build_production_gpu_tome(production)
            result = CLIResult(
                "build",
                report.get("status", "fail"),
                0 if report.get("status") in {"pass", "warn"} else 7,
                artifact={"workspace": str(intent.outputs.output_dir)},
                reports={"production": report},
                config={
                    "schema_version": "radjax_tome_build_intent_v1",
                    "selection_authority_hash": resolved.selection_authority_hash,
                },
            )
            if result.exit_code == 0:
                receipt = intent.outputs.output_dir / "m9_cli_receipt.json"
                receipt.parent.mkdir(parents=True, exist_ok=True)
                result.artifact["receipt_path"] = str(receipt)
                result.receipt_path = str(receipt)
                receipt.write_text(
                    json.dumps(result.to_dict(), sort_keys=True, default=str) + "\n"
                )
            return result
        if args.command == "validate":
            from radjax_tome.tome.artifact_dispatch import validate_artifact

            report = validate_artifact(
                args.artifact,
                mode=args.mode,
                expected=args.expected,
                attestation=args.attestation,
                attestation_policy=args.attestation_policy,
                evaluation_time=args.evaluation_time,
            )
            code = 0 if report["status"] == "pass" else 4
            return CLIResult(
                "validate",
                report["status"],
                code,
                artifact={"input": str(args.artifact), "format": report["kind"]},
                reports=report,
            )
        if args.command == "inspect":
            from radjax_tome.tome.artifact_dispatch import inspect_artifact

            item = inspect_artifact(args.artifact)
            return CLIResult(
                "inspect",
                "pass",
                0,
                artifact={"input": str(item.path), "format": item.kind},
                reports={"validation": item.validation, "metadata": item.metadata},
            )
        if args.command == "package":
            from radjax_tome.tome.packaging import package_tome_artifact

            archive = "tgz" if args.transport == "tgz" else "none"
            result = package_tome_artifact(
                args.workspace,
                args.output,
                profile=args.profile,
                archive=archive,
                overwrite=args.overwrite,
                student_contract_profile=args.student_contract_profile,
            )
            cli_result = CLIResult(
                "package",
                "pass",
                0,
                artifact={
                    "output": str(result.output_path),
                    "profile": args.profile,
                    "transport": args.transport,
                },
            )
            receipt = args.output.with_name(args.output.name + ".m9_receipt.json")
            cli_result.artifact["receipt_path"] = str(receipt)
            cli_result.receipt_path = str(receipt)
            receipt.write_text(
                json.dumps(cli_result.to_dict(), sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
            return cli_result
        if args.command == "doctor":
            from radjax_tome.backends import TeacherBackendConfig
            from radjax_tome.builder.production_stages.delivery import backend_config
            from radjax_tome.builder.production_stages.preflight import (
                validate_required_inputs,
            )
            from radjax_tome.reports import build_runtime_doctor_report

            if args.config:
                intent = load_tome_build_intent(args.config)
                resolved = resolve_tome_build_intent(intent, source="m9_doctor")
                production = production_build_config_from_resolved(resolved)
                blockers: list[str] = []
                validate_required_inputs(production, blockers)
                runtime = build_runtime_doctor_report(
                    backend_config(production),
                    exemplar_selection_enabled=production.exemplar_selection_enabled,
                )
                status = "pass" if not blockers else "fail"
                return CLIResult(
                    "doctor",
                    status,
                    0 if status == "pass" else 2,
                    reports={
                        "config": str(args.config),
                        "preflight": {"status": status, "blockers": blockers},
                        "runtime": runtime,
                    },
                )
            runtime = build_runtime_doctor_report(
                TeacherBackendConfig(backend_id="cpu_reference", runtime_mode="cpu")
            )
            return CLIResult(
                "doctor",
                "pass",
                0,
                reports={
                    "python": sys.version,
                    "radjax_tome": "ok",
                    "runtime": runtime,
                },
            )
        return CLIResult(
            "research",
            "fail",
            2,
            error=CLIError(
                "RESEARCH_ROUTING",
                "use a retained command after research",
                "invocation",
            ),
        )
    except KeyboardInterrupt:
        return _error(args.command, "INTERRUPTED", "command interrupted", 130)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        message = str(exc)
        if args.command == "validate":
            code = 3 if "unsupported" in message else 4
            error_code = "UNSUPPORTED_ARTIFACT" if code == 3 else "VALIDATION_FAILED"
        elif args.command == "package" and "already exists" in message:
            code, error_code = 5, "OUTPUT_CONFLICT"
        else:
            code, error_code = 2, "COMMAND_FAILED"
        return _error(args.command, error_code, message, code)


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if "corpus" in raw:
        return _main_corpus(raw)
    args = parser().parse_args(argv)
    result = run(args)
    try:
        emit(result, machine=args.machine, quiet=args.quiet)
    except BrokenPipeError:
        return 141
    return result.exit_code


def _corpus_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="radjax-tome")
    root.add_argument("--json", action="store_true", dest="machine")
    root.add_argument("--quiet", action="store_true")
    commands = root.add_subparsers(dest="command", required=True)
    corpus = commands.add_parser("corpus")
    children = corpus.add_subparsers(dest="corpus_command", required=True)
    build = children.add_parser("build")
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--resume", action="store_true")
    build.add_argument("--overwrite", action="store_true")
    validate = children.add_parser("validate")
    validate.add_argument("artifact", type=Path)
    inspect = children.add_parser("inspect")
    inspect.add_argument("artifact", type=Path)
    return root


def _main_corpus(raw: list[str]) -> int:
    args = _corpus_parser().parse_args(raw)
    from dataclasses import replace

    from radjax_tome.corpora import (
        build_corpus_artifact_v2,
        inspect_corpus_artifact_v2,
        load_corpus_build_intent,
        validate_corpus_artifact_v2,
    )

    try:
        if args.corpus_command == "build":
            intent = load_corpus_build_intent(args.config)
            if args.resume and args.overwrite:
                raise ValueError("--resume and --overwrite are mutually exclusive")
            execution = dict(intent.execution)
            execution["resume"] = args.resume or execution.get("resume", False)
            execution["overwrite"] = args.overwrite or execution.get("overwrite", False)
            if execution != dict(intent.execution):
                intent = replace(intent, execution=execution)
            result = CLIResult(
                "corpus build", "pass", 0, reports=build_corpus_artifact_v2(intent)
            )
        elif args.corpus_command == "validate":
            report = validate_corpus_artifact_v2(args.artifact)
            result = CLIResult(
                "corpus validate",
                report.status,
                0 if report.ok else 4,
                reports=report.to_dict(),
            )
        else:
            result = CLIResult(
                "corpus inspect",
                "pass",
                0,
                reports=inspect_corpus_artifact_v2(args.artifact).to_dict(),
            )
    except KeyboardInterrupt:
        result = _error("corpus", "INTERRUPTED", "command interrupted", 130)
    except (OSError, TypeError, ValueError, KeyError) as exc:
        result = _error("corpus", "COMMAND_FAILED", str(exc), 2)
    try:
        emit(result, machine=args.machine, quiet=args.quiet)
    except BrokenPipeError:
        return 141
    return result.exit_code
