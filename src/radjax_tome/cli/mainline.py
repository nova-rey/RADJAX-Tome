"""Opinionated six-command public CLI."""

from __future__ import annotations

import argparse
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
        description="Opinionated RADJAX-Tome production lifecycle CLI",
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
        if args.command == "build":
            intent = load_tome_build_intent(args.config)
            overrides = {"output_dir": args.output} if args.output else {}
            if overrides:
                intent = apply_production_advanced_overrides(intent, overrides)
            assessment = assess_production_preflight(
                intent.outputs.output_dir, resume=args.resume, overwrite=args.overwrite
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
            intent = intent.__class__(
                **{
                    **intent.__dict__,
                    "execution": intent.execution.__class__(
                        **{
                            **intent.execution.__dict__,
                            "resume": args.resume,
                            "overwrite": args.overwrite,
                        }
                    ),
                }
            )
            from radjax_tome.builder.production import build_production_gpu_tome

            resolved = resolve_tome_build_intent(intent, source="m9_cli")
            report = build_production_gpu_tome(
                production_build_config_from_resolved(resolved)
            )
            result = CLIResult(
                "build",
                report.get("status", "fail"),
                0 if report.get("status") in {"pass", "warn"} else 7,
                artifact={"workspace": str(intent.outputs.output_dir)},
                reports={"production": report},
            )
            if result.exit_code == 0:
                receipt = intent.outputs.output_dir / "m9_cli_receipt.json"
                receipt.parent.mkdir(parents=True, exist_ok=True)
                receipt.write_text(
                    json.dumps(result.to_dict(), sort_keys=True, default=str) + "\n"
                )
                result.artifact["receipt_path"] = str(receipt)
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
            receipt.write_text(
                json.dumps(cli_result.to_dict(), sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
            cli_result.artifact["receipt_path"] = str(receipt)
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
    args = parser().parse_args(argv)
    result = run(args)
    emit(result, machine=args.machine, quiet=args.quiet)
    return result.exit_code
