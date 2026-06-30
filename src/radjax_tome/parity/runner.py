from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from radjax_tome.parity.artifact_compare import compare_teacher_textbook_artifacts


@dataclass(frozen=True)
class AbParityCase:
    case_id: str
    target_type: str
    logits_dtype: str = "float32"
    max_examples: int = 2
    batch_size: int = 4
    sequence_length: int = 8
    vocab_size: int = 16
    dataset: str = "builtin"
    top_k: int = 4
    top_log_probs_dtype: str = "float16"
    bucket_edges: tuple[float, ...] = (1.0, 1e-3, 1e-6, 1e-9, 1e-12, 0.0)
    bucket_mass_dtype: str = "float32"
    bucket_mean_logp_dtype: str = "float32"


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    status: str
    old_command: str
    new_command: str
    old_output_dir: str
    new_output_dir: str
    sidecar_comparison: dict[str, Any] = field(default_factory=dict)
    shard_comparison: dict[str, Any] = field(default_factory=dict)
    allowed_differences: list[dict[str, Any]] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AbParitySummary:
    status: str
    case_set: str
    old_repo: str
    new_repo: str
    work_dir: str
    cases: tuple[CaseResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "case_set": self.case_set,
            "old_repo": self.old_repo,
            "new_repo": self.new_repo,
            "work_dir": self.work_dir,
            "cases": [case.to_dict() for case in self.cases],
        }


def fake_default_cases() -> tuple[AbParityCase, ...]:
    return (
        AbParityCase(
            case_id="dense_float32_builtin_single_shard",
            target_type="dense_logits",
            logits_dtype="float32",
            dataset="builtin",
            max_examples=2,
            batch_size=4,
        ),
        AbParityCase(
            case_id="dense_float16_builtin_single_shard",
            target_type="dense_logits",
            logits_dtype="float16",
            dataset="builtin",
            max_examples=2,
            batch_size=4,
        ),
        AbParityCase(
            case_id="dense_float32_jsonl_multishard",
            target_type="dense_logits",
            logits_dtype="float32",
            dataset="jsonl",
            max_examples=3,
            batch_size=2,
        ),
        AbParityCase(
            case_id="topk_k1_float32_builtin",
            target_type="topk_with_tail_v0",
            logits_dtype="float32",
            dataset="builtin",
            max_examples=2,
            batch_size=4,
            top_k=1,
        ),
        AbParityCase(
            case_id="topk_k4_float32_jsonl_multishard",
            target_type="topk_with_tail_v0",
            logits_dtype="float32",
            dataset="jsonl",
            max_examples=3,
            batch_size=2,
            top_k=4,
        ),
        AbParityCase(
            case_id="topk_k_vocab_safe_float16_builtin",
            target_type="topk_with_tail_v0",
            logits_dtype="float16",
            dataset="builtin",
            max_examples=2,
            batch_size=4,
            top_k=16,
        ),
        AbParityCase(
            case_id="cascade_default_edges_float32_builtin",
            target_type="cascaded_soft_labels_v1",
            logits_dtype="float32",
            dataset="builtin",
            max_examples=2,
            batch_size=4,
            top_k=4,
        ),
        AbParityCase(
            case_id="cascade_custom_edges_float32_jsonl_multishard",
            target_type="cascaded_soft_labels_v1",
            logits_dtype="float32",
            dataset="jsonl",
            max_examples=3,
            batch_size=2,
            top_k=4,
            bucket_edges=(1.0, 0.5, 0.05, 0.0),
        ),
        AbParityCase(
            case_id="cascade_float16_or_compressed_dtype_variant",
            target_type="cascaded_soft_labels_v1",
            logits_dtype="float16",
            dataset="builtin",
            max_examples=2,
            batch_size=4,
            top_k=4,
            top_log_probs_dtype="float16",
            bucket_mass_dtype="float16",
            bucket_mean_logp_dtype="float16",
        ),
    )


def run_ab_parity(
    *,
    old_repo: str | Path,
    new_repo: str | Path,
    work_dir: str | Path,
    case_set: str = "fake-default",
    overwrite: bool = False,
) -> AbParitySummary:
    old_root = Path(old_repo).resolve()
    new_root = Path(new_repo).resolve()
    work_root = Path(work_dir).resolve()
    if case_set != "fake-default":
        raise ValueError(f"unsupported case_set: {case_set!r}")
    if work_root.exists():
        if not overwrite:
            raise ValueError(
                f"A/B parity work dir already exists: {work_root}. "
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    results = tuple(
        _run_case(case, old_repo=old_root, new_repo=new_root, work_root=work_root)
        for case in fake_default_cases()
    )
    summary = AbParitySummary(
        status="fail" if any(case.status == "fail" for case in results) else "pass",
        case_set=case_set,
        old_repo=str(old_root),
        new_repo=str(new_root),
        work_dir=str(work_root),
        cases=results,
    )
    _write_summary_files(summary, work_root)
    return summary


def _run_case(
    case: AbParityCase,
    *,
    old_repo: Path,
    new_repo: Path,
    work_root: Path,
) -> CaseResult:
    case_root = work_root / "cases" / case.case_id
    old_output = case_root / "old"
    new_output = case_root / "new"
    case_root.mkdir(parents=True, exist_ok=True)
    dataset = _materialize_dataset(case, case_root)
    old_command = _build_command(old_repo, old_output, case, dataset)
    new_command = _build_command(new_repo, new_output, case, dataset)
    blockers: list[str] = []
    warnings: list[str] = []

    for label, repo, command in (
        ("old", old_repo, old_command),
        ("new", new_repo, new_command),
    ):
        script = repo / "scripts" / "build_teacher_textbook.py"
        if not script.is_file():
            return _write_case_report(
                case_root,
                _skip_case(
                    case,
                    old_command,
                    new_command,
                    old_output,
                    new_output,
                    f"{label} builder script not found: {script}",
                ),
            )
        result = _run_command(command, repo)
        if result.returncode != 0:
            blockers.append(
                f"{label} builder failed with exit {result.returncode}: "
                f"{_compact_process_output(result)}"
            )
            return _write_case_report(
                case_root,
                _fail_case(
                    case,
                    old_command,
                    new_command,
                    old_output,
                    new_output,
                    blockers,
                    warnings,
                ),
            )
        validate = _validate_command(repo, old_output if label == "old" else new_output)
        if validate is None:
            warnings.append(f"{label} validator script not found")
            continue
        validation = _run_command(validate, repo)
        if validation.returncode != 0:
            blockers.append(
                f"{label} validator failed with exit {validation.returncode}: "
                f"{_compact_process_output(validation)}"
            )
            return _write_case_report(
                case_root,
                _fail_case(
                    case,
                    old_command,
                    new_command,
                    old_output,
                    new_output,
                    blockers,
                    warnings,
                ),
            )

    comparison = compare_teacher_textbook_artifacts(old_output, new_output)
    result = CaseResult(
        case_id=case.case_id,
        status=comparison.status,
        old_command=_format_command(old_command),
        new_command=_format_command(new_command),
        old_output_dir=str(old_output),
        new_output_dir=str(new_output),
        sidecar_comparison=comparison.sidecar_comparison,
        shard_comparison=comparison.shard_comparison,
        allowed_differences=[item.to_dict() for item in comparison.allowed_differences],
        blockers=[*blockers, *comparison.blockers],
        warnings=[*warnings, *comparison.warnings],
    )
    return _write_case_report(case_root, result)


def _skip_case(
    case: AbParityCase,
    old_command: list[str],
    new_command: list[str],
    old_output: Path,
    new_output: Path,
    reason: str,
) -> CaseResult:
    return CaseResult(
        case_id=case.case_id,
        status="skip",
        old_command=_format_command(old_command),
        new_command=_format_command(new_command),
        old_output_dir=str(old_output),
        new_output_dir=str(new_output),
        blockers=[],
        warnings=[reason],
    )


def _fail_case(
    case: AbParityCase,
    old_command: list[str],
    new_command: list[str],
    old_output: Path,
    new_output: Path,
    blockers: list[str],
    warnings: list[str],
) -> CaseResult:
    return CaseResult(
        case_id=case.case_id,
        status="fail",
        old_command=_format_command(old_command),
        new_command=_format_command(new_command),
        old_output_dir=str(old_output),
        new_output_dir=str(new_output),
        blockers=blockers,
        warnings=warnings,
    )


def _build_command(
    repo: Path,
    output: Path,
    case: AbParityCase,
    dataset: Path | None,
) -> list[str]:
    command = [
        sys.executable,
        str(repo / "scripts" / "build_teacher_textbook.py"),
        "--output",
        str(output),
        "--teacher-mode",
        "fake",
        "--sequence-length",
        str(case.sequence_length),
        "--batch-size",
        str(case.batch_size),
        "--max-examples",
        str(case.max_examples),
        "--logits-dtype",
        case.logits_dtype,
        "--target-type",
        case.target_type,
        "--top-k",
        str(case.top_k),
        "--top-log-probs-dtype",
        case.top_log_probs_dtype,
        "--bucket-edges",
        ",".join(f"{edge:.12g}" for edge in case.bucket_edges),
        "--bucket-mass-dtype",
        case.bucket_mass_dtype,
        "--bucket-mean-logp-dtype",
        case.bucket_mean_logp_dtype,
        "--vocab-size",
        str(case.vocab_size),
        "--seed",
        "0",
        "--overwrite",
    ]
    if dataset is not None:
        command.extend(["--dataset", str(dataset)])
    return command


def _validate_command(repo: Path, output: Path) -> list[str] | None:
    script = repo / "scripts" / "validate_teacher_textbook.py"
    if not script.is_file():
        return None
    return [
        sys.executable,
        str(script),
        "--path",
        str(output),
        "--write-report",
    ]


def _run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    existing_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath_parts = [str(cwd / "src")]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(pythonpath_parts),
    }
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _materialize_dataset(case: AbParityCase, case_root: Path) -> Path | None:
    if case.dataset == "builtin":
        return None
    if case.dataset != "jsonl":
        raise ValueError(f"unsupported dataset fixture: {case.dataset!r}")
    path = case_root / "dataset.jsonl"
    rows = (
        {"example_id": "ab-alpha", "text": "alpha parity example"},
        {"example_id": "ab-beta", "text": "beta parity example"},
        {"example_id": "ab-gamma", "text": "gamma parity example"},
    )
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _write_summary_files(summary: AbParitySummary, work_root: Path) -> None:
    payload = summary.to_dict()
    (work_root / "ab_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# TeacherTextbook A/B Parity Summary",
        "",
        f"- status: `{summary.status}`",
        f"- case_set: `{summary.case_set}`",
        f"- old_repo: `{summary.old_repo}`",
        f"- new_repo: `{summary.new_repo}`",
        "",
        "| case | status | blockers | warnings |",
        "| --- | --- | ---: | ---: |",
    ]
    for case in summary.cases:
        lines.append(
            f"| `{case.case_id}` | `{case.status}` | "
            f"{len(case.blockers)} | {len(case.warnings)} |"
        )
    (work_root / "ab_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_case_report(case_root: Path, result: CaseResult) -> CaseResult:
    (case_root / "report.json").write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _compact_process_output(result: subprocess.CompletedProcess[str]) -> str:
    text = "\n".join(part for part in (result.stdout, result.stderr) if part)
    return text.strip()[-2000:]


def _format_command(command: list[str]) -> str:
    return " ".join(_shell_quote(part) for part in command)


def _shell_quote(part: str) -> str:
    if not part:
        return "''"
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+-=.,/:")
    if all(char in safe for char in part):
        return part
    return "'" + part.replace("'", "'\"'\"'") + "'"
