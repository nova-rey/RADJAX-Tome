#!/usr/bin/env python3
"""Collect private M8B.1 cap-eight selected-staging baseline evidence.

This successor intentionally reuses the canonical M8A checkpoint/replay owner
but has its own frozen receipt schema.  It writes path-bearing raw evidence
outside the repository and has no package entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import run_m8a_selected_pass_measurements as m8a

from radjax_tome.builder.delivery.measurement import M8BStagingStatistics
from radjax_tome.builder.delivery.replay import ImmutablePostC5Checkpoint


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _initial_staging_seconds(run: dict[str, object]) -> float:
    metrics = run.get("metrics")
    if not isinstance(metrics, dict):
        raise AssertionError("M8B run has no selected-pass metrics")
    diagnostics = metrics.get("selected_pass_execution_v1")
    if not isinstance(diagnostics, dict):
        raise AssertionError("M8B run has no selected-pass diagnostics")
    phases = diagnostics.get("phases")
    if not isinstance(phases, dict):
        raise AssertionError("M8B run has no phase ledger")
    names = (
        "canonical_body_encoding_hash",
        "staging_json_encoding",
        "temporary_file_write",
        "temporary_file_close",
        "atomic_replacement",
    )
    seconds = 0.0
    for name in names:
        phase = phases.get(name)
        if not isinstance(phase, dict) or phase.get("status") != "measured_host_wall":
            raise AssertionError(f"M8B run did not measure {name}")
        value = phase.get("seconds")
        if not isinstance(value, (int, float)):
            raise AssertionError(f"M8B phase {name} has no numeric duration")
        seconds += float(value)
    return seconds


def _commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _write_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _receipt(
    *,
    checkpoint: ImmutablePostC5Checkpoint,
    runs: list[dict[str, object]],
    expected_tome_commit: str,
) -> dict[str, object]:
    statistics = M8BStagingStatistics()
    staging = [_initial_staging_seconds(run) for run in runs]
    selected_wall = [
        float(
            run["metrics"]["selected_pass_execution_v1"]["selected_pass_wall_seconds"]
        )
        for run in runs
    ]
    fractions = [
        value / wall for value, wall in zip(staging, selected_wall, strict=True)
    ]
    return {
        "schema_version": "m8b_selected_staging_baseline_v1",
        "tome_commit": expected_tome_commit,
        "statistics": statistics.receipt_projection(),
        "checkpoint": {
            "digest": checkpoint.digest,
            "file_count": len(checkpoint.file_digests),
            "manifest_sha256": _sha256(checkpoint.manifest_path),
        },
        "runs": runs,
        "initial_staging_seconds": statistics.summarize(staging),
        "selected_pass_wall_seconds": statistics.summarize(selected_wall),
        "initial_staging_fraction_of_selected_pass": statistics.summarize(fractions),
        "initial_staging_gate_passes": all(
            value >= statistics.staging_gate_fraction for value in fractions
        ),
    }


def _candidate_receipt(
    *,
    checkpoint: ImmutablePostC5Checkpoint,
    runs: list[dict[str, object]],
    expected_tome_commit: str,
    baseline_path: Path,
) -> dict[str, object]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline.get("schema_version") != "m8b_selected_staging_baseline_v1":
        raise ValueError("candidate comparison requires an M8B.1 baseline receipt")
    if baseline.get("checkpoint", {}).get("digest") != checkpoint.digest:
        raise ValueError("candidate checkpoint does not match baseline checkpoint")
    candidate = _receipt(
        checkpoint=checkpoint, runs=runs, expected_tome_commit=expected_tome_commit
    )
    statistics = M8BStagingStatistics()
    baseline_staging = float(baseline["initial_staging_seconds"]["median"])
    candidate_staging = float(candidate["initial_staging_seconds"]["median"])
    baseline_wall = float(baseline["selected_pass_wall_seconds"]["median"])
    candidate_wall = float(candidate["selected_pass_wall_seconds"]["median"])
    return {
        **candidate,
        "schema_version": "m8b_selected_staging_candidate_v1",
        "baseline_report_sha256": _sha256(baseline_path),
        "baseline_tome_commit": baseline.get("tome_commit"),
        "initial_staging_improvement_fraction": (
            (baseline_staging - candidate_staging) / baseline_staging
        ),
        "selected_pass_improvement_fraction": (
            (baseline_wall - candidate_wall) / baseline_wall
        ),
        "initial_staging_improvement_beyond_noise": statistics.improvement_beyond_noise(
            [_initial_staging_seconds(run) for run in baseline["runs"]],
            [_initial_staging_seconds(run) for run in runs],
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--tokenizer-id", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--teacher-model-provenance", type=Path, required=True)
    parser.add_argument("--expected-tome-commit", required=True)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--candidate-warmup", action="store_true")
    args = parser.parse_args()

    if _commit() != args.expected_tome_commit:
        raise ValueError("Tome HEAD does not match the M8B baseline commit")
    checkpoint = m8a._build_snapshot(args)
    if args.candidate_warmup:
        m8a._run_once(
            args,
            checkpoint,
            label="m8b-candidate-cap-eight-warmup",
            cap=8,
            records=m8a._records(checkpoint),
        )
    run_count = int(os.environ.get("M8D_RUN_COUNT", "3"))
    if run_count < 1 or run_count > 3:
        raise ValueError("M8D_RUN_COUNT must be between one and three")
    runs = [
        m8a._run_once(
            args,
            checkpoint,
            label=f"m8b-current-base-cap-eight-{iteration}",
            cap=8,
            records=m8a._records(checkpoint),
        )
        for iteration in range(1, run_count + 1)
    ]
    for run in runs[1:]:
        m8a._assert_equivalent(runs[0], run)
    if args.baseline_report is None:
        receipt = _receipt(
            checkpoint=checkpoint,
            runs=runs,
            expected_tome_commit=args.expected_tome_commit,
        )
        output_name = "m8b_selected_staging_baseline.json"
    else:
        receipt = _candidate_receipt(
            checkpoint=checkpoint,
            runs=runs,
            expected_tome_commit=args.expected_tome_commit,
            baseline_path=args.baseline_report,
        )
        output_name = "m8b_selected_staging_candidate.json"
    _write_atomic(args.evidence_dir / output_name, receipt)


if __name__ == "__main__":
    main()
