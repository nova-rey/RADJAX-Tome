#!/usr/bin/env python3
"""Run private M8A selected-pass measurements from a completed Golden anchor.

This development-only driver deliberately has no package entry point.  It
creates a content-addressed post-C5 snapshot, invokes the production rerun
owner, and writes raw evidence outside the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from radjax_tome.builder.c6_integration import c5_records_for_delivery
from radjax_tome.builder.delivery.measurement import deterministic_source_sample
from radjax_tome.builder.delivery.replay import (
    ImmutablePostC5Checkpoint,
    SelectedPassMeasurementControl,
    run_selected_delivery_replay,
)
from radjax_tome.builder.production import ProductionBuildConfig
from radjax_tome.builder.production_stages.delivery import exemplar_delivery_config
from radjax_tome.fingerprint.multi_role_selection import (
    load_multi_role_selection_artifact,
)

_ROLES = {
    "score": "metadata.json",
    "corridor": "corridors/corridor_summary.json",
    "authority": "c6/authority_manifest.json",
    "c2": "c6/corridor-features/manifest.json",
    "c3": "c6/corridor-leaderboards/manifest.json",
    "c4": "c6/claims/claim_manifest.json",
    "c5": "c6/multi-role-selection/manifest.json",
    "passports": "c6/source-passports.json",
    "model": "teacher_manifest.json",
    "tokenizer": "teacher_manifest.json",
    "corpus": "input/corpus_manifest.json",
    "config": "emission_config.json",
}


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _copy(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _build_snapshot(args: argparse.Namespace) -> ImmutablePostC5Checkpoint:
    anchor = args.anchor.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    if checkpoint_root.exists():
        raise ValueError(f"checkpoint root must be fresh: {checkpoint_root}")
    if not (anchor / "production_build_report.json").is_file():
        raise ValueError("anchor is missing terminal production_build_report.json")
    report = json.loads((anchor / "production_build_report.json").read_text())
    if report.get("status") != "pass":
        raise ValueError("anchor production_build_report status is not pass")
    for member in (
        "metadata.json",
        "shards",
        "c6",
        "corridors",
        "teacher_manifest.json",
        "run_manifest.json",
        "emission_config.json",
    ):
        _copy(anchor / member, checkpoint_root / member)
    _copy(args.dataset, checkpoint_root / "input" / "corpus.jsonl")
    _copy(args.corpus_manifest, checkpoint_root / "input" / "corpus_manifest.json")
    _copy(
        args.teacher_model_provenance,
        checkpoint_root / "input" / "teacher_model_provenance.json",
    )
    role_paths = {role: checkpoint_root / relative for role, relative in _ROLES.items()}
    return ImmutablePostC5Checkpoint.capture(
        checkpoint_root,
        role_paths=role_paths,
        manifest_path=checkpoint_root.parent / "post-c5-checkpoint-manifest.json",
    )


def _environment() -> dict[str, object]:
    facts: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        import torch

        facts["torch"] = torch.__version__
        facts["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            facts["cuda_runtime"] = torch.version.cuda
            facts["gpu"] = torch.cuda.get_device_name(0)
            facts["gpu_properties"] = {
                "total_memory": int(torch.cuda.get_device_properties(0).total_memory),
                "major": int(torch.cuda.get_device_properties(0).major),
                "minor": int(torch.cuda.get_device_properties(0).minor),
            }
    except ImportError:
        facts["torch"] = "unavailable"
    return facts


def _records(checkpoint: ImmutablePostC5Checkpoint) -> list[dict[str, Any]]:
    selected = load_multi_role_selection_artifact(
        checkpoint.root / "c6" / "multi-role-selection"
    )
    return c5_records_for_delivery(selected, delivery_path="two_pass_rerun_selected")


def _sample_records(
    records: list[dict[str, Any]], sample: dict[str, object]
) -> list[dict[str, Any]]:
    keys = {tuple(item) for item in sample["sample_source_keys"]}
    return [
        record
        for record in records
        if (
            int(record["source_shard_id"]),
            int(record["source_row"]),
            str(record["selected_example_id"]),
        )
        in keys
    ]


def _run_once(
    args: argparse.Namespace,
    checkpoint: ImmutablePostC5Checkpoint,
    *,
    label: str,
    cap: int,
    records: list[dict[str, Any]],
) -> dict[str, object]:
    output = Path(tempfile.mkdtemp(prefix="radjax-m8a-selected-"))
    try:
        checkpoint.prepare_temporary_output(output)
        execution = ProductionBuildConfig(
            teacher_model=args.teacher_model,
            tokenizer_id=args.tokenizer_id,
            dataset_path=output / "input" / "corpus.jsonl",
            corpus_manifest_path=output / "input" / "corpus_manifest.json",
            teacher_model_provenance_path=(
                output / "input" / "teacher_model_provenance.json"
            ),
            output_dir=output,
            teacher_backend="gpu_torch",
            runtime_mode="cpu_gpu",
            target_policy="corridor_exemplar_v1",
            sequence_length=128,
            vocab_size=262144,
            top_k=32,
            num_buckets=4,
            dynamic_top_k_min=32,
            dynamic_top_k_max=262144,
            dynamic_mass_threshold=0.99,
            gpu_batch_size_mode="preset",
            gpu_batch_size_preset=8,
            shard_size_examples=1024,
            payload_records_per_shard=128,
            max_examples=1000,
            exemplar_delivery_path="two_pass_rerun_selected",
            exemplar_selection_enabled=True,
            selected_exemplar_budget=256,
            selected_rerun_batch_size=8,
            selection_integration_policy="corridor_first_global_backfill_v1",
            total_selected_exemplar_budget=256,
            track_delivery_timing=True,
        )
        authority = json.loads(
            (checkpoint.root / "c6" / "authority_manifest.json").read_text()
        )
        delivery = exemplar_delivery_config(
            execution,
            8,
            authoritative_records=tuple(records),
            delivery_authority_hash=str(authority["score_pass_authority_hash"]),
        )
        control = SelectedPassMeasurementControl(
            benchmark_only=True,
            effective_execution_cap=cap,
            immutable_checkpoint_digest=checkpoint.digest,
            checkpoint_root=checkpoint.root,
            temporary_output_root=output,
        )
        prepared = run_selected_delivery_replay(
            delivery, checkpoint=checkpoint, control=control
        )
        metrics = dict(delivery.rerun_metrics or {})
        diagnostics = dict(metrics["selected_pass_execution_v1"])
        if diagnostics.get("score_pass_invocation_count") != 0:
            raise AssertionError("M8A replay invoked a score pass")
        if diagnostics.get("selection_writer_invocation_count") != 0:
            raise AssertionError("M8A replay invoked a selection writer")
        if not diagnostics.get("accounting_within_five_percent"):
            raise AssertionError("selected-pass phase accounting did not reconcile")
        checkpoint.verify_unchanged()
        return {
            "label": label,
            "cap": cap,
            "record_count": len(records),
            "source_count": len({record["selected_example_id"] for record in records}),
            "authority_hash": authority["score_pass_authority_hash"],
            "record_digest": (
                "sha256:" + hashlib.sha256(_json_bytes(records)).hexdigest()
            ),
            "payload_hashes": [
                payload["payload_hash"] for payload in prepared.selected_payloads
            ],
            "metrics": metrics,
        }
    finally:
        shutil.rmtree(output, ignore_errors=True)


def _equivalence_mismatches(
    reference: dict[str, object], candidate: dict[str, object]
) -> list[str]:
    fields = ("authority_hash", "record_digest", "payload_hashes")
    return [field for field in fields if reference[field] != candidate[field]]


def _assert_equivalent(
    reference: dict[str, object], candidate: dict[str, object]
) -> None:
    mismatches = _equivalence_mismatches(reference, candidate)
    if mismatches:
        raise AssertionError(
            "non-equivalent selected payload evidence: " + ", ".join(mismatches)
        )


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
    args = parser.parse_args()

    checkpoint = _build_snapshot(args)
    all_records = _records(checkpoint)
    sample = deterministic_source_sample(all_records)
    evidence: dict[str, object] = {
        "schema_version": "m8a_selected_pass_baseline_v1",
        "environment": _environment(),
        "checkpoint": {
            "digest": checkpoint.digest,
            "manifest": str(checkpoint.manifest_path),
            "file_count": len(checkpoint.file_digests),
        },
        "sample": sample,
        "full_batch_8": [],
        "sample_caps": {},
    }
    full_runs: list[dict[str, object]] = []
    for iteration in range(1, 4):
        run = _run_once(
            args,
            checkpoint,
            label=f"full-batch-8-{iteration}",
            cap=8,
            records=_records(checkpoint),
        )
        full_runs.append(run)
    for run in full_runs[1:]:
        _assert_equivalent(full_runs[0], run)
    evidence["full_batch_8"] = full_runs
    cap_runs: dict[str, list[dict[str, object]]] = {}
    for cap in (1, 2, 4, 8):
        warmup = _run_once(
            args,
            checkpoint,
            label=f"sample-cap-{cap}-warmup",
            cap=cap,
            records=_sample_records(_records(checkpoint), sample),
        )
        measured = [
            _run_once(
                args,
                checkpoint,
                label=f"sample-cap-{cap}-measure-{iteration}",
                cap=cap,
                records=_sample_records(_records(checkpoint), sample),
            )
            for iteration in range(1, 4)
        ]
        for run in measured:
            _assert_equivalent(warmup, run)
        cap_runs[str(cap)] = [warmup, *measured]
    sample_reference = cap_runs["8"][1]
    evidence["sample_cross_cap_equivalence"] = {
        cap: {
            "exact": not _equivalence_mismatches(sample_reference, runs[1]),
            "mismatches": _equivalence_mismatches(sample_reference, runs[1]),
        }
        for cap, runs in cap_runs.items()
    }
    evidence["sample_caps"] = cap_runs
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    output = args.evidence_dir / "m8a_selected_pass_baseline.json"
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)


if __name__ == "__main__":
    main()
