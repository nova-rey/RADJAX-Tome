"""M4C adversarial matrix for read-only native Path-B resume resolution."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from radjax_tome.builder.native_path_b.api import (
    CANONICAL_DELIVERY_PATH,
    CANONICAL_SELECTION_INTEGRATION_POLICY,
    CANONICAL_TARGET_POLICY,
    resolve_canonical_path_b_config,
)
from radjax_tome.builder.native_path_b.resume import (
    NativePathBResumeResolution,
    resolve_native_path_b_resume,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_preflight(output_dir: Path) -> None:
    _write_json(output_dir / "run_plan.json", {"status": "pass"})


def _write_complete_score_pass(output_dir: Path) -> None:
    _write_preflight(output_dir)
    _write_json(
        output_dir / "run_manifest.json",
        {
            "status": "complete",
            "num_examples_completed": 2,
            "num_examples_planned": 2,
            "num_shards_completed": 1,
            "num_shards_planned": 1,
        },
    )


def _write_corridors(
    output_dir: Path,
    *,
    selected_count: int,
    selected_linked: bool,
) -> None:
    corridors = output_dir / "corridors"
    _write_json(
        corridors / "corridor_summary.json",
        {
            "corridor_artifact_built": True,
            "corridor_modes_built": True,
            "corridor_observation_basis": "full_token_position_corridor",
            "degraded_corridor_export": False,
            "corridor_positions_available": 2,
            "corridor_positions_used": 2,
            "fingerprint_count": 1,
            "mode_count": 1,
            "corridor_assignment_count": 2,
            "selected_exemplar_count": selected_count,
            "selected_exemplars_linked_to_corridor_modes": selected_linked,
        },
    )
    _write_json(corridors / "corridor_fingerprints.json", {"fingerprint_count": 1})
    _write_json(corridors / "corridor_modes.json", {"mode_count": 1})
    _write_json(corridors / "mode_assignments.json", {"num_assignments": 2})


def _write_selection_evidence(
    output_dir: Path, *, authority_hash: str = "sha256:a"
) -> None:
    global_supply = output_dir / "c6" / "global_board_supply.json"
    passports = output_dir / "c6" / "source_passports.json"
    _write_json(global_supply, {})
    _write_json(passports, {})
    _write_json(
        output_dir / "c6" / "authority_manifest.json",
        {
            "score_pass_authority_hash": authority_hash,
            "paths": {
                "global_board_supply": str(global_supply.relative_to(output_dir)),
                "source_passports": str(passports.relative_to(output_dir)),
            },
        },
    )
    _write_json(output_dir / "c6" / "coverage-plan" / "coverage_plan.json", {})
    _write_json(output_dir / "c6" / "claims" / "claim_manifest.json", {})
    _write_json(output_dir / "c6" / "multi-role-selection" / "manifest.json", {})


def _write_delivery_and_final_corridor(output_dir: Path) -> None:
    _write_selection_evidence(output_dir)
    _write_json(
        output_dir / "delivery_report.json",
        {
            "status": "pass",
            "num_selected_exemplars": 2,
            "delivery_authority_hash": "sha256:a",
        },
    )
    _write_corridors(output_dir, selected_count=2, selected_linked=True)


def _write_finalization(output_dir: Path) -> None:
    _write_json(output_dir / "validation_report.json", {"status": "pass"})
    _write_json(
        output_dir / "reports" / "c6_integrated_selection_validation.json",
        {"status": "pass"},
    )


def _write_complete_run(output_dir: Path) -> None:
    _write_complete_score_pass(output_dir)
    _write_delivery_and_final_corridor(output_dir)
    _write_finalization(output_dir)
    _write_json(output_dir / "production_build_report.json", {"status": "pass"})


def _assert_pending(
    resolution: NativePathBResumeResolution,
    *,
    stage: str,
    reason: str,
) -> None:
    assert resolution.complete is False
    assert resolution.stage == stage
    assert resolution.failure is not None
    assert resolution.failure.stage == stage
    assert resolution.failure.reason == reason
    assert resolution.failure.resumable is True
    assert resolution.failure.blockers


def test_resume_reader_import_isolates_optional_ml_and_research_modules() -> None:
    command = (
        "import json, sys\n"
        "import radjax_tome.builder.native_path_b.resume\n"
        "print(json.dumps(sorted(name for name in sys.modules if "
        "name == 'torch' or name.startswith(('torch.', 'transformers', "
        "'radjax_tome.backends.hf_export', 'radjax_tome.backends.hf_specimen', "
        "'radjax_tome.backends.qwen_policy', "
        "'radjax_tome.builder.multi_gpu_path_b')))))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=REPOSITORY_ROOT,
        env={"PYTHONPATH": str(REPOSITORY_ROOT / "src")},
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == []


def test_resume_resolver_starts_at_preflight_when_fresh_then_score_for_partial_output(
    tmp_path: Path,
) -> None:
    fresh = resolve_native_path_b_resume(tmp_path / "fresh")
    assert fresh.complete is False
    assert fresh.stage == "preflight"
    assert fresh.failure is not None
    assert fresh.failure.stage == "preflight"
    assert fresh.failure.resumable is True
    assert fresh.failure.blockers

    partial_output = tmp_path / "partial"
    _write_preflight(partial_output)
    _write_json(
        partial_output / "run_manifest.json",
        {
            "status": "running",
            "num_examples_completed": 1,
            "num_examples_planned": 2,
            "num_shards_completed": 0,
            "num_shards_planned": 1,
        },
    )
    partial = resolve_native_path_b_resume(partial_output)
    _assert_pending(partial, stage="score_pass", reason="score_pass_incomplete")


def test_resume_resolver_preserves_provisional_early_corridor_as_non_final(
    tmp_path: Path,
) -> None:
    _write_complete_score_pass(tmp_path)
    _write_corridors(tmp_path, selected_count=0, selected_linked=False)

    resolution = resolve_native_path_b_resume(tmp_path)

    _assert_pending(
        resolution,
        stage="fingerprint_corridor_selection_authority_export",
        reason="selection_authority_evidence_unavailable",
    )
    assert resolution.evidence is not None
    assert resolution.evidence.stage == "score_surface_corridor_materialization"


def test_resume_resolver_marks_selection_complete_but_delivery_pending(
    tmp_path: Path,
) -> None:
    _write_complete_score_pass(tmp_path)
    _write_corridors(tmp_path, selected_count=0, selected_linked=False)
    _write_selection_evidence(tmp_path)

    resolution = resolve_native_path_b_resume(tmp_path)

    _assert_pending(
        resolution,
        stage="selected_delivery_rerun",
        reason="selected_delivery_pending",
    )


def test_resume_resolver_identifies_finalization_only_after_late_corridor(
    tmp_path: Path,
) -> None:
    _write_complete_score_pass(tmp_path)
    _write_delivery_and_final_corridor(tmp_path)

    resolution = resolve_native_path_b_resume(tmp_path)

    _assert_pending(
        resolution,
        stage="validation_linkage",
        reason="validation_evidence_unavailable",
    )
    assert resolution.evidence is not None
    assert resolution.evidence.stage == "selected_artifact_corridor_finalization"


def test_resume_resolver_recognizes_an_already_complete_run(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)

    resolution = resolve_native_path_b_resume(tmp_path)

    assert resolution.complete is True
    assert resolution.stage is None
    assert resolution.failure is None
    assert resolution.evidence is not None
    assert resolution.evidence.stage == "final_reporting"


@pytest.mark.parametrize(
    ("mutate", "stage", "reason"),
    (
        (
            lambda output: _write_json(
                output / "delivery_report.json",
                {
                    "status": "pass",
                    "num_selected_exemplars": 2,
                    "delivery_authority_hash": "sha256:stale",
                },
            ),
            "selected_artifact_corridor_finalization",
            "selected_delivery_authority_mismatch",
        ),
        (
            lambda output: (output / "corridors" / "corridor_modes.json").unlink(),
            "selected_artifact_corridor_finalization",
            "corridor_evidence_unavailable",
        ),
        (
            lambda output: (output / "validation_report.json").write_text(
                "{invalid-json", encoding="utf-8"
            ),
            "validation_linkage",
            "validation_evidence_unavailable",
        ),
    ),
    ids=("stale-authority", "missing-corridor-evidence", "corrupt-validation"),
)
def test_resume_resolver_fails_closed_at_the_earliest_invalid_stage(
    tmp_path: Path,
    mutate: Any,
    stage: str,
    reason: str,
) -> None:
    _write_complete_run(tmp_path)
    mutate(tmp_path)

    resolution = resolve_native_path_b_resume(tmp_path)

    _assert_pending(resolution, stage=stage, reason=reason)


def test_resume_resolver_rejects_config_bound_authority_hash_drift(
    tmp_path: Path,
) -> None:
    _write_complete_score_pass(tmp_path)
    _write_corridors(tmp_path, selected_count=0, selected_linked=False)
    _write_selection_evidence(tmp_path)
    _write_json(
        tmp_path / "emission_config.json",
        {"selection_integration_config_hash": "sha256:stale"},
    )
    _write_json(
        tmp_path / "c6" / "authority_manifest.json",
        {
            "score_pass_authority_hash": "sha256:a",
            "selection_integration_config_hash": "sha256:stale",
        },
    )
    source = SimpleNamespace(
        target_policy=CANONICAL_TARGET_POLICY,
        selection_integration_policy=CANONICAL_SELECTION_INTEGRATION_POLICY,
        exemplar_selection_enabled=True,
        exemplar_delivery_path=CANONICAL_DELIVERY_PATH,
        teacher_model="teacher",
        tokenizer_id="tokenizer",
        dataset_path=tmp_path / "dataset.jsonl",
        corpus_manifest_path=tmp_path / "corpus_manifest.json",
        sequence_length=8,
        vocab_size=16,
        top_k=2,
        num_buckets=1,
        dynamic_top_k_min=1,
        dynamic_top_k_max=2,
        dynamic_mass_threshold=0.9,
        selected_rerun_batch_size=2,
        total_selected_exemplar_budget=2,
        fingerprint_corridor_budget_fraction=0.5,
        fingerprint_corridor_budget_max=2,
        fingerprint_corridor_mode_cap=2,
        fingerprint_corridor_candidate_pool_cap=2,
        require_full_selected_budget=True,
    )
    config = resolve_canonical_path_b_config(source)
    assert config is not None

    resolution = resolve_native_path_b_resume(tmp_path, config=config)

    _assert_pending(
        resolution,
        stage="preflight",
        reason="selection_integration_config_mismatch",
    )
