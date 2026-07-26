"""M4C resolver coverage for the native selected-artifact assembly boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from radjax_tome.builder.native_path_b.resume import (
    NativePathBResumeResolution,
    resolve_native_path_b_resume,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_completed_delivery_and_final_corridor(output_dir: Path) -> None:
    _write_json(output_dir / "run_plan.json", {"status": "pass"})
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
    _write_json(
        output_dir / "c6" / "authority_manifest.json",
        {"score_pass_authority_hash": "sha256:authority"},
    )
    _write_json(
        output_dir / "delivery_report.json",
        {
            "status": "pass",
            "num_selected_exemplars": 2,
            "delivery_authority_hash": "sha256:authority",
        },
    )
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
            "selected_exemplar_count": 2,
            "selected_exemplars_linked_to_corridor_modes": True,
        },
    )
    _write_json(corridors / "corridor_fingerprints.json", {"fingerprint_count": 1})
    _write_json(corridors / "corridor_modes.json", {"mode_count": 1})
    _write_json(corridors / "mode_assignments.json", {"num_assignments": 2})


def _write_assembled_selected_artifact(output_dir: Path) -> None:
    selected_records = [
        {"selected_example_id": "example-0", "selected_position": 0},
        {"selected_example_id": "example-1", "selected_position": 1},
    ]
    _write_json(
        output_dir / "leaderboards" / "selected_exemplars.json",
        {
            "schema_version": "selected_exemplars_v1",
            "selected_exemplars": selected_records,
        },
    )
    _write_json(
        output_dir / "selected_exemplars" / "payload_index.json",
        {
            "schema_version": "selected_exemplar_payload_index_v1",
            "selected_exemplars": selected_records,
        },
    )
    _write_json(
        output_dir / "selected_exemplars" / "selected-exemplars-00000.json",
        {
            "schema_version": "selected_exemplar_payload_shard_v1",
            "delivery_authority_hash": "sha256:authority",
            "selected_exemplars": selected_records,
        },
    )


def _assert_next_stage(
    resolution: NativePathBResumeResolution,
    stage: str,
) -> None:
    assert resolution.complete is False
    assert resolution.stage == stage
    assert resolution.failure is not None
    assert resolution.failure.stage == stage
    assert resolution.failure.blockers


def test_final_corridor_without_assembled_selected_artifact_resumes_assembly(
    tmp_path: Path,
) -> None:
    _write_completed_delivery_and_final_corridor(tmp_path)

    resolution = resolve_native_path_b_resume(tmp_path)

    _assert_next_stage(resolution, "artifact_assembly")


def test_assembled_selected_artifact_without_validation_resumes_validation(
    tmp_path: Path,
) -> None:
    _write_completed_delivery_and_final_corridor(tmp_path)
    _write_assembled_selected_artifact(tmp_path)

    resolution = resolve_native_path_b_resume(tmp_path)

    _assert_next_stage(resolution, "validation_linkage")
    assert resolution.evidence is not None
    assert resolution.evidence.stage == "artifact_assembly"
