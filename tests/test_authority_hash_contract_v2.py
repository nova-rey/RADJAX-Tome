from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from radjax_tome.builder.authority_hashes import (
    AUTHORITY_HASH_V1,
    AUTHORITY_HASH_V2,
    authority_hash_contract_version,
    authority_hashes_for_artifact,
    score_pass_authority_hash_v1,
)
from radjax_tome.golden.compare import compare_contracts
from radjax_tome.golden.contract import (
    GOLDEN_CONTRACT_SCHEMA_V2,
    build_contract,
    digest_active_payload_storage,
)
from radjax_tome.golden.projection import (
    _authority_for_capture,
    capture_golden_contract,
)
from radjax_tome.targets.schema import (
    TEACHER_TARGET_STORE_SCHEMA_VERSION,
    TEACHER_TARGET_STORE_VERSION,
)


def test_v2_ignores_runtime_timestamps_but_retains_raw_integrity(
    tmp_path: Path,
) -> None:
    artifact = _write_authority_artifact(tmp_path / "artifact")
    first = authority_hashes_for_artifact(
        artifact, selection_integration_config_hash="sha256:selection"
    )
    _rewrite_created_at(artifact / "metadata.json", "2026-07-24T00:00:00+00:00")
    _rewrite_created_at(
        artifact / "c6" / "production_global_selector.json", "2026-07-24T00:00:00+00:00"
    )
    second = authority_hashes_for_artifact(
        artifact, selection_integration_config_hash="sha256:selection"
    )

    assert (
        first.raw_artifact_digests["metadata.json"]
        != second.raw_artifact_digests["metadata.json"]
    )
    assert (
        first.raw_artifact_digests["c6/production_global_selector.json"]
        != second.raw_artifact_digests["c6/production_global_selector.json"]
    )
    assert first.score_pass_authority_hash_v1 != second.score_pass_authority_hash_v1
    assert first.score_pass_authority_hash_v2 == second.score_pass_authority_hash_v2
    assert _v2_root(first.score_pass_authority_hash_v2) == _v2_root(
        second.score_pass_authority_hash_v2
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "metadata",
        "assignment",
        "mode",
        "selector",
        "selection_config",
    ),
)
def test_v2_changes_for_every_authority_bearing_semantic_input(
    tmp_path: Path,
    mutation: str,
) -> None:
    artifact = _write_authority_artifact(tmp_path / mutation)
    first = authority_hashes_for_artifact(
        artifact, selection_integration_config_hash="sha256:selection"
    )
    selection_hash = "sha256:selection"
    if mutation == "metadata":
        payload = _read(artifact / "metadata.json")
        payload["vocab_size"] = 8192
        _write(artifact / "metadata.json", payload)
    elif mutation == "assignment":
        np.save(
            artifact / "corridors" / "mode_assignments" / "mode_id.npy",
            np.asarray([1, 0], dtype=np.int32),
        )
    elif mutation == "mode":
        payload = _read(artifact / "corridors" / "corridor_modes.json")
        payload["modes"][0]["bounds"]["entropy"]["mean"] = 2.0
        _write(artifact / "corridors" / "corridor_modes.json", payload)
    elif mutation == "selector":
        payload = _read(artifact / "c6" / "production_global_selector.json")
        payload["boards"][0]["ranked_candidates"][0]["score"] = 0.25
        _write(artifact / "c6" / "production_global_selector.json", payload)
    else:
        selection_hash = "sha256:selection-drift"

    second = authority_hashes_for_artifact(
        artifact, selection_integration_config_hash=selection_hash
    )
    assert first.score_pass_authority_hash_v2 != second.score_pass_authority_hash_v2


def test_v1_calculation_remains_the_historical_raw_input_recipe(tmp_path: Path) -> None:
    artifact = _write_authority_artifact(tmp_path / "artifact")
    hashes = authority_hashes_for_artifact(
        artifact, selection_integration_config_hash="sha256:selection"
    )
    payload = {
        "metadata_sha256": hashes.raw_artifact_digests["metadata.json"],
        "assignment_manifest_sha256": hashes.raw_artifact_digests[
            "corridors/mode_assignments.json"
        ],
        "modes_sha256": hashes.raw_artifact_digests["corridors/corridor_modes.json"],
        "selector_sha256": hashes.raw_artifact_digests[
            "c6/production_global_selector.json"
        ],
        "selection_integration_config_hash": "sha256:selection",
    }
    expected = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )

    assert (
        score_pass_authority_hash_v1(
            hashes.raw_artifact_digests,
            selection_integration_config_hash="sha256:selection",
        )
        == expected
    )
    assert hashes.score_pass_authority_hash_v1 == expected


def test_v2_capture_projects_a_historical_v1_authority_read_only(
    tmp_path: Path,
) -> None:
    artifact = _write_authority_artifact(tmp_path / "artifact")
    hashes = authority_hashes_for_artifact(
        artifact, selection_integration_config_hash="sha256:selection"
    )
    historical_authority = {
        "schema_version": "radjax.c6_selection_authority.v1",
        "score_pass_authority_hash": hashes.score_pass_authority_hash_v1,
        "selection_integration_config_hash": "sha256:selection",
    }

    projected, replacement = _authority_for_capture(
        artifact, historical_authority, GOLDEN_CONTRACT_SCHEMA_V2
    )

    assert authority_hash_contract_version(historical_authority) == AUTHORITY_HASH_V1
    assert historical_authority["score_pass_authority_hash"] == (
        hashes.score_pass_authority_hash_v1
    )
    assert projected["score_pass_authority_contract_version"] == AUTHORITY_HASH_V2
    assert projected["score_pass_authority_hash"] == hashes.score_pass_authority_hash_v2
    assert projected["score_pass_authority_hash_v1"] == (
        hashes.score_pass_authority_hash_v1
    )
    assert projected["raw_artifact_digests"] == hashes.raw_artifact_digests
    assert replacement == (
        hashes.score_pass_authority_hash_v1,
        hashes.score_pass_authority_hash_v2,
    )


def test_july_t4_artifacts_compare_under_v2_when_both_are_available(
    tmp_path: Path,
) -> None:
    july_19 = os.environ.get("RADJAX_TOME_T4_JULY19_ARTIFACT")
    july_24 = os.environ.get("RADJAX_TOME_T4_JULY24_ARTIFACT")
    if july_19 is None and july_24 is None:
        pytest.skip("July T4 source artifacts are not available locally")
    if not july_19 or not july_24:
        pytest.fail("both July T4 artifact environment variables are required")
    left = Path(july_19)
    right = Path(july_24)
    if not left.is_dir() or not right.is_dir():
        pytest.fail("configured July T4 artifact path is not a directory")
    left_contract = tmp_path / "july19-v2"
    right_contract = tmp_path / "july24-v2"

    capture_golden_contract(
        left, left_contract, contract_version=GOLDEN_CONTRACT_SCHEMA_V2
    )
    capture_golden_contract(
        right, right_contract, contract_version=GOLDEN_CONTRACT_SCHEMA_V2
    )

    assert compare_contracts(left_contract, right_contract)["status"] == "pass"


def _v2_root(authority_hash: str) -> str:
    rows = _rows(authority_hash)
    return build_contract(
        fixture_metadata={"canonical_pipeline": "native_two_pass"},
        input_identity={"corpus_hash": "sha256:corpus"},
        semantic_policy={
            "score_pass_authority_contract_version": AUTHORITY_HASH_V2,
        },
        stage_summary=[{"stage": "validation", "status": "pass"}],
        selected_obligations=rows,
        source_passports=rows,
        payload_semantics=_payload_rows(),
        board_summary={"authority": {"score_pass_authority_hash": authority_hash}},
        schema_version=GOLDEN_CONTRACT_SCHEMA_V2,
    )["semantic_root"]


def _rows(authority_hash: str) -> list[dict[str, object]]:
    return [
        {
            "selection_index": 1,
            "selected_example_id": "example-1",
            "selected_position": 2,
            "score_pass_authority_hash": authority_hash,
        }
    ]


def _payload_rows() -> list[dict[str, object]]:
    return [
        {
            "selection_index": 1,
            "selected_example_id": "example-1",
            "selected_position": 2,
            "effective_top_k": 1,
            **digest_active_payload_storage(
                {
                    "effective_top_k": 1,
                    "top_token_ids": [7],
                    "top_probs": [0.5],
                    "top_log_probs": [-0.6931471805599453],
                    "top_selection_mask": [True],
                }
            ),
        }
    ]


def _write_authority_artifact(root: Path) -> Path:
    assignments_dir = root / "corridors" / "mode_assignments"
    assignments_dir.mkdir(parents=True)
    _write(
        root / "metadata.json",
        {
            "schema_version": TEACHER_TARGET_STORE_SCHEMA_VERSION,
            "target_store_version": TEACHER_TARGET_STORE_VERSION,
            "model_id": "gemma-test",
            "model_family": "gpu_torch",
            "tokenizer_id": "gemma-test",
            "tokenizer_hash": "sha256:tokenizer",
            "vocab_size": 4096,
            "target_type": "corridor_exemplar_v1",
            "dtype": "float32",
            "sequence_length": 8,
            "num_examples": 2,
            "shard_count": 1,
            "created_by": "test",
            "created_at": "2026-07-19T00:00:00+00:00",
            "source": {
                "kind": "/rental/corpus.jsonl",
                "source_corpus_hash": "sha256:corpus",
                "source_corpus_manifest_hash": "sha256:manifest",
            },
            "provenance": {"phase": "runtime", "teacher_backend": "gpu_torch"},
            "target_params": {
                "target_policy": "corridor_exemplar_v1",
                "dynamic_top_k_min": "32",
                "dynamic_top_k_max": "4096",
                "dynamic_mass_threshold": "0.99",
                "corridor_stat_top_k": "32",
                "run_manifest_path": "/rental/run_manifest.json",
            },
        },
    )
    arrays = {
        "position_example_index": np.asarray([0, 1], dtype=np.int32),
        "position": np.asarray([2, 3], dtype=np.int32),
        "mode_id": np.asarray([0, 0], dtype=np.int32),
        "weight": np.asarray([1.0, 1.0], dtype=np.float32),
        "fingerprint_index": np.asarray([0, 1], dtype=np.int32),
    }
    for name, array in arrays.items():
        np.save(assignments_dir / f"{name}.npy", array)
    (assignments_dir / "examples_metadata.jsonl").write_text(
        "".join(
            json.dumps({"example_index": index, "example_id": f"example-{index}"})
            + "\n"
            for index in range(2)
        ),
        encoding="utf-8",
    )
    _write(
        root / "corridors" / "mode_assignments.json",
        {
            "schema_version": "corridor_mode_assignments_v3",
            "assignment_policy": "full_token_position_stat_bands_v0",
            "storage_kind": "packed_numpy_v1",
            "corridor_observation_basis": "full_token_position_corridor",
            "full_assignment_retained": True,
            "num_assignments": 2,
            "num_examples": 2,
            "arrays": {
                name: {
                    "path": f"corridors/mode_assignments/{name}.npy",
                    "dtype": str(array.dtype),
                    "shape": list(array.shape),
                }
                for name, array in arrays.items()
            },
            "examples_metadata": {
                "path": "corridors/mode_assignments/examples_metadata.jsonl",
                "num_examples": 2,
            },
        },
    )
    _write(
        root / "corridors" / "corridor_modes.json",
        {
            "schema_version": "corridor_modes_v2",
            "mode_policy": "stat_bands_v0",
            "corridor_mode_policy": "stat_bands_v0",
            "corridor_max_modes": 256,
            "corridor_stat_top_k": 32,
            "min_corridor_stat_top_k": 32,
            "tracked_stats": ["entropy"],
            "corridor_observation_basis": "full_token_position_corridor",
            "degraded_corridor_export": False,
            "corridor_positions_available": 2,
            "corridor_positions_used": 2,
            "mode_count": 1,
            "modes": [
                {
                    "mode_id": 0,
                    "mode_key": {"entropy_bin": 1},
                    "bounds": {"entropy": {"min": 0.0, "max": 2.0, "mean": 1.0}},
                }
            ],
        },
    )
    _write(
        root / "c6" / "production_global_selector.json",
        {
            "schema_version": "exemplar_selection_manifest_v1",
            "selection_policy": "multi_leaderboard_exemplar_selector_v1",
            "capture_mode": "two_pass_sparse_exemplar",
            "fulfillment_policy": "rerun_selected_capture",
            "selection_application": "rerun_selected_examples",
            "num_candidates_seen": 1,
            "num_boards": 1,
            "total_board_capacity": 1,
            "num_board_winners": 1,
            "num_unique_examples_selected": 1,
            "num_unique_positions_selected": 1,
            "deduplication_policy": "rank_aware_board_assignment_with_backfill_v1",
            "duplicate_candidate_count": 0,
            "backfill_attempt_count": 0,
            "backfill_success_count": 0,
            "boards_with_backfill": [],
            "runner_up_pool_multiplier": 4,
            "score_aware_budget_trimming": True,
            "budget_trimming_policy": "score_aware_assigned_board_rank_v1",
            "budget_requested_examples": None,
            "budget_requested_fraction": None,
            "budget_applied": False,
            "budget_trimmed_example_count": 0,
            "budget_trimmed_position_count": 0,
            "production_global_selector": True,
            "semantic_diversity_used": False,
            "utility_calibrated": False,
            "retention_policy": "rerun_requisition_only",
            "created_at": "2026-07-19T00:00:00+00:00",
            "boards": [
                {
                    "board_id": "global_max_entropy",
                    "ranked_candidates": [
                        {
                            "example_id": "example-0",
                            "selected_position": 2,
                            "score": 0.5,
                        }
                    ],
                }
            ],
            "selected_examples": [
                {"example_id": "example-0", "selected_positions": [2]}
            ],
        },
    )
    return root


def _rewrite_created_at(path: Path, value: str) -> None:
    payload = _read(path)
    payload["created_at"] = value
    _write(path, payload)


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
