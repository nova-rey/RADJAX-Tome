from __future__ import annotations

import json
from pathlib import Path

import pytest

from radjax_tome.builder import (
    EXEMPLAR_SELECTION_MANIFEST_FILENAME,
    MULTI_LEADERBOARD_SELECTOR_POLICY,
    BackendTeacherTextbookBuildConfig,
    ExemplarCandidate,
    build_backend_teacher_textbook,
    extract_one_pass_candidates,
    extract_score_pass_candidates,
    select_exemplars,
    validate_exemplar_selection_manifest,
)
from radjax_tome.targets.store import TeacherTargetStore


def _config(tmp_path: Path, **overrides: object) -> BackendTeacherTextbookBuildConfig:
    payload = {
        "output_dir": tmp_path / "backend_tome",
        "teacher_backend": "cpu_reference",
        "runtime_mode": "cpu",
        "target_policy": "corridor_exemplar_v1",
        "sequence_length": 5,
        "batch_size": 2,
        "max_examples": 3,
        "vocab_size": 11,
        "top_k": 4,
        "num_buckets": 3,
        "overwrite": True,
    }
    payload.update(overrides)
    return BackendTeacherTextbookBuildConfig(**payload)


def _candidate(
    example_id: str,
    *,
    position: int,
    shard_id: int = 0,
    max_entropy: float = 1.0,
    mean_entropy: float = 1.0,
    confidence: float = 0.5,
    selected_position_entropy: float | None = None,
    tail_mass: float | None = None,
    effective_top_k: float | None = None,
) -> ExemplarCandidate:
    scores = {
        "max_entropy": max_entropy,
        "mean_entropy": mean_entropy,
        "confidence": confidence,
        "selected_position_entropy": (
            max_entropy
            if selected_position_entropy is None
            else selected_position_entropy
        ),
        "position_bucket": float(position // 2),
        "length_bucket": 0.0,
    }
    if tail_mass is not None:
        scores["tail_mass"] = tail_mass
    if effective_top_k is not None:
        scores["effective_top_k"] = effective_top_k
    return ExemplarCandidate(
        example_id=example_id,
        source_shard_id=shard_id,
        source_row=0,
        selected_position=position,
        candidate_positions=(position,),
        sequence_length=8,
        capture_mode="one_pass_candidate",
        source_policy="1",
        score_fields=scores,
        payload_ref={"kind": "test", "source_shard_id": shard_id, "source_row": 0},
    )


def _board(manifest: dict[str, object], board_id: str) -> dict[str, object]:
    boards = {
        str(board["board_id"]): board
        for board in manifest["boards"]  # type: ignore[index]
    }
    return boards[board_id]


def test_selector_extracts_path_a_corridor_candidates(tmp_path: Path) -> None:
    output = tmp_path / "one_pass"
    build_backend_teacher_textbook(
        _config(
            tmp_path,
            output_dir=output,
            exemplar_capture_mode="one_pass_candidate",
        )
    )
    store = TeacherTargetStore.open(output)
    shard = store.read_shard(0)

    candidates = extract_one_pass_candidates(
        shard,
        example_ids=("example-a", "example-b"),
        source_shard_id=0,
    )

    assert candidates
    assert candidates[0].capture_mode == "one_pass_candidate"
    assert candidates[0].payload_ref["kind"] == "corridor_exemplar_v1"
    assert "max_entropy" in candidates[0].score_fields
    assert "tail_mass" in candidates[0].score_fields
    assert "effective_top_k" in candidates[0].score_fields


def test_selector_extracts_path_b_score_pass_candidates(tmp_path: Path) -> None:
    output = tmp_path / "score_pass"
    build_backend_teacher_textbook(
        _config(
            tmp_path,
            output_dir=output,
            exemplar_capture_mode="two_pass_sparse_exemplar",
        )
    )
    store = TeacherTargetStore.open(output)
    shard = store.read_shard(0)

    candidates = extract_score_pass_candidates(
        shard,
        example_ids=("example-a", "example-b"),
        source_shard_id=0,
    )

    assert len(candidates) == 2
    assert candidates[0].capture_mode == "two_pass_sparse_exemplar"
    assert candidates[0].payload_ref["kind"] == "corridor_exemplar_score_pass_v1"
    assert "selected_position_entropy" in candidates[0].score_fields
    assert "confidence" in candidates[0].score_fields


def test_global_max_entropy_board_keeps_bounded_top_winners() -> None:
    manifest = select_exemplars(
        (
            _candidate("a", position=0, max_entropy=1.0),
            _candidate("b", position=1, max_entropy=3.0),
            _candidate("c", position=2, max_entropy=2.0),
        ),
        capture_mode="one_pass_candidate",
        fulfillment_policy="select_from_existing_capture",
        board_capacity=2,
        created_at="2026-07-07T00:00:00+00:00",
    )

    board = _board(manifest, "global_max_entropy")
    assert board["winner_count"] == 2
    assert [winner["example_id"] for winner in board["winners"]] == ["b", "c"]


def test_low_confidence_board_keeps_bounded_top_winners() -> None:
    manifest = select_exemplars(
        (
            _candidate("a", position=0, confidence=0.9),
            _candidate("b", position=1, confidence=0.2),
            _candidate("c", position=2, confidence=0.4),
        ),
        capture_mode="one_pass_candidate",
        fulfillment_policy="select_from_existing_capture",
        board_capacity=2,
        created_at="2026-07-07T00:00:00+00:00",
    )

    board = _board(manifest, "low_confidence")
    assert [winner["example_id"] for winner in board["winners"]] == ["b", "c"]


def test_position_bucket_and_shard_boards_keep_local_winners() -> None:
    manifest = select_exemplars(
        (
            _candidate("a", position=0, shard_id=0, selected_position_entropy=1.0),
            _candidate("b", position=1, shard_id=0, selected_position_entropy=3.0),
            _candidate("c", position=4, shard_id=1, selected_position_entropy=2.0),
            _candidate("d", position=5, shard_id=1, selected_position_entropy=5.0),
        ),
        capture_mode="one_pass_candidate",
        fulfillment_policy="select_from_existing_capture",
        board_capacity=1,
        created_at="2026-07-07T00:00:00+00:00",
    )

    assert (
        _board(manifest, "position_bucket_entropy:0")["winners"][0]["example_id"] == "b"
    )
    assert (
        _board(manifest, "position_bucket_entropy:2")["winners"][0]["example_id"] == "d"
    )
    assert _board(manifest, "shard_coverage:0")["winners"][0]["example_id"] == "b"
    assert _board(manifest, "shard_coverage:1")["winners"][0]["example_id"] == "d"


def test_duplicate_winners_are_deduped_with_reasons_and_deterministic() -> None:
    candidates = (
        _candidate("a", position=0, max_entropy=9.0, confidence=0.1),
        _candidate("b", position=1, max_entropy=1.0, confidence=0.9),
    )

    first = select_exemplars(
        candidates,
        capture_mode="one_pass_candidate",
        fulfillment_policy="select_from_existing_capture",
        board_capacity=1,
        created_at="2026-07-07T00:00:00+00:00",
    )
    second = select_exemplars(
        candidates,
        capture_mode="one_pass_candidate",
        fulfillment_policy="select_from_existing_capture",
        board_capacity=1,
        created_at="2026-07-07T00:00:00+00:00",
    )

    assert first == second
    validate_exemplar_selection_manifest(first)
    selected = first["selected_examples"][0]
    assert selected["example_id"] == "a"
    assert selected["selected_positions"] == [0]
    assert "global_max_entropy" in selected["winning_boards"]
    assert "low_confidence" in selected["winning_boards"]
    assert first["semantic_diversity_used"] is False
    assert first["utility_calibrated"] is False


def test_path_a_manifest_uses_select_from_existing_capture(tmp_path: Path) -> None:
    output = tmp_path / "selected_one_pass"
    build_backend_teacher_textbook(
        _config(
            tmp_path,
            output_dir=output,
            exemplar_capture_mode="one_pass_candidate",
            exemplar_selection_enabled=True,
            exemplar_selection_board_capacity=2,
        )
    )

    manifest = json.loads(
        (output / EXEMPLAR_SELECTION_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    params = TeacherTargetStore.open(output).metadata.target_params

    assert manifest["schema_version"] == "exemplar_selection_manifest_v1"
    assert manifest["selection_policy"] == MULTI_LEADERBOARD_SELECTOR_POLICY
    assert manifest["fulfillment_policy"] == "select_from_existing_capture"
    assert manifest["selection_application"] == "retain_selected_candidates"
    assert manifest["selected_examples"]
    assert manifest["selected_examples"][0]["retain_from_existing_capture"] is True
    assert manifest["selected_examples"][0]["rerun_for_pass_2"] is False
    assert params["exemplar_selection_enabled"] == "true"
    assert params["exemplar_selection_manifest_path"] == (
        EXEMPLAR_SELECTION_MANIFEST_FILENAME
    )
    assert params["selected_from_existing_capture"] == "true"
    assert params["production_global_selector"] == "false"
    assert params["semantic_diversity_used"] == "false"
    assert params["utility_calibrated"] == "false"


def test_path_b_manifest_uses_rerun_requisition(tmp_path: Path) -> None:
    output = tmp_path / "selected_score_pass"
    build_backend_teacher_textbook(
        _config(
            tmp_path,
            output_dir=output,
            exemplar_capture_mode="two_pass_sparse_exemplar",
            exemplar_selection_enabled=True,
            exemplar_selection_board_capacity=2,
        )
    )

    manifest = json.loads(
        (output / EXEMPLAR_SELECTION_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    params = TeacherTargetStore.open(output).metadata.target_params

    assert manifest["fulfillment_policy"] == "rerun_selected_capture"
    assert manifest["selection_application"] == "rerun_selected_examples"
    assert manifest["selected_examples"]
    assert manifest["selected_examples"][0]["rerun_for_pass_2"] is True
    assert manifest["selected_examples"][0]["retain_from_existing_capture"] is False
    assert params["rerun_manifest_ready"] == "true"
    assert params["selected_pass_rerun_performed"] == "false"


def test_selection_off_by_default_and_unsupported_targets_fail(
    tmp_path: Path,
) -> None:
    output = tmp_path / "default_off"
    build_backend_teacher_textbook(_config(tmp_path, output_dir=output))

    assert not (output / EXEMPLAR_SELECTION_MANIFEST_FILENAME).exists()
    assert (
        "exemplar_selection_enabled"
        not in TeacherTargetStore.open(output).metadata.target_params
    )

    with pytest.raises(ValueError, match="exemplar selection requires"):
        build_backend_teacher_textbook(
            _config(
                tmp_path,
                output_dir=tmp_path / "bad",
                target_policy="dynamic_cascaded_soft_labels_v1",
                exemplar_selection_enabled=True,
            )
        )
