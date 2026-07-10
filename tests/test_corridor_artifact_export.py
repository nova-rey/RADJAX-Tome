from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from radjax_tome.builder import build_production_gpu_tome, validate_teacher_textbook
from radjax_tome.builder.corridor_artifacts import (
    CORRIDOR_TRACKED_STATS,
    DEFAULT_CORRIDOR_MAX_MODES,
    build_corridor_artifacts,
)
from radjax_tome.builder.exemplar_delivery import (
    SELECTED_EXEMPLARS_FILENAME,
    _load_examples,
)
from radjax_tome.builder.teacher_textbook import TinyTextExample
from radjax_tome.targets import (
    TEACHER_TARGET_STORE_SCHEMA_VERSION,
    TEACHER_TARGET_STORE_VERSION,
    TargetStoreMetadata,
    TeacherTargetStore,
)
from tests.test_selected_exemplar_delivery import _config, _json


def test_path_b_emits_first_class_corridor_artifacts(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        output_name="path_b_corridors",
        delivery_path="two_pass_rerun_selected",
    )

    report = build_production_gpu_tome(config)
    output = config.output_dir
    summary = _json(output / "corridors" / "corridor_summary.json")
    fingerprints = _json(output / "corridors" / "corridor_fingerprints.json")
    modes = _json(output / "corridors" / "corridor_modes.json")
    assignments = _json(output / "corridors" / "mode_assignments.json")
    selected_payloads = _json(
        output / "selected_exemplars" / "selected-exemplars-00000.json"
    )["selected_exemplars"]

    assert report["status"] == "pass"
    assert summary["corridor_artifact_built"] is True
    assert summary["corridor_modes_built"] is True
    assert summary["corridor_observation_basis"] == "full_token_position_corridor"
    assert summary["degraded_corridor_export"] is False
    assert summary["corridor_positions_available"] == 25
    assert summary["corridor_positions_used"] == 25
    assert summary["corridor_observation_count"] == 25
    assert summary["mode_count"] >= 1
    assert summary["mode_policy"] == "stat_bands_v0"
    assert summary["corridor_mode_policy"] == "stat_bands_v0"
    assert summary["corridor_max_modes"] == DEFAULT_CORRIDOR_MAX_MODES
    assert summary["corridor_tracked_stats"] == list(CORRIDOR_TRACKED_STATS)
    assert summary["corridor_stat_top_k"] >= 32
    assert summary["min_corridor_stat_top_k"] == 32
    assert summary["corridor_assignment_storage_kind"] == "packed_numpy_v1"
    assert summary["corridor_assignment_count"] == summary["corridor_positions_used"]
    assert summary["mode_count"] <= DEFAULT_CORRIDOR_MAX_MODES
    assert summary["mode_count"] < summary["corridor_positions_used"]
    assert summary["mode_count"] <= 125
    assert summary["fingerprint_count"] >= 1
    assert fingerprints["fingerprint_count"] == summary["fingerprint_count"]
    assert modes["mode_count"] == summary["mode_count"]
    assert modes["mode_policy"] == "stat_bands_v0"
    assert modes["tracked_stats"] == list(CORRIDOR_TRACKED_STATS)
    for stat in CORRIDOR_TRACKED_STATS:
        assert stat in modes["modes"][0]["bounds"]
    assert assignments["schema_version"] == "corridor_mode_assignments_v3"
    assert assignments["assignment_policy"] == "full_token_position_stat_bands_v0"
    assert assignments["storage_kind"] == "packed_numpy_v1"
    assert assignments["full_assignment_retained"] is True
    assert assignments["num_assignments"] == summary["corridor_positions_used"]
    assert "assignments" not in assignments
    for spec in assignments["arrays"].values():
        assert (output / spec["path"]).is_file()
    assert selected_payloads
    assert selected_payloads[0]["corridor_mode_id"] is not None
    assert selected_payloads[0]["corridor_fingerprint_id"] is not None
    assert selected_payloads[0]["corridor_assignment_status"] == "linked"
    assert (output / "corridors" / "corridor_summary.txt").is_file()


def test_stat_band_modes_do_not_explode_on_distinct_top_tokens(
    tmp_path: Path,
) -> None:
    output = tmp_path / "top_token_collapse"
    sequence_length = 40
    metadata = TargetStoreMetadata(
        schema_version=TEACHER_TARGET_STORE_SCHEMA_VERSION,
        target_store_version=TEACHER_TARGET_STORE_VERSION,
        model_id="fake-teacher",
        model_family="fake",
        tokenizer_id="fake-tokenizer",
        tokenizer_hash=None,
        vocab_size=512,
        target_type="corridor_exemplar_score_pass_v1",
        dtype="float32",
        sequence_length=sequence_length,
        num_examples=1,
        shard_count=1,
        created_by="test",
        created_at="2026-07-09T00:00:00Z",
        target_params={
            "corridor_stat_top_k": "32",
            "min_corridor_stat_top_k": "32",
        },
    )
    store = TeacherTargetStore.create(output, metadata, overwrite=True)
    input_ids = np.arange(sequence_length, dtype=np.int32)[None, :]
    attention_mask = np.ones((1, sequence_length), dtype=np.int32)
    entropy = np.full((1, sequence_length), 2.0, dtype=np.float32)
    top1_margin = np.full((1, sequence_length), 0.10, dtype=np.float32)
    top8_mass = np.full((1, sequence_length), 0.80, dtype=np.float32)
    top32_mass = np.full((1, sequence_length), 0.80, dtype=np.float32)
    store.write_shard(
        0,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "corridor_top_token_ids": np.arange(
                sequence_length,
                dtype=np.int32,
            )[None, :],
            "corridor_teacher_entropy": entropy,
            "corridor_entropy": entropy,
            "corridor_top1_margin": top1_margin,
            "corridor_top8_mass": top8_mass,
            "corridor_top32_mass": top32_mass,
            "corridor_tail_mass": np.full(
                (1, sequence_length),
                0.20,
                dtype=np.float32,
            ),
            "corridor_confidence": np.full(
                (1, sequence_length),
                0.45,
                dtype=np.float32,
            ),
            "corridor_lengths": np.asarray([sequence_length], dtype=np.int32),
            "score_example_ids": np.asarray([0], dtype=np.int32),
            "score_max_entropy": np.asarray([2.0], dtype=np.float32),
            "score_mean_entropy": np.asarray([2.0], dtype=np.float32),
            "score_selected_position": np.asarray([0], dtype=np.int32),
            "score_top_token_id": np.asarray([0], dtype=np.int32),
            "score_selected_position_entropy": np.asarray([2.0], dtype=np.float32),
            "score_confidence_at_selected_position": np.asarray(
                [0.45],
                dtype=np.float32,
            ),
            "score_source_policy_ids": np.asarray([0], dtype=np.int32),
            "score_lengths": np.asarray([sequence_length], dtype=np.int32),
        },
    )
    selected = {
        "selected_example_id": "example-0",
        "selected_position": 0,
        "selected_score": 2.0,
    }

    build_corridor_artifacts(
        output_dir=output,
        examples=(TinyTextExample(example_id="example-0", text="hello"),),
        selected_records=[selected],
        selected_payloads=[selected.copy()],
        delivery_path="two_pass_rerun_selected",
        non_selected_exemplar_payload_retained=False,
    )
    summary = _json(output / "corridors" / "corridor_summary.json")
    modes = _json(output / "corridors" / "corridor_modes.json")
    assignments = _json(output / "corridors" / "mode_assignments.json")

    assert summary["mode_policy"] == "stat_bands_v0"
    assert summary["mode_count"] == 1
    assert summary["fingerprint_count"] == sequence_length
    assert modes["modes"][0]["mode_key"] == {
        "entropy_bin": 1,
        "top1_margin_bin": 1,
        "top32_mass_bin": 2,
    }
    assert set(CORRIDOR_TRACKED_STATS).issubset(modes["modes"][0]["bounds"])
    assert assignments["num_assignments"] == sequence_length
    assert assignments["storage_kind"] == "packed_numpy_v1"
    mode_ids = np.load(output / assignments["arrays"]["mode_id"]["path"])
    assert set(mode_ids.tolist()) == {0}
    assert selected["corridor_mode_id"] == 0
    assert selected["corridor_assignment_status"] == "linked"


def test_path_a_emits_corridors_and_prunes_only_candidate_arrays(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        output_name="path_a_corridors",
        delivery_path="one_pass_pruned_candidate",
    )

    assert build_production_gpu_tome(config)["status"] == "pass"
    output = config.output_dir
    delivery = _json(output / "delivery_report.json")
    summary = _json(output / "corridors" / "corridor_summary.json")

    assert delivery["teacher_rerun_count"] == 0
    assert delivery["selected_payload_source"] == "one_pass_candidate_shard_capture"
    assert delivery["corridor_artifact_built"] is True
    assert delivery["corridor_modes_built"] is True
    assert delivery["corridor_observation_basis"] == "full_token_position_corridor"
    assert delivery["corridor_positions_used"] == 25
    assert summary["mode_count"] >= 1
    assert summary["mode_policy"] == "stat_bands_v0"
    assert summary["mode_count"] <= DEFAULT_CORRIDOR_MAX_MODES
    assert (output / "corridors" / "corridor_fingerprints.json").is_file()
    assert (output / "corridors" / "corridor_modes.json").is_file()
    assert (output / "corridors" / "mode_assignments.json").is_file()
    for shard_path in sorted((output / "shards").glob("shard-*.npz")):
        with np.load(shard_path) as shard:
            retained = [
                key for key in shard.files if key.startswith("exemplar_source_")
            ]
        assert retained == []
    assert validate_teacher_textbook(output).status == "pass"


def test_validation_fails_for_selected_only_artifact_missing_corridors(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        output_name="missing_corridors",
        delivery_path="two_pass_rerun_selected",
    )
    assert build_production_gpu_tome(config)["status"] == "pass"
    shutil.rmtree(config.output_dir / "corridors")

    validation = validate_teacher_textbook(config.output_dir)

    assert validation.status == "fail"
    assert (
        "corridor_exemplar_v1 selected-only run did not emit corridor modes"
        in validation.blockers
    )
    assert validation.corridor_artifact_ok is False


def test_validation_fails_for_packed_assignment_invalid_mode_id(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        output_name="invalid_assignment_mode",
        delivery_path="two_pass_rerun_selected",
    )
    assert build_production_gpu_tome(config)["status"] == "pass"
    manifest = _json(config.output_dir / "corridors" / "mode_assignments.json")
    mode_path = config.output_dir / manifest["arrays"]["mode_id"]["path"]
    mode_ids = np.load(mode_path)
    mode_ids[0] = 999_999
    np.save(mode_path, mode_ids)

    validation = validate_teacher_textbook(config.output_dir)

    assert validation.status == "fail"
    assert "mode_assignments references nonexistent mode_id" in validation.blockers


def test_validation_fails_for_packed_assignment_shape_mismatch(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        output_name="invalid_assignment_shape",
        delivery_path="two_pass_rerun_selected",
    )
    assert build_production_gpu_tome(config)["status"] == "pass"
    manifest = _json(config.output_dir / "corridors" / "mode_assignments.json")
    position_path = config.output_dir / manifest["arrays"]["position"]["path"]
    positions = np.load(position_path)
    np.save(position_path, positions[:-1])

    validation = validate_teacher_textbook(config.output_dir)

    assert validation.status == "fail"
    assert "mode_assignments array shape mismatch: position" in validation.blockers


def test_score_only_corridor_export_is_degraded_and_rejected(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        output_name="score_only_degraded",
        delivery_path="two_pass_rerun_selected",
    )
    assert build_production_gpu_tome(config)["status"] == "pass"
    output = config.output_dir
    shutil.rmtree(output / "corridors")
    for shard_path in sorted((output / "shards").glob("shard-*.npz")):
        with np.load(shard_path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        for key in (
            "corridor_top_token_ids",
            "corridor_teacher_entropy",
            "corridor_entropy",
            "corridor_top1_margin",
            "corridor_top8_mass",
            "corridor_top32_mass",
            "corridor_tail_mass",
            "corridor_confidence",
            "corridor_lengths",
        ):
            arrays.pop(key, None)
        np.savez(shard_path, **arrays)
    selected_records = _json(output / "leaderboards" / SELECTED_EXEMPLARS_FILENAME)[
        "selected_exemplars"
    ]
    selected_payloads = _json(
        output / "selected_exemplars" / "selected-exemplars-00000.json"
    )["selected_exemplars"]
    examples = _load_examples(config.dataset_path, max_examples=5)

    build_corridor_artifacts(
        output_dir=output,
        examples=examples,
        selected_records=selected_records,
        selected_payloads=selected_payloads,
        delivery_path="two_pass_rerun_selected",
        non_selected_exemplar_payload_retained=False,
        allow_degraded_score_only=True,
    )
    summary = _json(output / "corridors" / "corridor_summary.json")
    validation = validate_teacher_textbook(output)

    assert summary["corridor_observation_basis"] == "score_selected_position_only"
    assert summary["degraded_corridor_export"] is True
    assert summary["corridor_positions_available"] == 25
    assert summary["corridor_positions_used"] == 5
    assert validation.status == "fail"
    assert any(
        "corridor artifact was built from score-selected positions only" in blocker
        for blocker in validation.blockers
    )


def test_reports_and_cover_page_include_corridor_counts(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        output_name="reports_corridors",
        delivery_path="two_pass_rerun_selected",
    )
    assert build_production_gpu_tome(config)["status"] == "pass"
    output = config.output_dir
    production = _json(output / "production_build_report.json")
    delivery = _json(output / "delivery_report.json")
    validation = _json(output / "validation_report.json")
    cover_page = _json(output / "cover_page.json")
    contents = {entry["path"]: entry["role"] for entry in cover_page["contents"]}

    for report in (production, delivery):
        assert report["corridor_artifact_built"] is True
        assert report["corridor_modes_built"] is True
        assert report["corridor_observation_basis"] == "full_token_position_corridor"
        assert report["degraded_corridor_export"] is False
        assert report["corridor_positions_available"] == 25
        assert report["corridor_positions_used"] == 25
        assert report["corridor_mode_count"] >= 1
        assert report["corridor_mode_policy"] == "stat_bands_v0"
        assert report["corridor_max_modes"] == DEFAULT_CORRIDOR_MAX_MODES
        assert report["corridor_tracked_stats"] == list(CORRIDOR_TRACKED_STATS)
        assert report["corridor_stat_top_k"] >= 32
        assert report["min_corridor_stat_top_k"] == 32
        assert report["corridor_assignment_storage_kind"] == "packed_numpy_v1"
        assert report["corridor_assignment_count"] == report["corridor_positions_used"]
        assert report["selected_exemplars_linked_to_corridor_modes"] is True
        assert report["corridor_fingerprint_count"] >= 1
    assert validation["corridor_artifact_ok"] is True
    assert validation["corridor_modes_ok"] is True
    assert validation["corridor_mode_count"] >= 1
    assert validation["corridor_mode_policy"] == "stat_bands_v0"
    assert validation["corridor_stat_top_k"] >= 32
    assert validation["corridor_assignment_storage_kind"] == "packed_numpy_v1"
    assert validation["corridor_assignment_count"] == 25
    assert validation["corridor_observation_basis"] == "full_token_position_corridor"
    assert validation["corridor_positions_used"] == 25
    assert contents["corridors/corridor_summary.json"] == "corridor_summary"
    assert contents["corridors/corridor_fingerprints.json"] == ("corridor_fingerprints")
    assert contents["corridors/corridor_modes.json"] == "corridor_mode_table"
    assert contents["corridors/mode_assignments.json"] == (
        "corridor_assignment_manifest"
    )
