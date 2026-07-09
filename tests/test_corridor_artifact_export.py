from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from radjax_tome.builder import build_production_gpu_tome, validate_teacher_textbook
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
    assert summary["mode_count"] >= 1
    assert summary["fingerprint_count"] >= 1
    assert fingerprints["fingerprint_count"] == summary["fingerprint_count"]
    assert modes["mode_count"] == summary["mode_count"]
    assert assignments["assignments"]
    assert selected_payloads
    assert selected_payloads[0]["corridor_mode_id"] is not None
    assert selected_payloads[0]["corridor_fingerprint_id"] is not None
    assert selected_payloads[0]["corridor_assignment_status"] == "linked"
    assert (output / "corridors" / "corridor_summary.txt").is_file()


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
    assert summary["mode_count"] >= 1
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
        assert report["corridor_mode_count"] >= 1
        assert report["corridor_fingerprint_count"] >= 1
    assert validation["corridor_artifact_ok"] is True
    assert validation["corridor_modes_ok"] is True
    assert validation["corridor_mode_count"] >= 1
    assert contents["corridors/corridor_summary.json"] == "corridor_summary"
    assert contents["corridors/corridor_fingerprints.json"] == ("corridor_fingerprints")
    assert contents["corridors/corridor_modes.json"] == "corridor_modes"
    assert contents["corridors/mode_assignments.json"] == ("corridor_mode_assignments")
