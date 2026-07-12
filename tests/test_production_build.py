from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import radjax_tome.builder.production as production
from radjax_tome.backends import (
    CPUReferenceTeacherEmissionBackend,
    TeacherBackendConfig,
)
from radjax_tome.builder import ProductionBuildConfig, build_production_gpu_tome
from radjax_tome.corpora import CorpusBuildConfig, build_corpus_artifact
from radjax_tome.io.json import write_json
from radjax_tome.provenance import inspect_teacher_model, write_teacher_model_provenance
from radjax_tome.reports import TomeParityReport
from radjax_tome.targets.store import TeacherTargetStore
from radjax_tome.tome import STUDENT, package_tome_artifact, validate_tome_package
from tests.helpers.subprocess import run_cli

ROOT = Path(__file__).resolve().parents[1]


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fake_model_dir(root: Path) -> Path:
    model = root / "model"
    model.mkdir(parents=True, exist_ok=True)
    (model / "config.json").write_text(
        json.dumps({"_name_or_path": "radjax/local-test", "model_type": "tiny"}),
        encoding="utf-8",
    )
    (model / "tokenizer.json").write_text('{"version": "1.0"}', encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")
    return model


def _production_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    sources = []
    for index in range(6):
        source = tmp_path / f"source-{index}.txt"
        source.write_text(f"production example {index}", encoding="utf-8")
        sources.append(source)
    corpus_dir = tmp_path / "corpus_out"
    build_corpus_artifact(
        CorpusBuildConfig(inputs=tuple(sources), output_dir=corpus_dir, overwrite=True)
    )
    model = _fake_model_dir(tmp_path)
    provenance_path = tmp_path / "teacher_model_provenance.json"
    write_teacher_model_provenance(
        inspect_teacher_model(model, model_name="radjax/local-test"),
        provenance_path,
    )
    return (
        corpus_dir / "corpus.jsonl",
        corpus_dir / "corpus_manifest.json",
        provenance_path,
        model,
    )


def _config(tmp_path: Path, **overrides: object) -> ProductionBuildConfig:
    dataset, corpus_manifest, provenance_path, model = _production_inputs(tmp_path)
    payload = {
        "teacher_model": str(model),
        "dataset_path": dataset,
        "corpus_manifest_path": corpus_manifest,
        "teacher_model_provenance_path": provenance_path,
        "output_dir": tmp_path / "production_tome",
        "teacher_backend": "cpu_reference",
        "runtime_mode": "cpu",
        "target_policy": "dynamic_cascaded_soft_labels_v1",
        "sequence_length": 5,
        "vocab_size": 13,
        "top_k": 4,
        "num_buckets": 3,
        "gpu_batch_size_mode": "preset",
        "gpu_batch_size_preset": 2,
        "shard_size_examples": 2,
        "max_examples": 5,
    }
    payload.update(overrides)
    return ProductionBuildConfig(**payload)


def test_production_build_writes_plan_report_cover_and_valid_artifact(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    report = build_production_gpu_tome(config)
    output = config.output_dir

    assert report["schema_version"] == "production_build_report_v1"
    assert report["status"] == "pass"
    assert report["run_plan_status"] == "pass"
    assert report["effective_batch_size"] == 2
    assert report["validation_status"] == "pass"
    assert report["streaming_build"] is True
    assert report["parity_status"] == "not_run"
    assert report["claims_not_made"]["no_model_download"] is True
    assert report["claims_not_made"]["no_silent_cpu_fallback"] is True
    assert report["inputs"]["allow_downloads"] is False
    assert (output / "run_plan.json").is_file()
    assert (output / "production_build_report.json").is_file()
    assert (output / "cover_page.json").is_file()
    assert (output / "run_manifest.json").is_file()
    assert (output / "progress_log.jsonl").is_file()
    assert _json(output / "run_manifest.json")["status"] == "complete"
    assert report["selection_integration_policy"] == "global_only_v1"
    assert not (output / "c6").exists()


def test_production_build_progress_sidecar_reaches_complete(tmp_path: Path) -> None:
    config = _config(tmp_path, progress=True)

    report = build_production_gpu_tome(config)
    progress = _json(config.output_dir / "production_progress.json")

    assert report["status"] == "pass"
    assert report["production_progress_path"].endswith("production_progress.json")
    assert progress["schema_version"] == "production_progress_v1"
    assert progress["status"] == "complete"
    assert progress["phase"] == "complete"
    assert progress["production_status"] == "pass"
    assert progress["score_pass"]["status"] == "complete"
    assert progress["score_pass"]["examples_processed"] == 5
    assert progress["score_pass"]["examples_total"] == 5
    assert progress["score_pass"]["shard_count_written"] == 3
    assert progress["validation"]["status"] == "complete"
    assert progress["report_writing"]["status"] == "complete"


def test_production_config_passes_dynamic_top_k_controls_to_backend(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        dynamic_top_k_min=3,
        dynamic_top_k_max=128,
        dynamic_mass_threshold=0.975,
    )

    backend_config = production._backend_config(config)

    assert backend_config.dynamic_top_k_min == 3
    assert backend_config.dynamic_top_k_max == 128
    assert backend_config.dynamic_mass_threshold == 0.975


def test_c6_resume_hash_changes_for_every_policy_input(tmp_path: Path) -> None:
    baseline = _config(
        tmp_path / "baseline",
        selection_integration_policy="corridor_first_global_backfill_v1",
        total_selected_exemplar_budget=8,
    )
    baseline_hash = production._selection_integration_hash(baseline)
    mutations = (
        {"fingerprint_corridor_budget_fraction": "0.25"},
        {"fingerprint_corridor_budget_max": 3},
        {"fingerprint_corridor_mode_cap": 2},
        {"fingerprint_corridor_candidate_pool_cap": 8},
        {"require_full_selected_budget": False},
        {"exemplar_delivery_path": "two_pass_rerun_selected"},
    )

    for index, mutation in enumerate(mutations):
        changed = _config(
            tmp_path / f"changed-{index}",
            selection_integration_policy="corridor_first_global_backfill_v1",
            total_selected_exemplar_budget=8,
            **mutation,
        )
        assert production._selection_integration_hash(changed) != baseline_hash


@pytest.mark.parametrize(
    "delivery_path",
    ("one_pass_pruned_candidate", "two_pass_rerun_selected"),
)
def test_c6_cpu_path_generates_features_audit_and_curriculum(
    tmp_path: Path,
    delivery_path: str,
) -> None:
    config = _config(
        tmp_path / "c6",
        target_policy="corridor_exemplar_v1",
        vocab_size=64,
        top_k=32,
        exemplar_selection_enabled=True,
        exemplar_delivery_path=delivery_path,
        retain_unselected_exemplar_payloads=False,
        selection_integration_policy="corridor_first_global_backfill_v1",
        total_selected_exemplar_budget=4,
    )
    report = build_production_gpu_tome(config)

    assert report["status"] == "pass", report["blockers"]
    assert (
        _json(
            config.output_dir / "reports" / "c6_integrated_selection_validation.json"
        )["status"]
        == "pass"
    )
    assert (
        _json(config.output_dir / "selected_linkage_audit.json")["c6_integration"][
            "status"
        ]
        == "pass"
    )
    assert (
        _json(config.output_dir / "curriculum" / "selected_routes.json")[
            "unique_coordinate_count"
        ]
        == 4
    )
    assert (config.output_dir / "c6" / "corridor-features" / "manifest.json").is_file()
    authority = _json(config.output_dir / "c6" / "authority_manifest.json")
    assert authority["production_grade"] is True
    assert authority["score_pass_authority_hash"]
    assert (config.output_dir / "c6" / "global-board-supply.json").is_file()
    assert (config.output_dir / "c6" / "source-passports.json").is_file()
    global_supply = _json(config.output_dir / "c6" / "global-board-supply.json")
    passports = _json(config.output_dir / "c6" / "source-passports.json")
    features = _json(config.output_dir / "c6" / "corridor-features" / "manifest.json")
    authority_hash = authority["score_pass_authority_hash"]
    assert (
        global_supply["source_provenance"]["score_pass_authority_hash"]
        == authority_hash
    )
    assert passports["score_pass_authority_hash"] == authority_hash
    assert features["score_pass_authority_hash"] == authority_hash
    assert report["full_teacher_pass_count"] == 1
    assert report["external_authority_override_used"] is False
    if delivery_path == "two_pass_rerun_selected":
        assert report["selected_teacher_rerun_count"] == 1
        assert report["legacy_selected_teacher_rerun_count"] == 0
        assert report["native_c6_selected_teacher_rerun_count"] == 1
        # One rerun input can satisfy multiple selected positions from the
        # same example; payload count remains the C5 coordinate count.
        assert 0 < report["selected_teacher_rerun_example_count"] <= 4
        assert report["num_selected_exemplars"] == 4
        emission = _json(config.output_dir / "emission_config.json")
        delivery = _json(config.output_dir / "delivery_report.json")
        assert emission["exemplar_selection_enabled"] is True
        assert emission["selection_integration_policy"] == (
            "corridor_first_global_backfill_v1"
        )
        assert emission["exemplar_selection_manifest"] == (
            "c6/production_global_selector.json"
        )
        assert delivery["execution_mode"] == "native_c6_path_b_v1"
        assert delivery["selected_rerun_batch_count"] == 1
        assert delivery["selected_rerun_peak_host_memory_bytes"] > 0
        assert "score_pass_complete" in report["phase_host_memory_bytes"]
        assert "c2_c5_selection_complete" in report["phase_host_memory_bytes"]
    else:
        assert report["selected_teacher_rerun_count"] == 0
    package = tmp_path / f"student-{delivery_path}"
    package_tome_artifact(config.output_dir, package, profile=STUDENT, overwrite=True)
    assert validate_tome_package(package, profile=STUDENT).ok
    assert (
        _json(package / "selected_linkage_audit.json")["c6_integration"]["status"]
        == "pass"
    )


def test_c6_resume_finalization_reuses_complete_delivery_without_teacher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        target_policy="corridor_exemplar_v1",
        vocab_size=64,
        top_k=32,
        exemplar_selection_enabled=True,
        exemplar_delivery_path="two_pass_rerun_selected",
        retain_unselected_exemplar_payloads=False,
        selection_integration_policy="corridor_first_global_backfill_v1",
        total_selected_exemplar_budget=4,
    )
    first = build_production_gpu_tome(config)
    (config.output_dir / "reports" / "c6_integrated_selection_validation.json").unlink()

    resume_config = _config(
        tmp_path,
        output_dir=config.output_dir,
        resume=True,
        target_policy="corridor_exemplar_v1",
        vocab_size=64,
        top_k=32,
        exemplar_selection_enabled=True,
        exemplar_delivery_path="two_pass_rerun_selected",
        retain_unselected_exemplar_payloads=False,
        selection_integration_policy="corridor_first_global_backfill_v1",
        total_selected_exemplar_budget=4,
    )
    eligibility = production.probe_c6_finalization_only_resume(resume_config)
    assert eligibility.eligible, eligibility.to_dict()

    def fail_if_teacher_runs(*args, **kwargs):
        raise AssertionError("teacher pass must not run during finalization resume")

    monkeypatch.setattr(
        production, "build_streaming_backend_teacher_textbook", fail_if_teacher_runs
    )
    monkeypatch.setattr(
        production,
        "build_runtime_doctor_report",
        fail_if_teacher_runs,
    )
    monkeypatch.setattr(production, "build_gpu_run_plan", fail_if_teacher_runs)
    monkeypatch.setattr(
        production,
        "materialize_selected_exemplar_delivery",
        fail_if_teacher_runs,
    )
    resumed = build_production_gpu_tome(resume_config)

    assert first["status"] == "pass"
    assert resumed["status"] == "pass", resumed["blockers"]
    assert resumed["resume_finalization_only"] is True
    assert resumed["teacher_pass_resumed"] is False
    assert resumed["selected_delivery_status"] == "pass"


def test_c6_finalization_probe_rejects_incomplete_delivery_before_doctor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        target_policy="corridor_exemplar_v1",
        vocab_size=64,
        top_k=32,
        exemplar_selection_enabled=True,
        exemplar_delivery_path="two_pass_rerun_selected",
        retain_unselected_exemplar_payloads=False,
        selection_integration_policy="corridor_first_global_backfill_v1",
        total_selected_exemplar_budget=4,
    )
    first = build_production_gpu_tome(config)
    (config.output_dir / "reports" / "c6_integrated_selection_validation.json").unlink()
    payload_path = next(
        (config.output_dir / "selected_exemplars").glob("selected-exemplars-*.json")
    )
    payload_path.unlink()
    resume_config = _config(
        tmp_path,
        output_dir=config.output_dir,
        resume=True,
        target_policy="corridor_exemplar_v1",
        vocab_size=64,
        top_k=32,
        exemplar_selection_enabled=True,
        exemplar_delivery_path="two_pass_rerun_selected",
        retain_unselected_exemplar_payloads=False,
        selection_integration_policy="corridor_first_global_backfill_v1",
        total_selected_exemplar_budget=4,
    )
    eligibility = production.probe_c6_finalization_only_resume(resume_config)
    assert eligibility.eligible is False
    assert any(
        "payload_shard_count_mismatch" in reason for reason in eligibility.reasons
    )

    doctor_called = False

    def doctor_probe(_config):
        nonlocal doctor_called
        doctor_called = True
        raise RuntimeError("accelerator preflight required")

    monkeypatch.setattr(production, "build_runtime_doctor_report", doctor_probe)
    with pytest.raises(RuntimeError, match="accelerator preflight required"):
        build_production_gpu_tome(resume_config)
    assert doctor_called
    assert first["status"] == "pass"


def test_c6_finalization_probe_rejects_configuration_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        target_policy="corridor_exemplar_v1",
        vocab_size=64,
        top_k=32,
        exemplar_selection_enabled=True,
        exemplar_delivery_path="two_pass_rerun_selected",
        retain_unselected_exemplar_payloads=False,
        selection_integration_policy="corridor_first_global_backfill_v1",
        total_selected_exemplar_budget=4,
    )
    first = build_production_gpu_tome(config)
    assert first["status"] == "pass", first["blockers"]
    resume_config = _config(
        tmp_path,
        output_dir=config.output_dir,
        resume=True,
        dynamic_top_k_max=128,
        target_policy="corridor_exemplar_v1",
        vocab_size=64,
        top_k=32,
        exemplar_selection_enabled=True,
        exemplar_delivery_path="two_pass_rerun_selected",
        retain_unselected_exemplar_payloads=False,
        selection_integration_policy="corridor_first_global_backfill_v1",
        total_selected_exemplar_budget=4,
    )

    eligibility = production.probe_c6_finalization_only_resume(resume_config)

    assert eligibility.eligible is False
    assert any("dynamic_top_k_max_mismatch" in reason for reason in eligibility.reasons)

    doctor_called = False

    def doctor_probe(_config):
        nonlocal doctor_called
        doctor_called = True
        raise RuntimeError("accelerator preflight required")

    monkeypatch.setattr(production, "build_runtime_doctor_report", doctor_probe)
    with pytest.raises(RuntimeError, match="accelerator preflight required"):
        build_production_gpu_tome(resume_config)
    assert doctor_called


def test_c6_underfilled_budget_stops_before_native_selected_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        target_policy="corridor_exemplar_v1",
        vocab_size=64,
        top_k=32,
        exemplar_selection_enabled=True,
        exemplar_delivery_path="two_pass_rerun_selected",
        retain_unselected_exemplar_payloads=False,
        selection_integration_policy="corridor_first_global_backfill_v1",
        total_selected_exemplar_budget=99,
        require_full_selected_budget=True,
    )

    def unexpected_delivery(*args, **kwargs):
        raise AssertionError("native selected rerun must not start under budget")

    monkeypatch.setattr(
        production,
        "materialize_selected_exemplar_delivery",
        unexpected_delivery,
    )
    report = build_production_gpu_tome(config)

    assert report["status"] == "fail"
    assert report["build_status"] == "selection_underfilled_before_selected_rerun"
    diagnostics = _json(config.output_dir / "c6" / "selection_budget_diagnostics.json")
    assert diagnostics["budget_shortfall"] > 0
    assert diagnostics["budget_shortfall_reason"]
    assert not (config.output_dir / "delivery_report.json").exists()


def test_c6_selected_rerun_batch_override_is_independent_of_score_batch(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        target_policy="corridor_exemplar_v1",
        vocab_size=64,
        top_k=32,
        gpu_batch_size_preset=1,
        exemplar_selection_enabled=True,
        exemplar_delivery_path="two_pass_rerun_selected",
        retain_unselected_exemplar_payloads=False,
        selection_integration_policy="corridor_first_global_backfill_v1",
        total_selected_exemplar_budget=4,
        selected_rerun_batch_size=2,
    )

    report = build_production_gpu_tome(config)
    delivery = _json(config.output_dir / "delivery_report.json")

    assert report["status"] == "pass", report["blockers"]
    assert report["effective_batch_size"] == 1
    assert delivery["selected_rerun_batch_size"] == 2
    assert delivery["selected_rerun_batch_count"] == 1
    assert delivery["selected_payload_shard_count"] == 4
    shards = sorted(
        (config.output_dir / "selected_exemplars").glob("selected-exemplars-*.json")
    )
    assert len(shards) == 4
    assert all(len(_json(path)["selected_exemplars"]) == 1 for path in shards)
    assert (config.output_dir / "selected_exemplars" / "payload_index.json").is_file()


def test_c6_budget_diagnostics_use_unique_coordinate_sets() -> None:
    config = SimpleNamespace(total_selected_exemplar_budget=4)
    claims = SimpleNamespace(
        selected_coordinates=(object(), object(), object()),
        corridor_claims=(
            SimpleNamespace(example_id="a", position=0),
            SimpleNamespace(example_id="b", position=0),
        ),
        global_claims=(SimpleNamespace(example_id="c", position=0, global_rank=3),),
        collision_obligations=(object(),),
        summary={"board_summaries": []},
    )
    leaderboards = SimpleNamespace(
        modes=(
            SimpleNamespace(
                candidates=(
                    SimpleNamespace(candidate_id="a", position=0),
                    SimpleNamespace(candidate_id="b", position=0),
                )
            ),
        )
    )
    plan = SimpleNamespace(modes=(SimpleNamespace(allocated_slots=2),))
    global_supply = {
        "boards": [
            {
                "candidates": [
                    {"example_id": "b", "position": 0},
                    {"example_id": "c", "position": 0},
                ]
            },
            {"candidates": [{"example_id": "b", "position": 0}]},
        ]
    }

    diagnostics = production._c6_budget_diagnostics(
        config,
        claims=claims,
        leaderboards=leaderboards,
        plan=plan,
        global_supply=global_supply,
    )

    assert diagnostics["fingerprint_corridor_budget_requested"] == 2
    assert diagnostics["fingerprint_corridor_candidates_eligible_unique"] == 2
    assert diagnostics["global_supply_exported"] == 2
    assert diagnostics["within_role_duplicate_count"] == 1
    assert diagnostics["cross_role_duplicate_count"] == 1
    assert diagnostics["fingerprint_corridor_global_jaccard"] == pytest.approx(1 / 3)
    assert (
        diagnostics["budget_shortfall_reason"]
        == "insufficient_eligible_unique_candidates"
    )


def _c6_source_inputs(
    artifact: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    assignments = _json(artifact / "corridors" / "mode_assignments.json")
    metadata_path = artifact / assignments["examples_metadata"]["path"]
    example_ids = [
        json.loads(line)["example_id"]
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
    ]
    arrays = assignments["arrays"]
    example_index = np.load(
        artifact / arrays["position_example_index"]["path"], allow_pickle=False
    )
    positions = np.load(artifact / arrays["position"]["path"], allow_pickle=False)
    mode_ids = np.load(artifact / arrays["mode_id"]["path"], allow_pickle=False)
    mode_by_coordinate = {
        (example_ids[int(index)], int(position)): int(mode_id)
        for index, position, mode_id in zip(
            example_index, positions, mode_ids, strict=True
        )
    }
    store = TeacherTargetStore.open(artifact)
    passports: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    offset = 0
    for shard_id in range(store.metadata.shard_count):
        shard = store.read_shard(shard_id)
        rows = int(np.asarray(shard["input_ids"]).shape[0])
        for row in range(rows):
            example_id = example_ids[offset + row]
            for position in range(store.metadata.sequence_length):
                entropy = float(np.asarray(shard["corridor_entropy"])[row, position])
                passports.append(
                    {
                        "example_id": example_id,
                        "position": position,
                        "source_shard_id": shard_id,
                        "source_row": row,
                        "source_position": position,
                        "source_score": entropy,
                        "source_top_token_id": int(
                            np.asarray(shard["exemplar_source_top_token_ids"])[
                                row, position, 0
                            ]
                        ),
                        "source_score_policy": "entropy_top_n_v1",
                        "corridor_mode_id": mode_by_coordinate[(example_id, position)],
                        "corridor_assignment_status": "linked",
                    }
                )
                candidates.append(
                    {
                        "example_id": example_id,
                        "position": position,
                        "rank": len(candidates) + 1,
                        "score": entropy,
                        "eligible": True,
                    }
                )
        offset += rows
    candidates.sort(
        key=lambda item: (
            -float(item["score"]),
            str(item["example_id"]),
            int(item["position"]),
        )
    )
    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank
    return passports, {
        "schema_version": "radjax.c4_global_board_supply.v1",
        "source_provenance": {
            "production_grade": True,
            "source_artifact_id": "test",
            "selector_policy": "multi_leaderboard_exemplar_selector_v1",
            "selector_schema_version": "exemplar_selection_manifest_v1",
        },
        "boards": [
            {
                "board_id": "global_max_entropy",
                "priority": 0,
                "requested_slots": 4,
                "candidates": candidates,
            }
        ],
    }


def test_production_build_stops_on_planner_failure(tmp_path: Path) -> None:
    config = _config(tmp_path, max_artifact_bytes=1)

    report = build_production_gpu_tome(config)

    assert report["status"] == "fail"
    assert report["run_plan_status"] == "fail"
    assert any("estimated artifact size" in item for item in report["blockers"])
    assert (config.output_dir / "run_plan.json").is_file()
    assert not (config.output_dir / "metadata.json").exists()


def test_production_build_warns_by_default_and_can_fail_on_plan_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_plan(plan_config):
        status = "fail" if plan_config.fail_on_warnings else "warn"
        blockers = (
            ["fail_on_warnings enabled and planner emitted warnings"]
            if plan_config.fail_on_warnings
            else []
        )
        return {
            "status": status,
            "warnings": ["planner warning"],
            "blockers": blockers,
            "resolved_batch_policy": {"effective_gpu_batch_size": 2},
            "artifact_estimates": {"estimated_total_artifact_bytes": 123},
        }

    monkeypatch.setattr(production, "build_gpu_run_plan", fake_plan)
    warn_report = build_production_gpu_tome(_config(tmp_path / "warn"))
    fail_report = build_production_gpu_tome(
        _config(tmp_path / "fail", fail_on_plan_warnings=True)
    )

    assert warn_report["status"] == "warn"
    assert warn_report["run_plan_status"] == "warn"
    assert warn_report["effective_batch_size"] == 2
    assert fail_report["status"] == "fail"
    assert fail_report["run_plan_status"] == "fail"
    assert any("fail_on_warnings" in item for item in fail_report["blockers"])
    assert not (Path(fail_report["output_dir"]) / "metadata.json").exists()


def test_production_build_honors_no_build_if_plan_warn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        production,
        "build_gpu_run_plan",
        lambda _config: {
            "status": "warn",
            "warnings": ["planner warning"],
            "blockers": [],
            "resolved_batch_policy": {"effective_gpu_batch_size": 2},
        },
    )

    report = build_production_gpu_tome(_config(tmp_path, no_build_if_plan_warn=True))

    assert report["status"] == "fail"
    assert any("no_build_if_plan_warn" in item for item in report["blockers"])
    assert not (config_output := Path(report["output_dir"]) / "metadata.json").exists()
    assert config_output.name == "metadata.json"


def test_production_build_passes_effective_batch_to_streaming_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        production,
        "build_gpu_run_plan",
        lambda _config: {
            "status": "pass",
            "warnings": [],
            "blockers": [],
            "resolved_batch_policy": {"effective_gpu_batch_size": 3},
        },
    )

    config = _config(tmp_path, shard_size_examples=3)
    report = build_production_gpu_tome(config)
    manifest = _json(config.output_dir / "run_manifest.json")

    assert report["status"] == "pass"
    assert report["effective_batch_size"] == 3
    assert manifest["batch_size"] == 3


def test_production_build_missing_inputs_fail_before_build(tmp_path: Path) -> None:
    missing_corpus = build_production_gpu_tome(
        _config(tmp_path / "corpus_missing", corpus_manifest_path=tmp_path / "missing")
    )
    missing_provenance = build_production_gpu_tome(
        _config(
            tmp_path / "provenance_missing",
            teacher_model_provenance_path=tmp_path / "missing_provenance.json",
        )
    )

    assert missing_corpus["status"] == "fail"
    assert any(
        "corpus manifest path missing" in item for item in missing_corpus["blockers"]
    )
    assert missing_provenance["status"] == "fail"
    assert any(
        "teacher model provenance path missing" in item
        for item in missing_provenance["blockers"]
    )


def test_production_build_resume_completed_reports_already_complete(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    first = build_production_gpu_tome(config)
    resumed = build_production_gpu_tome(
        _config(tmp_path, output_dir=config.output_dir, resume=True)
    )

    assert first["status"] == "pass"
    assert resumed["status"] == "pass"
    assert resumed["resume_requested"] is True
    assert resumed["already_complete"] is True


def test_production_build_completed_resume_skips_failing_planner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    first = build_production_gpu_tome(config)

    def fail_if_called(_config):
        raise AssertionError("planner should not run")

    monkeypatch.setattr(production, "build_gpu_run_plan", fail_if_called)
    resumed = build_production_gpu_tome(
        _config(tmp_path, output_dir=config.output_dir, resume=True)
    )

    assert first["status"] == "pass"
    assert resumed["status"] == "pass"
    assert resumed["already_complete"] is True
    assert resumed["resume_requested"] is True
    assert resumed["validation_status"] == "pass"
    assert resumed["run_plan_status"] == "not_run"
    assert resumed["build_status"] == "already_complete"
    assert "planner should not run" not in resumed["blockers"]
    assert (config.output_dir / "production_build_report.json").is_file()


def test_production_build_completed_invalid_resume_fails_without_planner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    first = build_production_gpu_tome(config)
    (config.output_dir / "metadata.json").unlink()

    def fail_if_called(_config):
        raise AssertionError("planner should not run")

    monkeypatch.setattr(production, "build_gpu_run_plan", fail_if_called)
    resumed = build_production_gpu_tome(
        _config(tmp_path, output_dir=config.output_dir, resume=True)
    )

    assert first["status"] == "pass"
    assert resumed["status"] == "fail"
    assert resumed["already_complete"] is True
    assert resumed["validation_status"] == "fail"
    assert resumed["run_plan_status"] == "not_run"
    assert resumed["build_status"] == "already_complete_invalid"
    assert resumed["blockers"]
    assert "planner should not run" not in resumed["blockers"]


def test_production_build_overwrite_rebuilds_and_preserves_run_plan(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    first = build_production_gpu_tome(config)
    stale = config.output_dir / "stale.txt"
    stale.write_text("old", encoding="utf-8")

    rebuilt = build_production_gpu_tome(
        _config(tmp_path, output_dir=config.output_dir, overwrite=True)
    )

    assert first["status"] == "pass"
    assert rebuilt["status"] == "pass"
    assert not stale.exists()
    assert (config.output_dir / "run_plan.json").is_file()
    assert (config.output_dir / "production_build_report.json").is_file()


def test_production_build_failure_report_is_preserved_on_streaming_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_backend = CPUReferenceTeacherEmissionBackend

    class FailingBackend:
        def __init__(self, config: TeacherBackendConfig) -> None:
            self.backend = real_backend(config)
            self.calls = 0

        def emit_batch(self, batch):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("injected production failure")
            return self.backend.emit_batch(batch)

        def close(self) -> None:
            self.backend.close()

    monkeypatch.setattr(
        "radjax_tome.builder.backend_textbook.create_backend",
        lambda config: FailingBackend(config),
    )

    config = _config(tmp_path, shard_size_examples=2)
    report = build_production_gpu_tome(config)

    assert report["status"] == "fail"
    assert any(
        "completed shards remain available" in item for item in report["blockers"]
    )
    failure = _json(config.output_dir / "failure_report.json")
    assert failure["resume_available"] is True


def test_production_build_records_parity_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_compare(*_args, **_kwargs):
        return TomeParityReport(status="fail", blockers=("parity failed",))

    def fake_write(report: TomeParityReport, path: Path) -> None:
        write_json(path, report.to_dict())

    monkeypatch.setattr(production, "compare_tome_artifacts", fake_compare)
    monkeypatch.setattr(production, "write_tome_parity_report", fake_write)

    config = _config(tmp_path, parity_left=tmp_path / "baseline")
    report = build_production_gpu_tome(config)

    assert report["status"] == "fail"
    assert report["parity_status"] == "fail"
    assert any("parity failed" in item for item in report["blockers"])
    assert (config.output_dir / "parity_report.json").is_file()


def test_production_build_custom_manifest_and_progress_paths_are_truthful(
    tmp_path: Path,
) -> None:
    output = tmp_path / "production_tome"
    config = _config(
        tmp_path,
        output_dir=output,
        run_manifest_path=output / "custom" / "run_manifest.json",
        progress_log_path=output / "custom" / "progress_log.jsonl",
    )

    report = build_production_gpu_tome(config)
    metadata = _json(output / "metadata.json")
    teacher_manifest = _json(output / "teacher_manifest.json")
    emission_config = _json(output / "emission_config.json")
    cover_page = _json(output / "cover_page.json")

    assert report["status"] == "pass"
    assert metadata["target_params"]["run_manifest_path"] == "custom/run_manifest.json"
    assert metadata["target_params"]["progress_log_path"] == "custom/progress_log.jsonl"
    assert teacher_manifest["run_manifest_path"] == "custom/run_manifest.json"
    assert emission_config["progress_log_path"] == "custom/progress_log.jsonl"
    assert cover_page["streaming"]["run_manifest_path"] == "custom/run_manifest.json"
    assert report["run_manifest_path"].endswith("custom/run_manifest.json")
    assert report["progress_log_path"].endswith("custom/progress_log.jsonl")


def test_production_build_cli_smoke_and_no_allow_downloads_flag(
    tmp_path: Path,
) -> None:
    dataset, corpus_manifest, provenance_path, model = _production_inputs(tmp_path)
    output = tmp_path / "cli_production"

    result = run_cli(
        ROOT,
        "production-build",
        "--teacher-backend",
        "cpu_reference",
        "--runtime-mode",
        "cpu",
        "--target-policy",
        "dynamic",
        "--teacher-model",
        str(model),
        "--dataset",
        str(dataset),
        "--corpus-manifest",
        str(corpus_manifest),
        "--teacher-model-provenance",
        str(provenance_path),
        "--output",
        str(output),
        "--gpu-batch-size-mode",
        "preset",
        "--gpu-batch-size-preset",
        "2",
        "--shard-size-examples",
        "2",
        "--max-examples",
        "5",
        "--sequence-length",
        "5",
        "--vocab-size",
        "13",
        "--top-k",
        "4",
        "--dynamic-top-k-max",
        "128",
    )
    help_result = run_cli(ROOT, "production-build", "--help")
    production_report = _json(output / "production_build_report.json")
    emission_config = _json(output / "emission_config.json")

    assert result.returncode == 0, result.stderr
    assert "status=pass" in result.stdout
    assert "phase=score_pass" in result.stdout
    assert (output / "production_build_report.json").is_file()
    assert _json(output / "production_progress.json")["status"] == "complete"
    assert production_report["dynamic_top_k_max"] == 128
    assert production_report["inputs"]["dynamic_top_k_max"] == 128
    assert emission_config["dynamic_top_k_max"] == 128
    assert "--allow-downloads" not in help_result.stdout
    assert "--dynamic-top-k-min" in help_result.stdout
    assert "--dynamic-top-k-max" in help_result.stdout
    assert "--dynamic-mass-threshold" in help_result.stdout
    assert "--progress" in help_result.stdout
    assert "--no-progress" in help_result.stdout


def test_fail_fast_is_not_user_facing() -> None:
    build_help = run_cli(ROOT, "build", "--help")
    production_help = run_cli(ROOT, "production-build", "--help")

    assert build_help.returncode == 0, build_help.stderr
    assert production_help.returncode == 0, production_help.stderr
    assert "--fail-fast" not in build_help.stdout
    assert "--fail-fast" not in production_help.stdout
