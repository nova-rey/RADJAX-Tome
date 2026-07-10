from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import radjax_tome.builder.exemplar_delivery as exemplar_delivery
import radjax_tome.builder.production as production
from radjax_tome.builder import (
    ProductionBuildConfig,
    build_production_gpu_tome,
    compare_exemplar_delivery_artifacts,
    validate_teacher_textbook,
)
from radjax_tome.corpora import CorpusBuildConfig, build_corpus_artifact
from radjax_tome.io.json import write_json
from radjax_tome.provenance import inspect_teacher_model, write_teacher_model_provenance
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


def _production_inputs(
    tmp_path: Path,
    *,
    example_count: int = 5,
) -> tuple[Path, Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    sources = []
    for index in range(example_count):
        source = tmp_path / f"source-{index}.txt"
        source.write_text(f"selected delivery example {index}", encoding="utf-8")
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


def _config(
    tmp_path: Path,
    *,
    output_name: str,
    delivery_path: str,
    retain_unselected: bool = False,
    example_count: int = 5,
    max_examples: int | None = 5,
    gpu_batch_size_preset: int = 2,
    shard_size_examples: int = 2,
    track_delivery_timing: bool = False,
    progress: bool = False,
) -> ProductionBuildConfig:
    dataset, corpus_manifest, provenance_path, model = _production_inputs(
        tmp_path,
        example_count=example_count,
    )
    return ProductionBuildConfig(
        teacher_model=str(model),
        tokenizer_id=str(model),
        dataset_path=dataset,
        corpus_manifest_path=corpus_manifest,
        teacher_model_provenance_path=provenance_path,
        output_dir=tmp_path / output_name,
        teacher_backend="cpu_reference",
        runtime_mode="cpu",
        target_policy="corridor_exemplar_v1",
        sequence_length=5,
        vocab_size=64,
        top_k=4,
        num_buckets=3,
        gpu_batch_size_mode="preset",
        gpu_batch_size_preset=gpu_batch_size_preset,
        shard_size_examples=shard_size_examples,
        max_examples=max_examples,
        exemplar_selection_enabled=True,
        exemplar_delivery_path=delivery_path,
        selected_exemplar_budget=2,
        retain_unselected_exemplar_payloads=retain_unselected,
        track_delivery_timing=track_delivery_timing,
        progress=progress,
    )


def test_cli_exposes_selected_only_exemplar_delivery_flags() -> None:
    production_help = run_cli(ROOT, "production-build", "--help")
    parity_help = run_cli(ROOT, "exemplar-delivery-parity", "--help")

    assert production_help.returncode == 0, production_help.stderr
    assert "--exemplar-delivery-path" in production_help.stdout
    assert "--exemplar-selection-enabled" in production_help.stdout
    assert "--selected-exemplar-budget" in production_help.stdout
    assert "--no-retain-unselected-exemplar-payloads" in production_help.stdout
    assert "--exemplar-score-policy" in production_help.stdout
    assert "--track-delivery-timing" in production_help.stdout
    assert parity_help.returncode == 0, parity_help.stderr
    assert "--path-a" in parity_help.stdout
    assert "--path-b" in parity_help.stdout


def test_path_b_selected_only_delivery_writes_payloads_without_unselected_retention(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        output_name="path_b",
        delivery_path="two_pass_rerun_selected",
    )

    report = build_production_gpu_tome(config)
    output = config.output_dir
    delivery = _json(output / "delivery_report.json")
    selected = _json(output / "leaderboards" / "selected_exemplars.json")
    payload = _json(output / "selected_exemplars" / "selected-exemplars-00000.json")

    assert report["status"] == "pass"
    assert report["delivery_path"] == "two_pass_rerun_selected"
    assert report["num_examples_scored"] == 5
    assert report["num_selected_exemplars"] == 2
    assert report["non_selected_exemplar_payload_retained"] is False
    assert delivery["num_positions_scored"] == 25
    assert delivery["teacher_rerun_count"] == delivery["selected_example_count"]
    assert "timing_enabled" not in delivery
    assert "timing_enabled" not in report
    assert selected["selected_exemplars"]
    selected_payload = payload["selected_exemplars"][0]
    for field in (
        "selected_example_id",
        "selected_position",
        "selected_score",
        "top_token_ids",
        "top_log_probs",
        "top_probs",
        "top_selection_mask",
        "bucket_masses",
        "dynamic_top_k",
    ):
        assert field in selected_payload
    assert not (output / "unselected_candidate_payloads").exists()
    assert validate_teacher_textbook(output).status == "pass"


def test_path_b_progress_sidecar_records_rerun_and_corridor_export(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        output_name="path_b_progress",
        delivery_path="two_pass_rerun_selected",
        progress=True,
    )

    report = build_production_gpu_tome(config)
    progress = _json(config.output_dir / "production_progress.json")
    delivery = _json(config.output_dir / "delivery_report.json")

    assert report["status"] == "pass"
    assert progress["status"] == "complete"
    assert progress["selected_rerun"]["status"] == "complete"
    assert (
        progress["selected_rerun"]["selected_examples_processed"]
        == delivery["selected_example_count"]
    )
    assert (
        progress["selected_rerun"]["selected_examples_total"]
        == delivery["selected_example_count"]
    )
    assert progress["corridor_export"]["status"] == "complete"
    assert (
        progress["corridor_export"]["positions_processed"]
        == report["corridor_positions_used"]
    )
    assert (
        progress["corridor_export"]["modes_discovered"] == report["corridor_mode_count"]
    )
    assert (
        progress["corridor_export"]["fingerprints_discovered"]
        == report["corridor_fingerprint_count"]
    )
    assert progress["corridor_export"]["assignment_storage_kind"] == "packed_numpy_v1"


def test_path_b_main_artifact_is_score_pass_without_broad_exemplar_payloads(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        output_name="path_b_score_pass",
        delivery_path="two_pass_rerun_selected",
    )

    assert build_production_gpu_tome(config)["status"] == "pass"
    metadata = _json(config.output_dir / "metadata.json")
    shard_path = config.output_dir / "shards" / "shard-00000.npz"

    assert metadata["target_type"] == "corridor_exemplar_score_pass_v1"
    with np.load(shard_path) as shard:
        assert "corridor_top_token_ids" in shard.files
        assert "corridor_teacher_entropy" in shard.files
        assert "corridor_confidence" in shard.files
        assert "score_selected_position" in shard.files
        assert "score_selected_position_entropy" in shard.files
        assert "exemplar_positions" not in shard.files
        assert "exemplar_selection_mask" not in shard.files
        assert "exemplar_source_top_mass" not in shard.files


def test_selected_payload_linkage_matches_score_pass_rows(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        output_name="path_b_linkage",
        delivery_path="two_pass_rerun_selected",
    )

    assert build_production_gpu_tome(config)["status"] == "pass"
    _assert_selected_source_coordinate_invariants(config.output_dir)
    selected_path = config.output_dir / "leaderboards" / "selected_exemplars.json"
    selected = _json(selected_path)
    selected["selected_exemplars"][0]["selected_position"] += 1
    write_json(selected_path, selected)

    validation = validate_teacher_textbook(config.output_dir)

    assert validation.status == "fail"
    assert (
        "selected exemplar linkage mismatch: selected record/payload does not match "
        "source candidate coordinate"
    ) in validation.blockers


def test_path_a_selected_only_delivery_prunes_temporary_candidates(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        output_name="path_a",
        delivery_path="one_pass_pruned_candidate",
    )

    report = build_production_gpu_tome(config)
    delivery = _json(config.output_dir / "delivery_report.json")

    assert report["status"] == "pass"
    assert delivery["delivery_path"] == "one_pass_pruned_candidate"
    assert delivery["teacher_rerun_count"] == 0
    assert delivery["selected_payload_source"] == "one_pass_candidate_shard_capture"
    assert delivery["pruned_candidate_payload_bytes"] > 0
    assert delivery["temporary_candidate_bytes"] > 0
    assert delivery["non_selected_exemplar_payload_retained"] is False
    _assert_selected_source_coordinate_invariants(config.output_dir)
    with np.load(config.output_dir / "shards" / "shard-00000.npz") as shard:
        assert not any(key.startswith("exemplar_source_") for key in shard.files)
    assert not (config.output_dir / "temporary_candidates").exists()
    assert not (config.output_dir / "unselected_candidate_payloads").exists()
    assert validate_teacher_textbook(config.output_dir).status == "pass"


def test_path_a_materializes_payloads_from_capture_without_backend_rerun(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fail_create_backend(_backend_config):
        raise AssertionError("Path A must not rerun the backend")

    monkeypatch.setattr(exemplar_delivery, "create_backend", fail_create_backend)
    config = _config(
        tmp_path,
        output_name="path_a_no_rerun",
        delivery_path="one_pass_pruned_candidate",
    )

    assert build_production_gpu_tome(config)["status"] == "pass"
    payload = _json(
        config.output_dir / "selected_exemplars" / "selected-exemplars-00000.json"
    )
    selected_payload = payload["selected_exemplars"][0]

    assert selected_payload["dynamic_top_k"]["source_payload"] == (
        "one_pass_candidate_shard"
    )
    assert len(selected_payload["top_token_ids"]) >= 32
    assert len(selected_payload["bucket_masses"]) == config.num_buckets


def test_path_a_one_pass_payload_uses_candidate_payload_ref_slot() -> None:
    shard = _compact_one_pass_shard_for_slot_test()
    record = _path_a_slot_record(candidate_rank=1)
    config = SimpleNamespace(
        sequence_length=5,
        vocab_size=512,
        num_buckets=3,
        top_k=2,
        score_policy="entropy_top_n_v1",
        backend_config=None,
    )

    payload = exemplar_delivery._selected_payload_from_one_pass_shard(
        record,
        shard=shard,
        row=0,
        config=config,
    )

    assert payload["top_token_ids"][0] == 222
    assert payload["selected_position"] == 2
    assert payload["selected_score"] == 7.0
    assert payload["teacher_entropy"] == 7.0
    assert payload["payload_ref"] == record["payload_ref"]


def test_path_a_one_pass_payload_searches_candidate_slot_when_ref_rank_is_stale() -> (
    None
):
    shard = _compact_one_pass_shard_for_slot_test()
    record = _path_a_slot_record(candidate_rank=0)
    config = SimpleNamespace(
        sequence_length=5,
        vocab_size=512,
        num_buckets=3,
        top_k=2,
        score_policy="entropy_top_n_v1",
        backend_config=None,
    )

    payload = exemplar_delivery._selected_payload_from_one_pass_shard(
        record,
        shard=shard,
        row=0,
        config=config,
    )

    assert payload["top_token_ids"][0] == 222
    assert np.allclose(payload["top_probs"], [0.7, 0.2])
    assert payload["payload_ref"] == record["payload_ref"]


def test_path_a_one_pass_payload_uses_source_position_for_full_sequence_arrays() -> (
    None
):
    shard = _full_sequence_one_pass_shard_for_slot_test()
    record = _path_a_slot_record(candidate_rank=1)
    config = SimpleNamespace(
        sequence_length=5,
        vocab_size=512,
        num_buckets=3,
        top_k=2,
        score_policy="entropy_top_n_v1",
        backend_config=None,
    )

    payload = exemplar_delivery._selected_payload_from_one_pass_shard(
        record,
        shard=shard,
        row=0,
        config=config,
    )

    assert payload["top_token_ids"][0] == 222
    assert payload["selected_position"] == 2


def test_path_a_payload_token_authority_can_differ_from_corridor_token() -> None:
    shard = _full_sequence_one_pass_shard_for_slot_test()
    shard["corridor_top_token_ids"] = np.asarray(
        [[101, 999, 751, 303, 111]],
        dtype=np.int32,
    )
    shard["exemplar_source_top_token_ids"][0, 2, 0] = 649
    record = _path_a_slot_record(candidate_rank=1)
    record["score_top_token_id"] = 751
    record["source_top_token_id"] = 649
    record["payload_ref"] = {
        **record["payload_ref"],
        "source_top_token_id": 649,
    }
    config = SimpleNamespace(
        sequence_length=5,
        vocab_size=512,
        num_buckets=3,
        top_k=2,
        score_policy="entropy_top_n_v1",
        backend_config=None,
    )

    payload = exemplar_delivery._selected_payload_from_one_pass_shard(
        record,
        shard=shard,
        row=0,
        config=config,
    )

    assert payload["score_top_token_id"] == 751
    assert payload["source_top_token_id"] == 649
    assert payload["top_token_ids"][0] == 649
    assert not exemplar_delivery._source_coordinate_linkage_mismatch(
        shard,
        row=0,
        record=record,
        payload=payload,
    )


def test_path_a_slot_mismatch_includes_source_coordinate_diagnostic() -> None:
    shard = _compact_one_pass_shard_for_slot_test()
    shard["exemplar_source_top_token_ids"][0, 1, 0] = 999
    record = _path_a_slot_record(candidate_rank=1)
    config = SimpleNamespace(
        sequence_length=5,
        vocab_size=512,
        num_buckets=3,
        top_k=2,
        score_policy="entropy_top_n_v1",
        backend_config=None,
    )

    with pytest.raises(exemplar_delivery.SelectedExemplarDeliveryError) as raised:
        exemplar_delivery._selected_payload_from_one_pass_shard(
            record,
            shard=shard,
            row=0,
            config=config,
        )

    diagnostic = raised.value.diagnostic
    assert diagnostic["failure_stage"] == "selected_exemplar_delivery"
    assert diagnostic["delivery_path"] == "one_pass_pruned_candidate"
    assert diagnostic["source_position"] == 2
    assert diagnostic["source_top_token_id"] == 222
    assert diagnostic["payload_array_storage_kind"] == "compact_candidate_rank"
    assert diagnostic["exemplar_source_top_token_ids_shape"] == [1, 2, 2]
    assert diagnostic["exemplar_positions_shape"] == [1, 2]
    assert diagnostic["candidate_ranks_searched"][1]["position_match"] is True
    assert diagnostic["candidate_ranks_searched"][1]["top_token_match"] is False


def test_path_a_rejects_payload_ref_coordinate_drift() -> None:
    shard = _compact_one_pass_shard_for_slot_test()
    record = _path_a_slot_record(candidate_rank=1)
    record["source_row"] = 0
    record["payload_ref"] = {
        **record["payload_ref"],
        "source_row": 1,
    }
    config = SimpleNamespace(
        sequence_length=5,
        vocab_size=512,
        num_buckets=3,
        top_k=2,
        score_policy="entropy_top_n_v1",
        backend_config=None,
    )
    store = SimpleNamespace(read_shard=lambda _shard_id: shard)

    with pytest.raises(exemplar_delivery.SelectedExemplarDeliveryError) as raised:
        exemplar_delivery._selected_payloads_from_one_pass_capture(
            [record],
            store=store,
            config=config,
        )

    diagnostic = raised.value.diagnostic
    assert diagnostic["source_row"] == 0
    assert diagnostic["resolved_source_row"] == 0
    assert diagnostic["payload_ref"]["source_row"] == 1
    assert diagnostic["mismatch_fields"] == ["payload_ref.source_row"]


def test_production_report_names_selected_delivery_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config(
        tmp_path,
        output_name="path_a_delivery_failure",
        delivery_path="one_pass_pruned_candidate",
    )
    diagnostic = {
        "failure_stage": "selected_exemplar_delivery",
        "delivery_path": "one_pass_pruned_candidate",
        "selected_example_id": "example-3",
        "source_row": 3,
        "source_position": 2,
    }

    def fail_delivery(_config):
        raise exemplar_delivery.SelectedExemplarDeliveryError(diagnostic)

    monkeypatch.setattr(
        production,
        "materialize_selected_exemplar_delivery",
        fail_delivery,
    )

    report = build_production_gpu_tome(config)

    assert report["status"] == "fail"
    assert report["validation_status"] == "pass"
    assert report["selected_delivery_status"] == "fail"
    assert report["failure_stage"] == "selected_exemplar_delivery"
    assert report["selected_delivery_failure"] == diagnostic
    assert not (config.output_dir / "delivery_report.json").exists()


def test_path_a_path_b_delivery_parity_matches_selection_and_scores(
    tmp_path: Path,
) -> None:
    path_a = _config(
        tmp_path / "a",
        output_name="path_a",
        delivery_path="one_pass_pruned_candidate",
    )
    path_b = _config(
        tmp_path / "b",
        output_name="path_b",
        delivery_path="two_pass_rerun_selected",
    )
    assert build_production_gpu_tome(path_a)["status"] == "pass"
    assert build_production_gpu_tome(path_b)["status"] == "pass"

    report = compare_exemplar_delivery_artifacts(
        path_a.output_dir,
        path_b.output_dir,
        output=tmp_path / "parity_report.json",
    )

    assert report["status"] == "pass"
    assert report["selected_example_ids_match"] is True
    assert report["selected_positions_match"] is True
    assert report["selected_score_ranks_match"] is True
    assert report["path_a_teacher_rerun_count"] == 0
    assert (
        report["path_b_teacher_rerun_count"]
        == _json(path_b.output_dir / "delivery_report.json")["selected_example_count"]
    )


def test_path_a_path_b_1000_example_selected_only_parity_and_pruning(
    tmp_path: Path,
) -> None:
    path_a = _config(
        tmp_path / "a1000",
        output_name="path_a",
        delivery_path="one_pass_pruned_candidate",
        example_count=1000,
        max_examples=1000,
        gpu_batch_size_preset=64,
        shard_size_examples=250,
        track_delivery_timing=True,
    )
    path_b = _config(
        tmp_path / "b1000",
        output_name="path_b",
        delivery_path="two_pass_rerun_selected",
        example_count=1000,
        max_examples=1000,
        gpu_batch_size_preset=64,
        shard_size_examples=250,
        track_delivery_timing=True,
    )

    assert build_production_gpu_tome(path_a)["status"] == "pass"
    assert build_production_gpu_tome(path_b)["status"] == "pass"
    report = compare_exemplar_delivery_artifacts(
        path_a.output_dir,
        path_b.output_dir,
        output=tmp_path / "parity_1000.json",
    )
    path_a_delivery = _json(path_a.output_dir / "delivery_report.json")
    path_b_delivery = _json(path_b.output_dir / "delivery_report.json")
    path_a_production = _json(path_a.output_dir / "production_build_report.json")
    path_b_production = _json(path_b.output_dir / "production_build_report.json")

    assert report["status"] == "pass"
    assert report["blockers"] == []
    assert report["selected_example_ids_match"] is True
    assert report["selected_positions_match"] is True
    assert report["selected_score_ranks_match"] is True
    assert report["selected_mode_keys_match"] is True
    assert report["payload_shape_compatible"] is True
    assert report["corridor_artifact_shape_match"] is True
    assert report["path_a_corridor_mode_count"] >= 1
    assert report["path_b_corridor_mode_count"] >= 1
    assert report["path_a_corridor_mode_policy"] == "stat_bands_v0"
    assert report["path_b_corridor_mode_policy"] == "stat_bands_v0"
    assert report["corridor_mode_policy_match"] is True
    assert report["corridor_mode_count_match"] is True
    assert report["corridor_tracked_stats_match"] is True
    assert report["corridor_mode_table_match"] is True
    assert report["path_a_corridor_mode_count"] <= 125
    assert report["path_b_corridor_mode_count"] <= 125
    assert report["timing_enabled"] is True
    assert report["path_a_wall_seconds"] >= 0.0
    assert report["path_b_wall_seconds"] >= 0.0
    assert report["faster_path"] in {"path_a", "path_b", "tie", "unknown"}
    assert report["timing_claims_not_made"]["no_speed_parity_requirement"] is True
    assert path_a_delivery["teacher_rerun_count"] == 0
    assert path_a_delivery["corridor_mode_policy"] == "stat_bands_v0"
    assert path_b_delivery["corridor_mode_policy"] == "stat_bands_v0"
    assert path_a_delivery["corridor_max_modes"] == 256
    assert path_b_delivery["corridor_max_modes"] == 256
    assert (
        path_a_delivery["corridor_mode_count"] <= path_a_delivery["corridor_max_modes"]
    )
    assert (
        path_b_delivery["corridor_mode_count"] <= path_b_delivery["corridor_max_modes"]
    )
    assert (
        path_b_delivery["teacher_rerun_count"]
        == path_b_delivery["selected_example_count"]
    )
    assert path_a_delivery["timing_enabled"] is True
    assert path_b_delivery["timing_enabled"] is True
    assert path_a_delivery["path_a_wall_seconds"] >= 0.0
    assert path_b_delivery["path_b_wall_seconds"] >= 0.0
    assert path_a_production["timing_enabled"] is True
    assert path_b_production["timing_enabled"] is True
    assert path_a_production["production_wall_seconds"] >= 0.0
    assert path_b_production["production_wall_seconds"] >= 0.0
    assert path_a_production["score_or_main_pass_wall_seconds"] >= 0.0
    assert path_b_production["score_or_main_pass_wall_seconds"] >= 0.0
    assert path_a_delivery["teacher_rerun_count"] == 0
    assert (
        path_b_delivery["teacher_rerun_count"]
        == path_b_delivery["selected_example_count"]
    )
    for shard_path in sorted((path_a.output_dir / "shards").glob("shard-*.npz")):
        with np.load(shard_path) as shard:
            retained = [
                key for key in shard.files if key.startswith("exemplar_source_")
            ]
        assert retained == []


def test_path_b_selected_rerun_invokes_backend_only_for_selected_examples(
    tmp_path: Path,
    monkeypatch,
) -> None:
    invocations: list[tuple[str, ...]] = []
    backend_configs: list[object] = []

    class MarkerBackend:
        def __init__(self, config):
            self.config = config
            backend_configs.append(config)

        def emit_batch(self, batch):
            invocations.append(tuple(batch.example_ids))
            payload = _marker_dynamic_payload(batch, self.config)
            _patch_marker_payload_with_score_pass(
                payload,
                batch=batch,
                dataset_path=config.dataset_path,
                output_dir=config.output_dir,
            )
            return SimpleNamespace(payload=payload)

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        exemplar_delivery,
        "create_backend",
        lambda backend_config: MarkerBackend(backend_config),
    )
    config = _config(
        tmp_path,
        output_name="path_b_backend_marker",
        delivery_path="two_pass_rerun_selected",
    )

    assert build_production_gpu_tome(config)["status"] == "pass"
    delivery = _json(config.output_dir / "delivery_report.json")

    assert len(invocations) == 1
    assert invocations[0] == tuple(delivery["selected_rerun_example_ids"])
    assert len(invocations[0]) == delivery["teacher_rerun_count"]
    assert backend_configs[0].target_policy == "dynamic_cascaded_soft_labels_v1"


def test_selected_payload_values_come_from_backend_emission(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class MarkerBackend:
        def __init__(self, config):
            self.config = config

        def emit_batch(self, batch):
            payload = _marker_dynamic_payload(batch, self.config)
            _patch_marker_payload_with_score_pass(
                payload,
                batch=batch,
                dataset_path=config.dataset_path,
                output_dir=config.output_dir,
            )
            return SimpleNamespace(payload=payload)

    monkeypatch.setattr(
        exemplar_delivery,
        "create_backend",
        lambda backend_config: MarkerBackend(backend_config),
    )
    config = _config(
        tmp_path,
        output_name="path_b_marker_payload",
        delivery_path="two_pass_rerun_selected",
    )

    assert build_production_gpu_tome(config)["status"] == "pass"
    delivery = _json(config.output_dir / "delivery_report.json")
    payload = _json(
        config.output_dir / "selected_exemplars" / "selected-exemplars-00000.json"
    )
    selected_payload = payload["selected_exemplars"][0]
    row = delivery["selected_rerun_example_ids"].index(
        selected_payload["selected_example_id"]
    )
    position = selected_payload["selected_position"]
    score_top_token_id = selected_payload["score_top_token_id"]

    assert selected_payload["top_token_ids"][:3] == [
        score_top_token_id,
        *[_marker_token(row, position, index) for index in range(1, 3)],
    ]
    assert selected_payload["top_probs"][:3] == [
        _marker_prob(row, position, index) for index in range(3)
    ]
    assert selected_payload["top_mass"] == _marker_top_mass(row, position)
    assert selected_payload["bucket_masses"] == [
        _marker_bucket(row, position, index) for index in range(config.num_buckets)
    ]


def test_validation_fails_when_non_selected_payload_retention_is_claimed(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        output_name="retained",
        delivery_path="one_pass_pruned_candidate",
    )
    assert build_production_gpu_tome(config)["status"] == "pass"
    delivery_path = config.output_dir / "delivery_report.json"
    delivery = _json(delivery_path)
    delivery["non_selected_exemplar_payload_retained"] = True
    write_json(delivery_path, delivery)

    validation = validate_teacher_textbook(config.output_dir)

    assert validation.status == "fail"
    assert "non_selected_exemplar_payload_retained=true" in validation.blockers


def test_path_a_validation_fails_when_compact_payload_ref_is_missing(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        output_name="path_a_missing_payload_ref",
        delivery_path="one_pass_pruned_candidate",
    )
    assert build_production_gpu_tome(config)["status"] == "pass"
    selected_path = config.output_dir / "leaderboards" / "selected_exemplars.json"
    selected = _json(selected_path)
    selected["selected_exemplars"][0]["payload_ref"] = None
    write_json(selected_path, selected)

    validation = validate_teacher_textbook(config.output_dir)

    assert validation.status == "fail"
    assert (
        "selected exemplar linkage mismatch: selected record/payload does not match "
        "source candidate coordinate"
    ) in validation.blockers


def _compact_one_pass_shard_for_slot_test() -> dict[str, np.ndarray]:
    return {
        "exemplar_positions": np.asarray([[4, 2]], dtype=np.int32),
        "exemplar_source_top_token_ids": np.asarray(
            [[[111, 11], [222, 22]]],
            dtype=np.int32,
        ),
        "exemplar_source_top_log_probs": np.asarray(
            [[[-0.4, -1.4], [-0.2, -1.2]]],
            dtype=np.float32,
        ),
        "exemplar_source_top_probs": np.asarray(
            [[[0.6, 0.3], [0.7, 0.2]]],
            dtype=np.float32,
        ),
        "exemplar_source_top_selection_mask": np.asarray(
            [[[True, True], [True, True]]],
            dtype=bool,
        ),
        "exemplar_source_effective_top_k": np.asarray([[2, 2]], dtype=np.int32),
        "exemplar_source_top_mass": np.asarray([[0.9, 0.9]], dtype=np.float32),
        "exemplar_source_tail_mass": np.asarray([[0.1, 0.1]], dtype=np.float32),
        "exemplar_source_bucket_masses": np.asarray(
            [[[0.03, 0.03, 0.04], [0.01, 0.04, 0.05]]],
            dtype=np.float32,
        ),
        "corridor_teacher_entropy": np.asarray(
            [[1.0, 2.0, 7.0, 3.0, 4.0]],
            dtype=np.float32,
        ),
    }


def _full_sequence_one_pass_shard_for_slot_test() -> dict[str, np.ndarray]:
    top_token_ids = np.asarray(
        [[[101, 11], [999, 99], [222, 22], [303, 33], [111, 44]]],
        dtype=np.int32,
    )
    top_probs = np.asarray(
        [[[0.6, 0.3], [0.7, 0.2], [0.7, 0.2], [0.6, 0.3], [0.5, 0.4]]],
        dtype=np.float32,
    )
    return {
        "exemplar_positions": np.asarray([[4, 2]], dtype=np.int32),
        "exemplar_source_top_token_ids": top_token_ids,
        "exemplar_source_top_log_probs": -top_probs,
        "exemplar_source_top_probs": top_probs,
        "exemplar_source_top_selection_mask": np.ones_like(top_token_ids, dtype=bool),
        "exemplar_source_effective_top_k": np.full((1, 5), 2, dtype=np.int32),
        "exemplar_source_top_mass": np.full((1, 5), 0.9, dtype=np.float32),
        "exemplar_source_tail_mass": np.full((1, 5), 0.1, dtype=np.float32),
        "exemplar_source_bucket_masses": np.full((1, 5, 3), 0.03, dtype=np.float32),
        "corridor_teacher_entropy": np.asarray(
            [[1.0, 2.0, 7.0, 3.0, 4.0]],
            dtype=np.float32,
        ),
    }


def _path_a_slot_record(*, candidate_rank: int) -> dict[str, object]:
    payload_ref = {
        "kind": "one_pass_candidate_v1",
        "source_shard_id": 3,
        "source_row": 0,
        "source_position": 2,
        "candidate_rank": candidate_rank,
        "position_index": candidate_rank,
        "source_top_token_id": 222,
        "source_score": 7.0,
    }
    return {
        "selected_example_id": "example-slot-b",
        "selected_position": 2,
        "selected_score": 7.0,
        "score_top_token_id": 222,
        "source_shard_id": 3,
        "source_row": 0,
        "source_position": 2,
        "source_score": 7.0,
        "source_top_token_id": 222,
        "source_score_policy": "entropy_top_n_v1",
        "payload_ref": payload_ref,
        "selected_policy": "entropy_top_n_v1",
        "source_delivery_path": "one_pass_pruned_candidate",
    }


def _marker_dynamic_payload(batch, config) -> dict[str, np.ndarray]:
    batch_size = len(batch.example_ids)
    sequence_length = config.sequence_length
    top_k = min(5, config.vocab_size)
    shape = (batch_size, sequence_length, top_k)
    top_token_ids = np.zeros(shape, dtype=np.int32)
    top_probs = np.zeros(shape, dtype=np.float32)
    top_log_probs = np.zeros(shape, dtype=np.float32)
    top_selection_mask = np.ones(shape, dtype=bool)
    bucket_masses = np.zeros(
        (batch_size, sequence_length, config.num_buckets),
        dtype=np.float32,
    )
    top_mass = np.zeros((batch_size, sequence_length), dtype=np.float32)
    tail_mass = np.zeros((batch_size, sequence_length), dtype=np.float32)
    teacher_entropy = np.zeros((batch_size, sequence_length), dtype=np.float32)
    effective_top_k = np.full((batch_size, sequence_length), top_k, dtype=np.int32)
    for row in range(batch_size):
        for position in range(sequence_length):
            for index in range(top_k):
                top_token_ids[row, position, index] = _marker_token(
                    row,
                    position,
                    index,
                )
                top_probs[row, position, index] = _marker_prob(row, position, index)
                top_log_probs[row, position, index] = -_marker_prob(
                    row,
                    position,
                    index,
                )
            for bucket in range(config.num_buckets):
                bucket_masses[row, position, bucket] = _marker_bucket(
                    row,
                    position,
                    bucket,
                )
            top_mass[row, position] = _marker_top_mass(row, position)
            tail_mass[row, position] = 1.0 - _marker_top_mass(row, position)
            teacher_entropy[row, position] = 3.0 + row + (position / 100.0)
    return {
        "top_token_ids": top_token_ids,
        "top_log_probs": top_log_probs,
        "top_probs": top_probs,
        "top_selection_mask": top_selection_mask,
        "effective_top_k": effective_top_k,
        "top_mass": top_mass,
        "tail_mass": tail_mass,
        "bucket_masses": bucket_masses,
        "teacher_entropy": teacher_entropy,
    }


def _patch_marker_payload_with_score_pass(
    payload: dict[str, np.ndarray],
    *,
    batch,
    dataset_path: Path,
    output_dir: Path,
) -> None:
    score_rows = _score_rows_by_example_id(
        dataset_path=dataset_path,
        output_dir=output_dir,
    )
    for row, example_id in enumerate(batch.example_ids):
        score_row = score_rows[str(example_id)]
        position = score_row["position"]
        payload["teacher_entropy"][row, position] = score_row["entropy"]
        payload["top_token_ids"][row, position, 0] = score_row["top_token_id"]


def _score_rows_by_example_id(
    *,
    dataset_path: Path,
    output_dir: Path,
) -> dict[str, dict[str, int | float]]:
    example_ids = [
        str(json.loads(line)["example_id"])
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows: dict[str, dict[str, int | float]] = {}
    offset = 0
    for shard_path in sorted((output_dir / "shards").glob("shard-*.npz")):
        with np.load(shard_path) as shard:
            row_count = int(shard["input_ids"].shape[0])
            for row in range(row_count):
                example_id = example_ids[offset + row]
                rows[example_id] = {
                    "position": int(shard["score_selected_position"][row]),
                    "entropy": float(shard["score_selected_position_entropy"][row]),
                    "top_token_id": int(shard["score_top_token_id"][row]),
                }
        offset += row_count
    return rows


def _assert_selected_source_coordinate_invariants(output_dir: Path) -> None:
    selected = _json(output_dir / "leaderboards" / "selected_exemplars.json")[
        "selected_exemplars"
    ]
    payloads = _json(
        output_dir / "selected_exemplars" / "selected-exemplars-00000.json"
    )["selected_exemplars"]
    for record, payload in zip(selected, payloads, strict=True):
        assert isinstance(record["payload_ref"], dict)
        assert record["payload_ref"]
        assert payload["payload_ref"] == record["payload_ref"]
        source_shard_id = int(record["source_shard_id"])
        source_row = int(record["source_row"])
        source_position = int(record["source_position"])
        with np.load(
            output_dir / "shards" / f"shard-{source_shard_id:05d}.npz"
        ) as shard:
            source_score = float(record["source_score"])
            source_top_token_id = int(record["source_top_token_id"])
            assert record["selected_position"] == source_position
            assert payload["selected_position"] == source_position
            assert np.isclose(record["selected_score"], source_score)
            assert np.isclose(payload["selected_score"], source_score)
            assert np.isclose(payload["source_score"], source_score)
            assert np.isclose(payload["teacher_entropy"], source_score)
            assert payload["top_token_ids"][0] == source_top_token_id
            assert np.isclose(
                float(shard["corridor_entropy"][source_row, source_position]),
                source_score,
            )
            if record["source_delivery_path"] == "one_pass_pruned_candidate":
                if "exemplar_source_top_token_ids" in shard:
                    payload_position = (
                        exemplar_delivery._one_pass_payload_position_index(
                            dict(shard),
                            row=source_row,
                            record=record,
                        )
                    )
                    assert (
                        int(
                            shard["exemplar_source_top_token_ids"][
                                source_row,
                                payload_position,
                                0,
                            ]
                        )
                        == source_top_token_id
                    )
            elif record["payload_ref"]["kind"] == "corridor_exemplar_score_pass_v1":
                assert (
                    int(shard["corridor_top_token_ids"][source_row, source_position])
                    == source_top_token_id
                )
                score_position = int(shard["score_selected_position"][source_row])
                score_entropy = float(
                    shard["score_selected_position_entropy"][source_row]
                )
                score_top_token_id = int(shard["score_top_token_id"][source_row])
                assert source_position == score_position
                assert np.isclose(source_score, score_entropy)
                assert source_top_token_id == score_top_token_id
        for field in (
            "selected_example_id",
            "selected_position",
            "selected_score",
            "source_shard_id",
            "source_row",
            "source_position",
            "source_score",
            "source_top_token_id",
            "source_score_policy",
            "payload_ref",
            "corridor_mode_id",
            "corridor_assignment_status",
        ):
            assert record[field] == payload[field]


def _marker_token(row: int, position: int, index: int) -> int:
    return 1000 + row * 100 + position * 10 + index


def _marker_prob(row: int, position: int, index: int) -> float:
    return np.float32(0.01 * (row + 1) + 0.001 * position + 0.0001 * index).item()


def _marker_top_mass(row: int, position: int) -> float:
    return np.float32(0.4 + 0.01 * row + 0.001 * position).item()


def _marker_bucket(row: int, position: int, index: int) -> float:
    return np.float32(0.03 + 0.01 * index + 0.001 * row + 0.0001 * position).item()
