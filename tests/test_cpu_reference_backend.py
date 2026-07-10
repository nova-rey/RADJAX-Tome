from __future__ import annotations

import numpy as np
import pytest

from radjax_tome.backends import (
    CPUReferenceTeacherEmissionBackend,
    TeacherBackendConfig,
    TeacherBatchInput,
    create_backend,
    list_backend_capabilities,
)
from radjax_tome.backends.cpu import _corridor_stat_arrays


def _batch() -> TeacherBatchInput:
    return TeacherBatchInput(
        example_ids=("ex-1", "ex-2"),
        texts=("alpha", "beta"),
    )


def _config(**overrides: object) -> TeacherBackendConfig:
    payload = {
        "backend_id": "cpu_reference",
        "runtime_mode": "cpu",
        "cpu_orchestration_mode": "auto",
        "target_policy": "dense_logits",
        "sequence_length": 5,
        "batch_size": 99,
        "vocab_size": 64,
        "top_k": 4,
        "num_buckets": 3,
        "exemplar_top_n": 2,
        "dynamic_top_k_min": 2,
        "dynamic_top_k_max": 5,
        "dynamic_mass_threshold": 0.75,
        "dynamic_top_k_policy": "mass_threshold_v1",
    }
    payload.update(overrides)
    return TeacherBackendConfig(**payload)


def test_cpu_reference_backend_created_through_registry() -> None:
    backend = create_backend(_config())

    assert backend.backend_id == "cpu_reference"
    assert backend.backend_family == "cpu_reference"
    assert backend.runtime_mode == "cpu"
    assert backend.config.exemplar_capture_mode == "one_pass_candidate"


def test_registry_lists_fake_and_cpu_reference_backends() -> None:
    backend_ids = [capability.backend_id for capability in list_backend_capabilities()]

    assert "fake_numpy" in backend_ids
    assert "cpu_reference" in backend_ids
    assert backend_ids == sorted(backend_ids)


def test_dense_logits_payload_shape_and_determinism() -> None:
    config = _config(target_policy="dense_logits")
    backend = create_backend(config)

    first = backend.emit_batch(_batch())
    second = backend.emit_batch(_batch())

    assert first.payload["logits"].shape == (2, 5, config.vocab_size)
    np.testing.assert_array_equal(first.input_ids, second.input_ids)
    np.testing.assert_array_equal(first.attention_mask, second.attention_mask)
    np.testing.assert_array_equal(first.payload["logits"], second.payload["logits"])
    assert first.metadata["capability_status"] == "supported_debug"
    assert first.metadata["configured_batch_size"] == 99
    assert first.metadata["actual_batch_size"] == 2


def test_topk_payload_fields_shapes_and_mass_accounting() -> None:
    backend = create_backend(_config(target_policy="topk_with_tail_v0", top_k=4))

    result = backend.emit_batch(_batch())
    payload = result.payload

    assert {
        "top_token_ids",
        "top_log_probs",
        "top_probs",
        "top_mass",
        "tail_mass",
        "teacher_entropy",
    } <= set(payload)
    assert payload["top_token_ids"].shape == (2, 5, 4)
    assert payload["top_log_probs"].shape == (2, 5, 4)
    assert payload["top_probs"].shape == (2, 5, 4)
    assert payload["top_mass"].shape == (2, 5)
    assert payload["tail_mass"].shape == (2, 5)
    assert payload["teacher_entropy"].shape == (2, 5)
    assert np.isfinite(payload["top_log_probs"]).all()
    assert np.isfinite(payload["top_probs"]).all()
    assert np.isfinite(payload["tail_mass"]).all()
    assert np.isfinite(payload["teacher_entropy"]).all()
    np.testing.assert_allclose(
        payload["tail_mass"],
        1.0 - payload["top_mass"],
        rtol=1e-6,
        atol=1e-6,
    )
    assert result.metadata["capability_status"] == "supported"
    assert result.metadata["requested_top_k"] == 4
    assert result.metadata["effective_top_k"] == 4


def test_topk_effective_top_k_clamps_to_vocab_size() -> None:
    backend = create_backend(
        _config(target_policy="topk_with_tail_v0", top_k=99, vocab_size=7)
    )

    result = backend.emit_batch(_batch())

    assert result.payload["top_token_ids"].shape == (2, 5, 7)
    assert result.metadata["requested_top_k"] == 99
    assert result.metadata["effective_top_k"] == 7


def test_cascaded_payload_shapes_and_bucket_mass_accounting() -> None:
    backend = create_backend(
        _config(target_policy="cascaded_soft_labels_v1", top_k=4, num_buckets=3)
    )

    result = backend.emit_batch(_batch())
    payload = result.payload

    assert {
        "top_token_ids",
        "top_log_probs",
        "top_probs",
        "top_mass",
        "tail_mass",
        "bucket_masses",
        "teacher_entropy",
    } <= set(payload)
    assert payload["bucket_masses"].shape == (2, 5, 3)
    assert np.isfinite(payload["bucket_masses"]).all()
    np.testing.assert_allclose(
        np.sum(payload["bucket_masses"], axis=-1),
        payload["tail_mass"],
        rtol=1e-6,
        atol=1e-6,
    )
    assert result.metadata["capability_status"] == "supported"
    assert result.metadata["num_buckets"] == 3
    assert (
        result.metadata["bucket_policy"]
        == "contiguous_descending_tail_probability_mass"
    )


def test_dynamic_cascaded_payload_shapes_mask_and_mass_accounting() -> None:
    backend = create_backend(
        _config(
            target_policy="dynamic_cascaded_soft_labels_v1",
            dynamic_top_k_min=2,
            dynamic_top_k_max=5,
            dynamic_mass_threshold=0.75,
            num_buckets=3,
        )
    )

    result = backend.emit_batch(_batch())
    payload = result.payload

    assert {
        "top_token_ids",
        "top_log_probs",
        "top_probs",
        "top_selection_mask",
        "effective_top_k",
        "top_mass",
        "tail_mass",
        "bucket_masses",
        "teacher_entropy",
    } <= set(payload)
    assert payload["top_token_ids"].shape == (2, 5, 5)
    assert payload["top_log_probs"].shape == (2, 5, 5)
    assert payload["top_probs"].shape == (2, 5, 5)
    assert payload["top_selection_mask"].shape == (2, 5, 5)
    assert payload["top_selection_mask"].dtype == np.bool_
    assert payload["effective_top_k"].shape == (2, 5)
    assert np.issubdtype(payload["effective_top_k"].dtype, np.integer)
    assert payload["top_mass"].shape == (2, 5)
    assert payload["tail_mass"].shape == (2, 5)
    assert payload["bucket_masses"].shape == (2, 5, 3)
    assert payload["teacher_entropy"].shape == (2, 5)
    assert np.isfinite(payload["teacher_entropy"]).all()

    mask = payload["top_selection_mask"]
    np.testing.assert_array_equal(
        payload["effective_top_k"],
        np.sum(mask, axis=-1).astype(np.int32),
    )
    assert int(np.min(payload["effective_top_k"])) >= 2
    assert int(np.max(payload["effective_top_k"])) <= 5
    assert (payload["top_token_ids"][~mask] == 0).all()
    assert (payload["top_probs"][~mask] == 0.0).all()
    assert (payload["top_log_probs"][~mask] == 0.0).all()
    np.testing.assert_allclose(
        np.sum(payload["bucket_masses"], axis=-1),
        payload["tail_mass"],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        payload["top_mass"] + payload["tail_mass"],
        1.0,
        rtol=1e-6,
        atol=1e-6,
    )

    not_max_clamped = payload["effective_top_k"] < 5
    assert (payload["top_mass"][not_max_clamped] >= 0.75).all()
    below_threshold = payload["top_mass"] < 0.75
    assert (payload["effective_top_k"][below_threshold] == 5).all()

    metadata = result.metadata
    assert metadata["capability_status"] == "supported"
    assert metadata["dynamic_top_k_policy"] == "mass_threshold_v1"
    assert metadata["dynamic_top_k_min_configured"] == 2
    assert metadata["dynamic_top_k_max_configured"] == 5
    assert metadata["dynamic_top_k_min_effective"] == 2
    assert metadata["dynamic_top_k_max_effective"] == 5
    assert metadata["dynamic_mass_threshold"] == 0.75
    assert metadata["num_buckets"] == 3
    assert metadata["bucket_policy"] == "contiguous_descending_tail_probability_mass"
    assert metadata["padding_policy"] == "pad_to_dynamic_top_k_max_effective"
    assert metadata["selection_mask_field"] == "top_selection_mask"
    assert "must be ignored" in metadata["masked_slot_policy"]
    assert metadata["top_selection_mask_semantics"] == (
        "true means explicit selected token"
    )
    assert metadata["effective_top_k_min_observed"] == int(
        np.min(payload["effective_top_k"])
    )
    assert metadata["effective_top_k_max_observed"] == int(
        np.max(payload["effective_top_k"])
    )
    assert metadata["effective_top_k_mean_observed"] == pytest.approx(
        float(np.mean(payload["effective_top_k"], dtype=np.float32))
    )


def test_dynamic_cascaded_min_k_is_honored_when_threshold_is_low() -> None:
    backend = create_backend(
        _config(
            target_policy="dynamic_cascaded_soft_labels_v1",
            dynamic_top_k_min=3,
            dynamic_top_k_max=5,
            dynamic_mass_threshold=0.01,
        )
    )

    payload = backend.emit_batch(_batch()).payload

    assert (payload["effective_top_k"] >= 3).all()
    assert (payload["effective_top_k"] == 3).any()


def test_dynamic_cascaded_max_k_clamps_to_vocab_size() -> None:
    backend = create_backend(
        _config(
            target_policy="dynamic_cascaded_soft_labels_v1",
            vocab_size=4,
            dynamic_top_k_min=2,
            dynamic_top_k_max=99,
            dynamic_mass_threshold=1.0,
        )
    )

    result = backend.emit_batch(_batch())

    assert result.payload["top_token_ids"].shape == (2, 5, 4)
    assert (result.payload["effective_top_k"] == 4).all()
    assert result.metadata["dynamic_top_k_max_configured"] == 99
    assert result.metadata["dynamic_top_k_max_effective"] == 4
    assert result.metadata["dynamic_top_k_min_effective"] == 2


def test_corridor_stat_arrays_require_real_top32_support() -> None:
    source = {
        "top_probs": np.full((1, 2, 8), 0.1, dtype=np.float32),
        "teacher_entropy": np.ones((1, 2), dtype=np.float32),
    }

    with pytest.raises(
        ValueError,
        match=(
            "corridor stat export requires top_probs depth >= 32 to compute "
            "top32_mass and tail_mass; got K=8"
        ),
    ):
        _corridor_stat_arrays(source)


def test_corridor_stat_arrays_compute_top32_stats() -> None:
    probs = np.zeros((1, 1, 32), dtype=np.float32)
    probs[0, 0] = np.linspace(0.05, 0.001, 32, dtype=np.float32)
    source = {
        "top_probs": probs,
        "teacher_entropy": np.asarray([[1.5]], dtype=np.float32),
    }

    stats = _corridor_stat_arrays(source)

    assert stats["top8_mass"][0, 0] <= stats["top32_mass"][0, 0]
    assert stats["tail_mass"][0, 0] == pytest.approx(
        1.0 - stats["top32_mass"][0, 0],
    )
    assert stats["top1_margin"][0, 0] == pytest.approx(probs[0, 0, 0] - probs[0, 0, 1])
    for value in stats.values():
        assert np.isfinite(value).all()


def test_corridor_exemplar_payload_and_metadata_are_deterministic() -> None:
    backend = create_backend(
        _config(target_policy="corridor_exemplar_v1", exemplar_top_n=2)
    )

    first = backend.emit_batch(_batch())
    second = backend.emit_batch(_batch())
    payload = first.payload

    assert first.target_policy == "corridor_exemplar_v1"
    assert {
        "corridor_records",
        "corridor_summary",
        "exemplar_records",
        "exemplar_summary",
        "mode_records",
        "source_policy_summary",
        "schema_metadata",
        "corridor_top_token_ids",
        "corridor_top_probs",
        "corridor_teacher_entropy",
        "corridor_confidence",
        "corridor_lengths",
        "exemplar_positions",
        "exemplar_scores",
        "exemplar_selection_mask",
        "exemplar_source_policy_ids",
        "exemplar_source_effective_top_k",
        "exemplar_source_top_mass",
        "exemplar_source_tail_mass",
    } <= set(payload)
    assert payload["corridor_records"]["record_count"] == 2
    assert payload["corridor_summary"]["record_count"] == 2
    assert payload["exemplar_records"]["record_count"] == 4
    assert payload["exemplar_summary"]["positions_per_example"] == 2
    assert payload["mode_records"]["mode_record_policy"] == "top_mode_summary_v1"
    assert (
        payload["source_policy_summary"]["exemplar_source_policy"]
        == "dynamic_cascaded_soft_labels_v1"
    )
    assert payload["source_policy_summary"]["source_policy_kind"] == (
        "dynamic_cascaded"
    )
    assert payload["source_policy_summary"]["source_policy_uses_bucketed_tail"] is True
    assert payload["source_policy_summary"]["source_policy_dynamic_top_k"] is True
    assert payload["schema_metadata"]["schema_version"] == "corridor_exemplar_v1"
    assert payload["schema_metadata"]["corridor_payload_flavor"] == "production_v1"
    assert payload["schema_metadata"]["production_corridor_schema"] is True
    assert payload["schema_metadata"]["historical_parity_claimed"] is False
    assert (
        payload["schema_metadata"]["historical_reference_source"]
        == "cpu_reference_proxy"
    )
    assert payload["schema_metadata"]["exemplar_capture_mode_requested"] == (
        "one_pass_candidate"
    )
    assert payload["schema_metadata"]["exemplar_capture_mode_effective"] == (
        "one_pass_candidate"
    )
    assert payload["schema_metadata"]["exemplar_capture_mode_policy"] == (
        "explicit_one_pass_candidate_v1"
    )
    assert payload["schema_metadata"]["exemplar_candidate_scope"] == (
        "batch_all_examples"
    )
    assert payload["schema_metadata"]["corpus_level_exemplar_finalization"] is False
    assert (
        payload["schema_metadata"]["requires_second_pass_for_final_exemplars"] is False
    )
    assert payload["corridor_top_token_ids"].shape == (2, 5)
    assert payload["corridor_top_probs"].shape == (2, 5)
    assert payload["corridor_teacher_entropy"].shape == (2, 5)
    assert payload["corridor_confidence"].shape == (2, 5)
    assert payload["corridor_lengths"].shape == (2,)
    assert payload["exemplar_positions"].shape == (2, 2)
    assert payload["exemplar_scores"].shape == (2, 2)
    assert payload["exemplar_selection_mask"].shape == (2, 5)
    assert payload["exemplar_source_policy_ids"].shape == (2, 5)
    assert payload["exemplar_source_effective_top_k"].shape == (2, 5)
    assert payload["exemplar_source_top_mass"].shape == (2, 5)
    assert payload["exemplar_source_tail_mass"].shape == (2, 5)
    assert (payload["exemplar_source_policy_ids"] == 3).all()
    assert np.count_nonzero(payload["exemplar_selection_mask"]) == 4
    assert np.isfinite(payload["corridor_top_probs"]).all()
    assert np.isfinite(payload["corridor_teacher_entropy"]).all()
    assert np.isfinite(payload["exemplar_scores"]).all()
    assert np.isfinite(payload["exemplar_source_top_mass"]).all()
    assert np.isfinite(payload["exemplar_source_tail_mass"]).all()

    for key in (
        "mode_records",
        "corridor_top_token_ids",
        "corridor_top_probs",
        "corridor_teacher_entropy",
        "corridor_confidence",
        "corridor_lengths",
        "exemplar_positions",
        "exemplar_scores",
        "exemplar_selection_mask",
        "exemplar_source_policy_ids",
        "exemplar_source_effective_top_k",
        "exemplar_source_top_mass",
        "exemplar_source_tail_mass",
    ):
        np.testing.assert_array_equal(first.payload[key], second.payload[key])
    assert first.payload["corridor_records"] == second.payload["corridor_records"]
    assert first.payload["corridor_summary"] == second.payload["corridor_summary"]
    assert first.payload["exemplar_records"] == second.payload["exemplar_records"]
    assert first.payload["exemplar_summary"] == second.payload["exemplar_summary"]
    assert first.payload["mode_records"] == second.payload["mode_records"]
    assert (
        first.payload["source_policy_summary"]
        == second.payload["source_policy_summary"]
    )
    assert first.payload["schema_metadata"] == second.payload["schema_metadata"]

    assert first.metadata["effective_runtime_mode"] == "cpu"
    assert first.metadata["effective_cpu_orchestration_mode"] == "serial"
    assert first.metadata["optimized_path_used"] is False
    assert first.metadata["fallback_used"] is False
    assert first.metadata["capability_status"] == "supported"
    assert first.metadata["schema_version"] == "corridor_exemplar_v1"
    assert first.metadata["corridor_payload_flavor"] == "production_v1"
    assert first.metadata["production_corridor_schema"] is True
    assert first.metadata["historical_parity_claimed"] is False
    assert first.metadata["historical_reference_source"] == "cpu_reference_proxy"
    assert first.metadata["exemplar_source_policy"] == (
        "dynamic_cascaded_soft_labels_v1"
    )
    assert first.metadata["exemplar_selection_policy"] == "entropy_top_n_v1"
    assert first.metadata["corridor_policy"] == "production_corridor_records_v1"
    assert first.metadata["mode_record_policy"] == "top_mode_summary_v1"
    assert first.metadata["fingerprint_topology_policy"] == "sequence_position_v1"
    assert first.metadata["corridor_confidence_policy"] == "top_probability_v1"
    assert first.metadata["exemplar_capture_mode_requested"] == "one_pass_candidate"
    assert first.metadata["exemplar_capture_mode_effective"] == "one_pass_candidate"
    assert first.metadata["exemplar_capture_mode_policy"] == (
        "explicit_one_pass_candidate_v1"
    )
    assert first.metadata["exemplar_candidate_scope"] == "batch_all_examples"
    assert first.metadata["corpus_level_exemplar_finalization"] is False
    assert first.metadata["requires_second_pass_for_final_exemplars"] is False
    assert first.metadata["source_policy_kind"] == "dynamic_cascaded"
    assert first.metadata["source_policy_uses_bucketed_tail"] is True
    assert first.metadata["source_policy_dynamic_top_k"] is True
    assert first.metadata["exemplar_records"] == 2


def test_corridor_two_pass_score_payload_is_batch_scale() -> None:
    backend = create_backend(
        _config(
            target_policy="corridor_exemplar_v1",
            exemplar_capture_mode="two_pass_sparse_exemplar",
            exemplar_sparse_selection_top_n=1,
        )
    )

    result = backend.emit_batch(_batch())
    payload = result.payload

    assert {
        "score_records",
        "score_summary",
        "score_metadata",
        "corridor_top_token_ids",
        "corridor_teacher_entropy",
        "corridor_confidence",
        "corridor_lengths",
        "score_example_ids",
        "score_max_entropy",
        "score_mean_entropy",
        "score_selected_position",
        "score_top_token_id",
        "score_selected_position_entropy",
        "score_confidence_at_selected_position",
        "score_source_policy_ids",
        "score_lengths",
    } <= set(payload)
    assert "corridor_teacher_entropy" in payload
    assert "exemplar_positions" not in payload
    for field in (
        "corridor_lengths",
        "score_example_ids",
        "score_max_entropy",
        "score_mean_entropy",
        "score_selected_position",
        "score_top_token_id",
        "score_selected_position_entropy",
        "score_confidence_at_selected_position",
        "score_source_policy_ids",
        "score_lengths",
    ):
        assert payload[field].shape == (2,)
    for field in (
        "corridor_top_token_ids",
        "corridor_teacher_entropy",
        "corridor_confidence",
    ):
        assert payload[field].shape == (2, 5)

    rerun = create_backend(
        _config(target_policy="dynamic_cascaded_soft_labels_v1")
    ).emit_batch(_batch())
    for row, position in enumerate(payload["score_selected_position"]):
        assert (
            payload["corridor_top_token_ids"][row, position]
            == payload["score_top_token_id"][row]
        )
        assert (
            rerun.payload["top_token_ids"][row, position, 0]
            == payload["score_top_token_id"][row]
        )

    one_pass = create_backend(_config(target_policy="corridor_exemplar_v1")).emit_batch(
        _batch()
    )
    score_bytes = sum(
        value.nbytes for value in payload.values() if hasattr(value, "nbytes")
    )
    one_pass_bytes = sum(
        value.nbytes for value in one_pass.payload.values() if hasattr(value, "nbytes")
    )
    assert score_bytes < one_pass_bytes

    assert result.metadata["schema_version"] == "corridor_exemplar_score_pass_v1"
    assert result.metadata["production_corridor_schema"] is False
    assert result.metadata["exemplar_capture_mode_requested"] == (
        "two_pass_sparse_exemplar"
    )
    assert result.metadata["exemplar_capture_mode_effective"] == (
        "two_pass_sparse_exemplar"
    )
    assert result.metadata["exemplar_capture_stage"] == "score_pass"
    assert result.metadata["exemplar_candidate_scope"] == (
        "batch_score_and_corridor_evidence"
    )
    assert result.metadata["requires_second_pass_for_final_exemplars"] is True
    assert result.metadata["rerun_teacher_for_selected_examples"] is True


def test_corridor_two_pass_selected_helper_emits_production_schema() -> None:
    backend = CPUReferenceTeacherEmissionBackend(
        _config(
            target_policy="corridor_exemplar_v1",
            exemplar_capture_mode="two_pass_sparse_exemplar",
            exemplar_second_pass_source_policy="cascaded_soft_labels_v1",
        )
    )

    result = backend.emit_corridor_exemplar_selected_batch(
        _batch(),
        corpus_level_finalized=True,
    )
    payload = result.payload

    assert payload["schema_metadata"]["schema_version"] == "corridor_exemplar_v1"
    assert payload["schema_metadata"]["production_corridor_schema"] is True
    assert payload["schema_metadata"]["exemplar_capture_stage"] == (
        "selected_exemplar_pass"
    )
    assert payload["schema_metadata"]["exemplar_candidate_scope"] == (
        "selected_examples_only"
    )
    assert payload["schema_metadata"]["corpus_level_exemplar_finalization"] is True
    assert (
        payload["schema_metadata"]["requires_second_pass_for_final_exemplars"] is False
    )
    assert payload["schema_metadata"]["rerun_teacher_for_selected_examples"] is True
    assert payload["corridor_teacher_entropy"].shape == (2, 5)
    assert payload["exemplar_positions"].shape == (2, 2)
    assert payload["source_policy_summary"]["exemplar_source_policy"] == (
        "cascaded_soft_labels_v1"
    )


def test_corridor_auto_policy_chooses_one_pass_for_small_estimate() -> None:
    backend = create_backend(
        _config(
            target_policy="corridor_exemplar_v1",
            exemplar_capture_mode="auto",
            exemplar_auto_num_examples=4,
            exemplar_auto_expected_selected_fraction=0.25,
            exemplar_auto_teacher_inference_cost_hint="cheap",
        )
    )

    result = backend.emit_batch(_batch())

    assert "corridor_teacher_entropy" in result.payload
    assert result.payload["score_max_entropy"].shape == (2,)
    assert result.payload["score_selected_position"].shape == (2,)
    assert result.metadata["exemplar_capture_mode_requested"] == "auto"
    assert result.metadata["exemplar_capture_mode_effective"] == "one_pass_candidate"
    assert (
        result.metadata["exemplar_capture_policy"] == "auto_exemplar_capture_policy_v1"
    )
    assert result.metadata["manual_override_used"] is False
    assert "small enough" in result.metadata["auto_policy_reason"]
    assert result.metadata["estimated_one_pass_candidate_bytes"] > 0
    assert result.metadata["estimated_two_pass_score_bytes"] > 0
    assert result.metadata["estimated_two_pass_selected_bytes"] > 0
    assert result.metadata["estimated_two_pass_total_bytes"] > 0
    assert result.metadata["expected_selected_exemplar_fraction"] == 0.25
    assert result.metadata["available_disk_budget_bytes"] is None
    assert (
        "available_disk_budget_bytes" in result.metadata["auto_policy_inputs_missing"]
    )


def test_corridor_auto_policy_chooses_two_pass_for_huge_estimate() -> None:
    backend = create_backend(
        _config(
            target_policy="corridor_exemplar_v1",
            exemplar_capture_mode="auto",
            sequence_length=64,
            exemplar_auto_num_examples=1_000_000,
            exemplar_auto_expected_selected_fraction=0.01,
            exemplar_auto_available_disk_budget_bytes=1_000_000,
        )
    )

    result = backend.emit_batch(_batch())

    assert "score_max_entropy" in result.payload
    assert "corridor_teacher_entropy" in result.payload
    assert "exemplar_positions" not in result.payload
    assert result.metadata["exemplar_capture_mode_requested"] == "auto"
    assert result.metadata["exemplar_capture_mode_effective"] == (
        "two_pass_sparse_exemplar"
    )
    assert (
        result.metadata["exemplar_capture_policy"] == "auto_exemplar_capture_policy_v1"
    )
    assert result.metadata["manual_override_used"] is False
    assert "budget" in result.metadata["auto_policy_reason"]
    assert (
        result.metadata["estimated_one_pass_candidate_bytes"]
        > (result.metadata["available_disk_budget_bytes"])
    )
    assert result.metadata["exemplar_capture_stage"] == "score_pass"
    assert result.metadata["requires_second_pass_for_final_exemplars"] is True


def test_corridor_manual_one_pass_overrides_auto_estimates() -> None:
    backend = create_backend(
        _config(
            target_policy="corridor_exemplar_v1",
            exemplar_capture_mode="one_pass_candidate",
            sequence_length=64,
            exemplar_auto_num_examples=1_000_000,
            exemplar_auto_available_disk_budget_bytes=1_000_000,
        )
    )

    result = backend.emit_batch(_batch())

    assert "corridor_teacher_entropy" in result.payload
    assert result.metadata["exemplar_capture_mode_requested"] == "one_pass_candidate"
    assert result.metadata["exemplar_capture_mode_effective"] == "one_pass_candidate"
    assert (
        result.metadata["exemplar_capture_policy"]
        == "manual_exemplar_capture_policy_v1"
    )
    assert result.metadata["manual_override_used"] is True
    assert "manual override" in result.metadata["auto_policy_reason"]


@pytest.mark.parametrize(
    ("source_policy", "kind", "bucketed", "dynamic", "policy_id"),
    (
        ("dense_logits", "full_resolution", False, False, 1),
        ("cascaded_soft_labels_v1", "fixed_cascaded", True, False, 2),
        ("dynamic_cascaded_soft_labels_v1", "dynamic_cascaded", True, True, 3),
    ),
)
def test_corridor_exemplar_source_policy_summary_is_truthful(
    source_policy: str,
    kind: str,
    bucketed: bool,
    dynamic: bool,
    policy_id: int,
) -> None:
    backend = create_backend(
        _config(
            target_policy="corridor_exemplar_v1",
            exemplar_source_policy=source_policy,
        )
    )

    result = backend.emit_batch(_batch())
    payload = result.payload
    summary = payload["source_policy_summary"]

    assert summary["exemplar_source_policy"] == source_policy
    assert summary["source_policy_kind"] == kind
    assert summary["source_policy_uses_bucketed_tail"] is bucketed
    assert summary["source_policy_dynamic_top_k"] is dynamic
    assert result.metadata["exemplar_source_policy"] == source_policy
    assert result.metadata["source_policy_kind"] == kind
    assert result.metadata["source_policy_uses_bucketed_tail"] is bucketed
    assert result.metadata["source_policy_dynamic_top_k"] is dynamic
    assert (payload["exemplar_source_policy_ids"] == policy_id).all()
    if source_policy == "dense_logits":
        assert summary["source_effective_top_k_semantics"] == "top_mode_only"
        assert (payload["exemplar_source_effective_top_k"] == 1).all()
    if source_policy == "cascaded_soft_labels_v1":
        assert summary["requested_top_k"] == 4
        assert summary["effective_top_k"] == 4
        assert summary["num_buckets"] == 3
    if source_policy == "dynamic_cascaded_soft_labels_v1":
        assert summary["dynamic_top_k_policy"] == "mass_threshold_v1"
        assert summary["dynamic_top_k_min_effective"] == 2
        assert summary["dynamic_top_k_max_effective"] == 5
        assert "effective_top_k_mean_observed" in summary


def test_cpu_reference_metadata_records_requested_and_effective_values() -> None:
    backend = create_backend(
        _config(
            cpu_orchestration_mode="staged",
            target_policy="topk_with_tail_v0",
        )
    )

    metadata = backend.emit_batch(_batch()).metadata

    assert metadata["requested_runtime_mode"] == "cpu"
    assert metadata["effective_runtime_mode"] == "cpu"
    assert metadata["requested_cpu_orchestration_mode"] == "staged"
    assert metadata["effective_cpu_orchestration_mode"] == "serial"
    assert metadata["requested_target_policy"] == "topk_with_tail_v0"
    assert metadata["effective_target_policy"] == "topk_with_tail_v0"
    assert metadata["backend_id"] == "cpu_reference"
    assert metadata["backend_family"] == "cpu_reference"
    assert metadata["runtime_kind"] == "cpu_reference"
    assert metadata["device_kind"] == "cpu"
    assert metadata["optimized_path_used"] is False
    assert metadata["fallback_used"] is False
    assert metadata["sequence_length"] == 5
    assert metadata["batch_size"] == 99
    assert metadata["vocab_size"] == 64


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("backend_id", "fake_numpy", "backend_id"),
        ("runtime_mode", "cpu_gpu", "runtime_mode"),
        ("sequence_length", 0, "sequence_length"),
        ("batch_size", 0, "batch_size"),
        ("vocab_size", 0, "vocab_size"),
        ("top_k", 0, "top_k"),
        ("num_buckets", 0, "num_buckets"),
        ("exemplar_top_n", 0, "exemplar_top_n"),
        ("dynamic_top_k_min", 0, "dynamic_top_k_min"),
        ("dynamic_top_k_max", 1, "dynamic_top_k_max"),
        ("dynamic_mass_threshold", 0.0, "dynamic_mass_threshold"),
        ("dynamic_mass_threshold", 1.1, "dynamic_mass_threshold"),
        ("dynamic_top_k_policy", "unknown", "dynamic_top_k_policy"),
        ("exemplar_source_policy", "unknown", "exemplar_source_policy"),
        ("exemplar_selection_policy", "unknown", "exemplar_selection_policy"),
        ("exemplar_capture_mode", "unknown", "exemplar_capture_mode"),
        (
            "exemplar_first_pass_score_policy",
            "unknown",
            "exemplar_first_pass_score_policy",
        ),
        (
            "exemplar_second_pass_source_policy",
            "unknown",
            "exemplar_second_pass_source_policy",
        ),
        (
            "exemplar_sparse_selection_top_n",
            0,
            "exemplar_sparse_selection_top_n",
        ),
        (
            "exemplar_sparse_selection_fraction",
            0.0,
            "exemplar_sparse_selection_fraction",
        ),
        (
            "exemplar_sparse_selection_fraction",
            1.1,
            "exemplar_sparse_selection_fraction",
        ),
        ("exemplar_auto_num_examples", 0, "exemplar_auto_num_examples"),
        (
            "exemplar_auto_expected_selected_fraction",
            0.0,
            "exemplar_auto_expected_selected_fraction",
        ),
        (
            "exemplar_auto_expected_selected_fraction",
            1.1,
            "exemplar_auto_expected_selected_fraction",
        ),
        (
            "exemplar_auto_available_disk_budget_bytes",
            0,
            "exemplar_auto_available_disk_budget_bytes",
        ),
        ("corridor_payload_flavor", "proxy_v0", "corridor_payload_flavor"),
    ),
)
def test_invalid_cpu_reference_config_values_fail_clearly(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        config = _config(**{field: value})
        if field == "backend_id":
            CPUReferenceTeacherEmissionBackend(config)
        else:
            create_backend(config)


def test_cpu_reference_capabilities_mark_corridor_exemplar_supported() -> None:
    capabilities = {
        capability.target_policy: capability
        for capability in create_backend(_config()).capabilities()
    }

    corridor = capabilities["corridor_exemplar_v1"]
    assert corridor.status == "supported"
    assert corridor.implemented_now
    assert not corridor.optimized
    assert "Spec 3.3F8" in corridor.notes
    assert "locked production schema" in corridor.notes


def test_cpu_reference_capabilities_mark_dynamic_cascaded_supported() -> None:
    capabilities = {
        capability.target_policy: capability
        for capability in create_backend(_config()).capabilities()
    }

    dynamic = capabilities["dynamic_cascaded_soft_labels_v1"]
    assert dynamic.status == "supported"
    assert dynamic.implemented_now
    assert not dynamic.optimized
    assert "Spec 3.3F6" in dynamic.notes
    assert "dynamic top-k explicit head plus bucketed-tail" in dynamic.notes
