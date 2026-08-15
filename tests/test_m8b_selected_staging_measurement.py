"""Frozen M8B.1 comparison rules; these are benchmark-only, never authority."""

from __future__ import annotations

import pytest

from radjax_tome.builder.delivery.measurement import (
    M8BStagingStatistics,
    SelectedPassExecutionDiagnostics,
    SelectedPassMeasurementControl,
    validate_m8b_statistics_receipt,
)
from radjax_tome.builder.delivery.staging import _write_native_payload_shard


def test_frozen_statistics_follow_the_approved_formulas() -> None:
    statistics = M8BStagingStatistics()

    assert statistics.summarize([5.0, 1.0, 3.0]) == {
        "median": 3.0,
        "spread": 4.0,
        "normalized_spread": 4.0 / 3.0,
    }
    assert statistics.summarize([0.0, 0.0, 0.0])["normalized_spread"] == 0.0
    assert statistics.combined_spread([10.0, 12.0, 11.0], [7.0, 8.0, 9.0]) == (8.0**0.5)
    assert statistics.improvement_beyond_noise(
        [100.0, 100.0, 100.0], [90.0, 90.0, 90.0]
    )
    assert statistics.materially_regresses([100.0, 100.0, 100.0], [106.0, 106.0, 106.0])
    assert statistics.memory_limit([10, 12, 11]) == 14


def test_statistics_require_three_measurements_and_reject_changed_receipt() -> None:
    statistics = M8BStagingStatistics()
    with pytest.raises(ValueError, match="exactly three"):
        statistics.summarize([1.0, 2.0])
    with pytest.raises(ValueError, match="exactly three"):
        statistics.memory_limit([1, 2])

    receipt: dict[str, object] = {"statistics": statistics.receipt_projection()}
    validate_m8b_statistics_receipt(receipt)
    receipt["statistics"] = {**statistics.receipt_projection(), "noise_multiplier": 1.0}
    with pytest.raises(ValueError, match="frozen definitions"):
        validate_m8b_statistics_receipt(receipt)


def test_zero_median_with_nonzero_spread_is_explicitly_undefined() -> None:
    assert M8BStagingStatistics().summarize([-1.0, 0.0, 1.0]) == {
        "median": 0.0,
        "spread": 2.0,
        "normalized_spread": None,
    }


def test_private_staging_diagnostics_split_initial_write_without_relabeling_m8a(
    tmp_path,
) -> None:
    control = SelectedPassMeasurementControl(
        benchmark_only=True,
        effective_execution_cap=8,
        immutable_checkpoint_digest="sha256:" + "a" * 64,
        checkpoint_root=tmp_path / "checkpoint",
        temporary_output_root=tmp_path / "output",
    )
    diagnostics = SelectedPassExecutionDiagnostics(
        control=control,
        requested_batch_size=8,
    )
    _write_native_payload_shard(
        tmp_path / "stage",
        record_index=0,
        payload={"delivery_authority_hash": "sha256:authority"},
        delivery_path="two_pass_rerun_selected",
        _measurement_diagnostics=diagnostics,
    )
    report = diagnostics.finish()
    phases = report["phases"]
    assert phases["hashing_json_atomic_write_fsync"]["seconds"] == 0.0
    for phase in (
        "canonical_body_encoding_hash",
        "staging_json_encoding",
        "temporary_file_write",
        "temporary_file_close",
        "atomic_replacement",
    ):
        assert phases[phase]["status"] == "measured_host_wall"
        assert phases[phase]["seconds"] >= 0.0
    counts = report["operation_counts"]
    assert report["operation_counts_schema_version"] == (
        "selected_pass_operation_counts_v1"
    )
    assert counts["canonical_body_encoding_hash"]["hashes"] == 1
    assert counts["canonical_body_encoding_hash"]["bytes_read"] > 0
    assert counts["staging_json_encoding"]["bytes_written"] > 0
    assert counts["temporary_file_write"]["bytes_written"] > 0
    assert counts["atomic_replacement"]["files_replaced"] == 1
