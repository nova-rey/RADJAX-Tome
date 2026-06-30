from __future__ import annotations

from pathlib import Path

from radjax_tome.reports import (
    BaselineArmReport,
    FingerprintArtifactByteBudget,
    FingerprintBaselineComparisonReport,
    FingerprintQualityPerByteReport,
    build_fingerprint_arc_report,
    build_quality_per_byte_delta,
    read_fingerprint_arc_report,
    read_fingerprint_baseline_report,
    read_fingerprint_quality_report,
    render_fingerprint_arc_summary,
    render_fingerprint_baseline_summary,
    render_fingerprint_quality_summary,
    write_fingerprint_arc_report,
    write_fingerprint_baseline_report,
    write_fingerprint_quality_report,
)
from radjax_tome.reports.arc import REQUIRED_ARC2_FLAGS


def test_quality_per_byte_report_round_trip(tmp_path: Path) -> None:
    budget = FingerprintArtifactByteBudget(
        artifact_kind="behavioral_fingerprint",
        fingerprint_artifact_size_bytes=1_000_000,
        targets_size_bytes=800_000,
        exemplars_size_bytes=200_000,
        target_records=10,
        exemplar_records=2,
        modes_discovered=1,
    )
    delta = build_quality_per_byte_delta(
        reference_score=1.0,
        fingerprint_score=0.75,
        artifact_size_bytes=budget.fingerprint_artifact_size_bytes,
    )
    report = FingerprintQualityPerByteReport(
        status="pass",
        artifact_budget=budget,
        quality_per_byte=delta,
        arms=({"arm_id": "fingerprint_corridor", "status": "pass"},),
        limitations=("tiny smoke only",),
        claims={"winner_declared": False},
    )
    path = write_fingerprint_quality_report(tmp_path / "quality.json", report)
    loaded = read_fingerprint_quality_report(path)

    assert loaded.quality_per_byte.delta_per_mb == 0.25
    assert "Artifact bytes: 1000000" in render_fingerprint_quality_summary(loaded)


def test_baseline_report_summary_round_trip(tmp_path: Path) -> None:
    report = FingerprintBaselineComparisonReport(
        status="pass",
        arms=(
            BaselineArmReport(
                arm_id="baseline_init_only",
                status="pass",
                arm_kind="reference_baseline",
                trained=False,
                artifact_kind="none",
                artifact_size_bytes=0,
                teacher_required_during_training=False,
            ),
            BaselineArmReport(
                arm_id="fingerprint_corridor",
                status="pass",
                arm_kind="fingerprint_method",
                trained=True,
                artifact_kind="behavioral_fingerprint",
                artifact_size_bytes=12,
                teacher_required_during_training=False,
            ),
        ),
        claims={"winner_declared": False},
    )
    path = write_fingerprint_baseline_report(tmp_path / "baseline.json", report)
    loaded = read_fingerprint_baseline_report(path)
    summary = render_fingerprint_baseline_summary(loaded)

    assert loaded.arms[1].artifact_kind == "behavioral_fingerprint"
    assert "fingerprint_corridor" in summary


def test_arc_report_from_snapshot_round_trip(tmp_path: Path) -> None:
    snapshot = {"main_contains": {flag: True for flag, _ in REQUIRED_ARC2_FLAGS}}
    report = build_fingerprint_arc_report(snapshot, snapshot_path="snapshot.yaml")
    path = write_fingerprint_arc_report(tmp_path / "arc.json", report)
    loaded = read_fingerprint_arc_report(path)
    summary = render_fingerprint_arc_summary(loaded)

    assert loaded.status == "pass"
    assert loaded.recommendation == "go_with_constraints"
    assert "P148 quality-per-byte smoke: True" in summary
