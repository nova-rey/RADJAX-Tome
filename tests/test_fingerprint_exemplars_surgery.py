from __future__ import annotations

from pathlib import Path

from radjax_tome.fingerprint.capture_summary import (
    RealTeacherCaptureSummary,
    SourceCorpusReference,
    TeacherIdentity,
    read_real_teacher_capture_summary,
    validate_real_teacher_capture_summary,
    write_real_teacher_capture_summary,
)
from radjax_tome.fingerprint.corridor import (
    CorridorMeasurementRecord,
    CorridorMeasurementReport,
    read_corridor_measurement_report,
    validate_corridor_measurement_report,
    write_corridor_measurement_report,
)
from radjax_tome.fingerprint.exemplars import (
    FingerprintExemplarManifest,
    FingerprintExemplarRecord,
    FingerprintExemplarShard,
    read_fingerprint_exemplar_records,
    validate_fingerprint_exemplar_records,
    write_fingerprint_exemplar_manifest,
    write_fingerprint_exemplar_records,
)


def test_exemplar_manifest_records_round_trip_and_validate(tmp_path: Path) -> None:
    manifest = FingerprintExemplarManifest(
        payload_type="dense_probs",
        shards=(FingerprintExemplarShard(path="exemplars-00000.jsonl", num_records=1),),
        num_records=1,
    )
    write_fingerprint_exemplar_manifest(tmp_path / "exemplar_manifest.json", manifest)

    record = FingerprintExemplarRecord(
        example_id="ex-1",
        input_ids=(1, 2, 3),
        position=1,
        weight=0.5,
        teacher_probs=(0.2, 0.3, 0.5),
        mode_id=0,
        interestingness_score=0.9,
        reason_codes=("high_margin",),
    )
    path = tmp_path / "exemplars-00000.jsonl"
    write_fingerprint_exemplar_records(path, [record])
    records = read_fingerprint_exemplar_records(path, payload_type="dense_probs")

    result = validate_fingerprint_exemplar_records(
        records,
        max_seq_len=3,
        vocab_size=3,
        payload_type="dense_probs",
    )
    assert result.ok
    assert result.metadata["records"] == 1

    bad = validate_fingerprint_exemplar_records(
        records,
        max_seq_len=4,
        vocab_size=3,
        payload_type="dense_probs",
    )
    assert not bad.ok
    assert "input_ids length" in bad.blockers[0]


def test_corridor_measurement_report_round_trip(tmp_path: Path) -> None:
    report = CorridorMeasurementReport(
        status="pass",
        artifact_dir="fingerprint",
        records_measured=1,
        modes_measured=1,
        tracked_stats=("top1_margin",),
        measurements=(
            CorridorMeasurementRecord(
                example_id="ex-1",
                position=0,
                mode_id=0,
                stats={"top1_margin": 0.2},
                bounds={"top1_margin": (0.1, 0.3)},
            ),
        ),
    )
    path = write_corridor_measurement_report(tmp_path / "corridor.json", report)
    loaded = read_corridor_measurement_report(path)
    ok, blockers = validate_corridor_measurement_report(loaded)

    assert ok, blockers
    assert loaded.measurements[0].bounds["top1_margin"] == (0.1, 0.3)


def test_real_teacher_capture_summary_round_trip(tmp_path: Path) -> None:
    summary = RealTeacherCaptureSummary(
        status="pass",
        artifact_dir="fingerprint",
        teacher=TeacherIdentity(
            model_name_or_path="local/tiny",
            tokenizer_name_or_path="local/tok",
            local_files_only=True,
            allow_downloads=False,
        ),
        source_corpus=SourceCorpusReference(
            path="prompts.jsonl",
            corpus_id="unit",
            prompt_count=2,
        ),
        manifest_path="manifest.json",
        modes_path="modes.json",
        target_shards=("targets/targets-00000.jsonl",),
        examples_processed=2,
        tokens_processed=6,
        target_positions_processed=2,
        modes_discovered=1,
        exemplars_retained=0,
    )
    path = write_real_teacher_capture_summary(tmp_path / "capture.json", summary)
    loaded = read_real_teacher_capture_summary(path)
    ok, blockers = validate_real_teacher_capture_summary(loaded)

    assert ok, blockers
    assert loaded.teacher.model_name_or_path == "local/tiny"
    assert loaded.source_corpus.path == "prompts.jsonl"
