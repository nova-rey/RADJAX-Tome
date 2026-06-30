from __future__ import annotations

import sys
from pathlib import Path

import pytest

from radjax_tome.backends import (
    HFTeacherExportConfig,
    HFTeacherSpecimenConfig,
    build_hf_export_metadata,
    build_hf_teacher_specimen_dry_run,
    build_hf_teacher_specimen_swap_report,
    read_hf_export_metadata,
    read_hf_teacher_specimen_report,
    run_hf_teacher_specimen_smoke,
    validate_hf_export_config,
    validate_hf_teacher_specimen_config,
    write_hf_export_metadata,
    write_hf_teacher_specimen_report,
)


def test_hf_specimen_dry_run_round_trip_without_heavy_imports(tmp_path: Path) -> None:
    config = HFTeacherSpecimenConfig(
        model_id="local/tiny",
        prompts=("hello", "world"),
        sequence_length=4,
        local_files_only=True,
        allow_downloads=False,
    )
    result = build_hf_teacher_specimen_dry_run(
        config,
        target_store=tmp_path / "targets",
    )
    path = write_hf_teacher_specimen_report(tmp_path / "specimen.json", result)
    loaded = read_hf_teacher_specimen_report(path)

    assert loaded.status == "dry_run"
    assert loaded.local_files_only
    assert loaded.num_examples == 2
    assert not any(
        name.startswith(("jax", "torch", "transformers")) for name in sys.modules
    )


def test_hf_specimen_validation_and_swap_report(tmp_path: Path) -> None:
    invalid = HFTeacherSpecimenConfig(
        model_id="local/tiny",
        local_files_only=True,
        allow_downloads=True,
    )
    with pytest.raises(ValueError, match="allow_downloads"):
        validate_hf_teacher_specimen_config(invalid)

    unavailable = run_hf_teacher_specimen_smoke(
        HFTeacherSpecimenConfig(model_id="local/tiny"),
        target_store=tmp_path / "targets",
    )
    swap = build_hf_teacher_specimen_swap_report((unavailable,))
    assert swap.status == "pass"
    assert swap.unavailable == 1


def test_hf_export_metadata_round_trip(tmp_path: Path) -> None:
    config = HFTeacherExportConfig(
        resolved_model_id="local/tiny",
        tokenizer_id="local/tok",
        sequence_length=8,
        prompt_count=3,
        vocab_size=16,
        include_logits=True,
        include_attention_targets=False,
        local_files_only=True,
        allow_downloads=False,
    )
    ok, blockers = validate_hf_export_config(config)
    assert ok, blockers

    metadata = build_hf_export_metadata(config)
    path = write_hf_export_metadata(tmp_path / "hf_export.json", metadata)
    loaded = read_hf_export_metadata(path)

    assert loaded.teacher_model_id == "local/tiny"
    assert loaded.targets["logits"]
    assert loaded.prompt_count == 3
