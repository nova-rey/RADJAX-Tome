"""M9 prerequisite regressions for accepted M8 authority."""

from pathlib import Path

from radjax_tome.builder.config import (
    PRODUCTION_PRESETS,
    apply_production_preset,
    canonical_production_build_intent,
)


def test_contract_pin_is_the_accepted_buffer_native_commit() -> None:
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert "RADJAX-Contract.git@373e3d17060d4ce1c4a0db6065c9289da714bde7" in pyproject


def test_accepted_buffer_native_contract_api_imports() -> None:
    from radjax_contract.tome.m8g import (  # noqa: PLC0415
        compact_body_from_buffers,
        encode_compact_body_packed_from_buffers,
    )

    assert callable(compact_body_from_buffers)
    assert callable(encode_compact_body_packed_from_buffers)


def test_every_canonical_production_preset_uses_selected_batch_one() -> None:
    intent = canonical_production_build_intent(
        teacher_model="model",
        dataset_path=Path("dataset"),
        corpus_manifest_path=Path("manifest"),
        teacher_model_provenance_path=Path("provenance"),
        output_dir=Path("output"),
    )
    assert PRODUCTION_PRESETS == ("smoke", "t4-1k", "t4-10k", "production-100k")
    for preset in PRODUCTION_PRESETS:
        assert (
            apply_production_preset(intent, preset).selection.selected_rerun_batch_size
            == 1
        )
