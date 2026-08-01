"""Native Path-B v4 publication uses the durable shard transaction."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from radjax_tome.builder import TeacherTextbookBuildConfig, build_fake_teacher_textbook
from radjax_tome.builder.production_stages import v4_publication
from radjax_tome.builder.production_stages.v4_publication import (
    native_v4_archive_path,
    native_v4_directory_path,
    native_v4_staging_path,
    publish_native_path_b_v4,
)


def _record(index: int) -> dict[str, object]:
    return {
        "selected_example_id": f"example-{index}",
        "selected_position": index,
        "selected_score": 1.0,
        "score_selected_position_entropy": 1.0,
        "score_top_token_id": 3,
        "source_shard_id": 0,
        "source_row": index,
        "source_position": index,
        "source_score": 1.0,
        "source_top_token_id": 3,
        "source_score_policy": "fixture",
        "payload_ref": {"fixture": "payload"},
        "selected_policy": "fixture",
        "source_delivery_path": "fixture",
        "top_token_ids": [3],
        "top_log_probs": [-0.1],
        "top_probs": [0.9],
        "top_selection_mask": [True],
        "effective_top_k": 1,
        "top_mass": 0.9,
        "tail_mass": 0.1,
        "bucket_masses": [0.1],
        "teacher_entropy": 1.0,
        "sequence_length": 8,
        "vocab_size": 16,
        "num_buckets": 1,
        "dynamic_top_k": False,
        "dynamic_mass_threshold": 0.9,
        "dynamic_top_k_max": 1,
        "top_k_saturated": False,
        "long_tail_class": "fixture",
        "long_tail_warnings": [],
        "effective_top_k_fraction_of_vocab": 0.0625,
        "semantic_tail_tag": "fixture",
        "selected_board": "primary",
        "corridor_mode_id": 0,
        "corridor_fingerprint_id": 0,
        "corridor_assignment_status": "fixture",
    }


def _source(tmp_path: Path, *, invalid_position: bool = False) -> Path:
    source = tmp_path / "fake_tome"
    build_fake_teacher_textbook(
        TeacherTextbookBuildConfig(
            output_dir=source,
            teacher_mode="fake",
            max_examples=2,
            sequence_length=8,
            overwrite=True,
        )
    )
    selected = source / "selected_exemplars"
    selected.mkdir(exist_ok=True)
    records = [_record(0), _record(1), _record(2)]
    if invalid_position:
        records[0]["selected_position"] = "not-an-int"
    (selected / "selected-exemplars-00000.json").write_text(
        json.dumps(
            {
                "schema_version": "selected_exemplar_payload_shard_v1",
                "selected_exemplars": records,
            }
        ),
        encoding="utf-8",
    )
    return source


def _config(source: Path, **changes: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "output_dir": source,
        "overwrite": False,
        "resume": False,
        "payload_records_per_shard": 2,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_native_v4_resume_rejects_corrupt_completed_publication(tmp_path: Path) -> None:
    source = _source(tmp_path)
    publish_native_path_b_v4(_config(source))
    shard = (
        native_v4_directory_path(source) / "selected_exemplars/shards/shard-00000.jsonl"
    )
    shard.write_bytes(b"corrupt\n")

    with pytest.raises(ValueError, match="Contract validation.*digest_mismatch"):
        publish_native_path_b_v4(_config(source, resume=True))


def test_native_v4_resume_reuses_sealed_shards_after_interruption(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)

    def interrupt(event: str) -> None:
        if event == "after_v4_shard_sealed":
            raise RuntimeError("interrupt after sealed shard")

    with pytest.raises(RuntimeError, match="sealed shard"):
        publish_native_path_b_v4(_config(source, v4_publication_hook=interrupt))
    stage = native_v4_staging_path(source)
    assert (stage / "v4-shard-receipt.json").is_file()
    assert not native_v4_directory_path(source).exists()

    publication = publish_native_path_b_v4(_config(source, resume=True))

    assert publication.directory.selected_count == 3
    assert publication.archive_path.is_file()


def test_native_v4_interruption_before_promotion_leaves_no_final_archive(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)

    def interrupt(event: str) -> None:
        if event == "before_v4_final_promotion":
            raise RuntimeError("interrupt before promotion")

    with pytest.raises(RuntimeError, match="before promotion"):
        publish_native_path_b_v4(_config(source, v4_publication_hook=interrupt))
    assert not native_v4_directory_path(source).exists()
    assert not native_v4_archive_path(source).exists()

    publication = publish_native_path_b_v4(_config(source, resume=True))
    assert publication.directory.root == native_v4_directory_path(source)
    assert publication.archive_path == native_v4_archive_path(source)


@pytest.mark.parametrize(
    "event",
    ("before_v4_directory_materialization", "before_v4_final_promotion"),
)
def test_native_v4_prepublication_interruptions_leave_no_public_outputs(
    tmp_path: Path, event: str
) -> None:
    source = _source(tmp_path)

    def interrupt(observed: str) -> None:
        if observed == event:
            raise RuntimeError(event)

    with pytest.raises(RuntimeError, match=event):
        publish_native_path_b_v4(_config(source, v4_publication_hook=interrupt))
    assert not native_v4_directory_path(source).exists()
    assert not native_v4_archive_path(source).exists()


def test_native_v4_archive_packing_failure_never_publishes_partial_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    original = v4_publication.pack_sharded_tome_v4

    def interrupting_pack(root: Path, output: Path, **kwargs: object) -> Path:
        output.write_bytes(b"partial archive")
        raise RuntimeError("interrupt during archive packing")

    monkeypatch.setattr(v4_publication, "pack_sharded_tome_v4", interrupting_pack)
    with pytest.raises(RuntimeError, match="during archive packing"):
        publish_native_path_b_v4(_config(source))
    assert not native_v4_archive_path(source).exists()
    assert not native_v4_directory_path(source).exists()

    monkeypatch.setattr(v4_publication, "pack_sharded_tome_v4", original)
    publication = publish_native_path_b_v4(_config(source, resume=True))
    assert publication.archive_path.is_file()


def test_native_v4_contract_validation_happens_before_promotion(tmp_path: Path) -> None:
    source = _source(tmp_path, invalid_position=True)

    with pytest.raises(
        ValueError, match="Contract validation.*payload_semantic_projection_invalid"
    ):
        publish_native_path_b_v4(_config(source))

    assert not native_v4_directory_path(source).exists()
    assert not native_v4_archive_path(source).exists()
