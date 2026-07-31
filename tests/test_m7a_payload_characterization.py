"""M7A characterization locks for the pre-sharding selected-payload surface."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

from radjax_tome.builder.delivery._shared import _REQUIRED_SELECTED_PAYLOAD_FIELDS
from radjax_tome.tome.canonical_artifact import derive_tome_semantic_identity
from radjax_tome.tome.packaging import StudentTomeReader


def _write(root: Path, relative: str, payload: object) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _artifact(root: Path, payload_shards: tuple[list[dict[str, object]], ...]) -> None:
    _write(
        root,
        "metadata.json",
        {
            "target_type": "corridor_exemplar_v1",
            "sequence_length": 128,
            "vocab_size": 256,
            "tome_version": 1,
        },
    )
    for relative in (
        "vocab_contract.json",
        "teacher_manifest.json",
        "emission_config.json",
        "corridors/corridor_summary.json",
        "corridors/corridor_modes.json",
        "corridors/mode_assignments.json",
        "leaderboards/selected_exemplars.json",
    ):
        _write(root, relative, {"schema_version": "test"})
    for index, records in enumerate(payload_shards):
        _write(
            root,
            f"selected_exemplars/selected-exemplars-{index:05d}.json",
            {
                "schema_version": "selected_exemplar_payload_shard_v1",
                "selected_exemplars": records,
            },
        )


def test_m7a_locks_current_selected_payload_field_surface() -> None:
    assert _REQUIRED_SELECTED_PAYLOAD_FIELDS == (
        "selected_example_id",
        "selected_position",
        "selected_score",
        "score_selected_position_entropy",
        "score_top_token_id",
        "source_shard_id",
        "source_row",
        "source_position",
        "source_score",
        "source_top_token_id",
        "source_score_policy",
        "payload_ref",
        "selected_policy",
        "source_delivery_path",
        "top_token_ids",
        "top_log_probs",
        "top_probs",
        "top_selection_mask",
        "effective_top_k",
        "top_mass",
        "tail_mass",
        "bucket_masses",
        "teacher_entropy",
        "sequence_length",
        "vocab_size",
        "num_buckets",
        "dynamic_top_k",
        "dynamic_mass_threshold",
        "dynamic_top_k_max",
        "top_k_saturated",
        "long_tail_class",
        "long_tail_warnings",
        "effective_top_k_fraction_of_vocab",
        "semantic_tail_tag",
        "selected_board",
        "corridor_mode_id",
        "corridor_fingerprint_id",
        "corridor_assignment_status",
    )


def test_m7a_v3_identity_is_currently_sensitive_to_payload_file_grouping(
    tmp_path: Path,
) -> None:
    first = tmp_path / "one-shard"
    second = tmp_path / "two-shards"
    first_record = {"selected_example_id": "one", "selected_position": 1}
    second_record = {"selected_example_id": "two", "selected_position": 2}
    _artifact(first, ([first_record, second_record],))
    _artifact(second, ([first_record], [second_record]))

    assert (
        derive_tome_semantic_identity(first).semantic_digest
        != derive_tome_semantic_identity(second).semantic_digest
    )


def test_m7a_public_student_reader_owns_an_eager_payload_tuple() -> None:
    assert "selected_payloads" in {field.name for field in fields(StudentTomeReader)}
