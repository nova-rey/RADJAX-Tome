from __future__ import annotations

import hashlib
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from radjax_tome.builder.corridor_artifacts import build_corridor_artifacts
from radjax_tome.builder.teacher_textbook import (
    EMISSION_CONFIG_FIELDS,
    TinyTextExample,
    validate_teacher_textbook,
    write_teacher_textbook_validation_report,
)
from radjax_tome.io.json import write_json
from radjax_tome.targets.schema import (
    TEACHER_TARGET_STORE_SCHEMA_VERSION,
    TEACHER_TARGET_STORE_VERSION,
    TargetStoreMetadata,
)
from radjax_tome.targets.store import TeacherTargetStore
from radjax_tome.tome.cover_page import validate_tome_cover_page, write_cover_page

FIXTURE_ID = "production_multi_surface_v1"
FIXTURE_SCHEMA_VERSION = "production_tome_fixture_v1"
FIXTURE_EPOCH = "2000-01-01T00:00:00Z"
FIXTURE_PRODUCER_BASELINE = "c8fb9ac4d92a33d8342c2249bae939e21221125a"
NUM_EXAMPLES = 8
SEQUENCE_LENGTH = 4
VOCAB_SIZE = 32
SELECTED_COUNT = 4
PAYLOAD_WIDTH = 5
BUCKET_COUNT = 3


def build_production_contract_fixture(
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    producer_commit: str = FIXTURE_PRODUCER_BASELINE,
) -> Path:
    fixture_root = Path(output_dir)
    if fixture_root.exists():
        if not overwrite:
            raise ValueError(f"fixture output already exists: {fixture_root}")
        shutil.rmtree(fixture_root)
    artifact = fixture_root / "artifact"
    metadata = TargetStoreMetadata(
        schema_version=TEACHER_TARGET_STORE_SCHEMA_VERSION,
        target_store_version=TEACHER_TARGET_STORE_VERSION,
        model_id="fake-production-teacher",
        model_family="fake",
        tokenizer_id="fake-production-tokenizer",
        tokenizer_hash="sha256:fixture-tokenizer-v1",
        vocab_size=VOCAB_SIZE,
        target_type="corridor_exemplar_score_pass_v1",
        dtype="float32",
        sequence_length=SEQUENCE_LENGTH,
        num_examples=NUM_EXAMPLES,
        shard_count=1,
        created_by="radjax-tome.production-contract-fixture",
        created_at=FIXTURE_EPOCH,
        source={"kind": "deterministic_synthetic_fixture"},
        provenance={"fixture_id": FIXTURE_ID},
        target_params={
            "corridor_stat_top_k": "32",
            "min_corridor_stat_top_k": "32",
            "dynamic_top_k_min": "2",
            "dynamic_top_k_max": "5",
            "dynamic_mass_threshold": "0.95",
        },
    )
    store = TeacherTargetStore.create(artifact, metadata)
    examples = tuple(
        TinyTextExample(example_id=f"fixture-{index:02d}", text=f"fixture {index}")
        for index in range(NUM_EXAMPLES)
    )
    store.write_shard(0, _score_pass_arrays())
    _write_sidecars(artifact)

    selected_records, selected_payloads = _selected_exemplars()
    corridor = build_corridor_artifacts(
        output_dir=artifact,
        examples=examples,
        selected_records=selected_records,
        selected_payloads=selected_payloads,
        delivery_path="two_pass_rerun_selected",
        non_selected_exemplar_payload_retained=False,
    )
    _write_selected_delivery(
        artifact,
        selected_records=selected_records,
        selected_payloads=selected_payloads,
        corridor_fields=corridor.report_fields(),
    )
    initial_report = validate_teacher_textbook(artifact)
    write_teacher_textbook_validation_report(
        initial_report,
        artifact / "validation_report.json",
    )
    if initial_report.status != "pass":
        raise ValueError(
            "generated fixture failed producer validation: "
            + "; ".join(initial_report.blockers)
        )
    write_cover_page(artifact)
    cover_report = validate_tome_cover_page(artifact)
    if not cover_report.ok:
        raise ValueError(
            "generated fixture failed cover-page validation: "
            + "; ".join(cover_report.blockers)
        )
    write_json(
        fixture_root / "FIXTURE_PROVENANCE.json",
        {
            "contract_commit": None,
            "created_at": FIXTURE_EPOCH,
            "fixture_id": FIXTURE_ID,
            "fixture_schema_version": FIXTURE_SCHEMA_VERSION,
            "generator": "radjax_tome.tome.golden_fixture",
            "producer_commit": producer_commit,
            "source_examples": NUM_EXAMPLES,
            "tree_digest": artifact_tree_digest(artifact),
            "claims_not_made": [
                "no_model_quality_claim",
                "no_network_verification_claim",
                "no_student_training_claim",
                "no_delivery_path_quality_parity_claim",
            ],
        },
    )
    return artifact


def artifact_tree_digest(artifact_dir: str | Path) -> str:
    root = Path(artifact_dir)
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _score_pass_arrays() -> dict[str, np.ndarray]:
    input_ids = (
        np.arange(
            NUM_EXAMPLES * SEQUENCE_LENGTH,
            dtype=np.int32,
        ).reshape(NUM_EXAMPLES, SEQUENCE_LENGTH)
        % VOCAB_SIZE
    )
    attention_mask = np.ones_like(input_ids, dtype=np.int32)
    low_mode = np.arange(NUM_EXAMPLES)[:, None] < (NUM_EXAMPLES // 2)
    entropy = np.where(low_mode, 0.5, 3.0).astype(np.float32)
    entropy = np.broadcast_to(entropy, input_ids.shape).copy()
    margin = np.where(low_mode, 0.02, 0.20).astype(np.float32)
    margin = np.broadcast_to(margin, input_ids.shape).copy()
    top8_mass = np.where(low_mode, 0.50, 0.80).astype(np.float32)
    top8_mass = np.broadcast_to(top8_mass, input_ids.shape).copy()
    top32_mass = np.where(low_mode, 0.70, 0.95).astype(np.float32)
    top32_mass = np.broadcast_to(top32_mass, input_ids.shape).copy()
    confidence = np.where(low_mode, 0.75, 0.40).astype(np.float32)
    confidence = np.broadcast_to(confidence, input_ids.shape).copy()
    selected_positions = np.arange(NUM_EXAMPLES, dtype=np.int32) % SEQUENCE_LENGTH
    row = np.arange(NUM_EXAMPLES, dtype=np.int32)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "corridor_top_token_ids": input_ids.copy(),
        "corridor_teacher_entropy": entropy,
        "corridor_entropy": entropy.copy(),
        "corridor_top1_margin": margin,
        "corridor_top8_mass": top8_mass,
        "corridor_top32_mass": top32_mass,
        "corridor_tail_mass": (1.0 - top32_mass).astype(np.float32),
        "corridor_confidence": confidence,
        "corridor_lengths": np.full(
            (NUM_EXAMPLES,),
            SEQUENCE_LENGTH,
            dtype=np.int32,
        ),
        "score_example_ids": row.copy(),
        "score_max_entropy": np.max(entropy, axis=1),
        "score_mean_entropy": np.mean(entropy, axis=1, dtype=np.float32),
        "score_selected_position": selected_positions,
        "score_top_token_id": input_ids[row, selected_positions],
        "score_selected_position_entropy": entropy[row, selected_positions],
        "score_confidence_at_selected_position": confidence[
            row,
            selected_positions,
        ],
        "score_source_policy_ids": np.zeros((NUM_EXAMPLES,), dtype=np.int32),
        "score_lengths": np.full(
            (NUM_EXAMPLES,),
            SEQUENCE_LENGTH,
            dtype=np.int32,
        ),
    }


def _selected_exemplars() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    base_probs = np.asarray([0.24, 0.18, 0.12, 0.08, 0.05], dtype=np.float32)
    for rank, effective_top_k in enumerate((2, 3, 4, 5), start=1):
        example_index = rank - 1
        position = example_index % SEQUENCE_LENGTH
        example_id = f"fixture-{example_index:02d}"
        selected_score = 0.5 if example_index < (NUM_EXAMPLES // 2) else 3.0
        score_top_token_id = (example_index * SEQUENCE_LENGTH + position) % VOCAB_SIZE
        mask = [index < effective_top_k for index in range(PAYLOAD_WIDTH)]
        probs = [
            float(value) if mask[index] else 0.0
            for index, value in enumerate(base_probs.tolist())
        ]
        log_probs = [
            float(math.log(value)) if mask[index] else -100.0
            for index, value in enumerate(base_probs.tolist())
        ]
        top_mass = float(sum(probs))
        tail_mass = float(1.0 - top_mass)
        bucket_masses = [tail_mass / BUCKET_COUNT for _ in range(BUCKET_COUNT)]
        record = {
            "rank": rank,
            "selected_example_id": example_id,
            "selected_position": position,
            "selected_score": selected_score,
            "score_selected_position_entropy": selected_score,
            "score_top_token_id": score_top_token_id,
            "source_shard_id": 0,
            "source_row": example_index,
            "selected_policy": "entropy_top_n_v1",
            "source_delivery_path": "two_pass_rerun_selected",
        }
        payload = {
            **record,
            "top_token_ids": [
                score_top_token_id
                if index == 0
                else (example_index * PAYLOAD_WIDTH + index) % VOCAB_SIZE
                for index in range(PAYLOAD_WIDTH)
            ],
            "top_log_probs": log_probs,
            "top_probs": probs,
            "top_selection_mask": mask,
            "effective_top_k": effective_top_k,
            "top_mass": top_mass,
            "tail_mass": tail_mass,
            "bucket_masses": bucket_masses,
            "teacher_entropy": selected_score,
            "sequence_length": SEQUENCE_LENGTH,
            "vocab_size": VOCAB_SIZE,
            "num_buckets": BUCKET_COUNT,
            "dynamic_top_k": {
                "effective_top_k": effective_top_k,
                "policy": "mass_threshold_v1",
                "requested_top_k": PAYLOAD_WIDTH,
                "score_policy": "entropy_top_n_v1",
                "source_payload": "deterministic_fixture",
            },
        }
        records.append(record)
        payloads.append(payload)
    return records, payloads


def _write_sidecars(artifact: Path) -> None:
    write_json(
        artifact / "vocab_contract.json",
        {
            "model_family": "fake",
            "model_id": "fake-production-teacher",
            "special_tokens": {"bos_token_id": 1, "eos_token_id": 2},
            "tokenizer_hash": "sha256:fixture-tokenizer-v1",
            "tokenizer_id": "fake-production-tokenizer",
            "vocab_size": VOCAB_SIZE,
        },
    )
    write_json(
        artifact / "teacher_manifest.json",
        {
            "allow_downloads": False,
            "artifact_type": "teacher_textbook",
            "artifact_version": 1,
            "claims_not_made": [
                "no_model_quality_claim",
                "no_network_verification_claim",
                "no_student_training_claim",
            ],
            "corpus_provenance": {
                "corpus_hash": "sha256:fixture-corpus-v1",
                "source": "deterministic_synthetic_fixture",
                "split": "training",
            },
            "created_at": FIXTURE_EPOCH,
            "dtype": "float32",
            "local_files_only": True,
            "num_examples": NUM_EXAMPLES,
            "sequence_length": SEQUENCE_LENGTH,
            "shard_count": 1,
            "target_type": "corridor_exemplar_score_pass_v1",
            "teacher_backend_type": "fake",
            "teacher_model_id": "fake-production-teacher",
            "teacher_model_provenance": {
                "allow_downloads": False,
                "config_hash": "sha256:fixture-config-v1",
                "local_files_only": True,
                "model_identity_confidence": "fixture",
                "model_name": "fake-production-teacher",
                "model_revision": "fixture-v1",
                "model_source_kind": "deterministic_fixture",
                "network_used": False,
                "schema_version": "teacher_model_provenance_v1",
                "tokenizer_hash": "sha256:fixture-tokenizer-v1",
                "weights_hash": "sha256:fixture-weights-v1",
            },
            "tokenizer_id": "fake-production-tokenizer",
            "vocab_contract_path": "vocab_contract.json",
            "vocab_size": VOCAB_SIZE,
        },
    )
    emission = {field: None for field in EMISSION_CONFIG_FIELDS}
    emission.update(
        {
            "batch_size": NUM_EXAMPLES,
            "dataset_source": "deterministic_synthetic_fixture",
            "dynamic_mass_threshold": 0.95,
            "dynamic_top_k_max": 5,
            "dynamic_top_k_min": 2,
            "include_hidden_states": False,
            "logits_dtype": "float32",
            "max_examples": NUM_EXAMPLES,
            "sampling_used": False,
            "seed": 0,
            "sequence_length": SEQUENCE_LENGTH,
            "teacher_mode": "fake",
            "temperature": 0.0,
            "top_k": PAYLOAD_WIDTH,
            "top_p": 1.0,
        }
    )
    write_json(artifact / "emission_config.json", emission)
    write_json(
        artifact / "validation_report.json",
        {
            "blockers": [],
            "claims_not_made": ["no_student_training_claim"],
            "status": "pass",
            "warnings": [],
        },
    )


def _write_selected_delivery(
    artifact: Path,
    *,
    selected_records: list[dict[str, Any]],
    selected_payloads: list[dict[str, Any]],
    corridor_fields: dict[str, Any],
) -> None:
    write_json(
        artifact / "leaderboards" / "selected_exemplars.json",
        {
            "created_at": FIXTURE_EPOCH,
            "delivery_path": "two_pass_rerun_selected",
            "schema_version": "selected_exemplars_v1",
            "score_policy": "entropy_top_n_v1",
            "selected_exemplars": selected_records,
        },
    )
    write_json(
        artifact / "leaderboards" / "leaderboard_report.json",
        {
            "created_at": FIXTURE_EPOCH,
            "schema_version": "leaderboard_report_v1",
            "status": "pass",
        },
    )
    write_json(
        artifact / "selected_exemplars" / "selected-exemplars-00000.json",
        {
            "delivery_path": "two_pass_rerun_selected",
            "schema_version": "selected_exemplar_payload_shard_v1",
            "selected_exemplars": selected_payloads,
        },
    )
    report = {
        "blockers": [],
        "claims_not_made": {
            "no_student_training_quality_claim": True,
        },
        "created_at": FIXTURE_EPOCH,
        "delivery_path": "two_pass_rerun_selected",
        "non_selected_exemplar_payload_retained": False,
        "num_examples_scored": NUM_EXAMPLES,
        "num_positions_scored": NUM_EXAMPLES * SEQUENCE_LENGTH,
        "num_selected_exemplars": SELECTED_COUNT,
        "schema_version": "selected_exemplar_delivery_report_v1",
        "selected_example_count": SELECTED_COUNT,
        "selected_exemplar_payload_retained": True,
        "selection_enabled": True,
        "status": "pass",
        "teacher_rerun_count": SELECTED_COUNT,
        "warnings": [],
    }
    report.update(
        {
            key: _fixture_relative_value(artifact, key, value)
            for key, value in corridor_fields.items()
        }
    )
    write_json(artifact / "delivery_report.json", report)


def _fixture_relative_value(
    artifact: Path,
    key: str,
    value: Any,
) -> Any:
    if not key.endswith("_path") or not isinstance(value, str):
        return value
    path = Path(value)
    try:
        return path.relative_to(artifact).as_posix()
    except ValueError:
        return value
