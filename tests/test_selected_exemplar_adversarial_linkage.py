from __future__ import annotations

import copy
import json
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import radjax_tome.builder.exemplar_delivery as exemplar_delivery
from radjax_tome.audit import audit_selected_linkage
from radjax_tome.backends import TeacherBackendConfig
from radjax_tome.builder import ProductionBuildConfig, build_production_gpu_tome
from radjax_tome.builder.exemplar_delivery import ExemplarDeliveryConfig
from radjax_tome.builder.teacher_textbook import TinyTextExample
from radjax_tome.corpora import CorpusBuildConfig, build_corpus_artifact
from radjax_tome.io.json import read_json_object, write_json
from radjax_tome.provenance import (
    inspect_teacher_model,
    write_teacher_model_provenance,
)
from tests.helpers.subprocess import run_cli

ROOT = Path(__file__).resolve().parents[1]
PATH_A = "one_pass_pruned_candidate"
PATH_B = "two_pass_rerun_selected"
SEQUENCE_LENGTH = 11
TOP_K = 5
EXAMPLE_IDS = (
    "corpus_000999",
    "corpus_000003",
    "corpus_000020",
    "corpus_000777",
    "corpus_000042",
    "corpus_000001",
    "corpus_000888",
    "corpus_000006",
    "corpus_000005",
    "corpus_000123",
    "corpus_000004",
    "corpus_000314",
)
SHARD_SIZES = (4, 3, 5)
SELECTED_COORDINATES = (
    (2, 3, 1),
    (0, 1, 0),
    (1, 2, 2),
    (0, 3, 1),
    (2, 0, 0),
    (1, 0, 2),
    (0, 0, 1),
    (2, 4, 0),
    (2, 1, 2),
)


def test_clean_adversarial_artifacts_pass_and_cli_audits_all_payload_shards(
    tmp_path: Path,
) -> None:
    path_a = _build_synthetic_artifact(tmp_path / "path-a", delivery_path=PATH_A)
    path_b = _build_synthetic_artifact(tmp_path / "path-b", delivery_path=PATH_B)

    path_a_report = audit_selected_linkage(path_a, strict=True)
    path_b_report = audit_selected_linkage(path_b, strict=True)
    cli = run_cli(
        ROOT,
        "audit-selected-linkage",
        "--artifact",
        str(path_a),
        "--strict",
    )

    assert path_a_report.status == "pass"
    assert path_a_report.selected_count == 9
    assert path_a_report.checked_count == 9
    assert path_a_report.bad_count == 0
    assert path_a_report.path_a_payload_authority == "pass"
    assert path_a_report.path_b_score_pass_authority is None
    assert path_b_report.status == "pass"
    assert path_b_report.path_b_score_pass_authority == "pass"
    assert path_b_report.path_a_payload_authority is None
    assert len(list((path_a / "selected_exemplars").glob("*.json"))) == 3
    assert cli.returncode == 0, cli.stderr
    assert json.loads(cli.stdout)["bad_count"] == 0


@pytest.mark.parametrize(
    ("delivery_path", "mutation", "expected_field"),
    (
        (PATH_A, "selected_position", "selected_position"),
        (PATH_A, "source_position", "source_position"),
        (PATH_A, "source_row", "selected_example_id"),
        (PATH_A, "source_shard_id", "selected_example_id"),
        (PATH_A, "source_score", "source_score"),
        (PATH_A, "teacher_entropy", "teacher_entropy"),
        (PATH_A, "source_top_token_id", "source_top_token_id"),
        (PATH_A, "score_top_token_id", "score_top_token_id"),
        (PATH_A, "payload_top_token", "source_top_token_id"),
        (PATH_A, "candidate_rank", "payload_ref.candidate_rank"),
        (PATH_A, "payload_ref_source_position", "payload_ref"),
        (PATH_A, "payload_ref_source_score", "payload_ref"),
        (PATH_A, "payload_ref_source_top_token", "payload_ref"),
        (PATH_A, "corridor_mode_id", "corridor_mode_id"),
        (PATH_A, "corridor_assignment_status", "corridor_assignment_status"),
        (PATH_A, "payload_order", "selected_example_id"),
        (PATH_A, "leaderboard_order", "selected_example_id"),
        (PATH_A, "example_id_suffix", "selected_example_id"),
        (PATH_B, "source_position", "source_position"),
        (PATH_B, "source_score", "source_score"),
        (PATH_B, "source_top_token_id", "source_top_token_id"),
        (PATH_B, "payload_ref_kind", "payload_ref"),
        (PATH_B, "payload_ref_source_position", "payload_ref"),
        (PATH_B, "payload_ref_source_score", "payload_ref"),
        (PATH_B, "payload_ref_source_top_token", "payload_ref"),
        (PATH_B, "payload_top_token", "top_token_ids[0]"),
    ),
)
def test_single_field_mutations_fail_with_structured_passport_diagnostic(
    tmp_path: Path,
    delivery_path: str,
    mutation: str,
    expected_field: str,
) -> None:
    artifact = _build_synthetic_artifact(
        tmp_path / delivery_path,
        delivery_path=delivery_path,
    )
    _apply_mutation(artifact, mutation)

    report = audit_selected_linkage(artifact, strict=True)

    assert report.status == "fail"
    assert report.bad_count >= 1
    matching = [
        error
        for error in report.errors
        if expected_field in error["mismatch_fields"]
        or any(
            field.startswith(f"{expected_field}.") for field in error["mismatch_fields"]
        )
    ]
    assert matching, report.to_dict()
    error = matching[0]
    assert error["failure_stage"] == "selected_linkage_audit"
    assert error["selected_example_id"] is not None
    assert error["rank"] is not None
    assert error["source_shard_id"] is not None
    assert error["source_row"] is not None
    assert error["source_position"] is not None


def test_path_a_compact_rank_is_not_token_position_and_corridor_token_can_diverge(
    tmp_path: Path,
) -> None:
    artifact = _build_synthetic_artifact(tmp_path / "path-a", delivery_path=PATH_A)
    records = _selected_records(artifact)
    payloads = _selected_payloads(artifact)
    record = next(item for item in records if item["source_position"] == 2)
    payload = payloads[records.index(record)]

    assert record["payload_ref"]["candidate_rank"] == 1
    assert record["source_position"] == 2
    assert record["score_top_token_id"] != record["source_top_token_id"]
    assert payload["top_token_ids"][0] == record["source_top_token_id"]
    assert audit_selected_linkage(artifact, strict=True).status == "pass"


def test_path_b_rerun_mapping_preserves_duplicate_records_for_one_example(
    tmp_path: Path,
    monkeypatch,
) -> None:
    emitted_example_ids: list[tuple[str, ...]] = []
    example_a = "corpus_000999"
    example_b = "corpus_000003"
    example_c = "corpus_000020"
    records = [
        _path_b_direct_record(
            example_c,
            rank=1,
            source_row=2,
            position=4,
            score=7.5,
            token=18,
        ),
        _path_b_direct_record(
            example_a,
            rank=2,
            source_row=0,
            position=6,
            score=7.5,
            token=26,
        ),
        _path_b_direct_record(
            example_c,
            rank=3,
            source_row=2,
            position=9,
            score=7.5,
            token=29,
        ),
        _path_b_direct_record(
            example_b,
            rank=4,
            source_row=1,
            position=1,
            score=7.5,
            token=31,
        ),
    ]

    class DuplicateRecordBackend:
        def emit_batch(self, batch):
            emitted_example_ids.append(tuple(batch.example_ids))
            payload = _empty_rerun_payload(len(batch.example_ids))
            row_by_example_id = {
                example_id: row for row, example_id in enumerate(batch.example_ids)
            }
            expected = {
                example_a: ((6, 26),),
                example_b: ((1, 31),),
                example_c: ((4, 18), (9, 29)),
            }
            for example_id, coordinates in expected.items():
                row = row_by_example_id[example_id]
                for position, token in coordinates:
                    payload["teacher_entropy"][row, position] = 7.5
                    payload["top_token_ids"][row, position, 0] = token
            return SimpleNamespace(payload=payload)

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        exemplar_delivery,
        "create_backend",
        lambda _config: DuplicateRecordBackend(),
    )
    config = ExemplarDeliveryConfig(
        artifact_dir=tmp_path,
        dataset_path=tmp_path / "unused.jsonl",
        delivery_path=PATH_B,
        selection_enabled=True,
        sequence_length=SEQUENCE_LENGTH,
        vocab_size=37,
        top_k=TOP_K,
        num_buckets=4,
        backend_config=TeacherBackendConfig(
            backend_id="cpu_reference",
            runtime_mode="cpu",
            target_policy="corridor_exemplar_v1",
            sequence_length=SEQUENCE_LENGTH,
            vocab_size=37,
            top_k=TOP_K,
            num_buckets=4,
        ),
        selected_rerun_batch_size=3,
    )

    payloads = exemplar_delivery._selected_payloads_from_backend(
        records,
        store=SimpleNamespace(),
        examples=(
            TinyTextExample(example_id=example_c, text="example c"),
            TinyTextExample(example_id=example_a, text="example a"),
            TinyTextExample(example_id=example_b, text="example b"),
        ),
        config=config,
    )

    assert emitted_example_ids == [(example_a, example_b, example_c)]
    assert len(payloads) == 4
    assert [payload["selected_position"] for payload in payloads] == [4, 6, 9, 1]
    assert [payload["top_token_ids"][0] for payload in payloads] == [18, 26, 29, 31]


def test_deterministic_builder_artifacts_pass_strict_audit_for_both_paths(
    tmp_path: Path,
) -> None:
    path_a = _production_config(
        tmp_path / "a",
        delivery_path=PATH_A,
    )
    path_b = _production_config(
        tmp_path / "b",
        delivery_path=PATH_B,
    )

    assert build_production_gpu_tome(path_a)["status"] == "pass"
    assert build_production_gpu_tome(path_b)["status"] == "pass"

    path_a_audit = audit_selected_linkage(path_a.output_dir, strict=True)
    path_b_audit = audit_selected_linkage(path_b.output_dir, strict=True)

    assert path_a_audit.status == "pass", path_a_audit.to_dict()
    assert path_b_audit.status == "pass", path_b_audit.to_dict()
    assert path_a_audit.checked_count == 4
    assert path_b_audit.checked_count == 4


def test_selected_linkage_audit_accepts_one_quantization_step_entropy_deltas(
    tmp_path: Path,
) -> None:
    artifact = _build_synthetic_artifact(tmp_path / "quantized", delivery_path=PATH_A)
    payload_paths = sorted(
        (artifact / "selected_exemplars").glob("selected-exemplars-*.json")
    )
    remaining = 8
    for path in payload_paths:
        document = read_json_object(path)
        for payload in document["selected_exemplars"]:
            if remaining <= 0:
                break
            payload["teacher_entropy"] += 0.00390625
            remaining -= 1
        write_json(path, document)

    report = audit_selected_linkage(artifact, strict=True)

    assert report.status == "pass", report.to_dict()
    assert report.entropy_absolute_delta == 0.00390625
    assert report.entropy_allowed_tolerance == 0.00390625
    assert report.entropy_parity_status == "pass"


@pytest.mark.parametrize("delta", (0.00390625 * 2, float("nan")))
def test_selected_linkage_audit_rejects_large_or_nonfinite_entropy_deltas(
    tmp_path: Path,
    delta: float,
) -> None:
    artifact = _build_synthetic_artifact(tmp_path / str(delta), delivery_path=PATH_A)
    path = next((artifact / "selected_exemplars").glob("selected-exemplars-*.json"))
    document = read_json_object(path)
    document["selected_exemplars"][0]["teacher_entropy"] += delta
    write_json(path, document)

    report = audit_selected_linkage(artifact, strict=True)

    assert report.status == "fail"
    assert any("teacher_entropy" in error["mismatch_fields"] for error in report.errors)
    assert report.entropy_parity_status == "fail"


def test_score_pass_diagnostic_lookup_uses_exact_coordinate_for_duplicate_example(
    tmp_path: Path,
) -> None:
    from radjax_tome.builder.exemplar_delivery import _resolve_score_pass_evidence_row

    shard = {
        "score_example_ids": np.asarray([b"same", b"same"]),
        "score_selected_position": np.asarray([2, 7], dtype=np.int32),
    }

    assert (
        _resolve_score_pass_evidence_row(
            shard,
            {"selected_example_id": "same", "source_row": 1, "source_position": 7},
        )
        == 1
    )


def test_selected_linkage_audit_streams_256_payload_shards(
    tmp_path: Path,
) -> None:
    artifact = _build_many_shard_audit_artifact(tmp_path / "256-shards", count=256)
    tracemalloc.start()
    report = audit_selected_linkage(artifact, strict=True)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert report.status == "pass", report.to_dict()
    assert report.selected_count == 256
    assert report.checked_count == 256
    # Payload records are streamed one shard at a time; the test payloads are
    # deliberately large enough to catch retaining every record dictionary.
    assert peak < 32 * 1024 * 1024


def _build_synthetic_artifact(root: Path, *, delivery_path: str) -> Path:
    (root / "shards").mkdir(parents=True)
    (root / "leaderboards").mkdir(parents=True)
    (root / "selected_exemplars").mkdir(parents=True)
    selected: list[dict[str, object]] = []
    payloads: list[dict[str, object]] = []
    example_offset = 0
    for shard_id, row_count in enumerate(SHARD_SIZES):
        arrays = _synthetic_shard_arrays(
            delivery_path=delivery_path,
            shard_id=shard_id,
            row_count=row_count,
            global_offset=example_offset,
        )
        np.savez(root / "shards" / f"shard-{shard_id:05d}.npz", **arrays)
        example_offset += row_count
    for rank, (shard_id, row, candidate_rank) in enumerate(
        SELECTED_COORDINATES,
        start=1,
    ):
        global_row = sum(SHARD_SIZES[:shard_id]) + row
        example_id = EXAMPLE_IDS[global_row]
        shard = _load_npz(root / "shards" / f"shard-{shard_id:05d}.npz")
        if delivery_path == PATH_A:
            position = int(shard["exemplar_positions"][row, candidate_rank])
            score = float(shard["corridor_entropy"][row, position])
            source_top_token_id = int(
                shard["exemplar_source_top_token_ids"][row, candidate_rank, 0]
            )
            score_top_token_id = int(shard["corridor_top_token_ids"][row, position])
            payload_ref = {
                "kind": "one_pass_candidate_v1",
                "source_shard_id": shard_id,
                "source_row": row,
                "source_position": position,
                "candidate_rank": candidate_rank,
                "position_index": candidate_rank,
                "source_score": score,
                "source_top_token_id": source_top_token_id,
            }
        else:
            position = int(shard["score_selected_position"][row])
            score = float(shard["score_selected_position_entropy"][row])
            source_top_token_id = int(shard["score_top_token_id"][row])
            score_top_token_id = source_top_token_id
            payload_ref = {
                "kind": "corridor_exemplar_score_pass_v1",
                "source_shard_id": shard_id,
                "source_row": row,
                "source_position": position,
                "source_score": score,
                "source_top_token_id": source_top_token_id,
            }
        mode_id = _mode_id(global_row, position)
        record = {
            "rank": rank,
            "selected_example_id": example_id,
            "selected_position": position,
            "selected_score": score,
            "score_selected_position_entropy": score,
            "score_top_token_id": score_top_token_id,
            "source_shard_id": shard_id,
            "source_row": row,
            "source_position": position,
            "source_score": score,
            "source_top_token_id": source_top_token_id,
            "source_score_policy": "entropy_top_n_v1",
            "selected_policy": "entropy_top_n_v1",
            "source_delivery_path": delivery_path,
            "payload_ref": payload_ref,
            "corridor_mode_id": mode_id,
            "corridor_fingerprint_id": f"fp-{mode_id}",
            "corridor_assignment_status": "linked",
        }
        payload = {
            key: copy.deepcopy(value) for key, value in record.items() if key != "rank"
        }
        payload.update(
            {
                "top_token_ids": [source_top_token_id, 2, 3, 4, 5],
                "top_log_probs": [-0.1, -1.0, -2.0, -3.0, -4.0],
                "top_probs": [0.8, 0.1, 0.05, 0.03, 0.02],
                "top_selection_mask": [True] * TOP_K,
                "effective_top_k": TOP_K,
                "top_mass": 1.0,
                "tail_mass": 0.0,
                "bucket_masses": [0.0, 0.0, 0.0, 0.0],
                "teacher_entropy": score,
                "sequence_length": SEQUENCE_LENGTH,
                "vocab_size": 37,
                "num_buckets": 4,
                "dynamic_top_k": {"policy": "mass_threshold_v1"},
            }
        )
        selected.append(record)
        payloads.append(payload)
    write_json(
        root / "leaderboards" / "selected_exemplars.json",
        {
            "schema_version": "selected_exemplars_v1",
            "delivery_path": delivery_path,
            "selected_exemplars": selected,
        },
    )
    for shard_index in range(3):
        start = shard_index * 3
        write_json(
            root / "selected_exemplars" / f"selected-exemplars-{shard_index:05d}.json",
            {
                "schema_version": "selected_exemplar_payload_shard_v1",
                "delivery_path": delivery_path,
                "selected_exemplars": payloads[start : start + 3],
            },
        )
    write_json(
        root / "delivery_report.json",
        {"status": "pass", "delivery_path": delivery_path},
    )
    _write_mode_assignments(root)
    return root


def _build_many_shard_audit_artifact(root: Path, *, count: int) -> Path:
    (root / "shards").mkdir(parents=True)
    (root / "leaderboards").mkdir()
    (root / "selected_exemplars").mkdir()
    records: list[dict[str, object]] = []
    metadata: list[dict[str, object]] = []
    for index in range(count):
        example_id = f"stream_{index:06d}"
        score = float(2.0 + (index % 13) / 10.0)
        token = 10_000 + index
        entropy = np.full((1, SEQUENCE_LENGTH), score, dtype=np.float32)
        top_tokens = np.full((1, SEQUENCE_LENGTH), token, dtype=np.int32)
        source_tokens = np.zeros((1, SEQUENCE_LENGTH, TOP_K), dtype=np.int32)
        source_tokens[:, :, 0] = token
        np.savez(
            root / "shards" / f"shard-{index:05d}.npz",
            input_ids=np.zeros((1, SEQUENCE_LENGTH), dtype=np.int32),
            corridor_entropy=entropy,
            corridor_teacher_entropy=entropy,
            corridor_top_token_ids=top_tokens,
            exemplar_positions=np.zeros((1, TOP_K), dtype=np.int32),
            exemplar_source_top_token_ids=source_tokens,
        )
        payload_ref = {
            "kind": "one_pass_candidate_v1",
            "source_shard_id": index,
            "source_row": 0,
            "source_position": 0,
            "candidate_rank": 0,
            "source_score": score,
            "source_top_token_id": token,
        }
        record = {
            "rank": index + 1,
            "selected_example_id": example_id,
            "selected_position": 0,
            "selected_score": score,
            "score_selected_position_entropy": score,
            "score_top_token_id": token,
            "source_shard_id": index,
            "source_row": 0,
            "source_position": 0,
            "source_score": score,
            "source_top_token_id": token,
            "source_score_policy": "entropy_top_n_v1",
            "selected_policy": "entropy_top_n_v1",
            "source_delivery_path": PATH_A,
            "payload_ref": payload_ref,
            "corridor_mode_id": 0,
            "corridor_fingerprint_id": "fp-0",
            "corridor_assignment_status": "linked",
            "selected_board": "primary",
        }
        payload = copy.deepcopy(record)
        payload.update(
            {
                "top_token_ids": [token, *range(1, 1024)],
                "top_log_probs": [-0.1] * 1024,
                "top_probs": [1.0 / 1024] * 1024,
                "top_selection_mask": [True] * 1024,
                "effective_top_k": 1024,
                "top_mass": 1.0,
                "tail_mass": 0.0,
                "bucket_masses": [0.25] * 4,
                "teacher_entropy": score,
                "sequence_length": SEQUENCE_LENGTH,
                "vocab_size": 1024,
                "num_buckets": 4,
                "dynamic_top_k": {"policy": "mass_threshold_v1"},
            }
        )
        records.append(record)
        metadata.append({"example_index": index, "example_id": example_id})
        write_json(
            root / "selected_exemplars" / f"selected-exemplars-{index:05d}.json",
            {
                "schema_version": "selected_exemplar_payload_shard_v1",
                "delivery_path": PATH_A,
                "selected_exemplars": [payload],
            },
        )
    write_json(
        root / "leaderboards" / "selected_exemplars.json",
        {
            "schema_version": "selected_exemplars_v1",
            "delivery_path": PATH_A,
            "selected_exemplars": records,
        },
    )
    write_json(
        root / "delivery_report.json",
        {"status": "pass", "delivery_path": PATH_A},
    )
    assignment_dir = root / "corridors" / "mode_assignments"
    assignment_dir.mkdir(parents=True)
    arrays = {
        "position_example_index": np.arange(count, dtype=np.int32),
        "position": np.zeros(count, dtype=np.int32),
        "mode_id": np.zeros(count, dtype=np.int32),
        "weight": np.ones(count, dtype=np.float32),
    }
    for name, array in arrays.items():
        np.save(assignment_dir / f"{name}.npy", array)
    metadata_path = assignment_dir / "examples_metadata.jsonl"
    metadata_path.write_text(
        "".join(json.dumps(row) + "\n" for row in metadata),
        encoding="utf-8",
    )
    write_json(
        root / "corridors" / "mode_assignments.json",
        {
            "schema_version": "corridor_mode_assignments_v3",
            "storage_kind": "packed_numpy_v1",
            "num_assignments": count,
            "num_examples": count,
            "arrays": {
                name: {
                    "path": f"corridors/mode_assignments/{name}.npy",
                    "dtype": str(array.dtype),
                    "shape": list(array.shape),
                }
                for name, array in arrays.items()
            },
            "examples_metadata": {
                "path": "corridors/mode_assignments/examples_metadata.jsonl",
                "num_examples": count,
            },
        },
    )
    return root


def _production_config(root: Path, *, delivery_path: str) -> ProductionBuildConfig:
    root.mkdir(parents=True)
    source_paths: list[Path] = []
    for index, example_id in enumerate(EXAMPLE_IDS[:8]):
        source = root / f"source-{index}.txt"
        source.write_text(
            f"{example_id} adversarial deterministic source {index}",
            encoding="utf-8",
        )
        source_paths.append(source)
    corpus_dir = root / "corpus"
    build_corpus_artifact(
        CorpusBuildConfig(
            inputs=tuple(source_paths),
            output_dir=corpus_dir,
            overwrite=True,
        )
    )
    model = root / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"_name_or_path": "radjax/audit", "model_type": "tiny"}),
        encoding="utf-8",
    )
    (model / "tokenizer.json").write_text('{"version": "1.0"}', encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")
    provenance = root / "teacher_model_provenance.json"
    write_teacher_model_provenance(
        inspect_teacher_model(model, model_name="radjax/audit"),
        provenance,
    )
    return ProductionBuildConfig(
        teacher_model=str(model),
        tokenizer_id=str(model),
        dataset_path=corpus_dir / "corpus.jsonl",
        corpus_manifest_path=corpus_dir / "corpus_manifest.json",
        teacher_model_provenance_path=provenance,
        output_dir=root / "artifact",
        teacher_backend="cpu_reference",
        runtime_mode="cpu",
        target_policy="corridor_exemplar_v1",
        sequence_length=SEQUENCE_LENGTH,
        vocab_size=64,
        top_k=TOP_K,
        num_buckets=4,
        gpu_batch_size_mode="preset",
        gpu_batch_size_preset=2,
        shard_size_examples=3,
        max_examples=8,
        exemplar_selection_enabled=True,
        exemplar_delivery_path=delivery_path,
        selected_exemplar_budget=4,
        retain_unselected_exemplar_payloads=False,
    )


def _synthetic_shard_arrays(
    *,
    delivery_path: str,
    shard_id: int,
    row_count: int,
    global_offset: int,
) -> dict[str, np.ndarray]:
    input_ids = np.zeros((row_count, SEQUENCE_LENGTH), dtype=np.int32)
    corridor_entropy = np.zeros((row_count, SEQUENCE_LENGTH), dtype=np.float32)
    corridor_top_token_ids = np.zeros((row_count, SEQUENCE_LENGTH), dtype=np.int32)
    for row in range(row_count):
        global_row = global_offset + row
        input_ids[row] = 1000 + global_row
        for position in range(SEQUENCE_LENGTH):
            corridor_entropy[row, position] = np.float32(
                7.5 if position in {2, 6} else 1.0 + global_row + position / 100
            )
            corridor_top_token_ids[row, position] = 700 + global_row * 20 + position
    arrays: dict[str, np.ndarray] = {
        "input_ids": input_ids,
        "attention_mask": np.ones_like(input_ids, dtype=np.int32),
        "corridor_entropy": corridor_entropy,
        "corridor_teacher_entropy": corridor_entropy.copy(),
        "corridor_top_token_ids": corridor_top_token_ids,
    }
    if delivery_path == PATH_A:
        positions = np.tile(np.asarray([[8, 2, 10]], dtype=np.int32), (row_count, 1))
        source_top_ids = np.zeros((row_count, 3, TOP_K), dtype=np.int32)
        for row in range(row_count):
            global_row = global_offset + row
            source_top_ids[row, 0, 0] = 801 + global_row
            source_top_ids[row, 1, 0] = 222 + global_row
            source_top_ids[row, 2, 0] = 910 + global_row
        arrays.update(
            {
                "exemplar_positions": positions,
                "exemplar_source_top_token_ids": source_top_ids,
            }
        )
    else:
        selected_positions = np.full((row_count,), 6, dtype=np.int32)
        score_top_ids = np.asarray(
            [18 + global_offset + row for row in range(row_count)],
            dtype=np.int32,
        )
        for row in range(row_count):
            corridor_top_token_ids[row, 6] = score_top_ids[row]
        arrays.update(
            {
                "score_selected_position": selected_positions,
                "score_selected_position_entropy": np.full(
                    (row_count,),
                    7.5,
                    dtype=np.float32,
                ),
                "score_top_token_id": score_top_ids,
            }
        )
    return arrays


def _write_mode_assignments(root: Path) -> None:
    assignment_dir = root / "corridors" / "mode_assignments"
    assignment_dir.mkdir(parents=True)
    count = len(EXAMPLE_IDS) * SEQUENCE_LENGTH
    position_example_index = np.repeat(
        np.arange(len(EXAMPLE_IDS), dtype=np.int32),
        SEQUENCE_LENGTH,
    )
    positions = np.tile(
        np.arange(SEQUENCE_LENGTH, dtype=np.int32),
        len(EXAMPLE_IDS),
    )
    mode_ids = np.asarray(
        [
            _mode_id(example_index, position)
            for example_index in range(len(EXAMPLE_IDS))
            for position in range(SEQUENCE_LENGTH)
        ],
        dtype=np.int32,
    )
    arrays = {
        "position_example_index": position_example_index,
        "position": positions,
        "mode_id": mode_ids,
        "weight": np.ones((count,), dtype=np.float32),
    }
    for name, array in arrays.items():
        np.save(assignment_dir / f"{name}.npy", array)
    metadata_path = assignment_dir / "examples_metadata.jsonl"
    metadata_path.write_text(
        "".join(
            json.dumps({"example_index": index, "example_id": example_id}) + "\n"
            for index, example_id in enumerate(EXAMPLE_IDS)
        ),
        encoding="utf-8",
    )
    write_json(
        root / "corridors" / "mode_assignments.json",
        {
            "schema_version": "corridor_mode_assignments_v3",
            "storage_kind": "packed_numpy_v1",
            "num_assignments": count,
            "num_examples": len(EXAMPLE_IDS),
            "arrays": {
                name: {
                    "path": f"corridors/mode_assignments/{name}.npy",
                    "dtype": str(array.dtype),
                    "shape": list(array.shape),
                }
                for name, array in arrays.items()
            },
            "examples_metadata": {
                "path": "corridors/mode_assignments/examples_metadata.jsonl",
                "num_examples": len(EXAMPLE_IDS),
            },
        },
    )


def _apply_mutation(root: Path, mutation: str) -> None:
    records_doc = read_json_object(root / "leaderboards" / "selected_exemplars.json")
    payload_paths = sorted((root / "selected_exemplars").glob("*.json"))
    payload_docs = [read_json_object(path) for path in payload_paths]
    records = records_doc["selected_exemplars"]
    payloads = [
        item for document in payload_docs for item in document["selected_exemplars"]
    ]
    record = records[0]
    payload = payloads[0]
    if mutation == "selected_position":
        record["selected_position"] += 1
    elif mutation == "source_position":
        record["source_position"] += 1
        payload["source_position"] += 1
        payload["selected_position"] += 1
    elif mutation == "source_row":
        record["source_row"] = (record["source_row"] + 1) % SHARD_SIZES[
            record["source_shard_id"]
        ]
        payload["source_row"] = record["source_row"]
        record["payload_ref"]["source_row"] = record["source_row"]
        payload["payload_ref"]["source_row"] = record["source_row"]
    elif mutation == "source_shard_id":
        record["source_shard_id"] = (record["source_shard_id"] + 1) % len(SHARD_SIZES)
        record["source_row"] %= SHARD_SIZES[record["source_shard_id"]]
        payload["source_shard_id"] = record["source_shard_id"]
        payload["source_row"] = record["source_row"]
        for item in (record, payload):
            item["payload_ref"]["source_shard_id"] = record["source_shard_id"]
            item["payload_ref"]["source_row"] = record["source_row"]
    elif mutation == "source_score":
        record["source_score"] += 0.5
        record["selected_score"] += 0.5
        payload["source_score"] += 0.5
        payload["selected_score"] += 0.5
        record["payload_ref"]["source_score"] += 0.5
        payload["payload_ref"]["source_score"] += 0.5
    elif mutation == "teacher_entropy":
        payload["teacher_entropy"] += 0.5
    elif mutation == "source_top_token_id":
        record["source_top_token_id"] += 99
        payload["source_top_token_id"] = record["source_top_token_id"]
        record["payload_ref"]["source_top_token_id"] = record["source_top_token_id"]
        payload["payload_ref"]["source_top_token_id"] = record["source_top_token_id"]
    elif mutation == "score_top_token_id":
        record["score_top_token_id"] += 99
        payload["score_top_token_id"] = record["score_top_token_id"]
    elif mutation == "payload_top_token":
        payload["top_token_ids"][0] += 99
    elif mutation == "candidate_rank":
        record["payload_ref"]["candidate_rank"] = 0
        payload["payload_ref"]["candidate_rank"] = 0
    elif mutation == "payload_ref_source_position":
        record["payload_ref"]["source_position"] += 1
        payload["payload_ref"]["source_position"] += 1
    elif mutation == "payload_ref_source_score":
        record["payload_ref"]["source_score"] += 0.5
        payload["payload_ref"]["source_score"] += 0.5
    elif mutation == "payload_ref_source_top_token":
        record["payload_ref"]["source_top_token_id"] += 99
        payload["payload_ref"]["source_top_token_id"] += 99
    elif mutation == "payload_ref_kind":
        record["payload_ref"]["kind"] = "wrong"
        payload["payload_ref"]["kind"] = "wrong"
    elif mutation == "corridor_mode_id":
        record["corridor_mode_id"] += 1
        payload["corridor_mode_id"] += 1
    elif mutation == "corridor_assignment_status":
        record["corridor_assignment_status"] = "unlinked"
        payload["corridor_assignment_status"] = "unlinked"
    elif mutation == "payload_order":
        (
            payload_docs[0]["selected_exemplars"][0],
            payload_docs[1]["selected_exemplars"][0],
        ) = (
            payload_docs[1]["selected_exemplars"][0],
            payload_docs[0]["selected_exemplars"][0],
        )
    elif mutation == "leaderboard_order":
        records.reverse()
    elif mutation == "example_id_suffix":
        record["selected_example_id"] = "corpus_000002"
        payload["selected_example_id"] = "corpus_000002"
    else:
        raise AssertionError(f"unsupported mutation: {mutation}")
    write_json(root / "leaderboards" / "selected_exemplars.json", records_doc)
    for path, document in zip(payload_paths, payload_docs, strict=True):
        write_json(path, document)


def _selected_records(root: Path) -> list[dict[str, object]]:
    return read_json_object(root / "leaderboards" / "selected_exemplars.json")[
        "selected_exemplars"
    ]


def _selected_payloads(root: Path) -> list[dict[str, object]]:
    return [
        item
        for path in sorted((root / "selected_exemplars").glob("*.json"))
        for item in read_json_object(path)["selected_exemplars"]
    ]


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _mode_id(global_row: int, position: int) -> int:
    return (global_row * 3 + position) % 17


def _path_b_direct_record(
    example_id: str,
    *,
    rank: int,
    source_row: int,
    position: int,
    score: float,
    token: int,
) -> dict[str, object]:
    payload_ref = {
        "kind": "corridor_exemplar_score_pass_v1",
        "source_shard_id": 0,
        "source_row": source_row,
        "source_position": position,
        "source_score": score,
        "source_top_token_id": token,
    }
    return {
        "rank": rank,
        "selected_example_id": example_id,
        "selected_position": position,
        "selected_score": score,
        "score_selected_position_entropy": score,
        "score_top_token_id": token,
        "source_shard_id": 0,
        "source_row": source_row,
        "source_position": position,
        "source_score": score,
        "source_top_token_id": token,
        "source_score_policy": "entropy_top_n_v1",
        "payload_ref": payload_ref,
        "selected_policy": "entropy_top_n_v1",
        "source_delivery_path": PATH_B,
    }


def _empty_rerun_payload(batch_size: int) -> dict[str, np.ndarray]:
    shape = (batch_size, SEQUENCE_LENGTH, TOP_K)
    return {
        "top_token_ids": np.zeros(shape, dtype=np.int32),
        "top_log_probs": np.zeros(shape, dtype=np.float32),
        "top_probs": np.zeros(shape, dtype=np.float32),
        "top_selection_mask": np.ones(shape, dtype=bool),
        "effective_top_k": np.full(
            (batch_size, SEQUENCE_LENGTH),
            TOP_K,
            dtype=np.int32,
        ),
        "top_mass": np.ones((batch_size, SEQUENCE_LENGTH), dtype=np.float32),
        "tail_mass": np.zeros((batch_size, SEQUENCE_LENGTH), dtype=np.float32),
        "bucket_masses": np.zeros(
            (batch_size, SEQUENCE_LENGTH, 4),
            dtype=np.float32,
        ),
        "teacher_entropy": np.zeros(
            (batch_size, SEQUENCE_LENGTH),
            dtype=np.float32,
        ),
    }
