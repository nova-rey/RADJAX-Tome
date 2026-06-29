from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from radjax_tome.parity import compare_teacher_textbook_artifacts


def test_identical_sidecars_pass(tmp_path: Path) -> None:
    old = _artifact(tmp_path / "old")
    new = _copy_artifact(old, tmp_path / "new")

    result = compare_teacher_textbook_artifacts(old, new)

    assert result.status == "pass"
    assert result.blockers == ()


def test_allowed_timestamp_and_provenance_differences_pass(tmp_path: Path) -> None:
    old = _artifact(tmp_path / "old")
    new = _copy_artifact(old, tmp_path / "new")
    _mutate_json(
        new / "metadata.json",
        {
            "created_at": "2026-06-29T12:00:01+00:00",
            "created_by": "radjax_tome.builder.teacher_textbook",
            "provenance": {"phase": "radjax-tome-migration", "teacher_mode": "fake"},
        },
    )
    _mutate_json(
        new / "teacher_manifest.json",
        {"created_at": "2026-06-29T12:00:01+00:00"},
    )

    result = compare_teacher_textbook_artifacts(old, new)

    assert result.status == "pass"
    assert len(result.allowed_differences) == 4


def test_target_type_mismatch_fails(tmp_path: Path) -> None:
    old = _artifact(tmp_path / "old")
    new = _copy_artifact(old, tmp_path / "new")
    _mutate_json(new / "metadata.json", {"target_type": "topk_with_tail_v0"})

    result = compare_teacher_textbook_artifacts(old, new)

    assert result.status == "fail"
    assert any("target_type" in blocker for blocker in result.blockers)


def test_missing_shard_fails(tmp_path: Path) -> None:
    old = _artifact(tmp_path / "old")
    new = _copy_artifact(old, tmp_path / "new")
    (new / "shards" / "shard-00000.npz").unlink()

    result = compare_teacher_textbook_artifacts(old, new)

    assert result.status == "fail"
    assert any("missing new shard" in blocker for blocker in result.blockers)


def test_array_shape_mismatch_fails(tmp_path: Path) -> None:
    old = _artifact(tmp_path / "old")
    new = _copy_artifact(old, tmp_path / "new")
    _write_shard(new, logits=np.zeros((1, 3, 4), dtype=np.float32))

    result = compare_teacher_textbook_artifacts(old, new)

    assert result.status == "fail"
    assert any("shape mismatch" in blocker for blocker in result.blockers)


def test_array_dtype_mismatch_fails(tmp_path: Path) -> None:
    old = _artifact(tmp_path / "old")
    new = _copy_artifact(old, tmp_path / "new")
    _write_shard(new, logits=np.zeros((1, 2, 4), dtype=np.float16))

    result = compare_teacher_textbook_artifacts(old, new)

    assert result.status == "fail"
    assert any("dtype mismatch" in blocker for blocker in result.blockers)


def test_array_value_mismatch_fails(tmp_path: Path) -> None:
    old = _artifact(tmp_path / "old")
    new = _copy_artifact(old, tmp_path / "new")
    logits = np.zeros((1, 2, 4), dtype=np.float32)
    logits[0, 0, 0] = 1.0
    _write_shard(new, logits=logits)

    result = compare_teacher_textbook_artifacts(old, new)

    assert result.status == "fail"
    assert any("value mismatch" in blocker for blocker in result.blockers)


def test_topk_metadata_mismatch_fails(tmp_path: Path) -> None:
    old = _artifact(tmp_path / "old", target_type="topk_with_tail_v0", top_k=4)
    new = _copy_artifact(old, tmp_path / "new")
    _mutate_json(new / "metadata.json", {"target_params": {"top_k": "1"}})

    result = compare_teacher_textbook_artifacts(old, new)

    assert result.status == "fail"
    assert any("/target_params/top_k" in blocker for blocker in result.blockers)


def test_cascaded_bucket_metadata_mismatch_fails(tmp_path: Path) -> None:
    old = _artifact(
        tmp_path / "old",
        target_type="cascaded_soft_labels_v1",
        top_k=4,
    )
    new = _copy_artifact(old, tmp_path / "new")
    _mutate_json(
        new / "metadata.json",
        {"target_params": {"bucket_edges": "1,0.5,0"}},
    )

    result = compare_teacher_textbook_artifacts(old, new)

    assert result.status == "fail"
    assert any("/target_params/bucket_edges" in blocker for blocker in result.blockers)


def test_unknown_sidecar_difference_fails(tmp_path: Path) -> None:
    old = _artifact(tmp_path / "old")
    new = _copy_artifact(old, tmp_path / "new")
    _mutate_json(new / "emission_config.json", {"sequence_length": 3})

    result = compare_teacher_textbook_artifacts(old, new)

    assert result.status == "fail"
    assert any(
        "emission_config.json/sequence_length" in blocker
        for blocker in result.blockers
    )


def _artifact(
    root: Path,
    *,
    target_type: str = "dense_logits",
    top_k: int = 4,
) -> Path:
    root.mkdir(parents=True)
    (root / "shards").mkdir()
    target_params: dict[str, str] = {}
    if target_type in {"topk_with_tail_v0", "cascaded_soft_labels_v1"}:
        target_params.update({"top_k": str(top_k), "top_log_probs_dtype": "float16"})
    if target_type == "cascaded_soft_labels_v1":
        target_params.update(
            {
                "bucket_edges": "1,0.001,1e-06,1e-09,1e-12,0",
                "bucket_edge_type": "probability",
                "bucket_count": "5",
                "bucket_mass_dtype": "float32",
                "bucket_mean_logp_dtype": "float32",
            }
        )
    metadata = {
        "schema_version": "qrwkv_xla.teacher_target_store.v1",
        "target_store_version": "p93",
        "model_id": "fake-deterministic-teacher",
        "model_family": "fake",
        "tokenizer_id": "fake-deterministic-tokenizer",
        "tokenizer_hash": None,
        "vocab_size": 4,
        "target_type": target_type,
        "dtype": "float32",
        "sequence_length": 2,
        "num_examples": 1,
        "shard_count": 1,
        "created_by": "qrwkv_xla.artifacts.teacher_textbook_builder",
        "created_at": "2026-06-29T12:00:00+00:00",
        "source": {"kind": "builtin_examples"},
        "provenance": {"phase": "P119", "teacher_mode": "fake"},
        "target_params": target_params,
    }
    _write_json(root / "metadata.json", metadata)
    _write_json(
        root / "vocab_contract.json",
        {
            "tokenizer_id": "fake-deterministic-tokenizer",
            "tokenizer_hash": None,
            "vocab_size": 4,
            "model_id": "fake-deterministic-teacher",
            "model_family": "fake",
        },
    )
    manifest = {
        "artifact_type": "teacher_textbook",
        "artifact_version": 0,
        "teacher_model_id": "fake-deterministic-teacher",
        "teacher_backend_type": "fake",
        "teacher_revision_or_hash": None,
        "tokenizer_id": "fake-deterministic-tokenizer",
        "vocab_size": 4,
        "vocab_contract_path": "vocab_contract.json",
        "target_type": target_type,
        "dtype": "float32",
        "sequence_length": 2,
        "num_examples": 1,
        "shard_count": 1,
        "created_at": "2026-06-29T12:00:00+00:00",
        "local_files_only": True,
        "allow_downloads": False,
        "claims_not_made": ["no_model_quality_claim"],
    }
    if target_type in {"topk_with_tail_v0", "cascaded_soft_labels_v1"}:
        manifest["top_k"] = top_k
        manifest["top_log_probs_dtype"] = "float16"
    if target_type == "cascaded_soft_labels_v1":
        manifest["bucket_edges"] = [1.0, 1e-3, 1e-6, 1e-9, 1e-12, 0.0]
        manifest["bucket_count"] = 5
        manifest["bucket_mass_dtype"] = "float32"
        manifest["bucket_mean_logp_dtype"] = "float32"
    _write_json(root / "teacher_manifest.json", manifest)
    emission = {
        "dataset_source": "builtin_examples",
        "max_examples": 1,
        "batch_size": 4,
        "sequence_length": 2,
        "logits_dtype": "float32",
        "target_type": target_type,
        "include_hidden_states": False,
        "sampling_used": False,
        "temperature": None,
        "top_p": None,
        "top_k": None if target_type == "dense_logits" else top_k,
        "seed": 0,
        "teacher_mode": "fake",
    }
    _write_json(root / "emission_config.json", emission)
    _write_json(
        root / "validation_report.json",
        {
            "artifact_type": "teacher_textbook",
            "artifact_version": 0,
            "status": "pass",
            "checks": [],
            "blockers": [],
            "warnings": [],
            "target_type": target_type,
        },
    )
    _write_shard(root)
    return root


def _copy_artifact(source: Path, target: Path) -> Path:
    shutil.copytree(source, target)
    return target


def _write_shard(root: Path, *, logits: np.ndarray | None = None) -> None:
    if logits is None:
        logits = np.zeros((1, 2, 4), dtype=np.float32)
    np.savez(
        root / "shards" / "shard-00000.npz",
        input_ids=np.array([[1, 2]], dtype=np.int32),
        attention_mask=np.array([[1, 1]], dtype=np.int32),
        logits=logits,
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _mutate_json(path: Path, updates: dict[str, object]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _merge(payload, updates)
    _write_json(path, payload)


def _merge(payload: dict[str, object], updates: dict[str, object]) -> None:
    for key, value in updates.items():
        if (
            isinstance(value, dict)
            and isinstance(payload.get(key), dict)
        ):
            _merge(payload[key], value)  # type: ignore[arg-type]
        else:
            payload[key] = value
