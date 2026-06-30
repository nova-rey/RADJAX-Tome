from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from radjax_tome.targets.schema import (
    TEACHER_TARGET_STORE_SCHEMA_VERSION,
    TEACHER_TARGET_STORE_VERSION,
    TargetStoreMetadata,
)
from radjax_tome.targets.store import TeacherTargetStore


def dense_logits_to_topk_tail(
    logits: np.ndarray,
    *,
    top_k: int,
) -> dict[str, np.ndarray]:
    values = np.asarray(logits, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("logits must have shape [N,T,V]")
    if top_k <= 0 or top_k > values.shape[-1]:
        raise ValueError("top_k must be in [1, vocab_size]")
    probs = _softmax(values)
    log_probs = np.log(np.maximum(probs, np.finfo(np.float32).tiny))
    top_ids = np.argsort(log_probs, axis=-1)[..., -top_k:][..., ::-1]
    top_log_probs = np.take_along_axis(log_probs, top_ids, axis=-1)
    top_probs = np.take_along_axis(probs, top_ids, axis=-1)
    top_mass = np.sum(top_probs, axis=-1).astype(np.float32)
    tail_mass = np.maximum(1.0 - top_mass, 0.0).astype(np.float32)
    entropy = -np.sum(probs * log_probs, axis=-1).astype(np.float32)
    return {
        "top_token_ids": top_ids.astype(np.int32),
        "top_log_probs": top_log_probs.astype(np.float32),
        "top_mass": top_mass,
        "tail_mass": tail_mass,
        "teacher_entropy": entropy,
    }


def dense_logits_to_cascaded_soft_labels(
    logits: np.ndarray,
    *,
    top_k: int,
    bucket_edges: tuple[float, ...] = (1.0, 0.1, 0.0),
) -> dict[str, np.ndarray]:
    values = np.asarray(logits, dtype=np.float32)
    compressed = dense_logits_to_topk_tail(values, top_k=top_k)
    probs = _softmax(values)
    log_probs = np.log(np.maximum(probs, np.finfo(np.float32).tiny))
    top_ids = compressed["top_token_ids"]
    top_mask = np.zeros(probs.shape, dtype=bool)
    np.put_along_axis(top_mask, top_ids, True, axis=-1)
    tail_probs = np.where(top_mask, 0.0, probs)
    tail_log_probs = np.where(top_mask, 0.0, log_probs)
    edges = np.asarray(bucket_edges, dtype=np.float32)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("bucket_edges must contain at least two values")
    if not np.isclose(edges[0], 1.0) or not np.isclose(edges[-1], 0.0):
        raise ValueError("bucket_edges must start at 1.0 and end at 0.0")
    if not np.all(np.diff(edges) < 0):
        raise ValueError("bucket_edges must be strictly descending")
    bucket_count = edges.size - 1
    out_shape = (*probs.shape[:2], bucket_count)
    bucket_mass = np.zeros(out_shape, dtype=np.float32)
    bucket_item_count = np.zeros(out_shape, dtype=np.int32)
    bucket_mean_logp = np.zeros(out_shape, dtype=np.float32)
    for bucket_id in range(bucket_count):
        high = edges[bucket_id]
        low = edges[bucket_id + 1]
        if bucket_id == bucket_count - 1:
            mask = (~top_mask) & (tail_probs <= high) & (tail_probs >= low)
        else:
            mask = (~top_mask) & (tail_probs <= high) & (tail_probs > low)
        count = np.sum(mask, axis=-1)
        mass = np.sum(np.where(mask, tail_probs, 0.0), axis=-1)
        logp_sum = np.sum(np.where(mask, tail_log_probs, 0.0), axis=-1)
        bucket_item_count[..., bucket_id] = count.astype(np.int32)
        bucket_mass[..., bucket_id] = mass.astype(np.float32)
        bucket_mean_logp[..., bucket_id] = np.divide(
            logp_sum,
            count,
            out=np.zeros_like(mass, dtype=np.float32),
            where=count > 0,
        ).astype(np.float32)
    compressed.update(
        {
            "bucket_mass": bucket_mass,
            "bucket_count": bucket_item_count,
            "bucket_mean_logp": bucket_mean_logp,
        }
    )
    return compressed


def write_compressed_target_store(
    source_store: TeacherTargetStore | str | Path,
    output_dir: str | Path,
    *,
    target_type: str,
    top_k: int = 2,
    bucket_edges: tuple[float, ...] = (1.0, 0.1, 0.0),
    overwrite: bool = False,
) -> TeacherTargetStore:
    source = (
        source_store
        if isinstance(source_store, TeacherTargetStore)
        else TeacherTargetStore.open(source_store)
    )
    source.validate()
    output = Path(output_dir)
    if output.exists():
        if not overwrite:
            raise ValueError(f"target store already exists: {output}")
        shutil.rmtree(output)
    metadata = _compressed_metadata(
        source,
        target_type=target_type,
        top_k=top_k,
        bucket_edges=bucket_edges,
    )
    store = TeacherTargetStore.create(output, metadata, overwrite=overwrite)
    arrays = source.read_shard(0)
    logits = np.asarray(arrays["logits"], dtype=np.float32)
    compressed = dense_logits_to_topk_tail(logits, top_k=top_k)
    if target_type == "cascaded_soft_labels_v1":
        compressed = dense_logits_to_cascaded_soft_labels(
            logits,
            top_k=top_k,
            bucket_edges=bucket_edges,
        )
    elif target_type != "topk_with_tail_v0":
        raise ValueError(
            "target_type must be topk_with_tail_v0 or cascaded_soft_labels_v1"
        )
    store.write_shard(
        0,
        {
            "input_ids": np.asarray(arrays["input_ids"], dtype=np.int32),
            "attention_mask": np.asarray(arrays["attention_mask"], dtype=np.int32),
            **compressed,
        },
    )
    store.validate()
    return TeacherTargetStore.open(output)


def _compressed_metadata(
    source: TeacherTargetStore,
    *,
    target_type: str,
    top_k: int,
    bucket_edges: tuple[float, ...],
) -> TargetStoreMetadata:
    params = {
        "top_k": str(top_k),
        "top_log_probs_dtype": "float32",
    }
    if target_type == "cascaded_soft_labels_v1":
        params.update(
            {
                "bucket_edge_type": "probability",
                "bucket_edges": ",".join(str(edge) for edge in bucket_edges),
                "bucket_count": str(len(bucket_edges) - 1),
                "bucket_mass_dtype": "float32",
                "bucket_mean_logp_dtype": "float32",
            }
        )
    return TargetStoreMetadata(
        schema_version=TEACHER_TARGET_STORE_SCHEMA_VERSION,
        target_store_version=TEACHER_TARGET_STORE_VERSION,
        model_id=source.metadata.model_id,
        model_family=source.metadata.model_family,
        tokenizer_id=source.metadata.tokenizer_id,
        tokenizer_hash=source.metadata.tokenizer_hash,
        vocab_size=source.metadata.vocab_size,
        target_type=target_type,
        dtype="float32",
        sequence_length=source.metadata.sequence_length,
        num_examples=source.metadata.num_examples,
        shard_count=1,
        created_by="radjax_tome.targets.compression",
        created_at=source.metadata.created_at,
        source={"kind": "compressed_from_dense", "path": str(source.root)},
        provenance={"compression": target_type},
        target_params=params,
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted, dtype=np.float32)
    return (exp / np.sum(exp, axis=-1, keepdims=True)).astype(np.float32)
