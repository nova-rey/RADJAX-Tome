from __future__ import annotations

import numpy as np

from radjax_tome.backends.base import (
    BackendCapability,
    TargetPolicy,
    TeacherBackendConfig,
    TeacherBatchInput,
    TeacherEmissionResult,
)

_SUPPORTED_EMISSION_POLICIES = {
    "dense_logits",
    "topk_with_tail_v0",
    "cascaded_soft_labels_v1",
    "corridor_exemplar_v1",
}
_CAPABILITY_STATUS: dict[TargetPolicy, str] = {
    "dense_logits": "supported_debug",
    "topk_with_tail_v0": "supported",
    "cascaded_soft_labels_v1": "supported",
    "corridor_exemplar_v1": "supported",
}


class CPUReferenceTeacherEmissionBackend:
    backend_id = "cpu_reference"
    backend_family = "cpu_reference"
    runtime_mode = "cpu"

    def __init__(self, config: TeacherBackendConfig) -> None:
        if config.backend_id != self.backend_id:
            raise ValueError(
                f"cpu_reference requires backend_id={self.backend_id!r}, "
                f"got {config.backend_id!r}"
            )
        if config.runtime_mode != "cpu":
            raise ValueError("cpu_reference supports only runtime_mode='cpu'")
        if config.target_policy not in _SUPPORTED_EMISSION_POLICIES:
            raise ValueError(
                "cpu_reference supports dense_logits, topk_with_tail_v0, "
                "cascaded_soft_labels_v1, and corridor_exemplar_v1"
            )
        if config.sequence_length <= 0:
            raise ValueError("sequence_length must be > 0")
        if config.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if config.vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if config.top_k <= 0:
            raise ValueError("top_k must be > 0")
        if config.num_buckets <= 0:
            raise ValueError("num_buckets must be > 0")
        if config.exemplar_top_n <= 0:
            raise ValueError("exemplar_top_n must be > 0")
        self.config = config

    def capabilities(self) -> tuple[BackendCapability, ...]:
        return (
            BackendCapability(
                backend_id=self.backend_id,
                backend_family=self.backend_family,
                runtime_mode="cpu",
                target_policy="dense_logits",
                status="supported_debug",
                optimized=False,
                implemented_now=True,
                notes=(
                    "Spec 3.3C CPU reference backend emits deterministic dense "
                    "logits through the backend contract for debug/reference use."
                ),
            ),
            BackendCapability(
                backend_id=self.backend_id,
                backend_family=self.backend_family,
                runtime_mode="cpu",
                target_policy="topk_with_tail_v0",
                status="supported",
                optimized=False,
                implemented_now=True,
                notes=(
                    "Spec 3.3C CPU reference backend emits deterministic top-k "
                    "plus tail payloads through the backend contract."
                ),
            ),
            BackendCapability(
                backend_id=self.backend_id,
                backend_family=self.backend_family,
                runtime_mode="cpu",
                target_policy="cascaded_soft_labels_v1",
                status="supported",
                optimized=False,
                implemented_now=True,
                notes=(
                    "Spec 3.3C CPU reference backend emits deterministic "
                    "cascaded soft-label payloads through the backend contract."
                ),
            ),
            BackendCapability(
                backend_id=self.backend_id,
                backend_family=self.backend_family,
                runtime_mode="cpu",
                target_policy="corridor_exemplar_v1",
                status="supported",
                optimized=False,
                implemented_now=True,
                notes=(
                    "Spec 3.3C.1 adds serial/reference CPU corridor/exemplar "
                    "support through the backend contract."
                ),
            ),
        )

    def emit_batch(self, batch: TeacherBatchInput) -> TeacherEmissionResult:
        input_ids, attention_mask = _encode_texts(
            batch.texts,
            sequence_length=self.config.sequence_length,
            vocab_size=self.config.vocab_size,
        )
        logits = _deterministic_logits(input_ids, vocab_size=self.config.vocab_size)
        payload = self._payload_for_policy(logits)
        metadata = self.metadata(actual_batch_size=len(batch.texts))
        return TeacherEmissionResult(
            backend_id=self.backend_id,
            runtime_mode="cpu",
            target_policy=self.config.target_policy,
            input_ids=input_ids,
            attention_mask=attention_mask,
            payload=payload,
            metadata=metadata,
        )

    def metadata(self, actual_batch_size: int | None = None) -> dict[str, object]:
        metadata: dict[str, object] = {
            "requested_runtime_mode": self.config.runtime_mode,
            "effective_runtime_mode": "cpu",
            "requested_cpu_orchestration_mode": self.config.cpu_orchestration_mode,
            "effective_cpu_orchestration_mode": "serial",
            "requested_target_policy": self.config.target_policy,
            "effective_target_policy": self.config.target_policy,
            "backend_id": self.backend_id,
            "backend_family": self.backend_family,
            "runtime_kind": "cpu_reference",
            "device_kind": "cpu",
            "capability_status": _CAPABILITY_STATUS[self.config.target_policy],
            "optimized_path_used": False,
            "fallback_used": False,
            "sequence_length": self.config.sequence_length,
            "batch_size": self.config.batch_size,
            "configured_batch_size": self.config.batch_size,
            "actual_batch_size": (
                self.config.batch_size
                if actual_batch_size is None
                else actual_batch_size
            ),
            "vocab_size": self.config.vocab_size,
        }
        if self.config.target_policy in {
            "topk_with_tail_v0",
            "cascaded_soft_labels_v1",
        }:
            metadata.update(
                {
                    "requested_top_k": self.config.top_k,
                    "effective_top_k": min(self.config.top_k, self.config.vocab_size),
                }
            )
        if self.config.target_policy == "cascaded_soft_labels_v1":
            metadata.update(
                {
                    "num_buckets": self.config.num_buckets,
                    "bucket_policy": "contiguous_descending_tail_probability_mass",
                }
            )
        if self.config.target_policy == "corridor_exemplar_v1":
            effective_exemplar_top_n = min(
                self.config.exemplar_top_n,
                self.config.sequence_length,
            )
            metadata.update(
                {
                    "corridor_policy": "deterministic_reference_corridor_v1",
                    "exemplar_selection_policy": (
                        "deterministic_high_entropy_top_n_v1"
                    ),
                    "requested_exemplar_top_n": self.config.exemplar_top_n,
                    "effective_exemplar_top_n": effective_exemplar_top_n,
                    "exemplar_records": effective_exemplar_top_n,
                }
            )
        return metadata

    def close(self) -> None:
        return None

    def _payload_for_policy(self, logits: np.ndarray) -> dict[str, np.ndarray]:
        if self.config.target_policy == "dense_logits":
            return {"logits": logits}
        topk = _dense_logits_to_topk_tail(logits, top_k=self.config.top_k)
        if self.config.target_policy == "topk_with_tail_v0":
            return topk
        if self.config.target_policy == "cascaded_soft_labels_v1":
            return {
                **topk,
                "bucket_masses": _tail_bucket_masses(
                    logits,
                    top_token_ids=topk["top_token_ids"],
                    num_buckets=self.config.num_buckets,
                ),
            }
        if self.config.target_policy == "corridor_exemplar_v1":
            return _corridor_exemplar_payload(
                logits,
                top_k=self.config.top_k,
                exemplar_top_n=self.config.exemplar_top_n,
            )
        raise ValueError(f"unsupported target_policy: {self.config.target_policy}")


def _encode_texts(
    texts: tuple[str, ...],
    *,
    sequence_length: int,
    vocab_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    input_ids = np.zeros((len(texts), sequence_length), dtype=np.int32)
    attention_mask = np.zeros((len(texts), sequence_length), dtype=np.int32)
    for row, text in enumerate(texts):
        encoded = [
            (ord(char) + (17 * row) + position) % vocab_size
            for position, char in enumerate(text[:sequence_length])
        ]
        if encoded:
            input_ids[row, : len(encoded)] = np.asarray(encoded, dtype=np.int32)
            attention_mask[row, : len(encoded)] = 1
    return input_ids, attention_mask


def _deterministic_logits(input_ids: np.ndarray, *, vocab_size: int) -> np.ndarray:
    ids = np.asarray(input_ids, dtype=np.float32)
    rows = np.arange(ids.shape[0], dtype=np.float32)[:, None, None]
    positions = np.arange(ids.shape[1], dtype=np.float32)[None, :, None]
    vocab = np.arange(vocab_size, dtype=np.float32)[None, None, :]
    logits = (
        np.sin(ids[:, :, None] * 0.13 + positions * 0.07 + vocab * 0.17)
        + np.cos(rows * 0.19 + vocab * 0.11)
        + positions * 0.001
    )
    return logits.astype(np.float32)


def _dense_logits_to_topk_tail(
    logits: np.ndarray,
    *,
    top_k: int,
) -> dict[str, np.ndarray]:
    values = np.asarray(logits, dtype=np.float32)
    effective_top_k = min(top_k, values.shape[-1])
    probs = _softmax(values)
    log_probs = np.log(np.maximum(probs, np.finfo(np.float32).tiny))
    top_ids = np.argsort(-log_probs, axis=-1, kind="stable")[..., :effective_top_k]
    top_log_probs = np.take_along_axis(log_probs, top_ids, axis=-1).astype(np.float32)
    top_probs = np.take_along_axis(probs, top_ids, axis=-1).astype(np.float32)
    top_mass = np.sum(top_probs, axis=-1).astype(np.float32)
    tail_mass = np.clip(1.0 - top_mass, 0.0, 1.0).astype(np.float32)
    teacher_entropy = -np.sum(probs * log_probs, axis=-1).astype(np.float32)
    return {
        "top_token_ids": top_ids.astype(np.int32),
        "top_log_probs": top_log_probs,
        "top_probs": top_probs,
        "top_mass": top_mass,
        "tail_mass": tail_mass,
        "teacher_entropy": teacher_entropy,
    }


def _tail_bucket_masses(
    logits: np.ndarray,
    *,
    top_token_ids: np.ndarray,
    num_buckets: int,
) -> np.ndarray:
    probs = _softmax(np.asarray(logits, dtype=np.float32))
    top_mask = np.zeros(probs.shape, dtype=bool)
    np.put_along_axis(top_mask, top_token_ids, True, axis=-1)
    bucket_masses = np.zeros((*probs.shape[:2], num_buckets), dtype=np.float32)
    for row in range(probs.shape[0]):
        for position in range(probs.shape[1]):
            tail_probs = probs[row, position][~top_mask[row, position]]
            if tail_probs.size == 0:
                continue
            sorted_tail = np.sort(tail_probs)[::-1]
            buckets = np.array_split(sorted_tail, num_buckets)
            for bucket_id, bucket in enumerate(buckets):
                bucket_masses[row, position, bucket_id] = np.sum(
                    bucket,
                    dtype=np.float32,
                )
    return bucket_masses


def _corridor_exemplar_payload(
    logits: np.ndarray,
    *,
    top_k: int,
    exemplar_top_n: int,
) -> dict[str, object]:
    topk = _dense_logits_to_topk_tail(logits, top_k=top_k)
    effective_exemplar_top_n = min(exemplar_top_n, logits.shape[1])
    entropy = topk["teacher_entropy"]
    confidence = np.max(topk["top_probs"], axis=-1).astype(np.float32)
    exemplar_positions = np.argsort(
        -entropy,
        axis=-1,
        kind="stable",
    )[..., :effective_exemplar_top_n].astype(np.int32)
    selection_mask = np.zeros(entropy.shape, dtype=np.int32)
    np.put_along_axis(selection_mask, exemplar_positions, 1, axis=-1)
    exemplar_scores = np.take_along_axis(
        entropy,
        exemplar_positions,
        axis=-1,
    ).astype(np.float32)
    top_ids = topk["top_token_ids"][..., 0].astype(np.int32)
    top_probs = topk["top_probs"][..., 0].astype(np.float32)
    lengths = np.full((logits.shape[0],), logits.shape[1], dtype=np.int32)
    corridor_records = {
        "policy": "deterministic_reference_corridor_v1",
        "record_count": int(logits.shape[0]),
        "fields": (
            "corridor_top_token_ids",
            "corridor_top_probs",
            "corridor_teacher_entropy",
            "corridor_confidence",
        ),
    }
    exemplar_records = {
        "policy": "deterministic_high_entropy_top_n_v1",
        "record_count": int(logits.shape[0] * effective_exemplar_top_n),
        "positions_per_example": int(effective_exemplar_top_n),
    }
    return {
        "corridor_records": corridor_records,
        "corridor_summary": {
            "record_count": int(logits.shape[0]),
            "mean_confidence": float(np.mean(confidence, dtype=np.float32)),
            "mean_entropy": float(np.mean(entropy, dtype=np.float32)),
            "sequence_length": int(logits.shape[1]),
        },
        "exemplar_records": exemplar_records,
        "exemplar_summary": {
            "selection_policy": "deterministic_high_entropy_top_n_v1",
            "record_count": int(logits.shape[0] * effective_exemplar_top_n),
            "positions_per_example": int(effective_exemplar_top_n),
        },
        "mode_records": np.zeros((logits.shape[0],), dtype=np.int32),
        "corridor_top_token_ids": top_ids,
        "corridor_top_probs": top_probs,
        "corridor_teacher_entropy": entropy,
        "corridor_confidence": confidence,
        "corridor_lengths": lengths,
        "exemplar_positions": exemplar_positions,
        "exemplar_scores": exemplar_scores,
        "exemplar_selection_mask": selection_mask,
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted, dtype=np.float32)
    return (exp / np.sum(exp, axis=-1, keepdims=True)).astype(np.float32)
