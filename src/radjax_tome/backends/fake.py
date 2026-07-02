from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from radjax_tome.backends.base import (
    BackendCapability,
    TeacherBackendConfig,
    TeacherBatchInput,
    TeacherEmissionResult,
    resolve_gpu_batch_size_policy,
    validate_gpu_batch_size_policy_config,
)


@dataclass(frozen=True)
class FakeTeacherBackend:
    vocab_size: int = 8
    backend_id: str = "fake"

    def emit_logits(self, input_ids: np.ndarray) -> np.ndarray:
        return _deterministic_logits(input_ids, vocab_size=self.vocab_size)


class FakeNumpyTeacherEmissionBackend:
    backend_id = "fake_numpy"
    backend_family = "fake_numpy"
    runtime_mode = "cpu"

    def __init__(self, config: TeacherBackendConfig) -> None:
        if config.backend_id != self.backend_id:
            raise ValueError(
                f"fake backend requires backend_id={self.backend_id!r}, "
                f"got {config.backend_id!r}"
            )
        if config.runtime_mode != "cpu":
            raise ValueError("fake_numpy supports only runtime_mode='cpu'")
        if config.target_policy != "dense_logits":
            raise ValueError("fake_numpy supports only target_policy='dense_logits'")
        validate_gpu_batch_size_policy_config(config)
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
                    "Spec 3.3B fake backend proves the backend contract for "
                    "dense_logits."
                ),
            ),
        )

    def emit_batch(self, batch: TeacherBatchInput) -> TeacherEmissionResult:
        input_ids, attention_mask = _encode_texts(
            batch.texts,
            sequence_length=self.config.sequence_length,
            vocab_size=self.config.vocab_size,
        )
        payload = {"logits": _deterministic_logits(input_ids, self.config.vocab_size)}
        return TeacherEmissionResult(
            backend_id=self.backend_id,
            runtime_mode="cpu",
            target_policy="dense_logits",
            input_ids=input_ids,
            attention_mask=attention_mask,
            payload=payload,
            metadata=self.metadata(payload=payload),
        )

    def metadata(
        self, payload: dict[str, np.ndarray] | None = None
    ) -> dict[str, object]:
        return {
            "requested_runtime_mode": self.config.runtime_mode,
            "effective_runtime_mode": "cpu",
            "requested_target_policy": self.config.target_policy,
            "effective_target_policy": "dense_logits",
            "backend_id": self.backend_id,
            "backend_family": self.backend_family,
            "runtime_kind": "fake_numpy",
            "device_kind": "cpu",
            "optimized_path_used": False,
            "fallback_used": False,
            **resolve_gpu_batch_size_policy(self.config, payload=payload),
        }

    def close(self) -> None:
        return None


def _encode_texts(
    texts: tuple[str, ...],
    *,
    sequence_length: int,
    vocab_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be > 0")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be > 0")
    input_ids = np.zeros((len(texts), sequence_length), dtype=np.int32)
    attention_mask = np.zeros((len(texts), sequence_length), dtype=np.int32)
    for row, text in enumerate(texts):
        encoded = [(ord(char) + row) % vocab_size for char in text[:sequence_length]]
        if encoded:
            input_ids[row, : len(encoded)] = np.asarray(encoded, dtype=np.int32)
            attention_mask[row, : len(encoded)] = 1
    return input_ids, attention_mask


def _deterministic_logits(input_ids: np.ndarray, vocab_size: int) -> np.ndarray:
    ids = np.asarray(input_ids, dtype=np.float32)
    vocab = np.arange(vocab_size, dtype=np.float32)[None, None, :]
    return np.sin(ids[:, :, None] * 0.17 + vocab * 0.23).astype(np.float32)
