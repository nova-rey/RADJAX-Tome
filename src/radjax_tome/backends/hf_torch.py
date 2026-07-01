from __future__ import annotations

from importlib import import_module
from typing import Any

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
}
_CAPABILITY_STATUS: dict[TargetPolicy, str] = {
    "dense_logits": "supported_debug",
    "topk_with_tail_v0": "supported",
    "cascaded_soft_labels_v1": "supported",
    "corridor_exemplar_v1": "planned",
}
_DEFAULT_FAKE_TOKENIZER_ID = "fake-deterministic-tokenizer"


class HFTorchTeacherEmissionBackend:
    backend_id = "hf_torch"
    backend_family = "hf_torch"
    runtime_mode = "cpu"

    def __init__(self, config: TeacherBackendConfig) -> None:
        if config.backend_id != self.backend_id:
            raise ValueError(
                f"hf_torch requires backend_id={self.backend_id!r}, "
                f"got {config.backend_id!r}"
            )
        if config.runtime_mode != "cpu":
            raise ValueError("hf_torch supports only runtime_mode='cpu'")
        if config.target_policy not in _SUPPORTED_EMISSION_POLICIES:
            raise ValueError(
                "hf_torch supports dense_logits, topk_with_tail_v0, "
                "and cascaded_soft_labels_v1"
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
        self.config = config
        self._torch: Any | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    @classmethod
    def available(cls, config: TeacherBackendConfig) -> bool:
        try:
            backend = cls(config)
            backend._load_model_and_tokenizer()
        except Exception:
            return False
        return True

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
                    "Spec 3.3E HF Torch backend emits real causal-LM dense "
                    "logits through the backend contract when optional local "
                    "HF dependencies and model files are available."
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
                    "Spec 3.3E HF Torch backend emits top-k plus tail payloads "
                    "by CPU-reducing real HF logits through the backend contract."
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
                    "Spec 3.3E HF Torch backend emits cascaded soft-label "
                    "payloads by CPU-reducing real HF logits through the "
                    "backend contract."
                ),
            ),
            BackendCapability(
                backend_id=self.backend_id,
                backend_family=self.backend_family,
                runtime_mode="cpu",
                target_policy="corridor_exemplar_v1",
                status="planned",
                optimized=False,
                implemented_now=False,
                notes=(
                    "HF Torch corridor/exemplar support remains planned; "
                    "Spec 3.3E does not implement this behavioral pass."
                ),
            ),
        )

    def emit_batch(self, batch: TeacherBatchInput) -> TeacherEmissionResult:
        torch, tokenizer, model = self._load_model_and_tokenizer()
        encoded = tokenizer(
            list(batch.texts),
            padding="max_length",
            truncation=True,
            max_length=self.config.sequence_length,
            return_tensors="pt",
        )
        input_ids_tensor = encoded["input_ids"].to("cpu")
        attention_mask_tensor = encoded["attention_mask"].to("cpu")
        with torch.no_grad():
            output = model(
                input_ids=input_ids_tensor,
                attention_mask=attention_mask_tensor,
            )
        logits = output.logits.detach().to("cpu").numpy().astype(np.float32)
        input_ids = input_ids_tensor.detach().to("cpu").numpy().astype(np.int32)
        attention_mask = (
            attention_mask_tensor.detach().to("cpu").numpy().astype(np.int32)
        )
        effective_vocab_size = int(logits.shape[-1])
        payload = self._payload_for_policy(logits)
        return TeacherEmissionResult(
            backend_id=self.backend_id,
            runtime_mode="cpu",
            target_policy=self.config.target_policy,
            input_ids=input_ids,
            attention_mask=attention_mask,
            payload=payload,
            metadata=self.metadata(
                actual_batch_size=len(batch.texts),
                effective_vocab_size=effective_vocab_size,
            ),
        )

    def metadata(
        self,
        *,
        actual_batch_size: int | None = None,
        effective_vocab_size: int | None = None,
    ) -> dict[str, object]:
        tokenizer_id = _effective_tokenizer_id(self.config)
        local_files_only = (
            self.config.local_files_only or not self.config.allow_downloads
        )
        effective_vocab = (
            self.config.vocab_size
            if effective_vocab_size is None
            else effective_vocab_size
        )
        metadata: dict[str, object] = {
            "requested_runtime_mode": self.config.runtime_mode,
            "effective_runtime_mode": "cpu",
            "requested_cpu_orchestration_mode": self.config.cpu_orchestration_mode,
            "effective_cpu_orchestration_mode": "serial",
            "requested_target_policy": self.config.target_policy,
            "effective_target_policy": self.config.target_policy,
            "backend_id": self.backend_id,
            "backend_family": self.backend_family,
            "runtime_kind": "hf_torch",
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
            "model_id": self.config.model_id,
            "tokenizer_id": tokenizer_id,
            "local_files_only": local_files_only,
            "allow_downloads": self.config.allow_downloads,
            "configured_vocab_size": self.config.vocab_size,
            "effective_vocab_size": effective_vocab,
            "vocab_size_mismatch": self.config.vocab_size != effective_vocab,
        }
        if self.config.target_policy in {
            "topk_with_tail_v0",
            "cascaded_soft_labels_v1",
        }:
            metadata.update(
                {
                    "requested_top_k": self.config.top_k,
                    "effective_top_k": min(self.config.top_k, effective_vocab),
                }
            )
        if self.config.target_policy == "cascaded_soft_labels_v1":
            metadata.update(
                {
                    "num_buckets": self.config.num_buckets,
                    "bucket_policy": "contiguous_descending_tail_probability_mass",
                }
            )
        return metadata

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        self._torch = None

    def _load_model_and_tokenizer(self) -> tuple[Any, Any, Any]:
        if (
            self._torch is not None
            and self._tokenizer is not None
            and self._model is not None
        ):
            return self._torch, self._tokenizer, self._model
        try:
            torch = import_module("torch")
        except ImportError as exc:
            raise RuntimeError(
                "hf_torch backend requires optional dependency torch. "
                "Install the teacher-hf extra to use torch/transformers emission."
            ) from exc
        try:
            transformers = import_module("transformers")
        except ImportError as exc:
            raise RuntimeError(
                "hf_torch backend requires optional dependency transformers. "
                "Install the teacher-hf extra to use torch/transformers emission."
            ) from exc
        tokenizer_id = _effective_tokenizer_id(self.config)
        local_files_only = (
            self.config.local_files_only or not self.config.allow_downloads
        )
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                tokenizer_id,
                local_files_only=local_files_only,
            )
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                local_files_only=local_files_only,
            )
        except Exception as exc:
            raise RuntimeError(
                "hf_torch backend could not load local torch/transformers model "
                f"or tokenizer for model_id={self.config.model_id!r}. "
                "Install the teacher-hf extra and provide local model files, or "
                "set allow_downloads only in an explicitly network-enabled run."
            ) from exc
        if getattr(tokenizer, "pad_token_id", None) is None:
            eos_token = getattr(tokenizer, "eos_token", None)
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            if eos_token is not None and eos_token_id is not None:
                tokenizer.pad_token = eos_token
        model.eval()
        if hasattr(model, "to"):
            model = model.to("cpu")
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        return torch, tokenizer, model

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
        raise ValueError(f"unsupported target_policy: {self.config.target_policy}")


def _effective_tokenizer_id(config: TeacherBackendConfig) -> str:
    if config.tokenizer_id and config.tokenizer_id != _DEFAULT_FAKE_TOKENIZER_ID:
        return config.tokenizer_id
    return config.model_id


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


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted, dtype=np.float32)
    return (exp / np.sum(exp, axis=-1, keepdims=True)).astype(np.float32)
