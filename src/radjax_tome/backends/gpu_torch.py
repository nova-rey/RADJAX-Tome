from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np

from radjax_tome.backends.base import (
    BackendCapability,
    TeacherBackendConfig,
    TeacherBatchInput,
    TeacherEmissionResult,
)
from radjax_tome.backends.hf_torch import _effective_tokenizer_id

_SUPPORTED_EMISSION_POLICIES = {
    "dense_logits",
    "topk_with_tail_v0",
    "cascaded_soft_labels_v1",
}
_CAPABILITY_STATUS = {
    "dense_logits": "supported_debug",
    "topk_with_tail_v0": "optimized",
    "cascaded_soft_labels_v1": "optimized",
}


@dataclass(frozen=True)
class TorchAcceleratorDetection:
    available: bool
    device_kind: str
    device: str | None
    reason: str
    torch_version: str | None = None
    cuda_available: bool = False
    mps_available: bool = False


class GPUTorchTeacherEmissionBackend:
    backend_id = "gpu_torch"
    backend_family = "gpu_torch"
    runtime_mode = "cpu_gpu"

    def __init__(self, config: TeacherBackendConfig) -> None:
        if config.backend_id != self.backend_id:
            raise ValueError(
                f"gpu_torch requires backend_id={self.backend_id!r}, "
                f"got {config.backend_id!r}"
            )
        if config.runtime_mode != "cpu_gpu":
            raise ValueError("gpu_torch supports only runtime_mode='cpu_gpu'")
        if config.target_policy not in _SUPPORTED_EMISSION_POLICIES:
            raise ValueError(
                "gpu_torch supports dense_logits, topk_with_tail_v0, "
                "and cascaded_soft_labels_v1; "
                f"{config.target_policy!r} is not implemented yet"
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
        self._device: TorchAcceleratorDetection | None = None

    @classmethod
    def available(cls, config: TeacherBackendConfig) -> bool:
        try:
            backend = cls(config)
            backend._load_model_and_tokenizer()
        except Exception:
            return False
        return True

    @classmethod
    def detect_device(cls) -> TorchAcceleratorDetection:
        return detect_torch_accelerator()

    def capabilities(self) -> tuple[BackendCapability, ...]:
        return (
            BackendCapability(
                backend_id=self.backend_id,
                backend_family=self.backend_family,
                runtime_mode="cpu_gpu",
                target_policy="dense_logits",
                status="supported_debug",
                optimized=False,
                implemented_now=True,
                notes=(
                    "Spec 3.3F1 gpu_torch emits dense debug HF logits on an "
                    "available CUDA or MPS Torch accelerator. This path is "
                    "not optimized and transfers dense logits back to host."
                ),
            ),
            BackendCapability(
                backend_id=self.backend_id,
                backend_family=self.backend_family,
                runtime_mode="cpu_gpu",
                target_policy="topk_with_tail_v0",
                status="optimized",
                optimized=True,
                implemented_now=True,
                notes=(
                    "Spec 3.3F2 gpu_torch computes top-k plus tail compact "
                    "reduction on the selected CUDA or MPS accelerator and "
                    "transfers only compact payload arrays back to host."
                ),
            ),
            BackendCapability(
                backend_id=self.backend_id,
                backend_family=self.backend_family,
                runtime_mode="cpu_gpu",
                target_policy="cascaded_soft_labels_v1",
                status="optimized",
                optimized=True,
                implemented_now=True,
                notes=(
                    "Spec 3.3F3 gpu_torch computes cascaded soft-label "
                    "compact reduction on the selected CUDA or MPS "
                    "accelerator, including bucket masses, and transfers only "
                    "compact payload arrays back to host."
                ),
            ),
            BackendCapability(
                backend_id=self.backend_id,
                backend_family=self.backend_family,
                runtime_mode="cpu_gpu",
                target_policy="corridor_exemplar_v1",
                status="historical_reference_exists",
                optimized=False,
                implemented_now=False,
                notes=(
                    "Historical QRWKV-XLA optimization work can guide future "
                    "corridor/exemplar acceleration; active gpu_torch support "
                    "is not implemented."
                ),
            ),
        )

    def emit_batch(self, batch: TeacherBatchInput) -> TeacherEmissionResult:
        torch, tokenizer, model, device = self._load_model_and_tokenizer()
        encoded = tokenizer(
            list(batch.texts),
            padding="max_length",
            truncation=True,
            max_length=self.config.sequence_length,
            return_tensors="pt",
        )
        input_ids_tensor = encoded["input_ids"].to(device.device)
        attention_mask_tensor = encoded["attention_mask"].to(device.device)
        with torch.no_grad():
            output = model(
                input_ids=input_ids_tensor,
                attention_mask=attention_mask_tensor,
            )
        input_ids = input_ids_tensor.detach().to("cpu").numpy().astype(np.int32)
        attention_mask = (
            attention_mask_tensor.detach().to("cpu").numpy().astype(np.int32)
        )
        logits = output.logits
        effective_vocab_size = int(logits.shape[-1])
        estimated_dense_logits_dtype = _logits_dtype_name(logits)
        if self.config.target_policy == "dense_logits":
            dense_logits = logits.detach().to("cpu").numpy().astype(np.float32)
            payload = {"logits": dense_logits}
        elif self.config.target_policy == "cascaded_soft_labels_v1":
            payload = _compact_payload_to_numpy(
                _gpu_cascaded_reduce(
                    torch,
                    logits,
                    top_k=self.config.top_k,
                    num_buckets=self.config.num_buckets,
                )
            )
        else:
            payload = _compact_payload_to_numpy(
                _gpu_topk_tail_reduce(torch, logits, top_k=self.config.top_k)
            )
        return TeacherEmissionResult(
            backend_id=self.backend_id,
            runtime_mode="cpu_gpu",
            target_policy=self.config.target_policy,
            input_ids=input_ids,
            attention_mask=attention_mask,
            payload=payload,
            metadata=self.metadata(
                actual_batch_size=len(batch.texts),
                effective_vocab_size=effective_vocab_size,
                compact_payload=payload,
                estimated_dense_logits_dtype=estimated_dense_logits_dtype,
            ),
        )

    def metadata(
        self,
        *,
        actual_batch_size: int | None = None,
        effective_vocab_size: int | None = None,
        compact_payload: dict[str, np.ndarray] | None = None,
        estimated_dense_logits_dtype: str = "float32",
    ) -> dict[str, object]:
        device = self._device or detect_torch_accelerator()
        tokenizer_id = _effective_tokenizer_id(self.config)
        local_files_only = (
            self.config.local_files_only or not self.config.allow_downloads
        )
        effective_vocab = (
            self.config.vocab_size
            if effective_vocab_size is None
            else effective_vocab_size
        )
        actual_batch = (
            self.config.batch_size if actual_batch_size is None else actual_batch_size
        )
        dense_debug_path = self.config.target_policy == "dense_logits"
        metadata: dict[str, object] = {
            "requested_runtime_mode": self.config.runtime_mode,
            "effective_runtime_mode": "cpu_gpu",
            "requested_target_policy": self.config.target_policy,
            "effective_target_policy": self.config.target_policy,
            "backend_id": self.backend_id,
            "backend_family": self.backend_family,
            "runtime_kind": "gpu_torch",
            "device_kind": device.device_kind,
            "torch_device": device.device,
            "capability_status": _CAPABILITY_STATUS[self.config.target_policy],
            "optimized_path_used": not dense_debug_path,
            "dense_debug_path": dense_debug_path,
            "dense_logits_transferred_to_host": dense_debug_path,
            "compact_reduction_used": not dense_debug_path,
            "gpu_compact_reduction_implemented": not dense_debug_path,
            "fallback_used": False,
            "model_id": self.config.model_id,
            "tokenizer_id": tokenizer_id,
            "configured_batch_size": self.config.batch_size,
            "actual_batch_size": actual_batch,
            "sequence_length": self.config.sequence_length,
            "configured_vocab_size": self.config.vocab_size,
            "effective_vocab_size": effective_vocab,
            "local_files_only": local_files_only,
            "allow_downloads": self.config.allow_downloads,
            "torch_version": device.torch_version,
            "cuda_available": device.cuda_available,
            "mps_available": device.mps_available,
        }
        if self.config.target_policy in {
            "topk_with_tail_v0",
            "cascaded_soft_labels_v1",
        }:
            effective_top_k = min(self.config.top_k, effective_vocab)
            compact_payload = compact_payload or {}
            compact_fields = [
                "top_token_ids",
                "top_log_probs",
                "top_probs",
                "top_mass",
                "tail_mass",
            ]
            gpu_reduction_mode = "compact_topk_tail"
            if self.config.target_policy == "cascaded_soft_labels_v1":
                compact_fields.append("bucket_masses")
                gpu_reduction_mode = "compact_cascaded_soft_labels"
            compact_fields.append("teacher_entropy")
            compact_metadata = {
                "requested_top_k": self.config.top_k,
                "effective_top_k": effective_top_k,
                "gpu_reduction_mode": gpu_reduction_mode,
                "compact_payload_fields": compact_fields,
                "compact_payload_arrays": [
                    field for field in compact_fields if field in compact_payload
                ],
                "compact_bytes_transferred_to_host": sum(
                    int(compact_payload[field].nbytes)
                    for field in compact_fields
                    if field in compact_payload
                ),
                "estimated_dense_logits_bytes": _estimate_dense_logits_bytes(
                    actual_batch_size=actual_batch,
                    sequence_length=self.config.sequence_length,
                    effective_vocab_size=effective_vocab,
                    dtype_name=estimated_dense_logits_dtype,
                ),
                "estimated_dense_logits_dtype": estimated_dense_logits_dtype,
            }
            if self.config.target_policy == "cascaded_soft_labels_v1":
                compact_metadata.update(
                    {
                        "num_buckets": self.config.num_buckets,
                        "bucket_policy": (
                            "contiguous_descending_tail_probability_mass"
                        ),
                    }
                )
            metadata.update(compact_metadata)
        return metadata

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._device = None

    def _load_model_and_tokenizer(
        self,
    ) -> tuple[Any, Any, Any, TorchAcceleratorDetection]:
        if (
            self._torch is not None
            and self._tokenizer is not None
            and self._model is not None
            and self._device is not None
        ):
            return self._torch, self._tokenizer, self._model, self._device
        device = detect_torch_accelerator()
        if not device.available:
            if device.reason == "missing torch":
                raise RuntimeError(
                    "gpu_torch backend requires optional dependency torch. "
                    "Install the teacher-hf extra to use torch/transformers "
                    "emission."
                )
            raise RuntimeError(
                "gpu_torch requires runtime_mode='cpu_gpu' with an available "
                "cuda or mps Torch accelerator; no CPU fallback is used."
            )
        try:
            torch = import_module("torch")
        except ImportError as exc:
            raise RuntimeError(
                "gpu_torch backend requires optional dependency torch. "
                "Install the teacher-hf extra to use torch/transformers emission."
            ) from exc
        try:
            transformers = import_module("transformers")
        except ImportError as exc:
            raise RuntimeError(
                "gpu_torch backend requires optional dependency transformers. "
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
                "gpu_torch backend could not load local torch/transformers "
                f"model or tokenizer for model_id={self.config.model_id!r}. "
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
            model = model.to(device.device)
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        return torch, tokenizer, model, device


def check_gpu_torch_backend_available(config: TeacherBackendConfig) -> bool:
    return GPUTorchTeacherEmissionBackend.available(config)


def detect_torch_accelerator() -> TorchAcceleratorDetection:
    try:
        torch = import_module("torch")
    except ImportError:
        return TorchAcceleratorDetection(
            available=False,
            device_kind="unavailable",
            device=None,
            reason="missing torch",
        )
    torch_version = getattr(torch, "__version__", None)
    cuda_available = _safe_is_available(getattr(torch, "cuda", None))
    mps_available = _safe_is_available(
        getattr(getattr(torch, "backends", None), "mps", None)
    )
    if cuda_available:
        return TorchAcceleratorDetection(
            available=True,
            device_kind="cuda",
            device="cuda",
            reason="cuda available",
            torch_version=torch_version,
            cuda_available=True,
            mps_available=mps_available,
        )
    if mps_available:
        return TorchAcceleratorDetection(
            available=True,
            device_kind="mps",
            device="mps",
            reason="mps available",
            torch_version=torch_version,
            cuda_available=False,
            mps_available=True,
        )
    return TorchAcceleratorDetection(
        available=False,
        device_kind="unavailable",
        device=None,
        reason="no cuda or mps accelerator available",
        torch_version=torch_version,
        cuda_available=False,
        mps_available=False,
    )


def _safe_is_available(namespace: Any) -> bool:
    is_available = getattr(namespace, "is_available", None)
    if not callable(is_available):
        return False
    try:
        return bool(is_available())
    except Exception:
        return False


def _gpu_topk_tail_reduce(
    torch: Any,
    logits: Any,
    *,
    top_k: int,
) -> dict[str, Any]:
    effective_top_k = min(top_k, int(logits.shape[-1]))
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    top_log_probs, top_token_ids = torch.topk(
        log_probs,
        k=effective_top_k,
        dim=-1,
    )
    top_probs = torch.exp(top_log_probs)
    top_mass = torch.sum(top_probs, dim=-1)
    tail_mass = torch.clamp(1.0 - top_mass, min=0.0, max=1.0)
    teacher_entropy = -torch.sum(probs * log_probs, dim=-1)
    return {
        "top_token_ids": top_token_ids,
        "top_log_probs": top_log_probs,
        "top_probs": top_probs,
        "top_mass": top_mass,
        "tail_mass": tail_mass,
        "teacher_entropy": teacher_entropy,
    }


def _gpu_cascaded_reduce(
    torch: Any,
    logits: Any,
    *,
    top_k: int,
    num_buckets: int,
) -> dict[str, Any]:
    payload = _gpu_topk_tail_reduce(torch, logits, top_k=top_k)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    payload["bucket_masses"] = _tail_bucket_masses_on_device(
        torch,
        probs,
        top_token_ids=payload["top_token_ids"],
        num_buckets=num_buckets,
    )
    return payload


def _tail_bucket_masses_on_device(
    torch: Any,
    probs: Any,
    *,
    top_token_ids: Any,
    num_buckets: int,
) -> Any:
    top_mask = torch.zeros_like(probs, dtype=torch.bool)
    top_mask.scatter_(-1, top_token_ids, True)
    bucket_masses = torch.zeros(
        (*probs.shape[:2], num_buckets),
        dtype=probs.dtype,
        device=probs.device,
    )
    for row in range(int(probs.shape[0])):
        for position in range(int(probs.shape[1])):
            tail_probs = probs[row, position][~top_mask[row, position]]
            if int(tail_probs.shape[0]) == 0:
                continue
            sorted_tail = torch.sort(tail_probs, descending=True).values
            tail_count = int(sorted_tail.shape[0])
            quotient, remainder = divmod(tail_count, num_buckets)
            for bucket_id in range(num_buckets):
                start = bucket_id * quotient + min(bucket_id, remainder)
                stop = start + quotient + (1 if bucket_id < remainder else 0)
                if start < stop:
                    bucket_masses[row, position, bucket_id] = torch.sum(
                        sorted_tail[start:stop]
                    )
    return bucket_masses


def _compact_payload_to_numpy(payload: dict[str, Any]) -> dict[str, np.ndarray]:
    compact = {
        "top_token_ids": _tensor_to_numpy(payload["top_token_ids"], np.int32),
        "top_log_probs": _tensor_to_numpy(payload["top_log_probs"], np.float32),
        "top_probs": _tensor_to_numpy(payload["top_probs"], np.float32),
        "top_mass": _tensor_to_numpy(payload["top_mass"], np.float32),
        "tail_mass": _tensor_to_numpy(payload["tail_mass"], np.float32),
        "teacher_entropy": _tensor_to_numpy(
            payload["teacher_entropy"],
            np.float32,
        ),
    }
    if "bucket_masses" in payload:
        compact["bucket_masses"] = _tensor_to_numpy(
            payload["bucket_masses"],
            np.float32,
        )
    return compact


def _tensor_to_numpy(tensor: Any, dtype: type[np.generic]) -> np.ndarray:
    return tensor.detach().to("cpu").numpy().astype(dtype)


def _estimate_dense_logits_bytes(
    *,
    actual_batch_size: int,
    sequence_length: int,
    effective_vocab_size: int,
    dtype_name: str,
) -> int:
    return (
        actual_batch_size
        * sequence_length
        * effective_vocab_size
        * _dtype_nbytes(dtype_name)
    )


def _logits_dtype_name(logits: Any) -> str:
    dtype = getattr(logits, "dtype", None)
    if dtype is None:
        return "float32"
    name = str(dtype)
    if name.startswith("torch."):
        name = name.removeprefix("torch.")
    return name


def _dtype_nbytes(dtype_name: str) -> int:
    normalized = dtype_name.removeprefix("torch.")
    try:
        return int(np.dtype(normalized).itemsize)
    except TypeError:
        return 4
