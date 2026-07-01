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
        if config.target_policy != "dense_logits":
            raise ValueError(
                "gpu_torch Spec 3.3F1 supports only target_policy='dense_logits'"
            )
        if config.sequence_length <= 0:
            raise ValueError("sequence_length must be > 0")
        if config.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if config.vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
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
                status="historical_reference_exists",
                optimized=False,
                implemented_now=False,
                notes=(
                    "Historical QRWKV-XLA compact GPU reduction exists as "
                    "migration reference; active gpu_torch top-k/tail "
                    "reduction starts in a later 3.3F spec."
                ),
            ),
            BackendCapability(
                backend_id=self.backend_id,
                backend_family=self.backend_family,
                runtime_mode="cpu_gpu",
                target_policy="cascaded_soft_labels_v1",
                status="historical_reference_exists",
                optimized=False,
                implemented_now=False,
                notes=(
                    "Historical QRWKV-XLA compact GPU reduction can guide "
                    "future cascaded soft-label migration; Spec 3.3F1 does "
                    "not implement this reduction."
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
        logits = output.logits.detach().to("cpu").numpy().astype(np.float32)
        input_ids = input_ids_tensor.detach().to("cpu").numpy().astype(np.int32)
        attention_mask = (
            attention_mask_tensor.detach().to("cpu").numpy().astype(np.int32)
        )
        effective_vocab_size = int(logits.shape[-1])
        return TeacherEmissionResult(
            backend_id=self.backend_id,
            runtime_mode="cpu_gpu",
            target_policy="dense_logits",
            input_ids=input_ids,
            attention_mask=attention_mask,
            payload={"logits": logits},
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
        return {
            "requested_runtime_mode": self.config.runtime_mode,
            "effective_runtime_mode": "cpu_gpu",
            "requested_target_policy": self.config.target_policy,
            "effective_target_policy": "dense_logits",
            "backend_id": self.backend_id,
            "backend_family": self.backend_family,
            "runtime_kind": "gpu_torch",
            "device_kind": device.device_kind,
            "torch_device": device.device,
            "capability_status": "supported_debug",
            "optimized_path_used": False,
            "dense_debug_path": True,
            "dense_logits_transferred_to_host": True,
            "compact_reduction_used": False,
            "gpu_compact_reduction_implemented": False,
            "fallback_used": False,
            "model_id": self.config.model_id,
            "tokenizer_id": tokenizer_id,
            "configured_batch_size": self.config.batch_size,
            "actual_batch_size": (
                self.config.batch_size
                if actual_batch_size is None
                else actual_batch_size
            ),
            "sequence_length": self.config.sequence_length,
            "configured_vocab_size": self.config.vocab_size,
            "effective_vocab_size": effective_vocab,
            "local_files_only": local_files_only,
            "allow_downloads": self.config.allow_downloads,
            "torch_version": device.torch_version,
            "cuda_available": device.cuda_available,
            "mps_available": device.mps_available,
        }

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
