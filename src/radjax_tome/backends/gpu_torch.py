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


class TeacherBackendError(RuntimeError):
    """Base error for runtime backend failures."""


class TeacherBackendDependencyError(TeacherBackendError):
    """Raised when an optional backend dependency is unavailable."""


class TeacherBackendUnavailableError(TeacherBackendError):
    """Raised when a backend cannot run in the requested runtime."""


class TeacherBackendModelLoadError(TeacherBackendError):
    """Raised when a configured local model/tokenizer cannot be loaded."""


class TeacherBackendDeviceError(TeacherBackendError):
    """Raised when device transfer, forward, or reduction work fails."""


class TeacherBackendUnsupportedPolicyError(TeacherBackendError):
    """Raised when a backend target policy is not implemented."""


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
_DEFAULT_GPU_VOCAB_CHUNK_SIZE = 8192
_REDUCER_WORKSPACE_FACTOR = 3


@dataclass(frozen=True)
class TorchAcceleratorDetection:
    available: bool
    device_kind: str
    device: str | None
    reason: str
    torch_version: str | None = None
    cuda_available: bool = False
    mps_available: bool = False


@dataclass(frozen=True)
class VocabChunkingPlan:
    requested: bool
    requested_size: int | None
    effective_size: int | None
    used: bool
    reason: str | None


class GPUTorchTeacherEmissionBackend:
    backend_id = "gpu_torch"
    backend_family = "gpu_torch"
    runtime_mode = "cpu_gpu"

    def __init__(self, config: TeacherBackendConfig) -> None:
        _validate_gpu_torch_config(config)
        self.config = config
        self._torch: Any | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._device: TorchAcceleratorDetection | None = None

    @classmethod
    def available(cls, config: TeacherBackendConfig) -> bool:
        return bool(cls.diagnostics(config)["can_emit"])

    @classmethod
    def diagnostics(cls, config: TeacherBackendConfig) -> dict[str, object]:
        return diagnose_gpu_torch_backend(config)

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
                    "transfers only compact payload arrays back to host. "
                    "Spec 3.3F4 adds optional vocab-axis chunking and "
                    "workspace metadata for this compact reducer."
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
                    "compact payload arrays back to host. Spec 3.3F4 adds "
                    "optional vocab-axis chunking, workspace metadata, and "
                    "shared probability workspace reuse."
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
        try:
            input_ids_tensor = encoded["input_ids"].to(device.device)
            attention_mask_tensor = encoded["attention_mask"].to(device.device)
        except Exception as exc:
            raise _wrap_device_error(
                "input tensor transfer to device", exc, device
            ) from exc
        try:
            with torch.no_grad():
                output = model(
                    input_ids=input_ids_tensor,
                    attention_mask=attention_mask_tensor,
                )
        except Exception as exc:
            raise _wrap_device_error("model forward", exc, device) from exc
        try:
            input_ids = input_ids_tensor.detach().to("cpu").numpy().astype(np.int32)
            attention_mask = (
                attention_mask_tensor.detach().to("cpu").numpy().astype(np.int32)
            )
        except Exception as exc:
            raise _wrap_device_error(
                "input tensor transfer to CPU", exc, device
            ) from exc
        logits = output.logits
        effective_vocab_size = int(logits.shape[-1])
        estimated_dense_logits_dtype = _logits_dtype_name(logits)
        chunking_plan = _vocab_chunking_plan(self.config, self.config.target_policy)
        if self.config.target_policy == "dense_logits":
            try:
                dense_logits = logits.detach().to("cpu").numpy().astype(np.float32)
            except Exception as exc:
                raise _wrap_device_error(
                    "dense logits transfer to CPU", exc, device
                ) from exc
            payload = {"logits": dense_logits}
        elif self.config.target_policy == "cascaded_soft_labels_v1":
            try:
                compact_payload = _gpu_cascaded_reduce(
                    torch,
                    logits,
                    top_k=self.config.top_k,
                    num_buckets=self.config.num_buckets,
                    vocab_chunk_size=chunking_plan.effective_size,
                )
            except Exception as exc:
                raise _wrap_device_error("compact reduction", exc, device) from exc
            try:
                payload = _compact_payload_to_numpy(compact_payload)
            except Exception as exc:
                raise _wrap_device_error(
                    "compact tensor transfer to CPU", exc, device
                ) from exc
        else:
            try:
                compact_payload = _gpu_topk_tail_reduce(
                    torch,
                    logits,
                    top_k=self.config.top_k,
                    vocab_chunk_size=chunking_plan.effective_size,
                )
            except Exception as exc:
                raise _wrap_device_error("compact reduction", exc, device) from exc
            try:
                payload = _compact_payload_to_numpy(compact_payload)
            except Exception as exc:
                raise _wrap_device_error(
                    "compact tensor transfer to CPU", exc, device
                ) from exc
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
        chunking_plan = _vocab_chunking_plan(self.config, self.config.target_policy)
        chunks_per_batch = _vocab_chunks_per_batch(
            effective_vocab_size=effective_vocab,
            vocab_chunk_size=chunking_plan.effective_size,
        )
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
            "fallback_policy": self.config.fallback_policy,
            "fallback_allowed": self.config.fallback_policy == "auto",
            "fallback_handled_by": "none",
            "diagnostic_status": "ok",
            "failure_stage": "none",
            "failure_reason": None,
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
            "vocab_chunking_requested": chunking_plan.requested,
            "vocab_chunking_used": chunking_plan.used,
            "gpu_vocab_chunk_size_requested": chunking_plan.requested_size,
            "gpu_vocab_chunk_size_effective": chunking_plan.effective_size,
            "gpu_vocab_chunks_per_batch": chunks_per_batch,
        }
        if chunking_plan.reason is not None:
            metadata["vocab_chunking_reason"] = chunking_plan.reason
        if self.config.target_policy in {
            "topk_with_tail_v0",
            "cascaded_soft_labels_v1",
        }:
            effective_top_k = min(self.config.top_k, effective_vocab)
            compact_payload = compact_payload or {}
            compact_fields = _compact_payload_fields(self.config.target_policy)
            gpu_reduction_mode = _gpu_reduction_mode(
                self.config.target_policy,
                vocab_chunking_used=chunking_plan.used,
            )
            estimated_dense_logits_bytes = _estimate_dense_logits_bytes(
                actual_batch_size=actual_batch,
                sequence_length=self.config.sequence_length,
                effective_vocab_size=effective_vocab,
                dtype_name=estimated_dense_logits_dtype,
            )
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
                "estimated_dense_logits_bytes": estimated_dense_logits_bytes,
                "estimated_reducer_workspace_bytes": (
                    _estimate_reducer_workspace_bytes(
                        actual_batch_size=actual_batch,
                        sequence_length=self.config.sequence_length,
                        effective_vocab_size=effective_vocab,
                        vocab_chunk_size=chunking_plan.effective_size,
                        dtype_name=estimated_dense_logits_dtype,
                    )
                ),
                "estimated_reducer_workspace_formula": (
                    "batch*sequence*effective_vocab_or_chunk*bytes_per_logit*"
                    f"{_REDUCER_WORKSPACE_FACTOR}"
                ),
                "estimated_reducer_workspace_is_measured": False,
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
                raise TeacherBackendDependencyError(_missing_torch_message())
            raise TeacherBackendUnavailableError(
                _no_accelerator_message(self.config.fallback_policy)
            )
        try:
            torch = import_module("torch")
        except ImportError as exc:
            raise TeacherBackendDependencyError(_missing_torch_message()) from exc
        try:
            transformers = import_module("transformers")
        except ImportError as exc:
            raise TeacherBackendDependencyError(
                _missing_transformers_message()
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
            raise TeacherBackendModelLoadError(
                _model_load_message(
                    self.config,
                    tokenizer_id=tokenizer_id,
                    local_files_only=local_files_only,
                )
            ) from exc
        if getattr(tokenizer, "pad_token_id", None) is None:
            eos_token = getattr(tokenizer, "eos_token", None)
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            if eos_token is not None and eos_token_id is not None:
                tokenizer.pad_token = eos_token
        model.eval()
        if hasattr(model, "to"):
            try:
                model = model.to(device.device)
            except Exception as exc:
                raise _wrap_device_error("model.to(device)", exc, device) from exc
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        return torch, tokenizer, model, device


def check_gpu_torch_backend_available(config: TeacherBackendConfig) -> bool:
    return GPUTorchTeacherEmissionBackend.available(config)


def diagnose_gpu_torch_backend(config: TeacherBackendConfig) -> dict[str, object]:
    tokenizer_id = _effective_tokenizer_id(config)
    local_files_only = config.local_files_only or not config.allow_downloads
    diagnostics = _base_diagnostics(
        config,
        tokenizer_id=tokenizer_id,
        local_files_only=local_files_only,
    )
    try:
        _validate_gpu_torch_config(config)
    except TeacherBackendUnsupportedPolicyError as exc:
        diagnostics.update(
            {
                "can_emit": False,
                "failure_stage": "unsupported_target",
                "failure_reason": str(exc),
            }
        )
        return diagnostics
    except ValueError as exc:
        diagnostics.update(
            {
                "can_emit": False,
                "failure_stage": "invalid_config",
                "failure_reason": str(exc),
            }
        )
        return diagnostics

    device = detect_torch_accelerator()
    diagnostics.update(_device_diagnostics(device))
    if not device.available:
        if device.reason == "missing torch":
            diagnostics.update(
                {
                    "dependency_status": "missing_torch",
                    "torch_available": False,
                    "can_emit": False,
                    "failure_stage": "missing_dependency",
                    "failure_reason": _failure_reason(
                        _missing_torch_message(),
                        config.fallback_policy,
                    ),
                }
            )
            return diagnostics
        diagnostics.update(
            {
                "can_emit": False,
                "failure_stage": "no_accelerator",
                "failure_reason": _failure_reason(
                    _no_accelerator_message(config.fallback_policy),
                    config.fallback_policy,
                ),
            }
        )
        return diagnostics

    try:
        import_module("torch")
    except ImportError:
        diagnostics.update(
            {
                "dependency_status": "missing_torch",
                "torch_available": False,
                "can_emit": False,
                "failure_stage": "missing_dependency",
                "failure_reason": _failure_reason(
                    _missing_torch_message(),
                    config.fallback_policy,
                ),
            }
        )
        return diagnostics
    diagnostics["torch_available"] = True

    try:
        transformers = import_module("transformers")
    except ImportError:
        diagnostics.update(
            {
                "dependency_status": "missing_transformers",
                "transformers_available": False,
                "can_emit": False,
                "failure_stage": "missing_dependency",
                "failure_reason": _failure_reason(
                    _missing_transformers_message(),
                    config.fallback_policy,
                ),
            }
        )
        return diagnostics
    diagnostics.update(
        {
            "dependency_status": "ok",
            "transformers_available": True,
        }
    )

    try:
        transformers.AutoTokenizer.from_pretrained(
            tokenizer_id,
            local_files_only=local_files_only,
        )
        diagnostics["tokenizer_available"] = True
        transformers.AutoModelForCausalLM.from_pretrained(
            config.model_id,
            local_files_only=local_files_only,
        )
        diagnostics["model_available"] = True
    except Exception:
        diagnostics.update(
            {
                "can_emit": False,
                "failure_stage": "model_load",
                "failure_reason": _failure_reason(
                    _model_load_message(
                        config,
                        tokenizer_id=tokenizer_id,
                        local_files_only=local_files_only,
                    ),
                    config.fallback_policy,
                ),
            }
        )
        return diagnostics

    diagnostics.update(
        {
            "can_emit": True,
            "failure_stage": "none",
            "failure_reason": None,
        }
    )
    return diagnostics


def _validate_gpu_torch_config(config: TeacherBackendConfig) -> None:
    if config.backend_id != GPUTorchTeacherEmissionBackend.backend_id:
        raise ValueError(
            f"gpu_torch requires backend_id="
            f"{GPUTorchTeacherEmissionBackend.backend_id!r}, "
            f"got {config.backend_id!r}"
        )
    if config.runtime_mode != "cpu_gpu":
        raise ValueError("gpu_torch supports only runtime_mode='cpu_gpu'")
    if config.target_policy not in _SUPPORTED_EMISSION_POLICIES:
        raise TeacherBackendUnsupportedPolicyError(
            "gpu_torch target_policy "
            f"{config.target_policy!r} is not implemented; supported policies "
            "are dense_logits, topk_with_tail_v0, and cascaded_soft_labels_v1"
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
    if config.gpu_vocab_chunk_size is not None and config.gpu_vocab_chunk_size <= 0:
        raise ValueError("gpu_vocab_chunk_size must be None or > 0")


def _base_diagnostics(
    config: TeacherBackendConfig,
    *,
    tokenizer_id: str,
    local_files_only: bool,
) -> dict[str, object]:
    chunk_fields = _diagnostic_chunking_fields(config)
    return {
        "backend_id": "gpu_torch",
        "runtime_mode": config.runtime_mode,
        "target_policy": config.target_policy,
        "dependency_status": "unknown",
        "torch_available": False,
        "transformers_available": False,
        "accelerator_available": False,
        "device_kind": "unavailable",
        "torch_device": None,
        "cuda_available": False,
        "mps_available": False,
        "model_available": False,
        "tokenizer_available": False,
        "model_id": config.model_id,
        "tokenizer_id": tokenizer_id,
        "local_files_only": local_files_only,
        "allow_downloads": config.allow_downloads,
        "can_emit": False,
        "failure_reason": None,
        "failure_stage": "none",
        "fallback_policy": config.fallback_policy,
        "fallback_used": False,
        "fallback_allowed": config.fallback_policy == "auto",
        "fallback_handled_by": "none",
        **chunk_fields,
    }


def _device_diagnostics(device: TorchAcceleratorDetection) -> dict[str, object]:
    return {
        "torch_available": device.reason != "missing torch",
        "accelerator_available": device.available,
        "device_kind": device.device_kind,
        "torch_device": device.device,
        "cuda_available": device.cuda_available,
        "mps_available": device.mps_available,
        "torch_version": device.torch_version,
    }


def _diagnostic_chunking_fields(
    config: TeacherBackendConfig,
) -> dict[str, object]:
    if config.target_policy in _SUPPORTED_EMISSION_POLICIES and (
        config.gpu_vocab_chunk_size is None or config.gpu_vocab_chunk_size > 0
    ):
        chunking_plan = _vocab_chunking_plan(config, config.target_policy)
        effective_size = chunking_plan.effective_size
        used = chunking_plan.used
    else:
        effective_size = None
        used = False
    return {
        "gpu_enable_vocab_chunking": config.gpu_enable_vocab_chunking,
        "gpu_vocab_chunk_size_requested": config.gpu_vocab_chunk_size,
        "gpu_vocab_chunk_size_effective": effective_size,
        "vocab_chunking_used": used,
    }


def _missing_torch_message() -> str:
    return (
        "gpu_torch backend requires optional dependency torch. Install the "
        "teacher-hf extra to use torch/transformers emission."
    )


def _missing_transformers_message() -> str:
    return (
        "gpu_torch backend requires optional dependency transformers. Install "
        "the teacher-hf extra to use torch/transformers emission."
    )


def _no_accelerator_message(fallback_policy: str) -> str:
    message = (
        "gpu_torch requires runtime_mode='cpu_gpu' with an available cuda or "
        "mps Torch accelerator; no CPU fallback is used."
    )
    if fallback_policy == "auto":
        message += " fallback_policy='auto' must be handled by an orchestrator."
    return message


def _model_load_message(
    config: TeacherBackendConfig,
    *,
    tokenizer_id: str,
    local_files_only: bool,
) -> str:
    return (
        "gpu_torch backend could not load torch/transformers model or tokenizer "
        f"for model_id={config.model_id!r}, tokenizer_id={tokenizer_id!r}, "
        f"local_files_only={local_files_only!r}, "
        f"allow_downloads={config.allow_downloads!r}. Install the teacher-hf "
        "extra and provide local model files, or set allow_downloads only in "
        "an explicitly network-enabled run."
    )


def _failure_reason(message: str, fallback_policy: str) -> str:
    if fallback_policy != "auto":
        return message
    return f"{message} Orchestrator fallback is required; gpu_torch did not fall back."


def _wrap_device_error(
    stage: str,
    exc: Exception,
    device: TorchAcceleratorDetection,
) -> TeacherBackendDeviceError:
    message = (
        f"gpu_torch {stage} failed on device_kind={device.device_kind!r}, "
        f"torch_device={device.device!r}: out of memory or device failure; "
        "no CPU fallback is used."
    )
    original = str(exc).lower()
    if device.device_kind == "mps" and (
        "unsupported" in original or "not implemented" in original
    ):
        message += " mps unsupported operation."
    return TeacherBackendDeviceError(message)


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
    vocab_chunk_size: int | None = None,
) -> dict[str, Any]:
    workspace = _gpu_probability_workspace(
        torch,
        logits,
        vocab_chunk_size=vocab_chunk_size,
    )
    return _gpu_topk_tail_reduce_from_workspace(
        torch,
        workspace,
        top_k=top_k,
    )


def _gpu_topk_tail_reduce_from_workspace(
    torch: Any,
    workspace: dict[str, Any],
    *,
    top_k: int,
) -> dict[str, Any]:
    logits = workspace["logits"]
    effective_top_k = min(top_k, int(logits.shape[-1]))
    if workspace["chunked"]:
        top_log_prob_chunks = []
        top_token_id_chunks = []
        entropy = None
        for start, stop in _chunk_slices(
            effective_vocab_size=int(logits.shape[-1]),
            vocab_chunk_size=int(workspace["vocab_chunk_size"]),
        ):
            chunk_log_probs = logits[..., start:stop] - workspace["logsumexp"]
            chunk_probs = torch.exp(chunk_log_probs)
            chunk_top_k = min(effective_top_k, stop - start)
            chunk_top_log_probs, chunk_top_token_ids = torch.topk(
                chunk_log_probs,
                k=chunk_top_k,
                dim=-1,
            )
            top_log_prob_chunks.append(chunk_top_log_probs)
            top_token_id_chunks.append(chunk_top_token_ids + start)
            contribution = -torch.sum(chunk_probs * chunk_log_probs, dim=-1)
            entropy = contribution if entropy is None else entropy + contribution
        candidate_log_probs = torch.cat(top_log_prob_chunks, dim=-1)
        candidate_token_ids = torch.cat(top_token_id_chunks, dim=-1)
        top_log_probs, candidate_indices = torch.topk(
            candidate_log_probs,
            k=effective_top_k,
            dim=-1,
        )
        top_token_ids = torch.gather(candidate_token_ids, -1, candidate_indices)
        top_probs = torch.exp(top_log_probs)
        top_mass = torch.sum(top_probs, dim=-1)
        tail_mass = torch.clamp(1.0 - top_mass, min=0.0, max=1.0)
        return {
            "top_token_ids": top_token_ids,
            "top_log_probs": top_log_probs,
            "top_probs": top_probs,
            "top_mass": top_mass,
            "tail_mass": tail_mass,
            "teacher_entropy": entropy,
        }
    log_probs = workspace["log_probs"]
    probs = workspace["probs"]
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
    vocab_chunk_size: int | None = None,
) -> dict[str, Any]:
    workspace = _gpu_probability_workspace(
        torch,
        logits,
        vocab_chunk_size=vocab_chunk_size,
    )
    payload = _gpu_topk_tail_reduce_from_workspace(torch, workspace, top_k=top_k)
    payload["bucket_masses"] = _tail_bucket_masses_on_device(
        torch,
        _probs_from_workspace(torch, workspace),
        top_token_ids=payload["top_token_ids"],
        num_buckets=num_buckets,
    )
    return payload


def _gpu_probability_workspace(
    torch: Any,
    logits: Any,
    *,
    vocab_chunk_size: int | None,
) -> dict[str, Any]:
    if vocab_chunk_size is None:
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return {
            "chunked": False,
            "logits": logits,
            "log_probs": log_probs,
            "probs": torch.exp(log_probs),
        }
    logsumexp = None
    for start, stop in _chunk_slices(
        effective_vocab_size=int(logits.shape[-1]),
        vocab_chunk_size=vocab_chunk_size,
    ):
        chunk_logsumexp = torch.logsumexp(logits[..., start:stop], dim=-1, keepdim=True)
        logsumexp = (
            chunk_logsumexp
            if logsumexp is None
            else torch.logaddexp(logsumexp, chunk_logsumexp)
        )
    return {
        "chunked": True,
        "logits": logits,
        "logsumexp": logsumexp,
        "vocab_chunk_size": vocab_chunk_size,
    }


def _probs_from_workspace(torch: Any, workspace: dict[str, Any]) -> Any:
    if not workspace["chunked"]:
        return workspace["probs"]
    logits = workspace["logits"]
    chunks = []
    for start, stop in _chunk_slices(
        effective_vocab_size=int(logits.shape[-1]),
        vocab_chunk_size=int(workspace["vocab_chunk_size"]),
    ):
        chunks.append(torch.exp(logits[..., start:stop] - workspace["logsumexp"]))
    return torch.cat(chunks, dim=-1)


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


def _estimate_reducer_workspace_bytes(
    *,
    actual_batch_size: int,
    sequence_length: int,
    effective_vocab_size: int,
    vocab_chunk_size: int | None,
    dtype_name: str,
) -> int:
    effective_axis = (
        effective_vocab_size
        if vocab_chunk_size is None
        else min(vocab_chunk_size, effective_vocab_size)
    )
    return (
        actual_batch_size
        * sequence_length
        * effective_axis
        * _dtype_nbytes(dtype_name)
        * _REDUCER_WORKSPACE_FACTOR
    )


def _vocab_chunking_plan(
    config: TeacherBackendConfig,
    target_policy: str,
) -> VocabChunkingPlan:
    requested_size = config.gpu_vocab_chunk_size
    if not config.gpu_enable_vocab_chunking:
        return VocabChunkingPlan(
            requested=False,
            requested_size=requested_size,
            effective_size=None,
            used=False,
            reason=None,
        )
    if target_policy == "dense_logits":
        return VocabChunkingPlan(
            requested=True,
            requested_size=requested_size,
            effective_size=None,
            used=False,
            reason="disabled_for_dense_debug",
        )
    if target_policy == "cascaded_soft_labels_v1":
        return VocabChunkingPlan(
            requested=True,
            requested_size=requested_size,
            effective_size=None,
            used=False,
            reason="exact_bucket_policy_requires_full_probability_workspace",
        )
    return VocabChunkingPlan(
        requested=True,
        requested_size=requested_size,
        effective_size=requested_size or _DEFAULT_GPU_VOCAB_CHUNK_SIZE,
        used=True,
        reason=None,
    )


def _vocab_chunks_per_batch(
    *,
    effective_vocab_size: int,
    vocab_chunk_size: int | None,
) -> int:
    if vocab_chunk_size is None:
        return 1
    return max(1, (effective_vocab_size + vocab_chunk_size - 1) // vocab_chunk_size)


def _chunk_slices(
    *,
    effective_vocab_size: int,
    vocab_chunk_size: int,
) -> list[tuple[int, int]]:
    return [
        (start, min(start + vocab_chunk_size, effective_vocab_size))
        for start in range(0, effective_vocab_size, vocab_chunk_size)
    ]


def _compact_payload_fields(target_policy: str) -> list[str]:
    fields = [
        "top_token_ids",
        "top_log_probs",
        "top_probs",
        "top_mass",
        "tail_mass",
    ]
    if target_policy == "cascaded_soft_labels_v1":
        fields.append("bucket_masses")
    fields.append("teacher_entropy")
    return fields


def _gpu_reduction_mode(
    target_policy: str,
    *,
    vocab_chunking_used: bool,
) -> str:
    if target_policy == "cascaded_soft_labels_v1":
        return (
            "compact_cascaded_soft_labels_chunked"
            if vocab_chunking_used
            else "compact_cascaded_soft_labels"
        )
    return "compact_topk_tail_chunked" if vocab_chunking_used else "compact_topk_tail"


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
