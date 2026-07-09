from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from importlib import import_module
from typing import Any

import numpy as np

from radjax_tome.backends.base import (
    BackendCapability,
    TeacherBackendConfig,
    TeacherBatchInput,
    TeacherEmissionResult,
    resolve_exemplar_capture_policy,
    resolve_gpu_batch_size_policy,
    validate_gpu_batch_size_policy_config,
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
    "dynamic_cascaded_soft_labels_v1",
    "corridor_exemplar_v1",
}
_CAPABILITY_STATUS = {
    "dense_logits": "supported_debug",
    "topk_with_tail_v0": "optimized",
    "cascaded_soft_labels_v1": "optimized",
    "dynamic_cascaded_soft_labels_v1": "optimized",
    "corridor_exemplar_v1": "optimized",
}
_DEFAULT_GPU_VOCAB_CHUNK_SIZE = 8192
_REDUCER_WORKSPACE_FACTOR = 3
_DYNAMIC_TOP_K_POLICY = "mass_threshold_v1"
_TAIL_BUCKET_POLICY = "contiguous_descending_tail_probability_mass"
_CORRIDOR_PAYLOAD_FLAVOR = "production_v1"
_EXEMPLAR_SELECTION_POLICY = "entropy_top_n_v1"
_CORRIDOR_POLICY = "production_corridor_records_v1"
_MODE_RECORD_POLICY = "top_mode_summary_v1"
_FINGERPRINT_TOPOLOGY_POLICY = "sequence_position_v1"
_CORRIDOR_CONFIDENCE_POLICY = "top_probability_v1"
_EXEMPLAR_CAPTURE_MODE = "one_pass_candidate"
_EXEMPLAR_CAPTURE_MODES = {"auto", "one_pass_candidate", "two_pass_sparse_exemplar"}
_EXEMPLAR_FIRST_PASS_SCORE_POLICY = "entropy_score_v1"
_EXEMPLAR_SOURCE_POLICIES = {
    "dense_logits": 1,
    "cascaded_soft_labels_v1": 2,
    "dynamic_cascaded_soft_labels_v1": 3,
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
                status="optimized",
                optimized=True,
                implemented_now=True,
                notes=(
                    "Spec 3.3F9 implements optimized gpu_torch "
                    "corridor/exemplar emission against the F8 production "
                    "schema and transfers only compact production arrays back "
                    "to host."
                ),
            ),
            BackendCapability(
                backend_id=self.backend_id,
                backend_family=self.backend_family,
                runtime_mode="cpu_gpu",
                target_policy="dynamic_cascaded_soft_labels_v1",
                status="optimized",
                optimized=True,
                implemented_now=True,
                notes=(
                    "Spec 3.3F7 gpu_torch computes dynamic top-k explicit "
                    "head plus bucketed tail on the selected CUDA or MPS "
                    "accelerator and transfers only compact payload arrays "
                    "back to host."
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
        elif self.config.target_policy == "dynamic_cascaded_soft_labels_v1":
            try:
                compact_payload = _gpu_dynamic_cascaded_reduce(
                    torch,
                    logits,
                    dynamic_top_k_min=self.config.dynamic_top_k_min,
                    dynamic_top_k_max=self.config.dynamic_top_k_max,
                    dynamic_mass_threshold=self.config.dynamic_mass_threshold,
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
        elif self.config.target_policy == "corridor_exemplar_v1":
            try:
                if (
                    _effective_exemplar_capture_mode(
                        self.config,
                        actual_batch_size=len(batch.texts),
                        effective_vocab_size=effective_vocab_size,
                    )
                    == "two_pass_sparse_exemplar"
                ):
                    compact_payload = _gpu_corridor_exemplar_score_reduce(
                        torch,
                        logits,
                        config=self.config,
                        vocab_chunk_size=chunking_plan.effective_size,
                    )
                else:
                    compact_payload = _gpu_corridor_exemplar_reduce(
                        torch,
                        logits,
                        config=self.config,
                        vocab_chunk_size=chunking_plan.effective_size,
                    )
            except Exception as exc:
                raise _wrap_device_error(
                    "corridor/exemplar compact reduction", exc, device
                ) from exc
            try:
                payload = _compact_payload_to_numpy(compact_payload)
                if (
                    _effective_exemplar_capture_mode(
                        self.config,
                        actual_batch_size=len(batch.texts),
                        effective_vocab_size=effective_vocab_size,
                    )
                    == "two_pass_sparse_exemplar"
                ):
                    payload.update(
                        _gpu_corridor_score_records(
                            self.config,
                            payload,
                            effective_vocab_size=effective_vocab_size,
                        )
                    )
                else:
                    payload.update(
                        _gpu_corridor_payload_records(
                            self.config,
                            payload,
                            effective_vocab_size=effective_vocab_size,
                        )
                    )
            except Exception as exc:
                raise _wrap_device_error(
                    "corridor/exemplar compact tensor transfer to CPU",
                    exc,
                    device,
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
        metadata.update(
            resolve_gpu_batch_size_policy(
                self.config,
                payload=compact_payload,
            )
        )
        if chunking_plan.reason is not None:
            metadata["vocab_chunking_reason"] = chunking_plan.reason
        if self.config.target_policy in {
            "topk_with_tail_v0",
            "cascaded_soft_labels_v1",
            "dynamic_cascaded_soft_labels_v1",
            "corridor_exemplar_v1",
        }:
            compact_payload = compact_payload or {}
            compact_fields = _compact_payload_fields(
                self.config.target_policy,
                exemplar_capture_mode=_effective_exemplar_capture_mode(
                    self.config,
                    actual_batch_size=actual_batch,
                    effective_vocab_size=effective_vocab,
                ),
            )
            gpu_reduction_mode = _gpu_reduction_mode(
                self.config.target_policy,
                exemplar_capture_mode=_effective_exemplar_capture_mode(
                    self.config,
                    actual_batch_size=actual_batch,
                    effective_vocab_size=effective_vocab,
                ),
                vocab_chunking_used=chunking_plan.used,
            )
            estimated_dense_logits_bytes = _estimate_dense_logits_bytes(
                actual_batch_size=actual_batch,
                sequence_length=self.config.sequence_length,
                effective_vocab_size=effective_vocab,
                dtype_name=estimated_dense_logits_dtype,
            )
            compact_metadata = {
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
            if self.config.target_policy in {
                "topk_with_tail_v0",
                "cascaded_soft_labels_v1",
            }:
                effective_top_k = min(self.config.top_k, effective_vocab)
                compact_metadata.update(
                    {
                        "requested_top_k": self.config.top_k,
                        "effective_top_k": effective_top_k,
                    }
                )
            if self.config.target_policy == "cascaded_soft_labels_v1":
                compact_metadata.update(
                    {
                        "num_buckets": self.config.num_buckets,
                        "bucket_policy": _TAIL_BUCKET_POLICY,
                    }
                )
            if self.config.target_policy == "dynamic_cascaded_soft_labels_v1":
                compact_metadata.update(
                    _dynamic_cascaded_metadata(
                        self.config,
                        effective_vocab_size=effective_vocab,
                        compact_payload=compact_payload,
                    )
                )
            if self.config.target_policy == "corridor_exemplar_v1":
                compact_metadata.update(
                    _gpu_corridor_metadata(
                        self.config,
                        compact_payload=compact_payload,
                        actual_batch_size=actual_batch,
                        effective_vocab_size=effective_vocab,
                    )
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
            "are dense_logits, topk_with_tail_v0, cascaded_soft_labels_v1, "
            "dynamic_cascaded_soft_labels_v1, and corridor_exemplar_v1"
        )
    if config.sequence_length <= 0:
        raise ValueError("sequence_length must be > 0")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    validate_gpu_batch_size_policy_config(config)
    if config.vocab_size <= 0:
        raise ValueError("vocab_size must be > 0")
    if config.top_k <= 0:
        raise ValueError("top_k must be > 0")
    if config.num_buckets <= 0:
        raise ValueError("num_buckets must be > 0")
    if config.exemplar_top_n <= 0:
        raise ValueError("exemplar_top_n must be > 0")
    if config.exemplar_source_policy not in _EXEMPLAR_SOURCE_POLICIES:
        raise ValueError(
            "exemplar_source_policy must be dense_logits, "
            "cascaded_soft_labels_v1, or dynamic_cascaded_soft_labels_v1"
        )
    if config.exemplar_selection_policy != _EXEMPLAR_SELECTION_POLICY:
        raise ValueError("exemplar_selection_policy must be 'entropy_top_n_v1'")
    if config.exemplar_capture_mode not in _EXEMPLAR_CAPTURE_MODES:
        raise ValueError(
            "exemplar_capture_mode must be 'auto', 'one_pass_candidate', "
            "or 'two_pass_sparse_exemplar'"
        )
    if config.exemplar_first_pass_score_policy != _EXEMPLAR_FIRST_PASS_SCORE_POLICY:
        raise ValueError("exemplar_first_pass_score_policy must be 'entropy_score_v1'")
    if config.exemplar_second_pass_source_policy not in _EXEMPLAR_SOURCE_POLICIES:
        raise ValueError(
            "exemplar_second_pass_source_policy must be dense_logits, "
            "cascaded_soft_labels_v1, or dynamic_cascaded_soft_labels_v1"
        )
    if config.exemplar_sparse_selection_top_n < 1:
        raise ValueError("exemplar_sparse_selection_top_n must be >= 1")
    if config.exemplar_sparse_selection_fraction is not None and not (
        0.0 < config.exemplar_sparse_selection_fraction <= 1.0
    ):
        raise ValueError(
            "exemplar_sparse_selection_fraction must be None or > 0 and <= 1"
        )
    if config.exemplar_auto_num_examples is not None and (
        config.exemplar_auto_num_examples < 1
    ):
        raise ValueError("exemplar_auto_num_examples must be None or >= 1")
    if config.exemplar_auto_expected_selected_fraction is not None and not (
        0.0 < config.exemplar_auto_expected_selected_fraction <= 1.0
    ):
        raise ValueError(
            "exemplar_auto_expected_selected_fraction must be None or > 0 and <= 1"
        )
    if config.exemplar_auto_available_disk_budget_bytes is not None and (
        config.exemplar_auto_available_disk_budget_bytes < 1
    ):
        raise ValueError(
            "exemplar_auto_available_disk_budget_bytes must be None or >= 1"
        )
    if config.corridor_payload_flavor != _CORRIDOR_PAYLOAD_FLAVOR:
        raise ValueError("corridor_payload_flavor must be 'production_v1'")
    if config.gpu_vocab_chunk_size is not None and config.gpu_vocab_chunk_size <= 0:
        raise ValueError("gpu_vocab_chunk_size must be None or > 0")
    if config.dynamic_top_k_min < 1:
        raise ValueError("dynamic_top_k_min must be >= 1")
    if config.dynamic_top_k_max < config.dynamic_top_k_min:
        raise ValueError("dynamic_top_k_max must be >= dynamic_top_k_min")
    if not 0.0 < config.dynamic_mass_threshold <= 1.0:
        raise ValueError("dynamic_mass_threshold must be > 0 and <= 1")
    if config.dynamic_top_k_policy != _DYNAMIC_TOP_K_POLICY:
        raise ValueError("dynamic_top_k_policy must be 'mass_threshold_v1'")


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
        "gpu-teacher extra, or the equivalent teacher-hf extra, to use "
        "torch/transformers GPU emission."
    )


def _missing_transformers_message() -> str:
    return (
        "gpu_torch backend requires optional dependency transformers. Install "
        "the gpu-teacher extra, or the equivalent teacher-hf extra, to use "
        "torch/transformers GPU emission."
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
        f"allow_downloads={config.allow_downloads!r}. Install the gpu-teacher "
        "extra, or the equivalent teacher-hf extra, and provide/cache local "
        "model files. RADJAX-Tome did not download a model."
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


def _dynamic_cascaded_metadata(
    config: TeacherBackendConfig,
    *,
    effective_vocab_size: int,
    compact_payload: dict[str, np.ndarray],
) -> dict[str, object]:
    effective_max_k = min(config.dynamic_top_k_max, effective_vocab_size)
    effective_min_k = min(config.dynamic_top_k_min, effective_max_k)
    effective_top_k = compact_payload.get("effective_top_k")
    if effective_top_k is None or effective_top_k.size == 0:
        observed_min = effective_min_k
        observed_max = effective_min_k
        observed_mean = float(effective_min_k)
    else:
        observed_min = int(np.min(effective_top_k))
        observed_max = int(np.max(effective_top_k))
        observed_mean = float(np.mean(effective_top_k, dtype=np.float32))
    return {
        "dynamic_top_k_policy": config.dynamic_top_k_policy,
        "dynamic_top_k_min_configured": config.dynamic_top_k_min,
        "dynamic_top_k_max_configured": config.dynamic_top_k_max,
        "dynamic_top_k_min_effective": effective_min_k,
        "dynamic_top_k_max_effective": effective_max_k,
        "dynamic_mass_threshold": config.dynamic_mass_threshold,
        "num_buckets": config.num_buckets,
        "bucket_policy": _TAIL_BUCKET_POLICY,
        "padding_policy": "pad_to_dynamic_top_k_max_effective",
        "masked_slot_policy": (
            "masked top slots use token_id=0, top_prob=0.0, "
            "top_log_prob=0.0 and must be ignored"
        ),
        "selection_mask_field": "top_selection_mask",
        "top_selection_mask_semantics": "true means explicit selected token",
        "dynamic_head_selection_vectorized": True,
        "dynamic_tail_bucket_vectorized": True,
        "effective_top_k_min_observed": observed_min,
        "effective_top_k_max_observed": observed_max,
        "effective_top_k_mean_observed": observed_mean,
    }


def _gpu_corridor_metadata(
    config: TeacherBackendConfig,
    *,
    compact_payload: dict[str, np.ndarray],
    actual_batch_size: int | None,
    effective_vocab_size: int,
) -> dict[str, object]:
    policy = resolve_exemplar_capture_policy(
        config,
        actual_batch_size=actual_batch_size,
        effective_vocab_size=effective_vocab_size,
    )
    effective_mode = str(policy["exemplar_capture_mode_effective"])
    if effective_mode == "two_pass_sparse_exemplar":
        source_summary = _gpu_corridor_source_summary(
            _second_pass_source_config(config),
            compact_payload=compact_payload,
            effective_vocab_size=effective_vocab_size,
        )
        return {
            "schema_version": "corridor_exemplar_score_pass_v1",
            "corridor_payload_flavor": config.corridor_payload_flavor,
            "production_corridor_schema": False,
            "historical_parity_claimed": False,
            "historical_reference_source": "gpu_torch_score_pass",
            "exemplar_source_policy": config.exemplar_source_policy,
            "exemplar_first_pass_score_policy": (
                config.exemplar_first_pass_score_policy
            ),
            "exemplar_second_pass_source_policy": (
                config.exemplar_second_pass_source_policy
            ),
            "score_source_policy": config.exemplar_second_pass_source_policy,
            "exemplar_selection_policy": config.exemplar_selection_policy,
            **_gpu_corridor_score_pass_metadata(config, policy=policy),
            "requested_exemplar_top_n": config.exemplar_top_n,
            "effective_exemplar_top_n": min(
                config.exemplar_top_n,
                config.sequence_length,
            ),
            "requested_exemplar_sparse_selection_top_n": (
                config.exemplar_sparse_selection_top_n
            ),
            "exemplar_sparse_selection_fraction": (
                config.exemplar_sparse_selection_fraction
            ),
            "score_records": int(compact_payload["score_example_ids"].shape[0])
            if "score_example_ids" in compact_payload
            else 0,
            "source_policy_kind": source_summary.get("source_policy_kind"),
            "source_policy_uses_bucketed_tail": source_summary.get(
                "source_policy_uses_bucketed_tail"
            ),
            "source_policy_dynamic_top_k": source_summary.get(
                "source_policy_dynamic_top_k"
            ),
        }
    source_summary = _gpu_corridor_source_summary(
        config,
        compact_payload=compact_payload,
        effective_vocab_size=effective_vocab_size,
    )
    return {
        **_gpu_corridor_schema_metadata(config),
        **_gpu_corridor_capture_mode_metadata(config, policy=policy),
        **source_summary,
        "requested_exemplar_top_n": config.exemplar_top_n,
        "effective_exemplar_top_n": min(config.exemplar_top_n, config.sequence_length),
        "exemplar_records": min(config.exemplar_top_n, config.sequence_length),
    }


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


def _gpu_dynamic_cascaded_reduce(
    torch: Any,
    logits: Any,
    *,
    dynamic_top_k_min: int,
    dynamic_top_k_max: int,
    dynamic_mass_threshold: float,
    num_buckets: int,
    vocab_chunk_size: int | None = None,
) -> dict[str, Any]:
    workspace = _gpu_probability_workspace(
        torch,
        logits,
        vocab_chunk_size=vocab_chunk_size,
    )
    probs = _probs_from_workspace(torch, workspace)
    log_probs = torch.log(torch.clamp(probs, min=torch.finfo(probs.dtype).tiny))
    effective_max_k = min(dynamic_top_k_max, int(logits.shape[-1]))
    effective_min_k = min(dynamic_top_k_min, effective_max_k)
    head_payload = _dynamic_cascaded_head_from_probs(
        torch,
        probs,
        log_probs,
        dynamic_top_k_min=effective_min_k,
        dynamic_top_k_max=effective_max_k,
        dynamic_mass_threshold=dynamic_mass_threshold,
    )
    bucket_masses = _dynamic_bucket_masses_from_sorted_tail(
        torch,
        head_payload["sorted_probs"],
        effective_top_k=head_payload["effective_top_k"],
        num_buckets=num_buckets,
    )
    tail_mass = torch.clamp(1.0 - head_payload["top_mass"], min=0.0, max=1.0)
    teacher_entropy = -torch.sum(probs * log_probs, dim=-1)
    return {
        "top_token_ids": head_payload["top_token_ids"],
        "top_log_probs": head_payload["top_log_probs"],
        "top_probs": head_payload["top_probs"],
        "top_selection_mask": head_payload["top_selection_mask"],
        "effective_top_k": head_payload["effective_top_k"],
        "top_mass": head_payload["top_mass"],
        "tail_mass": tail_mass,
        "bucket_masses": bucket_masses,
        "teacher_entropy": teacher_entropy,
    }


def _gpu_corridor_exemplar_reduce(
    torch: Any,
    logits: Any,
    *,
    config: TeacherBackendConfig,
    vocab_chunk_size: int | None = None,
) -> dict[str, Any]:
    source = _gpu_corridor_source_payload(
        torch,
        logits,
        config=config,
        vocab_chunk_size=vocab_chunk_size,
    )
    teacher_entropy = source["teacher_entropy"]
    effective_exemplar_top_n = min(config.exemplar_top_n, int(logits.shape[1]))
    exemplar_scores, exemplar_positions = torch.topk(
        teacher_entropy,
        k=effective_exemplar_top_n,
        dim=-1,
    )
    exemplar_selection_mask = torch.zeros(
        teacher_entropy.shape,
        dtype=torch.int32,
        device=teacher_entropy.device,
    )
    exemplar_selection_mask.scatter_(
        -1,
        exemplar_positions,
        torch.ones_like(exemplar_positions, dtype=torch.int32),
    )
    score_selected_position = torch.argmax(teacher_entropy, dim=-1).to(torch.int32)
    gather_positions = score_selected_position.to(torch.int64).unsqueeze(-1)
    score_selected_entropy = torch.gather(
        teacher_entropy,
        -1,
        gather_positions,
    ).squeeze(-1)
    score_selected_confidence = torch.gather(
        source["top_probs"][..., 0],
        -1,
        gather_positions,
    ).squeeze(-1)
    corridor_top_token_ids = source["top_token_ids"][..., 0]
    score_top_token_id = torch.gather(
        corridor_top_token_ids,
        -1,
        gather_positions,
    ).squeeze(-1)
    corridor_top_probs = source["top_probs"][..., 0]
    batch_size = int(logits.shape[0])
    sequence_length = int(logits.shape[1])
    policy_id = _EXEMPLAR_SOURCE_POLICIES[config.exemplar_source_policy]
    return {
        "corridor_top_token_ids": corridor_top_token_ids,
        "corridor_top_probs": corridor_top_probs,
        "corridor_teacher_entropy": teacher_entropy,
        "corridor_confidence": corridor_top_probs,
        "corridor_lengths": torch.full(
            (batch_size,),
            sequence_length,
            dtype=torch.int32,
            device=logits.device,
        ),
        "exemplar_positions": exemplar_positions,
        "exemplar_scores": exemplar_scores,
        "exemplar_selection_mask": exemplar_selection_mask,
        "exemplar_source_policy_ids": torch.full(
            teacher_entropy.shape,
            policy_id,
            dtype=torch.int32,
            device=logits.device,
        ),
        "exemplar_source_top_token_ids": source["top_token_ids"],
        "exemplar_source_top_log_probs": source["top_log_probs"],
        "exemplar_source_top_probs": source["top_probs"],
        "exemplar_source_top_selection_mask": source.get(
            "top_selection_mask",
            torch.ones_like(source["top_token_ids"], dtype=torch.bool),
        ),
        "exemplar_source_effective_top_k": source["effective_top_k"],
        "exemplar_source_top_mass": source["top_mass"],
        "exemplar_source_tail_mass": source["tail_mass"],
        "exemplar_source_bucket_masses": source.get(
            "bucket_masses",
            torch.zeros(
                (*teacher_entropy.shape, config.num_buckets),
                dtype=torch.float32,
                device=logits.device,
            ),
        ),
        "score_example_ids": torch.arange(
            batch_size,
            dtype=torch.int32,
            device=logits.device,
        ),
        "score_max_entropy": torch.max(teacher_entropy, dim=-1).values,
        "score_mean_entropy": torch.mean(teacher_entropy, dim=-1),
        "score_selected_position": score_selected_position,
        "score_top_token_id": score_top_token_id.to(torch.int32),
        "score_selected_position_entropy": score_selected_entropy,
        "score_confidence_at_selected_position": score_selected_confidence,
        "score_source_policy_ids": torch.full(
            (batch_size,),
            policy_id,
            dtype=torch.int32,
            device=logits.device,
        ),
        "score_lengths": torch.full(
            (batch_size,),
            sequence_length,
            dtype=torch.int32,
            device=logits.device,
        ),
    }


def _gpu_corridor_exemplar_score_reduce(
    torch: Any,
    logits: Any,
    *,
    config: TeacherBackendConfig,
    vocab_chunk_size: int | None = None,
) -> dict[str, Any]:
    source_config = _second_pass_source_config(config)
    source = _gpu_corridor_source_payload(
        torch,
        logits,
        config=source_config,
        vocab_chunk_size=vocab_chunk_size,
    )
    teacher_entropy = source["teacher_entropy"]
    corridor_top_token_ids = source["top_token_ids"][..., 0]
    confidence = source["top_probs"][..., 0]
    score_selected_position = torch.argmax(teacher_entropy, dim=-1).to(torch.int32)
    gather_positions = score_selected_position.to(torch.int64).unsqueeze(-1)
    selected_entropy = torch.gather(
        teacher_entropy,
        -1,
        gather_positions,
    ).squeeze(-1)
    selected_confidence = torch.gather(
        confidence,
        -1,
        gather_positions,
    ).squeeze(-1)
    selected_top_token_id = torch.gather(
        corridor_top_token_ids,
        -1,
        gather_positions,
    ).squeeze(-1)
    batch_size = int(logits.shape[0])
    sequence_length = int(logits.shape[1])
    policy_id = _EXEMPLAR_SOURCE_POLICIES[config.exemplar_second_pass_source_policy]
    return {
        "corridor_top_token_ids": corridor_top_token_ids,
        "corridor_teacher_entropy": teacher_entropy,
        "corridor_confidence": confidence,
        "corridor_lengths": torch.full(
            (batch_size,),
            sequence_length,
            dtype=torch.int32,
            device=logits.device,
        ),
        "score_example_ids": torch.arange(
            batch_size,
            dtype=torch.int32,
            device=logits.device,
        ),
        "score_max_entropy": torch.max(teacher_entropy, dim=-1).values,
        "score_mean_entropy": torch.mean(teacher_entropy, dim=-1),
        "score_selected_position": score_selected_position,
        "score_top_token_id": selected_top_token_id.to(torch.int32),
        "score_selected_position_entropy": selected_entropy,
        "score_confidence_at_selected_position": selected_confidence,
        "score_source_policy_ids": torch.full(
            (batch_size,),
            policy_id,
            dtype=torch.int32,
            device=logits.device,
        ),
        "score_lengths": torch.full(
            (batch_size,),
            sequence_length,
            dtype=torch.int32,
            device=logits.device,
        ),
    }


def _gpu_corridor_exemplar_selected_reduce(
    torch: Any,
    logits: Any,
    *,
    config: TeacherBackendConfig,
    vocab_chunk_size: int | None = None,
) -> dict[str, Any]:
    return _gpu_corridor_exemplar_reduce(
        torch,
        logits,
        config=_second_pass_source_config(config),
        vocab_chunk_size=vocab_chunk_size,
    )


def _second_pass_source_config(
    config: TeacherBackendConfig,
) -> TeacherBackendConfig:
    return replace(
        config,
        exemplar_capture_mode="one_pass_candidate",
        exemplar_source_policy=config.exemplar_second_pass_source_policy,
    )


def _gpu_corridor_source_payload(
    torch: Any,
    logits: Any,
    *,
    config: TeacherBackendConfig,
    vocab_chunk_size: int | None = None,
) -> dict[str, Any]:
    if config.exemplar_source_policy == "dense_logits":
        workspace = _gpu_probability_workspace(
            torch,
            logits,
            vocab_chunk_size=vocab_chunk_size,
        )
        source = _gpu_topk_tail_reduce_from_workspace(torch, workspace, top_k=1)
        source["effective_top_k"] = torch.ones(
            logits.shape[:2],
            dtype=torch.int32,
            device=logits.device,
        )
        return source
    if config.exemplar_source_policy == "cascaded_soft_labels_v1":
        source = _gpu_cascaded_reduce(
            torch,
            logits,
            top_k=config.top_k,
            num_buckets=config.num_buckets,
            vocab_chunk_size=vocab_chunk_size,
        )
        source["effective_top_k"] = torch.full(
            logits.shape[:2],
            min(config.top_k, int(logits.shape[-1])),
            dtype=torch.int32,
            device=logits.device,
        )
        return source
    return _gpu_dynamic_cascaded_reduce(
        torch,
        logits,
        dynamic_top_k_min=config.dynamic_top_k_min,
        dynamic_top_k_max=config.dynamic_top_k_max,
        dynamic_mass_threshold=config.dynamic_mass_threshold,
        num_buckets=config.num_buckets,
        vocab_chunk_size=vocab_chunk_size,
    )


def _gpu_corridor_payload_records(
    config: TeacherBackendConfig,
    compact_payload: dict[str, np.ndarray],
    *,
    effective_vocab_size: int,
) -> dict[str, object]:
    effective_exemplar_top_n = min(config.exemplar_top_n, config.sequence_length)
    confidence = compact_payload["corridor_confidence"]
    entropy = compact_payload["corridor_teacher_entropy"]
    return {
        "corridor_records": {
            "policy": _CORRIDOR_POLICY,
            "record_count": int(confidence.shape[0]),
            "fields": (
                "corridor_top_token_ids",
                "corridor_top_probs",
                "corridor_teacher_entropy",
                "corridor_confidence",
            ),
        },
        "corridor_summary": {
            "record_count": int(confidence.shape[0]),
            "mean_confidence": float(np.mean(confidence, dtype=np.float32)),
            "mean_entropy": float(np.mean(entropy, dtype=np.float32)),
            "sequence_length": int(confidence.shape[1]),
        },
        "exemplar_records": {
            "policy": config.exemplar_selection_policy,
            "record_count": int(confidence.shape[0] * effective_exemplar_top_n),
            "positions_per_example": int(effective_exemplar_top_n),
        },
        "exemplar_summary": {
            "selection_policy": config.exemplar_selection_policy,
            "record_count": int(confidence.shape[0] * effective_exemplar_top_n),
            "positions_per_example": int(effective_exemplar_top_n),
        },
        "mode_records": {
            "mode_record_policy": _MODE_RECORD_POLICY,
            "record_count": int(confidence.shape[0]),
            "mode_field": "corridor_top_token_ids",
        },
        "source_policy_summary": _gpu_corridor_source_summary(
            config,
            compact_payload=compact_payload,
            effective_vocab_size=effective_vocab_size,
        ),
        "schema_metadata": _gpu_corridor_schema_metadata(config),
    }


def _gpu_corridor_score_records(
    config: TeacherBackendConfig,
    compact_payload: dict[str, np.ndarray],
    *,
    effective_vocab_size: int,
) -> dict[str, object]:
    score_fields = tuple(_corridor_score_payload_fields())
    return {
        "score_records": {
            "policy": config.exemplar_first_pass_score_policy,
            "record_count": int(compact_payload["score_example_ids"].shape[0]),
            "fields": score_fields,
        },
        "score_summary": {
            "record_count": int(compact_payload["score_example_ids"].shape[0]),
            "mean_max_entropy": float(
                np.mean(compact_payload["score_max_entropy"], dtype=np.float32)
            ),
            "mean_selected_position_entropy": float(
                np.mean(
                    compact_payload["score_selected_position_entropy"],
                    dtype=np.float32,
                )
            ),
            "sequence_length": config.sequence_length,
        },
        "source_policy_summary": _gpu_corridor_source_summary(
            _second_pass_source_config(config),
            compact_payload=compact_payload,
            effective_vocab_size=effective_vocab_size,
        ),
        "score_metadata": _gpu_corridor_score_pass_metadata(config),
    }


def _gpu_corridor_source_summary(
    config: TeacherBackendConfig,
    *,
    compact_payload: dict[str, np.ndarray],
    effective_vocab_size: int,
) -> dict[str, object]:
    if config.exemplar_source_policy == "dense_logits":
        return {
            "exemplar_source_policy": "dense_logits",
            "source_policy_kind": "full_resolution",
            "source_policy_uses_bucketed_tail": False,
            "source_policy_dynamic_top_k": False,
            "source_effective_top_k_semantics": "top_mode_only",
        }
    if config.exemplar_source_policy == "cascaded_soft_labels_v1":
        return {
            "exemplar_source_policy": "cascaded_soft_labels_v1",
            "source_policy_kind": "fixed_cascaded",
            "source_policy_uses_bucketed_tail": True,
            "source_policy_dynamic_top_k": False,
            "requested_top_k": config.top_k,
            "effective_top_k": min(config.top_k, effective_vocab_size),
            "num_buckets": config.num_buckets,
            "bucket_policy": _TAIL_BUCKET_POLICY,
        }
    effective_top_k = compact_payload.get("exemplar_source_effective_top_k")
    if effective_top_k is None or effective_top_k.size == 0:
        observed_min = min(config.dynamic_top_k_min, config.dynamic_top_k_max)
        observed_max = observed_min
        observed_mean = float(observed_min)
    else:
        observed_min = int(np.min(effective_top_k))
        observed_max = int(np.max(effective_top_k))
        observed_mean = float(np.mean(effective_top_k, dtype=np.float32))
    return {
        "exemplar_source_policy": "dynamic_cascaded_soft_labels_v1",
        "source_policy_kind": "dynamic_cascaded",
        "source_policy_uses_bucketed_tail": True,
        "source_policy_dynamic_top_k": True,
        "dynamic_top_k_policy": config.dynamic_top_k_policy,
        "dynamic_top_k_min_effective": min(
            config.dynamic_top_k_min,
            min(config.dynamic_top_k_max, effective_vocab_size),
        ),
        "dynamic_top_k_max_effective": min(
            config.dynamic_top_k_max,
            effective_vocab_size,
        ),
        "dynamic_mass_threshold": config.dynamic_mass_threshold,
        "num_buckets": config.num_buckets,
        "bucket_policy": _TAIL_BUCKET_POLICY,
        "effective_top_k_min_observed": observed_min,
        "effective_top_k_max_observed": observed_max,
        "effective_top_k_mean_observed": observed_mean,
    }


def _gpu_corridor_schema_metadata(config: TeacherBackendConfig) -> dict[str, object]:
    return {
        "schema_version": "corridor_exemplar_v1",
        "corridor_payload_flavor": config.corridor_payload_flavor,
        "production_corridor_schema": True,
        "historical_parity_claimed": False,
        "historical_reference_source": "gpu_torch_production",
        "exemplar_source_policy": config.exemplar_source_policy,
        "exemplar_selection_policy": config.exemplar_selection_policy,
        "corridor_policy": _CORRIDOR_POLICY,
        "mode_record_policy": _MODE_RECORD_POLICY,
        "fingerprint_topology_policy": _FINGERPRINT_TOPOLOGY_POLICY,
        "corridor_confidence_policy": _CORRIDOR_CONFIDENCE_POLICY,
        **_gpu_corridor_capture_mode_metadata(config, policy=None),
    }


def _effective_exemplar_capture_mode(
    config: TeacherBackendConfig,
    *,
    actual_batch_size: int | None = None,
    effective_vocab_size: int | None = None,
) -> str:
    return str(
        resolve_exemplar_capture_policy(
            config,
            actual_batch_size=actual_batch_size,
            effective_vocab_size=effective_vocab_size,
        )["exemplar_capture_mode_effective"]
    )


def _gpu_corridor_capture_mode_metadata(
    config: TeacherBackendConfig,
    *,
    policy: dict[str, object] | None,
) -> dict[str, object]:
    if policy is None:
        policy = resolve_exemplar_capture_policy(config)
    return {
        **policy,
        "exemplar_capture_mode_policy": "explicit_one_pass_candidate_v1",
        "exemplar_candidate_scope": "batch_all_examples",
        "corpus_level_exemplar_finalization": False,
        "requires_second_pass_for_final_exemplars": False,
        "rerun_teacher_for_selected_examples": False,
    }


def _gpu_corridor_score_pass_metadata(
    config: TeacherBackendConfig,
    *,
    policy: dict[str, object] | None = None,
) -> dict[str, object]:
    if policy is None:
        policy = resolve_exemplar_capture_policy(config)
    return {
        **policy,
        "exemplar_capture_stage": "score_pass",
        "exemplar_capture_mode_policy": "explicit_two_pass_sparse_exemplar_v1",
        "exemplar_candidate_scope": "batch_score_and_corridor_evidence",
        "corpus_level_exemplar_finalization": False,
        "requires_second_pass_for_final_exemplars": True,
        "rerun_teacher_for_selected_examples": True,
    }


def _gpu_corridor_selected_pass_metadata(
    config: TeacherBackendConfig,
    *,
    corpus_level_finalized: bool,
) -> dict[str, object]:
    policy = resolve_exemplar_capture_policy(config)
    policy = {**policy, "exemplar_capture_mode_effective": "two_pass_sparse_exemplar"}
    return {
        **policy,
        "exemplar_capture_stage": "selected_exemplar_pass",
        "exemplar_capture_mode_policy": "explicit_two_pass_sparse_exemplar_v1",
        "exemplar_candidate_scope": "selected_examples_only",
        "corpus_level_exemplar_finalization": corpus_level_finalized,
        "requires_second_pass_for_final_exemplars": False,
        "rerun_teacher_for_selected_examples": True,
    }


def _dynamic_cascaded_head_from_probs(
    torch: Any,
    probs: Any,
    log_probs: Any,
    *,
    dynamic_top_k_min: int,
    dynamic_top_k_max: int,
    dynamic_mass_threshold: float,
) -> dict[str, Any]:
    sorted_probs, sorted_ids = torch.sort(probs, dim=-1, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    threshold_reached = cumulative >= dynamic_mass_threshold
    has_hit = torch.any(threshold_reached, dim=-1)
    first_hit = torch.argmax(threshold_reached.to(torch.int64), dim=-1)
    threshold_count = torch.where(
        has_hit,
        first_hit + 1,
        torch.full_like(first_hit, dynamic_top_k_max),
    )
    effective_top_k = torch.clamp(
        threshold_count,
        min=dynamic_top_k_min,
        max=dynamic_top_k_max,
    )
    head_ids = sorted_ids[..., :dynamic_top_k_max]
    head_probs = sorted_probs[..., :dynamic_top_k_max]
    head_log_probs = torch.gather(log_probs, -1, head_ids)
    slots = torch.arange(dynamic_top_k_max, device=probs.device)
    top_selection_mask = slots < effective_top_k[..., None]
    top_token_ids = torch.where(
        top_selection_mask, head_ids, torch.zeros_like(head_ids)
    )
    top_probs = torch.where(
        top_selection_mask,
        head_probs,
        torch.zeros_like(head_probs),
    )
    top_log_probs = torch.where(
        top_selection_mask,
        head_log_probs,
        torch.zeros_like(head_log_probs),
    )
    top_mass = torch.sum(top_probs, dim=-1)
    return {
        "sorted_probs": sorted_probs,
        "top_token_ids": top_token_ids,
        "top_log_probs": top_log_probs,
        "top_probs": top_probs,
        "top_selection_mask": top_selection_mask,
        "effective_top_k": effective_top_k,
        "top_mass": top_mass,
    }


def _dynamic_bucket_masses_from_sorted_tail(
    torch: Any,
    sorted_probs: Any,
    *,
    effective_top_k: Any,
    num_buckets: int,
) -> Any:
    vocab_size = int(sorted_probs.shape[-1])
    tail_count = vocab_size - effective_top_k
    quotient = torch.div(tail_count, num_buckets, rounding_mode="floor")
    remainder = torch.remainder(tail_count, num_buckets)
    rank_shape = (1,) * (len(sorted_probs.shape) - 1) + (vocab_size,)
    ranks = torch.arange(vocab_size, device=sorted_probs.device).view(rank_shape)
    bucket_masses = []
    for bucket_id in range(num_buckets):
        tail_start = bucket_id * quotient + torch.clamp(remainder, max=bucket_id)
        tail_width = quotient + (remainder > bucket_id).to(quotient.dtype)
        tail_stop = tail_start + tail_width
        absolute_start = effective_top_k + tail_start
        absolute_stop = effective_top_k + tail_stop
        bucket_mask = (ranks >= absolute_start[..., None]) & (
            ranks < absolute_stop[..., None]
        )
        bucket_masses.append(
            torch.sum(
                torch.where(bucket_mask, sorted_probs, torch.zeros_like(sorted_probs)),
                dim=-1,
            )
        )
    return torch.stack(bucket_masses, dim=-1)


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


_SCORE_PAYLOAD_DTYPES: Mapping[str, type[np.generic]] = {
    "corridor_top_token_ids": np.int32,
    "corridor_teacher_entropy": np.float32,
    "corridor_confidence": np.float32,
    "corridor_lengths": np.int32,
    "score_example_ids": np.int32,
    "score_max_entropy": np.float32,
    "score_mean_entropy": np.float32,
    "score_selected_position": np.int32,
    "score_top_token_id": np.int32,
    "score_selected_position_entropy": np.float32,
    "score_confidence_at_selected_position": np.float32,
    "score_source_policy_ids": np.int32,
    "score_lengths": np.int32,
}


def _compact_score_payload_to_numpy(
    payload: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    return {
        name: _tensor_to_numpy(payload[name], dtype)
        for name, dtype in _SCORE_PAYLOAD_DTYPES.items()
    }


def _is_full_corridor_exemplar_payload(payload: Mapping[str, Any]) -> bool:
    return all(
        name in payload
        for name in (
            "corridor_top_token_ids",
            "corridor_top_probs",
            "exemplar_positions",
            "exemplar_source_top_token_ids",
            "exemplar_source_top_probs",
            "exemplar_source_bucket_masses",
        )
    )


def _compact_payload_to_numpy(payload: dict[str, Any]) -> dict[str, np.ndarray]:
    if "score_example_ids" in payload and not _is_full_corridor_exemplar_payload(
        payload
    ):
        return _compact_score_payload_to_numpy(payload)
    if _is_full_corridor_exemplar_payload(payload):
        compact = {
            "corridor_top_token_ids": _tensor_to_numpy(
                payload["corridor_top_token_ids"],
                np.int32,
            ),
            "corridor_top_probs": _tensor_to_numpy(
                payload["corridor_top_probs"],
                np.float32,
            ),
            "corridor_teacher_entropy": _tensor_to_numpy(
                payload["corridor_teacher_entropy"],
                np.float32,
            ),
            "corridor_confidence": _tensor_to_numpy(
                payload["corridor_confidence"],
                np.float32,
            ),
            "corridor_lengths": _tensor_to_numpy(
                payload["corridor_lengths"],
                np.int32,
            ),
            "exemplar_positions": _tensor_to_numpy(
                payload["exemplar_positions"],
                np.int32,
            ),
            "exemplar_scores": _tensor_to_numpy(
                payload["exemplar_scores"],
                np.float32,
            ),
            "exemplar_selection_mask": _tensor_to_numpy(
                payload["exemplar_selection_mask"],
                np.int32,
            ),
            "exemplar_source_policy_ids": _tensor_to_numpy(
                payload["exemplar_source_policy_ids"],
                np.int32,
            ),
            "exemplar_source_top_token_ids": _tensor_to_numpy(
                payload["exemplar_source_top_token_ids"],
                np.int32,
            ),
            "exemplar_source_top_log_probs": _tensor_to_numpy(
                payload["exemplar_source_top_log_probs"],
                np.float32,
            ),
            "exemplar_source_top_probs": _tensor_to_numpy(
                payload["exemplar_source_top_probs"],
                np.float32,
            ),
            "exemplar_source_top_selection_mask": _tensor_to_numpy(
                payload["exemplar_source_top_selection_mask"],
                np.bool_,
            ),
            "exemplar_source_effective_top_k": _tensor_to_numpy(
                payload["exemplar_source_effective_top_k"],
                np.int32,
            ),
            "exemplar_source_top_mass": _tensor_to_numpy(
                payload["exemplar_source_top_mass"],
                np.float32,
            ),
            "exemplar_source_tail_mass": _tensor_to_numpy(
                payload["exemplar_source_tail_mass"],
                np.float32,
            ),
            "exemplar_source_bucket_masses": _tensor_to_numpy(
                payload["exemplar_source_bucket_masses"],
                np.float32,
            ),
        }
        if "score_example_ids" in payload:
            compact.update(_compact_score_payload_to_numpy(payload))
        return compact
    if "score_example_ids" in payload:
        return _compact_score_payload_to_numpy(payload)
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
    if "top_selection_mask" in payload:
        compact["top_selection_mask"] = _tensor_to_numpy(
            payload["top_selection_mask"],
            np.bool_,
        )
    if "effective_top_k" in payload:
        compact["effective_top_k"] = _tensor_to_numpy(
            payload["effective_top_k"],
            np.int32,
        )
    if "bucket_masses" in payload:
        compact["bucket_masses"] = _tensor_to_numpy(
            payload["bucket_masses"],
            np.float32,
        )
    return compact


def _tensor_to_numpy(tensor: Any, dtype: type[np.generic]) -> np.ndarray:
    detached = tensor.detach()
    if np.issubdtype(np.dtype(dtype), np.floating):
        detached = detached.float()
    return detached.to("cpu").numpy().astype(dtype)


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
    if target_policy == "dynamic_cascaded_soft_labels_v1":
        return VocabChunkingPlan(
            requested=True,
            requested_size=requested_size,
            effective_size=None,
            used=False,
            reason="dynamic_exact_bucket_policy_requires_full_probability_workspace",
        )
    if target_policy == "corridor_exemplar_v1":
        if config.exemplar_source_policy == "dense_logits":
            reason = "corridor_dense_source_requires_full_probability_workspace"
        elif config.exemplar_source_policy == "cascaded_soft_labels_v1":
            reason = "exact_bucket_policy_requires_full_probability_workspace"
        else:
            reason = "dynamic_exact_bucket_policy_requires_full_probability_workspace"
        return VocabChunkingPlan(
            requested=True,
            requested_size=requested_size,
            effective_size=None,
            used=False,
            reason=reason,
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


def _corridor_score_payload_fields() -> list[str]:
    return [
        "corridor_top_token_ids",
        "corridor_teacher_entropy",
        "corridor_confidence",
        "corridor_lengths",
        "score_example_ids",
        "score_max_entropy",
        "score_mean_entropy",
        "score_selected_position",
        "score_top_token_id",
        "score_selected_position_entropy",
        "score_confidence_at_selected_position",
        "score_source_policy_ids",
        "score_lengths",
    ]


def _compact_payload_fields(
    target_policy: str,
    *,
    exemplar_capture_mode: str = "one_pass_candidate",
) -> list[str]:
    if target_policy == "corridor_exemplar_v1":
        if exemplar_capture_mode == "two_pass_sparse_exemplar":
            return _corridor_score_payload_fields()
        return [
            "corridor_top_token_ids",
            "corridor_top_probs",
            "corridor_teacher_entropy",
            "corridor_confidence",
            "corridor_lengths",
            "exemplar_positions",
            "exemplar_scores",
            "exemplar_selection_mask",
            "exemplar_source_policy_ids",
            "exemplar_source_effective_top_k",
            "exemplar_source_top_mass",
            "exemplar_source_tail_mass",
        ]
    if target_policy == "dynamic_cascaded_soft_labels_v1":
        return [
            "top_token_ids",
            "top_log_probs",
            "top_probs",
            "top_selection_mask",
            "effective_top_k",
            "top_mass",
            "tail_mass",
            "bucket_masses",
            "teacher_entropy",
        ]
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
    exemplar_capture_mode: str = "one_pass_candidate",
    vocab_chunking_used: bool,
) -> str:
    if target_policy == "dynamic_cascaded_soft_labels_v1":
        return "compact_dynamic_cascaded_soft_labels"
    if target_policy == "corridor_exemplar_v1":
        if exemplar_capture_mode == "two_pass_sparse_exemplar":
            return "compact_corridor_exemplar_score_pass"
        return "compact_corridor_exemplar"
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
