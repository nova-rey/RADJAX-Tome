from __future__ import annotations

from dataclasses import replace

import numpy as np

from radjax_tome.backends.base import (
    BackendCapability,
    TargetPolicy,
    TeacherBackendConfig,
    TeacherBatchInput,
    TeacherEmissionResult,
    resolve_exemplar_capture_policy,
    resolve_gpu_batch_size_policy,
    validate_gpu_batch_size_policy_config,
)

_SUPPORTED_EMISSION_POLICIES = {
    "dense_logits",
    "topk_with_tail_v0",
    "cascaded_soft_labels_v1",
    "dynamic_cascaded_soft_labels_v1",
    "corridor_exemplar_v1",
}
_CAPABILITY_STATUS: dict[TargetPolicy, str] = {
    "dense_logits": "supported_debug",
    "topk_with_tail_v0": "supported",
    "cascaded_soft_labels_v1": "supported",
    "dynamic_cascaded_soft_labels_v1": "supported",
    "corridor_exemplar_v1": "supported",
}
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
            raise ValueError(
                "exemplar_first_pass_score_policy must be 'entropy_score_v1'"
            )
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
        if config.dynamic_top_k_min < 1:
            raise ValueError("dynamic_top_k_min must be >= 1")
        if config.dynamic_top_k_max < config.dynamic_top_k_min:
            raise ValueError("dynamic_top_k_max must be >= dynamic_top_k_min")
        if not 0.0 < config.dynamic_mass_threshold <= 1.0:
            raise ValueError("dynamic_mass_threshold must be > 0 and <= 1")
        if config.dynamic_top_k_policy != _DYNAMIC_TOP_K_POLICY:
            raise ValueError("dynamic_top_k_policy must be 'mass_threshold_v1'")
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
                    "Spec 3.3F8 CPU reference corridor_exemplar_v1 emits the "
                    "locked production schema with source-policy-aware "
                    "metadata through the backend contract."
                ),
            ),
            BackendCapability(
                backend_id=self.backend_id,
                backend_family=self.backend_family,
                runtime_mode="cpu",
                target_policy="dynamic_cascaded_soft_labels_v1",
                status="supported",
                optimized=False,
                implemented_now=True,
                notes=(
                    "Spec 3.3F6 CPU reference backend emits dynamic top-k "
                    "explicit head plus bucketed-tail soft-label payloads "
                    "through the backend contract."
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
        metadata = self.metadata(
            actual_batch_size=len(batch.texts),
            payload=payload,
        )
        return TeacherEmissionResult(
            backend_id=self.backend_id,
            runtime_mode="cpu",
            target_policy=self.config.target_policy,
            input_ids=input_ids,
            attention_mask=attention_mask,
            payload=payload,
            metadata=metadata,
        )

    def metadata(
        self,
        actual_batch_size: int | None = None,
        payload: dict[str, np.ndarray] | None = None,
    ) -> dict[str, object]:
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
            "fallback_policy": self.config.fallback_policy,
            "fallback_handled_by": "none",
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
        metadata.update(resolve_gpu_batch_size_policy(self.config, payload=payload))
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
                    "bucket_policy": _TAIL_BUCKET_POLICY,
                }
            )
        if self.config.target_policy == "dynamic_cascaded_soft_labels_v1":
            dynamic_metadata = _dynamic_cascaded_metadata(
                self.config,
                payload=payload,
            )
            metadata.update(dynamic_metadata)
        if self.config.target_policy == "corridor_exemplar_v1":
            metadata.update(
                _corridor_metadata(
                    self.config,
                    payload=payload,
                    actual_batch_size=actual_batch_size,
                )
            )
        return metadata

    def close(self) -> None:
        return None

    def emit_corridor_exemplar_score_batch(
        self,
        batch: TeacherBatchInput,
    ) -> TeacherEmissionResult:
        input_ids, attention_mask = _encode_texts(
            batch.texts,
            sequence_length=self.config.sequence_length,
            vocab_size=self.config.vocab_size,
        )
        logits = _deterministic_logits(input_ids, vocab_size=self.config.vocab_size)
        payload = _corridor_exemplar_score_payload(logits, config=self.config)
        return TeacherEmissionResult(
            backend_id=self.backend_id,
            runtime_mode="cpu",
            target_policy="corridor_exemplar_v1",
            input_ids=input_ids,
            attention_mask=attention_mask,
            payload=payload,
            metadata=self.metadata(actual_batch_size=len(batch.texts), payload=payload),
        )

    def emit_corridor_exemplar_selected_batch(
        self,
        batch: TeacherBatchInput,
        *,
        corpus_level_finalized: bool = False,
    ) -> TeacherEmissionResult:
        input_ids, attention_mask = _encode_texts(
            batch.texts,
            sequence_length=self.config.sequence_length,
            vocab_size=self.config.vocab_size,
        )
        logits = _deterministic_logits(input_ids, vocab_size=self.config.vocab_size)
        payload = _corridor_exemplar_selected_payload(
            logits,
            config=self.config,
            corpus_level_finalized=corpus_level_finalized,
        )
        metadata = dict(
            self.metadata(actual_batch_size=len(batch.texts), payload=payload)
        )
        metadata.update(payload["schema_metadata"])
        return TeacherEmissionResult(
            backend_id=self.backend_id,
            runtime_mode="cpu",
            target_policy="corridor_exemplar_v1",
            input_ids=input_ids,
            attention_mask=attention_mask,
            payload=payload,
            metadata=metadata,
        )

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
        if self.config.target_policy == "dynamic_cascaded_soft_labels_v1":
            return _dense_logits_to_dynamic_cascaded(
                logits,
                dynamic_top_k_min=self.config.dynamic_top_k_min,
                dynamic_top_k_max=self.config.dynamic_top_k_max,
                dynamic_mass_threshold=self.config.dynamic_mass_threshold,
                num_buckets=self.config.num_buckets,
            )
        if self.config.target_policy == "corridor_exemplar_v1":
            if (
                _effective_exemplar_capture_mode(
                    self.config,
                    actual_batch_size=logits.shape[0],
                )
                == "two_pass_sparse_exemplar"
            ):
                return _corridor_exemplar_score_payload(
                    logits,
                    config=self.config,
                )
            return _corridor_exemplar_payload(
                logits,
                config=self.config,
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


def _dense_logits_to_dynamic_cascaded(
    logits: np.ndarray,
    *,
    dynamic_top_k_min: int,
    dynamic_top_k_max: int,
    dynamic_mass_threshold: float,
    num_buckets: int,
) -> dict[str, np.ndarray]:
    values = np.asarray(logits, dtype=np.float32)
    probs = _softmax(values)
    log_probs = np.log(np.maximum(probs, np.finfo(np.float32).tiny))
    effective_max_k = min(dynamic_top_k_max, values.shape[-1])
    effective_min_k = min(dynamic_top_k_min, effective_max_k)

    top_token_ids = np.zeros((*values.shape[:2], effective_max_k), dtype=np.int32)
    top_log_probs = np.zeros((*values.shape[:2], effective_max_k), dtype=np.float32)
    top_probs = np.zeros((*values.shape[:2], effective_max_k), dtype=np.float32)
    top_selection_mask = np.zeros((*values.shape[:2], effective_max_k), dtype=bool)
    effective_top_k = np.zeros(values.shape[:2], dtype=np.int32)
    top_mass = np.zeros(values.shape[:2], dtype=np.float32)
    bucket_masses = np.zeros((*values.shape[:2], num_buckets), dtype=np.float32)
    teacher_entropy = -np.sum(probs * log_probs, axis=-1).astype(np.float32)
    sorted_ids = np.argsort(-probs, axis=-1, kind="stable").astype(np.int32)

    for row in range(values.shape[0]):
        for position in range(values.shape[1]):
            ids = sorted_ids[row, position]
            sorted_probs = probs[row, position, ids]
            cumulative = np.cumsum(sorted_probs, dtype=np.float32)
            threshold_hits = np.flatnonzero(cumulative >= dynamic_mass_threshold)
            threshold_count = (
                int(threshold_hits[0]) + 1 if threshold_hits.size else effective_max_k
            )
            selected_count = min(
                max(threshold_count, effective_min_k),
                effective_max_k,
            )
            selected_ids = ids[:selected_count]
            selected_probs = probs[row, position, selected_ids]
            selected_log_probs = log_probs[row, position, selected_ids]

            top_token_ids[row, position, :selected_count] = selected_ids
            top_probs[row, position, :selected_count] = selected_probs
            top_log_probs[row, position, :selected_count] = selected_log_probs
            top_selection_mask[row, position, :selected_count] = True
            effective_top_k[row, position] = selected_count
            top_mass[row, position] = np.sum(selected_probs, dtype=np.float32)

            selected_mask = np.zeros(values.shape[-1], dtype=bool)
            selected_mask[selected_ids] = True
            tail_probs = probs[row, position][~selected_mask]
            if tail_probs.size == 0:
                continue
            sorted_tail = np.sort(tail_probs)[::-1]
            buckets = np.array_split(sorted_tail, num_buckets)
            for bucket_id, bucket in enumerate(buckets):
                bucket_masses[row, position, bucket_id] = np.sum(
                    bucket,
                    dtype=np.float32,
                )

    tail_mass = np.clip(1.0 - top_mass, 0.0, 1.0).astype(np.float32)
    return {
        "top_token_ids": top_token_ids,
        "top_log_probs": top_log_probs,
        "top_probs": top_probs,
        "top_selection_mask": top_selection_mask,
        "effective_top_k": effective_top_k,
        "top_mass": top_mass.astype(np.float32),
        "tail_mass": tail_mass,
        "bucket_masses": bucket_masses,
        "teacher_entropy": teacher_entropy,
    }


def _dynamic_cascaded_metadata(
    config: TeacherBackendConfig,
    *,
    payload: dict[str, np.ndarray] | None,
) -> dict[str, object]:
    effective_max_k = min(config.dynamic_top_k_max, config.vocab_size)
    effective_min_k = min(config.dynamic_top_k_min, effective_max_k)
    effective_top_k = (
        np.asarray(payload["effective_top_k"], dtype=np.int32)
        if payload is not None and "effective_top_k" in payload
        else None
    )
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
        "effective_top_k_min_observed": observed_min,
        "effective_top_k_max_observed": observed_max,
        "effective_top_k_mean_observed": observed_mean,
    }


def _corridor_exemplar_payload(
    logits: np.ndarray,
    *,
    config: TeacherBackendConfig,
) -> dict[str, object]:
    source = _corridor_source_payload(logits, config=config)
    source_summary = _corridor_source_policy_summary(config, source_payload=source)
    effective_exemplar_top_n = min(config.exemplar_top_n, logits.shape[1])
    entropy = source["teacher_entropy"].astype(np.float32)
    top_ids = source["top_token_ids"][..., 0].astype(np.int32)
    top_probs = source["top_probs"][..., 0].astype(np.float32)
    confidence = top_probs.astype(np.float32)
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
    lengths = np.full((logits.shape[0],), logits.shape[1], dtype=np.int32)
    policy_id = _EXEMPLAR_SOURCE_POLICIES[config.exemplar_source_policy]
    source_policy_ids = np.full(entropy.shape, policy_id, dtype=np.int32)
    score_selected_position = np.argmax(entropy, axis=-1).astype(np.int32)
    score_selected_entropy = np.take_along_axis(
        entropy,
        score_selected_position[:, None],
        axis=-1,
    )[:, 0].astype(np.float32)
    score_selected_confidence = np.take_along_axis(
        confidence,
        score_selected_position[:, None],
        axis=-1,
    )[:, 0].astype(np.float32)
    score_top_token_id = np.take_along_axis(
        top_ids,
        score_selected_position[:, None],
        axis=-1,
    )[:, 0].astype(np.int32)
    schema_metadata = _corridor_schema_metadata(config)
    corridor_records = {
        "policy": _CORRIDOR_POLICY,
        "record_count": int(logits.shape[0]),
        "fields": (
            "corridor_top_token_ids",
            "corridor_top_probs",
            "corridor_teacher_entropy",
            "corridor_confidence",
        ),
    }
    exemplar_records = {
        "policy": config.exemplar_selection_policy,
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
            "selection_policy": config.exemplar_selection_policy,
            "record_count": int(logits.shape[0] * effective_exemplar_top_n),
            "positions_per_example": int(effective_exemplar_top_n),
        },
        "mode_records": {
            "mode_record_policy": _MODE_RECORD_POLICY,
            "record_count": int(logits.shape[0]),
            "mode_field": "corridor_top_token_ids",
        },
        "source_policy_summary": source_summary,
        "schema_metadata": schema_metadata,
        "corridor_top_token_ids": top_ids,
        "corridor_top_probs": top_probs,
        "corridor_teacher_entropy": entropy,
        "corridor_confidence": confidence,
        "corridor_lengths": lengths,
        "exemplar_positions": exemplar_positions,
        "exemplar_scores": exemplar_scores,
        "exemplar_selection_mask": selection_mask,
        "exemplar_source_policy_ids": source_policy_ids,
        "exemplar_source_top_token_ids": source["top_token_ids"].astype(np.int32),
        "exemplar_source_top_log_probs": source["top_log_probs"].astype(np.float32),
        "exemplar_source_top_probs": source["top_probs"].astype(np.float32),
        "exemplar_source_top_selection_mask": _source_selection_mask(source),
        "exemplar_source_effective_top_k": source["effective_top_k"].astype(np.int32),
        "exemplar_source_top_mass": source["top_mass"].astype(np.float32),
        "exemplar_source_tail_mass": source["tail_mass"].astype(np.float32),
        "exemplar_source_bucket_masses": _source_bucket_masses(
            source,
            logits_shape=logits.shape,
            num_buckets=config.num_buckets,
        ),
        "score_example_ids": np.arange(logits.shape[0], dtype=np.int32),
        "score_max_entropy": np.max(entropy, axis=-1).astype(np.float32),
        "score_mean_entropy": np.mean(entropy, axis=-1, dtype=np.float32).astype(
            np.float32
        ),
        "score_selected_position": score_selected_position,
        "score_top_token_id": score_top_token_id,
        "score_selected_position_entropy": score_selected_entropy,
        "score_confidence_at_selected_position": score_selected_confidence,
        "score_source_policy_ids": np.full(
            (logits.shape[0],), policy_id, dtype=np.int32
        ),
        "score_lengths": lengths,
    }


def _corridor_exemplar_score_payload(
    logits: np.ndarray,
    *,
    config: TeacherBackendConfig,
) -> dict[str, object]:
    source_config = _second_pass_source_config(config)
    source = _corridor_source_payload(logits, config=source_config)
    source_summary = _corridor_source_policy_summary(
        source_config,
        source_payload=source,
    )
    entropy = source["teacher_entropy"].astype(np.float32)
    confidence = source["top_probs"][..., 0].astype(np.float32)
    top_ids = source["top_token_ids"][..., 0].astype(np.int32)
    selected_position = np.argmax(entropy, axis=-1).astype(np.int32)
    selected_entropy = np.take_along_axis(
        entropy,
        selected_position[:, None],
        axis=-1,
    )[:, 0].astype(np.float32)
    selected_confidence = np.take_along_axis(
        confidence,
        selected_position[:, None],
        axis=-1,
    )[:, 0].astype(np.float32)
    selected_top_token_id = np.take_along_axis(
        top_ids,
        selected_position[:, None],
        axis=-1,
    )[:, 0].astype(np.int32)
    policy_id = _EXEMPLAR_SOURCE_POLICIES[config.exemplar_second_pass_source_policy]
    batch_size = int(logits.shape[0])
    fields = (
        "score_example_ids",
        "score_max_entropy",
        "score_mean_entropy",
        "score_selected_position",
        "score_top_token_id",
        "score_selected_position_entropy",
        "score_confidence_at_selected_position",
        "score_source_policy_ids",
        "score_lengths",
    )
    return {
        "score_records": {
            "policy": config.exemplar_first_pass_score_policy,
            "record_count": batch_size,
            "fields": fields,
        },
        "score_summary": {
            "record_count": batch_size,
            "mean_max_entropy": float(np.mean(np.max(entropy, axis=-1))),
            "mean_selected_position_entropy": float(np.mean(selected_entropy)),
            "sequence_length": int(logits.shape[1]),
        },
        "source_policy_summary": source_summary,
        "score_metadata": _corridor_score_pass_metadata(config),
        "corridor_top_token_ids": top_ids,
        "corridor_teacher_entropy": entropy,
        "corridor_confidence": confidence,
        "corridor_lengths": np.full((batch_size,), logits.shape[1], dtype=np.int32),
        "score_example_ids": np.arange(batch_size, dtype=np.int32),
        "score_max_entropy": np.max(entropy, axis=-1).astype(np.float32),
        "score_mean_entropy": np.mean(entropy, axis=-1, dtype=np.float32).astype(
            np.float32
        ),
        "score_selected_position": selected_position,
        "score_top_token_id": selected_top_token_id,
        "score_selected_position_entropy": selected_entropy,
        "score_confidence_at_selected_position": selected_confidence,
        "score_source_policy_ids": np.full((batch_size,), policy_id, dtype=np.int32),
        "score_lengths": np.full((batch_size,), logits.shape[1], dtype=np.int32),
    }


def _corridor_exemplar_selected_payload(
    logits: np.ndarray,
    *,
    config: TeacherBackendConfig,
    corpus_level_finalized: bool = False,
) -> dict[str, object]:
    selected_config = replace(
        config,
        exemplar_capture_mode="one_pass_candidate",
        exemplar_source_policy=config.exemplar_second_pass_source_policy,
    )
    payload = _corridor_exemplar_payload(logits, config=selected_config)
    payload["schema_metadata"] = {
        **payload["schema_metadata"],
        **_corridor_selected_pass_metadata(
            config,
            corpus_level_finalized=corpus_level_finalized,
        ),
    }
    payload["source_policy_summary"] = _corridor_source_policy_summary(
        selected_config,
        source_payload=None,
    )
    return payload


def _source_selection_mask(source: dict[str, np.ndarray]) -> np.ndarray:
    if "top_selection_mask" in source:
        return source["top_selection_mask"].astype(bool)
    return np.ones(np.asarray(source["top_token_ids"]).shape, dtype=bool)


def _source_bucket_masses(
    source: dict[str, np.ndarray],
    *,
    logits_shape: tuple[int, ...],
    num_buckets: int,
) -> np.ndarray:
    if "bucket_masses" in source:
        return source["bucket_masses"].astype(np.float32)
    return np.zeros((*logits_shape[:2], num_buckets), dtype=np.float32)


def _second_pass_source_config(
    config: TeacherBackendConfig,
) -> TeacherBackendConfig:
    return replace(
        config,
        exemplar_source_policy=config.exemplar_second_pass_source_policy,
    )


def _corridor_source_payload(
    logits: np.ndarray,
    *,
    config: TeacherBackendConfig,
) -> dict[str, np.ndarray]:
    if config.exemplar_source_policy == "dense_logits":
        topk = _dense_logits_to_topk_tail(logits, top_k=1)
        return {
            **topk,
            "effective_top_k": np.ones(logits.shape[:2], dtype=np.int32),
        }
    if config.exemplar_source_policy == "cascaded_soft_labels_v1":
        topk = _dense_logits_to_topk_tail(logits, top_k=config.top_k)
        return {
            **topk,
            "bucket_masses": _tail_bucket_masses(
                logits,
                top_token_ids=topk["top_token_ids"],
                num_buckets=config.num_buckets,
            ),
            "effective_top_k": np.full(
                logits.shape[:2],
                min(config.top_k, logits.shape[-1]),
                dtype=np.int32,
            ),
        }
    return _dense_logits_to_dynamic_cascaded(
        logits,
        dynamic_top_k_min=config.dynamic_top_k_min,
        dynamic_top_k_max=config.dynamic_top_k_max,
        dynamic_mass_threshold=config.dynamic_mass_threshold,
        num_buckets=config.num_buckets,
    )


def _corridor_source_policy_summary(
    config: TeacherBackendConfig,
    *,
    source_payload: dict[str, np.ndarray] | None = None,
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
            "effective_top_k": min(config.top_k, config.vocab_size),
            "num_buckets": config.num_buckets,
            "bucket_policy": _TAIL_BUCKET_POLICY,
        }
    dynamic_metadata = _dynamic_cascaded_metadata(config, payload=source_payload)
    return {
        "exemplar_source_policy": "dynamic_cascaded_soft_labels_v1",
        "source_policy_kind": "dynamic_cascaded",
        "source_policy_uses_bucketed_tail": True,
        "source_policy_dynamic_top_k": True,
        "dynamic_top_k_policy": dynamic_metadata["dynamic_top_k_policy"],
        "dynamic_top_k_min_effective": dynamic_metadata["dynamic_top_k_min_effective"],
        "dynamic_top_k_max_effective": dynamic_metadata["dynamic_top_k_max_effective"],
        "dynamic_mass_threshold": dynamic_metadata["dynamic_mass_threshold"],
        "num_buckets": dynamic_metadata["num_buckets"],
        "bucket_policy": dynamic_metadata["bucket_policy"],
        "effective_top_k_min_observed": dynamic_metadata[
            "effective_top_k_min_observed"
        ],
        "effective_top_k_max_observed": dynamic_metadata[
            "effective_top_k_max_observed"
        ],
        "effective_top_k_mean_observed": dynamic_metadata[
            "effective_top_k_mean_observed"
        ],
    }


def _corridor_metadata(
    config: TeacherBackendConfig,
    *,
    payload: dict[str, object] | None,
    actual_batch_size: int | None,
) -> dict[str, object]:
    policy = resolve_exemplar_capture_policy(
        config,
        actual_batch_size=actual_batch_size,
    )
    effective_mode = str(policy["exemplar_capture_mode_effective"])
    if effective_mode == "two_pass_sparse_exemplar":
        source_summary = (
            payload.get("source_policy_summary", {})
            if payload is not None
            else _corridor_source_policy_summary(_second_pass_source_config(config))
        )
        return {
            "schema_version": "corridor_exemplar_score_pass_v1",
            "corridor_payload_flavor": config.corridor_payload_flavor,
            "production_corridor_schema": False,
            "historical_parity_claimed": False,
            "historical_reference_source": "cpu_reference_score_pass",
            "exemplar_source_policy": config.exemplar_source_policy,
            "exemplar_first_pass_score_policy": config.exemplar_first_pass_score_policy,
            "exemplar_second_pass_source_policy": (
                config.exemplar_second_pass_source_policy
            ),
            "score_source_policy": config.exemplar_second_pass_source_policy,
            "exemplar_selection_policy": config.exemplar_selection_policy,
            **_corridor_score_pass_metadata(config, policy=policy),
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
            "score_records": (
                payload.get("score_records", {}).get("record_count", 0)
                if payload is not None
                else 0
            ),
            "source_policy_kind": source_summary.get("source_policy_kind"),
            "source_policy_uses_bucketed_tail": source_summary.get(
                "source_policy_uses_bucketed_tail"
            ),
            "source_policy_dynamic_top_k": source_summary.get(
                "source_policy_dynamic_top_k"
            ),
        }
    effective_exemplar_top_n = min(config.exemplar_top_n, config.sequence_length)
    source_summary = (
        payload.get("source_policy_summary", {})
        if payload is not None
        else _corridor_source_policy_summary(config)
    )
    return {
        "schema_version": "corridor_exemplar_v1",
        "corridor_payload_flavor": config.corridor_payload_flavor,
        "production_corridor_schema": True,
        "historical_parity_claimed": False,
        "historical_reference_source": "cpu_reference_proxy",
        "exemplar_source_policy": config.exemplar_source_policy,
        "exemplar_selection_policy": config.exemplar_selection_policy,
        "corridor_policy": _CORRIDOR_POLICY,
        "mode_record_policy": _MODE_RECORD_POLICY,
        "fingerprint_topology_policy": _FINGERPRINT_TOPOLOGY_POLICY,
        "corridor_confidence_policy": _CORRIDOR_CONFIDENCE_POLICY,
        **_corridor_capture_mode_metadata(config, policy=policy),
        "requested_exemplar_top_n": config.exemplar_top_n,
        "effective_exemplar_top_n": effective_exemplar_top_n,
        "exemplar_records": effective_exemplar_top_n,
        "source_policy_kind": source_summary.get("source_policy_kind"),
        "source_policy_uses_bucketed_tail": source_summary.get(
            "source_policy_uses_bucketed_tail"
        ),
        "source_policy_dynamic_top_k": source_summary.get(
            "source_policy_dynamic_top_k"
        ),
    }


def _corridor_schema_metadata(config: TeacherBackendConfig) -> dict[str, object]:
    return {
        "schema_version": "corridor_exemplar_v1",
        "corridor_payload_flavor": config.corridor_payload_flavor,
        "production_corridor_schema": True,
        "historical_parity_claimed": False,
        "historical_reference_source": "cpu_reference_proxy",
        "exemplar_source_policy": config.exemplar_source_policy,
        "exemplar_selection_policy": config.exemplar_selection_policy,
        "corridor_policy": _CORRIDOR_POLICY,
        "mode_record_policy": _MODE_RECORD_POLICY,
        "fingerprint_topology_policy": _FINGERPRINT_TOPOLOGY_POLICY,
        "corridor_confidence_policy": _CORRIDOR_CONFIDENCE_POLICY,
        **_corridor_capture_mode_metadata(config, policy=None),
    }


def _effective_exemplar_capture_mode(
    config: TeacherBackendConfig,
    *,
    actual_batch_size: int | None = None,
) -> str:
    return str(
        resolve_exemplar_capture_policy(
            config,
            actual_batch_size=actual_batch_size,
        )["exemplar_capture_mode_effective"]
    )


def _corridor_capture_mode_metadata(
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


def _corridor_score_pass_metadata(
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


def _corridor_selected_pass_metadata(
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


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted, dtype=np.float32)
    return (exp / np.sum(exp, axis=-1, keepdims=True)).astype(np.float32)
