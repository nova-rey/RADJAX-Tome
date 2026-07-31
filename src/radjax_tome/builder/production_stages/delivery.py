"""Selected-only rerun stage and delivery configuration adapter."""

from __future__ import annotations

from typing import Any

from radjax_tome.backends import TeacherBackendConfig
from radjax_tome.builder.exemplar_delivery import (
    ExemplarDeliveryConfig,
    run_selected_delivery_rerun,
)
from radjax_tome.builder.production_stages.evidence import native_file_evidence
from radjax_tome.builder.production_stages.shared import (
    exemplar_capture_mode,
    native_c6_path_b_enabled,
)


def backend_config(config: Any) -> TeacherBackendConfig:
    return TeacherBackendConfig(
        backend_id=config.teacher_backend,
        runtime_mode=config.runtime_mode,
        target_policy=config.target_policy,
        model_id=config.teacher_model,
        tokenizer_id=config.tokenizer_id or config.teacher_model,
        sequence_length=config.sequence_length,
        batch_size=1,
        vocab_size=config.vocab_size,
        top_k=config.top_k,
        num_buckets=config.num_buckets,
        dynamic_top_k_min=config.dynamic_top_k_min,
        dynamic_top_k_max=config.dynamic_top_k_max,
        dynamic_mass_threshold=config.dynamic_mass_threshold,
        gpu_batch_size_mode=config.gpu_batch_size_mode,
        gpu_batch_size_preset=config.gpu_batch_size_preset,
        gpu_batch_size_custom=config.gpu_batch_size_custom,
        gpu_batch_size_auto_min=config.gpu_batch_size_auto_min,
        gpu_batch_size_auto_max=config.gpu_batch_size_auto_max,
        local_files_only=True,
        allow_downloads=False,
        fallback_policy="error",
        exemplar_capture_mode=exemplar_capture_mode(config),
    )


def exemplar_delivery_config(
    config: Any,
    effective_batch_size: int,
    *,
    progress_callback: Any = None,
    authoritative_records: tuple[dict[str, Any], ...] | None = None,
    delivery_authority_hash: str | None = None,
) -> ExemplarDeliveryConfig:
    return ExemplarDeliveryConfig(
        artifact_dir=config.output_dir,
        dataset_path=config.dataset_path,
        delivery_path=config.exemplar_delivery_path or "one_pass_pruned_candidate",
        selection_enabled=config.exemplar_selection_enabled,
        leaderboard_capacity=config.exemplar_leaderboard_capacity,
        selected_exemplar_budget=config.selected_exemplar_budget,
        selected_exemplar_fraction=config.selected_exemplar_fraction,
        retain_unselected_exemplar_payloads=config.retain_unselected_exemplar_payloads,
        score_policy=config.exemplar_score_policy,
        sequence_length=config.sequence_length,
        vocab_size=config.vocab_size,
        top_k=config.top_k,
        num_buckets=config.num_buckets,
        max_examples=config.max_examples,
        backend_config=backend_config(config),
        selected_rerun_batch_size=config.selected_rerun_batch_size
        or effective_batch_size,
        track_timing=config.track_delivery_timing,
        long_tail_warning_k=config.long_tail_warning_k,
        very_long_tail_warning_k=config.very_long_tail_warning_k,
        perverse_tail_warning_k=config.perverse_tail_warning_k,
        reject_perverse_exemplars=config.reject_perverse_exemplars,
        primary_selected_exemplar_budget=(
            config.primary_selected_exemplar_budget
            if config.primary_selected_exemplar_budget is not None
            else config.selected_exemplar_budget
        ),
        long_tail_side_board_cap=config.long_tail_side_board_cap,
        perverse_tail_side_board_cap=config.perverse_tail_side_board_cap,
        include_long_tail_in_primary=config.include_long_tail_in_primary,
        include_perverse_tail_in_primary=config.include_perverse_tail_in_primary,
        include_perverse_tail_in_student=config.include_perverse_tail_in_student,
        progress_callback=progress_callback,
        authoritative_selection=authoritative_records is not None,
        authoritative_records=authoritative_records,
        execution_mode=(
            "native_c6_path_b_v1"
            if authoritative_records is not None and native_c6_path_b_enabled(config)
            else "legacy_delivery_v1"
        ),
        rerun_metrics={},
        delivery_authority_hash=delivery_authority_hash,
    )


def native_selected_rerun_operation(state: Any, inputs: Any) -> Any:
    from radjax_tome.builder.native_path_b.contracts import EvidenceCount, StageResult
    from radjax_tome.builder.native_path_b.delivery import SelectedRerunHandoff

    context = inputs.selection
    config = exemplar_delivery_config(
        state.config,
        state.effective_batch_size,
        progress_callback=state.progress.handle_delivery_event,
        authoritative_records=tuple(context["delivery_records"]),
        delivery_authority_hash=str(
            (context.get("authorities") or {}).get("score_pass_authority_hash") or ""
        ),
    )
    prepared = run_selected_delivery_rerun(config)
    staging_root = (
        config.artifact_dir
        / ".staging-native-c6"
        / str(config.delivery_authority_hash or "unbound").replace(":", "-")
    )
    staging_paths = tuple(
        sorted(path for path in staging_root.rglob("*") if path.is_file())
    )
    if not staging_paths:
        raise ValueError("selected rerun produced no native staged payload evidence")
    evidence = native_file_evidence(
        "selected_delivery_rerun",
        staging_paths,
        counts=(
            EvidenceCount("selected_record_count", len(prepared.selected_records)),
            EvidenceCount("selected_payload_count", len(prepared.selected_payloads)),
            EvidenceCount(
                "selected_rerun_example_count", prepared.rerun_selected_example_count
            ),
        ),
        prior=inputs.c5_evidence,
    )
    return StageResult(
        status="pass",
        value=SelectedRerunHandoff(
            value={"prepared": prepared, "context": context}, stage_evidence=evidence
        ),
        evidence=evidence,
    )
