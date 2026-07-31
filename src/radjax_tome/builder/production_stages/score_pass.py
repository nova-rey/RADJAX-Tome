"""Streaming teacher score-pass configuration and execution stage."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from radjax_tome.builder.backend_textbook import (
    BackendTeacherTextbookBuildConfig,
    build_streaming_backend_teacher_textbook,
)
from radjax_tome.builder.production_stages.delivery import backend_config
from radjax_tome.builder.production_stages.evidence import native_file_evidence
from radjax_tome.builder.production_stages.shared import (
    exemplar_capture_mode,
    native_c6_path_b_enabled,
    selection_integration_hash,
)


def streaming_config(
    config: Any, effective_batch_size: int, *, progress_callback: Any = None
) -> BackendTeacherTextbookBuildConfig:
    return BackendTeacherTextbookBuildConfig(
        output_dir=config.output_dir,
        dataset_path=config.dataset_path,
        teacher_backend=config.teacher_backend,
        runtime_mode=config.runtime_mode,
        target_policy=config.target_policy,
        teacher_model_id=config.teacher_model,
        tokenizer_id=config.tokenizer_id or config.teacher_model,
        sequence_length=config.sequence_length,
        batch_size=effective_batch_size,
        max_examples=config.max_examples,
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
        fallback_policy="error",
        exemplar_capture_mode=exemplar_capture_mode(config),
        local_files_only=True,
        allow_downloads=False,
        overwrite=False,
        corpus_manifest_path=config.corpus_manifest_path,
        teacher_model_provenance_path=config.teacher_model_provenance_path,
        streaming=True,
        resume=config.resume,
        shard_size_examples=config.shard_size_examples,
        progress_log_path=config.progress_log_path,
        run_manifest_path=config.run_manifest_path,
        progress_callback=progress_callback,
        selection_integration_policy=config.selection_integration_policy,
        selection_integration_config_hash=selection_integration_hash(config),
        exemplar_selection_enabled=native_c6_path_b_enabled(config),
        native_c6_path_b_execution=native_c6_path_b_enabled(config),
    )


def native_score_pass_operation(state: Any) -> Any:
    """Execute the existing streaming score pass and preserve typed evidence."""
    from radjax_tome.builder.native_path_b.contracts import StageResult

    started = perf_counter()
    plan = state.plan or {}
    state.progress.start_score_pass(
        examples_total=planned_example_count(plan),
        shard_count_total=planned_shard_count(state.config, plan),
    )
    report = build_streaming_backend_teacher_textbook(
        streaming_config(
            state.config,
            state.effective_batch_size,
            progress_callback=state.progress.handle_streaming_event,
        )
    )
    state.build_report = report
    state.main_pass_wall_seconds = perf_counter() - started
    state.progress.memory_checkpoint("score_pass_complete")
    evidence = native_file_evidence(
        "score_pass",
        (
            state.config.output_dir / "run_manifest.json",
            state.config.output_dir / "metadata.json",
        ),
    )
    return StageResult(status="pass", value=state, evidence=evidence)


@dataclass(frozen=True)
class ScorePassOperations:
    """Facade-owned policy callbacks needed for terminal compatibility reports."""

    report: Callable[..., dict[str, Any]]
    record_terminal_report: Callable[[Any, dict[str, Any]], None]
    terminal_stage_failure: Callable[[Any, str], Any]
    stage_success: Callable[..., Any]
    now: Callable[[], str]
    build_streaming: Callable[[BackendTeacherTextbookBuildConfig], Any] = (
        build_streaming_backend_teacher_textbook
    )


def run_score_pass(state: Any, *, operations: ScorePassOperations) -> Any:
    """Execute the historical score-pass body outside ``builder.production``."""

    config = state.config
    started = perf_counter()
    plan = state.plan or {}
    state.progress.start_score_pass(
        examples_total=planned_example_count(plan),
        shard_count_total=planned_shard_count(config, plan),
    )
    try:
        build_report = operations.build_streaming(
            streaming_config(
                config,
                state.effective_batch_size,
                progress_callback=state.progress.handle_streaming_event,
            )
        )
    except Exception as exc:
        state.blockers.append(str(exc))
        report = operations.report(
            config,
            created_at=state.created_at,
            completed_at=operations.now(),
            status="fail",
            blockers=state.blockers,
            warnings=state.warnings,
            doctor_report=state.doctor_report or {},
            run_plan_path=state.run_plan_path,
            run_plan=plan,
            effective_batch_size=state.effective_batch_size,
            already_complete=state.already_complete,
            parity_report_path=state.parity_report_path,
            parity_status="not_run",
        )
        operations.record_terminal_report(state, report)
        return operations.terminal_stage_failure(state, "score_pass")
    state.main_pass_wall_seconds = perf_counter() - started
    state.build_report = build_report
    state.progress.memory_checkpoint("score_pass_complete")
    return operations.stage_success(
        state,
        "score_pass",
        paths=(
            config.output_dir / "run_manifest.json",
            config.output_dir / "metadata.json",
        ),
        prior_stage="preflight",
        prior_paths=(state.run_plan_path,),
    )


def planned_example_count(plan: dict[str, Any]) -> int | None:
    estimates = plan.get("artifact_estimates")
    if not isinstance(estimates, dict):
        return None
    value = estimates.get("num_examples_effective")
    return int(value) if value is not None else None


def planned_shard_count(config: Any, plan: dict[str, Any]) -> int | None:
    count = planned_example_count(plan)
    if count is None:
        return None
    return max(
        1, (count + config.shard_size_examples - 1) // config.shard_size_examples
    )


__all__ = ["backend_config", "native_score_pass_operation", "streaming_config"]
