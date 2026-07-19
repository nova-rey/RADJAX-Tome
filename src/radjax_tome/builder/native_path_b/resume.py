"""Read-only native Path-B resume classification from canonical evidence.

The resolver does not create a checkpoint format and deliberately does not run
the existing compatibility migration.  It only identifies the first native
stage that must run (or repair its evidence) from the artifacts already on
disk.  Callers that support legacy metadata retain responsibility for applying
their established migration before executing the returned action.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from radjax_tome.builder.native_path_b.api import CanonicalPathBConfig
from radjax_tome.builder.native_path_b.contracts import (
    EvidenceDiagnostic,
    FileHash,
    PriorStageProof,
    StageEvidence,
    StageFailure,
)
from radjax_tome.builder.native_path_b.evidence import (
    read_score_surface_corridor_evidence,
    read_selected_artifact_corridor_evidence,
)

PREFLIGHT_STAGE = "preflight"
SCORE_PASS_STAGE = "score_pass"
EARLY_CORRIDOR_STAGE = "score_surface_corridor_materialization"
FINGERPRINT_AUTHORITY_STAGE = "fingerprint_corridor_selection_authority_export"
GLOBAL_AUTHORITY_STAGE = "global_authority_export"
INTEGRATED_SELECTION_STAGE = "integrated_selection"
SELECTED_DELIVERY_STAGE = "selected_delivery_rerun"
LATE_CORRIDOR_STAGE = "selected_artifact_corridor_finalization"
ARTIFACT_ASSEMBLY_STAGE = "artifact_assembly"
VALIDATION_STAGE = "validation_linkage"
RECONCILIATION_STAGE = "reconciliation_cover"
FINAL_REPORTING_STAGE = "final_reporting"


@dataclass(frozen=True)
class NativePathBResumeResolution:
    """The next canonical Path-B action, derived without writing artifacts."""

    stage: str | None
    complete: bool
    failure: StageFailure | None
    evidence: StageEvidence | None = None

    def __post_init__(self) -> None:
        if self.complete:
            if self.stage is not None or self.failure is not None:
                raise ValueError("a complete resume resolution has no next stage")
        elif self.stage is None or self.failure is None:
            raise ValueError("an incomplete resume resolution needs stage and failure")


def resolve_native_path_b_resume(
    output_dir: Path,
    *,
    config: CanonicalPathBConfig | None = None,
    run_manifest_path: Path | None = None,
) -> NativePathBResumeResolution:
    """Return the earliest incomplete or invalid native Path-B stage.

    ``config`` is optional because artifact inspection remains useful to
    compatibility callers.  When it is supplied, the resolver verifies the
    stored C6 selection-integration hash against the current request.  It does
    not attempt the broader production input/provenance checks; those remain
    preflight ownership and require their existing production readers.
    """

    preflight_path = output_dir / "run_plan.json"
    preflight = _read_object(preflight_path)
    if isinstance(preflight, StageFailure):
        return _next(
            PREFLIGHT_STAGE,
            "preflight_evidence_unavailable",
            preflight.blockers,
            remediation="restore or rerun the native Path-B preflight plan",
        )
    if preflight.get("status") not in {"pass", "warn"}:
        return _next(
            PREFLIGHT_STAGE,
            "preflight_incomplete",
            ("run_plan.status must be pass or warn before score resume",),
            diagnostics=(
                EvidenceDiagnostic("run_plan_status", str(preflight.get("status"))),
            ),
            remediation="rerun preflight and resolve its blockers",
        )

    manifest_path = run_manifest_path or output_dir / "run_manifest.json"
    manifest = _read_object(manifest_path)
    if isinstance(manifest, StageFailure):
        return _next(
            SCORE_PASS_STAGE,
            "score_pass_evidence_unavailable",
            manifest.blockers,
            remediation="start or restore the streaming score pass manifest",
        )
    score_failure = _validate_score_manifest(manifest)
    if score_failure is not None:
        return _from_failure(score_failure)
    score_evidence = _file_evidence(SCORE_PASS_STAGE, (manifest_path,))

    late = read_selected_artifact_corridor_evidence(output_dir)
    if late.status == "pass":
        assert late.evidence is not None
        config_failure = _config_binding_failure(output_dir, config)
        if config_failure is not None:
            return _from_failure(config_failure)
        return _after_final_corridor(output_dir, late.evidence)

    early = read_score_surface_corridor_evidence(
        output_dir,
        run_manifest_path=manifest_path,
    )
    if early.status == "pass":
        assert early.evidence is not None
        return _after_provisional_corridor(output_dir, early.evidence, config)

    if _passing_delivery_exists(output_dir):
        assert late.failure is not None
        return _from_failure(late.failure)

    assert early.failure is not None
    if early.failure.reason == "provisional_corridor_evidence_overwritten":
        return _next(
            SELECTED_DELIVERY_STAGE,
            "selected_delivery_pending",
            ("final corridor evidence is present without a passing delivery proof",),
            remediation="complete selected delivery before final corridor validation",
            evidence=score_evidence,
        )
    return _from_failure(early.failure)


def _after_provisional_corridor(
    output_dir: Path,
    early_evidence: StageEvidence,
    config: CanonicalPathBConfig | None,
) -> NativePathBResumeResolution:
    authority_path = output_dir / "c6" / "authority_manifest.json"
    authority = _read_object(authority_path)
    if isinstance(authority, StageFailure):
        return _next(
            FINGERPRINT_AUTHORITY_STAGE,
            "selection_authority_evidence_unavailable",
            authority.blockers,
            remediation="export C6 selection authority from provisional corridors",
            evidence=early_evidence,
        )
    authority_failure = _validate_authority(authority)
    if authority_failure is not None:
        return _next(
            FINGERPRINT_AUTHORITY_STAGE,
            authority_failure.reason,
            authority_failure.blockers,
            remediation=authority_failure.remediation,
            evidence=early_evidence,
        )
    config_failure = _config_binding_failure(output_dir, config, authority=authority)
    if config_failure is not None:
        return _from_failure(config_failure)
    global_failure = _global_authority_failure(output_dir, authority)
    if global_failure is not None:
        return _next(
            GLOBAL_AUTHORITY_STAGE,
            global_failure.reason,
            global_failure.blockers,
            remediation=global_failure.remediation,
            evidence=early_evidence,
        )
    selection_failure = _selection_failure(output_dir)
    if selection_failure is not None:
        return _next(
            INTEGRATED_SELECTION_STAGE,
            selection_failure.reason,
            selection_failure.blockers,
            remediation=selection_failure.remediation,
            evidence=early_evidence,
        )
    return _next(
        SELECTED_DELIVERY_STAGE,
        "selected_delivery_pending",
        ("C2-C5 selection evidence exists but selected delivery is not final",),
        remediation="rerun and materialize the selected exemplar payloads",
        evidence=early_evidence,
    )


def _after_final_corridor(
    output_dir: Path,
    final_evidence: StageEvidence,
) -> NativePathBResumeResolution:
    assembly = _artifact_assembly_evidence(output_dir, final_evidence)
    if isinstance(assembly, StageFailure):
        return _next(
            ARTIFACT_ASSEMBLY_STAGE,
            assembly.reason,
            assembly.blockers,
            remediation=assembly.remediation,
            evidence=final_evidence,
        )
    return _after_artifact_assembly(output_dir, assembly)


def _after_artifact_assembly(
    output_dir: Path,
    assembly_evidence: StageEvidence,
) -> NativePathBResumeResolution:
    validation_path = output_dir / "validation_report.json"
    validation = _read_object(validation_path)
    if isinstance(validation, StageFailure):
        return _next(
            VALIDATION_STAGE,
            "validation_evidence_unavailable",
            validation.blockers,
            remediation="run strict artifact validation and selected-linkage audit",
            evidence=assembly_evidence,
        )
    if validation.get("status") != "pass":
        return _next(
            VALIDATION_STAGE,
            "validation_incomplete",
            ("validation_report.status must be pass",),
            diagnostics=(
                EvidenceDiagnostic("validation_status", str(validation.get("status"))),
            ),
            remediation="repair validation blockers before finalization",
            evidence=assembly_evidence,
        )

    reconciliation_path = (
        output_dir / "reports" / "c6_integrated_selection_validation.json"
    )
    reconciliation = _read_object(reconciliation_path)
    if isinstance(reconciliation, StageFailure):
        return _next(
            RECONCILIATION_STAGE,
            "reconciliation_evidence_unavailable",
            reconciliation.blockers,
            remediation="write C6 reconciliation and coverage evidence",
            evidence=assembly_evidence,
        )
    if reconciliation.get("status") != "pass":
        return _next(
            RECONCILIATION_STAGE,
            "reconciliation_incomplete",
            ("c6_integrated_selection_validation.status must be pass",),
            diagnostics=(
                EvidenceDiagnostic(
                    "reconciliation_status", str(reconciliation.get("status"))
                ),
            ),
            remediation="repair C6 reconciliation blockers before reporting",
            evidence=assembly_evidence,
        )

    report_path = output_dir / "production_build_report.json"
    report = _read_object(report_path)
    if isinstance(report, StageFailure):
        return _next(
            FINAL_REPORTING_STAGE,
            "final_reporting_evidence_unavailable",
            report.blockers,
            remediation="render the terminal production report",
            evidence=assembly_evidence,
        )
    if report.get("status") != "pass":
        return _next(
            FINAL_REPORTING_STAGE,
            "final_reporting_incomplete",
            ("production_build_report.status must be pass",),
            diagnostics=(
                EvidenceDiagnostic("report_status", str(report.get("status"))),
            ),
            remediation="complete final reporting after resolving its blockers",
            evidence=assembly_evidence,
        )
    terminal_paths = (validation_path, reconciliation_path, report_path)
    terminal_evidence = StageEvidence(
        stage=FINAL_REPORTING_STAGE,
        paths=terminal_paths,
        hashes=_file_evidence(FINAL_REPORTING_STAGE, terminal_paths).hashes,
        counts=(),
        prior_stage_proof=PriorStageProof(
            stage=ARTIFACT_ASSEMBLY_STAGE,
            paths=assembly_evidence.paths,
            hashes=assembly_evidence.hashes,
            counts=assembly_evidence.counts,
        ),
    )
    return NativePathBResumeResolution(
        stage=None,
        complete=True,
        failure=None,
        evidence=terminal_evidence,
    )


def _artifact_assembly_evidence(
    output_dir: Path,
    final_evidence: StageEvidence,
) -> StageEvidence | StageFailure:
    """Require promoted selected payload/index evidence after late corridors."""

    selected_dir = output_dir / "selected_exemplars"
    payload_shards = tuple(sorted(selected_dir.glob("selected-exemplars-*.json")))
    paths = (
        output_dir / "leaderboards" / "selected_exemplars.json",
        selected_dir / "payload_index.json",
        *payload_shards,
    )
    unavailable = [
        path for path in paths if isinstance(_read_object(path), StageFailure)
    ]
    if not payload_shards:
        unavailable.append(selected_dir / "selected-exemplars-*.json")
    if unavailable:
        return _failure(
            ARTIFACT_ASSEMBLY_STAGE,
            "artifact_assembly_evidence_unavailable",
            tuple(
                f"missing or invalid assembled selected artifact: {path}"
                for path in unavailable
            ),
            remediation=(
                "promote selected payloads and write their index after final corridors"
            ),
        )
    return StageEvidence(
        stage=ARTIFACT_ASSEMBLY_STAGE,
        paths=paths,
        hashes=_file_evidence(ARTIFACT_ASSEMBLY_STAGE, paths).hashes,
        counts=(),
        prior_stage_proof=PriorStageProof(
            stage=LATE_CORRIDOR_STAGE,
            paths=final_evidence.paths,
            hashes=final_evidence.hashes,
            counts=final_evidence.counts,
        ),
    )


def _validate_score_manifest(manifest: Mapping[str, Any]) -> StageFailure | None:
    if manifest.get("status") != "complete":
        return _failure(
            SCORE_PASS_STAGE,
            "score_pass_incomplete",
            ("run_manifest.status must be complete",),
            remediation="resume the streaming score pass",
        )
    pairs = (
        ("num_examples_completed", "num_examples_planned"),
        ("num_shards_completed", "num_shards_planned"),
    )
    for completed_key, planned_key in pairs:
        completed = _non_negative_integer(manifest.get(completed_key))
        planned = _non_negative_integer(manifest.get(planned_key))
        if completed is None or planned is None:
            return _failure(
                SCORE_PASS_STAGE,
                "score_pass_evidence_invalid",
                (f"run_manifest.{completed_key} and {planned_key} must be integers",),
                remediation="restore the complete streaming run manifest",
            )
        if completed != planned:
            return _failure(
                SCORE_PASS_STAGE,
                "score_pass_checkpoint_incomplete",
                (f"run_manifest.{completed_key} must equal {planned_key}",),
                remediation="resume the incomplete streaming score pass",
            )
    return None


def _validate_authority(authority: Mapping[str, Any]) -> StageFailure | None:
    authority_hash = authority.get("score_pass_authority_hash")
    if not isinstance(authority_hash, str) or not authority_hash:
        return _failure(
            FINGERPRINT_AUTHORITY_STAGE,
            "selection_authority_evidence_invalid",
            ("authority_manifest.score_pass_authority_hash must be non-empty",),
            remediation="re-export C6 selection authority from score evidence",
        )
    return None


def _global_authority_failure(
    output_dir: Path,
    authority: Mapping[str, Any],
) -> StageFailure | None:
    paths = authority.get("paths")
    if not isinstance(paths, Mapping):
        return _failure(
            GLOBAL_AUTHORITY_STAGE,
            "global_authority_evidence_unavailable",
            ("authority_manifest.paths must describe global authority files",),
            remediation="export the global authority and source passports",
        )
    required = ("global_board_supply", "source_passports")
    missing = [
        name
        for name in required
        if not isinstance(paths.get(name), str)
        or not (output_dir / str(paths[name])).is_file()
    ]
    if missing:
        return _failure(
            GLOBAL_AUTHORITY_STAGE,
            "global_authority_evidence_unavailable",
            tuple(f"missing global authority artifact: {name}" for name in missing),
            remediation="export the global authority and source passports",
        )
    return None


def _selection_failure(output_dir: Path) -> StageFailure | None:
    required = (
        output_dir / "c6" / "coverage-plan" / "coverage_plan.json",
        output_dir / "c6" / "claims" / "claim_manifest.json",
        output_dir / "c6" / "multi-role-selection" / "manifest.json",
    )
    unavailable = [
        path for path in required if isinstance(_read_object(path), StageFailure)
    ]
    if unavailable:
        return _failure(
            INTEGRATED_SELECTION_STAGE,
            "integrated_selection_evidence_unavailable",
            tuple(f"missing or invalid C2-C5 evidence: {path}" for path in unavailable),
            remediation="complete integrated C2-C5 selection",
        )
    return None


def _passing_delivery_exists(output_dir: Path) -> bool:
    delivery = _read_object(output_dir / "delivery_report.json")
    return not isinstance(delivery, StageFailure) and delivery.get("status") == "pass"


def _config_binding_failure(
    output_dir: Path,
    config: CanonicalPathBConfig | None,
    *,
    authority: Mapping[str, Any] | None = None,
) -> StageFailure | None:
    if config is None:
        return None
    expected = _selection_integration_hash(config.source_config)
    if expected is None:
        return None
    documents: list[tuple[str, Mapping[str, Any]]] = []
    if authority is not None:
        documents.append(("authority_manifest", authority))
    else:
        artifact = _read_object(output_dir / "c6" / "authority_manifest.json")
        if not isinstance(artifact, StageFailure):
            documents.append(("authority_manifest", artifact))
    emission = _read_object(output_dir / "emission_config.json")
    if not isinstance(emission, StageFailure):
        documents.append(("emission_config", emission))
    mismatches = tuple(
        f"{name}.selection_integration_config_hash does not match the request"
        for name, document in documents
        if document.get("selection_integration_config_hash") != expected
    )
    if not mismatches:
        return None
    return _failure(
        PREFLIGHT_STAGE,
        "selection_integration_config_mismatch",
        mismatches,
        diagnostics=(EvidenceDiagnostic("expected_selection_hash", expected),),
        remediation="restart from preflight with configuration matching C6 evidence",
    )


def _selection_integration_hash(source: Any) -> str | None:
    names = (
        "selection_integration_policy",
        "teacher_model",
        "tokenizer_id",
        "dataset_path",
        "corpus_manifest_path",
        "target_policy",
        "sequence_length",
        "vocab_size",
        "top_k",
        "num_buckets",
        "dynamic_top_k_min",
        "dynamic_top_k_max",
        "dynamic_mass_threshold",
        "selected_rerun_batch_size",
        "total_selected_exemplar_budget",
        "fingerprint_corridor_budget_fraction",
        "fingerprint_corridor_budget_max",
        "fingerprint_corridor_mode_cap",
        "fingerprint_corridor_candidate_pool_cap",
        "require_full_selected_budget",
        "exemplar_delivery_path",
    )
    if any(not hasattr(source, name) for name in names):
        return None
    tokenizer_id = source.tokenizer_id or source.teacher_model
    corridor_budget_fraction = source.fingerprint_corridor_budget_fraction
    corridor_candidate_pool_cap = source.fingerprint_corridor_candidate_pool_cap
    payload = {
        "selection_integration_policy": source.selection_integration_policy,
        "teacher_model": source.teacher_model,
        "tokenizer_id": tokenizer_id,
        "dataset_path": str(source.dataset_path),
        "corpus_manifest_path": str(source.corpus_manifest_path),
        "target_policy": source.target_policy,
        "sequence_length": source.sequence_length,
        "vocab_size": source.vocab_size,
        "top_k": source.top_k,
        "num_buckets": source.num_buckets,
        "dynamic_top_k_min": source.dynamic_top_k_min,
        "dynamic_top_k_max": source.dynamic_top_k_max,
        "dynamic_mass_threshold": source.dynamic_mass_threshold,
        "selected_rerun_batch_size": source.selected_rerun_batch_size,
        "total_selected_exemplar_budget": source.total_selected_exemplar_budget,
        "fingerprint_corridor_budget_fraction": corridor_budget_fraction,
        "fingerprint_corridor_budget_max": source.fingerprint_corridor_budget_max,
        "fingerprint_corridor_mode_cap": source.fingerprint_corridor_mode_cap,
        "fingerprint_corridor_candidate_pool_cap": corridor_candidate_pool_cap,
        "require_full_selected_budget": source.require_full_selected_budget,
        "c2_schema": "radjax.c2_corridor_candidate_leaderboards.v1",
        "c3_schema": "radjax.c3_corridor_coverage_plan.v1",
        "c4_schema": "radjax.c4_corridor_global_claims.v1",
        "c5_schema": "radjax.multi_role_selected_exemplar.v1",
        "delivery_path": source.exemplar_delivery_path,
    }
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(body).hexdigest()


def _read_object(path: Path) -> dict[str, Any] | StageFailure:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return _failure(
            "evidence_reader",
            "json_unavailable_or_invalid",
            (f"cannot read JSON evidence {path}: {exc}",),
            remediation="restore the existing JSON evidence",
        )
    if not isinstance(payload, dict):
        return _failure(
            "evidence_reader",
            "json_object_required",
            (f"JSON evidence must be an object: {path}",),
            remediation="restore the existing JSON object evidence",
        )
    return payload


def _file_evidence(stage: str, paths: tuple[Path, ...]) -> StageEvidence:
    return StageEvidence(
        stage=stage,
        paths=paths,
        hashes=tuple(
            FileHash(
                path=path,
                sha256="sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for path in paths
        ),
        counts=(),
    )


def _non_negative_integer(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _failure(
    stage: str,
    reason: str,
    blockers: tuple[str, ...],
    *,
    diagnostics: tuple[EvidenceDiagnostic, ...] = (),
    remediation: str | None = None,
) -> StageFailure:
    return StageFailure(
        stage=stage,
        reason=reason,
        blockers=blockers,
        diagnostics=diagnostics,
        resumable=True,
        remediation=remediation,
    )


def _next(
    stage: str,
    reason: str,
    blockers: tuple[str, ...],
    *,
    diagnostics: tuple[EvidenceDiagnostic, ...] = (),
    remediation: str | None = None,
    evidence: StageEvidence | None = None,
) -> NativePathBResumeResolution:
    return NativePathBResumeResolution(
        stage=stage,
        complete=False,
        failure=_failure(
            stage,
            reason,
            blockers,
            diagnostics=diagnostics,
            remediation=remediation,
        ),
        evidence=evidence,
    )


def _from_failure(failure: StageFailure) -> NativePathBResumeResolution:
    return NativePathBResumeResolution(
        stage=failure.stage,
        complete=False,
        failure=failure,
    )
