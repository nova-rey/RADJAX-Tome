"""C2--C5 integrated selection stage for canonical Path B."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from radjax_tome.builder.c6_integration import c5_records_for_delivery
from radjax_tome.builder.production_stages.evidence import native_file_evidence
from radjax_tome.fingerprint.corridor_budget import (
    CorridorBudgetPolicy,
    allocate_corridor_coverage,
    inspect_corridor_coverage_plan,
    write_corridor_coverage_plan,
)
from radjax_tome.fingerprint.corridor_claims import (
    CorridorGlobalClaimPolicy,
    claim_corridor_then_backfill_global,
    load_global_board_input,
    write_corridor_global_claim_result,
)
from radjax_tome.fingerprint.corridor_leaderboards import (
    CorridorLeaderboardPolicy,
    build_corridor_candidate_leaderboards,
    inspect_corridor_candidate_leaderboards,
    load_candidate_records_jsonl,
    write_corridor_candidate_leaderboards,
)
from radjax_tome.fingerprint.multi_role_selection import (
    build_multi_role_selected_exemplars,
    load_source_passports_for_coordinates,
    write_multi_role_selection_artifact,
)
from radjax_tome.io.json import write_json


class C6BudgetShortfallError(ValueError):
    """Stops Path B before selected rerun with retained diagnostics."""

    def __init__(self, diagnostic: dict[str, Any]) -> None:
        self.diagnostic = diagnostic
        super().__init__("C6 selected budget underfilled before selected rerun")


def prepare_c6_selection(config: Any, authorities: Mapping[str, Any]) -> dict[str, Any]:
    if config.total_selected_exemplar_budget is None:
        raise ValueError("C6 total_selected_exemplar_budget is required")
    c6_root = config.output_dir / "c6"
    c6_root.mkdir(parents=True, exist_ok=True)
    feature_path = Path(str(authorities["feature_path"]))
    feature_records = load_candidate_records_jsonl(
        feature_path, source_artifact_id=str(feature_path)
    )
    leaderboards = build_corridor_candidate_leaderboards(
        feature_records,
        CorridorLeaderboardPolicy(
            candidate_pool_cap=config.fingerprint_corridor_candidate_pool_cap
        ),
    )
    c2_path = write_corridor_candidate_leaderboards(
        leaderboards, c6_root / "corridor-leaderboards", overwrite=True
    )
    c2_summary = inspect_corridor_candidate_leaderboards(c2_path)
    plan = allocate_corridor_coverage(
        leaderboards,
        CorridorBudgetPolicy(
            total_selected_exemplar_budget=config.total_selected_exemplar_budget,
            corridor_budget_fraction=config.fingerprint_corridor_budget_fraction,
            corridor_budget_max=config.fingerprint_corridor_budget_max,
            corridor_mode_cap=config.fingerprint_corridor_mode_cap,
        ),
        source_leaderboard_provenance=c2_summary,
    )
    c3_path = write_corridor_coverage_plan(
        plan, c6_root / "coverage-plan", overwrite=True
    )
    c3_summary = inspect_corridor_coverage_plan(c3_path)
    c3_summary["mode_allocations"] = [
        {
            "mode_id": mode.corridor_mode_id,
            "allocated_slots": mode.allocated_slots,
            "zero_allocation_reason": mode.zero_allocation_reason,
        }
        for mode in plan.modes
    ]
    global_input = load_global_board_input(
        Path(str(authorities["global_board_supply_path"])), production_grade=True
    )
    provenance = global_input.source_provenance
    if (
        provenance.get("selector_policy") != "multi_leaderboard_exemplar_selector_v1"
        or provenance.get("selector_schema_version") != "exemplar_selection_manifest_v1"
    ):
        raise ValueError(
            "C6 global board supply must be exported by the production global selector"
        )
    global_supply = global_input.to_dict()
    claims = claim_corridor_then_backfill_global(
        leaderboards,
        plan,
        global_input,
        CorridorGlobalClaimPolicy(
            total_selected_exemplar_budget=config.total_selected_exemplar_budget,
            require_full_budget=False,
        ),
    )
    write_corridor_global_claim_result(claims, c6_root / "claims", overwrite=True)
    budget_diagnostics = c6_budget_diagnostics(
        config,
        claims=claims,
        leaderboards=leaderboards,
        plan=plan,
        global_supply=global_supply,
    )
    write_json(c6_root / "selection_budget_diagnostics.json", budget_diagnostics)
    if config.require_full_selected_budget and budget_diagnostics["budget_shortfall"]:
        raise C6BudgetShortfallError(budget_diagnostics)
    source_passports = load_source_passports_for_coordinates(
        Path(str(authorities["source_passports_path"])),
        {
            (coordinate.example_id, coordinate.position)
            for coordinate in claims.selected_coordinates
        },
    )
    selected = build_multi_role_selected_exemplars(
        claims, source_passports=source_passports
    )
    write_multi_role_selection_artifact(
        selected, c6_root / "multi-role-selection", overwrite=True
    )
    delivery_path = config.exemplar_delivery_path or "one_pass_pruned_candidate"
    return {
        "claims": claims,
        "selected": selected,
        "source_passports": [
            dict(record.source_passport) for record in selected.records
        ],
        "delivery_records": c5_records_for_delivery(
            selected, delivery_path=delivery_path
        ),
        "c2_summary": c2_summary,
        "c3_summary": c3_summary,
        "global_supply": global_supply,
        "budget_diagnostics": budget_diagnostics,
        "authorities": dict(authorities),
    }


def c6_budget_diagnostics(
    config: Any,
    *,
    claims: Any,
    leaderboards: Any,
    plan: Any,
    global_supply: Mapping[str, Any],
) -> dict[str, Any]:
    requested = int(config.total_selected_exemplar_budget or 0)
    final_count = len(claims.selected_coordinates)
    corridor_candidate_entries = [
        (candidate.candidate_id, candidate.position)
        for mode in leaderboards.modes
        for candidate in mode.candidates
    ]
    corridor_candidates = set(corridor_candidate_entries)
    global_candidate_entries = [
        (str(candidate["example_id"]), int(candidate["position"]))
        for board in global_supply.get("boards", [])
        if isinstance(board, Mapping)
        for candidate in board.get("candidates", [])
        if isinstance(candidate, Mapping)
    ]
    global_candidates = set(global_candidate_entries)
    corridor_claim_set = {
        (claim.example_id, claim.position) for claim in claims.corridor_claims
    }
    global_claim_set = {
        (claim.example_id, claim.position) for claim in claims.global_claims
    }
    intersection = corridor_claim_set & global_candidates
    union = corridor_claim_set | global_candidates
    corridor_budget_requested = sum(int(mode.allocated_slots) for mode in plan.modes)
    global_claims = len(claims.global_claims)
    corridor_claims = len(claims.corridor_claims)
    collisions = list(claims.collision_obligations)
    global_examined = sum(
        int(item.get("candidate_count_seen") or 0)
        for item in (claims.summary or {}).get("board_summaries", [])
        if isinstance(item, Mapping)
    )
    shortfall = max(0, requested - final_count)
    if not shortfall:
        reason = None
    elif len(corridor_candidates | global_candidates) < requested:
        reason = "insufficient_eligible_unique_candidates"
    elif global_claims < requested - corridor_claims:
        reason = "global_ranked_supply_exhaustion"
    elif collisions:
        reason = "deduplication_overlap_exhaustion"
    else:
        reason = "fingerprint_corridor_allocation_or_cap_exhaustion"
    return {
        "total_budget_requested": requested,
        "fingerprint_corridor_budget_requested": corridor_budget_requested,
        "fingerprint_corridor_candidates_eligible_unique": len(corridor_candidates),
        "fingerprint_corridor_claims_before_dedup": len(corridor_claim_set),
        "fingerprint_corridor_claims_accepted": corridor_claims,
        "global_supply_exported": len(global_candidates),
        "global_candidates_examined": global_examined,
        "global_claims_accepted": global_claims,
        "cross_role_duplicate_count": len(intersection),
        "accepted_cross_role_overlap": len(corridor_claim_set & global_claim_set),
        "within_role_duplicate_count": len(corridor_candidate_entries)
        - len(corridor_candidates)
        + len(global_candidate_entries)
        - len(global_candidates),
        "final_unique_selected_count": final_count,
        "budget_shortfall": shortfall,
        "budget_shortfall_reason": reason,
        "global_supply_remaining": max(0, len(global_candidates) - global_examined),
        "fingerprint_corridor_global_intersection_size": len(intersection),
        "fingerprint_corridor_global_jaccard": float(len(intersection))
        / float(max(len(union), 1)),
        "accepted_global_rank_depth": max(
            (claim.global_rank for claim in claims.global_claims), default=0
        ),
    }


def native_integrated_selection_operation(state: Any, inputs: Any) -> Any:
    from radjax_tome.builder.native_path_b.contracts import StageResult
    from radjax_tome.builder.native_path_b.selection import IntegratedSelectionHandoff

    state.progress.stage("corridor_global_selection")
    context = prepare_c6_selection(state.config, inputs.global_value)
    root = state.config.output_dir / "c6"
    c2 = native_file_evidence(
        "c2_corridor_candidate_leaderboards",
        (root / "corridor-leaderboards" / "manifest.json",),
        prior=inputs.fingerprint_evidence,
    )
    c3 = native_file_evidence(
        "c3_corridor_coverage_plan",
        (root / "coverage-plan" / "coverage_plan.json",),
        prior=c2,
    )
    c4 = native_file_evidence(
        "c4_corridor_global_claims",
        (root / "claims" / "claim_manifest.json",),
        prior=c3,
    )
    c5 = native_file_evidence(
        "c5_multi_role_selection",
        (root / "multi-role-selection" / "manifest.json",),
        prior=c4,
    )
    evidence = native_file_evidence(
        "integrated_selection",
        (
            root / "selection_budget_diagnostics.json",
            root / "multi-role-selection" / "manifest.json",
        ),
        prior=c5,
    )
    state.progress.memory_checkpoint("c2_c5_selection_complete")
    return StageResult(
        status="pass",
        value=IntegratedSelectionHandoff(
            value=context,
            stage_evidence=evidence,
            c2_evidence=c2,
            c3_evidence=c3,
            c4_evidence=c4,
            c5_evidence=c5,
        ),
        evidence=evidence,
    )
