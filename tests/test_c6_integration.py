from __future__ import annotations

import json
from pathlib import Path

import pytest

from radjax_tome.builder.c6_integration import (
    C6IntegrationError,
    build_corridor_coverage_report,
    c5_records_for_delivery,
    export_corridor_candidate_features,
    export_production_global_board_supply,
    validate_integrated_selection_contract,
)
from radjax_tome.builder.exemplar_selection import ExemplarCandidate, select_exemplars
from radjax_tome.fingerprint.corridor_leaderboards import load_candidate_records_jsonl
from radjax_tome.fingerprint.multi_role_selection import (
    build_multi_role_selected_exemplars,
)
from radjax_tome.tome.golden_fixture import build_production_contract_fixture
from tests.helpers.subprocess import run_cli
from tests.test_multi_role_selected_exemplars import _claims

ROOT = Path(__file__).resolve().parents[1]


def test_c6_coverage_recomputes_unique_and_obligation_counts() -> None:
    claims = _claims()
    selected = build_multi_role_selected_exemplars(claims)
    report = build_corridor_coverage_report(claims, selected)

    assert report["selection_integration_policy"] == (
        "corridor_first_global_backfill_v1"
    )
    assert report["selected_unique_count"] == len(selected.records)
    assert report["selected_obligation_count"] == 5
    assert report["corridor_budget_actual"] == len(claims.corridor_claims)
    assert report["collision_count"] == len(claims.collision_obligations)


def test_c6_strict_contract_compares_legacy_payload_and_curriculum_sets() -> None:
    claims = _claims()
    selected = build_multi_role_selected_exemplars(claims)
    legacy = list(selected.legacy_records)
    curriculum = [
        {
            "selected_example_id": item["selected_example_id"],
            "selected_position": item["selected_position"],
            "curriculum_board": "primary",
            "payload_key": item["payload_identity"]["payload_key"],
        }
        for item in legacy
    ]
    result = validate_integrated_selection_contract(
        claims,
        selected,
        legacy_records=legacy,
        payload_records=legacy,
        curriculum_records=curriculum,
        package_records=legacy,
        production_grade=False,
    )

    assert result["status"] == "pass"
    assert result["selected_unique_count"] == len(selected.records)

    broken = [*legacy, legacy[0]]
    failed = validate_integrated_selection_contract(
        claims,
        selected,
        legacy_records=broken,
        production_grade=False,
    )
    assert failed["status"] == "fail"
    assert any("duplicate" in item or "count" in item for item in failed["blockers"])


def test_c6_curriculum_routes_allow_multi_board_routes_but_reject_duplicates() -> None:
    claims = _claims()
    selected = build_multi_role_selected_exemplars(claims)
    legacy = list(selected.legacy_records)
    routes = [
        {
            "selected_example_id": item["selected_example_id"],
            "selected_position": item["selected_position"],
            "curriculum_board": "primary",
        }
        for item in legacy
    ]
    routes.append({**routes[0], "curriculum_board": "auxiliary"})
    passed = validate_integrated_selection_contract(
        claims,
        selected,
        legacy_records=legacy,
        payload_records=legacy,
        curriculum_records=routes,
        production_grade=False,
    )
    assert passed["status"] == "pass"

    failed = validate_integrated_selection_contract(
        claims,
        selected,
        curriculum_records=[*routes, routes[0]],
        production_grade=False,
    )
    assert failed["status"] == "fail"
    assert any("duplicate board routes" in item for item in failed["blockers"])


def test_production_contract_requires_real_passports() -> None:
    claims = _claims()
    selected = build_multi_role_selected_exemplars(claims)
    result = validate_integrated_selection_contract(
        claims,
        selected,
        legacy_records=list(selected.legacy_records),
        production_grade=True,
    )

    assert result["status"] == "fail"
    assert any("real source passports" in item for item in result["blockers"])


def test_c5_delivery_projection_requires_source_coordinates() -> None:
    claims = _claims()
    selected = build_multi_role_selected_exemplars(claims)

    with pytest.raises(C6IntegrationError, match="missing delivery fields"):
        c5_records_for_delivery(selected, delivery_path="two_pass_rerun_selected")


def test_global_export_rejects_development_selector_manifest() -> None:
    manifest = {
        "schema_version": "exemplar_selection_manifest_v1",
        "selection_policy": "multi_leaderboard_exemplar_selector_v1",
        "production_global_selector": False,
    }

    with pytest.raises(C6IntegrationError, match="development global selector"):
        export_production_global_board_supply(
            manifest,
            source_artifact_id="selector",
            source_artifact_hash="a" * 64,
        )


def test_global_export_preserves_production_ranked_supply() -> None:
    manifest = {
        "schema_version": "exemplar_selection_manifest_v1",
        "selection_policy": "multi_leaderboard_exemplar_selector_v1",
        "production_global_selector": True,
        "boards": [
            {
                "board_id": "global_max_entropy",
                "priority": 0,
                "capacity": 2,
                "ranked_candidates": [
                    {"example_id": "a", "selected_position": 1, "score": 3.0},
                    {"example_id": "b", "selected_position": 2, "score": 2.0},
                ],
            }
        ],
    }
    exported = export_production_global_board_supply(
        manifest,
        source_artifact_id="selector",
        source_artifact_hash="a" * 64,
    )

    assert exported["source_provenance"]["production_grade"] is True
    assert exported["boards"][0]["candidates"][1]["rank"] == 2


def test_real_selector_can_export_ranked_production_supply() -> None:
    candidates = tuple(
        ExemplarCandidate(
            example_id=example_id,
            source_shard_id=0,
            source_row=index,
            selected_position=2,
            candidate_positions=(2,),
            sequence_length=8,
            capture_mode="one_pass_candidate",
            source_policy="dynamic_cascaded_soft_labels_v1",
            score_fields={
                "max_entropy": score,
                "mean_entropy": score,
                "confidence": 0.2,
                "selected_position_entropy": score,
                "position_bucket": 0.0,
                "length_bucket": 0.0,
            },
            payload_ref={"kind": "one_pass_candidate_v1"},
        )
        for index, (example_id, score) in enumerate(
            (("example-a", 3.0), ("example-b", 2.0))
        )
    )
    manifest = select_exemplars(
        candidates,
        capture_mode="one_pass_candidate",
        fulfillment_policy="select_from_existing_capture",
        board_capacity=2,
        created_at="2026-07-11T00:00:00+00:00",
        production_global_selector=True,
    )

    assert manifest["production_global_selector"] is True
    board = next(
        board
        for board in manifest["boards"]
        if board["board_id"] == "global_max_entropy"
    )
    ranked = board["ranked_candidates"]
    assert [item["example_id"] for item in ranked] == ["example-a", "example-b"]


def test_production_cli_exposes_c6_policy_flags() -> None:
    result = run_cli(ROOT, "production-build", "--help")

    assert result.returncode == 0, result.stderr
    for flag in (
        "--selection-integration-policy",
        "--total-selected-exemplar-budget",
        "--fingerprint-corridor-budget-fraction",
        "--fingerprint-corridor-budget-max",
        "--fingerprint-corridor-mode-cap",
        "--fingerprint-corridor-candidate-pool-cap",
        "--require-full-selected-budget",
        "--allow-selected-underfill",
    ):
        assert flag in result.stdout


def test_cli_exposes_production_global_supply_exporter() -> None:
    result = run_cli(ROOT, "export-production-global-board-supply", "--help")

    assert result.returncode == 0, result.stderr
    assert "--selector-manifest" in result.stdout
    assert "Development manifests are rejected" in result.stdout


def test_c6_exports_strict_features_from_current_corridor_artifact(
    tmp_path: Path,
) -> None:
    artifact = build_production_contract_fixture(tmp_path / "artifact")
    features = export_corridor_candidate_features(
        artifact_dir=artifact,
        output_dir=artifact / "c6" / "corridor-features",
    )
    records = list(load_candidate_records_jsonl(features))

    assert features.is_file()
    assert records
    assert records[0].feature_provenance.fidelity == "derived"
    assert 0.0 <= records[0].features.membership_strength <= 1.0
    assert 0.0 <= records[0].features.core_distance <= 1.0
    manifest = json.loads((features.parent / "manifest.json").read_text())
    assert manifest["assignment_manifest_sha256"].startswith("sha256:")
    assert manifest["modes_sha256"].startswith("sha256:")
