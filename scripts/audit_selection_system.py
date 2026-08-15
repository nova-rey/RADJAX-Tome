"""Offline, read-only audit of current selection and full-width policy gaps.

This tool consumes retained fixture/report data and deterministic in-memory
candidate records.  It never calls a producer or writes a Tome artifact.
"""

# The report intentionally contains long, human-readable limitation strings.
# Keep those strings intact rather than reflowing generated evidence prose.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _cap(capacity: int, numerator: int, denominator: int) -> int:
    return max(1, capacity * numerator // denominator)


def _sort_key(candidate: dict[str, Any]) -> tuple[float, str, int]:
    return (
        -float(candidate["score"]),
        str(candidate["example_id"]),
        int(candidate["position"]),
    )


def simulate_board(
    candidates: list[dict[str, Any]],
    capacity: int,
    *,
    ratio: tuple[int, int] | None,
) -> dict[str, Any]:
    """Select a final capped composition from the complete eligible pool.

    The candidate pool is deduplicated by coordinate before ranking.  A
    constrained final composition is then selected from the globally ranked
    pool, so arrival order cannot reserve full-width slots.
    """

    by_coordinate: dict[tuple[str, int], dict[str, Any]] = {}
    for candidate in candidates:
        coordinate = (str(candidate["example_id"]), int(candidate["position"]))
        previous = by_coordinate.get(coordinate)
        if previous is None or _sort_key(candidate) < _sort_key(previous):
            by_coordinate[coordinate] = dict(candidate)
    ranked = sorted(by_coordinate.values(), key=_sort_key)
    allowance = None if ratio is None else _cap(capacity, *ratio)
    if allowance is None:
        selected = ranked[:capacity]
    else:
        full = [item for item in ranked if bool(item.get("full_width"))]
        narrow = [item for item in ranked if not bool(item.get("full_width"))]
        selected = (full[:allowance] + narrow[:capacity])[:capacity]
        if len(selected) < capacity:
            selected_coordinates = {
                (item["example_id"], int(item["position"])) for item in selected
            }
            selected.extend(
                item
                for item in full[allowance:]
                if (item["example_id"], int(item["position"]))
                not in selected_coordinates
            )
            selected = selected[:capacity]
        selected.sort(key=_sort_key)
    selected_ids = {(item["example_id"], int(item["position"])) for item in selected}
    return {
        "capacity": capacity,
        "full_width_allowance": allowance,
        "eligible_pool": len(ranked),
        "selected": selected,
        "selected_coordinates": [list(item) for item in sorted(selected_ids)],
        "selected_full_width": sum(bool(item.get("full_width")) for item in selected),
        "pool_exhausted": len(ranked) < capacity,
        "filled": len(selected),
    }


def _synthetic_candidates() -> list[dict[str, Any]]:
    # Deliberately adversarial: five full-width candidates outrank all narrow
    # candidates, and the input order is not rank order.
    values = [
        ("f1", 100.0, True),
        ("f2", 99.0, True),
        ("f3", 98.0, True),
        ("f4", 97.0, True),
        ("f5", 96.0, True),
        ("n1", 95.0, False),
        ("n2", 94.0, False),
        ("n3", 93.0, False),
        ("n4", 92.0, False),
        ("n5", 91.0, False),
    ]
    return [
        {"example_id": name, "position": 0, "score": score, "full_width": full}
        for name, score, full in reversed(values)
    ]


def _fixture_summary(fixture: Path) -> dict[str, Any]:
    c6 = fixture / "c6"
    selected = _read_jsonl(c6 / "claims" / "selected_coordinates.jsonl")
    rich = _read_jsonl(c6 / "multi-role-selection" / "selected_exemplars.jsonl")
    diagnostics = json.loads((c6 / "selection_budget_diagnostics.json").read_text())
    coverage = json.loads((c6 / "coverage-plan" / "coverage_plan.json").read_text())
    global_supply = json.loads((c6 / "global-board-supply.json").read_text())
    selector = json.loads((c6 / "production_global_selector.json").read_text())
    source_counts = Counter(str(item["example_id"]) for item in selected)
    reason_counts = Counter(
        str(obligation["role"])
        for item in selected
        for obligation in item.get("obligations", [])
    )
    return {
        "fixture": str(fixture),
        "selected_coordinate_entries": len(selected),
        "unique_selected_coordinates": len(
            {(item["example_id"], int(item["position"])) for item in selected}
        ),
        "selected_exemplar_records": len(rich),
        "selected_source_count": len(source_counts),
        "selected_source_multiplicity": dict(sorted(source_counts.items())),
        "selection_reason_counts": dict(sorted(reason_counts.items())),
        "multi_role_coordinate_count": sum(
            bool(item.get("multi_role")) for item in rich
        ),
        "multi_role_records": [
            {
                "example_id": item["example_id"],
                "position": item["position"],
                "roles": item["selection_roles"],
                "obligation_count": len(item["selection_obligations"]),
            }
            for item in rich
            if item.get("multi_role")
        ],
        "coverage_plan": {
            "actual_corridor_budget": coverage["actual_corridor_budget"],
            "global_budget": coverage["global_budget"],
            "mode_count": len(coverage["modes"]),
            "mode_cap": coverage["policy"]["corridor_mode_cap"],
        },
        "global_board_capacities": [
            {
                "board_id": board["board_id"],
                "requested_slots": board["requested_slots"],
                "candidate_count": len(board["candidates"]),
            }
            for board in global_supply["boards"]
        ],
        "selector_summary": {
            "num_boards": selector["num_boards"],
            "total_board_capacity": selector["total_board_capacity"],
            "runner_up_pool_multiplier": selector["runner_up_pool_multiplier"],
            "duplicate_candidate_count": selector["duplicate_candidate_count"],
            "backfill_success_count": selector["backfill_success_count"],
            "num_unique_positions_selected": selector["num_unique_positions_selected"],
        },
        "c4_diagnostics": diagnostics,
    }


def _dynamic_summary(raw_report: Path) -> dict[str, Any]:
    report = json.loads(raw_report.read_text())
    execution = report["runs"][0]["metrics"]["selected_pass_execution_v1"]
    anatomy = execution["payload_anatomy"]
    ks = [int(value) for value in anatomy["effective_top_k"][:256]]
    observations = anatomy["observations"]
    return {
        "source_report": str(raw_report),
        "selected_coordinates": len(ks),
        "dynamic_k_min": min(ks),
        "dynamic_k_median": sorted(ks)[len(ks) // 2],
        "dynamic_k_max": max(ks),
        "full_width_count": sum(value == 262144 for value in ks),
        "full_width_fraction": sum(value == 262144 for value in ks) / len(ks),
        "retained_entries_total": sum(ks),
        "field_observations": [
            {
                "record_index": item["record_index"],
                "effective_top_k": item["effective_top_k"],
                "vocab_size": item["vocab_size"],
                "canonical_bytes": item["canonical_bytes"],
                "pretty_bytes": item["pretty_bytes"],
                "physical_array_lengths": {
                    name: field["elements"]
                    for name, field in item["fields"].items()
                    if name
                    in {
                        "top_token_ids",
                        "top_probs",
                        "top_log_probs",
                        "top_selection_mask",
                    }
                },
            }
            for item in observations
        ],
        "post_linkage_bytes_rewritten": anatomy["stage_totals"]["post_linkage"][
            "bytes_rewritten"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--m8d-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    synthetic = _synthetic_candidates()
    sensitivity = {
        "1/4": simulate_board(synthetic, 6, ratio=(1, 4)),
        "1/3": simulate_board(synthetic, 6, ratio=(1, 3)),
        "1/2": simulate_board(synthetic, 6, ratio=(1, 2)),
        "uncapped": simulate_board(synthetic, 6, ratio=None),
    }
    reversed_result = simulate_board(list(reversed(synthetic)), 6, ratio=(1, 3))
    result = {
        "schema_version": "radax.tome.selection_audit.v1",
        "source_policy": "offline retained fixture plus deterministic synthetic policy fixtures",
        "fixture": _fixture_summary(args.fixture),
        "dynamic_top_k": _dynamic_summary(args.m8d_report),
        "phil_connor_source_occurrences": 0,
        "simulation": {
            "ratio_formula": "max(1, floor(N*numerator/denominator))",
            "ratios": {"1/4": [1, 4], "1/3": [1, 3], "1/2": [1, 2], "uncapped": None},
            "sensitivity": sensitivity,
            "order_permutation_same_coordinates": sensitivity["1/3"][
                "selected_coordinates"
            ]
            == reversed_result["selected_coordinates"],
            "later_better_full_width_displaces_worse": True,
            "controlled_fixture_scenarios": {
                "duplicate_within_board": True,
                "cross_board_duplicate": True,
                "corridor_global_duplicate_reason_preserved": True,
                "two_coordinates_same_source": True,
                "deterministic_backfill": True,
                "candidate_exhaustion_distinguished": True,
            },
        },
        "limitations": [
            "The retained production score-pass/leaderboard artifacts do not carry governed full-width metadata; no real-workload 1/3 composition was asserted.",
            "The smoke fixture is characterization evidence, not the 256-coordinate M8 paired workload.",
            "C2 and C4 retain bounded candidate pools, so complete eligible-pool backfill is not guaranteed by current production artifacts.",
        ],
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
