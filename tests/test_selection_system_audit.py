from radjax_tome.builder.full_width_policy import (
    FullWidthCompositionPolicy,
    select_final_composition,
)
from scripts.audit_selection_system import (
    _controlled_fixture_scenarios,
    _synthetic_candidates,
    simulate_board,
)


def test_authoritative_full_width_ratio_is_exact_and_final():
    policy = FullWidthCompositionPolicy()
    assert policy.to_dict() == {"numerator": 1, "denominator": 3}
    candidates = [
        {"example_id": f"f{i}", "position": 0, "score": 100 - i, "full_width": True}
        for i in range(5)
    ] + [
        {"example_id": f"n{i}", "position": 0, "score": 95 - i, "full_width": False}
        for i in range(5)
    ]
    selected = select_final_composition(list(reversed(candidates)), 6, policy=policy)
    assert sum(item["full_width"] for item in selected) == 2
    assert {item["example_id"] for item in selected} == {
        "f0",
        "f1",
        "n0",
        "n1",
        "n2",
        "n3",
    }


def test_exact_one_third_cap_is_final_composition_and_order_independent():
    candidates = _synthetic_candidates()
    forward = simulate_board(candidates, 6, ratio=(1, 3))
    reverse = simulate_board(list(reversed(candidates)), 6, ratio=(1, 3))
    assert forward["full_width_allowance"] == 2
    assert forward["selected_full_width"] == 2
    assert forward["filled"] == 6
    assert forward["selected_coordinates"] == reverse["selected_coordinates"]


def test_cap_sensitivity_and_full_width_competition():
    candidates = _synthetic_candidates()
    assert simulate_board(candidates, 6, ratio=(1, 4))["selected_full_width"] == 1
    assert simulate_board(candidates, 6, ratio=(1, 2))["selected_full_width"] == 3
    assert simulate_board(candidates, 6, ratio=None)["selected_full_width"] == 5
    result = simulate_board(list(reversed(candidates)), 6, ratio=(1, 3))
    full_ids = [item["example_id"] for item in result["selected"] if item["full_width"]]
    assert full_ids == ["f1", "f2"]


def test_full_width_cap_is_not_violated_when_narrow_pool_is_exhausted():
    full_only = [item for item in _synthetic_candidates() if item["full_width"]]
    full_only.append(
        {"example_id": "f6", "position": 0, "score": 90.0, "full_width": True}
    )
    result = simulate_board(full_only, 6, ratio=(1, 3))
    assert result["selected_full_width"] == 2
    assert result["filled"] == 2
    assert result["category_shortfall"] == 4
    assert not result["pool_exhausted"]


def test_controlled_audit_scenarios_are_derived():
    assert all(_controlled_fixture_scenarios().values())
