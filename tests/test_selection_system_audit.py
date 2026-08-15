from scripts.audit_selection_system import _synthetic_candidates, simulate_board


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
