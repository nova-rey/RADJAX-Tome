from pathlib import Path

from radjax_tome.builder.production_stages.preflight import assess_production_preflight


def test_preflight_destination_matrix_is_mutation_free(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    assessment = assess_production_preflight(missing)
    assert (assessment.status, assessment.action) == ("pass", "create")
    assert not missing.exists()
    empty = tmp_path / "empty"
    empty.mkdir()
    assert assess_production_preflight(empty).action == "use"
    (empty / "keep").write_text("preserve")
    blocked = assess_production_preflight(empty)
    assert blocked.status == "fail"
    assert (empty / "keep").read_text() == "preserve"


def test_preflight_rejects_unsafe_combinations(tmp_path: Path) -> None:
    both = assess_production_preflight(tmp_path / "out", resume=True, overwrite=True)
    assert both.status == "fail"
    assert assess_production_preflight(tmp_path / "none", resume=True).status == "fail"
