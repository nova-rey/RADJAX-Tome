from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from radjax_tome.fingerprint.corridor_duckdb import (
    DuckDBCandidateStore,
    build_corridor_candidate_leaderboards_duckdb,
)
from radjax_tome.fingerprint.corridor_leaderboards import (
    CorridorLeaderboardError,
    CorridorLeaderboardPolicy,
    build_corridor_candidate_leaderboards,
    validate_corridor_candidate_leaderboards,
    write_corridor_candidate_leaderboards,
)
from tests.test_corridor_candidate_leaderboards import _record


def _policy() -> CorridorLeaderboardPolicy:
    return CorridorLeaderboardPolicy(retain_complete_candidate_pool=True)


def _coords(artifact):
    return [
        [(x.candidate_id, x.position) for x in mode.candidates]
        for mode in artifact.modes
    ]


def test_duckdb_matches_reference_across_batches(tmp_path: Path) -> None:
    records = [
        _record(f"candidate-{i}", mode=i % 3, difficulty=i / 100.0) for i in range(30)
    ]
    expected = build_corridor_candidate_leaderboards(records, _policy())
    actual = build_corridor_candidate_leaderboards_duckdb(
        records, _policy(), scratch_dir=tmp_path / "db", ingestion_batch_size=5
    )
    assert _coords(actual) == _coords(expected)
    assert actual.summary == expected.summary
    actual.backend.close()


def test_duplicate_coordinates_collapse_and_conflicts_reject(tmp_path: Path) -> None:
    record = _record("same", mode=1)
    actual = build_corridor_candidate_leaderboards_duckdb(
        [record, record], _policy(), scratch_dir=tmp_path / "dup"
    )
    assert actual.summary["duplicates_collapsed"] == 1
    actual.backend.close()
    conflicting = _record("same", mode=1, difficulty=0.2)
    with pytest.raises(CorridorLeaderboardError, match="conflicting duplicate"):
        build_corridor_candidate_leaderboards_duckdb(
            [record, conflicting], _policy(), scratch_dir=tmp_path / "conflict"
        )


def test_empty_and_ineligible_inputs_are_bounded(tmp_path: Path) -> None:
    empty = build_corridor_candidate_leaderboards_duckdb(
        [], _policy(), scratch_dir=tmp_path / "empty"
    )
    assert empty.modes == ()
    empty.backend.close()
    rejected = _record("rejected", mode=1, membership=0.0)
    artifact = build_corridor_candidate_leaderboards_duckdb(
        [rejected], _policy(), scratch_dir=tmp_path / "reject"
    )
    assert artifact.summary["retained_candidate_count"] == 0
    artifact.backend.close()


def test_thread_and_batch_configuration_is_semantic_invariant(tmp_path: Path) -> None:
    records = [
        _record(f"candidate-{i}", mode=i % 4, difficulty=(i * 7) % 11)
        for i in range(64)
    ]
    results = []
    for threads, batch in ((1, 3), (2, 7), (4, 16)):
        artifact = build_corridor_candidate_leaderboards_duckdb(
            records,
            _policy(),
            scratch_dir=tmp_path / f"db-{threads}-{batch}",
            threads=threads,
            ingestion_batch_size=batch,
            fetch_batch_size=4,
        )
        results.append((_coords(artifact), artifact.summary))
        artifact.backend.close()
    assert results[0] == results[1] == results[2]


def test_no_silent_in_memory_fallback(monkeypatch, tmp_path: Path) -> None:
    import radjax_tome.fingerprint.corridor_duckdb as module

    monkeypatch.setattr(module, "duckdb", None)
    with pytest.raises(CorridorLeaderboardError, match="DuckDB 1.4.5"):
        DuckDBCandidateStore(scratch_dir=tmp_path / "missing", policy=_policy())


def test_existing_cache_is_inspected_and_rebuilt(tmp_path: Path) -> None:
    scratch = tmp_path / "cache"
    first = build_corridor_candidate_leaderboards_duckdb(
        [_record("first", mode=1)],
        _policy(),
        scratch_dir=scratch,
        input_authority="sha256:first",
        implementation_authority="test-impl",
    )
    first.backend.close(remove_scratch=False)
    second = build_corridor_candidate_leaderboards_duckdb(
        [_record("second", mode=1)],
        _policy(),
        scratch_dir=scratch,
        input_authority="sha256:second",
        implementation_authority="test-impl",
    )
    assert _coords(second) == [[("second", 0)]]
    assert second.backend.cache_rebuilt is True
    assert second.backend.cache_rebuild_reason == "input_authority_mismatch"
    second.backend.close()


def test_streamed_backend_artifact_has_valid_readback_for_small_fixture(
    tmp_path: Path,
) -> None:
    artifact = build_corridor_candidate_leaderboards_duckdb(
        [_record(f"candidate-{i}", mode=i % 3) for i in range(12)],
        _policy(),
        scratch_dir=tmp_path / "db",
    )
    output = write_corridor_candidate_leaderboards(
        artifact, tmp_path / "leaderboards", overwrite=True
    )
    assert validate_corridor_candidate_leaderboards(output).ok
    artifact.backend.close()


def test_coordinate_overlap_is_counted_without_reserve_materialization(
    tmp_path: Path,
) -> None:
    artifact = build_corridor_candidate_leaderboards_duckdb(
        [_record("shared", mode=1), _record("only-corridor", mode=1)],
        _policy(),
        scratch_dir=tmp_path / "db",
    )
    assert (
        artifact.backend.count_coordinate_overlap([("shared", 0), ("external", 0)]) == 1
    )
    artifact.backend.close()


def test_ranked_reserve_has_no_offset_pagination() -> None:
    import radjax_tome.fingerprint.corridor_duckdb as module

    assert "OFFSET" not in inspect.getsource(module.DuckDBRankedReserve)
