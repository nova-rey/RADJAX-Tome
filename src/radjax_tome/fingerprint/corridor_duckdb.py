from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

try:
    import duckdb
except ImportError as exc:  # pragma: no cover
    duckdb = None
    _DUCKDB_IMPORT_ERROR = exc
else:
    _DUCKDB_IMPORT_ERROR = None

from radjax_tome.fingerprint.corridor_archetypes import (
    CorridorArchetypeScore,
    score_corridor_archetype_candidate,
)
from radjax_tome.fingerprint.corridor_leaderboards import (
    CorridorCandidateRecord,
    CorridorFeatureProvenance,
    CorridorLeaderboardArtifact,
    CorridorLeaderboardError,
    CorridorLeaderboardPolicy,
    CorridorModeLeaderboard,
)


class DuckDBRankedReserve(Sequence[CorridorArchetypeScore]):
    """Lazy, deterministic ordered view of one DuckDB corridor reserve."""

    def __init__(self, store: DuckDBCandidateStore, mode_id: int, count: int):
        self._store = store
        self._mode_id = mode_id
        self._count = count

    def __len__(self) -> int:
        return self._count

    def __iter__(self) -> Iterator[CorridorArchetypeScore]:
        cursor = self._store.connection.execute(
            self._store._ordered_query(self._mode_id)
        )
        while True:
            rows = cursor.fetchmany(self._store.fetch_batch_size)
            if not rows:
                return
            for row in rows:
                yield self._store._score_from_row(row)

    def __getitem__(self, index: int) -> CorridorArchetypeScore:
        if not isinstance(index, int):
            raise TypeError("ranked reserve indices must be integers")
        if index < 0:
            index += self._count
        if index < 0 or index >= self._count:
            raise IndexError(index)
        row = self._store.connection.execute(
            self._store._ordered_query(self._mode_id) + " LIMIT 1 OFFSET ?",
            [index],
        ).fetchone()
        if row is None:
            raise IndexError(index)
        return self._store._score_from_row(row)


class DuckDBCandidateStore:
    """Rebuildable, run-owned DuckDB C2 candidate/reserve store."""

    SCHEMA_VERSION = "radjax.c2_duckdb_store.v1"

    def __init__(
        self,
        *,
        scratch_dir: str | Path,
        policy: CorridorLeaderboardPolicy,
        memory_limit: str = "512MiB",
        threads: int = 4,
        ingestion_batch_size: int = 4096,
        fetch_batch_size: int = 1024,
        input_authority: str | None = None,
        implementation_authority: str = "unknown",
    ) -> None:
        if duckdb is None:
            raise CorridorLeaderboardError(
                "DuckDB 1.4.5 is required for canonical C2; install duckdb==1.4.5"
            ) from _DUCKDB_IMPORT_ERROR
        self.scratch_dir = Path(scratch_dir)
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.scratch_dir / "c2-candidates.duckdb"
        self.tmp_dir = self.scratch_dir / "duckdb-tmp"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.memory_limit = memory_limit
        self.threads = max(1, int(threads))
        self.ingestion_batch_size = max(1, int(ingestion_batch_size))
        self.fetch_batch_size = max(1, int(fetch_batch_size))
        self.policy = policy
        self.input_authority = input_authority
        self.implementation_authority = implementation_authority
        # This is a rebuildable execution cache. Never append to an existing
        # file: stale or partial reserves must not contaminate a new run.
        self.cache_rebuilt = False
        self.cache_rebuild_reason: str | None = None
        self._discard_existing_cache()
        self.connection = duckdb.connect(str(self.db_path))
        self.connection.execute(f"PRAGMA memory_limit='{memory_limit}'")
        self.connection.execute(f"PRAGMA threads={self.threads}")
        escaped_tmp = str(self.tmp_dir).replace("'", "''")
        self.connection.execute(f"PRAGMA temp_directory='{escaped_tmp}'")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS execution_metadata (
                schema_version VARCHAR,
                input_authority VARCHAR,
                policy_json VARCHAR,
                implementation_authority VARCHAR,
                complete BOOLEAN,
                duckdb_version VARCHAR,
                memory_limit VARCHAR,
                threads BIGINT,
                temp_directory VARCHAR,
                ingestion_batch_size BIGINT,
                fetch_batch_size BIGINT
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS candidates (
                candidate_id VARCHAR NOT NULL,
                position BIGINT NOT NULL,
                mode_id BIGINT NOT NULL,
                mode_support BIGINT NOT NULL,
                corridor_fingerprint_id VARCHAR,
                membership_score DOUBLE NOT NULL,
                centrality_score DOUBLE NOT NULL,
                useful_difficulty_score DOUBLE NOT NULL,
                quality_score DOUBLE NOT NULL,
                corridor_training_utility DOUBLE,
                policy_id VARCHAR NOT NULL,
                eligible BOOLEAN NOT NULL,
                eligibility_reasons VARCHAR NOT NULL,
                full_width BOOLEAN NOT NULL,
                arrival_ordinal BIGINT NOT NULL,
                record_digest VARCHAR NOT NULL,
                PRIMARY KEY (candidate_id, position)
            )
            """
        )
        self.connection.execute(
            "CREATE TEMP TABLE IF NOT EXISTS incoming_candidates AS "
            "SELECT * FROM candidates LIMIT 0"
        )
        self._seen = 0
        self._eligible = 0
        self._duplicates = 0
        self._duplicate_modes: Counter[int] = Counter()
        self._rejections: Counter[str] = Counter()
        self._states: dict[int, dict[str, Any]] = {}
        self._provenance: CorridorFeatureProvenance | None = None
        self._arrival = 0
        self._batch: list[tuple[Any, ...]] = []
        self._pending_keys: dict[tuple[str, int], tuple[str, int]] = {}
        self._metadata_written = False

    def _discard_existing_cache(self) -> None:
        if not self.db_path.exists():
            return
        reason = "existing_cache_rebuilt"
        try:
            prior = duckdb.connect(str(self.db_path), read_only=True)
            columns = prior.execute(
                "PRAGMA table_info('execution_metadata')"
            ).fetchall()
            names = {str(row[1]) for row in columns}
            required = {
                "schema_version",
                "input_authority",
                "policy_json",
                "implementation_authority",
                "complete",
                "duckdb_version",
                "memory_limit",
                "threads",
                "temp_directory",
                "ingestion_batch_size",
                "fetch_batch_size",
            }
            if not required.issubset(names):
                reason = "incompatible_cache_metadata"
            else:
                row = prior.execute(
                    "SELECT schema_version, input_authority, policy_json, "
                    "implementation_authority, complete, duckdb_version, "
                    "memory_limit, threads, temp_directory, ingestion_batch_size, "
                    "fetch_batch_size FROM execution_metadata"
                ).fetchone()
                if row is None or not bool(row[4]):
                    reason = "incomplete_cache"
                elif row[0] != self.SCHEMA_VERSION:
                    reason = "schema_mismatch"
                elif row[1] != self.input_authority:
                    reason = "input_authority_mismatch"
                elif row[2] != json.dumps(self.policy.to_dict(), sort_keys=True):
                    reason = "policy_mismatch"
                elif row[3] != self.implementation_authority:
                    reason = "implementation_authority_mismatch"
                elif row[5] != str(duckdb.__version__):
                    reason = "duckdb_version_mismatch"
                elif row[6] != self.memory_limit or int(row[7]) != self.threads:
                    reason = "execution_limits_mismatch"
                elif row[8] != str(self.tmp_dir.resolve()):
                    reason = "temporary_directory_mismatch"
                elif (
                    int(row[9]) != self.ingestion_batch_size
                    or int(row[10]) != self.fetch_batch_size
                ):
                    reason = "batch_configuration_mismatch"
                else:
                    reason = "compatible_cache_rebuilt_for_clean_run"
            prior.close()
        except Exception:
            reason = "unreadable_cache_metadata"
        self.cache_rebuilt = True
        self.cache_rebuild_reason = reason
        self.db_path.unlink(missing_ok=True)
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def _write_metadata(self) -> None:
        if self._metadata_written:
            return
        self.connection.execute("DELETE FROM execution_metadata")
        self.connection.execute(
            "INSERT INTO execution_metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                self.SCHEMA_VERSION,
                self.input_authority,
                json.dumps(self.policy.to_dict(), sort_keys=True),
                self.implementation_authority,
                False,
                str(duckdb.__version__),
                self.memory_limit,
                self.threads,
                str(self.tmp_dir.resolve()),
                self.ingestion_batch_size,
                self.fetch_batch_size,
            ],
        )
        self._metadata_written = True

    def _flush(self) -> None:
        if not self._batch:
            return
        self._write_metadata()
        self.connection.execute("DELETE FROM incoming_candidates")
        self.connection.executemany(
            "INSERT INTO incoming_candidates VALUES ("
            "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ? )",
            self._batch,
        )
        conflicts = self.connection.execute(
            """
            SELECT i.candidate_id, i.position, i.record_digest,
                   c.record_digest, c.mode_id
            FROM incoming_candidates i
            JOIN candidates c
              ON c.candidate_id = i.candidate_id AND c.position = i.position
            """
        ).fetchall()
        for (
            candidate_id,
            position,
            incoming_digest,
            existing_digest,
            mode_id,
        ) in conflicts:
            if str(incoming_digest) != str(existing_digest):
                raise CorridorLeaderboardError(
                    "conflicting duplicate candidate coordinate: "
                    f"{candidate_id}:{position}"
                )
            self._duplicates += 1
            self._duplicate_modes[int(mode_id)] += 1
        self.connection.execute(
            """
            INSERT INTO candidates
            SELECT i.*
            FROM incoming_candidates i
            LEFT JOIN candidates c
              ON c.candidate_id = i.candidate_id AND c.position = i.position
            WHERE c.candidate_id IS NULL
            """
        )
        self._batch.clear()
        self._pending_keys.clear()

    def ingest(self, candidates: Iterable[CorridorCandidateRecord]) -> None:
        for record in candidates:
            if not isinstance(record, CorridorCandidateRecord):
                raise TypeError(
                    "candidates must contain CorridorCandidateRecord values"
                )
            self._seen += 1
            provenance = record.feature_provenance
            if (
                provenance.fidelity == "compatibility_proxy"
                and not self.policy.allow_compatibility_proxies
            ):
                raise CorridorLeaderboardError(
                    "real corridor features are required; compatibility_proxy "
                    "is disabled"
                )
            if self._provenance is None:
                self._provenance = provenance
            elif self._provenance != provenance:
                raise CorridorLeaderboardError(
                    "all candidates must share one feature provenance manifest"
                )
            digest = hashlib.sha256(
                json.dumps(
                    record.to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            mode_id = record.features.corridor_mode_id
            key = (record.coordinate[0], int(record.coordinate[1]))
            pending = self._pending_keys.get(key)
            if pending is not None:
                if pending[0] != digest:
                    raise CorridorLeaderboardError(
                        f"conflicting duplicate candidate coordinate: {key[0]}:{key[1]}"
                    )
                self._duplicates += 1
                self._duplicate_modes[pending[1]] += 1
                self._arrival += 1
                continue
            score = score_corridor_archetype_candidate(
                record.features, self.policy.archetype_policy
            )
            if mode_id is not None and mode_id >= 0:
                state = self._states.setdefault(
                    mode_id,
                    {
                        "mode_support": record.features.mode_support,
                        "seen": 0,
                        "eligible": 0,
                        "rejected": 0,
                        "reasons": Counter(),
                    },
                )
                if state["mode_support"] != record.features.mode_support:
                    raise CorridorLeaderboardError(
                        f"conflicting mode support for corridor mode {mode_id}"
                    )
                state["seen"] += 1
            if not score.eligible:
                if mode_id is not None and mode_id >= 0:
                    state["rejected"] += 1
                    state["reasons"].update(score.eligibility_reasons)
                self._rejections.update(score.eligibility_reasons)
                self._arrival += 1
                continue
            self._eligible += 1
            if mode_id is None or mode_id < 0:
                self._rejections["unassigned_corridor"] += 1
                self._arrival += 1
                continue
            state["eligible"] += 1
            self._batch.append(
                (
                    score.candidate_id,
                    score.position,
                    mode_id,
                    state["mode_support"],
                    score.corridor_fingerprint_id,
                    score.membership_score,
                    score.centrality_score,
                    score.useful_difficulty_score,
                    score.quality_score,
                    score.corridor_training_utility,
                    score.policy_id,
                    score.eligible,
                    json.dumps(list(score.eligibility_reasons), separators=(",", ":")),
                    score.full_width,
                    self._arrival,
                    digest,
                )
            )
            self._pending_keys[key] = (digest, int(mode_id))
            self._arrival += 1
            if len(self._batch) >= self.ingestion_batch_size:
                self._flush()
        self._flush()
        self.connection.commit()
        self.connection.execute("DROP TABLE IF EXISTS ranked_candidates")
        self.connection.execute(
            """
            CREATE TABLE ranked_candidates AS
            SELECT *,
              row_number() OVER (
                PARTITION BY mode_id
                ORDER BY corridor_training_utility DESC NULLS LAST,
                         membership_score DESC,
                         centrality_score DESC,
                         useful_difficulty_score DESC,
                         candidate_id ASC,
                         position ASC
              ) - 1 AS rank_ordinal
            FROM candidates
            WHERE eligible = TRUE
            """
        )
        self.connection.execute("UPDATE execution_metadata SET complete = TRUE")
        self.connection.commit()

    def _ordered_query(self, mode_id: int) -> str:
        return (
            "SELECT candidate_id, position, mode_id, corridor_fingerprint_id, "
            "eligible, eligibility_reasons, membership_score, centrality_score, "
            "useful_difficulty_score, corridor_training_utility, quality_score, "
            "policy_id, full_width FROM ranked_candidates "
            f"WHERE mode_id = {int(mode_id)} "
            "ORDER BY rank_ordinal"
        )

    @staticmethod
    def _score_from_row(row: tuple[Any, ...]) -> CorridorArchetypeScore:
        (
            candidate_id,
            position,
            mode_id,
            fingerprint_id,
            eligible,
            reasons,
            membership,
            centrality,
            difficulty,
            utility,
            quality,
            policy_id,
            full_width,
        ) = row
        return CorridorArchetypeScore(
            candidate_id=str(candidate_id),
            position=int(position),
            corridor_mode_id=int(mode_id),
            corridor_fingerprint_id=fingerprint_id,
            eligible=bool(eligible),
            eligibility_reasons=tuple(json.loads(reasons)),
            membership_score=float(membership),
            centrality_score=float(centrality),
            useful_difficulty_score=float(difficulty),
            quality_score=float(quality),
            corridor_training_utility=(None if utility is None else float(utility)),
            policy_id=str(policy_id),
            full_width=bool(full_width),
        )

    def validate_disjoint(self) -> None:
        row = self.connection.execute(
            """
            SELECT candidate_id, position
            FROM candidates
            WHERE eligible = TRUE
            GROUP BY candidate_id, position
            HAVING COUNT(DISTINCT mode_id) > 1
            LIMIT 1
            """
        ).fetchone()
        if row is not None:
            raise CorridorLeaderboardError(
                f"coordinate appears in multiple corridor pools: {row[0]}:{row[1]}"
            )

    def _mode_summary(
        self, mode_id: int, state: dict[str, Any]
    ) -> CorridorModeLeaderboard:
        count = int(
            self.connection.execute(
                "SELECT COUNT(*) FROM candidates WHERE mode_id = ? AND eligible = TRUE",
                [mode_id],
            ).fetchone()[0]
        )
        return CorridorModeLeaderboard(
            corridor_mode_id=mode_id,
            mode_support=int(state["mode_support"]),
            candidates=DuckDBRankedReserve(self, mode_id, count),
            candidates_seen=int(state["seen"]),
            candidates_eligible=int(state["eligible"]),
            candidates_rejected=int(state["rejected"]),
            rejection_counts_by_reason=dict(state["reasons"]),
            duplicate_count=self._duplicate_modes[mode_id],
        )

    def artifact(self) -> CorridorLeaderboardArtifact:
        modes = tuple(
            self._mode_summary(mode_id, state)
            for mode_id, state in sorted(self._states.items())
            if state["eligible"] > 0
        )
        summary = {
            "candidates_seen": self._seen,
            "candidates_eligible": self._eligible,
            "candidates_rejected": self._seen - self._eligible - self._duplicates,
            "duplicates_collapsed": self._duplicates,
            "retained_candidate_count": sum(len(mode.candidates) for mode in modes),
            "candidate_pool_cap": self.policy.candidate_pool_cap,
            "rejection_counts_by_reason": dict(sorted(self._rejections.items())),
            "modes_observed": len(modes),
            "modes_with_eligible_candidates": len(modes),
            "modes_with_empty_pools": len(self._states) - len(modes),
            "production_grade": self._provenance is None
            or self._provenance.fidelity != "compatibility_proxy",
            "compatibility_proxy_used": self._provenance is not None
            and self._provenance.fidelity == "compatibility_proxy",
        }
        artifact = CorridorLeaderboardArtifact(
            policy=self.policy,
            feature_provenance=self._provenance,
            modes=modes,
            summary=summary,
            warnings=(
                ("compatibility_proxy_used: non-production developer override",)
                if summary["compatibility_proxy_used"]
                else ()
            ),
        )
        object.__setattr__(artifact, "backend", self)
        return artifact

    def summary_for_path(self, path: Path) -> dict[str, Any]:
        from radjax_tome.fingerprint.corridor_leaderboards import _sha256

        manifest = path / "manifest.json"
        mode = path / "mode_leaderboards.jsonl"
        summary = dict(self.artifact().summary)
        summary.update(
            {
                "status": "pass",
                "leaderboard_artifact_id": str(path.resolve()),
                "leaderboard_manifest_sha256": _sha256(manifest),
                "mode_leaderboards_sha256": _sha256(mode),
                "feature_fidelity": (
                    self._provenance.fidelity if self._provenance else None
                ),
                "compatibility_proxy_used": bool(summary["compatibility_proxy_used"]),
                "cache_rebuilt": self.cache_rebuilt,
                "cache_rebuild_reason": self.cache_rebuild_reason,
                "duckdb_version": str(duckdb.__version__),
                "memory_limit": self.memory_limit,
                "threads": self.threads,
                "temp_directory": str(self.tmp_dir.resolve()),
                "ingestion_batch_size": self.ingestion_batch_size,
                "fetch_batch_size": self.fetch_batch_size,
            }
        )
        return summary

    def close(self, *, remove_scratch: bool = True) -> None:
        try:
            self.connection.close()
        finally:
            if remove_scratch:
                shutil.rmtree(self.scratch_dir, ignore_errors=True)


def build_corridor_candidate_leaderboards_duckdb(
    candidates: Iterable[CorridorCandidateRecord],
    policy: CorridorLeaderboardPolicy,
    *,
    scratch_dir: str | Path,
    memory_limit: str = "512MiB",
    threads: int = 4,
    ingestion_batch_size: int = 4096,
    fetch_batch_size: int = 1024,
    input_authority: str | None = None,
    implementation_authority: str = "unknown",
) -> CorridorLeaderboardArtifact:
    store = DuckDBCandidateStore(
        scratch_dir=scratch_dir,
        policy=policy,
        memory_limit=memory_limit,
        threads=threads,
        ingestion_batch_size=ingestion_batch_size,
        fetch_batch_size=fetch_batch_size,
        input_authority=input_authority,
        implementation_authority=implementation_authority,
    )
    try:
        store.ingest(candidates)
        return store.artifact()
    except Exception:
        store.close()
        raise
