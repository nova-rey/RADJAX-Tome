"""C2 offline fingerprint-corridor candidate micro-leaderboards."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sqlite3
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from radjax_tome.fingerprint.corridor_archetypes import (
    CorridorArchetypePolicy,
    CorridorArchetypeScore,
    CorridorCandidateFeatures,
    score_corridor_archetype_candidate,
)
from radjax_tome.io.json import read_json_object, write_json

CORRIDOR_LEADERBOARD_SCHEMA = "radjax.c2_corridor_candidate_leaderboards.v1"
CORRIDOR_MODE_LEADERBOARD_SCHEMA = "radjax.c2_corridor_mode_leaderboard.v1"
CORRIDOR_FEATURE_PROVENANCE_SCHEMA = "radjax.c2_corridor_feature_provenance.v1"
CORRIDOR_VALIDATION_SCHEMA = "radjax.c2_corridor_leaderboard_validation.v1"
CORRIDOR_LEADERBOARD_MANIFEST = "manifest.json"
CORRIDOR_MODE_LEADERBOARDS = "mode_leaderboards.jsonl"
CORRIDOR_VALIDATION_REPORT = "validation_report.json"
FEATURE_FIDELITIES = frozenset({"explicit", "derived", "compatibility_proxy"})


class CorridorLeaderboardError(ValueError):
    """Actionable input or artifact error for the offline C2 seam."""


@dataclass(frozen=True)
class CorridorFeatureProvenance:
    """Manifest-level evidence describing how compact features were obtained."""

    feature_schema_version: str = CORRIDOR_FEATURE_PROVENANCE_SCHEMA
    source_artifact_schema: str = "synthetic"
    source_artifact_id: str = "synthetic"
    source_artifact_hash: str | None = None
    membership_derivation: str = "provided_normalized_membership"
    core_distance_derivation: str = "provided_normalized_core_distance"
    difficulty_derivation: str = "provided_normalized_difficulty"
    fidelity: Literal["explicit", "derived", "compatibility_proxy"] = "explicit"
    compatibility_proxy_used: bool = False
    normalization_parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.feature_schema_version != CORRIDOR_FEATURE_PROVENANCE_SCHEMA:
            raise ValueError("unsupported corridor feature provenance schema")
        if not self.source_artifact_schema or not self.source_artifact_id:
            raise ValueError("source artifact schema and identity are required")
        if self.fidelity not in FEATURE_FIDELITIES:
            raise ValueError(f"unsupported feature fidelity: {self.fidelity}")
        if self.fidelity == "compatibility_proxy" and not self.compatibility_proxy_used:
            raise ValueError(
                "compatibility_proxy fidelity must set compatibility_proxy_used"
            )
        if self.compatibility_proxy_used and self.fidelity != "compatibility_proxy":
            raise ValueError(
                "compatibility_proxy_used requires compatibility_proxy fidelity"
            )
        if not isinstance(self.compatibility_proxy_used, bool):
            raise TypeError("compatibility_proxy_used must be a boolean")
        if not isinstance(self.normalization_parameters, Mapping):
            raise TypeError("normalization_parameters must be an object")
        object.__setattr__(
            self,
            "normalization_parameters",
            dict(self.normalization_parameters),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.feature_schema_version,
            "source_artifact_schema": self.source_artifact_schema,
            "source_artifact_id": self.source_artifact_id,
            "source_artifact_hash": self.source_artifact_hash,
            "membership_derivation": self.membership_derivation,
            "core_distance_derivation": self.core_distance_derivation,
            "difficulty_derivation": self.difficulty_derivation,
            "fidelity": self.fidelity,
            "compatibility_proxy_used": self.compatibility_proxy_used,
            "normalization_parameters": dict(self.normalization_parameters),
        }


@dataclass(frozen=True)
class CorridorLeaderboardPolicy:
    """C2 pool policy; budget allocation begins in C3."""

    candidate_pool_cap: int = 4
    archetype_policy: CorridorArchetypePolicy = field(
        default_factory=CorridorArchetypePolicy
    )
    allow_compatibility_proxies: bool = False
    proxy_override_reason: str | None = None
    policy_id: str = "corridor_candidate_micro_leaderboard_v1"

    def __post_init__(self) -> None:
        if isinstance(self.candidate_pool_cap, bool) or not isinstance(
            self.candidate_pool_cap, int
        ):
            raise ValueError("candidate_pool_cap must be a positive integer")
        if self.candidate_pool_cap < 1:
            raise ValueError("candidate_pool_cap must be a positive integer")
        if not isinstance(self.archetype_policy, CorridorArchetypePolicy):
            raise TypeError("archetype_policy must be CorridorArchetypePolicy")
        if not isinstance(self.allow_compatibility_proxies, bool):
            raise TypeError("allow_compatibility_proxies must be a boolean")
        if self.allow_compatibility_proxies and not self.proxy_override_reason:
            raise ValueError(
                "proxy_override_reason is required when compatibility "
                "proxies are allowed"
            )
        if not self.policy_id:
            raise ValueError("policy_id must be nonempty")

    @property
    def compatibility_proxy_override_enabled(self) -> bool:
        return self.allow_compatibility_proxies

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CORRIDOR_LEADERBOARD_SCHEMA,
            "policy_id": self.policy_id,
            "candidate_pool_cap": self.candidate_pool_cap,
            "archetype_policy": self.archetype_policy.to_dict(),
            "allow_compatibility_proxies": self.allow_compatibility_proxies,
            "proxy_override_reason": self.proxy_override_reason,
            "compatibility_proxy_override_enabled": (
                self.compatibility_proxy_override_enabled
            ),
        }


@dataclass(frozen=True)
class CorridorCandidateRecord:
    features: CorridorCandidateFeatures
    feature_provenance: CorridorFeatureProvenance = field(
        default_factory=CorridorFeatureProvenance
    )

    @property
    def coordinate(self) -> tuple[str, int]:
        return (self.features.candidate_id, self.features.position)

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": {
                "candidate_id": self.features.candidate_id,
                "position": self.features.position,
                "corridor_mode_id": self.features.corridor_mode_id,
                "assignment_status": self.features.assignment_status,
                "membership_strength": self.features.membership_strength,
                "core_distance": self.features.core_distance,
                "mode_support": self.features.mode_support,
                "difficulty_score": self.features.difficulty_score,
                "quality_score": self.features.quality_score,
                "corridor_fingerprint_id": self.features.corridor_fingerprint_id,
                "position_valid": self.features.position_valid,
            },
            "feature_provenance": self.feature_provenance.to_dict(),
        }


@dataclass(frozen=True)
class CorridorModeLeaderboard:
    corridor_mode_id: int
    mode_support: int
    candidates: tuple[CorridorArchetypeScore, ...]
    candidates_seen: int
    candidates_eligible: int
    candidates_rejected: int
    rejection_counts_by_reason: Mapping[str, int] = field(default_factory=dict)
    duplicate_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CORRIDOR_MODE_LEADERBOARD_SCHEMA,
            "corridor_mode_id": self.corridor_mode_id,
            "mode_support": self.mode_support,
            "candidates_seen": self.candidates_seen,
            "candidates_eligible": self.candidates_eligible,
            "candidates_rejected": self.candidates_rejected,
            "duplicate_count": self.duplicate_count,
            "rejection_counts_by_reason": dict(
                sorted(self.rejection_counts_by_reason.items())
            ),
            "retained_count": len(self.candidates),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class CorridorLeaderboardArtifact:
    policy: CorridorLeaderboardPolicy
    feature_provenance: CorridorFeatureProvenance | None
    modes: tuple[CorridorModeLeaderboard, ...]
    summary: Mapping[str, Any]
    warnings: tuple[str, ...] = ()

    @property
    def production_grade(self) -> bool:
        """Whether observed feature provenance is production-grade."""

        return bool(self.summary.get("production_grade", False))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CORRIDOR_LEADERBOARD_SCHEMA,
            "policy": self.policy.to_dict(),
            "feature_provenance": (
                self.feature_provenance.to_dict()
                if self.feature_provenance is not None
                else None
            ),
            "summary": dict(self.summary),
            "warnings": list(self.warnings),
            "modes": [mode.to_dict() for mode in self.modes],
        }


@dataclass(frozen=True)
class CorridorLeaderboardValidationResult:
    status: Literal["pass", "fail", "warn"]
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    summary: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CORRIDOR_VALIDATION_SCHEMA,
            "status": self.status,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "summary": dict(self.summary),
        }


@dataclass
class _ModeState:
    mode_support: int
    candidates_seen: int = 0
    candidates_eligible: int = 0
    candidates_rejected: int = 0
    rejection_counts: Counter[str] = field(default_factory=Counter)
    pool: list[CorridorArchetypeScore] = field(default_factory=list)
    duplicate_count: int = 0


def build_corridor_candidate_leaderboards(
    candidates: Iterable[CorridorCandidateRecord],
    policy: CorridorLeaderboardPolicy | None = None,
) -> CorridorLeaderboardArtifact:
    """Build bounded deterministic per-mode pools from compact candidate records."""

    policy = policy or CorridorLeaderboardPolicy()
    states: dict[int, _ModeState] = {}
    provenance: CorridorFeatureProvenance | None = None
    candidates_seen = 0
    candidates_eligible = 0
    duplicates = 0
    duplicate_modes: Counter[int] = Counter()
    rejection_counts: Counter[str] = Counter()

    with _temporary_coordinate_index() as connection:
        for record in candidates:
            if not isinstance(record, CorridorCandidateRecord):
                raise TypeError(
                    "candidates must contain CorridorCandidateRecord values"
                )
            candidates_seen += 1
            record_provenance = record.feature_provenance
            if record_provenance.fidelity == "compatibility_proxy" and not (
                policy.allow_compatibility_proxies
            ):
                raise CorridorLeaderboardError(
                    "real corridor features are required; compatibility_proxy "
                    "is disabled by default; provide explicit developer override"
                )
            if provenance is None:
                provenance = record_provenance
            elif provenance != record_provenance:
                raise CorridorLeaderboardError(
                    "all candidates must share one feature provenance manifest"
                )
            identity = record.coordinate
            serialized_record = json.dumps(
                record.to_dict(), sort_keys=True, separators=(",", ":")
            )
            previous = connection.execute(
                "SELECT record_json, mode_id FROM coordinates "
                "WHERE candidate_id = ? AND position = ?",
                identity,
            ).fetchone()
            if previous is not None:
                if previous[0] != serialized_record:
                    raise CorridorLeaderboardError(
                        "conflicting duplicate candidate coordinate: "
                        f"{identity[0]}:{identity[1]}"
                    )
                duplicates += 1
                if previous[1] is not None:
                    duplicate_modes[int(previous[1])] += 1
                continue
            connection.execute(
                "INSERT INTO coordinates "
                "(candidate_id, position, record_json, mode_id) VALUES (?, ?, ?, ?)",
                (
                    identity[0],
                    identity[1],
                    serialized_record,
                    record.features.corridor_mode_id,
                ),
            )

            mode_id = record.features.corridor_mode_id
            state = None
            if mode_id is not None and mode_id >= 0:
                state = states.setdefault(
                    mode_id,
                    _ModeState(mode_support=record.features.mode_support),
                )
                if state.mode_support != record.features.mode_support:
                    raise CorridorLeaderboardError(
                        f"conflicting mode support for corridor mode {mode_id}"
                    )
                state.candidates_seen += 1

            score = score_corridor_archetype_candidate(
                record.features,
                policy.archetype_policy,
            )
            if not score.eligible:
                if state is not None:
                    state.candidates_rejected += 1
                rejection_counts.update(score.eligibility_reasons)
                if state is not None:
                    state.rejection_counts.update(score.eligibility_reasons)
                continue
            candidates_eligible += 1
            if state is None:
                rejection_counts["unassigned_corridor"] += 1
                continue
            state.candidates_eligible += 1
            state.pool.append(score)
            state.pool.sort(key=_score_sort_key)
            del state.pool[policy.candidate_pool_cap :]
        connection.commit()

    for mode_id, state in states.items():
        state.duplicate_count = duplicate_modes[mode_id]
    modes = tuple(
        CorridorModeLeaderboard(
            corridor_mode_id=mode_id,
            mode_support=state.mode_support,
            candidates=tuple(state.pool),
            candidates_seen=state.candidates_seen,
            candidates_eligible=state.candidates_eligible,
            candidates_rejected=state.candidates_rejected,
            rejection_counts_by_reason=dict(state.rejection_counts),
            duplicate_count=state.duplicate_count,
        )
        for mode_id, state in sorted(states.items())
    )
    modes_with_eligible = sum(bool(mode.candidates_eligible) for mode in modes)
    summary = {
        "candidates_seen": candidates_seen,
        "candidates_eligible": candidates_eligible,
        "candidates_rejected": candidates_seen - candidates_eligible - duplicates,
        "duplicates_collapsed": duplicates,
        "retained_candidate_count": sum(len(mode.candidates) for mode in modes),
        "candidate_pool_cap": policy.candidate_pool_cap,
        "rejection_counts_by_reason": dict(sorted(rejection_counts.items())),
        "modes_observed": len(modes),
        "modes_with_eligible_candidates": modes_with_eligible,
        "modes_with_empty_pools": len(modes) - modes_with_eligible,
        "production_grade": provenance is None
        or provenance.fidelity != "compatibility_proxy",
        "compatibility_proxy_used": provenance is not None
        and provenance.fidelity == "compatibility_proxy",
    }
    warnings = (
        ("compatibility_proxy_used: non-production developer override",)
        if summary["compatibility_proxy_used"]
        else ()
    )
    return CorridorLeaderboardArtifact(
        policy=policy,
        feature_provenance=provenance,
        modes=modes,
        summary=summary,
        warnings=warnings,
    )


def write_corridor_candidate_leaderboards(
    artifact: CorridorLeaderboardArtifact,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write the compact C2 leaderboard directory."""

    output = Path(output_dir)
    if output.exists() and not overwrite:
        raise ValueError(f"leaderboard output exists: {output}")
    validation = _validate_artifact(artifact, production_grade=True)
    if not validation.ok and artifact.production_grade:
        raise CorridorLeaderboardError(
            "cannot write invalid leaderboard artifact: "
            + "; ".join(validation.blockers)
        )
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=parent))
    try:
        mode_path = temp / CORRIDOR_MODE_LEADERBOARDS
        mode_path.write_text(
            "".join(
                json.dumps(mode.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
                for mode in artifact.modes
            ),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": CORRIDOR_LEADERBOARD_SCHEMA,
            "policy": artifact.policy.to_dict(),
            "feature_provenance": (
                artifact.feature_provenance.to_dict()
                if artifact.feature_provenance is not None
                else None
            ),
            "summary": dict(artifact.summary),
            "warnings": list(artifact.warnings),
            "files": {
                CORRIDOR_MODE_LEADERBOARDS: {
                    "sha256": _sha256(mode_path),
                    "size_bytes": mode_path.stat().st_size,
                }
            },
        }
        write_json(temp / CORRIDOR_LEADERBOARD_MANIFEST, manifest)
        report = _validate_directory(temp, production_grade=False)
        write_json(temp / CORRIDOR_VALIDATION_REPORT, report.to_dict())
        if output.exists():
            shutil.rmtree(output)
        os.replace(temp, output)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return output


def validate_corridor_candidate_leaderboards(
    path: str | Path,
    *,
    production_grade: bool = True,
) -> CorridorLeaderboardValidationResult:
    return _validate_directory(Path(path), production_grade=production_grade)


def validate_corridor_candidate_leaderboard_artifact(
    artifact: CorridorLeaderboardArtifact,
    *,
    production_grade: bool = True,
) -> CorridorLeaderboardValidationResult:
    """Validate an in-memory C2 artifact using the canonical C2 rules."""

    if not isinstance(artifact, CorridorLeaderboardArtifact):
        raise TypeError("artifact must be a CorridorLeaderboardArtifact")
    return _validate_artifact(artifact, production_grade=production_grade)


def load_corridor_candidate_leaderboards(
    path: str | Path,
    *,
    production_grade: bool = True,
) -> CorridorLeaderboardArtifact:
    """Load a validated C2 artifact without exposing its storage format."""

    root = Path(path)
    validation = validate_corridor_candidate_leaderboards(
        root, production_grade=production_grade
    )
    if validation.status == "fail":
        raise CorridorLeaderboardError(
            "cannot load invalid corridor leaderboard artifact: "
            + "; ".join(validation.blockers)
        )
    manifest = read_json_object(root / CORRIDOR_LEADERBOARD_MANIFEST)
    policy_payload = manifest.get("policy") or {}
    archetype_payload = policy_payload.get("archetype_policy") or {}
    try:
        archetype_policy = CorridorArchetypePolicy(
            policy_id=str(archetype_payload["policy_id"]),
            minimum_membership_strength=float(
                archetype_payload["minimum_membership_strength"]
            ),
            maximum_core_distance=float(archetype_payload["maximum_core_distance"]),
            minimum_mode_support=int(archetype_payload["minimum_mode_support"]),
            membership_weight=float(archetype_payload["membership_weight"]),
            centrality_weight=float(archetype_payload["centrality_weight"]),
            difficulty_weight=float(archetype_payload["difficulty_weight"]),
            quality_weight=float(archetype_payload["quality_weight"]),
        )
        policy = CorridorLeaderboardPolicy(
            candidate_pool_cap=int(policy_payload["candidate_pool_cap"]),
            archetype_policy=archetype_policy,
            allow_compatibility_proxies=_strict_bool(
                policy_payload.get("allow_compatibility_proxies", False),
                "allow_compatibility_proxies",
            ),
            proxy_override_reason=policy_payload.get("proxy_override_reason"),
            policy_id=str(policy_payload["policy_id"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CorridorLeaderboardError(
            f"invalid C2 policy in leaderboard manifest: {exc}"
        ) from exc

    provenance_payload = manifest.get("feature_provenance")
    provenance = (
        None
        if provenance_payload is None
        else _provenance_from_dict(provenance_payload)
    )
    modes: list[CorridorModeLeaderboard] = []
    for mode_payload in _read_modes(root / CORRIDOR_MODE_LEADERBOARDS):
        try:
            candidates = tuple(
                _score_from_dict(candidate_payload)
                for candidate_payload in mode_payload["candidates"]
            )
            modes.append(
                CorridorModeLeaderboard(
                    corridor_mode_id=int(mode_payload["corridor_mode_id"]),
                    mode_support=int(mode_payload["mode_support"]),
                    candidates=candidates,
                    candidates_seen=int(mode_payload["candidates_seen"]),
                    candidates_eligible=int(mode_payload["candidates_eligible"]),
                    candidates_rejected=int(mode_payload["candidates_rejected"]),
                    rejection_counts_by_reason=dict(
                        mode_payload.get("rejection_counts_by_reason", {})
                    ),
                    duplicate_count=int(mode_payload.get("duplicate_count", 0)),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CorridorLeaderboardError(
                f"invalid C2 mode leaderboard payload: {exc}"
            ) from exc
    return CorridorLeaderboardArtifact(
        policy=policy,
        feature_provenance=provenance,
        modes=tuple(modes),
        summary=dict(manifest.get("summary") or {}),
        warnings=tuple(manifest.get("warnings") or ()),
    )


def inspect_corridor_candidate_leaderboards(path: str | Path) -> dict[str, Any]:
    result = validate_corridor_candidate_leaderboards(path, production_grade=False)
    manifest = read_json_object(Path(path) / CORRIDOR_LEADERBOARD_MANIFEST)
    summary = dict(manifest.get("summary", {}))
    modes = _read_modes(Path(path) / CORRIDOR_MODE_LEADERBOARDS)
    occupancy = [len(mode.get("candidates", ())) for mode in modes]
    return {
        "status": result.status,
        "blockers": list(result.blockers),
        "warnings": list(result.warnings),
        "modes_observed": summary.get("modes_observed", len(modes)),
        "coverage_holes": summary.get("modes_with_empty_pools", 0),
        "candidate_pool_cap": summary.get("candidate_pool_cap"),
        "retained_candidate_count": summary.get("retained_candidate_count", 0),
        "pool_occupancy": {
            "min": min(occupancy, default=0),
            "median": _median(occupancy),
            "max": max(occupancy, default=0),
        },
        "feature_fidelity": (manifest.get("feature_provenance", {}) or {}).get(
            "fidelity"
        ),
        "compatibility_proxy_used": summary.get("compatibility_proxy_used", False),
        "production_grade": summary.get("production_grade"),
        "c2_policy_id": (manifest.get("policy", {}) or {}).get("policy_id"),
        "leaderboard_artifact_id": str(Path(path).resolve()),
        "leaderboard_manifest_sha256": _sha256(
            Path(path) / CORRIDOR_LEADERBOARD_MANIFEST
        ),
        "mode_leaderboards_sha256": (
            manifest.get("files", {}).get(CORRIDOR_MODE_LEADERBOARDS, {}).get("sha256")
        ),
    }


def load_candidate_records_jsonl(
    path: str | Path,
    *,
    source_artifact_id: str | None = None,
) -> Iterator[CorridorCandidateRecord]:
    """Load explicit feature records for offline/developer workflows."""

    source = Path(path)
    if not source.is_file():
        raise CorridorLeaderboardError(f"candidate feature file missing: {source}")
    artifact_id = source_artifact_id or source.name
    artifact_hash = _sha256(source)

    def _records() -> Iterator[CorridorCandidateRecord]:
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        raise TypeError("record must be an object")
                    feature_payload = payload["features"]
                    if not isinstance(feature_payload, Mapping):
                        raise TypeError("features must be an object")
                    fidelity = _require_fidelity(payload)
                    _validate_loader_feature_contract(
                        feature_payload,
                        payload,
                        fidelity,
                    )
                    features = CorridorCandidateFeatures.from_mapping(feature_payload)
                    provenance = CorridorFeatureProvenance(
                        source_artifact_schema=str(
                            payload.get(
                                "source_artifact_schema", "candidate_features_jsonl_v1"
                            )
                        ),
                        source_artifact_id=artifact_id,
                        source_artifact_hash=artifact_hash,
                        membership_derivation=str(
                            payload.get(
                                "membership_derivation",
                                "provided_normalized_membership",
                            )
                        ),
                        core_distance_derivation=str(
                            payload.get(
                                "core_distance_derivation",
                                "provided_normalized_core_distance",
                            )
                        ),
                        difficulty_derivation=str(
                            payload.get(
                                "difficulty_derivation",
                                "provided_normalized_difficulty",
                            )
                        ),
                        fidelity=fidelity,
                        compatibility_proxy_used=(fidelity == "compatibility_proxy"),
                        normalization_parameters=payload.get(
                            "normalization_parameters", {}
                        ),
                    )
                    yield CorridorCandidateRecord(
                        features=features,
                        feature_provenance=provenance,
                    )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise CorridorLeaderboardError(
                        f"invalid candidate feature record at line {line_number}: {exc}"
                    ) from exc

    return _records()


def _validate_directory(
    root: Path,
    *,
    production_grade: bool,
) -> CorridorLeaderboardValidationResult:
    blockers: list[str] = []
    warnings: list[str] = []
    try:
        manifest = read_json_object(root / CORRIDOR_LEADERBOARD_MANIFEST)
        if manifest.get("schema_version") != CORRIDOR_LEADERBOARD_SCHEMA:
            blockers.append("unsupported leaderboard schema")
        files = manifest.get("files")
        if not isinstance(files, dict):
            blockers.append("leaderboard manifest files must be an object")
            files = {}
        mode_path = root / CORRIDOR_MODE_LEADERBOARDS
        if not mode_path.is_file():
            blockers.append("mode_leaderboards.jsonl is missing")
        elif CORRIDOR_MODE_LEADERBOARDS not in files:
            blockers.append(
                "leaderboard manifest does not hash mode_leaderboards.jsonl"
            )
        else:
            expected = files[CORRIDOR_MODE_LEADERBOARDS]
            if expected.get("sha256") != _sha256(mode_path):
                blockers.append("mode_leaderboards.jsonl hash mismatch")
            if int(expected.get("size_bytes", -1)) != mode_path.stat().st_size:
                blockers.append("mode_leaderboards.jsonl size mismatch")
        modes = _read_modes(mode_path) if mode_path.is_file() else []
        policy = manifest.get("policy") or {}
        cap = int(policy.get("candidate_pool_cap", -1))
        if cap < 1:
            blockers.append("leaderboard policy candidate_pool_cap is invalid")
        previous_mode = None
        all_coordinates: set[tuple[str, int]] = set()
        eligible_total = 0
        rejected_total = 0
        duplicate_total = 0
        for mode in modes:
            mode_id = mode.get("corridor_mode_id")
            if not isinstance(mode_id, int) or isinstance(mode_id, bool):
                blockers.append("mode id is not an integer")
                continue
            if previous_mode is not None and mode_id <= previous_mode:
                blockers.append("mode leaderboards are not strictly ordered")
            previous_mode = mode_id
            candidates = mode.get("candidates")
            if not isinstance(candidates, list):
                blockers.append(f"mode {mode_id} candidates must be a list")
                continue
            if len(candidates) > cap:
                blockers.append(f"mode {mode_id} exceeds candidate pool cap")
            seen_coordinates: set[tuple[str, int]] = set()
            previous_key: tuple[Any, ...] | None = None
            mode_seen = int(mode.get("candidates_seen", -1))
            mode_eligible = int(mode.get("candidates_eligible", -1))
            mode_rejected = int(mode.get("candidates_rejected", -1))
            mode_duplicate = int(mode.get("duplicate_count", -1))
            if mode_seen != mode_eligible + mode_rejected:
                blockers.append(f"mode {mode_id} count arithmetic mismatch")
            eligible_total += mode_eligible
            rejected_total += mode_rejected
            duplicate_total += mode_duplicate
            for candidate in candidates:
                if candidate.get("corridor_mode_id") != mode_id:
                    blockers.append(
                        f"mode {mode_id} contains candidate from another mode"
                    )
                if candidate.get("eligible") is not True:
                    blockers.append(f"mode {mode_id} contains ineligible candidate")
                if candidate.get("corridor_training_utility") is None:
                    blockers.append(f"mode {mode_id} contains null candidate utility")
                coordinate = (
                    str(candidate.get("candidate_id")),
                    int(candidate.get("position", -1)),
                )
                if coordinate in seen_coordinates:
                    blockers.append(f"mode {mode_id} contains duplicate coordinate")
                if coordinate in all_coordinates:
                    blockers.append("candidate coordinate appears in multiple pools")
                seen_coordinates.add(coordinate)
                all_coordinates.add(coordinate)
                key = _serialized_score_sort_key(candidate)
                if previous_key is not None and key < previous_key:
                    blockers.append(f"mode {mode_id} candidate ordering is invalid")
                previous_key = key
                for field_name in (
                    "membership_score",
                    "centrality_score",
                    "useful_difficulty_score",
                    "quality_score",
                    "corridor_training_utility",
                ):
                    value = candidate.get(field_name)
                    if not _bounded_finite(value):
                        blockers.append(
                            f"mode {mode_id} candidate {field_name} is not bounded"
                        )
        summary = manifest.get("summary") or {}
        if summary.get("modes_observed") != len(modes):
            blockers.append("summary modes_observed mismatch")
        retained = sum(len(mode.get("candidates", ())) for mode in modes)
        if summary.get("retained_candidate_count") != retained:
            blockers.append("summary retained_candidate_count mismatch")
        if summary.get("candidates_eligible") != eligible_total:
            blockers.append("summary candidates_eligible mismatch")
        if summary.get("candidates_rejected") != rejected_total:
            blockers.append("summary candidates_rejected mismatch")
        if summary.get("duplicates_collapsed") != duplicate_total:
            blockers.append("summary duplicates_collapsed mismatch")
        if (
            summary.get("candidates_seen")
            != eligible_total + rejected_total + duplicate_total
        ):
            blockers.append("summary candidate count arithmetic mismatch")
        provenance = manifest.get("feature_provenance")
        observed_proxy = (
            isinstance(provenance, dict)
            and provenance.get("fidelity") == "compatibility_proxy"
        )
        if not isinstance(provenance, dict):
            if modes:
                blockers.append("feature provenance is missing")
        elif observed_proxy:
            message = "leaderboard uses compatibility_proxy features"
            if production_grade:
                blockers.append(message)
            else:
                warnings.append(message)
        if summary.get("production_grade") != (not observed_proxy):
            blockers.append(
                "summary production_grade does not match observed provenance"
            )
        if summary.get("compatibility_proxy_used") != observed_proxy:
            blockers.append(
                "summary compatibility_proxy_used does not match observed provenance"
            )
        if (
            not blockers
            and not production_grade
            and summary.get("production_grade") is False
        ):
            warnings.extend(["artifact is non-production by explicit proxy override"])
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        blockers.append(f"leaderboard artifact unreadable: {exc}")
        summary = {}
    status: Literal["pass", "fail", "warn"] = "fail" if blockers else "pass"
    if status == "pass" and warnings:
        status = "warn"
    return CorridorLeaderboardValidationResult(
        status=status,
        blockers=tuple(dict.fromkeys(blockers)),
        warnings=tuple(dict.fromkeys(warnings)),
        summary=summary,
    )


def _validate_artifact(
    artifact: CorridorLeaderboardArtifact,
    *,
    production_grade: bool,
) -> CorridorLeaderboardValidationResult:
    blockers: list[str] = []
    warnings: list[str] = []
    previous_mode = -1
    retained = 0
    for mode in artifact.modes:
        if mode.corridor_mode_id <= previous_mode:
            blockers.append("mode leaderboards must be strictly ordered")
        previous_mode = mode.corridor_mode_id
        if len(mode.candidates) > artifact.policy.candidate_pool_cap:
            blockers.append(f"mode {mode.corridor_mode_id} exceeds candidate pool cap")
        previous_score: tuple[Any, ...] | None = None
        identities: set[tuple[str, int]] = set()
        for candidate in mode.candidates:
            retained += 1
            if not candidate.eligible or candidate.corridor_training_utility is None:
                blockers.append(
                    f"mode {mode.corridor_mode_id} has ineligible pool candidate"
                )
            identity = (candidate.candidate_id, candidate.position)
            if identity in identities:
                blockers.append(
                    f"mode {mode.corridor_mode_id} has duplicate coordinate"
                )
            identities.add(identity)
            current_score = _score_sort_key(candidate)
            if previous_score is not None and current_score < previous_score:
                blockers.append(f"mode {mode.corridor_mode_id} ordering is invalid")
            previous_score = current_score
    if artifact.feature_provenance is None and artifact.modes:
        blockers.append("feature provenance is required for non-empty artifacts")
    if (
        artifact.feature_provenance is not None
        and artifact.feature_provenance.fidelity == "compatibility_proxy"
    ):
        message = "leaderboard uses compatibility_proxy features"
        if production_grade:
            blockers.append(message)
        else:
            warnings.append(message)
    observed_proxy = (
        artifact.feature_provenance is not None
        and artifact.feature_provenance.fidelity == "compatibility_proxy"
    )
    if artifact.summary.get("production_grade") != (not observed_proxy):
        blockers.append("summary production_grade does not match observed provenance")
    if artifact.summary.get("compatibility_proxy_used") != observed_proxy:
        blockers.append(
            "summary compatibility_proxy_used does not match observed provenance"
        )
    if artifact.summary.get("retained_candidate_count") != retained:
        blockers.append("summary retained_candidate_count mismatch")
    if artifact.summary.get("modes_observed") != len(artifact.modes):
        blockers.append("summary modes_observed mismatch")
    status: Literal["pass", "fail", "warn"] = "fail" if blockers else "pass"
    if status == "pass" and warnings:
        status = "warn"
    return CorridorLeaderboardValidationResult(
        status=status,
        blockers=tuple(dict.fromkeys(blockers)),
        warnings=tuple(dict.fromkeys(warnings)),
        summary=dict(artifact.summary),
    )


def _read_modes(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    modes: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"mode leaderboard line {line_number} is not an object"
                )
            modes.append(payload)
    return modes


def _provenance_from_dict(payload: Mapping[str, Any]) -> CorridorFeatureProvenance:
    try:
        return CorridorFeatureProvenance(
            feature_schema_version=str(payload["schema_version"]),
            source_artifact_schema=str(payload["source_artifact_schema"]),
            source_artifact_id=str(payload["source_artifact_id"]),
            source_artifact_hash=payload.get("source_artifact_hash"),
            membership_derivation=str(payload["membership_derivation"]),
            core_distance_derivation=str(payload["core_distance_derivation"]),
            difficulty_derivation=str(payload["difficulty_derivation"]),
            fidelity=str(payload["fidelity"]),
            compatibility_proxy_used=_strict_bool(
                payload["compatibility_proxy_used"],
                "compatibility_proxy_used",
            ),
            normalization_parameters=payload.get("normalization_parameters", {}),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CorridorLeaderboardError(
            f"invalid C2 feature provenance payload: {exc}"
        ) from exc


def _score_from_dict(payload: Mapping[str, Any]) -> CorridorArchetypeScore:
    try:
        reasons = payload.get("eligibility_reasons", ())
        if not isinstance(reasons, list | tuple):
            raise TypeError("eligibility_reasons must be a list")
        return CorridorArchetypeScore(
            candidate_id=str(payload["candidate_id"]),
            position=int(payload["position"]),
            corridor_mode_id=payload.get("corridor_mode_id"),
            corridor_fingerprint_id=payload.get("corridor_fingerprint_id"),
            eligible=bool(payload["eligible"]),
            eligibility_reasons=tuple(str(reason) for reason in reasons),
            membership_score=float(payload["membership_score"]),
            centrality_score=float(payload["centrality_score"]),
            useful_difficulty_score=float(payload["useful_difficulty_score"]),
            quality_score=float(payload["quality_score"]),
            corridor_training_utility=(
                None
                if payload.get("corridor_training_utility") is None
                else float(payload["corridor_training_utility"])
            ),
            policy_id=str(payload["policy_id"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CorridorLeaderboardError(
            f"invalid C2 candidate score payload: {exc}"
        ) from exc


def _score_sort_key(score: CorridorArchetypeScore) -> tuple[Any, ...]:
    return (
        -float(score.corridor_training_utility or 0.0),
        -score.membership_score,
        -score.centrality_score,
        -score.useful_difficulty_score,
        score.candidate_id,
        score.position,
    )


def _serialized_score_sort_key(score: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -float(score.get("corridor_training_utility") or 0.0),
        -float(score.get("membership_score") or 0.0),
        -float(score.get("centrality_score") or 0.0),
        -float(score.get("useful_difficulty_score") or 0.0),
        str(score.get("candidate_id")),
        int(score.get("position", -1)),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _median(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _bounded_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0
    except (TypeError, ValueError):
        return False


def _require_fidelity(payload: Mapping[str, Any]) -> str:
    fidelity = payload.get("fidelity")
    if not isinstance(fidelity, str) or fidelity not in FEATURE_FIDELITIES:
        raise ValueError(
            "fidelity must be explicitly declared as explicit, derived, "
            "or compatibility_proxy"
        )
    return fidelity


def _validate_loader_feature_contract(
    features: Mapping[str, Any],
    record: Mapping[str, Any],
    fidelity: str,
) -> None:
    if fidelity == "compatibility_proxy":
        if (
            _strict_bool(
                record.get("compatibility_proxy_used"), "compatibility_proxy_used"
            )
            is not True
        ):
            raise ValueError(
                "compatibility_proxy records must set compatibility_proxy_used=true"
            )
        return

    required_features = (
        "membership_strength",
        "core_distance",
        "mode_support",
        "difficulty_score",
    )
    missing = [name for name in required_features if name not in features]
    if missing:
        raise ValueError(
            f"{fidelity} feature records require explicit fields: " + ", ".join(missing)
        )
    for name in ("membership_strength", "core_distance", "difficulty_score"):
        value = features[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{fidelity} field {name} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"{fidelity} field {name} must be finite")
    mode_support = features["mode_support"]
    if isinstance(mode_support, bool) or not isinstance(mode_support, int):
        raise ValueError(f"{fidelity} field mode_support must be an integer")
    if mode_support < 0:
        raise ValueError(f"{fidelity} field mode_support must be nonnegative")
    if fidelity == "derived":
        for name in (
            "membership_derivation",
            "core_distance_derivation",
            "difficulty_derivation",
        ):
            value = record.get(name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    "derived feature records require nonempty derivation metadata: "
                    + name
                )
    if "compatibility_proxy_used" in record and _strict_bool(
        record["compatibility_proxy_used"], "compatibility_proxy_used"
    ):
        raise ValueError(
            f"{fidelity} feature records cannot set compatibility_proxy_used=true"
        )


@contextmanager
def _temporary_coordinate_index():
    """Yield a disk-backed coordinate index for duplicate detection."""

    with tempfile.TemporaryDirectory(prefix="radjax-c2-coordinate-index-") as root:
        database = Path(root) / "coordinates.sqlite3"
        connection = sqlite3.connect(database)
        try:
            connection.execute("PRAGMA journal_mode = OFF")
            connection.execute("PRAGMA synchronous = OFF")
            connection.execute(
                "CREATE TABLE coordinates ("
                "candidate_id TEXT NOT NULL, "
                "position INTEGER NOT NULL, "
                "record_json TEXT NOT NULL, "
                "mode_id INTEGER, "
                "PRIMARY KEY (candidate_id, position)"
                ")"
            )
            yield connection
        finally:
            connection.close()


def _strict_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"{name} must be a boolean or explicit true/false string")
