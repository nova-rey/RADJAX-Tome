from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from radjax_tome.io.json import read_json_object

BEHAVIORAL_FINGERPRINT_ARTIFACT_TYPE = "behavioral_fingerprint"
BEHAVIORAL_FINGERPRINT_VERSION = "0.1"
SUPPORTED_BEHAVIORAL_FINGERPRINT_VERSIONS = (BEHAVIORAL_FINGERPRINT_VERSION,)


@dataclass(frozen=True)
class FingerprintManifest:
    artifact_type: str
    artifact_version: str
    created_by: str
    teacher: dict[str, Any]
    sequence: dict[str, Any]
    stats: dict[str, Any]
    modes_file: str
    target_shards: tuple[dict[str, Any], ...]
    target_payload: dict[str, Any] | None = None
    exemplar_reservoir: dict[str, Any] | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> FingerprintManifest:
        return cls(
            artifact_type=str(payload.get("artifact_type", "")),
            artifact_version=str(payload.get("artifact_version", "")),
            created_by=str(payload.get("created_by", "")),
            teacher=_mapping_or_empty(payload.get("teacher")),
            sequence=_mapping_or_empty(payload.get("sequence")),
            stats=_mapping_or_empty(payload.get("stats")),
            modes_file=str(payload.get("modes_file", "")),
            target_shards=tuple(
                _mapping_or_empty(item) for item in payload.get("target_shards", ())
            ),
            target_payload=(
                _mapping_or_empty(payload.get("target_payload"))
                if payload.get("target_payload") is not None
                else None
            ),
            exemplar_reservoir=(
                _mapping_or_empty(payload.get("exemplar_reservoir"))
                if payload.get("exemplar_reservoir") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class FingerprintValidationResult:
    ok: bool
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return "pass" if self.ok else "fail"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status
        return payload


@dataclass(frozen=True)
class FingerprintArtifactSummary:
    artifact_type: str
    artifact_version: str
    artifact_dir: str
    teacher_model_name: str
    tokenizer_name: str
    vocab_size: int
    max_seq_len: int
    tracked_stats: tuple[str, ...]
    num_modes: int
    num_corridor_records: int
    has_exemplars: bool
    exemplar_payload_type: str | None
    num_exemplar_records: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_fingerprint_artifact(path: str | Path) -> FingerprintValidationResult:
    root = Path(path)
    blockers: list[str] = []
    metadata: dict[str, Any] = {
        "artifact_type": BEHAVIORAL_FINGERPRINT_ARTIFACT_TYPE,
        "artifact_version": None,
        "shards": 0,
        "records": 0,
        "modes": 0,
        "exemplar_reservoir_enabled": False,
        "exemplar_payload_type": None,
        "exemplar_records": 0,
    }
    if not root.is_dir():
        return _result(blockers=[f"artifact path is not a directory: {root}"])
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return _result(blockers=[f"missing manifest.json: {manifest_path}"])
    try:
        manifest = FingerprintManifest.from_payload(read_json_object(manifest_path))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _result(blockers=[f"manifest.json invalid: {exc}"])

    metadata["artifact_version"] = manifest.artifact_version
    blockers.extend(_validate_manifest(manifest))
    tracked_stats = _tracked_stats(manifest.stats)

    modes_by_id: set[int] = set()
    if manifest.modes_file:
        modes_path = root / manifest.modes_file
        if not modes_path.is_file():
            blockers.append(f"modes_file does not exist: {manifest.modes_file}")
        else:
            mode_blockers, modes_by_id = _validate_modes(modes_path, tracked_stats)
            blockers.extend(mode_blockers)
            metadata["modes"] = len(modes_by_id)

    total_records = 0
    for shard_index, shard in enumerate(manifest.target_shards):
        shard_path_value = shard.get("path")
        if not isinstance(shard_path_value, str) or not shard_path_value.strip():
            blockers.append(f"target_shards[{shard_index}] missing non-empty path")
            continue
        shard_path = root / shard_path_value
        if not shard_path.is_file():
            blockers.append(f"target shard does not exist: {shard_path_value}")
            continue
        expected = _non_negative_int(shard.get("num_records"))
        actual = _count_jsonl_records(shard_path)
        total_records += actual
        if expected is not None and expected != actual:
            blockers.append(
                f"target shard record count mismatch for {shard_path_value}: "
                f"expected {expected}, got {actual}"
            )
    metadata["shards"] = len(manifest.target_shards)
    metadata["records"] = total_records

    target_positions = manifest.sequence.get("target_positions")
    if target_positions is not None:
        expected_total = _non_negative_int(target_positions)
        if expected_total is None:
            blockers.append("sequence.target_positions must be a non-negative integer")
        elif total_records != expected_total:
            blockers.append(
                "sequence.target_positions mismatch: "
                f"expected {expected_total}, got {total_records}"
            )

    reservoir = manifest.exemplar_reservoir
    if reservoir:
        metadata["exemplar_reservoir_enabled"] = True
        metadata["exemplar_payload_type"] = reservoir.get("payload_type")
        exemplar_count, exemplar_blockers = _validate_exemplar_reservoir(
            root, reservoir
        )
        metadata["exemplar_records"] = exemplar_count
        blockers.extend(exemplar_blockers)

    return _result(blockers=blockers, metadata=metadata)


def summarize_fingerprint_artifact(path: str | Path) -> FingerprintArtifactSummary:
    artifact_dir = Path(path)
    validation = validate_fingerprint_artifact(artifact_dir)
    if not validation.ok:
        joined = "; ".join(validation.blockers)
        raise ValueError(f"fingerprint artifact validation failed: {joined}")

    manifest = FingerprintManifest.from_payload(
        read_json_object(artifact_dir / "manifest.json")
    )
    tracked = manifest.stats.get("tracked", ())
    return FingerprintArtifactSummary(
        artifact_type=manifest.artifact_type,
        artifact_version=manifest.artifact_version,
        artifact_dir=str(artifact_dir),
        teacher_model_name=str(manifest.teacher.get("model_name", "")),
        tokenizer_name=str(manifest.teacher.get("tokenizer_name", "")),
        vocab_size=_positive_int(manifest.teacher.get("vocab_size"), "vocab_size"),
        max_seq_len=_positive_int(manifest.sequence.get("max_seq_len"), "max_seq_len"),
        tracked_stats=tuple(str(stat) for stat in tracked),
        num_modes=int(validation.metadata.get("modes", 0)),
        num_corridor_records=int(validation.metadata.get("records", 0)),
        has_exemplars=bool(validation.metadata.get("exemplar_reservoir_enabled")),
        exemplar_payload_type=validation.metadata.get("exemplar_payload_type"),
        num_exemplar_records=int(validation.metadata.get("exemplar_records", 0)),
    )


def _validate_manifest(manifest: FingerprintManifest) -> list[str]:
    blockers: list[str] = []
    if manifest.artifact_type != BEHAVIORAL_FINGERPRINT_ARTIFACT_TYPE:
        blockers.append(
            "manifest artifact_type must be "
            f"{BEHAVIORAL_FINGERPRINT_ARTIFACT_TYPE!r}, got {manifest.artifact_type!r}"
        )
    if manifest.artifact_version not in SUPPORTED_BEHAVIORAL_FINGERPRINT_VERSIONS:
        blockers.append(
            f"manifest artifact_version unsupported: {manifest.artifact_version!r}"
        )
    if not manifest.created_by:
        blockers.append("manifest created_by is required")
    if _positive_int(manifest.teacher.get("vocab_size"), "teacher.vocab_size") <= 0:
        blockers.append("teacher.vocab_size must be a positive integer")
    if _positive_int(manifest.sequence.get("max_seq_len"), "sequence.max_seq_len") <= 0:
        blockers.append("sequence.max_seq_len must be a positive integer")
    if not isinstance(manifest.stats.get("tracked", ()), (list, tuple)):
        blockers.append("stats.tracked must be a sequence")
    return blockers


def _validate_modes(
    path: Path, tracked_stats: tuple[str, ...]
) -> tuple[list[str], set[int]]:
    blockers: list[str] = []
    payload = read_json_object(path)
    modes = payload.get("modes", [])
    if not isinstance(modes, list):
        return ["modes payload must contain a list field named modes"], set()
    mode_ids: set[int] = set()
    for index, mode in enumerate(modes):
        if not isinstance(mode, dict):
            blockers.append(f"modes[{index}] must be an object")
            continue
        mode_id = _non_negative_int(mode.get("mode_id"))
        if mode_id is None:
            blockers.append(f"modes[{index}].mode_id must be a non-negative integer")
        else:
            mode_ids.add(mode_id)
        for stat in tracked_stats:
            if stat not in mode:
                blockers.append(f"modes[{index}] missing tracked stat {stat!r}")
    return blockers, mode_ids


def _validate_exemplar_reservoir(
    root: Path, reservoir: dict[str, Any]
) -> tuple[int, list[str]]:
    blockers: list[str] = []
    shards = reservoir.get("shards", ())
    if not isinstance(shards, (list, tuple)):
        return 0, ["exemplar_reservoir.shards must be a sequence"]
    total = 0
    for index, shard in enumerate(shards):
        if not isinstance(shard, dict):
            blockers.append(f"exemplar_reservoir.shards[{index}] must be an object")
            continue
        shard_path_value = shard.get("path")
        if not isinstance(shard_path_value, str) or not shard_path_value.strip():
            blockers.append(f"exemplar_reservoir.shards[{index}] missing path")
            continue
        shard_path = root / shard_path_value
        if not shard_path.is_file():
            blockers.append(f"exemplar shard does not exist: {shard_path_value}")
            continue
        total += _count_jsonl_records(shard_path)
    return total, blockers


def _tracked_stats(stats: dict[str, Any]) -> tuple[str, ...]:
    value = stats.get("tracked", ())
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value)


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _non_negative_int(value: Any) -> int | None:
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, int) and value > 0:
        return value
    raise ValueError(f"manifest {name} must be a positive integer")


def _count_jsonl_records(path: Path) -> int:
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        json.loads(line)
        count += 1
    return count


def _result(
    *,
    blockers: list[str],
    warnings: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> FingerprintValidationResult:
    return FingerprintValidationResult(
        ok=not blockers,
        blockers=tuple(blockers),
        warnings=tuple(warnings or ()),
        metadata=dict(metadata or {}),
    )
