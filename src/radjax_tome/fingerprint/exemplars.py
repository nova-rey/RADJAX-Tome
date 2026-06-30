from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from radjax_tome.io.json import read_json_object, write_json
from radjax_tome.io.jsonl import read_jsonl_objects, write_jsonl_objects

FINGERPRINT_EXEMPLAR_PAYLOAD_DENSE = "dense_probs"
FINGERPRINT_EXEMPLAR_PAYLOAD_CASCADED = "cascaded_soft_labels_v1"
SUPPORTED_FINGERPRINT_EXEMPLAR_PAYLOAD_TYPES = frozenset(
    (FINGERPRINT_EXEMPLAR_PAYLOAD_DENSE, FINGERPRINT_EXEMPLAR_PAYLOAD_CASCADED)
)


@dataclass(frozen=True)
class FingerprintExemplarRecord:
    example_id: str
    input_ids: tuple[int, ...]
    position: int
    weight: float
    target_type: str = FINGERPRINT_EXEMPLAR_PAYLOAD_DENSE
    teacher_probs: tuple[float, ...] | None = None
    top_token_ids: tuple[int, ...] | None = None
    top_log_probs: tuple[float, ...] | None = None
    top_mass: float | None = None
    tail_mass: float | None = None
    teacher_entropy: float | None = None
    bucket_edges: tuple[float, ...] | None = None
    bucket_mass: tuple[float, ...] | None = None
    bucket_count: tuple[int, ...] | None = None
    bucket_mean_logp: tuple[float, ...] | None = None
    mode_id: int | None = None
    interestingness_score: float | None = None
    reason_codes: tuple[str, ...] = ()

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        payload_type: str | None = None,
    ) -> FingerprintExemplarRecord:
        target_type = str(
            payload.get("target_type")
            or payload_type
            or FINGERPRINT_EXEMPLAR_PAYLOAD_DENSE
        )
        return cls(
            example_id=str(payload.get("example_id", "")),
            input_ids=tuple(int(value) for value in payload.get("input_ids", ())),
            position=int(payload.get("position", 0)),
            weight=float(payload.get("weight", 1.0)),
            target_type=target_type,
            teacher_probs=_optional_float_tuple(payload.get("teacher_probs")),
            top_token_ids=_optional_int_tuple(payload.get("top_token_ids")),
            top_log_probs=_optional_float_tuple(payload.get("top_log_probs")),
            top_mass=_optional_float(payload.get("top_mass")),
            tail_mass=_optional_float(payload.get("tail_mass")),
            teacher_entropy=_optional_float(payload.get("teacher_entropy")),
            bucket_edges=_optional_float_tuple(payload.get("bucket_edges")),
            bucket_mass=_optional_float_tuple(payload.get("bucket_mass")),
            bucket_count=_optional_int_tuple(payload.get("bucket_count")),
            bucket_mean_logp=_optional_float_tuple(payload.get("bucket_mean_logp")),
            mode_id=(
                int(payload["mode_id"]) if payload.get("mode_id") is not None else None
            ),
            interestingness_score=_optional_float(payload.get("interestingness_score")),
            reason_codes=tuple(str(item) for item in payload.get("reason_codes", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class FingerprintExemplarShard:
    path: str
    num_records: int
    payload_type: str | None = None
    sha256: str | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> FingerprintExemplarShard:
        return cls(
            path=str(payload.get("path", "")),
            num_records=_non_negative_int(payload.get("num_records"), "num_records"),
            payload_type=(
                str(payload["payload_type"])
                if payload.get("payload_type") is not None
                else None
            ),
            sha256=str(payload["sha256"]) if payload.get("sha256") else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class FingerprintExemplarManifest:
    payload_type: str
    shards: tuple[FingerprintExemplarShard, ...]
    num_records: int
    selection_policy: str = "top_interestingness_v0"
    encoding_contract: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> FingerprintExemplarManifest:
        return cls(
            payload_type=str(
                payload.get("payload_type", FINGERPRINT_EXEMPLAR_PAYLOAD_DENSE)
            ),
            shards=tuple(
                FingerprintExemplarShard.from_payload(dict(item))
                for item in payload.get("shards", ())
            ),
            num_records=_non_negative_int(payload.get("num_records", 0), "num_records"),
            selection_policy=str(
                payload.get("selection_policy", "top_interestingness_v0")
            ),
            encoding_contract=_mapping(payload.get("encoding_contract")),
            metadata={
                str(key): value
                for key, value in payload.items()
                if key
                not in {
                    "payload_type",
                    "shards",
                    "num_records",
                    "selection_policy",
                    "encoding_contract",
                }
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "payload_type": self.payload_type,
            "shards": [shard.to_dict() for shard in self.shards],
            "num_records": self.num_records,
            "selection_policy": self.selection_policy,
            "encoding_contract": dict(self.encoding_contract),
            **dict(self.metadata),
        }


@dataclass(frozen=True)
class FingerprintExemplarValidationResult:
    ok: bool
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return "pass" if self.ok else "fail"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "status": self.status,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class FingerprintExemplarReservoirSummary:
    payload_type: str
    shard_count: int
    num_records: int
    selection_policy: str
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_fingerprint_exemplar_manifest(path: str | Path) -> FingerprintExemplarManifest:
    return FingerprintExemplarManifest.from_payload(read_json_object(Path(path)))


def write_fingerprint_exemplar_manifest(
    path: str | Path, manifest: FingerprintExemplarManifest | dict[str, Any]
) -> Path:
    output = Path(path)
    payload = (
        manifest.to_dict()
        if isinstance(manifest, FingerprintExemplarManifest)
        else dict(manifest)
    )
    write_json(output, payload)
    return output


def read_fingerprint_exemplar_records(
    path: str | Path,
    *,
    payload_type: str | None = None,
) -> tuple[FingerprintExemplarRecord, ...]:
    return tuple(
        FingerprintExemplarRecord.from_payload(row, payload_type=payload_type)
        for row in read_jsonl_objects(Path(path))
    )


def write_fingerprint_exemplar_records(
    path: str | Path,
    records: tuple[FingerprintExemplarRecord, ...] | list[FingerprintExemplarRecord],
) -> Path:
    output = Path(path)
    write_jsonl_objects(output, (record.to_dict() for record in records))
    return output


def validate_fingerprint_exemplar_records(
    records: tuple[FingerprintExemplarRecord, ...] | list[FingerprintExemplarRecord],
    *,
    max_seq_len: int,
    vocab_size: int,
    payload_type: str | None = None,
) -> FingerprintExemplarValidationResult:
    blockers: list[str] = []
    reason_codes: set[str] = set()
    expected_payload_type = payload_type
    if expected_payload_type is not None:
        _validate_payload_type(expected_payload_type, blockers, "payload_type")
    if max_seq_len <= 0:
        blockers.append("max_seq_len must be positive")
    if vocab_size <= 0:
        blockers.append("vocab_size must be positive")
    for index, record in enumerate(records):
        if not record.example_id:
            blockers.append(f"records[{index}].example_id is required")
        if len(record.input_ids) != max_seq_len:
            blockers.append(f"records[{index}].input_ids length must equal max_seq_len")
        if record.position < 0 or record.position >= max_seq_len:
            blockers.append(f"records[{index}].position out of range")
        record_payload_type = record.target_type
        _validate_payload_type(record_payload_type, blockers, f"records[{index}]")
        if (
            expected_payload_type is not None
            and record_payload_type != expected_payload_type
        ):
            blockers.append(f"records[{index}].target_type mismatch")
        if record_payload_type == FINGERPRINT_EXEMPLAR_PAYLOAD_DENSE:
            if record.teacher_probs is None:
                blockers.append(f"records[{index}].teacher_probs is required")
            elif len(record.teacher_probs) != vocab_size:
                blockers.append(
                    f"records[{index}].teacher_probs length must equal vocab_size"
                )
        if record_payload_type == FINGERPRINT_EXEMPLAR_PAYLOAD_CASCADED:
            _validate_cascaded_record(record, blockers, index)
        reason_codes.update(record.reason_codes)
    return FingerprintExemplarValidationResult(
        ok=not blockers,
        blockers=tuple(blockers),
        metadata={
            "records": len(records),
            "payload_type": payload_type,
            "reason_codes": sorted(reason_codes),
        },
    )


def summarize_exemplar_reservoir(
    artifact_dir: str | Path,
) -> FingerprintExemplarReservoirSummary:
    from radjax_tome.fingerprint.artifacts import read_fingerprint_manifest

    root = Path(artifact_dir)
    manifest = read_fingerprint_manifest(root)
    reservoir = manifest.exemplar_reservoir
    if not reservoir:
        return FingerprintExemplarReservoirSummary(
            payload_type="",
            shard_count=0,
            num_records=0,
            selection_policy="",
            reason_codes=(),
        )
    exemplar_manifest = FingerprintExemplarManifest.from_payload(reservoir)
    reason_codes: set[str] = set()
    observed = 0
    for shard in exemplar_manifest.shards:
        records = read_fingerprint_exemplar_records(
            root / shard.path,
            payload_type=exemplar_manifest.payload_type,
        )
        observed += len(records)
        for record in records:
            reason_codes.update(record.reason_codes)
    return FingerprintExemplarReservoirSummary(
        payload_type=exemplar_manifest.payload_type,
        shard_count=len(exemplar_manifest.shards),
        num_records=observed,
        selection_policy=exemplar_manifest.selection_policy,
        reason_codes=tuple(sorted(reason_codes)),
    )


def _validate_cascaded_record(
    record: FingerprintExemplarRecord,
    blockers: list[str],
    index: int,
) -> None:
    required = (
        "top_token_ids",
        "top_log_probs",
        "top_mass",
        "tail_mass",
        "teacher_entropy",
        "bucket_edges",
        "bucket_mass",
        "bucket_count",
        "bucket_mean_logp",
    )
    for field_name in required:
        if getattr(record, field_name) is None:
            blockers.append(f"records[{index}].{field_name} is required")
    if record.top_token_ids is not None and record.top_log_probs is not None:
        if len(record.top_token_ids) != len(record.top_log_probs):
            blockers.append(
                f"records[{index}].top_token_ids and top_log_probs length mismatch"
            )


def _validate_payload_type(value: str, blockers: list[str], source: str) -> None:
    if value not in SUPPORTED_FINGERPRINT_EXEMPLAR_PAYLOAD_TYPES:
        blockers.append(f"{source} unsupported exemplar payload type: {value!r}")


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _optional_float_tuple(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    return tuple(float(item) for item in value)


def _optional_int_tuple(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    return tuple(int(item) for item in value)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, int) and value >= 0:
        return value
    raise ValueError(f"{name} must be a non-negative integer")
