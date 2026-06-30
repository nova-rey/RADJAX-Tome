from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from radjax_tome.io.json import read_json_object, write_json

DEFAULT_TRACKED_STATS = (
    "entropy",
    "top1_margin",
    "top8_mass",
    "top32_mass",
    "tail_mass",
)


@dataclass(frozen=True)
class CorridorMeasurementRecord:
    example_id: str
    position: int
    mode_id: int
    stats: dict[str, float]
    bounds: dict[str, tuple[float, float]]
    weight: float = 1.0

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> CorridorMeasurementRecord:
        return cls(
            example_id=str(payload.get("example_id", "")),
            position=int(payload.get("position", 0)),
            mode_id=int(payload.get("mode_id", 0)),
            stats={
                str(key): float(value)
                for key, value in _mapping(payload.get("stats")).items()
            },
            bounds={
                str(key): _bounds_tuple(value)
                for key, value in _mapping(payload.get("bounds")).items()
            },
            weight=float(payload.get("weight", 1.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CorridorMeasurementReport:
    status: str
    artifact_dir: str
    records_measured: int
    modes_measured: int
    tracked_stats: tuple[str, ...] = DEFAULT_TRACKED_STATS
    bounds_method: str = "minmax"
    measurements: tuple[CorridorMeasurementRecord, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> CorridorMeasurementReport:
        return cls(
            status=str(payload.get("status", "")),
            artifact_dir=str(payload.get("artifact_dir", "")),
            records_measured=_non_negative_int(
                payload.get("records_measured", 0),
                "records_measured",
            ),
            modes_measured=_non_negative_int(
                payload.get("modes_measured", 0),
                "modes_measured",
            ),
            tracked_stats=tuple(
                str(item)
                for item in payload.get("tracked_stats", DEFAULT_TRACKED_STATS)
            ),
            bounds_method=str(payload.get("bounds_method", "minmax")),
            measurements=tuple(
                CorridorMeasurementRecord.from_payload(dict(item))
                for item in payload.get("measurements", ())
            ),
            warnings=tuple(str(item) for item in payload.get("warnings", ())),
            metadata=_mapping(payload.get("metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "artifact_dir": self.artifact_dir,
            "records_measured": self.records_measured,
            "modes_measured": self.modes_measured,
            "tracked_stats": list(self.tracked_stats),
            "bounds_method": self.bounds_method,
            "measurements": [record.to_dict() for record in self.measurements],
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AggressivenessCalibrationProfile:
    profile_name: str
    corridor_scale: float
    exemplar_weight: float
    accepted: bool
    metrics: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> AggressivenessCalibrationProfile:
        return cls(
            profile_name=str(payload.get("profile_name", "")),
            corridor_scale=float(payload.get("corridor_scale", 1.0)),
            exemplar_weight=float(payload.get("exemplar_weight", 1.0)),
            accepted=bool(payload.get("accepted", False)),
            metrics={
                str(key): float(value)
                for key, value in _mapping(payload.get("metrics")).items()
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AggressivenessCalibrationReport:
    status: str
    selected_profile: str | None
    profiles: tuple[AggressivenessCalibrationProfile, ...]
    selection_allowed: bool
    blockers: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> AggressivenessCalibrationReport:
        return cls(
            status=str(payload.get("status", "")),
            selected_profile=(
                str(payload["selected_profile"])
                if payload.get("selected_profile") is not None
                else None
            ),
            profiles=tuple(
                AggressivenessCalibrationProfile.from_payload(dict(item))
                for item in payload.get("profiles", ())
            ),
            selection_allowed=bool(payload.get("selection_allowed", False)),
            blockers=tuple(str(item) for item in payload.get("blockers", ())),
            metadata=_mapping(payload.get("metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "selected_profile": self.selected_profile,
            "profiles": [profile.to_dict() for profile in self.profiles],
            "selection_allowed": self.selection_allowed,
            "blockers": list(self.blockers),
            "metadata": dict(self.metadata),
        }


def read_corridor_measurement_report(path: str | Path) -> CorridorMeasurementReport:
    return CorridorMeasurementReport.from_payload(read_json_object(Path(path)))


def write_corridor_measurement_report(
    path: str | Path,
    report: CorridorMeasurementReport | dict[str, Any],
) -> Path:
    output = Path(path)
    payload = (
        report.to_dict()
        if isinstance(report, CorridorMeasurementReport)
        else dict(report)
    )
    write_json(output, payload)
    return output


def validate_corridor_measurement_report(
    report: CorridorMeasurementReport | dict[str, Any],
) -> tuple[bool, tuple[str, ...]]:
    try:
        parsed = (
            report
            if isinstance(report, CorridorMeasurementReport)
            else CorridorMeasurementReport.from_payload(report)
        )
    except ValueError as exc:
        return False, (str(exc),)
    blockers: list[str] = []
    if parsed.status not in {"pass", "fail", "dry_run"}:
        blockers.append("status must be pass, fail, or dry_run")
    if not parsed.artifact_dir:
        blockers.append("artifact_dir is required")
    if not parsed.tracked_stats:
        blockers.append("tracked_stats must be non-empty")
    if parsed.records_measured != len(parsed.measurements) and parsed.measurements:
        blockers.append("records_measured does not match measurements length")
    for index, record in enumerate(parsed.measurements):
        if not record.example_id:
            blockers.append(f"measurements[{index}].example_id is required")
        if record.position < 0:
            blockers.append(f"measurements[{index}].position must be non-negative")
        for stat in parsed.tracked_stats:
            if stat not in record.stats:
                blockers.append(f"measurements[{index}] missing stat {stat!r}")
            if stat not in record.bounds:
                blockers.append(f"measurements[{index}] missing bounds {stat!r}")
    return not blockers, tuple(blockers)


def read_aggressiveness_calibration_report(
    path: str | Path,
) -> AggressivenessCalibrationReport:
    return AggressivenessCalibrationReport.from_payload(read_json_object(Path(path)))


def write_aggressiveness_calibration_report(
    path: str | Path,
    report: AggressivenessCalibrationReport | dict[str, Any],
) -> Path:
    output = Path(path)
    payload = (
        report.to_dict()
        if isinstance(report, AggressivenessCalibrationReport)
        else dict(report)
    )
    write_json(output, payload)
    return output


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _bounds_tuple(value: Any) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("corridor bounds must be [min, max]")
    lower, upper = float(value[0]), float(value[1])
    if lower > upper:
        raise ValueError("corridor lower bound exceeds upper bound")
    return lower, upper


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, int) and value >= 0:
        return value
    raise ValueError(f"{name} must be a non-negative integer")
