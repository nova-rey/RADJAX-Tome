from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from radjax_tome.io.json import read_json_object, write_json


@dataclass(frozen=True)
class BaselineArmReport:
    arm_id: str
    status: str
    arm_kind: str
    trained: bool
    artifact_kind: str
    artifact_size_bytes: int
    teacher_required_during_training: bool
    metrics: dict[str, float | int | bool | None] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> BaselineArmReport:
        metrics = _mapping(payload.get("metrics"))
        if not metrics and "eval" in payload:
            metrics = _mapping(payload.get("eval"))
        return cls(
            arm_id=str(payload.get("arm_id", "")),
            status=str(payload.get("status", "")),
            arm_kind=str(payload.get("arm_kind", payload.get("arm_id", ""))),
            trained=bool(payload.get("trained", False)),
            artifact_kind=str(payload.get("artifact_kind", "")),
            artifact_size_bytes=_non_negative_int(
                payload.get("artifact_size_bytes", 0),
                "artifact_size_bytes",
            ),
            teacher_required_during_training=bool(
                payload.get("teacher_required_during_training", False)
            ),
            metrics={str(key): value for key, value in metrics.items()},
            metadata={
                str(key): value
                for key, value in payload.items()
                if key
                not in {
                    "arm_id",
                    "status",
                    "arm_kind",
                    "trained",
                    "artifact_kind",
                    "artifact_size_bytes",
                    "teacher_required_during_training",
                    "metrics",
                    "eval",
                }
            },
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        metadata = payload.pop("metadata")
        return {**payload, **metadata}


@dataclass(frozen=True)
class FingerprintBaselineComparisonReport:
    status: str
    arms: tuple[BaselineArmReport, ...]
    artifact: dict[str, Any] = field(default_factory=dict)
    fairness: dict[str, Any] = field(default_factory=dict)
    claims: dict[str, bool] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()
    phase: str = "P147"

    @classmethod
    def from_payload(
        cls, payload: dict[str, Any]
    ) -> FingerprintBaselineComparisonReport:
        return cls(
            status=str(payload.get("status", "")),
            arms=tuple(
                BaselineArmReport.from_payload(dict(item))
                for item in payload.get("arms", ())
            ),
            artifact=_mapping(payload.get("artifact")),
            fairness=_mapping(payload.get("fairness")),
            claims={
                str(key): bool(value)
                for key, value in _mapping(payload.get("claims")).items()
            },
            limitations=tuple(str(item) for item in payload.get("limitations", ())),
            phase=str(payload.get("phase", "P147")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "status": self.status,
            "artifact": dict(self.artifact),
            "arms": [arm.to_dict() for arm in self.arms],
            "fairness": dict(self.fairness),
            "claims": dict(self.claims),
            "limitations": list(self.limitations),
        }


def read_fingerprint_baseline_report(
    path: str | Path,
) -> FingerprintBaselineComparisonReport:
    return FingerprintBaselineComparisonReport.from_payload(
        read_json_object(Path(path))
    )


def write_fingerprint_baseline_report(
    path: str | Path,
    report: FingerprintBaselineComparisonReport | dict[str, Any],
) -> Path:
    output = Path(path)
    payload = (
        report.to_dict()
        if isinstance(report, FingerprintBaselineComparisonReport)
        else dict(report)
    )
    write_json(output, payload)
    return output


def render_fingerprint_baseline_summary(
    report: FingerprintBaselineComparisonReport | dict[str, Any],
) -> str:
    parsed = (
        report
        if isinstance(report, FingerprintBaselineComparisonReport)
        else FingerprintBaselineComparisonReport.from_payload(report)
    )
    lines = [
        "# Fingerprint Baseline Comparison",
        "",
        f"Status: {parsed.status}",
        "",
        "## Arms",
    ]
    lines.extend(
        f"- {arm.arm_id}: status={arm.status}, trained={arm.trained}"
        for arm in parsed.arms
    )
    lines.append("")
    return "\n".join(lines)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, int) and value >= 0:
        return value
    raise ValueError(f"{name} must be a non-negative integer")
