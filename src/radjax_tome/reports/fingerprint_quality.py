from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from radjax_tome.io.json import read_json_object, write_json

EPSILON = 1.0e-8


@dataclass(frozen=True)
class FingerprintArtifactByteBudget:
    artifact_kind: str
    fingerprint_artifact_size_bytes: int
    manifest_size_bytes: int = 0
    targets_size_bytes: int = 0
    exemplars_size_bytes: int = 0
    modes_size_bytes: int = 0
    capture_summary_size_bytes: int = 0
    target_records: int = 0
    exemplar_records: int = 0
    modes_discovered: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> FingerprintArtifactByteBudget:
        return cls(
            artifact_kind=str(payload.get("artifact_kind", "")),
            fingerprint_artifact_size_bytes=_non_negative_int(
                payload.get("fingerprint_artifact_size_bytes", 0),
                "fingerprint_artifact_size_bytes",
            ),
            manifest_size_bytes=_non_negative_int(
                payload.get("manifest_size_bytes", 0),
                "manifest_size_bytes",
            ),
            targets_size_bytes=_non_negative_int(
                payload.get("targets_size_bytes", 0),
                "targets_size_bytes",
            ),
            exemplars_size_bytes=_non_negative_int(
                payload.get("exemplars_size_bytes", 0),
                "exemplars_size_bytes",
            ),
            modes_size_bytes=_non_negative_int(
                payload.get("modes_size_bytes", 0),
                "modes_size_bytes",
            ),
            capture_summary_size_bytes=_non_negative_int(
                payload.get("capture_summary_size_bytes", 0),
                "capture_summary_size_bytes",
            ),
            target_records=_non_negative_int(
                payload.get("target_records", 0),
                "target_records",
            ),
            exemplar_records=_non_negative_int(
                payload.get("exemplar_records", 0),
                "exemplar_records",
            ),
            modes_discovered=_non_negative_int(
                payload.get("modes_discovered", 0),
                "modes_discovered",
            ),
            metadata={
                str(key): value
                for key, value in payload.items()
                if key
                not in {
                    "artifact_kind",
                    "fingerprint_artifact_size_bytes",
                    "manifest_size_bytes",
                    "targets_size_bytes",
                    "exemplars_size_bytes",
                    "modes_size_bytes",
                    "capture_summary_size_bytes",
                    "target_records",
                    "exemplar_records",
                    "modes_discovered",
                }
            },
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        metadata = payload.pop("metadata")
        return {**payload, **metadata}


@dataclass(frozen=True)
class QualityPerByteDelta:
    quality_proxy: str
    artifact_byte_denominator: str
    absolute_delta: float
    relative_delta: float
    delta_per_mb: float
    trained_baseline_available: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> QualityPerByteDelta:
        return cls(
            quality_proxy=str(payload.get("quality_proxy", "")),
            artifact_byte_denominator=str(payload.get("artifact_byte_denominator", "")),
            absolute_delta=float(payload.get("absolute_delta", 0.0)),
            relative_delta=float(payload.get("relative_delta", 0.0)),
            delta_per_mb=float(payload.get("delta_per_mb", 0.0)),
            trained_baseline_available=bool(
                payload.get("trained_baseline_available", False)
            ),
            metadata={
                str(key): value
                for key, value in payload.items()
                if key
                not in {
                    "quality_proxy",
                    "artifact_byte_denominator",
                    "absolute_delta",
                    "relative_delta",
                    "delta_per_mb",
                    "trained_baseline_available",
                }
            },
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        metadata = payload.pop("metadata")
        return {**payload, **metadata}


@dataclass(frozen=True)
class FingerprintQualityPerByteReport:
    status: str
    artifact_budget: FingerprintArtifactByteBudget
    quality_per_byte: QualityPerByteDelta
    arms: tuple[dict[str, Any], ...] = ()
    limitations: tuple[str, ...] = ()
    claims: dict[str, bool] = field(default_factory=dict)
    experiment: dict[str, Any] = field(default_factory=dict)
    phase: str = "P148"

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> FingerprintQualityPerByteReport:
        return cls(
            status=str(payload.get("status", "")),
            artifact_budget=FingerprintArtifactByteBudget.from_payload(
                _mapping(payload.get("artifact_budget"))
            ),
            quality_per_byte=QualityPerByteDelta.from_payload(
                _quality_payload(payload.get("quality_per_byte"))
            ),
            arms=tuple(dict(item) for item in payload.get("arms", ())),
            limitations=tuple(str(item) for item in payload.get("limitations", ())),
            claims={
                str(key): bool(value)
                for key, value in _mapping(payload.get("claims")).items()
            },
            experiment=_mapping(payload.get("experiment")),
            phase=str(payload.get("phase", "P148")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "status": self.status,
            "experiment": dict(self.experiment),
            "artifact_budget": self.artifact_budget.to_dict(),
            "arms": [dict(arm) for arm in self.arms],
            "quality_per_byte": self.quality_per_byte.to_dict(),
            "claims": dict(self.claims),
            "limitations": list(self.limitations),
        }


def build_quality_per_byte_delta(
    *,
    reference_score: float,
    fingerprint_score: float,
    artifact_size_bytes: int,
    quality_proxy: str = "corridor_adherence",
) -> QualityPerByteDelta:
    size_mb = max(artifact_size_bytes / 1_000_000.0, EPSILON)
    absolute_delta = float(reference_score) - float(fingerprint_score)
    relative_delta = absolute_delta / max(abs(float(reference_score)), EPSILON)
    return QualityPerByteDelta(
        quality_proxy=quality_proxy,
        artifact_byte_denominator="fingerprint_artifact_size_bytes",
        absolute_delta=absolute_delta,
        relative_delta=relative_delta,
        delta_per_mb=absolute_delta / size_mb,
    )


def read_fingerprint_quality_report(
    path: str | Path,
) -> FingerprintQualityPerByteReport:
    return FingerprintQualityPerByteReport.from_payload(read_json_object(Path(path)))


def write_fingerprint_quality_report(
    path: str | Path,
    report: FingerprintQualityPerByteReport | dict[str, Any],
) -> Path:
    output = Path(path)
    payload = (
        report.to_dict()
        if isinstance(report, FingerprintQualityPerByteReport)
        else dict(report)
    )
    write_json(output, payload)
    return output


def render_fingerprint_quality_summary(
    report: FingerprintQualityPerByteReport | dict[str, Any],
) -> str:
    parsed = (
        report
        if isinstance(report, FingerprintQualityPerByteReport)
        else FingerprintQualityPerByteReport.from_payload(report)
    )
    return "\n".join(
        (
            "# Fingerprint Quality-Per-Byte Report",
            "",
            f"Status: {parsed.status}",
            f"Quality proxy: {parsed.quality_per_byte.quality_proxy}",
            f"Artifact bytes: {parsed.artifact_budget.fingerprint_artifact_size_bytes}",
            f"Absolute delta: {parsed.quality_per_byte.absolute_delta}",
            f"Delta per MB: {parsed.quality_per_byte.delta_per_mb}",
            "",
        )
    )


def _quality_payload(value: Any) -> dict[str, Any]:
    payload = _mapping(value)
    reference_delta = _mapping(payload.get("reference_delta_vs_init_only"))
    if reference_delta:
        return {
            "quality_proxy": payload.get("quality_proxy", ""),
            "artifact_byte_denominator": payload.get("artifact_byte_denominator", ""),
            "absolute_delta": reference_delta.get("absolute_corridor_loss_delta", 0.0),
            "relative_delta": reference_delta.get("relative_corridor_loss_delta", 0.0),
            "delta_per_mb": reference_delta.get("corridor_loss_delta_per_mb", 0.0),
            "trained_baseline_available": payload.get(
                "trained_baseline_available", False
            ),
        }
    return payload


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, int) and value >= 0:
        return value
    raise ValueError(f"{name} must be a non-negative integer")
