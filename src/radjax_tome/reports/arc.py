from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from radjax_tome.io.json import read_json_object, write_json

REQUIRED_ARC2_FLAGS: tuple[tuple[str, str], ...] = (
    ("p140_real_student_forward_smoke", "P140 real student forward smoke"),
    ("p141_main_runner_fingerprint_mode", "P141 main runner fingerprint mode"),
    ("p142_input_conditioned_rehearsal", "P142 input-conditioned rehearsal"),
    ("p143_teacher_side_capture_skeleton", "P143 teacher-side capture skeleton"),
    ("p144_known_stat_parity", "P144 capture parity"),
    ("p145_tiny_real_teacher_capture_path", "P145 tiny real teacher capture"),
    (
        "p146_real_teacher_artifact_training_rehearsal",
        "P146 real-teacher artifact training rehearsal",
    ),
    ("p147_baseline_comparison_harness", "P147 baseline comparison harness"),
    ("p148_quality_per_byte_experiment", "P148 quality-per-byte smoke"),
)


@dataclass(frozen=True)
class FingerprintArcReport:
    status: str
    recommendation: str
    evidence: tuple[dict[str, Any], ...]
    constraints: tuple[str, ...]
    open_gaps: tuple[dict[str, str], ...]
    claims: dict[str, bool]
    source_snapshot: str | None = None
    phase: str = "P149"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> FingerprintArcReport:
        go_no_go = _mapping(payload.get("go_no_go"))
        return cls(
            status=str(payload.get("status", "")),
            recommendation=str(
                payload.get("recommendation", go_no_go.get("recommendation", ""))
            ),
            evidence=tuple(dict(item) for item in payload.get("evidence", ())),
            constraints=tuple(str(item) for item in go_no_go.get("constraints", ())),
            open_gaps=tuple(dict(item) for item in payload.get("open_gaps", ())),
            claims={
                str(key): bool(value)
                for key, value in _mapping(payload.get("claims")).items()
            },
            source_snapshot=(
                str(payload["source_snapshot"])
                if payload.get("source_snapshot") is not None
                else None
            ),
            phase=str(payload.get("phase", "P149")),
            metadata={
                str(key): value
                for key, value in payload.items()
                if key
                not in {
                    "status",
                    "recommendation",
                    "evidence",
                    "go_no_go",
                    "open_gaps",
                    "claims",
                    "source_snapshot",
                    "phase",
                }
            },
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        metadata = payload.pop("metadata")
        recommendation = payload.pop("recommendation")
        constraints = payload.pop("constraints")
        payload["go_no_go"] = {
            "recommendation": recommendation,
            "go": self.status == "pass",
            "constraints": list(constraints),
            "no_go_blockers": [
                f"{item['flag']} missing"
                for item in self.evidence
                if not bool(item.get("present", False))
            ],
        }
        return {**payload, **metadata}


def build_fingerprint_arc_report(
    snapshot: dict[str, Any],
    *,
    snapshot_path: str | Path | None = None,
) -> FingerprintArcReport:
    main_contains = _mapping(snapshot.get("main_contains"))
    evidence = tuple(
        {
            "flag": flag,
            "description": description,
            "present": bool(main_contains.get(flag, False)),
        }
        for flag, description in REQUIRED_ARC2_FLAGS
    )
    missing = [item for item in evidence if not item["present"]]
    status = "pass" if not missing else "fail"
    return FingerprintArcReport(
        status=status,
        recommendation="go_with_constraints" if status == "pass" else "no_go",
        evidence=evidence,
        constraints=_constraints(),
        open_gaps=_open_gaps(),
        claims={
            "general_quality_claim_made": False,
            "trained_baseline_win_claim_made": False,
            "radlads_parity_claim_made": False,
            "scale_readiness_claim_made": False,
            "production_readiness_claim_made": False,
            "kernel_default_claim_made": False,
        },
        source_snapshot=str(snapshot_path) if snapshot_path is not None else None,
        metadata={
            "arc": {
                "id": "arc2_real_student_integration_and_teacher_pure_capture",
                "current_phase": snapshot.get("current_phase"),
                "checkpoint": snapshot.get("checkpoint"),
                "covered_phases": [f"P{number}" for number in range(140, 149)],
            }
        },
    )


def read_fingerprint_arc_report(path: str | Path) -> FingerprintArcReport:
    return FingerprintArcReport.from_payload(read_json_object(Path(path)))


def write_fingerprint_arc_report(
    path: str | Path,
    report: FingerprintArcReport | dict[str, Any],
) -> Path:
    output = Path(path)
    payload = (
        report.to_dict() if isinstance(report, FingerprintArcReport) else dict(report)
    )
    write_json(output, payload)
    return output


def render_fingerprint_arc_summary(
    report: FingerprintArcReport | dict[str, Any],
) -> str:
    parsed = (
        report
        if isinstance(report, FingerprintArcReport)
        else FingerprintArcReport.from_payload(report)
    )
    lines = [
        "# Fingerprint Arc Report",
        "",
        f"Status: {parsed.status}",
        f"Recommendation: {parsed.recommendation}",
        "",
        "## Evidence",
    ]
    lines.extend(
        f"- {item['description']}: {item['present']}" for item in parsed.evidence
    )
    lines.append("")
    return "\n".join(lines)


def _constraints() -> tuple[str, ...]:
    return (
        "Proceed only to larger controlled fingerprint experiments, "
        "not production scale.",
        "Add a trained non-fingerprint baseline before method-vs-method claims.",
        "Keep teacher/corpus/student budgets fixed and reported.",
        "Keep quality-per-byte claims scoped to measured tiny smoke settings.",
        "Do not claim RADLADS parity, scale readiness, or production readiness.",
    )


def _open_gaps() -> tuple[dict[str, str], ...]:
    return (
        {
            "id": "trained_baseline",
            "severity": "high",
            "summary": "No competitive trained non-fingerprint baseline exists yet.",
        },
        {
            "id": "generalization_eval",
            "severity": "high",
            "summary": (
                "Quality reports must distinguish train reuse from held-out eval."
            ),
        },
        {
            "id": "scale",
            "severity": "medium",
            "summary": "Artifacts and runs remain tiny JSONL CPU-safe smokes.",
        },
    )


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}
