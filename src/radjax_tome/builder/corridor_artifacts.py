from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from radjax_tome.builder.teacher_textbook import TinyTextExample
from radjax_tome.io.json import read_json_object, write_json
from radjax_tome.targets.store import TeacherTargetStore

FINGERPRINT_POLICY = "top_token_entropy_confidence_v1"
MODE_POLICY = "fingerprint_group_v1"
ASSIGNMENT_POLICY = "selected_and_representative_positions_v1"
CORRIDOR_SUMMARY_SCHEMA = "corridor_summary_v2"
CORRIDOR_FINGERPRINTS_SCHEMA = "corridor_fingerprints_v1"
CORRIDOR_MODES_SCHEMA = "corridor_modes_v1"
CORRIDOR_ASSIGNMENTS_SCHEMA = "corridor_mode_assignments_v1"


@dataclass(frozen=True)
class CorridorArtifactBuildResult:
    summary_path: Path
    fingerprints_path: Path
    modes_path: Path
    assignments_path: Path
    human_summary_path: Path
    fingerprint_count: int
    mode_count: int
    selected_exemplars_linked: bool

    def report_fields(self) -> dict[str, Any]:
        return {
            "corridor_artifact_built": True,
            "corridor_modes_built": True,
            "corridor_summary_path": str(self.summary_path),
            "corridor_fingerprints_path": str(self.fingerprints_path),
            "corridor_modes_path": str(self.modes_path),
            "corridor_mode_assignments_path": str(self.assignments_path),
            "corridor_fingerprint_count": self.fingerprint_count,
            "corridor_mode_count": self.mode_count,
            "selected_exemplars_linked_to_corridor_modes": (
                self.selected_exemplars_linked
            ),
        }


@dataclass(frozen=True)
class CorridorArtifactValidationResult:
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    corridor_artifact_ok: bool = False
    corridor_fingerprints_ok: bool = False
    corridor_modes_ok: bool = False
    corridor_mode_count: int = 0
    corridor_fingerprint_count: int = 0

    @property
    def ok(self) -> bool:
        return not self.blockers

    def report_fields(self) -> dict[str, Any]:
        return {
            "corridor_artifact_ok": self.corridor_artifact_ok,
            "corridor_fingerprints_ok": self.corridor_fingerprints_ok,
            "corridor_modes_ok": self.corridor_modes_ok,
            "corridor_mode_count": self.corridor_mode_count,
            "corridor_fingerprint_count": self.corridor_fingerprint_count,
        }


@dataclass(frozen=True)
class _Observation:
    example_id: str
    position: int
    top_token_id: int
    entropy: float
    confidence: float
    length: int
    source_policy_id: int

    @property
    def key(self) -> tuple[str, int]:
        return (self.example_id, self.position)


def build_corridor_artifacts(
    *,
    output_dir: Path,
    examples: tuple[TinyTextExample, ...],
    selected_records: list[dict[str, Any]],
    selected_payloads: list[dict[str, Any]],
    delivery_path: str,
    non_selected_exemplar_payload_retained: bool,
    fingerprint_policy: str = FINGERPRINT_POLICY,
    mode_policy: str = MODE_POLICY,
) -> CorridorArtifactBuildResult:
    store = TeacherTargetStore.open(output_dir)
    observations = _score_observations(store, examples)
    fingerprints, observation_to_fingerprint = _fingerprints(
        observations,
        target_type="corridor_exemplar_v1",
        source_target_type=store.metadata.target_type,
        num_examples_scored=store.metadata.num_examples,
        num_positions_scored=store.metadata.num_examples
        * store.metadata.sequence_length,
        fingerprint_policy=fingerprint_policy,
    )
    modes, fingerprint_to_mode = _modes(
        fingerprints,
        mode_policy=mode_policy,
        observation_count=len(observations),
    )
    linked = _link_selected_exemplars(
        selected_records,
        selected_payloads,
        observation_to_fingerprint=observation_to_fingerprint,
        fingerprint_to_mode=fingerprint_to_mode,
    )
    assignments = _mode_assignments(
        selected_records,
        fingerprints=fingerprints,
        fingerprint_to_mode=fingerprint_to_mode,
    )
    selected_linked = bool(selected_records) and linked == len(selected_records)

    corridors_dir = output_dir / "corridors"
    corridors_dir.mkdir(parents=True, exist_ok=True)
    summary_path = corridors_dir / "corridor_summary.json"
    fingerprints_path = corridors_dir / "corridor_fingerprints.json"
    modes_path = corridors_dir / "corridor_modes.json"
    assignments_path = corridors_dir / "mode_assignments.json"
    human_summary_path = corridors_dir / "corridor_summary.txt"

    summary = {
        "schema_version": CORRIDOR_SUMMARY_SCHEMA,
        "corridor_artifact_built": True,
        "corridor_fingerprints_retained": True,
        "corridor_modes_built": True,
        "fingerprint_count": len(fingerprints),
        "mode_count": len(modes),
        "num_examples_scored": store.metadata.num_examples,
        "num_positions_scored": store.metadata.num_examples
        * store.metadata.sequence_length,
        "fingerprint_policy": fingerprint_policy,
        "mode_policy": mode_policy,
        "selected_exemplar_count": len(selected_records),
        "selected_exemplars_linked_to_modes": selected_linked,
        "non_selected_exemplar_payload_retained": (
            non_selected_exemplar_payload_retained
        ),
        "delivery_path": delivery_path,
    }
    write_json(summary_path, summary)
    write_json(
        fingerprints_path,
        {
            "schema_version": CORRIDOR_FINGERPRINTS_SCHEMA,
            "target_type": "corridor_exemplar_v1",
            "source_target_type": "corridor_exemplar_score_pass_v1",
            "source_store_target_type": store.metadata.target_type,
            "num_examples_scored": store.metadata.num_examples,
            "num_positions_scored": store.metadata.num_examples
            * store.metadata.sequence_length,
            "fingerprint_policy": fingerprint_policy,
            "fingerprint_count": len(fingerprints),
            "fingerprints": fingerprints,
        },
    )
    write_json(
        modes_path,
        {
            "schema_version": CORRIDOR_MODES_SCHEMA,
            "mode_policy": mode_policy,
            "mode_count": len(modes),
            "modes": modes,
        },
    )
    write_json(
        assignments_path,
        {
            "schema_version": CORRIDOR_ASSIGNMENTS_SCHEMA,
            "assignment_policy": ASSIGNMENT_POLICY,
            "full_assignment_retained": False,
            "assignments": assignments,
        },
    )
    human_summary_path.write_text(
        _human_summary(
            num_examples=store.metadata.num_examples,
            num_positions=store.metadata.num_examples * store.metadata.sequence_length,
            mode_count=len(modes),
            fingerprint_count=len(fingerprints),
            selected_count=len(selected_records),
            selected_payload_retained=bool(selected_payloads),
            non_selected_payload_retained=non_selected_exemplar_payload_retained,
            delivery_path=delivery_path,
        ),
        encoding="utf-8",
    )
    return CorridorArtifactBuildResult(
        summary_path=summary_path,
        fingerprints_path=fingerprints_path,
        modes_path=modes_path,
        assignments_path=assignments_path,
        human_summary_path=human_summary_path,
        fingerprint_count=len(fingerprints),
        mode_count=len(modes),
        selected_exemplars_linked=selected_linked,
    )


def validate_corridor_artifacts(
    output_dir: Path,
    *,
    selected_records: list[dict[str, Any]] | None = None,
    selected_payloads: list[dict[str, Any]] | None = None,
    expected_selected_count: int | None = None,
) -> CorridorArtifactValidationResult:
    corridors_dir = output_dir / "corridors"
    required = {
        "summary": corridors_dir / "corridor_summary.json",
        "fingerprints": corridors_dir / "corridor_fingerprints.json",
        "modes": corridors_dir / "corridor_modes.json",
        "assignments": corridors_dir / "mode_assignments.json",
    }
    blockers: list[str] = []
    warnings: list[str] = []
    missing = [label for label, path in required.items() if not path.is_file()]
    if missing:
        if "modes" in missing:
            blockers.append(
                "corridor_exemplar_v1 selected-only run did not emit corridor modes"
            )
        blockers.extend(
            f"corridor artifact missing {label}: {path.name}"
            for label, path in required.items()
            if label in missing
        )
        return CorridorArtifactValidationResult(
            blockers=tuple(blockers),
            warnings=tuple(warnings),
        )

    try:
        summary = read_json_object(required["summary"])
        fingerprints = read_json_object(required["fingerprints"])
        modes = read_json_object(required["modes"])
        assignments = read_json_object(required["assignments"])
    except (OSError, ValueError) as exc:
        return CorridorArtifactValidationResult(
            blockers=(f"corridor artifact invalid JSON: {exc}",),
            warnings=tuple(warnings),
        )

    fingerprint_count = int(summary.get("fingerprint_count") or 0)
    mode_count = int(summary.get("mode_count") or 0)
    if summary.get("corridor_artifact_built") is not True:
        blockers.append("corridor_summary.corridor_artifact_built is not true")
    if summary.get("corridor_modes_built") is not True:
        blockers.append("corridor_summary.corridor_modes_built is not true")
    if fingerprint_count < 1:
        blockers.append("corridor_summary.fingerprint_count must be >= 1")
    if mode_count < 1:
        blockers.append("corridor_summary.mode_count must be >= 1")
    if fingerprints.get("fingerprint_count") != fingerprint_count:
        blockers.append("corridor_fingerprints fingerprint_count mismatch")
    if modes.get("mode_count") != mode_count:
        blockers.append("corridor_modes mode_count mismatch")
    if (
        expected_selected_count is not None
        and int(summary.get("selected_exemplar_count") or -1) != expected_selected_count
    ):
        blockers.append("corridor_summary selected_exemplar_count mismatch")
    _validate_selected_links(selected_records or (), blockers, source="selected")
    _validate_selected_links(selected_payloads or (), blockers, source="payload")
    assignment_items = assignments.get("assignments", [])
    if not isinstance(assignment_items, list):
        blockers.append("mode_assignments.assignments must be a list")
    elif selected_records and not any(
        item.get("source") == "selected_exemplar"
        for item in assignment_items
        if isinstance(item, dict)
    ):
        blockers.append("mode_assignments missing selected exemplar assignments")
    corridor_artifact_ok = not blockers
    return CorridorArtifactValidationResult(
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        corridor_artifact_ok=corridor_artifact_ok,
        corridor_fingerprints_ok=fingerprint_count >= 1 and not blockers,
        corridor_modes_ok=mode_count >= 1 and not blockers,
        corridor_mode_count=mode_count,
        corridor_fingerprint_count=fingerprint_count,
    )


def _score_observations(
    store: TeacherTargetStore,
    examples: tuple[TinyTextExample, ...],
) -> list[_Observation]:
    observations: list[_Observation] = []
    example_offset = 0
    for shard_id in range(store.metadata.shard_count):
        arrays = store.read_shard(shard_id)
        row_count = int(np.asarray(arrays["input_ids"]).shape[0])
        for row in range(row_count):
            example_index = example_offset + row
            example_id = (
                examples[example_index].example_id
                if example_index < len(examples)
                else f"row-{example_index:06d}"
            )
            observations.append(_observation_from_row(arrays, row, example_id))
        example_offset += row_count
    return observations


def _validate_selected_links(
    items: Any,
    blockers: list[str],
    *,
    source: str,
) -> None:
    if not isinstance(items, (list, tuple)):
        return
    for item in items:
        if not isinstance(item, dict):
            continue
        if "corridor_mode_id" not in item or "corridor_fingerprint_id" not in item:
            blockers.append(f"{source} selected exemplar missing corridor linkage")
            return
        if item.get("corridor_assignment_status") != "linked":
            blockers.append(f"{source} selected exemplar corridor assignment missing")
            return


def _observation_from_row(
    arrays: dict[str, np.ndarray],
    row: int,
    example_id: str,
) -> _Observation:
    position = int(np.asarray(arrays["score_selected_position"])[row])
    length = int(np.asarray(arrays["score_lengths"])[row])
    if "score_top_token_id" in arrays:
        top_token_id = int(np.asarray(arrays["score_top_token_id"])[row])
    elif "corridor_top_token_ids" in arrays:
        top_token_id = int(np.asarray(arrays["corridor_top_token_ids"])[row, position])
    else:
        top_token_id = int(np.asarray(arrays["input_ids"])[row, position])
    return _Observation(
        example_id=example_id,
        position=position,
        top_token_id=top_token_id,
        entropy=float(np.asarray(arrays["score_selected_position_entropy"])[row]),
        confidence=float(
            np.asarray(arrays["score_confidence_at_selected_position"])[row]
        ),
        length=length,
        source_policy_id=int(np.asarray(arrays["score_source_policy_ids"])[row]),
    )


def _fingerprints(
    observations: list[_Observation],
    *,
    target_type: str,
    source_target_type: str,
    num_examples_scored: int,
    num_positions_scored: int,
    fingerprint_policy: str,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], str]]:
    groups: dict[tuple[int, str, str, str], list[_Observation]] = {}
    for observation in observations:
        signature = (
            observation.top_token_id,
            _entropy_bucket(observation.entropy),
            _confidence_bucket(observation.confidence),
            _position_bucket(observation.position, observation.length),
        )
        groups.setdefault(signature, []).append(observation)
    fingerprints: list[dict[str, Any]] = []
    observation_to_fingerprint: dict[tuple[str, int], str] = {}
    sorted_groups = sorted(
        groups.items(),
        key=lambda item: (
            -len(item[1]),
            item[0][0],
            item[0][1],
            item[0][2],
            item[0][3],
        ),
    )
    for index, (signature, members) in enumerate(sorted_groups):
        fingerprint_id = f"fp_{index:06d}"
        for member in members:
            observation_to_fingerprint[member.key] = fingerprint_id
        entropies = [member.entropy for member in members]
        confidences = [member.confidence for member in members]
        representatives = sorted(
            members,
            key=lambda item: (-item.entropy, item.example_id, item.position),
        )[:3]
        fingerprints.append(
            {
                "fingerprint_id": fingerprint_id,
                "signature": {
                    "top_token_ids": [signature[0]],
                    "top_token_ranks": [0],
                    "bucket": "_".join(signature[1:]),
                    "entropy_bucket": signature[1],
                    "confidence_bucket": signature[2],
                    "position_bucket": signature[3],
                },
                "count": len(members),
                "mean_entropy": float(np.mean(entropies, dtype=np.float32)),
                "max_entropy": float(np.max(entropies)),
                "mean_confidence": float(np.mean(confidences, dtype=np.float32)),
                "positions_observed": len(members),
                "example_count": len({member.example_id for member in members}),
                "representatives": [
                    {
                        "example_id": member.example_id,
                        "position": member.position,
                        "entropy": member.entropy,
                        "confidence": member.confidence,
                    }
                    for member in representatives
                ],
                "target_type": target_type,
                "source_target_type": source_target_type,
                "num_examples_scored": num_examples_scored,
                "num_positions_scored": num_positions_scored,
                "fingerprint_policy": fingerprint_policy,
            }
        )
    return fingerprints, observation_to_fingerprint


def _modes(
    fingerprints: list[dict[str, Any]],
    *,
    mode_policy: str,
    observation_count: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    modes: list[dict[str, Any]] = []
    fingerprint_to_mode: dict[str, str] = {}
    denominator = max(observation_count, 1)
    for index, fingerprint in enumerate(fingerprints):
        mode_id = f"mode_{index:06d}"
        fingerprint_id = str(fingerprint["fingerprint_id"])
        signature = fingerprint["signature"]
        top_token_ids = list(signature["top_token_ids"])
        label = (
            f"token_{top_token_ids[0]}_{signature['entropy_bucket']}_entropy_"
            f"{signature['confidence_bucket']}_confidence"
        )
        fingerprint_to_mode[fingerprint_id] = mode_id
        modes.append(
            {
                "mode_id": mode_id,
                "fingerprint_ids": [fingerprint_id],
                "count": fingerprint["count"],
                "share": float(fingerprint["count"]) / float(denominator),
                "label": label,
                "top_token_ids": top_token_ids,
                "mean_entropy": fingerprint["mean_entropy"],
                "mean_confidence": fingerprint["mean_confidence"],
                "representative_examples": [
                    {
                        "example_id": item["example_id"],
                        "position": item["position"],
                    }
                    for item in fingerprint["representatives"]
                ],
                "mode_policy": mode_policy,
            }
        )
    return modes, fingerprint_to_mode


def _link_selected_exemplars(
    selected_records: list[dict[str, Any]],
    selected_payloads: list[dict[str, Any]],
    *,
    observation_to_fingerprint: dict[tuple[str, int], str],
    fingerprint_to_mode: dict[str, str],
) -> int:
    linked = 0
    for collection in (selected_records, selected_payloads):
        for item in collection:
            key = (
                str(item.get("selected_example_id")),
                int(item.get("selected_position", -1)),
            )
            fingerprint_id = observation_to_fingerprint.get(key)
            mode_id = (
                fingerprint_to_mode.get(fingerprint_id)
                if fingerprint_id is not None
                else None
            )
            item["corridor_fingerprint_id"] = fingerprint_id
            item["corridor_mode_id"] = mode_id
            is_linked = fingerprint_id is not None and mode_id is not None
            item["corridor_assignment_status"] = "linked" if is_linked else "missing"
            if collection is selected_records and is_linked:
                linked += 1
    return linked


def _mode_assignments(
    selected_records: list[dict[str, Any]],
    *,
    fingerprints: list[dict[str, Any]],
    fingerprint_to_mode: dict[str, str],
) -> list[dict[str, Any]]:
    assignments: dict[tuple[str, int, str], dict[str, Any]] = {}
    for item in selected_records:
        fingerprint_id = item.get("corridor_fingerprint_id")
        mode_id = item.get("corridor_mode_id")
        if fingerprint_id is None or mode_id is None:
            continue
        key = (
            str(item["selected_example_id"]),
            int(item["selected_position"]),
            "selected_exemplar",
        )
        assignments[key] = {
            "example_id": item["selected_example_id"],
            "position": item["selected_position"],
            "mode_id": mode_id,
            "fingerprint_id": fingerprint_id,
            "source": "selected_exemplar",
        }
    for fingerprint in fingerprints:
        fingerprint_id = str(fingerprint["fingerprint_id"])
        mode_id = fingerprint_to_mode[fingerprint_id]
        for representative in fingerprint["representatives"][:1]:
            key = (
                str(representative["example_id"]),
                int(representative["position"]),
                "representative",
            )
            assignments.setdefault(
                key,
                {
                    "example_id": representative["example_id"],
                    "position": representative["position"],
                    "mode_id": mode_id,
                    "fingerprint_id": fingerprint_id,
                    "source": "representative",
                },
            )
    return [
        assignments[key]
        for key in sorted(assignments, key=lambda item: (item[0], item[1], item[2]))
    ]


def _entropy_bucket(value: float) -> str:
    if value < 1.0:
        return "low"
    if value < 3.0:
        return "mid"
    if value < 6.0:
        return "high"
    return "very_high"


def _confidence_bucket(value: float) -> str:
    if value >= 0.75:
        return "high"
    if value >= 0.35:
        return "mid"
    if value >= 0.10:
        return "low"
    return "very_low"


def _position_bucket(position: int, length: int) -> str:
    if length <= 1 or position >= length - 1:
        return "final"
    ratio = float(position) / float(max(length - 1, 1))
    if ratio < 0.33:
        return "early"
    if ratio < 0.66:
        return "middle"
    return "late"


def _human_summary(
    *,
    num_examples: int,
    num_positions: int,
    mode_count: int,
    fingerprint_count: int,
    selected_count: int,
    selected_payload_retained: bool,
    non_selected_payload_retained: bool,
    delivery_path: str,
) -> str:
    return "\n".join(
        (
            "This Tome artifact contains:",
            f"  - {num_examples:,} scored examples",
            f"  - {num_positions:,} scored token positions",
            f"  - {mode_count:,} discovered corridor modes",
            f"  - {fingerprint_count:,} corridor fingerprints",
            f"  - {selected_count:,} selected exemplars",
            "  - selected exemplar payloads retained: "
            f"{'yes' if selected_payload_retained else 'no'}",
            "  - non-selected exemplar payloads retained: "
            f"{'yes' if non_selected_payload_retained else 'no'}",
            f"  - delivery path: {delivery_path}",
            "",
        )
    )
