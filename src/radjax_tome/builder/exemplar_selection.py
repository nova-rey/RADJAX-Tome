from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from radjax_tome.builder.teacher_textbook import TinyTextExample
from radjax_tome.io.json import write_json
from radjax_tome.targets.store import TeacherTargetStore

EXEMPLAR_SELECTION_MANIFEST_FILENAME = "exemplar_selection_manifest.json"
EXEMPLAR_SELECTION_MANIFEST_SCHEMA = "exemplar_selection_manifest_v1"
MULTI_LEADERBOARD_SELECTOR_POLICY = "multi_leaderboard_exemplar_selector_v1"
PATH_A_FULFILLMENT_POLICY = "select_from_existing_capture"
PATH_B_FULFILLMENT_POLICY = "rerun_selected_capture"
PATH_A_SELECTION_APPLICATION = "retain_selected_candidates"
PATH_B_SELECTION_APPLICATION = "rerun_selected_examples"


@dataclass(frozen=True)
class ExemplarCandidate:
    example_id: str
    source_shard_id: int
    source_row: int
    selected_position: int
    candidate_positions: tuple[int, ...]
    sequence_length: int
    capture_mode: str
    source_policy: str
    score_fields: dict[str, float]
    payload_ref: dict[str, object]

    @property
    def position_key(self) -> tuple[str, int]:
        return (self.example_id, self.selected_position)


@dataclass(frozen=True)
class _BoardEntry:
    score: float
    candidate: ExemplarCandidate


class _Board:
    def __init__(self, board_id: str, score_policy: str, capacity: int) -> None:
        self.board_id = board_id
        self.score_policy = score_policy
        self.capacity = capacity
        self.candidate_count_seen = 0
        self._winners: list[_BoardEntry] = []

    def consider(self, candidate: ExemplarCandidate, score: float | None) -> None:
        if score is None:
            return
        self.candidate_count_seen += 1
        entry = _BoardEntry(score=score, candidate=candidate)
        self._winners.append(entry)
        self._winners.sort(key=self._sort_key)
        del self._winners[self.capacity :]

    @property
    def winners(self) -> tuple[_BoardEntry, ...]:
        return tuple(self._winners)

    def to_manifest(self) -> dict[str, object]:
        cutoff_score = self._winners[-1].score if self._winners else None
        return {
            "board_id": self.board_id,
            "score_policy": self.score_policy,
            "capacity": self.capacity,
            "candidate_count_seen": self.candidate_count_seen,
            "winner_count": len(self._winners),
            "cutoff_score": cutoff_score,
            "winners": [
                {
                    "example_id": entry.candidate.example_id,
                    "selected_position": entry.candidate.selected_position,
                    "score": entry.score,
                }
                for entry in self._winners
            ],
        }

    def _sort_key(self, entry: _BoardEntry) -> tuple[float, str, int, str]:
        return (
            -entry.score,
            entry.candidate.example_id,
            entry.candidate.selected_position,
            self.board_id,
        )


def extract_one_pass_candidates(
    arrays: Mapping[str, np.ndarray],
    *,
    example_ids: tuple[str, ...],
    source_shard_id: int,
    capture_mode: str = "one_pass_candidate",
) -> tuple[ExemplarCandidate, ...]:
    entropy = np.asarray(arrays["corridor_teacher_entropy"])
    confidence = np.asarray(arrays["corridor_confidence"])
    positions = np.asarray(arrays["exemplar_positions"])
    scores = np.asarray(arrays["exemplar_scores"])
    effective_top_k = np.asarray(arrays["exemplar_source_effective_top_k"])
    tail_mass = np.asarray(arrays["exemplar_source_tail_mass"])
    source_policy_ids = np.asarray(arrays["exemplar_source_policy_ids"])
    sequence_length = int(entropy.shape[1])
    candidates: list[ExemplarCandidate] = []
    for row, example_id in enumerate(example_ids):
        row_positions = tuple(int(value) for value in positions[row].tolist())
        for position_index, selected_position in enumerate(row_positions):
            score_fields = {
                "max_entropy": float(np.max(entropy[row])),
                "mean_entropy": float(np.mean(entropy[row], dtype=np.float32)),
                "selected_position_entropy": float(
                    entropy[row, selected_position]
                    if selected_position < sequence_length
                    else scores[row, position_index]
                ),
                "confidence": float(confidence[row, selected_position]),
                "tail_mass": float(tail_mass[row, selected_position]),
                "effective_top_k": float(effective_top_k[row, selected_position]),
                "source_policy_id": float(source_policy_ids[row, selected_position]),
                "position_bucket": float(
                    _position_bucket(selected_position, sequence_length)
                ),
                "length_bucket": float(_length_bucket(sequence_length)),
            }
            candidates.append(
                ExemplarCandidate(
                    example_id=example_id,
                    source_shard_id=source_shard_id,
                    source_row=row,
                    selected_position=int(selected_position),
                    candidate_positions=row_positions,
                    sequence_length=sequence_length,
                    capture_mode=capture_mode,
                    source_policy=str(int(source_policy_ids[row, selected_position])),
                    score_fields=score_fields,
                    payload_ref={
                        "kind": "corridor_exemplar_v1",
                        "source_shard_id": source_shard_id,
                        "source_row": row,
                        "position_index": position_index,
                    },
                )
            )
    return tuple(candidates)


def extract_score_pass_candidates(
    arrays: Mapping[str, np.ndarray],
    *,
    example_ids: tuple[str, ...],
    source_shard_id: int,
    capture_mode: str = "two_pass_sparse_exemplar",
) -> tuple[ExemplarCandidate, ...]:
    selected_positions = np.asarray(arrays["score_selected_position"])
    sequence_lengths = np.asarray(arrays["score_lengths"])
    source_policy_ids = np.asarray(arrays["score_source_policy_ids"])
    candidates: list[ExemplarCandidate] = []
    for row, example_id in enumerate(example_ids):
        selected_position = int(selected_positions[row])
        sequence_length = int(sequence_lengths[row])
        score_fields = {
            "max_entropy": float(np.asarray(arrays["score_max_entropy"])[row]),
            "mean_entropy": float(np.asarray(arrays["score_mean_entropy"])[row]),
            "selected_position_entropy": float(
                np.asarray(arrays["score_selected_position_entropy"])[row]
            ),
            "confidence": float(
                np.asarray(arrays["score_confidence_at_selected_position"])[row]
            ),
            "source_policy_id": float(source_policy_ids[row]),
            "position_bucket": float(
                _position_bucket(selected_position, sequence_length)
            ),
            "length_bucket": float(_length_bucket(sequence_length)),
        }
        candidates.append(
            ExemplarCandidate(
                example_id=example_id,
                source_shard_id=source_shard_id,
                source_row=row,
                selected_position=selected_position,
                candidate_positions=(selected_position,),
                sequence_length=sequence_length,
                capture_mode=capture_mode,
                source_policy=str(int(source_policy_ids[row])),
                score_fields=score_fields,
                payload_ref={
                    "kind": "corridor_exemplar_score_pass_v1",
                    "source_shard_id": source_shard_id,
                    "source_row": row,
                },
            )
        )
    return tuple(candidates)


def select_exemplars(
    candidates: Iterable[ExemplarCandidate],
    *,
    capture_mode: str,
    fulfillment_policy: str,
    board_capacity: int,
    budget_examples: int | None = None,
    budget_fraction: float | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    if board_capacity < 1:
        raise ValueError("exemplar_selection_board_capacity must be positive")
    boards: dict[str, _Board] = {}
    num_candidates_seen = 0
    for candidate in candidates:
        num_candidates_seen += 1
        for board_id, score_policy, score in _scores_for_candidate(candidate):
            board = boards.setdefault(
                board_id,
                _Board(
                    board_id=board_id,
                    score_policy=score_policy,
                    capacity=board_capacity,
                ),
            )
            board.consider(candidate, score)

    selected_by_position: dict[tuple[str, int], dict[str, Any]] = {}
    num_board_winners = 0
    for board in boards.values():
        for entry in board.winners:
            num_board_winners += 1
            _merge_position_winner(selected_by_position, board, entry)

    selected_examples = _selected_examples(
        selected_by_position.values(),
        fulfillment_policy=fulfillment_policy,
        budget_examples=budget_examples,
        budget_fraction=budget_fraction,
    )
    application = _selection_application(fulfillment_policy)
    retention_policy = (
        "retain_all_candidates_debug"
        if fulfillment_policy == PATH_A_FULFILLMENT_POLICY
        else "rerun_requisition_only"
    )
    manifest: dict[str, Any] = {
        "schema_version": EXEMPLAR_SELECTION_MANIFEST_SCHEMA,
        "selection_policy": MULTI_LEADERBOARD_SELECTOR_POLICY,
        "capture_mode": capture_mode,
        "fulfillment_policy": fulfillment_policy,
        "selection_application": application,
        "num_candidates_seen": num_candidates_seen,
        "num_boards": len(boards),
        "total_board_capacity": sum(board.capacity for board in boards.values()),
        "num_board_winners": num_board_winners,
        "num_unique_examples_selected": len(selected_examples),
        "num_unique_positions_selected": sum(
            len(record["selected_positions"]) for record in selected_examples
        ),
        "deduplication_policy": "example_id_plus_selected_position_then_example_union",
        "production_global_selector": False,
        "semantic_diversity_used": False,
        "utility_calibrated": False,
        "retention_policy": retention_policy,
        "created_at": created_at
        or datetime.now(UTC).replace(microsecond=0).isoformat(),
        "boards": [
            board.to_manifest()
            for board in sorted(boards.values(), key=lambda item: item.board_id)
        ],
        "selected_examples": selected_examples,
    }
    validate_exemplar_selection_manifest(manifest)
    return manifest


def build_exemplar_selection_manifest(
    store: TeacherTargetStore,
    *,
    examples: tuple[TinyTextExample, ...],
    batch_size: int,
    capture_mode: str,
    fulfillment_policy: str,
    board_capacity: int,
    budget_examples: int | None,
    budget_fraction: float | None,
    created_at: str,
) -> dict[str, Any]:
    candidates: list[ExemplarCandidate] = []
    for shard_id in range(store.metadata.shard_count):
        shard = store.read_shard(shard_id)
        start = shard_id * batch_size
        row_count = int(np.asarray(shard["input_ids"]).shape[0])
        example_ids = tuple(
            example.example_id for example in examples[start : start + row_count]
        )
        if store.metadata.target_type == "corridor_exemplar_v1":
            candidates.extend(
                extract_one_pass_candidates(
                    shard,
                    example_ids=example_ids,
                    source_shard_id=shard_id,
                    capture_mode=capture_mode,
                )
            )
        elif store.metadata.target_type == "corridor_exemplar_score_pass_v1":
            candidates.extend(
                extract_score_pass_candidates(
                    shard,
                    example_ids=example_ids,
                    source_shard_id=shard_id,
                    capture_mode=capture_mode,
                )
            )
        else:
            raise ValueError(
                "exemplar selection requires corridor_exemplar_v1 or "
                "corridor_exemplar_score_pass_v1 target artifacts"
            )
    return select_exemplars(
        candidates,
        capture_mode=capture_mode,
        fulfillment_policy=fulfillment_policy,
        board_capacity=board_capacity,
        budget_examples=budget_examples,
        budget_fraction=budget_fraction,
        created_at=created_at,
    )


def write_exemplar_selection_manifest(
    output_dir: Path,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / EXEMPLAR_SELECTION_MANIFEST_FILENAME
    write_json(path, manifest)
    return path


def validate_exemplar_selection_manifest(manifest: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "selection_policy",
        "capture_mode",
        "fulfillment_policy",
        "selection_application",
        "num_candidates_seen",
        "num_boards",
        "total_board_capacity",
        "num_board_winners",
        "num_unique_examples_selected",
        "num_unique_positions_selected",
        "deduplication_policy",
        "production_global_selector",
        "semantic_diversity_used",
        "utility_calibrated",
        "created_at",
        "boards",
        "selected_examples",
    }
    missing = required - set(manifest)
    if missing:
        raise ValueError(
            f"exemplar selection manifest missing fields: {sorted(missing)}"
        )
    if manifest["schema_version"] != EXEMPLAR_SELECTION_MANIFEST_SCHEMA:
        raise ValueError("unsupported exemplar selection manifest schema")
    if manifest["selection_policy"] != MULTI_LEADERBOARD_SELECTOR_POLICY:
        raise ValueError("unsupported exemplar selection policy")
    if manifest["production_global_selector"] is not False:
        raise ValueError("manifest must not claim production global selection")
    if manifest["semantic_diversity_used"] is not False:
        raise ValueError("manifest must not claim semantic diversity")
    if manifest["utility_calibrated"] is not False:
        raise ValueError("manifest must not claim utility calibration")
    if not isinstance(manifest["boards"], list):
        raise ValueError("manifest boards must be a list")
    if not isinstance(manifest["selected_examples"], list):
        raise ValueError("manifest selected_examples must be a list")


def _scores_for_candidate(
    candidate: ExemplarCandidate,
) -> tuple[tuple[str, str, float | None], ...]:
    position_bucket = int(
        candidate.score_fields.get(
            "position_bucket",
            _position_bucket(candidate.selected_position, candidate.sequence_length),
        )
    )
    length_bucket = int(
        candidate.score_fields.get(
            "length_bucket",
            _length_bucket(candidate.sequence_length),
        )
    )
    return (
        (
            "global_max_entropy",
            "higher max_entropy wins",
            _field(candidate, "max_entropy"),
        ),
        (
            "global_mean_entropy",
            "higher mean_entropy wins",
            _field(candidate, "mean_entropy"),
        ),
        (
            "low_confidence",
            "lower confidence wins as 1 - confidence",
            _inverse_confidence(candidate),
        ),
        ("high_tail_mass", "higher tail_mass wins", _field(candidate, "tail_mass")),
        (
            "high_effective_top_k",
            "higher effective_top_k wins",
            _field(candidate, "effective_top_k"),
        ),
        (
            f"position_bucket_entropy:{position_bucket}",
            "higher selected_position_entropy wins within position bucket",
            _field(candidate, "selected_position_entropy"),
        ),
        (
            f"length_bucket_entropy:{length_bucket}",
            "higher selected_position_entropy wins within length bucket",
            _field(candidate, "selected_position_entropy"),
        ),
        (
            f"shard_coverage:{candidate.source_shard_id}",
            "higher selected_position_entropy wins within source shard",
            _field(candidate, "selected_position_entropy"),
        ),
    )


def _field(candidate: ExemplarCandidate, key: str) -> float | None:
    value = candidate.score_fields.get(key)
    return None if value is None else float(value)


def _inverse_confidence(candidate: ExemplarCandidate) -> float | None:
    confidence = _field(candidate, "confidence")
    return None if confidence is None else 1.0 - confidence


def _merge_position_winner(
    selected: dict[tuple[str, int], dict[str, Any]],
    board: _Board,
    entry: _BoardEntry,
) -> None:
    candidate = entry.candidate
    record = selected.setdefault(
        candidate.position_key,
        {
            "candidate": candidate,
            "winning_boards": [],
            "scores_by_board": {},
            "selection_reasons": [],
        },
    )
    if board.board_id not in record["winning_boards"]:
        record["winning_boards"].append(board.board_id)
        record["selection_reasons"].append(board.score_policy)
    record["scores_by_board"][board.board_id] = entry.score


def _selected_examples(
    position_records: Iterable[Mapping[str, Any]],
    *,
    fulfillment_policy: str,
    budget_examples: int | None,
    budget_fraction: float | None,
) -> list[dict[str, Any]]:
    examples: dict[str, dict[str, Any]] = {}
    for position_record in position_records:
        candidate = position_record["candidate"]
        record = examples.setdefault(
            candidate.example_id,
            {
                "example_id": candidate.example_id,
                "source_shard_id": candidate.source_shard_id,
                "source_row": candidate.source_row,
                "selected_positions": [],
                "winning_boards": [],
                "scores_by_board": {},
                "selection_reasons": [],
                "payload_ref": candidate.payload_ref,
                "rerun_for_pass_2": fulfillment_policy == PATH_B_FULFILLMENT_POLICY,
                "retain_from_existing_capture": (
                    fulfillment_policy == PATH_A_FULFILLMENT_POLICY
                ),
            },
        )
        record["selected_positions"].append(candidate.selected_position)
        for board_id in position_record["winning_boards"]:
            if board_id not in record["winning_boards"]:
                record["winning_boards"].append(board_id)
        record["scores_by_board"].update(position_record["scores_by_board"])
        for reason in position_record["selection_reasons"]:
            if reason not in record["selection_reasons"]:
                record["selection_reasons"].append(reason)

    selected = list(examples.values())
    for record in selected:
        record["selected_positions"] = sorted(set(record["selected_positions"]))
        record["winning_boards"] = sorted(record["winning_boards"])
        record["selection_reasons"] = sorted(record["selection_reasons"])
        record["scores_by_board"] = {
            key: record["scores_by_board"][key]
            for key in sorted(record["scores_by_board"])
        }
    selected.sort(key=lambda item: (item["example_id"], item["source_shard_id"]))
    limit = _selection_limit(
        selected_count=len(selected),
        budget_examples=budget_examples,
        budget_fraction=budget_fraction,
    )
    return selected[:limit]


def _selection_limit(
    *,
    selected_count: int,
    budget_examples: int | None,
    budget_fraction: float | None,
) -> int:
    limit = selected_count
    if budget_examples is not None:
        if budget_examples < 1:
            raise ValueError("exemplar_selection_budget_examples must be positive")
        limit = min(limit, budget_examples)
    if budget_fraction is not None:
        if not 0 < budget_fraction <= 1:
            raise ValueError("exemplar_selection_budget_fraction must be in (0, 1]")
        limit = min(limit, max(1, int(np.ceil(selected_count * budget_fraction))))
    return limit


def _selection_application(fulfillment_policy: str) -> str:
    if fulfillment_policy == PATH_A_FULFILLMENT_POLICY:
        return PATH_A_SELECTION_APPLICATION
    if fulfillment_policy == PATH_B_FULFILLMENT_POLICY:
        return PATH_B_SELECTION_APPLICATION
    raise ValueError(f"unsupported exemplar fulfillment policy: {fulfillment_policy}")


def _position_bucket(position: int, sequence_length: int) -> int:
    if sequence_length <= 0:
        return 0
    return min(3, int((position / sequence_length) * 4))


def _length_bucket(sequence_length: int) -> int:
    if sequence_length <= 16:
        return 0
    if sequence_length <= 64:
        return 1
    if sequence_length <= 256:
        return 2
    return 3
