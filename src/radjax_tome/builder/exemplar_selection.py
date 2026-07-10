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
RANK_AWARE_DEDUPLICATION_POLICY = "rank_aware_board_assignment_with_backfill_v1"
SCORE_AWARE_BUDGET_TRIMMING_POLICY = "score_aware_assigned_board_rank_v1"
RUNNER_UP_POOL_MULTIPLIER = 4
_BOARD_PRIORITY_PREFIXES = (
    "global_max_entropy",
    "low_confidence",
    "high_tail_mass",
    "high_effective_top_k",
    "global_mean_entropy",
    "position_bucket_entropy",
    "length_bucket_entropy",
    "shard_coverage",
)


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
    def __init__(
        self,
        board_id: str,
        score_policy: str,
        capacity: int,
        runner_up_pool_multiplier: int,
    ) -> None:
        self.board_id = board_id
        self.score_policy = score_policy
        self.capacity = capacity
        self.runner_up_pool_multiplier = runner_up_pool_multiplier
        self.pool_capacity = capacity * runner_up_pool_multiplier
        self.candidate_count_seen = 0
        self._pool: list[_BoardEntry] = []
        self.assigned_winners: list[_BoardEntry] = []
        self.backfill_count = 0
        self.duplicate_suppression_count = 0

    def consider(self, candidate: ExemplarCandidate, score: float | None) -> None:
        if score is None:
            return
        self.candidate_count_seen += 1
        entry = _BoardEntry(score=score, candidate=candidate)
        self._pool.append(entry)
        self._pool.sort(key=self._sort_key)
        del self._pool[self.pool_capacity :]

    @property
    def pool(self) -> tuple[_BoardEntry, ...]:
        return tuple(self._pool)

    def to_manifest(self) -> dict[str, object]:
        cutoff_score = (
            self.assigned_winners[-1].score if self.assigned_winners else None
        )
        assigned_identities = {
            entry.candidate.position_key for entry in self.assigned_winners
        }
        return {
            "board_id": self.board_id,
            "score_policy": self.score_policy,
            "capacity": self.capacity,
            "pool_capacity": self.pool_capacity,
            "candidate_count_seen": self.candidate_count_seen,
            "winner_count": len(self.assigned_winners),
            "assigned_winner_count": len(self.assigned_winners),
            "runner_up_count": len(self._pool) - len(assigned_identities),
            "cutoff_score": cutoff_score,
            "backfill_count": self.backfill_count,
            "duplicate_suppression_count": self.duplicate_suppression_count,
            "winners": [
                {
                    "example_id": entry.candidate.example_id,
                    "selected_position": entry.candidate.selected_position,
                    "score": entry.score,
                }
                for entry in self.assigned_winners
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
    canonical_score_fields_only: bool = False,
) -> tuple[ExemplarCandidate, ...]:
    entropy = np.asarray(arrays["corridor_teacher_entropy"])
    confidence = np.asarray(arrays["corridor_confidence"])
    corridor_top_token_ids = np.asarray(arrays["corridor_top_token_ids"])
    exemplar_source_top_token_ids = np.asarray(arrays["exemplar_source_top_token_ids"])
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
            source_top_token_id = _one_pass_source_top_token_id(
                exemplar_source_top_token_ids,
                row=row,
                selected_position=selected_position,
                candidate_rank=position_index,
                sequence_length=sequence_length,
            )
            score_fields = {
                "max_entropy": float(np.max(entropy[row])),
                "mean_entropy": float(np.mean(entropy[row], dtype=np.float32)),
                "selected_position_entropy": float(
                    entropy[row, selected_position]
                    if selected_position < sequence_length
                    else scores[row, position_index]
                ),
                "confidence": float(confidence[row, selected_position]),
                "score_top_token_id": float(
                    corridor_top_token_ids[row, selected_position]
                ),
                "source_policy_id": float(source_policy_ids[row, selected_position]),
                "position_bucket": float(
                    _position_bucket(selected_position, sequence_length)
                ),
                "length_bucket": float(_length_bucket(sequence_length)),
            }
            if not canonical_score_fields_only:
                score_fields.update(
                    {
                        "tail_mass": float(tail_mass[row, selected_position]),
                        "effective_top_k": float(
                            effective_top_k[row, selected_position]
                        ),
                    }
                )
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
                        "kind": "one_pass_candidate_v1",
                        "source_shard_id": source_shard_id,
                        "source_row": row,
                        "source_position": int(selected_position),
                        "candidate_rank": position_index,
                        "position_index": position_index,
                        "source_top_token_id": source_top_token_id,
                        "source_score": float(
                            entropy[row, selected_position]
                            if selected_position < sequence_length
                            else scores[row, position_index]
                        ),
                    },
                )
            )
    return tuple(candidates)


def _one_pass_source_top_token_id(
    exemplar_source_top_token_ids: np.ndarray,
    *,
    row: int,
    selected_position: int,
    candidate_rank: int,
    sequence_length: int,
) -> int:
    if (
        exemplar_source_top_token_ids.ndim >= 3
        and int(exemplar_source_top_token_ids.shape[1]) == sequence_length
    ):
        return int(exemplar_source_top_token_ids[row, selected_position, 0])
    return int(exemplar_source_top_token_ids[row, candidate_rank, 0])


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
    selected_top_token_ids = np.asarray(arrays["score_top_token_id"])
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
            "score_top_token_id": float(selected_top_token_ids[row]),
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
                    "source_position": selected_position,
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
    _validate_budget_request(
        budget_examples=budget_examples,
        budget_fraction=budget_fraction,
    )
    boards: dict[str, _Board] = {}
    num_candidates_seen = 0
    for candidate in candidates:
        num_candidates_seen += 1
        for board_id, score_policy, score in _scores_for_candidate(candidate):
            if score is None:
                continue
            board = boards.setdefault(
                board_id,
                _Board(
                    board_id=board_id,
                    score_policy=score_policy,
                    capacity=board_capacity,
                    runner_up_pool_multiplier=RUNNER_UP_POOL_MULTIPLIER,
                ),
            )
            board.consider(candidate, score)

    assignment = _assign_rank_aware_board_winners(boards)
    position_records = _position_records_from_assignment(boards)
    budget_result = _apply_score_aware_budget(
        position_records,
        fulfillment_policy=fulfillment_policy,
        budget_examples=budget_examples,
        budget_fraction=budget_fraction,
    )
    selected_examples = budget_result["selected_examples"]

    num_board_winners = sum(len(board.assigned_winners) for board in boards.values())
    duplicate_candidate_count = _duplicate_candidate_count(boards)
    boards_with_backfill = sorted(
        board_id for board_id, board in boards.items() if board.backfill_count
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
        "deduplication_policy": RANK_AWARE_DEDUPLICATION_POLICY,
        "duplicate_candidate_count": duplicate_candidate_count,
        "backfill_attempt_count": assignment["backfill_attempt_count"],
        "backfill_success_count": assignment["backfill_success_count"],
        "boards_with_backfill": boards_with_backfill,
        "runner_up_pool_multiplier": RUNNER_UP_POOL_MULTIPLIER,
        "score_aware_budget_trimming": True,
        "budget_trimming_policy": SCORE_AWARE_BUDGET_TRIMMING_POLICY,
        "budget_requested_examples": budget_examples,
        "budget_requested_fraction": budget_fraction,
        "budget_applied": budget_result["budget_applied"],
        "budget_trimmed_example_count": budget_result["trimmed_example_count"],
        "budget_trimmed_position_count": budget_result["trimmed_position_count"],
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
    canonical_score_fields_only: bool = False,
    use_score_pass_fields: bool = False,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    candidates: list[ExemplarCandidate] = []
    example_offset = 0
    for shard_id in range(store.metadata.shard_count):
        shard = store.read_shard(shard_id)
        row_count = int(np.asarray(shard["input_ids"]).shape[0])
        example_ids = tuple(
            example.example_id
            for example in examples[example_offset : example_offset + row_count]
        )
        example_offset += row_count
        if store.metadata.target_type == "corridor_exemplar_v1":
            if use_score_pass_fields:
                candidates.extend(
                    extract_score_pass_candidates(
                        shard,
                        example_ids=example_ids,
                        source_shard_id=shard_id,
                        capture_mode=capture_mode,
                    )
                )
            else:
                candidates.extend(
                    extract_one_pass_candidates(
                        shard,
                        example_ids=example_ids,
                        source_shard_id=shard_id,
                        capture_mode=capture_mode,
                        canonical_score_fields_only=canonical_score_fields_only,
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
        "duplicate_candidate_count",
        "backfill_attempt_count",
        "backfill_success_count",
        "boards_with_backfill",
        "runner_up_pool_multiplier",
        "score_aware_budget_trimming",
        "budget_trimming_policy",
        "budget_requested_examples",
        "budget_requested_fraction",
        "budget_applied",
        "budget_trimmed_example_count",
        "budget_trimmed_position_count",
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
    if manifest["deduplication_policy"] != RANK_AWARE_DEDUPLICATION_POLICY:
        raise ValueError("unsupported exemplar deduplication policy")
    if manifest["budget_trimming_policy"] != SCORE_AWARE_BUDGET_TRIMMING_POLICY:
        raise ValueError("unsupported exemplar budget trimming policy")
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


def _assign_rank_aware_board_winners(
    boards: Mapping[str, _Board],
) -> dict[str, int]:
    for board in boards.values():
        board.assigned_winners = list(board.pool[: board.capacity])
        board.backfill_count = 0
        board.duplicate_suppression_count = 0

    backfill_attempt_count = 0
    backfill_success_count = 0
    changed = True
    while changed:
        changed = False
        assigned_by_identity: dict[tuple[str, int], list[str]] = {}
        for board_id, board in boards.items():
            for entry in board.assigned_winners:
                assigned_by_identity.setdefault(
                    entry.candidate.position_key, []
                ).append(board_id)
        for identity, board_ids in sorted(assigned_by_identity.items()):
            if len(board_ids) < 2:
                continue
            strongest = _strongest_board_for_identity(identity, board_ids, boards)
            for board_id in board_ids:
                if board_id == strongest:
                    continue
                board = boards[board_id]
                before = len(board.assigned_winners)
                board.assigned_winners = [
                    entry
                    for entry in board.assigned_winners
                    if entry.candidate.position_key != identity
                ]
                if len(board.assigned_winners) != before:
                    board.duplicate_suppression_count += 1
                    changed = True

        assigned_identities = _assigned_identities(boards)
        for board_id in sorted(boards, key=_board_priority):
            board = boards[board_id]
            while len(board.assigned_winners) < board.capacity:
                backfill_attempt_count += 1
                replacement = _next_backfill_entry(board, assigned_identities)
                if replacement is None:
                    break
                board.assigned_winners.append(replacement)
                board.assigned_winners.sort(key=board._sort_key)
                board.backfill_count += 1
                backfill_success_count += 1
                assigned_identities.add(replacement.candidate.position_key)
                changed = True

    return {
        "backfill_attempt_count": backfill_attempt_count,
        "backfill_success_count": backfill_success_count,
    }


def _strongest_board_for_identity(
    identity: tuple[str, int],
    board_ids: Iterable[str],
    boards: Mapping[str, _Board],
) -> str:
    return min(
        board_ids,
        key=lambda board_id: _identity_board_strength(identity, board_id, boards),
    )


def _identity_board_strength(
    identity: tuple[str, int],
    board_id: str,
    boards: Mapping[str, _Board],
) -> tuple[int, float, int, str, int]:
    rank, entry = _entry_rank_for_identity(boards[board_id], identity)
    return (
        rank,
        -entry.score,
        _board_priority(board_id),
        identity[0],
        identity[1],
    )


def _entry_rank_for_identity(
    board: _Board,
    identity: tuple[str, int],
) -> tuple[int, _BoardEntry]:
    for index, entry in enumerate(board.pool, start=1):
        if entry.candidate.position_key == identity:
            return index, entry
    raise ValueError(f"candidate identity missing from board pool: {identity}")


def _next_backfill_entry(
    board: _Board,
    assigned_identities: set[tuple[str, int]],
) -> _BoardEntry | None:
    local_identities = {
        entry.candidate.position_key for entry in board.assigned_winners
    }
    for entry in board.pool:
        identity = entry.candidate.position_key
        if identity in local_identities:
            continue
        if identity in assigned_identities:
            continue
        return entry
    return None


def _assigned_identities(boards: Mapping[str, _Board]) -> set[tuple[str, int]]:
    return {
        entry.candidate.position_key
        for board in boards.values()
        for entry in board.assigned_winners
    }


def _position_records_from_assignment(
    boards: Mapping[str, _Board],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for board_id in sorted(boards, key=_board_priority):
        board = boards[board_id]
        for entry in board.assigned_winners:
            candidate = entry.candidate
            context = _candidate_board_context(candidate.position_key, boards)
            records.append(
                {
                    "candidate": candidate,
                    "selected_position": candidate.selected_position,
                    "assigned_board": board_id,
                    "winning_boards": context["winning_boards"],
                    "suppressed_duplicate_boards": [
                        value
                        for value in context["winning_boards"]
                        if value != board_id
                    ],
                    "rank_by_board": context["rank_by_board"],
                    "scores_by_board": context["scores_by_board"],
                    "selection_reasons": context["selection_reasons"],
                    "payload_ref": candidate.payload_ref,
                    "sort_key": _position_budget_sort_key(
                        assigned_board=board_id,
                        candidate=candidate,
                        rank_by_board=context["rank_by_board"],
                        scores_by_board=context["scores_by_board"],
                        winning_boards=context["winning_boards"],
                    ),
                }
            )
    records.sort(key=lambda item: item["sort_key"])
    return records


def _candidate_board_context(
    identity: tuple[str, int],
    boards: Mapping[str, _Board],
) -> dict[str, Any]:
    winning_boards: list[str] = []
    rank_by_board: dict[str, int] = {}
    scores_by_board: dict[str, float] = {}
    selection_reasons: dict[str, str] = {}
    for board_id, board in boards.items():
        for rank, entry in enumerate(board.pool, start=1):
            if entry.candidate.position_key != identity:
                continue
            winning_boards.append(board_id)
            rank_by_board[board_id] = rank
            scores_by_board[board_id] = entry.score
            selection_reasons[board_id] = board.score_policy
            break
    return {
        "winning_boards": sorted(winning_boards, key=_board_priority),
        "rank_by_board": {
            key: rank_by_board[key]
            for key in sorted(rank_by_board, key=_board_priority)
        },
        "scores_by_board": {
            key: scores_by_board[key]
            for key in sorted(scores_by_board, key=_board_priority)
        },
        "selection_reasons": [
            selection_reasons[key]
            for key in sorted(selection_reasons, key=_board_priority)
        ],
    }


def _apply_score_aware_budget(
    position_records: list[dict[str, Any]],
    *,
    fulfillment_policy: str,
    budget_examples: int | None,
    budget_fraction: float | None,
) -> dict[str, Any]:
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
                "selected_position_records": [],
                "sort_key": position_record["sort_key"],
                "rerun_for_pass_2": fulfillment_policy == PATH_B_FULFILLMENT_POLICY,
                "retain_from_existing_capture": (
                    fulfillment_policy == PATH_A_FULFILLMENT_POLICY
                ),
            },
        )
        if position_record["sort_key"] < record["sort_key"]:
            record["sort_key"] = position_record["sort_key"]
        record["selected_positions"].append(candidate.selected_position)
        record["selected_position_records"].append(
            _selected_position_manifest_record(position_record)
        )
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
        record["selected_position_records"].sort(
            key=lambda item: (
                item["rank_by_board"].get(item["assigned_board"], 999_999),
                _board_priority(item["assigned_board"]),
                item["selected_position"],
            )
        )
    selected.sort(key=lambda item: item["sort_key"])
    original_examples = len(selected)
    original_positions = sum(len(record["selected_positions"]) for record in selected)
    limit = _selection_limit_from_budget(
        selected_count=len(selected),
        budget_examples=budget_examples,
        budget_fraction=budget_fraction,
    )
    selected = selected[:limit]
    for record in selected:
        del record["sort_key"]
    retained_positions = sum(len(record["selected_positions"]) for record in selected)
    return {
        "selected_examples": selected,
        "budget_applied": limit < original_examples,
        "trimmed_example_count": original_examples - len(selected),
        "trimmed_position_count": original_positions - retained_positions,
    }


def _selected_position_manifest_record(
    position_record: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = position_record["candidate"]
    assigned_board = str(position_record["assigned_board"])
    selected_score = float(
        candidate.score_fields.get(
            "selected_position_entropy",
            position_record["scores_by_board"][assigned_board],
        )
    )
    score_top_token_id = candidate.score_fields.get("score_top_token_id")
    source_top_token_id = candidate.payload_ref.get(
        "source_top_token_id",
        score_top_token_id,
    )
    return {
        "selected_position": candidate.selected_position,
        "selected_score": selected_score,
        "score_selected_position_entropy": selected_score,
        "score_top_token_id": (
            None if score_top_token_id is None else int(score_top_token_id)
        ),
        "source_shard_id": candidate.source_shard_id,
        "source_row": candidate.source_row,
        "source_position": candidate.selected_position,
        "source_score": selected_score,
        "source_top_token_id": (
            None if source_top_token_id is None else int(source_top_token_id)
        ),
        "source_score_policy": "entropy_top_n_v1",
        "assigned_board": assigned_board,
        "winning_boards": position_record["winning_boards"],
        "suppressed_duplicate_boards": position_record["suppressed_duplicate_boards"],
        "rank_by_board": position_record["rank_by_board"],
        "scores_by_board": position_record["scores_by_board"],
        "selection_reasons": position_record["selection_reasons"],
        "payload_ref": position_record["payload_ref"],
    }


def _position_budget_sort_key(
    *,
    assigned_board: str,
    candidate: ExemplarCandidate,
    rank_by_board: Mapping[str, int],
    scores_by_board: Mapping[str, float],
    winning_boards: Iterable[str],
) -> tuple[int, int, float, int, str, int]:
    return (
        int(rank_by_board[assigned_board]),
        _board_priority(assigned_board),
        -float(scores_by_board[assigned_board]),
        -len(tuple(winning_boards)),
        candidate.example_id,
        candidate.selected_position,
    )


def _validate_budget_request(
    *,
    budget_examples: int | None,
    budget_fraction: float | None,
) -> None:
    if budget_examples is not None and budget_examples < 1:
        raise ValueError("exemplar_selection_budget_examples must be positive")
    if budget_fraction is not None and not 0 < budget_fraction <= 1:
        raise ValueError("exemplar_selection_budget_fraction must be in (0, 1]")


def _selection_limit_from_budget(
    *,
    selected_count: int,
    budget_examples: int | None,
    budget_fraction: float | None,
) -> int:
    limit = selected_count
    if budget_examples is not None:
        limit = min(limit, budget_examples)
    if budget_fraction is not None:
        limit = min(limit, max(1, int(np.ceil(selected_count * budget_fraction))))
    return limit


def _duplicate_candidate_count(boards: Mapping[str, _Board]) -> int:
    appearances: dict[tuple[str, int], set[str]] = {}
    for board_id, board in boards.items():
        for entry in board.pool:
            appearances.setdefault(entry.candidate.position_key, set()).add(board_id)
    return sum(1 for board_ids in appearances.values() if len(board_ids) > 1)


def _board_priority(board_id: str) -> int:
    for priority, prefix in enumerate(_BOARD_PRIORITY_PREFIXES):
        if board_id == prefix or board_id.startswith(f"{prefix}:"):
            return priority
    return len(_BOARD_PRIORITY_PREFIXES)


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
