"""M8A readiness checks using a deterministic fake backend, not performance evidence."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from radjax_tome.backends import TeacherBackendConfig
from radjax_tome.builder.delivery import replay, staging
from radjax_tome.builder.delivery.measurement import (
    SelectedPassMeasurementControl,
    deterministic_source_sample,
)
from radjax_tome.builder.exemplar_delivery_contracts import ExemplarDeliveryConfig
from radjax_tome.builder.teacher_textbook import TinyTextExample


def _control(
    checkpoint: replay.ImmutablePostC5Checkpoint, output: Path, *, cap: int = 1
):
    return SelectedPassMeasurementControl(
        benchmark_only=True,
        effective_execution_cap=cap,
        immutable_checkpoint_digest=checkpoint.digest,
        checkpoint_root=checkpoint.root,
        temporary_output_root=output,
    )


def _record(example_id: str, row: int, position: int, token: int) -> dict[str, object]:
    score = 7.5
    return {
        "rank": row + 1,
        "selected_example_id": example_id,
        "selected_position": position,
        "selected_score": score,
        "score_selected_position_entropy": score,
        "score_top_token_id": token,
        "source_shard_id": 0,
        "source_row": row,
        "source_position": position,
        "source_score": score,
        "source_top_token_id": token,
        "source_score_policy": "entropy_top_n_v1",
        "payload_ref": {"kind": "score", "source_row": row},
        "selected_policy": "entropy_top_n_v1",
        "source_delivery_path": "two_pass_rerun_selected",
    }


def _payload(batch_size: int) -> dict[str, np.ndarray]:
    shape = (batch_size, 4, 2)
    return {
        "top_token_ids": np.zeros(shape, dtype=np.int32),
        "top_log_probs": np.zeros(shape, dtype=np.float32),
        "top_probs": np.zeros(shape, dtype=np.float32),
        "top_selection_mask": np.ones(shape, dtype=bool),
        "effective_top_k": np.full((batch_size, 4), 2, dtype=np.int32),
        "top_mass": np.ones((batch_size, 4), dtype=np.float32),
        "tail_mass": np.zeros((batch_size, 4), dtype=np.float32),
        "bucket_masses": np.zeros((batch_size, 4, 1), dtype=np.float32),
        "teacher_entropy": np.zeros((batch_size, 4), dtype=np.float32),
    }


def test_measurement_control_rejects_production_or_unsafe_outputs(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoint"
    checkpoint_root.mkdir()
    (checkpoint_root / "evidence.json").write_text("{}", encoding="utf-8")
    checkpoint = replay.ImmutablePostC5Checkpoint.capture(checkpoint_root)

    with pytest.raises(ValueError, match="benchmark_only=True"):
        SelectedPassMeasurementControl(
            benchmark_only=False,
            effective_execution_cap=1,
            immutable_checkpoint_digest=checkpoint.digest,
            checkpoint_root=checkpoint.root,
            temporary_output_root=tmp_path / "output",
        )
    with pytest.raises(ValueError, match="one of 1, 2, 4, 8"):
        _control(checkpoint, tmp_path / "output", cap=3)
    with pytest.raises(ValueError, match="must not be the checkpoint"):
        _control(checkpoint, checkpoint.root)


def test_checkpoint_copy_is_hash_protected_and_replay_requires_frozen_c5(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoint"
    checkpoint_root.mkdir()
    (checkpoint_root / "score.json").write_text('{"score":"frozen"}', encoding="utf-8")
    checkpoint = replay.ImmutablePostC5Checkpoint.capture(checkpoint_root)
    output = tmp_path / "measurement-output"
    control = _control(checkpoint, output)
    checkpoint.prepare_temporary_output(output)
    assert (output / "score.json").read_text(encoding="utf-8") == '{"score":"frozen"}'

    config = SimpleNamespace(
        artifact_dir=output,
        authoritative_records=None,
        authoritative_selection=False,
    )
    with pytest.raises(ValueError, match="frozen authoritative C5"):
        replay.run_selected_delivery_replay(
            config,
            checkpoint=checkpoint,
            control=control,
        )
    checkpoint.verify_unchanged()


def test_replay_uses_the_canonical_owner_without_score_or_selection_writers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_root = tmp_path / "checkpoint"
    checkpoint_root.mkdir()
    (checkpoint_root / "authority.json").write_text("{}", encoding="utf-8")
    checkpoint = replay.ImmutablePostC5Checkpoint.capture(checkpoint_root)
    output = tmp_path / "measurement-output"
    control = _control(checkpoint, output)
    checkpoint.prepare_temporary_output(output)
    metrics: dict[str, object] = {
        "selected_pass_execution_v1": {"schema_version": "selected_pass_execution_v1"}
    }
    config = SimpleNamespace(
        artifact_dir=output,
        authoritative_records=({},),
        authoritative_selection=True,
        rerun_metrics=metrics,
    )
    calls: list[object] = []

    def canonical_owner(received_config, *, _measurement_control):
        calls.append((received_config, _measurement_control))
        return "canonical-rerun-result"

    import radjax_tome.builder.delivery.rerun as rerun

    monkeypatch.setattr(rerun, "run_selected_delivery_rerun", canonical_owner)
    assert (
        replay.run_selected_delivery_replay(
            config,
            checkpoint=checkpoint,
            control=control,
        )
        == "canonical-rerun-result"
    )
    assert calls == [(config, control)]
    diagnostics = metrics["selected_pass_execution_v1"]
    assert diagnostics["score_pass_invocation_count"] == 0
    assert diagnostics["selection_writer_invocation_count"] == 0
    assert (
        diagnostics["checkpoint_validation"]["included_in_selected_pass_wall_time"]
        is False
    )


def test_private_cap_reuses_canonical_batch_loop_and_preserves_requested_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nonrepresentative fake-backend check: this is not accelerator evidence."""
    checkpoint_root = tmp_path / "checkpoint"
    checkpoint_root.mkdir()
    (checkpoint_root / "c5.json").write_text("{}", encoding="utf-8")
    checkpoint = replay.ImmutablePostC5Checkpoint.capture(checkpoint_root)
    output = tmp_path / "measurement-output"
    control = _control(checkpoint, output, cap=1)
    emitted: list[tuple[str, ...]] = []

    class FakeBackend:
        def emit_batch(self, batch):
            emitted.append(tuple(batch.example_ids))
            payload = _payload(len(batch.example_ids))
            for row, example_id in enumerate(batch.example_ids):
                token = 10 + int(example_id.rsplit("-", 1)[1])
                payload["teacher_entropy"][row, 1] = 7.5
                payload["top_token_ids"][row, 1, 0] = token
            return SimpleNamespace(
                payload=payload,
                input_ids=np.zeros((len(batch.example_ids), 4), dtype=np.int32),
                attention_mask=np.ones((len(batch.example_ids), 4), dtype=np.int32),
                metadata={},
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr(staging, "create_backend", lambda _: FakeBackend())
    records = [_record(f"example-{index}", index, 1, 10 + index) for index in range(3)]
    metrics: dict[str, object] = {}
    config = ExemplarDeliveryConfig(
        artifact_dir=output,
        dataset_path=tmp_path / "unused.jsonl",
        delivery_path="two_pass_rerun_selected",
        selection_enabled=True,
        sequence_length=4,
        vocab_size=16,
        top_k=2,
        num_buckets=1,
        backend_config=TeacherBackendConfig(
            backend_id="cpu_reference",
            target_policy="corridor_exemplar_v1",
            sequence_length=4,
            vocab_size=16,
            top_k=2,
            num_buckets=1,
        ),
        selected_rerun_batch_size=8,
        rerun_metrics=metrics,
    )
    payloads = staging._selected_payloads_from_backend(
        records,
        store=SimpleNamespace(),
        examples=tuple(
            TinyTextExample(example_id=f"example-{index}", text=str(index))
            for index in range(3)
        ),
        config=config,
        _measurement_control=control,
    )

    assert emitted == [("example-0",), ("example-1",), ("example-2",)]
    assert [payload["selected_example_id"] for payload in payloads] == [
        "example-0",
        "example-1",
        "example-2",
    ]
    assert config.selected_rerun_batch_size == 8
    diagnostics = metrics["selected_pass_execution_v1"]
    assert diagnostics["requested_source_batch_size"] == 8
    assert diagnostics["benchmark_only_effective_execution_cap"] == 1
    assert diagnostics["accounting_within_five_percent"] is True
    assert diagnostics["compilation"]["status"] == "not_authorized"


def test_deterministic_sampler_uses_largest_remainder_and_stable_source_order() -> None:
    records = [
        _record(f"example-{source:03d}", source, position, source)
        for source in range(80)
        for position in range(1 + source % 3)
    ]
    first = deterministic_source_sample(records)
    second = deterministic_source_sample(reversed(records))

    assert first == second
    assert len(first["sample_source_keys"]) == 64
    assert first["approximation_only"] is True
    assert sum(first["allocation"].values()) == 64
