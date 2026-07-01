from __future__ import annotations

import subprocess
import sys

import pytest

from radjax_tome.backends import (
    BackendBatchEnvelope,
    BackendRunConfig,
    TeacherBackendConfig,
    TeacherBatchInput,
    create_backend,
    run_backend_batches,
)


def _batch(label: str, count: int = 2) -> TeacherBatchInput:
    return TeacherBatchInput(
        example_ids=tuple(f"{label}-{idx}" for idx in range(count)),
        texts=tuple(f"{label} text {idx}" for idx in range(count)),
    )


def _envelope(sequence_id: int, label: str, count: int = 2) -> BackendBatchEnvelope:
    return BackendBatchEnvelope(sequence_id=sequence_id, batch=_batch(label, count))


def _cpu_backend(target_policy: str = "dense_logits"):
    return create_backend(
        TeacherBackendConfig(
            backend_id="cpu_reference",
            target_policy=target_policy,
            sequence_length=4,
            batch_size=2,
            vocab_size=9,
        )
    )


def test_serial_mode_processes_batches_and_sorts_by_sequence_id() -> None:
    backend = _cpu_backend()
    result = run_backend_batches(
        backend,
        (_envelope(2, "two"), _envelope(1, "one")),
        BackendRunConfig(cpu_orchestration_mode="serial"),
    )

    assert result.requested_cpu_orchestration_mode == "serial"
    assert result.effective_cpu_orchestration_mode == "serial"
    assert [sequence_id for sequence_id, _ in result.results] == [1, 2]
    assert [emission.backend_id for _, emission in result.results] == [
        "cpu_reference",
        "cpu_reference",
    ]
    assert result.metadata["requested_cpu_orchestration_mode"] == "serial"
    assert result.metadata["effective_cpu_orchestration_mode"] == "serial"
    assert result.metadata["batch_count"] == 2
    assert result.metadata["example_count"] == 4
    assert result.metadata["sequence_id_min"] == 1
    assert result.metadata["sequence_id_max"] == 2
    assert result.metadata["input_order_was_sorted"] is False
    assert result.metadata["output_order_sorted"] is True
    assert result.metadata["auto_reason"] == "requested_serial"


def test_staged_mode_processes_batches_and_sorts_by_sequence_id() -> None:
    backend = _cpu_backend("topk_with_tail_v0")
    result = run_backend_batches(
        backend,
        (_envelope(3, "three"), _envelope(1, "one"), _envelope(2, "two")),
        BackendRunConfig(
            cpu_orchestration_mode="staged",
            max_workers=2,
            queue_depth=2,
        ),
    )

    assert result.requested_cpu_orchestration_mode == "staged"
    assert result.effective_cpu_orchestration_mode == "staged"
    assert [sequence_id for sequence_id, _ in result.results] == [1, 2, 3]
    assert result.metadata["effective_cpu_orchestration_mode"] == "staged"
    assert result.metadata["batch_count"] == 3
    assert result.metadata["example_count"] == 6
    assert result.metadata["input_order_was_sorted"] is False
    assert result.metadata["output_order_sorted"] is True
    assert result.metadata["auto_reason"] == "requested_staged"
    assert result.metadata["max_workers"] == 2
    assert result.metadata["queue_depth"] == 2
    assert result.metadata["staged_is_performance_optimized"] is False


def test_auto_resolves_to_serial_for_tiny_batch_count() -> None:
    result = run_backend_batches(
        _cpu_backend(),
        (_envelope(0, "tiny"),),
        BackendRunConfig(
            cpu_orchestration_mode="auto",
            auto_serial_max_batches=1,
            auto_serial_max_examples=0,
        ),
    )

    assert result.effective_cpu_orchestration_mode == "serial"
    assert result.metadata["auto_reason"] == "tiny_batch_count"


def test_auto_resolves_to_serial_for_tiny_example_count() -> None:
    result = run_backend_batches(
        _cpu_backend(),
        (_envelope(0, "a", 2), _envelope(1, "b", 2)),
        BackendRunConfig(
            cpu_orchestration_mode="auto",
            auto_serial_max_batches=1,
            auto_serial_max_examples=4,
        ),
    )

    assert result.effective_cpu_orchestration_mode == "serial"
    assert result.metadata["auto_reason"] == "tiny_example_count"


def test_auto_resolves_to_staged_for_normal_workload() -> None:
    result = run_backend_batches(
        _cpu_backend("cascaded_soft_labels_v1"),
        (_envelope(0, "a", 3), _envelope(1, "b", 3), _envelope(2, "c", 3)),
        BackendRunConfig(
            cpu_orchestration_mode="auto",
            auto_serial_max_batches=1,
            auto_serial_max_examples=4,
        ),
    )

    assert result.requested_cpu_orchestration_mode == "auto"
    assert result.effective_cpu_orchestration_mode == "staged"
    assert result.metadata["auto_reason"] == "normal_workload"
    assert [sequence_id for sequence_id, _ in result.results] == [0, 1, 2]


def test_duplicate_sequence_ids_fail_clearly() -> None:
    with pytest.raises(ValueError, match="duplicate sequence_id"):
        run_backend_batches(
            _cpu_backend(),
            (_envelope(1, "a"), _envelope(1, "b")),
            BackendRunConfig(cpu_orchestration_mode="serial"),
        )


def test_negative_sequence_id_fails_clearly() -> None:
    with pytest.raises(ValueError, match="sequence_id must be >= 0"):
        BackendBatchEnvelope(sequence_id=-1, batch=_batch("bad"))


def test_empty_run_is_allowed_and_records_empty_metadata() -> None:
    result = run_backend_batches(
        _cpu_backend(),
        (),
        BackendRunConfig(cpu_orchestration_mode="auto"),
    )

    assert result.results == ()
    assert result.effective_cpu_orchestration_mode == "serial"
    assert result.metadata["batch_count"] == 0
    assert result.metadata["example_count"] == 0
    assert result.metadata["sequence_id_min"] is None
    assert result.metadata["sequence_id_max"] is None
    assert result.metadata["input_order_was_sorted"] is True
    assert result.metadata["output_order_sorted"] is True
    assert result.metadata["auto_reason"] == "tiny_batch_count"


def test_unsupported_orchestration_mode_fails_clearly() -> None:
    with pytest.raises(ValueError, match="cpu_orchestration_mode"):
        BackendRunConfig(cpu_orchestration_mode="turbo")


def test_runner_works_with_fake_numpy_backend() -> None:
    backend = create_backend(
        TeacherBackendConfig(
            backend_id="fake_numpy",
            target_policy="dense_logits",
            sequence_length=3,
            batch_size=1,
            vocab_size=5,
        )
    )
    result = run_backend_batches(
        backend,
        (BackendBatchEnvelope(sequence_id=0, batch=_batch("fake", 1)),),
        BackendRunConfig(cpu_orchestration_mode="serial"),
    )

    assert len(result.results) == 1
    assert result.results[0][1].backend_id == "fake_numpy"
    assert result.results[0][1].payload["logits"].shape == (1, 3, 5)


def test_backend_orchestration_imports_without_heavy_optional_dependencies() -> None:
    script = """
import importlib
import sys

for name in ("torch", "transformers", "jax"):
    sys.modules.pop(name, None)

importlib.import_module("radjax_tome.backends.orchestration")

loaded = sorted(
    name for name in ("torch", "transformers", "jax") if name in sys.modules
)
print(",".join(loaded))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == ""
