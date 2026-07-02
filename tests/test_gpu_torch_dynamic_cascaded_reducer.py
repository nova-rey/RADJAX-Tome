from __future__ import annotations

from importlib import import_module

import numpy as np
import pytest

from radjax_tome.backends.cpu import _dense_logits_to_dynamic_cascaded


def test_gpu_dynamic_cascaded_reducer_matches_cpu_reference_contract() -> None:
    if not _optional_torch_available():
        pytest.skip("optional torch dependency is not installed")
    torch = import_module("torch")
    gpu_torch = import_module("radjax_tome.backends.gpu_torch")
    logits_np = np.asarray(
        [
            [
                [4.0, 1.0, 0.0, -1.0, -2.0, -3.0],
                [0.5, 2.0, -0.5, -1.5, -2.0, -3.0],
                [1.0, 0.0, 3.0, -2.0, -3.0, -4.0],
            ],
            [
                [3.0, 2.0, 1.0, 0.0, -1.0, -2.0],
                [0.0, 4.0, 1.0, -1.0, -2.0, -3.0],
                [1.0, 2.0, 4.0, 0.0, -1.0, -2.0],
            ],
        ],
        dtype=np.float32,
    )
    logits = torch.tensor(logits_np, dtype=torch.float32)

    gpu_payload = gpu_torch._compact_payload_to_numpy(
        gpu_torch._gpu_dynamic_cascaded_reduce(
            torch,
            logits,
            dynamic_top_k_min=2,
            dynamic_top_k_max=4,
            dynamic_mass_threshold=0.75,
            num_buckets=3,
        )
    )
    cpu_payload = _dense_logits_to_dynamic_cascaded(
        logits_np,
        dynamic_top_k_min=2,
        dynamic_top_k_max=4,
        dynamic_mass_threshold=0.75,
        num_buckets=3,
    )

    assert set(gpu_payload) == {
        "top_token_ids",
        "top_log_probs",
        "top_probs",
        "top_selection_mask",
        "effective_top_k",
        "top_mass",
        "tail_mass",
        "bucket_masses",
        "teacher_entropy",
    }
    assert gpu_payload["top_token_ids"].shape == (2, 3, 4)
    assert gpu_payload["top_log_probs"].shape == (2, 3, 4)
    assert gpu_payload["top_probs"].shape == (2, 3, 4)
    assert gpu_payload["top_selection_mask"].shape == (2, 3, 4)
    assert gpu_payload["top_selection_mask"].dtype == np.bool_
    assert gpu_payload["effective_top_k"].shape == (2, 3)
    assert np.issubdtype(gpu_payload["effective_top_k"].dtype, np.integer)
    assert gpu_payload["top_mass"].shape == (2, 3)
    assert gpu_payload["tail_mass"].shape == (2, 3)
    assert gpu_payload["bucket_masses"].shape == (2, 3, 3)
    assert gpu_payload["teacher_entropy"].shape == (2, 3)

    mask = gpu_payload["top_selection_mask"]
    np.testing.assert_array_equal(
        gpu_payload["effective_top_k"],
        np.sum(mask, axis=-1).astype(np.int32),
    )
    assert int(np.min(gpu_payload["effective_top_k"])) >= 2
    assert int(np.max(gpu_payload["effective_top_k"])) <= 4
    assert (gpu_payload["top_token_ids"][~mask] == 0).all()
    assert (gpu_payload["top_probs"][~mask] == 0.0).all()
    assert (gpu_payload["top_log_probs"][~mask] == 0.0).all()
    np.testing.assert_allclose(
        np.sum(gpu_payload["bucket_masses"], axis=-1),
        gpu_payload["tail_mass"],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        gpu_payload["top_mass"] + gpu_payload["tail_mass"],
        1.0,
        rtol=1e-6,
        atol=1e-6,
    )
    assert np.isfinite(gpu_payload["teacher_entropy"]).all()
    not_max_clamped = gpu_payload["effective_top_k"] < 4
    assert (gpu_payload["top_mass"][not_max_clamped] >= 0.75).all()
    below_threshold = gpu_payload["top_mass"] < 0.75
    assert (gpu_payload["effective_top_k"][below_threshold] == 4).all()

    np.testing.assert_array_equal(
        gpu_payload["top_token_ids"],
        cpu_payload["top_token_ids"],
    )
    np.testing.assert_array_equal(
        gpu_payload["top_selection_mask"],
        cpu_payload["top_selection_mask"],
    )
    np.testing.assert_array_equal(
        gpu_payload["effective_top_k"],
        cpu_payload["effective_top_k"],
    )
    for field in (
        "top_log_probs",
        "top_probs",
        "top_mass",
        "tail_mass",
        "bucket_masses",
        "teacher_entropy",
    ):
        np.testing.assert_allclose(
            gpu_payload[field],
            cpu_payload[field],
            rtol=1e-6,
            atol=1e-6,
        )


def test_gpu_dynamic_cascaded_reducer_honors_min_and_max() -> None:
    if not _optional_torch_available():
        pytest.skip("optional torch dependency is not installed")
    torch = import_module("torch")
    gpu_torch = import_module("radjax_tome.backends.gpu_torch")
    logits = torch.tensor(
        [[[5.0, 1.0, 0.0, -1.0], [0.25, 0.0, -0.25, -0.5]]],
        dtype=torch.float32,
    )

    min_payload = gpu_torch._compact_payload_to_numpy(
        gpu_torch._gpu_dynamic_cascaded_reduce(
            torch,
            logits,
            dynamic_top_k_min=3,
            dynamic_top_k_max=4,
            dynamic_mass_threshold=0.01,
            num_buckets=2,
        )
    )
    max_payload = gpu_torch._compact_payload_to_numpy(
        gpu_torch._gpu_dynamic_cascaded_reduce(
            torch,
            logits,
            dynamic_top_k_min=1,
            dynamic_top_k_max=2,
            dynamic_mass_threshold=1.0,
            num_buckets=2,
        )
    )

    assert (min_payload["effective_top_k"] >= 3).all()
    assert (max_payload["effective_top_k"] == 2).all()
    assert max_payload["top_token_ids"].shape == (1, 2, 2)


def _optional_torch_available() -> bool:
    try:
        import_module("torch")
    except ImportError:
        return False
    return True
