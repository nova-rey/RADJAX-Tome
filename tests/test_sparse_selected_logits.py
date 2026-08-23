from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from radjax_tome.backends.base import TeacherBatchInput  # noqa: E402
from radjax_tome.backends.gpu_torch import (  # noqa: E402
    _gpu_selected_position_logits,
    _selected_logits_to_keep,
    _torch_model_forward,
)


def test_union_indices_and_mapping_preserve_coordinate_order() -> None:
    batch = TeacherBatchInput(
        example_ids=("a", "b"),
        texts=("a", "b"),
        selected_positions_by_example=((3, 1), (2,)),
    )
    keep = _selected_logits_to_keep(torch, batch)
    assert keep.tolist() == [1, 2, 3]
    sparse = torch.tensor(
        [[[10, 11], [20, 21], [30, 31]], [[40, 41], [50, 51], [60, 61]]],
        dtype=torch.float32,
    )
    gathered = _gpu_selected_position_logits(torch, sparse, batch, logits_to_keep=keep)
    assert gathered.shape == (1, 3, 2)
    assert gathered.tolist() == [[[30, 31], [10, 11], [50, 51]]]


def test_forward_passes_official_logits_to_keep() -> None:
    seen = {}

    class Model:
        def forward(self, *, input_ids, attention_mask, logits_to_keep=None):
            seen["logits_to_keep"] = logits_to_keep
            return SimpleNamespace(logits=input_ids)

        __call__ = forward

    keep = torch.tensor([1, 4], dtype=torch.int64)
    output = _torch_model_forward(
        torch,
        Model(),
        input_ids=torch.zeros((1, 2, 3)),
        attention_mask=torch.ones((1, 2)),
        logits_to_keep=keep,
    )
    assert output.logits.shape == (1, 2, 3)
    assert seen["logits_to_keep"] is keep
