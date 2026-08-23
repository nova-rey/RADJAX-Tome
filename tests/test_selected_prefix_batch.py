from __future__ import annotations

from radjax_tome.backends.base import TeacherBatchInput
from radjax_tome.backends.gpu_torch import _tokenize_selected_prefix_batch


class _Tokenizer:
    padding_side = "right"
    pad_token_id = 0

    def __call__(self, text, *, padding, truncation, max_length, return_tensors):
        assert padding is False
        assert truncation is True
        assert return_tensors is None
        ids = list(range(1, min(len(text), max_length) + 1))
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def pad(self, rows, *, padding, max_length, return_tensors):
        assert padding == "max_length"
        assert return_tensors == "pt"
        width = max_length
        return {
            "input_ids": [
                row["input_ids"] + [0] * (width - len(row["input_ids"])) for row in rows
            ],
            "attention_mask": [
                row["attention_mask"] + [0] * (width - len(row["attention_mask"]))
                for row in rows
            ],
        }


def test_prefix_lengths_are_part_of_selected_batch_contract() -> None:
    batch = TeacherBatchInput(
        example_ids=("a", "b"),
        texts=("abcdef", "xy"),
        selected_positions_by_example=((2,), (0,)),
        selected_prefix_lengths_by_example=(3, 1),
    )
    encoded = _tokenize_selected_prefix_batch(
        _Tokenizer(), batch, max_sequence_length=8
    )
    assert encoded["input_ids"] == [[1, 2, 3], [1, 0, 0]]
    assert encoded["attention_mask"] == [[1, 1, 1], [1, 0, 0]]


def test_prefix_lengths_reject_excess_configured_width() -> None:
    batch = TeacherBatchInput(
        example_ids=("a",),
        texts=("abcdef",),
        selected_positions_by_example=((4,),),
        selected_prefix_lengths_by_example=(5,),
    )
    try:
        _tokenize_selected_prefix_batch(_Tokenizer(), batch, max_sequence_length=4)
    except ValueError as exc:
        assert "exceeds" in str(exc)
    else:
        raise AssertionError("expected prefix width validation")
