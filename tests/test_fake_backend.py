import numpy as np

from radjax_tome.backends import FakeTeacherBackend


def test_fake_backend_emits_deterministic_toy_logits() -> None:
    backend = FakeTeacherBackend(vocab_size=5)
    input_ids = np.asarray([[1, 2, 3]], dtype=np.int32)

    first = backend.emit_logits(input_ids)
    second = backend.emit_logits(input_ids)

    assert first.shape == (1, 3, 5)
    np.testing.assert_array_equal(first, second)
