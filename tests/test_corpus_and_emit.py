import json
from pathlib import Path

import numpy as np
from radjax_contract.io import read_json
from radjax_contract.validation import validate_teacher_tome

from radjax_tome.backends import FakeTeacherBackend
from radjax_tome.corpora import load_jsonl_corpus, split_corpus
from radjax_tome.emit import emit_toy_teacher_tome


def test_toy_corpus_loads_from_jsonl(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "\n".join(
            json.dumps({"example_id": f"e-{index}", "text": f"text {index}"})
            for index in range(3)
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_jsonl_corpus(corpus)

    assert [row["example_id"] for row in rows] == ["e-0", "e-1", "e-2"]


def test_split_writer_emits_three_disjoint_manifests() -> None:
    rows = [{"example_id": f"e-{index}", "text": f"text {index}"} for index in range(6)]

    splits = split_corpus(rows)

    assert set(splits) == {"train", "calibration", "final_test"}
    assert len(splits["train"]) == 2


def test_emitted_toy_tome_validates_using_contract(tmp_path: Path) -> None:
    rows = [{"example_id": "e-0", "text": "hello"}]
    output_dir = emit_toy_teacher_tome(
        output_dir=tmp_path / "toy_tome",
        backend=FakeTeacherBackend(vocab_size=7),
        records=rows,
        sequence_length=3,
    )
    manifest = read_json(output_dir / "manifest.json")
    logits = np.load(output_dir / "logits.npy", allow_pickle=False)

    assert validate_teacher_tome(output_dir).ok is True
    assert manifest["producer"] == "radjax-tome"
    assert manifest["schema_name"] == "teacher_tome_v0"
    assert logits.shape == (1, 3, 7)
