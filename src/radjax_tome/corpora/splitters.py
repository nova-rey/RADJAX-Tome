from __future__ import annotations

from radjax_contract.provenance import stable_hash, validate_three_way_split


def split_corpus(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    splits = {"train": [], "calibration": [], "final_test": []}
    names = tuple(splits)
    for index, row in enumerate(rows):
        split_name = names[index % len(names)]
        text = row["text"]
        record = {
            "example_id": row["example_id"],
            "text": text,
            "token_sequence_hash": stable_hash(text.split()),
            "source_text_hash": stable_hash(text),
        }
        splits[split_name].append(record)
    result = validate_three_way_split(
        train=splits["train"],
        calibration=splits["calibration"],
        final_test=splits["final_test"],
    )
    if not result.ok:
        raise ValueError("; ".join(result.blockers))
    return splits
