from __future__ import annotations

import json
from pathlib import Path


def load_jsonl_corpus(path: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
            raise ValueError(f"line {line_number} must contain string field text")
        rows.append(
            {
                "example_id": str(payload.get("example_id", f"example-{line_number}")),
                "text": payload["text"],
            }
        )
    return rows
