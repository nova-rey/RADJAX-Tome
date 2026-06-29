from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np


def write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def read_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise ValueError(f"missing npz file: {path}")
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}
