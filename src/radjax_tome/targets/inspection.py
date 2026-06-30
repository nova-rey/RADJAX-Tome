from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from radjax_tome.targets.store import TeacherTargetStore


def inspect_target_store(path: str | Path) -> dict[str, Any]:
    store = TeacherTargetStore.open(path)
    store.validate()
    shard_paths = store.list_shards()
    array_keys: set[str] = set()
    shard_shapes: list[dict[str, Any]] = []
    for shard_id in range(store.metadata.shard_count):
        arrays = store.read_shard(shard_id)
        array_keys.update(arrays)
        shard_shapes.append(
            {
                "shard_id": shard_id,
                "arrays": {
                    name: {
                        "shape": list(np.asarray(value).shape),
                        "dtype": str(np.asarray(value).dtype),
                    }
                    for name, value in sorted(arrays.items())
                },
            }
        )
    return {
        "path": str(Path(path)),
        "target_type": store.metadata.target_type,
        "target_store_version": store.metadata.target_store_version,
        "model_id": store.metadata.model_id,
        "tokenizer_id": store.metadata.tokenizer_id,
        "vocab_size": store.metadata.vocab_size,
        "sequence_length": store.metadata.sequence_length,
        "num_examples": store.metadata.num_examples,
        "shard_count": store.metadata.shard_count,
        "shard_files": [item.name for item in shard_paths],
        "array_keys": sorted(array_keys),
        "target_params": dict(store.metadata.target_params),
        "shards": shard_shapes,
    }
