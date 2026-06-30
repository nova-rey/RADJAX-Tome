"""Legacy TeacherTargetStore support used by the migrated builder."""

from radjax_tome.targets.compression import (
    dense_logits_to_cascaded_soft_labels,
    dense_logits_to_topk_tail,
    write_compressed_target_store,
)
from radjax_tome.targets.consumption import (
    OfflineTargetBatch,
    TeacherTargetBatch,
    load_offline_target_batch,
    load_teacher_target_batch,
)
from radjax_tome.targets.inspection import inspect_target_store
from radjax_tome.targets.multishard import (
    iter_offline_target_batches,
    iter_target_store_shard_ids,
    iter_teacher_target_batches,
    run_multishard_target_store_smoke,
)
from radjax_tome.targets.schema import (
    TEACHER_TARGET_STORE_SCHEMA_VERSION,
    TEACHER_TARGET_STORE_VERSION,
    TargetStoreMetadata,
    target_store_metadata_from_dict,
    target_store_metadata_to_dict,
    validate_target_store_metadata,
)
from radjax_tome.targets.store import TeacherTargetStore

__all__ = [
    "OfflineTargetBatch",
    "TEACHER_TARGET_STORE_SCHEMA_VERSION",
    "TEACHER_TARGET_STORE_VERSION",
    "TargetStoreMetadata",
    "TeacherTargetBatch",
    "TeacherTargetStore",
    "dense_logits_to_cascaded_soft_labels",
    "dense_logits_to_topk_tail",
    "inspect_target_store",
    "iter_offline_target_batches",
    "iter_target_store_shard_ids",
    "iter_teacher_target_batches",
    "load_offline_target_batch",
    "load_teacher_target_batch",
    "run_multishard_target_store_smoke",
    "target_store_metadata_from_dict",
    "target_store_metadata_to_dict",
    "validate_target_store_metadata",
    "write_compressed_target_store",
]
