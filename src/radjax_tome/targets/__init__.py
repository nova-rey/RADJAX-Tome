"""Legacy TeacherTargetStore support used by the migrated builder."""

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
    "TEACHER_TARGET_STORE_SCHEMA_VERSION",
    "TEACHER_TARGET_STORE_VERSION",
    "TargetStoreMetadata",
    "TeacherTargetStore",
    "target_store_metadata_from_dict",
    "target_store_metadata_to_dict",
    "validate_target_store_metadata",
]
