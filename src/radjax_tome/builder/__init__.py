"""Migrated legacy TeacherTextbook builder."""

from radjax_tome.builder.backend_textbook import (
    BackendTeacherTextbookBuildConfig,
    build_backend_teacher_textbook,
    teacher_backend_config_from_build_config,
)
from radjax_tome.builder.teacher_textbook import (
    TEACHER_TEXTBOOK_VERSION,
    TeacherTextbookBuildConfig,
    TeacherTextbookValidationReport,
    TinyTextExample,
    build_fake_teacher_textbook,
    build_hf_teacher_textbook,
    build_teacher_textbook,
    load_text_examples,
    validate_teacher_textbook,
    write_teacher_textbook_validation_report,
)

__all__ = [
    "TEACHER_TEXTBOOK_VERSION",
    "BackendTeacherTextbookBuildConfig",
    "TeacherTextbookBuildConfig",
    "TeacherTextbookValidationReport",
    "TinyTextExample",
    "build_backend_teacher_textbook",
    "build_fake_teacher_textbook",
    "build_hf_teacher_textbook",
    "build_teacher_textbook",
    "load_text_examples",
    "teacher_backend_config_from_build_config",
    "validate_teacher_textbook",
    "write_teacher_textbook_validation_report",
]
