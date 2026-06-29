"""Migrated legacy TeacherTextbook builder."""

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
    "TeacherTextbookBuildConfig",
    "TeacherTextbookValidationReport",
    "TinyTextExample",
    "build_fake_teacher_textbook",
    "build_hf_teacher_textbook",
    "build_teacher_textbook",
    "load_text_examples",
    "validate_teacher_textbook",
    "write_teacher_textbook_validation_report",
]
