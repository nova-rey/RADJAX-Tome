"""Migrated legacy TeacherTextbook builder."""

from radjax_tome.builder.backend_textbook import (
    BackendTeacherTextbookBuildConfig,
    build_backend_teacher_textbook,
    build_streaming_backend_teacher_textbook,
    teacher_backend_config_from_build_config,
)
from radjax_tome.builder.exemplar_selection import (
    EXEMPLAR_SELECTION_MANIFEST_FILENAME,
    EXEMPLAR_SELECTION_MANIFEST_SCHEMA,
    MULTI_LEADERBOARD_SELECTOR_POLICY,
    ExemplarCandidate,
    build_exemplar_selection_manifest,
    extract_one_pass_candidates,
    extract_score_pass_candidates,
    select_exemplars,
    validate_exemplar_selection_manifest,
    write_exemplar_selection_manifest,
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
    "EXEMPLAR_SELECTION_MANIFEST_FILENAME",
    "EXEMPLAR_SELECTION_MANIFEST_SCHEMA",
    "MULTI_LEADERBOARD_SELECTOR_POLICY",
    "ExemplarCandidate",
    "TeacherTextbookBuildConfig",
    "TeacherTextbookValidationReport",
    "TinyTextExample",
    "build_backend_teacher_textbook",
    "build_exemplar_selection_manifest",
    "build_fake_teacher_textbook",
    "build_hf_teacher_textbook",
    "build_streaming_backend_teacher_textbook",
    "build_teacher_textbook",
    "extract_one_pass_candidates",
    "extract_score_pass_candidates",
    "load_text_examples",
    "select_exemplars",
    "teacher_backend_config_from_build_config",
    "validate_teacher_textbook",
    "validate_exemplar_selection_manifest",
    "write_exemplar_selection_manifest",
    "write_teacher_textbook_validation_report",
]
