"""Teacher-side RADJAX artifact producer."""

from __future__ import annotations

from typing import Any

from radjax_tome._lazy_exports import (
    LazyExportMap,
    lazy_export_names,
    resolve_lazy_export,
)

_LAZY_EXPORTS: LazyExportMap = {
    "FakeTeacherBackend": ("radjax_tome.backends.fake", "FakeTeacherBackend"),
    "TeacherTextbookBuildConfig": (
        "radjax_tome.builder.teacher_textbook",
        "TeacherTextbookBuildConfig",
    ),
    "build_teacher_textbook": (
        "radjax_tome.builder.teacher_textbook",
        "build_teacher_textbook",
    ),
    "emit_toy_teacher_tome": ("radjax_tome.emit.teacher_tome", "emit_toy_teacher_tome"),
}

__all__ = [
    "FakeTeacherBackend",
    "TeacherTextbookBuildConfig",
    "build_teacher_textbook",
    "emit_toy_teacher_tome",
]


def __getattr__(name: str) -> Any:
    return resolve_lazy_export(globals(), _LAZY_EXPORTS, name)


def __dir__() -> list[str]:
    return lazy_export_names(globals(), _LAZY_EXPORTS)
