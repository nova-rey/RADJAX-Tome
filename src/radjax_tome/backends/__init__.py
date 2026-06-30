"""Teacher backend interfaces and fake/synthetic implementations."""

from radjax_tome.backends.base import TeacherBackend, TeacherTargetEmitter
from radjax_tome.backends.emission import emit_teacher_target_store
from radjax_tome.backends.fake import FakeTeacherBackend
from radjax_tome.backends.synthetic import SyntheticTeacherBackend

__all__ = [
    "FakeTeacherBackend",
    "SyntheticTeacherBackend",
    "TeacherBackend",
    "TeacherTargetEmitter",
    "emit_teacher_target_store",
]
