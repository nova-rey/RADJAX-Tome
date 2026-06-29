"""Teacher-side RADJAX artifact producer."""

from radjax_tome.backends.fake import FakeTeacherBackend
from radjax_tome.emit.teacher_tome import emit_toy_teacher_tome

__all__ = ["FakeTeacherBackend", "emit_toy_teacher_tome"]
