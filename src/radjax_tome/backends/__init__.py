"""Teacher backend interfaces and fake implementations."""

from radjax_tome.backends.base import TeacherBackend
from radjax_tome.backends.fake import FakeTeacherBackend

__all__ = ["FakeTeacherBackend", "TeacherBackend"]
