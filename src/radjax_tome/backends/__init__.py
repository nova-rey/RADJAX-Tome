"""Teacher backend interfaces and fake/synthetic implementations."""

from radjax_tome.backends.base import TeacherBackend, TeacherTargetEmitter
from radjax_tome.backends.emission import emit_teacher_target_store
from radjax_tome.backends.fake import FakeTeacherBackend
from radjax_tome.backends.qwen_policy import (
    QwenPolicyEntry,
    QwenPolicyMap,
    QwenResolution,
    load_qwen_policy,
    resolve_qwen_policy,
    resolve_qwen_policy_map,
)
from radjax_tome.backends.synthetic import SyntheticTeacherBackend

__all__ = [
    "FakeTeacherBackend",
    "QwenPolicyEntry",
    "QwenPolicyMap",
    "QwenResolution",
    "SyntheticTeacherBackend",
    "TeacherBackend",
    "TeacherTargetEmitter",
    "emit_teacher_target_store",
    "load_qwen_policy",
    "resolve_qwen_policy",
    "resolve_qwen_policy_map",
]
