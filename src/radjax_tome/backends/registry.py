from __future__ import annotations

from collections.abc import Callable

from radjax_tome.backends.base import (
    BackendCapability,
    TeacherBackendConfig,
    TeacherEmissionBackend,
)
from radjax_tome.backends.cpu import CPUReferenceTeacherEmissionBackend
from radjax_tome.backends.fake import FakeNumpyTeacherEmissionBackend
from radjax_tome.backends.hf_torch import HFTorchTeacherEmissionBackend

BackendFactory = Callable[[TeacherBackendConfig], TeacherEmissionBackend]

_BACKEND_FACTORIES: dict[str, BackendFactory] = {}


def register_backend(factory: BackendFactory) -> BackendFactory:
    backend_id = getattr(factory, "backend_id", None)
    if not isinstance(backend_id, str) or not backend_id:
        raise ValueError("backend factory must expose a non-empty backend_id")
    if backend_id in _BACKEND_FACTORIES:
        raise ValueError(f"backend is already registered: {backend_id}")
    _BACKEND_FACTORIES[backend_id] = factory
    return factory


def create_backend(config: TeacherBackendConfig) -> TeacherEmissionBackend:
    try:
        factory = _BACKEND_FACTORIES[config.backend_id]
    except KeyError as exc:
        raise ValueError(f"unknown teacher backend: {config.backend_id}") from exc
    return factory(config)


def list_backend_capabilities() -> tuple[BackendCapability, ...]:
    capabilities: list[BackendCapability] = []
    for backend_id in sorted(_BACKEND_FACTORIES):
        backend = create_backend(TeacherBackendConfig(backend_id=backend_id))
        capabilities.extend(backend.capabilities())
        backend.close()
    return tuple(capabilities)


register_backend(FakeNumpyTeacherEmissionBackend)
register_backend(CPUReferenceTeacherEmissionBackend)
register_backend(HFTorchTeacherEmissionBackend)
