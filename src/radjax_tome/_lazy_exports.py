"""Small shared helper for preserving package-level compatibility exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

LazyExportMap = dict[str, tuple[str, str]]


def resolve_lazy_export(
    namespace: dict[str, Any],
    exports: LazyExportMap,
    name: str,
) -> Any:
    """Load one declared compatibility export and cache it on its facade."""
    try:
        module_name, attribute_name = exports[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name), attribute_name)
    namespace[name] = value
    return value


def lazy_export_names(namespace: dict[str, Any], exports: LazyExportMap) -> list[str]:
    """Return a stable ``dir()`` view including declared lazy exports."""
    return sorted({*namespace, *exports})
