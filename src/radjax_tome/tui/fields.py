"""Small field helpers shared by the optional wizard screens."""

from __future__ import annotations

from typing import Any


def set_path(document: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current: dict[str, Any] = document
    for key in path[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            child = {}
            current[key] = child
        current = child
    current[path[-1]] = value


def get_path(
    document: dict[str, Any], path: tuple[str, ...], default: Any = None
) -> Any:
    current: Any = document
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


__all__ = ["get_path", "set_path"]
