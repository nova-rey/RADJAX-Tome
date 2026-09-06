"""Exclusive, fsynced publication for canonical wizard configuration drafts."""

from __future__ import annotations

import os
import secrets
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


def save_as_config(
    destination: str | Path,
    text: str,
    *,
    parse: Callable[[Any], Any],
) -> Path:
    """Validate and publish a new config without ever replacing an existing file."""
    target = Path(destination).resolve()
    if target.exists():
        raise FileExistsError(f"Save As destination already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    parsed = parse(text)
    if parsed is None:
        raise ValueError("config parser returned no document")
    temporary = target.with_name(f".{target.name}.m11-{secrets.token_hex(8)}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Save As destination already exists: {target}"
            ) from exc
        os.unlink(temporary)
        directory = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def export_config_document(
    destination: str | Path,
    document: Mapping[str, Any],
    *,
    serialize: Callable[[Mapping[str, Any]], str],
    parse: Callable[[Any], Any],
) -> Path:
    """Serialize, reparse, and exclusively publish a canonical document."""
    text = serialize(document)
    return save_as_config(destination, text, parse=parse)


__all__ = ["export_config_document", "save_as_config"]
