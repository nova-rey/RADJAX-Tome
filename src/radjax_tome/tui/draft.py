"""Canonical-data-only wizard drafts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class WizardDraft:
    workflow: str
    document: dict[str, Any]
    source_path: Path | None = None
    saved_path: Path | None = None
    saved_bytes: bytes | None = None
    provenance: dict[str, Any] = field(
        default_factory=lambda: {
            "surface": "tui",
            "schema_version": "m11_tui_provenance_v1",
        }
    )

    @property
    def dirty(self) -> bool:
        return self.saved_path is None or self.saved_bytes is None


__all__ = ["WizardDraft"]
