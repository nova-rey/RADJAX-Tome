"""Explicit validated source artifact handoff for package materialization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from radjax_tome.tome.canonical_artifact import derive_tome_semantic_identity
from radjax_tome.tome.contracts import TomeSemanticIdentity


@dataclass(frozen=True)
class ValidatedTomeArtifact:
    """A source root plus its profile-independent semantic identity.

    The source root and its training-authoritative semantic projection are
    checked before materialization.  A package is deliberately permitted to
    start from an older or subsequently augmented producer directory because
    packaging builds and validates its own profile-specific v3 inventory.
    """

    root: Path
    semantic_identity: TomeSemanticIdentity

    @classmethod
    def from_directory(cls, artifact_dir: Path) -> ValidatedTomeArtifact:
        root = artifact_dir.resolve()
        if not root.is_dir():
            raise ValueError(f"artifact directory does not exist: {root}")
        return cls(root=root, semantic_identity=derive_tome_semantic_identity(root))
