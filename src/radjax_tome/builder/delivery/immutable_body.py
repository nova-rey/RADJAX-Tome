"""Opt-in crash-consistent immutable-body transaction writer.

This is intentionally separate from the legacy staging path.  It writes and
validates a body before promotion, then atomically commits a small manifest;
linkage changes never rewrite the body.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from radjax_contract.tome.m8g import (
    CompactBody,
    _m8g_fv3,
    body_raw_digest,
    encode_compact_body,
    validate_body_bytes,
    validate_manifest,
)


class ImmutableBodyTransaction:
    def __init__(self, root: Path, *, profile: str = "producer_evidence") -> None:
        self.root = Path(root)
        self.profile = profile
        self.root.mkdir(parents=True, exist_ok=True)

    def commit(
        self,
        body: CompactBody,
        manifest: Mapping[str, Any],
        *,
        canonical_manifest_bytes: bytes,
    ) -> tuple[Path, Path]:
        """Atomically promote a validated body and its already-bound manifest."""

        body_bytes = encode_compact_body(body)
        validate_body_bytes(body_bytes, profile=self.profile)
        validate_manifest(dict(manifest), body)
        if manifest["body_raw_digest"] != body_raw_digest(body_bytes):
            raise ValueError("manifest body digest does not match body")
        expected_manifest_bytes = _m8g_fv3(dict(manifest))
        if canonical_manifest_bytes != expected_manifest_bytes:
            raise ValueError("manifest bytes do not match validated manifest")
        body_name = f"{body_raw_digest(body_bytes).hex()}.body"
        body_path = self.root / body_name
        manifest_path = self.root / (body_name + ".manifest")
        self._atomic_write(body_path, body_bytes)
        # Manifest bytes are canonical JSON only after the Contract validates
        # the closed record; callers provide the canonical serialized envelope.
        self._atomic_write(manifest_path, expected_manifest_bytes)
        return body_path, manifest_path

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError(f"immutable resource conflict: {path.name}")
            return
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


__all__ = ["ImmutableBodyTransaction"]
