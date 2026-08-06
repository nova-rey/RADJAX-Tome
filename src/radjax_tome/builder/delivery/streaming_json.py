"""Bounded canonical JSON-object streaming for private selected-payload staging."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


class CanonicalJSONObjectError(ValueError):
    """A supposedly canonical JSON object has an invalid outer grammar."""


@dataclass
class _ObjectTerminus:
    """Incrementally locate exactly one outer object closure without buffering it."""

    started: bool = False
    complete: bool = False
    depth: int = 0
    in_string: bool = False
    escaped: bool = False
    unicode_remaining: int = 0

    def consume(
        self,
        chunk: str,
        emit: Callable[[str], None],
        final_closure: Callable[[str], None],
    ) -> None:
        if not isinstance(chunk, str):
            raise CanonicalJSONObjectError("canonical encoder emitted non-text chunk")
        pending: list[str] = []
        closure: str | None = None
        for character in chunk:
            if self.complete:
                if not character.isspace():
                    raise CanonicalJSONObjectError("trailing data after object closure")
                continue
            if not self.started:
                if character.isspace():
                    continue
                if character != "{":
                    raise CanonicalJSONObjectError("canonical root must be an object")
                self.started = True
                self.depth = 1
                pending.append(character)
                continue
            if self.in_string:
                pending.append(character)
                if self.unicode_remaining:
                    if character not in "0123456789abcdefABCDEF":
                        raise CanonicalJSONObjectError("invalid unicode escape")
                    self.unicode_remaining -= 1
                elif self.escaped:
                    self.escaped = False
                    if character == "u":
                        self.unicode_remaining = 4
                elif character == "\\":
                    self.escaped = True
                elif character == '"':
                    self.in_string = False
                continue
            if character == '"':
                self.in_string = True
                pending.append(character)
            elif character in "[{":
                self.depth += 1
                pending.append(character)
            elif character in "]}":
                self.depth -= 1
                if self.depth < 0:
                    raise CanonicalJSONObjectError("premature outer closure")
                if self.depth == 0:
                    if character != "}":
                        raise CanonicalJSONObjectError(
                            "canonical root must close as object"
                        )
                    self.complete = True
                    closure = character
                else:
                    pending.append(character)
            else:
                pending.append(character)
        if pending:
            emit("".join(pending))
        if closure is not None:
            final_closure(closure)

    def finish(self) -> None:
        if (
            not self.started
            or not self.complete
            or self.depth != 0
            or self.in_string
            or self.escaped
            or self.unicode_remaining
        ):
            raise CanonicalJSONObjectError("missing or incomplete outer object closure")


def stream_canonical_object_with_hash(
    payload: Mapping[str, Any],
    *,
    write: Callable[[str], None],
) -> str:
    """Write compact canonical object bytes once and return their SHA-256 digest.

    The caller owns the destination transaction. Only a fixed grammar state and
    an encoder chunk are live here; no complete encoded representation is kept.
    """
    if not isinstance(payload, Mapping):
        raise CanonicalJSONObjectError("canonical root must be an object")
    digest = hashlib.sha256()
    terminus = _ObjectTerminus()

    def emit(text: str) -> None:
        digest.update(text.encode("utf-8"))
        write(text)

    def final_closure(text: str) -> None:
        digest.update(text.encode("utf-8"))

    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"))
    for chunk in encoder.iterencode(dict(payload)):
        terminus.consume(chunk, emit, final_closure)
    terminus.finish()
    return digest.hexdigest()


__all__ = [
    "CanonicalJSONObjectError",
    "stream_canonical_object_with_hash",
]
