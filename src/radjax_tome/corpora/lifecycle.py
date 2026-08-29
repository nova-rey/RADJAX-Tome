"""Restart-safe corpus v2 construction and publication."""

from __future__ import annotations

import json
import os
import secrets
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from radjax_tome.corpora.config import CorpusBuildIntent, canonical_bytes, sha256

JOURNAL_FILENAME = "corpus_build_journal_v1.jsonl"
STAGING_MARKER = ".radjax_corpus_staging"


class CorpusLifecycleError(ValueError):
    """A recoverable or fail-closed corpus publication error."""


def preflight_corpus_build(intent: CorpusBuildIntent) -> dict[str, Any]:
    """Validate only configuration and destination topology; perform no writes."""

    destination = intent.output_path
    minimum = intent.policy.get("filtering", {}).get("min_chars")
    maximum = intent.policy.get("chunking", {}).get("max_chars")
    if not isinstance(minimum, int) or minimum < 1:
        raise CorpusLifecycleError("policy.filtering.min_chars must be >= 1")
    if not isinstance(maximum, int) or maximum < minimum:
        raise CorpusLifecycleError("policy.chunking.max_chars must be >= min_chars")
    if destination.exists() and destination.is_symlink():
        raise CorpusLifecycleError("output destination may not be a symlink")
    for parent in [destination.parent, *destination.parent.parents]:
        if parent.is_symlink():
            raise CorpusLifecycleError("output parent may not contain a symlink")
    for source in intent.sources:
        if source.path.is_symlink():
            raise CorpusLifecycleError(
                f"source may not be a symlink: {source.source_id}"
            )
        if not source.path.exists():
            raise CorpusLifecycleError(f"source does not exist: {source.source_id}")
        if source.path.is_dir():
            for candidate in source.path.rglob("*"):
                if candidate.is_symlink():
                    raise CorpusLifecycleError(
                        f"source tree contains symlink: {candidate}"
                    )
    if destination.exists() and not intent.resume and not intent.overwrite:
        raise CorpusLifecycleError("output already exists; choose resume or overwrite")
    if intent.resume and not destination.exists():
        candidates = tuple(destination.parent.glob(f".{destination.name}.m10-*"))
        if not candidates:
            raise CorpusLifecycleError(
                "resume requires an existing output or owned staging state"
            )
    if destination.resolve() in {Path.cwd().resolve(), Path("/")}:
        raise CorpusLifecycleError("unsafe output destination")
    if destination.exists() and intent.overwrite:
        if not destination.is_dir():
            raise CorpusLifecycleError("overwrite requires an owned corpus directory")
        from radjax_tome.corpora.validation import validate_corpus_artifact_v2

        if not validate_corpus_artifact_v2(destination).ok:
            raise CorpusLifecycleError(
                "overwrite refuses a non-v2 or unowned destination"
            )
    destination_resolved = destination.resolve()
    for source in intent.sources:
        source_resolved = source.path.resolve()
        if (
            source_resolved == destination_resolved
            or source_resolved in destination_resolved.parents
            or destination_resolved in source_resolved.parents
        ):
            raise CorpusLifecycleError("source and output destination may not overlap")
    return {
        "status": "pass",
        "destination": str(destination),
        "action": "resume"
        if intent.resume
        else "overwrite"
        if intent.overwrite
        else "create",
    }


class CorpusJournal:
    def __init__(self, path: Path, transaction_id: str, config_identity: str):
        self.path = path
        self.transaction_id = transaction_id
        self.config_identity = config_identity
        self.sequence = 0
        self.previous_hash: str | None = None

    @classmethod
    def reopen(cls, path: Path, expected_config_identity: str) -> "CorpusJournal":
        """Reopen an owned journal without resetting its hash chain."""
        journal = cls(path, path.parents[1].name, expected_config_identity)
        if not path.is_file():
            raise CorpusLifecycleError("resume journal is missing")
        last: dict[str, Any] | None = None
        with path.open(encoding="utf-8") as handle:
            for sequence, line in enumerate(handle):
                event = json.loads(line)
                if event.get("sequence") != sequence:
                    raise CorpusLifecycleError("resume journal sequence is invalid")
                if event.get("config_identity") != expected_config_identity:
                    raise CorpusLifecycleError("resume journal identity mismatch")
                supplied = event.pop("event_hash", None)
                if supplied != sha256(canonical_bytes(event)):
                    raise CorpusLifecycleError("resume journal hash chain is invalid")
                if last is not None and event.get("previous_event_hash") != last.get(
                    "event_hash"
                ):
                    raise CorpusLifecycleError("resume journal predecessor is invalid")
                event["event_hash"] = supplied
                last = event
        if last is None:
            raise CorpusLifecycleError("resume journal is empty")
        journal.sequence = int(last["sequence"]) + 1
        journal.previous_hash = str(last["event_hash"])
        return journal

    def append(self, event_type: str, **fields: Any) -> dict[str, Any]:
        event = {
            "sequence": self.sequence,
            "event_type": event_type,
            "transaction_id": self.transaction_id,
            "config_identity": self.config_identity,
            **fields,
            "previous_event_hash": self.previous_hash,
        }
        event["event_hash"] = sha256(canonical_bytes(event))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as handle:
            handle.write(canonical_bytes(event) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.sequence += 1
        self.previous_hash = event["event_hash"]
        return event


def publish_staging(
    staging: Path,
    destination: Path,
    *,
    overwrite: bool,
    validate: Callable[[Path], Any],
    journal: CorpusJournal,
    ownership_check: Callable[[Path], bool] | None = None,
) -> None:
    """Promote a validated staging tree with explicit non-atomic overwrite semantics."""

    validate(staging)
    journal.append(
        "PROMOTION_INTENT", destination=str(destination), overwrite=overwrite
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        os.rename(staging, destination)
        _fsync_directory(destination.parent)
        journal.path = destination / "journal" / JOURNAL_FILENAME
        journal.append("PROMOTED", atomic_visibility=True)
        return
    if not overwrite:
        raise CorpusLifecycleError("output already exists")
    if ownership_check is not None and not ownership_check(destination):
        raise CorpusLifecycleError("overwrite destination ownership changed")
    before = destination.stat()
    quarantine = (
        destination.parent / f".{destination.name}.quarantine-{secrets.token_hex(8)}"
    )
    os.rename(destination, quarantine)
    _fsync_directory(destination.parent)
    journal.append(
        "OLD_QUARANTINED",
        destination=str(destination),
        quarantine=str(quarantine.name),
    )
    try:
        quarantined = quarantine.stat()
        if (quarantined.st_ino, quarantined.st_dev) != (before.st_ino, before.st_dev):
            raise CorpusLifecycleError(
                "overwrite destination changed before quarantine"
            )
        os.rename(staging, destination)
        _fsync_directory(destination.parent)
        journal.path = destination / "journal" / JOURNAL_FILENAME
        journal.append("NEW_PROMOTED", atomic_visibility=False)
        validate(destination)
    except Exception:
        if (
            destination.exists()
            and destination.is_dir()
            and (destination / STAGING_MARKER).is_file()
        ):
            shutil.rmtree(destination)
        if quarantine.exists():
            os.rename(quarantine, destination)
            _fsync_directory(destination.parent)
        journal.append("RESTORED")
        raise
    shutil.rmtree(quarantine)
    journal.append("COMMITTED")


def recover_publication(journal_path: str | Path, parent: Path) -> str:
    """Recover only names recorded by a journal; never guess or delete peers."""

    events = _read_journal(Path(journal_path))
    if not events:
        return "nothing_to_recover"
    last = events[-1]
    quarantine_name = last.get("quarantine")
    destination = Path(last.get("destination", "")) if last.get("destination") else None
    if last["event_type"] == "OLD_QUARANTINED" and quarantine_name and destination:
        if (
            destination.is_absolute()
            or destination.parent.resolve() != parent.resolve()
        ):
            return "no_safe_action"
        quarantine = parent / Path(quarantine_name).name
        if Path(quarantine_name).name != quarantine_name:
            return "no_safe_action"
        if not destination.exists() and quarantine.exists():
            os.rename(quarantine, destination)
            _fsync_directory(parent)
            return "restored"
    return "no_safe_action"


def _read_journal(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


__all__ = [
    "CorpusJournal",
    "CorpusLifecycleError",
    "JOURNAL_FILENAME",
    "preflight_corpus_build",
    "publish_staging",
    "recover_publication",
]
