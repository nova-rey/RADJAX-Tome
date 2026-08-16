"""Opt-in Contract-governed immutable-body transaction writer.

The body store is content addressed, while the manifest is the semantic
commit point.  Private transaction directories and receipts are never package
inventory members.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from radjax_contract.tome.m8g import (
    PROFILE_NAMES,
    CompactBody,
    JournalState,
    _digest,
    _m8g_fv3,
    body_raw_digest,
    encode_compact_body,
    validate_body_bytes,
    validate_manifest,
    validate_receipt,
)


class ImmutableBodyTransaction:
    """Crash-recoverable, race-resistant writer for one selected exemplar."""

    def __init__(self, root: Path, *, profile: str = "producer_evidence") -> None:
        self.root = Path(root)
        self.profile = profile
        self._reject_symlink_chain(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ("bodies", "manifests", ".transactions"):
            path = self.root / name
            self._reject_symlink(path)
            path.mkdir(exist_ok=True)
            if not stat.S_ISDIR(os.lstat(path).st_mode):
                raise ValueError(f"transaction path is not a directory: {name}")

    def commit(
        self,
        body: CompactBody,
        manifest: Mapping[str, Any],
        *,
        canonical_manifest_bytes: bytes,
    ) -> tuple[Path, Path]:
        body_bytes = encode_compact_body(body)
        validate_body_bytes(body_bytes, profile=self.profile)
        closed_manifest = dict(manifest)
        validate_manifest(closed_manifest, body)
        body_digest = body_raw_digest(body_bytes)
        if closed_manifest["body_raw_digest"] != body_digest:
            raise ValueError("manifest body digest does not match body")
        expected_manifest_bytes = _m8g_fv3(closed_manifest)
        if canonical_manifest_bytes != expected_manifest_bytes:
            raise ValueError("manifest bytes do not match validated manifest")

        txid = _digest(
            b"RDX-M8G-TX-1", body_digest + closed_manifest["manifest_semantic_id"]
        ).hex()
        body_path = self.root / "bodies" / f"{body_digest.hex()}.body"
        manifest_path = (
            self.root
            / "manifests"
            / f"{closed_manifest['manifest_semantic_id'].hex()}.manifest"
        )
        lock_path = self.root / ".transactions" / f"{txid}.lock"
        journal_dir = self.root / ".transactions" / txid
        self._preflight_target(body_path, body_bytes)
        self._preflight_target(manifest_path, expected_manifest_bytes)
        self._reserve(lock_path, txid)
        try:
            journal_dir.mkdir(exist_ok=True)
            self._receipt(
                journal_dir,
                txid,
                JournalState.BODY_GENERATING,
                body_digest,
                None,
                closed_manifest,
                len(body_bytes),
            )
            staged_body = journal_dir / "body.tmp"
            self._atomic_write(staged_body, body_bytes)
            self._receipt(
                journal_dir,
                txid,
                JournalState.BODY_WRITTEN,
                body_digest,
                None,
                closed_manifest,
                len(body_bytes),
            )
            validate_body_bytes(staged_body.read_bytes(), profile=self.profile)
            self._receipt(
                journal_dir,
                txid,
                JournalState.BODY_VALIDATED,
                body_digest,
                None,
                closed_manifest,
                len(body_bytes),
            )
            self._receipt(
                journal_dir,
                txid,
                JournalState.BODY_DIGESTED,
                body_digest,
                None,
                closed_manifest,
                len(body_bytes),
            )
            self._atomic_write(body_path, body_bytes)
            self._receipt(
                journal_dir,
                txid,
                JournalState.BODY_PROMOTED,
                body_digest,
                None,
                closed_manifest,
                len(body_bytes),
            )
            staged_manifest = journal_dir / "manifest.tmp"
            self._atomic_write(staged_manifest, expected_manifest_bytes)
            self._receipt(
                journal_dir,
                txid,
                JournalState.MANIFEST_PROVISIONAL,
                body_digest,
                None,
                closed_manifest,
                len(body_bytes),
            )
            self._receipt(
                journal_dir,
                txid,
                JournalState.LINKAGE_FINALIZED,
                body_digest,
                None,
                closed_manifest,
                len(body_bytes),
            )
            self._receipt(
                journal_dir,
                txid,
                JournalState.MANIFEST_VALIDATED,
                body_digest,
                None,
                closed_manifest,
                len(body_bytes),
            )
            self._atomic_write(manifest_path, expected_manifest_bytes)
            manifest_digest = hashlib.sha256(expected_manifest_bytes).digest()
            self._receipt(
                journal_dir,
                txid,
                JournalState.MANIFEST_PROMOTED,
                body_digest,
                manifest_digest,
                closed_manifest,
                len(body_bytes),
            )
            self._bind_inventory(body_path, manifest_path, body_digest, manifest_digest)
            self._receipt(
                journal_dir,
                txid,
                JournalState.INVENTORY_COMMITTED,
                body_digest,
                manifest_digest,
                closed_manifest,
                len(body_bytes),
            )
            self._receipt(
                journal_dir,
                txid,
                JournalState.PACKAGE_VALIDATED,
                body_digest,
                manifest_digest,
                closed_manifest,
                len(body_bytes),
            )
            self._receipt(
                journal_dir,
                txid,
                JournalState.COMMITTED,
                body_digest,
                manifest_digest,
                closed_manifest,
                len(body_bytes),
            )
            staged_body.unlink(missing_ok=True)
            staged_manifest.unlink(missing_ok=True)
            return body_path, manifest_path
        finally:
            self._release(lock_path)

    def _bind_inventory(
        self,
        body_path: Path,
        manifest_path: Path,
        body_digest: bytes,
        manifest_digest: bytes,
    ) -> None:
        inventory_path = self.root / "inventory.json"
        if inventory_path.exists() and (
            stat.S_ISLNK(os.lstat(inventory_path).st_mode)
            or not stat.S_ISREG(os.lstat(inventory_path).st_mode)
        ):
            raise ValueError("inventory path type invalid")
        inventory: dict[str, Any] = {"schema": "m8g_inventory_v1", "members": {}}
        if inventory_path.exists():
            inventory = json.loads(inventory_path.read_text())
        inventory.setdefault("members", {})[manifest_path.name] = {
            "body": body_path.name,
            "body_raw_digest": body_digest.hex(),
            "manifest_raw_digest": manifest_digest.hex(),
        }
        encoded = json.dumps(inventory, sort_keys=True).encode("utf-8")
        self._atomic_replace(inventory_path, encoded)
        if json.loads(inventory_path.read_text()) != inventory:
            raise ValueError("inventory binding validation failed")

    def recover(self) -> list[dict[str, Any]]:
        """Validate journal chains and quarantine incomplete transactions."""
        results = []
        tx_root = self.root / ".transactions"
        for path in sorted(tx_root.iterdir()):
            if path.name.endswith(".lock") or path.name.startswith(".quarantine-"):
                continue
            mode = os.lstat(path).st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise ValueError("symlink or non-directory transaction entry")
            receipts = sorted(path.glob("receipt-*.json"))
            states = []
            valid = True
            for receipt in receipts:
                try:
                    mode = os.lstat(receipt).st_mode
                    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                        raise ValueError("receipt file type invalid")
                    decoded = json.loads(receipt.read_text())
                    for key in (
                        "body_raw_digest",
                        "manifest_raw_digest",
                        "configuration_identity",
                        "semantic_authority_identity",
                        "receipt_digest",
                    ):
                        if isinstance(decoded.get(key), str) and decoded[key]:
                            decoded[key] = bytes.fromhex(decoded[key])
                    validate_receipt(decoded)
                    if decoded["transaction_id"] != path.name:
                        raise ValueError("receipt transaction binding invalid")
                    binary = receipt.with_suffix(".bin")
                    if binary.is_symlink() or binary.read_bytes() != _m8g_fv3(decoded):
                        raise ValueError("receipt binary/json mismatch")
                    if decoded["state"] >= int(JournalState.BODY_PROMOTED):
                        body_file = self.root / str(decoded["body_path"])
                        if (
                            body_file.is_symlink()
                            or body_raw_digest(body_file.read_bytes())
                            != decoded["body_raw_digest"]
                        ):
                            raise ValueError("receipt body binding invalid")
                    if decoded["state"] >= int(JournalState.MANIFEST_PROMOTED):
                        manifest_file = self.root / str(decoded["manifest_path"])
                        if (
                            manifest_file.is_symlink()
                            or hashlib.sha256(manifest_file.read_bytes()).digest()
                            != decoded["manifest_raw_digest"]
                        ):
                            raise ValueError("receipt manifest binding invalid")
                    states.append(int(decoded["state"]))
                except (OSError, ValueError, KeyError, TypeError):
                    valid = False
                    states.append(None)
            if states != list(range(1, len(states) + 1)):
                valid = False
            if valid and states == list(range(1, 13)):
                try:
                    inventory = json.loads((self.root / "inventory.json").read_text())
                    manifest_names = {
                        Path(json.loads(item.read_text()).get("manifest_path", "")).name
                        for item in receipts
                    }
                    if not all(
                        name in inventory.get("members", {})
                        for name in manifest_names
                        if name
                    ):
                        valid = False
                except (OSError, ValueError, TypeError):
                    valid = False
            if valid and states == list(range(1, 13)):
                status = "committed"
            else:
                quarantine = tx_root / f".quarantine-{path.name}"
                if not quarantine.exists():
                    os.replace(path, quarantine)
                status = "quarantined"
            results.append(
                {
                    "transaction_id": path.name,
                    "receipt_count": len(receipts),
                    "states": states,
                    "status": status,
                    "staging": path.name,
                }
            )
        return results

    def _receipt(
        self,
        directory: Path,
        txid: str,
        state: JournalState,
        body_digest: bytes,
        manifest_digest: bytes | None,
        manifest: Mapping[str, Any],
        body_size: int,
    ) -> None:
        manifest_name = f"manifests/{manifest['manifest_semantic_id'].hex()}.manifest"
        receipt: dict[str, Any] = {
            "transaction_id": txid,
            "schema_version": "radjax_contract_m8g_v1",
            "profile_code": PROFILE_NAMES[manifest["profile"]],
            "state": int(state),
            "parent_transaction_id": None,
            "body_path": f"bodies/{body_digest.hex()}.body"
            if state >= JournalState.BODY_PROMOTED
            else None,
            "manifest_path": manifest_name
            if state >= JournalState.MANIFEST_PROMOTED
            else None,
            "body_raw_digest": body_digest
            if state >= JournalState.BODY_PROMOTED
            else None,
            "body_size_bytes": body_size
            if state >= JournalState.BODY_PROMOTED
            else None,
            "manifest_raw_digest": manifest_digest,
            "committed_next_state": None
            if state == JournalState.COMMITTED
            else int(state) + 1,
            "configuration_identity": manifest["authority_id"],
            "semantic_authority_identity": manifest["selection_authority_id"],
            "receipt_digest": b"",
        }
        receipt["receipt_digest"] = _digest(
            b"RDX-RECEIPT-1",
            _m8g_fv3({k: v for k, v in receipt.items() if k != "receipt_digest"}),
        )
        validate_receipt(receipt)
        self._atomic_write(
            directory / f"receipt-{int(state):02d}.bin", _m8g_fv3(receipt)
        )
        serializable = {
            key: value.hex() if isinstance(value, bytes) else value
            for key, value in receipt.items()
        }
        self._atomic_write(
            directory / f"receipt-{int(state):02d}.json",
            json.dumps(serializable, sort_keys=True).encode("utf-8"),
        )

    @staticmethod
    def _reject_symlink(path: Path) -> None:
        if path.exists() and stat.S_ISLNK(os.lstat(path).st_mode):
            raise ValueError(f"symlink path rejected: {path}")

    @staticmethod
    def _reject_symlink_chain(path: Path) -> None:
        absolute = path.absolute()
        current = Path(absolute.anchor)
        for component in absolute.parts[1:]:
            current /= component
            if current.exists() and stat.S_ISLNK(os.lstat(current).st_mode):
                raise ValueError(f"symlink parent rejected: {current}")

    @staticmethod
    def _preflight_target(path: Path, expected: bytes) -> None:
        ImmutableBodyTransaction._reject_symlink_chain(path.parent)
        if not path.exists():
            return
        mode = os.lstat(path).st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise ValueError(f"unsafe immutable target: {path.name}")
        if path.read_bytes() != expected:
            raise ValueError(f"immutable resource conflict: {path.name}")

    @staticmethod
    def _reserve(path: Path, txid: str) -> None:
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise ValueError(f"transaction reservation conflict: {txid}") from exc
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(txid)
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _release(path: Path) -> None:
        try:
            if not stat.S_ISLNK(os.lstat(path).st_mode):
                path.unlink()
        except FileNotFoundError:
            return

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        ImmutableBodyTransaction._preflight_target(path, data)
        if path.exists():
            return
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                ImmutableBodyTransaction._preflight_target(path, data)
            else:
                os.unlink(temporary)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.lexists(temporary):
                os.unlink(temporary)

    @staticmethod
    def _atomic_replace(path: Path, data: bytes) -> None:
        ImmutableBodyTransaction._reject_symlink_chain(path.parent)
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
            if os.path.lexists(temporary):
                os.unlink(temporary)


__all__ = ["ImmutableBodyTransaction"]
