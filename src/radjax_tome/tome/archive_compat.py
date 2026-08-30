from __future__ import annotations

import tarfile
from pathlib import Path


def safe_extractall(archive: tarfile.TarFile, destination: str | Path) -> None:
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    members = archive.getmembers()
    for member in members:
        target = (root / member.name).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"unsafe archive member: {member.name}")
        if member.issym() or member.islnk():
            link_target = (target.parent / member.linkname).resolve()
            if link_target != root and root not in link_target.parents:
                raise ValueError(f"unsafe archive link: {member.name}")
    if callable(tarfile.TarFile.extractall):
        try:
            archive.extractall(root, filter="data")
            return
        except TypeError:
            pass
    archive.extractall(root)
