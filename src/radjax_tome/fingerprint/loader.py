from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from radjax_tome.fingerprint.artifacts import (
    FingerprintManifest,
    read_fingerprint_manifest,
    validate_fingerprint_artifact,
)
from radjax_tome.fingerprint.exemplars import (
    FingerprintExemplarRecord,
    read_fingerprint_exemplar_records,
)


@dataclass(frozen=True)
class LoadedFingerprintArtifact:
    artifact_dir: Path
    manifest: FingerprintManifest
    exemplars: tuple[FingerprintExemplarRecord, ...] = ()

    @property
    def has_exemplars(self) -> bool:
        return bool(self.exemplars)


def load_fingerprint_artifact(
    path: str | Path,
    *,
    include_exemplars: bool = False,
    validate: bool = True,
) -> LoadedFingerprintArtifact:
    root = Path(path)
    if validate:
        result = validate_fingerprint_artifact(root)
        if not result.ok:
            raise ValueError(
                "fingerprint artifact validation failed: " + "; ".join(result.blockers)
            )
    manifest = read_fingerprint_manifest(root)
    exemplars: list[FingerprintExemplarRecord] = []
    if include_exemplars and manifest.exemplar_reservoir:
        payload_type = str(manifest.exemplar_reservoir.get("payload_type", ""))
        for shard in manifest.exemplar_reservoir.get("shards", ()):
            exemplars.extend(
                read_fingerprint_exemplar_records(
                    root / str(shard.get("path", "")),
                    payload_type=payload_type,
                )
            )
    return LoadedFingerprintArtifact(
        artifact_dir=root,
        manifest=manifest,
        exemplars=tuple(exemplars),
    )


__all__ = [
    "LoadedFingerprintArtifact",
    "load_fingerprint_artifact",
]
