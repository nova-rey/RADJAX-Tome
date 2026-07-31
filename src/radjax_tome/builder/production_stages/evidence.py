"""Typed evidence construction used by extracted production stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from radjax_tome.builder.production_stages.shared import file_sha256


def native_file_evidence(
    stage: str,
    paths: tuple[Path, ...],
    *,
    counts: tuple[Any, ...] = (),
    prior: Any | None = None,
) -> Any:
    """Hash existing files as immutable typed evidence for one native stage."""

    from radjax_tome.builder.native_path_b.contracts import (
        FileHash,
        PriorStageProof,
        StageEvidence,
    )

    missing = tuple(path for path in paths if not path.is_file())
    if missing:
        raise ValueError(
            "native Path-B stage evidence is missing: "
            + ", ".join(str(path) for path in missing)
        )
    return StageEvidence(
        stage=stage,
        paths=paths,
        hashes=tuple(FileHash(path=path, sha256=file_sha256(path)) for path in paths),
        counts=counts,
        prior_stage_proof=(
            PriorStageProof(stage=prior.stage, paths=prior.paths, hashes=prior.hashes)
            if prior is not None
            else None
        ),
    )
