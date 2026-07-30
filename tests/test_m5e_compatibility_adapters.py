from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from radjax_tome.io.json import write_json
from radjax_tome.tome import (
    HISTORICAL_PACKAGE_COVER_SCHEMA,
    LEGACY_COVER_PAGE_V2,
    adapt_historical_tome_cover,
    build_cover_page,
    read_historical_tome_descriptor,
    validate_tome_cover_page,
)
from tests.helpers.fixtures import build_fake_teacher_textbook_artifact
from tests.helpers.subprocess import run_repo_python

ROOT = Path(__file__).resolve().parents[1]


def test_m5e_adapts_v2_only_from_known_cover_facts(tmp_path: Path) -> None:
    artifact = build_fake_teacher_textbook_artifact(tmp_path)
    legacy_cover = build_cover_page(artifact)
    write_json(artifact / "cover_page.json", legacy_cover)

    descriptor = read_historical_tome_descriptor(artifact)

    assert validate_tome_cover_page(artifact).ok
    assert descriptor.source_schema == LEGACY_COVER_PAGE_V2
    assert descriptor.identity is None
    assert descriptor.authority is None
    assert descriptor.package is None
    assert descriptor.training == {
        "target_type": "dense_logits",
        "sequence_length": 8,
        "vocab_size": 32,
        "tome_version": 1,
    }
    assert descriptor.manifests is not None
    assert descriptor.manifests["legacy_content_inventory"]["profile_complete"] is False
    assert "identity" in descriptor.unavailable_sections
    assert "rebuild or repackage" in descriptor.migration_diagnostic


def test_m5e_reads_legacy_outer_archive_without_reinterpreting_it(
    tmp_path: Path,
) -> None:
    artifact = build_fake_teacher_textbook_artifact(tmp_path)
    legacy_cover = build_cover_page(artifact)
    archive_path = tmp_path / "historical.tgz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in sorted(artifact.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(artifact).as_posix()
            payload = (
                json.dumps(legacy_cover, sort_keys=True).encode("utf-8")
                if relative == "cover_page.json"
                else path.read_bytes()
            )
            info = tarfile.TarInfo(f"historical/{relative}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))

    descriptor = read_historical_tome_descriptor(archive_path)

    assert descriptor.source_schema == LEGACY_COVER_PAGE_V2
    assert descriptor.package is None
    assert descriptor.identity is None


def test_m5e_package_v1_maps_profile_but_leaves_unknown_sections_absent() -> None:
    descriptor = adapt_historical_tome_cover(
        {
            "schema_version": HISTORICAL_PACKAGE_COVER_SCHEMA,
            "package_profile": "student",
            "layout": "unpacked_directory",
            "content_manifest": {
                "path": "manifests/content_manifest.json",
                "sha256": "legacy-raw-digest",
            },
            "top_level_summary": {"validation_status": "pass"},
        }
    )

    assert descriptor.source_schema == HISTORICAL_PACKAGE_COVER_SCHEMA
    assert descriptor.package == {"profile": "student", "transport": "directory"}
    assert descriptor.training is None
    assert descriptor.identity is None
    assert descriptor.authority is None
    assert descriptor.manifests == {
        "legacy_manifest_references": {
            "content_manifest": {
                "path": "manifests/content_manifest.json",
                "sha256": "legacy-raw-digest",
            }
        }
    }
    assert "identity" in descriptor.unavailable_sections


@pytest.mark.parametrize(
    "payload, message",
    (
        ({}, "unsupported historical Tome cover"),
        (
            {
                "artifact_kind": "radjax_tome",
                "cover_page_version": 2,
                "layout": "unpacked_directory",
                "contents": ["not-an-object"],
            },
            "contents entries must be objects",
        ),
        (
            {
                "schema_version": HISTORICAL_PACKAGE_COVER_SCHEMA,
                "package_profile": "student",
                "layout": "mystery-layout",
            },
            "layout is unsupported",
        ),
        (
            {
                "schema_version": HISTORICAL_PACKAGE_COVER_SCHEMA,
                "package_profile": "unknown-profile",
                "layout": "unpacked_directory",
            },
            "package_profile is unsupported",
        ),
    ),
)
def test_m5e_rejects_ambiguous_or_malformed_historical_input(
    payload: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        adapt_historical_tome_cover(payload)


def test_m5e_compatibility_reader_stays_lightweight() -> None:
    result = run_repo_python(
        ROOT,
        "-c",
        (
            "import sys; import radjax_tome.tome.compatibility; "
            "bad=[name for name in ('torch','transformers','jax') "
            "if name in sys.modules]; "
            "raise SystemExit(1 if bad else 0)"
        ),
    )

    assert result.returncode == 0, result.stderr
