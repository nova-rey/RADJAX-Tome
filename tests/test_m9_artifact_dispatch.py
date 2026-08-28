from pathlib import Path

import pytest

from radjax_tome.tome.artifact_dispatch import validate_artifact
from radjax_tome.tome.packaging import plan_package_destination


def test_historical_unknown_form_is_not_publicly_inferred(tmp_path: Path) -> None:
    path = tmp_path / "old.bin"
    path.write_bytes(b"old")
    with pytest.raises(ValueError, match="unsupported artifact path"):
        validate_artifact(path)


def test_package_destination_never_deletes_existing_data(tmp_path: Path) -> None:
    destination = tmp_path / "package"
    destination.mkdir()
    keep = destination / "keep.txt"
    keep.write_text("keep")
    with pytest.raises(ValueError, match="nonempty"):
        plan_package_destination(destination, overwrite=True)
    assert keep.read_text() == "keep"
