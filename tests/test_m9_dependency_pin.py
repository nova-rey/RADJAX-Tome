import tomllib
from pathlib import Path


def test_declared_contract_pin_is_exact():
    data = tomllib.loads(Path("pyproject.toml").read_text())
    dep = next(
        x for x in data["project"]["dependencies"] if x.startswith("radjax-contract @ ")
    )
    assert dep.endswith("@373e3d17060d4ce1c4a0db6065c9289da714bde7")


def test_declared_contract_exposes_buffer_native_codec():
    from radjax_tome.builder.delivery.simple_compact_body import (
        encode_compact_body_packed_from_buffers,
    )

    assert callable(encode_compact_body_packed_from_buffers)
