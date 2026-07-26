"""Offline semantic contracts for the canonical golden Tome artifact."""

from radjax_tome.golden.compare import compare_contracts, compare_fixture_artifact
from radjax_tome.golden.contract import (
    GOLDEN_CONTRACT_SCHEMA_V1,
    GOLDEN_CONTRACT_SCHEMA_V2,
    GOLDEN_CONTRACT_SCHEMA_VERSION,
    build_contract,
    validate_contract,
)
from radjax_tome.golden.projection import capture_golden_contract, validate_fixture

__all__ = [
    "GOLDEN_CONTRACT_SCHEMA_VERSION",
    "GOLDEN_CONTRACT_SCHEMA_V1",
    "GOLDEN_CONTRACT_SCHEMA_V2",
    "build_contract",
    "capture_golden_contract",
    "compare_contracts",
    "compare_fixture_artifact",
    "validate_contract",
    "validate_fixture",
]
