"""Direct M3C tests for the import-isolated native Path-B boundary."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from radjax_tome.builder.native_path_b.api import (
    CANONICAL_DELIVERY_PATH,
    CANONICAL_SELECTION_INTEGRATION_POLICY,
    CANONICAL_TARGET_POLICY,
    CanonicalPathBConfig,
    resolve_canonical_path_b_config,
    run_canonical_path_b,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _config(**overrides: object) -> SimpleNamespace:
    values = {
        "target_policy": CANONICAL_TARGET_POLICY,
        "selection_integration_policy": CANONICAL_SELECTION_INTEGRATION_POLICY,
        "exemplar_selection_enabled": True,
        "exemplar_delivery_path": CANONICAL_DELIVERY_PATH,
        "total_selected_exemplar_budget": 4,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_resolver_accepts_only_the_exact_native_c6_path_b_request() -> None:
    source = _config()

    resolved = resolve_canonical_path_b_config(source)

    assert resolved == CanonicalPathBConfig(
        source_config=source,
        target_policy="corridor_exemplar_v1",
        selection_integration_policy="corridor_first_global_backfill_v1",
        exemplar_delivery_path="two_pass_rerun_selected",
        total_selected_exemplar_budget=4,
    )
    assert resolved.source_config is source
    with pytest.raises(FrozenInstanceError):
        resolved.total_selected_exemplar_budget = 8  # type: ignore[misc]


@pytest.mark.parametrize(
    "overrides",
    (
        {"target_policy": "dynamic_cascaded_soft_labels_v1"},
        {"selection_integration_policy": "global_only_v1"},
        {"exemplar_selection_enabled": False},
        {"exemplar_delivery_path": "one_pass_pruned_candidate"},
        {"total_selected_exemplar_budget": None},
    ),
)
def test_non_native_requests_do_not_route_to_the_canonical_adapter(
    overrides: dict[str, object],
) -> None:
    assert resolve_canonical_path_b_config(_config(**overrides)) is None


def test_resolver_does_not_normalize_target_policy_aliases() -> None:
    assert resolve_canonical_path_b_config(_config(target_policy="corridor")) is None


def test_callback_adapter_passes_only_the_original_source_config() -> None:
    source = _config()
    resolved = resolve_canonical_path_b_config(source)
    assert resolved is not None
    received: list[SimpleNamespace] = []

    def compatibility_executor(value: SimpleNamespace) -> dict[str, str]:
        received.append(value)
        return {"status": "pass"}

    result = run_canonical_path_b(
        resolved,
        compatibility_executor=compatibility_executor,
    )

    assert result == {"status": "pass"}
    assert received == [source]


def test_api_import_loads_no_research_or_experimental_runtime_modules() -> None:
    command = (
        "import json, sys\n"
        "import radjax_tome.builder.native_path_b.api\n"
        "print(json.dumps(sorted(name for name in sys.modules "
        "if name in {'radjax_tome.builder.multi_gpu_path_b', "
        "'radjax_tome.backends.hf_export', 'radjax_tome.backends.hf_specimen', "
        "'radjax_tome.backends.qwen_policy'})))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "[]"
