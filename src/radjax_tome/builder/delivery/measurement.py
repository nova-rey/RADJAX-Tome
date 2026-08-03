"""Private M8A selected-pass measurement helpers.

Nothing in this module participates in Tome authority, package metadata, or
production configuration.  It is deliberately reachable only through the
underscore-prefixed measurement plumbing in ``rerun`` and ``staging``.
"""

from __future__ import annotations

import resource
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

_ALLOWED_EXECUTION_CAPS = frozenset({1, 2, 4, 8})


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class SelectedPassMeasurementControl:
    """Private, nonsemantic cap for a read-only selected-pass replay only."""

    benchmark_only: bool
    effective_execution_cap: int
    immutable_checkpoint_digest: str
    checkpoint_root: Path
    temporary_output_root: Path
    production_publication_disabled: bool = True

    def __post_init__(self) -> None:
        if self.benchmark_only is not True:
            raise ValueError(
                "selected-pass measurement control requires benchmark_only=True"
            )
        if self.effective_execution_cap not in _ALLOWED_EXECUTION_CAPS:
            raise ValueError("selected-pass measurement cap must be one of 1, 2, 4, 8")
        if not self.immutable_checkpoint_digest.startswith("sha256:"):
            raise ValueError(
                "selected-pass measurement requires a sha256 checkpoint digest"
            )
        if self.production_publication_disabled is not True:
            raise ValueError(
                "selected-pass measurement must disable production publication"
            )
        checkpoint = _resolved(self.checkpoint_root)
        output = _resolved(self.temporary_output_root)
        if checkpoint == output:
            raise ValueError("measurement output root must not be the checkpoint root")
        # A benchmark root is deliberately confined to the process temporary root.
        import tempfile

        if not _is_relative_to(output, _resolved(Path(tempfile.gettempdir()))):
            raise ValueError("measurement output root must be under the temporary root")

    def validate_for_output(self, artifact_dir: Path) -> None:
        if _resolved(artifact_dir) != _resolved(self.temporary_output_root):
            raise ValueError(
                "measurement output root must exactly match the temporary output root"
            )


def _host_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes while Linux reports KiB.
    import platform

    return value if platform.system() == "Darwin" else value * 1024


def _cuda_resources(*, synchronize: bool = False) -> dict[str, object]:
    """Collect optional CUDA facts without making torch/NVML a requirement."""

    try:
        import torch
    except ImportError:
        return {"status": "not_available", "reason": "torch_unavailable"}
    if not torch.cuda.is_available():
        return {"status": "not_available", "reason": "cuda_unavailable"}
    try:
        if synchronize:
            torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        return {
            "status": "available",
            "current_allocated_bytes": int(torch.cuda.memory_allocated()),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "current_reserved_bytes": int(torch.cuda.memory_reserved()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "free_bytes": int(free),
            "total_bytes": int(total),
            "observer": "torch.cuda",
            "nvml": {"status": "not_available", "reason": "not_requested"},
        }
    except RuntimeError as exc:
        return {"status": "not_available", "reason": f"cuda_observer_error:{exc}"}


_PHASES = (
    "backend_construction",
    "model_tokenizer_load",
    "tokenization_input_preparation",
    "h2d_input_transfer",
    "teacher_forward",
    "selected_position_index_preparation",
    "selected_row_gather",
    "compact_reduction",
    "compact_d2h_transfer",
    "payload_conversion_linkage_validation",
    "hashing_json_atomic_write_fsync",
    # M8B records these private sub-phases in place of the historical
    # aggregate when they are observed.  The aggregate remains in the schema
    # so pre-M8B evidence is neither reinterpreted nor invalidated.
    "canonical_body_encoding_hash",
    "temporary_file_write",
    "temporary_file_close",
    "atomic_replacement",
    "resume_validation",
    "corridor_synchronization_rewrite",
    "retry_reload",
    "backend_close_cleanup",
)


M8B_STATISTICS_SCHEMA_VERSION = "m8b_selected_staging_statistics_v1"


def _median(observations: Iterable[float]) -> float:
    values = sorted(float(value) for value in observations)
    if not values:
        raise ValueError("statistics require at least one observation")
    midpoint = len(values) // 2
    if len(values) % 2:
        return values[midpoint]
    return (values[midpoint - 1] + values[midpoint]) / 2.0


def _spread(observations: Iterable[float]) -> float:
    values = [float(value) for value in observations]
    if not values:
        raise ValueError("statistics require at least one observation")
    return max(values) - min(values)


@dataclass(frozen=True)
class M8BStagingStatistics:
    """Frozen, nonsemantic comparison rules for M8B evidence only."""

    schema_version: str = M8B_STATISTICS_SCHEMA_VERSION
    measured_run_count: int = 3
    staging_gate_fraction: float = 0.50
    initial_staging_improvement_fraction: float = 0.25
    selected_pass_improvement_fraction: float = 0.15
    selected_delivery_improvement_fraction: float = 0.10
    material_regression_fraction: float = 0.05
    noise_multiplier: float = 2.0

    def __post_init__(self) -> None:
        if self.schema_version != M8B_STATISTICS_SCHEMA_VERSION:
            raise ValueError("unsupported M8B statistics schema version")
        if self.measured_run_count != 3:
            raise ValueError("M8B comparisons require exactly three measured runs")
        for field_name in (
            "staging_gate_fraction",
            "initial_staging_improvement_fraction",
            "selected_pass_improvement_fraction",
            "selected_delivery_improvement_fraction",
            "material_regression_fraction",
            "noise_multiplier",
        ):
            if float(getattr(self, field_name)) < 0.0:
                raise ValueError(f"M8B statistic {field_name} must be nonnegative")

    def receipt_projection(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "measured_run_count": self.measured_run_count,
            "median": "sorted_middle_or_mean_of_two_middle_v1",
            "spread": "max_minus_min_v1",
            "combined_spread": (
                "sqrt(baseline_spread_squared_plus_candidate_spread_squared)_v1"
            ),
            "improvement_beyond_noise": (
                "improvement_gt_noise_multiplier_times_combined_spread_v1"
            ),
            "material_regression": (
                "candidate_minus_baseline_gt_max("
                "fraction_times_baseline,noise_multiplier_times_combined_spread)_v1"
            ),
            "host_memory_limit": "baseline_max_plus_baseline_range_v1",
            "device_memory_limit": "baseline_max_plus_baseline_range_v1",
            "staging_gate_fraction": self.staging_gate_fraction,
            "initial_staging_improvement_fraction": (
                self.initial_staging_improvement_fraction
            ),
            "selected_pass_improvement_fraction": (
                self.selected_pass_improvement_fraction
            ),
            "selected_delivery_improvement_fraction": (
                self.selected_delivery_improvement_fraction
            ),
            "material_regression_fraction": self.material_regression_fraction,
            "noise_multiplier": self.noise_multiplier,
        }

    def summarize(self, observations: Iterable[float]) -> dict[str, float | None]:
        values = self._three_measurements(observations)
        median = _median(values)
        spread = _spread(values)
        return {
            "median": median,
            "spread": spread,
            "normalized_spread": (
                0.0
                if median == 0.0 and spread == 0.0
                else spread / median
                if median != 0.0
                else None
            ),
        }

    def combined_spread(
        self, baseline: Iterable[float], candidate: Iterable[float]
    ) -> float:
        import math

        return math.sqrt(
            _spread(self._three_measurements(baseline)) ** 2
            + _spread(self._three_measurements(candidate)) ** 2
        )

    def improvement_beyond_noise(
        self, baseline: Iterable[float], candidate: Iterable[float]
    ) -> bool:
        baseline_values = self._three_measurements(baseline)
        candidate_values = self._three_measurements(candidate)
        return _median(baseline_values) - _median(candidate_values) > (
            self.noise_multiplier
            * self.combined_spread(baseline_values, candidate_values)
        )

    def materially_regresses(
        self, baseline: Iterable[float], candidate: Iterable[float]
    ) -> bool:
        baseline_values = self._three_measurements(baseline)
        candidate_values = self._three_measurements(candidate)
        baseline_median = _median(baseline_values)
        candidate_median = _median(candidate_values)
        return candidate_median - baseline_median > max(
            self.material_regression_fraction * baseline_median,
            self.noise_multiplier
            * self.combined_spread(baseline_values, candidate_values),
        )

    def memory_limit(self, baseline_peaks: Iterable[int]) -> int:
        values = [int(value) for value in baseline_peaks]
        if len(values) != self.measured_run_count:
            raise ValueError("M8B memory limits require exactly three baseline peaks")
        return max(values) + max(values) - min(values)

    def _three_measurements(self, observations: Iterable[float]) -> list[float]:
        values = [float(value) for value in observations]
        if len(values) != self.measured_run_count:
            raise ValueError(
                "M8B statistics require exactly three measured observations"
            )
        return values


def validate_m8b_statistics_receipt(receipt: Mapping[str, object]) -> None:
    """Reject evidence that silently changes the frozen M8B comparison rules."""

    projection = receipt.get("statistics")
    if not isinstance(projection, Mapping):
        raise ValueError("M8B receipt is missing frozen statistics")
    expected = M8BStagingStatistics().receipt_projection()
    if dict(projection) != expected:
        raise ValueError("M8B receipt statistics do not match frozen definitions")


@dataclass
class SelectedPassExecutionDiagnostics:
    """Phase ledger whose accounting is limited to the selected-pass denominator."""

    control: SelectedPassMeasurementControl
    requested_batch_size: int
    started_at: float = field(default_factory=perf_counter)
    entry_rss_bytes: int = field(default_factory=_host_rss_bytes)
    entry_cuda: dict[str, object] = field(default_factory=_cuda_resources)
    phase_seconds: dict[str, float] = field(
        default_factory=lambda: {phase: 0.0 for phase in _PHASES}
    )
    phase_status: dict[str, str] = field(
        default_factory=lambda: {phase: "measured" for phase in _PHASES}
    )
    batches: list[dict[str, object]] = field(default_factory=list)
    oom_events: list[dict[str, int]] = field(default_factory=list)
    peak_rss_bytes: int = 0
    peak_cuda: dict[str, object] = field(default_factory=dict)
    _seen_shapes: set[tuple[object, ...]] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.peak_rss_bytes = self.entry_rss_bytes
        self.peak_cuda = dict(self.entry_cuda)
        # Generic backends do not expose inner phases.  They remain explicit,
        # rather than being silently attributed to the forward pass.
        for phase in (
            "model_tokenizer_load",
            "tokenization_input_preparation",
            "h2d_input_transfer",
            "teacher_forward",
            "selected_position_index_preparation",
            "selected_row_gather",
            "compact_reduction",
            "compact_d2h_transfer",
        ):
            self.phase_status[phase] = "not_available"

    def add(self, phase: str, seconds: float) -> None:
        self.phase_seconds[phase] += max(0.0, seconds)
        if self.phase_status[phase] == "not_available":
            self.phase_status[phase] = "measured_host_wall"

    def observe(self) -> None:
        self.peak_rss_bytes = max(self.peak_rss_bytes, _host_rss_bytes())
        current = _cuda_resources()
        if current.get("status") == "available":
            self.peak_cuda = current

    def add_backend_diagnostics(self, metadata: Mapping[str, object]) -> None:
        value = metadata.get("selected_pass_execution_v1_backend", metadata)
        if not isinstance(value, Mapping):
            return
        phases = value.get("phases")
        statuses = value.get("phase_statuses")
        if isinstance(phases, Mapping):
            for phase, seconds in phases.items():
                status = statuses.get(phase) if isinstance(statuses, Mapping) else None
                if (
                    phase in self.phase_seconds
                    and isinstance(seconds, (int, float))
                    and status == "measured_host_wall"
                ):
                    self.add(phase, float(seconds))
                    self.phase_status[phase] = "measured_host_wall"
        if isinstance(statuses, Mapping):
            for phase, status in statuses.items():
                if phase in self.phase_status and isinstance(status, str):
                    if self.phase_status[phase] != "measured_host_wall":
                        self.phase_status[phase] = status

    def record_batch(
        self,
        *,
        source_count: int,
        coordinate_count: int,
        selected_positions_per_source: Iterable[int],
        result: object,
        effective_size: int,
    ) -> None:
        input_ids = getattr(result, "input_ids", None)
        attention = getattr(result, "attention_mask", None)
        payload = getattr(result, "payload", {})
        shapes = {
            "input_ids": _shape_dtype(input_ids),
            "attention_mask": _shape_dtype(attention),
            "compact_result": _mapping_shape_dtype(payload),
        }
        metadata = getattr(result, "metadata", {})
        backend_metadata = (
            metadata.get("selected_pass_execution_v1_backend", {})
            if isinstance(metadata, Mapping)
            else {}
        )
        observations = (
            backend_metadata.get("tensor_observations")
            if isinstance(backend_metadata, Mapping)
            else None
        )
        if isinstance(observations, Mapping):
            # The GPU observer owns device-side tensor facts.  Keep the
            # returned input/payload summaries for generic backend parity.
            shapes.update(
                {
                    str(name): value
                    for name, value in observations.items()
                    if name
                    in {
                        "input",
                        "attention_mask",
                        "logits",
                        "gathered_logits",
                        "compact_result",
                    }
                }
            )
        padded_tokens = _tensor_size(input_ids)
        attended_tokens = _tensor_sum(attention)
        shape_key = tuple(sorted((name, repr(value)) for name, value in shapes.items()))
        self.batches.append(
            {
                "effective_source_batch_size": effective_size,
                "source_count": source_count,
                "coordinate_count": coordinate_count,
                "selected_positions_per_source": list(selected_positions_per_source),
                "padded_token_count": padded_tokens,
                "attended_token_count": attended_tokens,
                "shapes": shapes,
                "first_occurrence_shape": shape_key not in self._seen_shapes,
                "execution_engine": "eager",
                "compilation_cache_status": "not_authorized",
            }
        )
        self._seen_shapes.add(shape_key)
        self.observe()

    def finish(self) -> dict[str, object]:
        self.peak_cuda = _cuda_resources(synchronize=True)
        self.observe()
        wall = perf_counter() - self.started_at
        source_count = sum(int(batch["source_count"]) for batch in self.batches)
        coordinate_count = sum(int(batch["coordinate_count"]) for batch in self.batches)
        included = sum(self.phase_seconds.values())
        unattributed = max(0.0, wall - included)
        accounted = included + unattributed
        reconciliation = abs(accounted - wall) / wall if wall else 0.0
        return {
            "schema_version": "selected_pass_execution_v1",
            "benchmark_only": True,
            "immutable_checkpoint_digest": self.control.immutable_checkpoint_digest,
            "requested_source_batch_size": self.requested_batch_size,
            "benchmark_only_effective_execution_cap": (
                self.control.effective_execution_cap
            ),
            "production_publication_disabled": True,
            "selected_pass_wall_seconds": wall,
            "selected_source_examples_per_second": (
                source_count / wall if wall else None
            ),
            "selected_coordinates_per_second": (
                coordinate_count / wall if wall else None
            ),
            "included_phase_seconds": included,
            "unattributed_control_seconds": unattributed,
            "accounting_reconciliation_fraction": reconciliation,
            "accounting_within_five_percent": reconciliation <= 0.05,
            "phases": {
                phase: {
                    "seconds": self.phase_seconds[phase],
                    "status": self.phase_status[phase],
                }
                for phase in _PHASES
            },
            "batches": self.batches,
            "resources": {
                "process_rss_at_selected_pass_entry_bytes": self.entry_rss_bytes,
                "process_rss_peak_bytes": self.peak_rss_bytes,
                "process_rss_incremental_peak_bytes": max(
                    0, self.peak_rss_bytes - self.entry_rss_bytes
                ),
                "cuda_at_selected_pass_entry": self.entry_cuda,
                "cuda_peak": self.peak_cuda,
            },
            "oom_or_fallback_events": self.oom_events,
            "compilation": {"status": "not_authorized"},
            "post_selected_pass_phases": {
                "m7_publication": "not_measured",
                "contract_validation": "not_measured",
                "archive_creation": "not_measured",
                "v5_packaging": "not_measured",
            },
        }


def _shape_dtype(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return {"shape": list(shape), "dtype": str(getattr(value, "dtype", "unknown"))}


def _mapping_shape_dtype(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): _shape_dtype(item) for key, item in value.items()}


def _tensor_size(value: object) -> int | None:
    size = getattr(value, "size", None)
    if isinstance(size, int):
        return size
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    result = 1
    for dimension in shape:
        result *= int(dimension)
    return result


def _tensor_sum(value: object) -> int | None:
    if value is None or not hasattr(value, "sum"):
        return None
    try:
        return int(value.sum())
    except (TypeError, ValueError):
        return None


def deterministic_source_sample(
    selected_records: Iterable[Mapping[str, object]], *, sample_size: int = 64
) -> dict[str, object]:
    """Return the M8A largest-remainder, stable-source approximation."""

    sources: dict[tuple[int, int, str], set[int]] = defaultdict(set)
    for record in selected_records:
        key = (
            int(record.get("source_shard_id", 999_999_999)),
            int(record.get("source_row", 999_999_999)),
            str(record.get("selected_example_id", "")),
        )
        sources[key].add(int(record["source_position"]))
    ordered = sorted(sources.items())
    if sample_size < 1 or sample_size > len(ordered):
        raise ValueError(
            "sample size must be between one and the selected source count"
        )
    strata: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    for source, positions in ordered:
        strata[len(positions)].append(source)
    total_sources = len(ordered)
    quotas = {
        count: sample_size * len(items) / total_sources
        for count, items in strata.items()
    }
    allocation = {count: int(quota) for count, quota in quotas.items()}
    remainder = sample_size - sum(allocation.values())
    for count in sorted(
        strata, key=lambda item: (-(quotas[item] - allocation[item]), item)
    )[:remainder]:
        allocation[count] += 1
    sample = [
        source
        for count in sorted(strata)
        for source in strata[count][: allocation[count]]
    ]
    full_histogram = dict(
        sorted(Counter(len(items) for items in strata.values() for _ in items).items())
    )
    sampled_histogram = dict(
        sorted(Counter(len(sources[item]) for item in sample).items())
    )
    deviations = [
        abs(sampled_histogram.get(count, 0) / sample_size - len(items) / total_sources)
        for count, items in strata.items()
    ]
    return {
        "sample_source_keys": [list(source) for source in sample],
        "full_source_multiplicity_histogram": full_histogram,
        "sample_source_multiplicity_histogram": sampled_histogram,
        "allocation": {str(count): allocation[count] for count in sorted(allocation)},
        "allocation_residuals": {
            str(count): quotas[count] - allocation[count] for count in sorted(quotas)
        },
        "full_selected_coordinate_count": sum(len(item) for item in sources.values()),
        "sampled_selected_coordinate_count": sum(len(sources[item]) for item in sample),
        "maximum_proportional_deviation": max(deviations, default=0.0),
        "approximation_only": True,
    }
