"""Scenario extension of PipeSim's three-tier max-plus evaluator.

The compute-unit rates stay fixed. Each scenario changes only DMA startup,
package HBM share, MIX IPC cost, and SIMT capacity-miss drift. The base
evaluator performs the SRAM and roofline checks anew for every scenario.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .dsl import AlgorithmFlow
from .evaluator import EvalOptions, EvaluationReport, evaluate
from .topology import NpuArchitecture, get_architecture
from .uncertainty import EnvironmentScenario, UncertaintySpace


@dataclass(frozen=True)
class ScenarioEvaluation:
    environment: EnvironmentScenario
    report: EvaluationReport

    @property
    def cycles(self) -> float:
        return self.report.total_cycles

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment": self.environment.to_dict(),
            "time_us": self.report.timing_us["total_us"],
            "limiting_stage": self.report.limiting_stage,
            "roofline_bound": self.report.roofline.binding_level,
        }


@dataclass(frozen=True)
class BoundedEvaluation:
    flow: AlgorithmFlow
    arch: NpuArchitecture
    samples: tuple[ScenarioEvaluation, ...]
    nominal: ScenarioEvaluation

    @property
    def feasible(self) -> bool:
        return all(sample.report.is_valid for sample in self.samples)

    @property
    def best_case_us(self) -> float:
        return min(sample.report.timing_us["total_us"] for sample in self.samples)

    @property
    def worst_case_us(self) -> float:
        return max(sample.report.timing_us["total_us"] for sample in self.samples)

    @property
    def nominal_us(self) -> float:
        return self.nominal.report.timing_us["total_us"]

    @property
    def peak_sram_occupancy_pct(self) -> float:
        return max(
            (fp.occupancy_pct for fp in self.nominal.report.sram_footprints.values()),
            default=0.0,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_case_us": self.best_case_us,
            "nominal_us": self.nominal_us,
            "worst_case_us": self.worst_case_us,
            "limiting_stage_nominal": self.nominal.report.limiting_stage,
        }


def evaluate_scenario(
    flow: AlgorithmFlow,
    arch: str | NpuArchitecture,
    scenario: EnvironmentScenario,
) -> ScenarioEvaluation:
    """Evaluate one point without mutating the shared architecture manifest."""
    arch = get_architecture(arch)
    scenario_arch = arch.evolve(
        hbm_bandwidth_gbps=arch.hbm_bandwidth_gbps
        * scenario.effective_hbm_bandwidth_fraction
    )
    for pipe in ("MTE2", "MTE2_UB"):
        if pipe in scenario_arch.pipes:
            channel = scenario_arch.pipe(pipe)
            # A lone core can use the local engine; multiple cores share HBM.
            hbm_share = scenario_arch.hbm_bandwidth_bpc_total() / flow.cores
            scenario_arch = scenario_arch.with_pipe_override(
                pipe,
                base_latency_cycles=scenario.dma_latency_cycles,
                bandwidth_bpc=min(channel.bandwidth_bpc, hbm_share),
            )
    options = EvalOptions(
        emit_hints=False,
        cross_core_sync_cycles=scenario.cross_core_sync_cycles,
        simt_cache_miss_rate_drift=scenario.simt_cache_miss_rate_drift,
    )
    return ScenarioEvaluation(scenario, evaluate(flow, scenario_arch, options))


def evaluate_parametric(
    flow: AlgorithmFlow,
    arch: str | NpuArchitecture,
    uncertainty_space: UncertaintySpace,
    *,
    scenarios: Sequence[EnvironmentScenario] | None = None,
) -> BoundedEvaluation:
    """Return scenario extrema and a nominal report for a typed flow.

    The four parameters are monotone in the current cost model, so extrema
    over the rectangular uncertainty box occur at its corners. Interior
    points are retained for regime and crossover analysis.
    """
    arch = get_architecture(arch)
    points = tuple(scenarios or uncertainty_space.scenarios())
    if not points:
        raise ValueError("at least one uncertainty scenario is required")
    nominal_point = uncertainty_space.nominal()
    samples = tuple(evaluate_scenario(flow, arch, point) for point in points)
    nominal = next((s for s in samples if s.environment == nominal_point), None)
    if nominal is None:
        nominal = evaluate_scenario(flow, arch, nominal_point)
    return BoundedEvaluation(flow, arch, samples, nominal)
