"""Explore typed graph and tile choices under environmental uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Iterator

from .autotiler import _subtile_values, candidate_tile_values
from .dsl import (
    AlgorithmFlow,
    BufferPlan,
    EpilogueOp,
    ExecutionGraph,
    ExecutionMode,
    KPosition,
    LoopOrder,
    OpType,
    OutputPath,
    TileConfig,
)
from .parametric_evaluator import (
    BoundedEvaluation,
    evaluate_parametric,
    evaluate_scenario,
)
from .topology import NpuArchitecture, get_architecture
from .uncertainty import UncertaintySpace


@dataclass(frozen=True)
class FrontierSearchOptions:
    top_k: int = 5
    max_evaluations: int = 240
    max_tile: int = 256
    max_tile_values: int = 5
    max_subtile_values: int = 3
    forbid_tails: bool = False
    dominance_policy: str = "scenario"
    buffer_plans: tuple[BufferPlan, ...] = (
        BufferPlan.single(),
        BufferPlan.double(),
        BufferPlan.triple(),
    )

    def __post_init__(self) -> None:
        if (
            min(
                self.top_k,
                self.max_evaluations,
                self.max_tile,
                self.max_tile_values,
                self.max_subtile_values,
            )
            < 1
        ):
            raise ValueError("frontier search limits must be positive")
        if self.dominance_policy not in ("scenario", "certified_interval"):
            raise ValueError("dominance_policy must be scenario or certified_interval")


@dataclass(frozen=True)
class FrontierCandidate:
    frontier_rank: int
    strategy_profile: str
    execution_mode: ExecutionMode
    dominance_regime: dict[str, str]
    predicted_bounds: BoundedEvaluation
    flow: AlgorithmFlow

    def to_dict(self) -> dict[str, Any]:
        return {
            "frontier_rank": self.frontier_rank,
            "strategy_profile": self.strategy_profile,
            "execution_mode": self.execution_mode.value,
            "dominance_regime": self.dominance_regime,
            "predicted_bounds": self.predicted_bounds.to_dict(),
            "flow": self.flow.to_dict(),
        }


@dataclass(frozen=True)
class FrontierResult:
    arch_name: str
    uncertainty_space: UncertaintySpace
    candidates: tuple[FrontierCandidate, ...]
    evaluated: int
    feasible: int
    dominated: int
    dominance_policy: str
    truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "arch": self.arch_name,
            "uncertainty_space": self.uncertainty_space.to_dict(),
            "evaluated": self.evaluated,
            "feasible": self.feasible,
            "dominated": self.dominated,
            "dominance_policy": self.dominance_policy,
            "truncated": self.truncated,
            "frontier": [candidate.to_dict() for candidate in self.candidates],
        }


def _tile_values(
    extent: int, granularity: int, options: FrontierSearchOptions, seed: int
) -> list[int]:
    values = candidate_tile_values(
        extent,
        granularity,
        max_value=options.max_tile,
        max_values=options.max_tile_values,
        include_divisors=True,
    )
    if seed <= extent and seed not in values:
        values.append(seed)
    return sorted(set(values), reverse=True)


def _modes(flow: AlgorithmFlow, arch: NpuArchitecture) -> tuple[ExecutionMode, ...]:
    if flow.op_type is OpType.VECTOR_GEMV:
        return (
            (ExecutionMode.HYBRID_SIMT_SIMD,)
            if flow.simt
            else (ExecutionMode.AIV_ONLY,)
        )
    if flow.op_type is OpType.ELEMWISE:
        return ()
    modes = [ExecutionMode.STANDALONE_CUBE]
    if arch.decoupled_vector_core and arch.vector_cores >= 2 * flow.cores:
        modes += [ExecutionMode.MIX_SPLIT_M, ExecutionMode.MIX_SPLIT_N]
    return tuple(modes)


def _lower(
    flow: AlgorithmFlow, mode: ExecutionMode, plan: BufferPlan, ring_space: str
) -> AlgorithmFlow:
    if mode in (ExecutionMode.MIX_SPLIT_M, ExecutionMode.MIX_SPLIT_N):
        epilogue = flow.epilogue
        if not any(op.unit == "vector" for op in epilogue):
            epilogue += (EpilogueOp("identity", "vector"),)
        return flow.evolve(
            buffers=plan,
            output_path=OutputPath.FIXPIPE_UB_MTE3,
            epilogue=epilogue,
            execution_graph=ExecutionGraph(mode, max(1, plan.ub_depth), ring_space, 2),
        )
    return flow.evolve(
        buffers=plan,
        execution_graph=ExecutionGraph(mode),
    )


def _candidate_flows(
    template: AlgorithmFlow, arch: NpuArchitecture, options: FrontierSearchOptions
) -> Iterator[AlgorithmFlow]:
    modes = _modes(template, arch)
    if not modes:
        return
    if template.op_type is OpType.VECTOR_GEMV:
        rows = _tile_values(template.loops.m, 16, options, template.tile.tile_m)
        for tile_rows, plan in product(rows, options.buffer_plans):
            base = template.with_tile(
                TileConfig(
                    tile_rows, 1, template.loops.k, tile_rows, 1, template.loops.k
                )
            )
            yield _lower(base, modes[0], plan, "UB")
        return
    k0 = arch.cube.profile(template.compute_dtype).k0
    axes = (
        _tile_values(template.loops.m, 16, options, template.tile.tile_m),
        _tile_values(template.loops.n, 16, options, template.tile.tile_n),
        _tile_values(template.loops.k, k0, options, template.tile.tile_k),
    )
    triples = list(product(*axes))
    triples.sort(
        key=lambda dims: (
            -(dims[0] * dims[1]) / (dims[0] + dims[1]),
            -dims[2],
            -(dims[0] * dims[1]),
        )
    )
    profile = arch.cube.profile(template.compute_dtype)
    itemsize = template.compute_dtype.size_bytes
    mte1_bpc = arch.pipe("MTE1").bandwidth_bpc
    outer: list[tuple[tuple[int, int, int], list[tuple[int, int, int]]]] = []
    for tm, tn, tk in triples:
        if tm % 16 or tn % 16 or tk % k0:
            continue
        subtiles = [
            (sm, sn, sk)
            for sm, sn, sk in product(
                _subtile_values(tm, 16, options.max_subtile_values),
                _subtile_values(tn, 16, options.max_subtile_values),
                _subtile_values(tk, k0, options.max_subtile_values),
            )
            if tm % sm == 0
            and tn % sn == 0
            and tk % sk == 0
            and sm % 16 == 0
            and sn % 16 == 0
            and sk % k0 == 0
        ]
        subtiles.sort(
            key=lambda sub: (
                abs(
                    (sub[0] * sub[1] * sub[2] / profile.macs_per_cycle)
                    - ((sub[0] * sub[2] + sub[2] * sub[1]) * itemsize / mte1_bpc)
                ),
                -(sub[0] * sub[1] * sub[2]),
            )
        )
        if subtiles:
            outer.append(((tm, tn, tk), subtiles))

    # Sweep one balanced subtile per outer tile before revisiting an outer
    # tile with its next subtile. This preserves tile diversity under a budget.
    for sub_rank in range(max((len(subs) for _, subs in outer), default=0)):
        for (tm, tn, tk), subtiles in outer:
            if sub_rank >= len(subtiles):
                continue
            sm, sn, sk = subtiles[sub_rank]
            tile = TileConfig(tm, tn, tk, sm, sn, sk)
            for plan, order in product(
                options.buffer_plans, (LoopOrder.M_FIRST, LoopOrder.N_FIRST)
            ):
                if (tm * tk + tk * tn) * itemsize * plan.l1_depth > arch.pool(
                    "L1"
                ).capacity_bytes:
                    continue
                if tm * tn * 4 * plan.l0c_depth > arch.pool("L0C").capacity_bytes:
                    continue
                if sm * sk * itemsize * plan.l0_depth > arch.pool("L0A").capacity_bytes:
                    continue
                if sk * sn * itemsize * plan.l0_depth > arch.pool("L0B").capacity_bytes:
                    continue
                base = template.evolve(
                    tile=tile, loop_order=order, k_position=KPosition.INNER
                )
                for mode in modes:
                    spaces = (
                        ("UB", "GM")
                        if mode is not ExecutionMode.STANDALONE_CUBE
                        else ("UB",)
                    )
                    for space in spaces:
                        yield _lower(base, mode, plan, space)


def _certified_dominates(a: BoundedEvaluation, b: BoundedEvaluation) -> bool:
    """Sufficient all-theta proof: A's upper bound beats B's lower bound.

    This deliberately retains uncertain crossovers. Scenario-only ordering is
    not a proof of dominance over the continuous parameter box.
    """
    return a.worst_case_us <= b.best_case_us and a.worst_case_us < b.worst_case_us


def _scenario_dominates(a: BoundedEvaluation, b: BoundedEvaluation) -> bool:
    """Exact Pareto dominance on the declared finite scenario set."""
    left = (sample.cycles for sample in a.samples)
    right = (sample.cycles for sample in b.samples)
    pairs = tuple(zip(left, right))
    return all(x <= y + 1e-9 for x, y in pairs) and any(x < y - 1e-9 for x, y in pairs)


def _regime(item: BoundedEvaluation, peers: list[BoundedEvaluation]) -> dict[str, str]:
    winning = []
    for idx, sample in enumerate(item.samples):
        if sample.cycles <= min(peer.samples[idx].cycles for peer in peers) + 1e-9:
            winning.append(sample.environment)
    if winning:
        dma_min = min(point.dma_latency_cycles for point in winning)
        dma_max = max(point.dma_latency_cycles for point in winning)
        bw_min = min(point.effective_hbm_bandwidth_fraction for point in winning)
        bw_max = max(point.effective_hbm_bandwidth_fraction for point in winning)
        optimal = (
            f"sampled DMA {dma_min:g}..{dma_max:g} cycles; "
            f"HBM fraction {bw_min:.2f}..{bw_max:.2f}"
        )
    else:
        optimal = "No sampled outright win; retained by pairwise Pareto comparison"
    if item.flow.buffers.l1_depth >= 3:
        category = "Latency-Tolerant"
    elif item.peak_sram_occupancy_pct <= 45 and not item.flow.tile.has_tail():
        category = "Resource-Conservative"
    else:
        category = "Throughput-Aggressive"
    return {
        "category": category,
        "optimal_when": optimal,
        "risk_profile": (
            f"worst/nominal {item.worst_case_us / item.nominal_us:.2f}x; "
            f"peak SRAM {item.peak_sram_occupancy_pct:.1f}%"
        ),
    }


def _profile(item: BoundedEvaluation) -> str:
    flow = item.flow
    if flow.execution_graph.mode is ExecutionMode.HYBRID_SIMT_SIMD:
        return "SIMT prefill / SIMD GEMV"
    if flow.execution_graph.mode is ExecutionMode.AIV_ONLY:
        return "AIV-only SIMD GEMV"
    if flow.execution_graph.mode is ExecutionMode.STANDALONE_CUBE:
        return "Standalone Cube with inline Fixpipe"
    depth = flow.buffers.l1_depth
    return f"MIX split {flow.execution_graph.mode.value[-1].upper()}, L1 depth {depth}"


def find_optimal_frontiers(
    flow_template: AlgorithmFlow,
    arch: str | NpuArchitecture,
    uncertainty_space: UncertaintySpace,
    options: FrontierSearchOptions | None = None,
) -> FrontierResult:
    """Return ranked, typed DSL flows and their uncertainty envelopes.

    The default policy finds the exact Pareto set on the declared 25-point
    scenario grid. ``certified_interval`` only prunes when the timing bounds
    prove dominance across the continuous interval box.
    """
    arch = get_architecture(arch)
    options = options or FrontierSearchOptions()
    feasible: list[BoundedEvaluation] = []
    evaluated = 0
    seen: set[tuple[Any, ...]] = set()
    truncated = False
    for flow in _candidate_flows(flow_template, arch, options):
        if options.forbid_tails and (flow.tile.has_tail() or flow.tile.has_sub_tail()):
            continue
        key = (
            flow.tile.as_tuple(),
            flow.buffers,
            flow.loop_order,
            flow.execution_graph,
            flow.output_path,
        )
        if key in seen:
            continue
        seen.add(key)
        if evaluated >= options.max_evaluations:
            truncated = True
            break
        # Tier 1 is scenario-independent. One nominal pass rejects capacity
        # violations before the full 25-point scenario sweep.
        nominal = evaluate_scenario(flow, arch, uncertainty_space.nominal())
        evaluated += 1
        if not nominal.report.is_valid:
            continue
        result = evaluate_parametric(flow, arch, uncertainty_space)
        if result.feasible:
            feasible.append(result)
    dominates = (
        _scenario_dominates
        if options.dominance_policy == "scenario"
        else _certified_dominates
    )
    survivors = [
        item
        for item in feasible
        if not any(dominates(other, item) for other in feasible if other is not item)
    ]
    survivors.sort(
        key=lambda item: (
            item.nominal_us,
            item.worst_case_us,
            item.peak_sram_occupancy_pct,
        )
    )
    selected: list[BoundedEvaluation] = []
    used: set[tuple[ExecutionMode, int, str]] = set()
    used_flows: set[tuple[Any, ...]] = set()

    def flow_signature(item: BoundedEvaluation) -> tuple[Any, ...]:
        flow = item.flow
        return (
            flow.execution_graph.mode,
            flow.execution_graph.ring_space,
            flow.tile.as_tuple(),
            flow.buffers,
        )

    for item in survivors:
        flow = item.flow
        signature = (
            flow.execution_graph.mode,
            flow.buffers.l1_depth,
            flow.execution_graph.ring_space,
        )
        if signature not in used:
            selected.append(item)
            used.add(signature)
            used_flows.add(flow_signature(item))
            if len(selected) == options.top_k:
                break
    if len(selected) < options.top_k:
        for item in survivors:
            signature = flow_signature(item)
            if signature not in used_flows:
                selected.append(item)
                used_flows.add(signature)
                if len(selected) == options.top_k:
                    break
    selected = selected[: options.top_k]
    candidates = tuple(
        FrontierCandidate(
            rank,
            _profile(item),
            item.flow.execution_graph.mode,
            _regime(item, survivors),
            item,
            item.flow,
        )
        for rank, item in enumerate(selected, 1)
    )
    return FrontierResult(
        arch.name,
        uncertainty_space,
        candidates,
        evaluated,
        len(feasible),
        len(feasible) - len(survivors),
        options.dominance_policy,
        truncated,
    )
