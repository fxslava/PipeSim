"""Three-tier analytical evaluation of a tiled DaVinci kernel.

The evaluator answers three questions, cheaply enough (tens of microseconds)
to sit inside a grid search:

**Tier 1 -- feasibility.** Do the staged buffers fit in the disjoint SRAM
pools at the requested multi-buffering depth, are the alignment invariants
satisfied, and how much MAC work is wasted on fractal padding?

**Tier 2 -- hierarchical roofline.** What is the operational intensity at each
level of the hierarchy (DRAM, MTE2, MTE1, Fixpipe), and which ceiling binds
first?

**Tier 3 -- max-plus pipeline scheduling.** Given per-stage service times and
the ping-pong depth of each staging buffer, what is the steady-state iteration
latency, where does the pipeline stall, and what is the resulting Cube
utilisation?

Traffic model
-------------
Every staging buffer is modelled as holding exactly one live tile (depth only
buys *overlap*, not *reuse*).  A tile is therefore re-fetched whenever its
index tuple changes between consecutive iterations of the loop nest.  For a
nest ordered outermost-first, operand ``X`` with index set ``deps`` is
re-fetched::

    d       = innermost loop position that X depends on and whose extent > 1
    loads   = product of the extents of loops[0..d]
    bytes   = loads x (bytes of one X tile)

This single rule reproduces every textbook GEMM reuse pattern, including the
degenerate ``n_tiles_k == 1`` case where the K loop vanishes and the outer
loop order starts to matter.  :func:`_resident_traffic` implements it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .dsl import (
    AlgorithmFlow,
    BufferPlan,
    KPosition,
    LoopOrder,
    OpType,
    OutputPath,
    TileConfig,
)
from .topology import DType, NpuArchitecture, get_architecture

__all__ = [
    "Severity",
    "Violation",
    "Diagnostic",
    "SramFootprint",
    "RooflineLevel",
    "HierarchicalRoofline",
    "PipelineStage",
    "PipelineSchedule",
    "EvaluationReport",
    "EvalOptions",
    "evaluate",
]


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def _ceil_div(a: int, b: int) -> int:
    return -(-int(a) // int(b))


def _pad(value: int, granularity: int) -> int:
    return _ceil_div(value, granularity) * granularity


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------


class Severity:
    """Violation severity levels (plain strings for painless JSON)."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class Violation:
    """A constraint the configuration breaks, or bends."""

    code: str
    severity: str
    message: str
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_fatal(self) -> bool:
        return self.severity == Severity.ERROR

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class Diagnostic:
    """An actionable recommendation, phrased for LLM prompt ingestion."""

    code: str
    message: str
    suggestion: str
    impact: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "suggestion": self.suggestion,
            "impact": self.impact,
        }


@dataclass(frozen=True)
class SramFootprint:
    """Allocation of one SRAM pool under the chosen buffer plan."""

    pool: str
    bytes_used: int
    capacity_bytes: int
    depth: int
    breakdown: Dict[str, int] = field(default_factory=dict)

    @property
    def fits(self) -> bool:
        return self.bytes_used <= self.capacity_bytes

    @property
    def occupancy_pct(self) -> float:
        return 100.0 * self.bytes_used / self.capacity_bytes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pool": self.pool,
            "bytes_used": self.bytes_used,
            "capacity_bytes": self.capacity_bytes,
            "depth": self.depth,
            "occupancy_pct": round(self.occupancy_pct, 2),
            "fits": self.fits,
            "breakdown": self.breakdown,
        }


@dataclass(frozen=True)
class RooflineLevel:
    """Operational intensity and the ceiling it implies at one level."""

    level: str
    bytes_moved: float
    intensity_flops_per_byte: float
    bandwidth_bpc: float
    ceiling_flops_per_cycle: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "bytes_moved": round(self.bytes_moved, 1),
            "intensity_flops_per_byte": round(self.intensity_flops_per_byte, 4),
            "bandwidth_bpc": self.bandwidth_bpc,
            "ceiling_flops_per_cycle": round(self.ceiling_flops_per_cycle, 2),
        }


@dataclass(frozen=True)
class HierarchicalRoofline:
    """The full set of ceilings plus the one that binds."""

    total_flops: int
    peak_flops_per_cycle: float
    levels: Tuple[RooflineLevel, ...]
    binding_level: str
    attainable_flops_per_cycle: float

    @property
    def roofline_efficiency_pct(self) -> float:
        return 100.0 * _safe_div(
            self.attainable_flops_per_cycle, self.peak_flops_per_cycle
        )

    def level(self, name: str) -> RooflineLevel:
        for lvl in self.levels:
            if lvl.level == name:
                return lvl
        raise KeyError(name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_flops": self.total_flops,
            "peak_flops_per_cycle": self.peak_flops_per_cycle,
            "attainable_flops_per_cycle": round(
                self.attainable_flops_per_cycle, 2
            ),
            "roofline_efficiency_pct": round(self.roofline_efficiency_pct, 2),
            "binding_level": self.binding_level,
            "levels": {lvl.level: lvl.to_dict() for lvl in self.levels},
        }


@dataclass(frozen=True)
class PipelineStage:
    """Steady-state service time of one hardware stage, per inner MAC step."""

    name: str
    cycles_per_step: float
    total_bytes: float
    overlapped: bool
    buffer_depth: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "cycles_per_step": round(self.cycles_per_step, 4),
            "total_bytes": round(self.total_bytes, 1),
            "overlapped": self.overlapped,
            "buffer_depth": self.buffer_depth,
        }


@dataclass(frozen=True)
class PipelineSchedule:
    """Outcome of the Tier-3 max-plus analysis."""

    stages: Tuple[PipelineStage, ...]
    steps: int
    steady_state_cycles_per_step: float
    fill_cycles: float
    drain_cycles: float
    total_cycles: float
    serialized_cycles: float
    ideal_cube_cycles: float
    issued_cube_cycles: float
    limiting_stage: str
    secondary_stage: str
    slack_pct: float

    @property
    def cube_utilization_pct(self) -> float:
        """Useful Cube work as a fraction of wall-clock cycles."""
        return 100.0 * _safe_div(self.ideal_cube_cycles, self.total_cycles)

    @property
    def cube_busy_pct(self) -> float:
        """Cycles the Cube is issuing anything, padding included."""
        return 100.0 * _safe_div(self.issued_cube_cycles, self.total_cycles)

    def stage(self, name: str) -> PipelineStage:
        for st in self.stages:
            if st.name == name:
                return st
        raise KeyError(name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": self.steps,
            "steady_state_cycles_per_step": round(
                self.steady_state_cycles_per_step, 4
            ),
            "fill_cycles": round(self.fill_cycles, 1),
            "drain_cycles": round(self.drain_cycles, 1),
            "total_cycles": round(self.total_cycles, 1),
            "serialized_cycles": round(self.serialized_cycles, 1),
            "ideal_cube_cycles": round(self.ideal_cube_cycles, 1),
            "issued_cube_cycles": round(self.issued_cube_cycles, 1),
            "limiting_stage": self.limiting_stage,
            "secondary_stage": self.secondary_stage,
            "slack_pct": round(self.slack_pct, 2),
            "cube_utilization_pct": round(self.cube_utilization_pct, 2),
            "cube_busy_pct": round(self.cube_busy_pct, 2),
            "stages": {st.name: st.to_dict() for st in self.stages},
        }


@dataclass(frozen=True)
class EvaluationReport:
    """Everything the three tiers produced, ready for JSON serialisation."""

    flow_name: str
    arch_name: str
    summary: str
    is_valid: bool
    violations: Tuple[Violation, ...]
    diagnostics: Tuple[Diagnostic, ...]
    sram_footprints: Dict[str, SramFootprint]
    fractal_efficiency: float
    roofline: HierarchicalRoofline
    schedule: PipelineSchedule
    t_min_cycles: float
    t_max_cycles: float
    code_generation_hints: Dict[str, Any]
    traffic: Dict[str, float]
    tile: Dict[str, int]
    timing_us: Dict[str, float]

    # -- headline metrics --------------------------------------------------
    @property
    def cube_utilization_pct(self) -> float:
        return self.schedule.cube_utilization_pct

    @property
    def limiting_stage(self) -> str:
        return self.schedule.limiting_stage

    @property
    def total_cycles(self) -> float:
        return self.schedule.total_cycles

    @property
    def achieved_tflops(self) -> float:
        return self.timing_us.get("achieved_tflops_single_core", 0.0)

    def errors(self) -> Tuple[Violation, ...]:
        return tuple(v for v in self.violations if v.is_fatal)

    def to_dict(self) -> Dict[str, Any]:
        """The structured payload consumed by the CLI and by LLM agents."""
        return {
            "flow": self.flow_name,
            "arch": self.arch_name,
            "summary": self.summary,
            "is_valid": self.is_valid,
            "cube_utilization_pct": round(self.cube_utilization_pct, 2),
            "limiting_stage": self.limiting_stage,
            "fractal_efficiency": round(self.fractal_efficiency, 4),
            "tail_penalty_pct": round(100.0 * (1.0 - self.fractal_efficiency), 2),
            "tile": self.tile,
            "hierarchical_roofline": self.roofline.to_dict(),
            "pipeline": self.schedule.to_dict(),
            "sram_footprints": {
                k: v.to_dict() for k, v in self.sram_footprints.items()
            },
            "traffic_bytes": {k: round(v, 1) for k, v in self.traffic.items()},
            "timing": {k: round(v, 4) for k, v in self.timing_us.items()},
            "bounds": {
                "t_min_cycles": round(self.t_min_cycles, 1),
                "t_max_cycles": round(self.t_max_cycles, 1),
                "t_actual_cycles": round(self.total_cycles, 1),
            },
            "violations": [v.to_dict() for v in self.violations],
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "code_generation_hints": self.code_generation_hints,
        }


@dataclass(frozen=True)
class EvalOptions:
    """Knobs that change the modelling assumptions, not the hardware."""

    #: Apply the per-core HBM share instead of the raw MTE2 width.
    model_hbm_contention: bool = True
    #: Treat the L2 as a reuse filter for the streamed operand.
    model_l2_reuse: bool = True
    #: Emit Ascend C code-generation hints (costs a few microseconds).
    emit_hints: bool = True
    #: Below this utilisation a diagnostic is raised.
    utilization_floor_pct: float = 60.0


# ---------------------------------------------------------------------------
# Loop-nest traffic model
# ---------------------------------------------------------------------------


def _resident_traffic(
    order: Sequence[Tuple[str, int]],
    deps: Sequence[str],
    axis_extents: Mapping[str, int],
    itemsize: int,
) -> Tuple[int, float]:
    """Loads and bytes for a depth-1 resident tile in a loop nest.

    Parameters
    ----------
    order:
        ``(loop_name, extent)`` pairs, outermost first.
    deps:
        Loop names the operand's address depends on.
    axis_extents:
        Full problem extent per dependent loop name (e.g. ``{"m": M, "k": K}``).
    itemsize:
        Element size of the operand in bytes.

    Returns
    -------
    ``(load_count, total_bytes)``
    """
    dep_set = set(deps)
    innermost = -1
    for idx, (loop_name, extent) in enumerate(order):
        if loop_name in dep_set and extent > 1:
            innermost = idx
    if innermost < 0:
        # Every dependent loop is degenerate: one load of the whole operand.
        total = itemsize
        for name in deps:
            total *= axis_extents[name]
        return 1, float(total)

    prefix = order[: innermost + 1]
    loads = 1
    repeat = 1
    for loop_name, extent in prefix:
        loads *= extent
        if loop_name not in dep_set:
            repeat *= extent
    operand_elements = 1
    for name in deps:
        operand_elements *= axis_extents[name]
    return loads, float(repeat * operand_elements * itemsize)


@dataclass(frozen=True)
class _AxisClass:
    """One equivalence class of tiles along a single axis."""

    extent: int           # real extent of this tile
    multiplicity: int     # how many tiles share this extent
    n_subs: int           # subtiles inside it
    padded_total: int     # sum of fractal-padded subtile lengths


def _axis_classes(extent: int, tile: int, sub: int, gran: int) -> List[_AxisClass]:
    """Split one axis into (full tile, tail tile) classes.

    All non-final tiles are identical, so at most two classes exist. Each
    class records the fractal-padded length actually issued to the Cube.
    """

    def padded_tile(t: int) -> Tuple[int, int]:
        n_s = _ceil_div(t, sub)
        sub_tail = t - (n_s - 1) * sub
        return n_s, (n_s - 1) * _pad(sub, gran) + _pad(sub_tail, gran)

    n_t = _ceil_div(extent, tile)
    tail = extent - (n_t - 1) * tile
    classes: List[_AxisClass] = []
    if n_t > 1:
        n_s, ptot = padded_tile(tile)
        classes.append(_AxisClass(tile, n_t - 1, n_s, ptot))
    n_s, ptot = padded_tile(tail)
    if classes and tail == tile:
        head = classes[0]
        classes[0] = _AxisClass(tile, head.multiplicity + 1, n_s, ptot)
    else:
        classes.append(_AxisClass(tail, 1, n_s, ptot))
    return classes


@dataclass(frozen=True)
class _InnerWork:
    """Aggregated inner-loop work, summed exactly over all tile classes."""

    issued_macs: int
    steps: int
    mte1_bytes: float
    max_l0a_elements: int
    max_l0b_elements: int
    max_l0c_elements: int


def _inner_work(
    flow: AlgorithmFlow, k0: int, itemsize: int
) -> _InnerWork:
    """Exact Cube and MTE1 totals, accounting for tails at both tile levels.

    The inner nest is ``for sk: for s_outer: for s_inner:`` where the outer
    spatial loop follows :attr:`AlgorithmFlow.loop_order`.  That makes the
    stationary operand (A for ``M_first``, B for ``N_first``) resident in its
    L0 buffer across the inner sweep, which is exactly the asymmetry the
    ``M_first`` / ``N_first`` choice is meant to express.
    """
    t = flow.tile
    loops = flow.loops
    m_classes = _axis_classes(loops.m, t.tile_m, t.sub_m, 16)
    n_classes = _axis_classes(loops.n, t.tile_n, t.sub_n, 16)
    k_classes = _axis_classes(loops.k, t.tile_k, t.sub_k, k0)

    issued = 0
    steps = 0
    mte1 = 0.0
    m_first = flow.loop_order is LoopOrder.M_FIRST
    for cm in m_classes:
        for cn in n_classes:
            for ck in k_classes:
                count = cm.multiplicity * cn.multiplicity * ck.multiplicity
                count *= loops.batch
                issued += count * cm.padded_total * cn.padded_total * ck.padded_total
                steps += count * cm.n_subs * cn.n_subs * ck.n_subs
                # Exact: sum over the resident sweep of padded subtile areas.
                if m_first:
                    a_bytes = cm.padded_total * ck.padded_total
                    b_bytes = cm.n_subs * ck.padded_total * cn.padded_total
                else:
                    a_bytes = cn.n_subs * cm.padded_total * ck.padded_total
                    b_bytes = ck.padded_total * cn.padded_total
                mte1 += count * (a_bytes + b_bytes) * itemsize

    return _InnerWork(
        issued_macs=issued,
        steps=steps,
        mte1_bytes=mte1,
        max_l0a_elements=_pad(t.sub_m, 16) * _pad(t.sub_k, k0),
        max_l0b_elements=_pad(t.sub_k, k0) * _pad(t.sub_n, 16),
        max_l0c_elements=_pad(t.tile_m, 16) * _pad(t.tile_n, 16),
    )


# ---------------------------------------------------------------------------
# Tier 1 -- feasibility
# ---------------------------------------------------------------------------


def _tier1_footprints(
    flow: AlgorithmFlow,
    arch: NpuArchitecture,
    work: _InnerWork,
    k0: int,
    itemsize: int,
    out_itemsize: int,
) -> Tuple[Dict[str, SramFootprint], List[Violation]]:
    """Buffer sizing and the hard capacity / alignment checks."""
    t = flow.tile
    plan = flow.buffers
    violations: List[Violation] = []
    footprints: Dict[str, SramFootprint] = {}

    if flow.op_type is OpType.ELEMWISE:
        return _tier1_footprints_elemwise(flow, arch, itemsize, out_itemsize)

    # -- L1: one A tile plus one B tile, each multi-buffered -------------
    l1 = arch.pool("L1")
    a_tile = l1.aligned_size(_pad(t.tile_m, 16) * _pad(t.tile_k, k0) * itemsize)
    b_tile = l1.aligned_size(_pad(t.tile_k, k0) * _pad(t.tile_n, 16) * itemsize)
    l1_bytes = (a_tile + b_tile) * plan.l1_depth
    footprints["L1"] = SramFootprint(
        "L1", l1_bytes, l1.capacity_bytes, plan.l1_depth,
        {"A_tile": a_tile, "B_tile": b_tile},
    )

    # -- L0A / L0B: one fractal subtile each ------------------------------
    l0a = arch.pool("L0A")
    l0b = arch.pool("L0B")
    a_sub = l0a.aligned_size(work.max_l0a_elements * itemsize)
    b_sub = l0b.aligned_size(work.max_l0b_elements * itemsize)
    footprints["L0A"] = SramFootprint(
        "L0A", a_sub * plan.l0_depth, l0a.capacity_bytes, plan.l0_depth,
        {"A_subtile": a_sub},
    )
    footprints["L0B"] = SramFootprint(
        "L0B", b_sub * plan.l0_depth, l0b.capacity_bytes, plan.l0_depth,
        {"B_subtile": b_sub},
    )

    # -- L0C: the accumulator must survive the whole reduction ------------
    l0c = arch.pool("L0C")
    acc_bytes = l0c.aligned_size(work.max_l0c_elements * 4)
    footprints["L0C"] = SramFootprint(
        "L0C", acc_bytes * plan.l0c_depth, l0c.capacity_bytes, plan.l0c_depth,
        {"accumulator_fp32": acc_bytes},
    )

    # -- UB: only when the epilogue leaves the Fixpipe --------------------
    ub = arch.pool("UB")
    ub_breakdown: Dict[str, int] = {}
    ub_bytes = 0
    if flow.uses_unified_buffer:
        out_tile = ub.aligned_size(
            _pad(t.tile_m, 16) * _pad(t.tile_n, 16) * out_itemsize
        )
        ub_breakdown["C_tile"] = out_tile
        ub_bytes += out_tile * plan.ub_depth
        for op in flow.vector_epilogue_ops():
            if op.operand is not None:
                operand_bytes = ub.aligned_size(op.operand.nbytes)
                ub_breakdown[op.operand.name] = operand_bytes
                ub_bytes += operand_bytes
    footprints["UB"] = SramFootprint(
        "UB", ub_bytes, ub.capacity_bytes, plan.ub_depth, ub_breakdown
    )

    # -- capacity violations ---------------------------------------------
    for name, fp in footprints.items():
        if not fp.fits:
            violations.append(
                Violation(
                    f"{name}_OVERFLOW",
                    Severity.ERROR,
                    f"{name} needs {fp.bytes_used} B at depth {fp.depth} but "
                    f"only {fp.capacity_bytes} B exist "
                    f"({fp.occupancy_pct:.0f}% of capacity)",
                    {
                        "pool": name,
                        "required_bytes": fp.bytes_used,
                        "capacity_bytes": fp.capacity_bytes,
                        "depth": fp.depth,
                        "breakdown": fp.breakdown,
                    },
                )
            )

    violations.extend(
        _alignment_violations(flow, arch, k0, itemsize, out_itemsize)
    )
    return footprints, violations


def _tier1_footprints_elemwise(
    flow: AlgorithmFlow,
    arch: NpuArchitecture,
    itemsize: int,
    out_itemsize: int,
) -> Tuple[Dict[str, SramFootprint], List[Violation]]:
    ub = arch.pool("UB")
    plan = flow.buffers
    tile_elems = flow.tile.tile_m
    n_in = flow.elemwise_inputs
    in_bytes = ub.aligned_size(tile_elems * itemsize) * n_in
    out_bytes = ub.aligned_size(tile_elems * out_itemsize)
    total = (in_bytes + out_bytes) * plan.ub_depth
    fp = SramFootprint(
        "UB", total, ub.capacity_bytes, plan.ub_depth,
        {"inputs": in_bytes, "output": out_bytes},
    )
    violations: List[Violation] = []
    if not fp.fits:
        violations.append(
            Violation(
                "UB_OVERFLOW", Severity.ERROR,
                f"UB needs {total} B at depth {plan.ub_depth} but only "
                f"{ub.capacity_bytes} B exist",
                {"required_bytes": total, "capacity_bytes": ub.capacity_bytes},
            )
        )
    return {"UB": fp}, violations


def _alignment_violations(
    flow: AlgorithmFlow,
    arch: NpuArchitecture,
    k0: int,
    itemsize: int,
    out_itemsize: int,
) -> List[Violation]:
    """Fractal, 32 B SRAM and 512 B GM-burst alignment checks."""
    t = flow.tile
    out: List[Violation] = []

    # The DaVinci fractal row is exactly 32 B by construction: k0 widens as
    # the element narrows.  A manifest that breaks that is misconfigured.
    if k0 * itemsize != 32:
        out.append(
            Violation(
                "FRACTAL_ROW_NOT_32B", Severity.WARNING,
                f"fractal row is {k0 * itemsize} B (k0={k0}, itemsize="
                f"{itemsize}); DaVinci fractals are 32 B rows",
                {"k0": k0, "itemsize": itemsize},
            )
        )

    for axis, value, gran in (
        ("tile_m", t.tile_m, 16), ("tile_n", t.tile_n, 16),
        ("tile_k", t.tile_k, k0),
        ("sub_m", t.sub_m, 16), ("sub_n", t.sub_n, 16), ("sub_k", t.sub_k, k0),
    ):
        if value % gran:
            out.append(
                Violation(
                    "TILE_NOT_FRACTAL_ALIGNED", Severity.WARNING,
                    f"{axis}={value} is not a multiple of {gran}; the Cube "
                    f"will issue padded fractals",
                    {"axis": axis, "value": value, "granularity": gran},
                )
            )

    for axis, extent, tile in (
        ("M", flow.loops.m, t.tile_m),
        ("N", flow.loops.n, t.tile_n),
        ("K", flow.loops.k, t.tile_k),
    ):
        if extent % tile:
            out.append(
                Violation(
                    "RAGGED_TILE_BOUNDARY", Severity.WARNING,
                    f"{axis}={extent} is not divisible by tile={tile}; the "
                    f"last tile is {extent % tile} wide and needs runtime "
                    f"tail handling (SetTail)",
                    {
                        "axis": axis, "extent": extent, "tile": tile,
                        "tail": extent % tile,
                        "requires_set_tail": True,
                    },
                )
            )

    # GM bursts: the contiguous run of one DataCopy row.
    gm_align = arch.pool("GM").alignment_bytes
    runs = {
        "A": (t.tile_k if not flow.transpose_a else t.tile_m) * itemsize,
        "B": (t.tile_n if not flow.transpose_b else t.tile_k) * itemsize,
        "C": t.tile_n * out_itemsize,
    }
    for operand, run in runs.items():
        if run % gm_align:
            out.append(
                Violation(
                    "GM_BURST_UNALIGNED", Severity.WARNING,
                    f"operand {operand}: the contiguous DataCopy run is "
                    f"{run} B, not a multiple of the {gm_align} B GM burst; "
                    f"MTE2/Fixpipe will waste partial bursts",
                    {
                        "operand": operand,
                        "run_bytes": run,
                        "burst_bytes": gm_align,
                        "burst_efficiency": round(
                            run / (_pad(run, gm_align) or 1), 3
                        ),
                    },
                )
            )

    # Epilogue ops the Fixpipe cannot actually absorb.
    for op in flow.fixpipe_epilogue_ops():
        if not op.is_hardware_supported():
            out.append(
                Violation(
                    "FIXPIPE_EPILOGUE_UNSUPPORTED", Severity.ERROR,
                    f"epilogue {op.kind!r} is scheduled on the Fixpipe but "
                    f"the hardware only folds in "
                    f"{sorted(op.FIXPIPE_CAPABLE)}; move it to the Vector unit",
                    {"kind": op.kind},
                )
            )

    # UB bank conflicts on the Vector epilogue.
    if flow.uses_unified_buffer:
        ub = arch.pool("UB")
        stride = t.tile_n * out_itemsize
        factor = ub.bank_conflict_factor(stride)
        if factor > 1.0:
            out.append(
                Violation(
                    "UB_BANK_CONFLICT", Severity.WARNING,
                    f"UB row stride {stride} B maps onto "
                    f"{ub.banks // max(1, int(factor))} of {ub.banks} banks; "
                    f"expect a {factor:.1f}x Vector slowdown. Pad the row.",
                    {"stride_bytes": stride, "slowdown": round(factor, 2)},
                )
            )
    return out


# ---------------------------------------------------------------------------
# Tier 2 -- hierarchical roofline
# ---------------------------------------------------------------------------


def _outer_traffic(
    flow: AlgorithmFlow, itemsize: int, out_itemsize: int
) -> Dict[str, float]:
    """GM-side traffic per operand, honouring loop order and K placement."""
    t = flow.tile
    loops = flow.loops
    n_m, n_n, n_k = t.n_tiles_m, t.n_tiles_n, t.n_tiles_k

    spatial = (("m", n_m), ("n", n_n))
    if flow.loop_order is LoopOrder.N_FIRST:
        spatial = (("n", n_n), ("m", n_m))
    if flow.k_position is KPosition.INNER:
        order = [*spatial, ("k", n_k)]
    else:
        order = [("k", n_k), *spatial]

    axis = {"m": loops.m, "n": loops.n, "k": loops.k}
    _, a_bytes = _resident_traffic(order, ("m", "k"), axis, itemsize)
    _, b_bytes = _resident_traffic(order, ("k", "n"), axis, itemsize)

    batch = loops.batch
    a_bytes *= batch
    b_bytes *= batch

    out_elements = batch * loops.m * loops.n
    if flow.k_position is KPosition.INNER:
        # One drain per output tile: C crosses the Fixpipe exactly once.
        c_write_bytes = float(out_elements * out_itemsize)
        c_read_bytes = 0.0
        c_drains = batch * n_m * n_n
    else:
        # The accumulator is re-materialised on every K step.
        c_write_bytes = float(out_elements * 4 * n_k)
        c_read_bytes = float(out_elements * 4 * (n_k - 1))
        c_drains = batch * n_m * n_n * n_k

    return {
        "mte2_a_bytes": a_bytes,
        "mte2_b_bytes": b_bytes,
        "mte2_c_readback_bytes": c_read_bytes,
        "mte2_bytes": a_bytes + b_bytes + c_read_bytes,
        "fixpipe_out_bytes": c_write_bytes,
        "_c_drains": float(c_drains),
    }


def _dram_bytes(
    flow: AlgorithmFlow,
    arch: NpuArchitecture,
    traffic: Mapping[str, float],
    itemsize: int,
    out_itemsize: int,
    options: EvalOptions,
) -> float:
    """MTE2 traffic after the L2 absorbs the re-reads it can hold.

    An operand whose whole footprint fits in L2 reaches DRAM exactly once, no
    matter how often the loop nest re-reads it.  One that does not fit streams
    from DRAM every time *and* pays for partial bursts, because a tile row
    shorter than the 512 B GM burst still drags a full burst across the bus.
    """
    loops = flow.loops
    a_footprint = float(loops.batch * loops.m * loops.k * itemsize)
    b_footprint = float(loops.batch * loops.k * loops.n * itemsize)
    a = traffic["mte2_a_bytes"]
    b = traffic["mte2_b_bytes"]

    l2 = arch.l2_capacity_bytes if options.model_l2_reuse else 0
    burst = arch.pool("GM").alignment_bytes
    t = flow.tile
    a_run = (t.tile_m if flow.transpose_a else t.tile_k) * itemsize
    b_run = (t.tile_k if flow.transpose_b else t.tile_n) * itemsize

    if a_footprint <= l2:
        a = a_footprint
    else:
        a *= _pad(a_run, burst) / a_run
    if b_footprint <= l2:
        b = b_footprint
    else:
        b *= _pad(b_run, burst) / b_run
    return a + b + traffic["mte2_c_readback_bytes"] + traffic["fixpipe_out_bytes"]


def _tier2_roofline(
    flow: AlgorithmFlow,
    arch: NpuArchitecture,
    traffic: Mapping[str, float],
    mte1_bytes: float,
    fixpipe_bytes: float,
    dram_bytes: float,
    peak_flops_per_cycle: float,
    mte2_bpc: float,
    options: EvalOptions,
) -> HierarchicalRoofline:
    """Assemble one ceiling per level of the hierarchy and pick the binder."""
    total_flops = flow.loops.flops
    levels: List[RooflineLevel] = []

    def add(level: str, moved: float, bpc: float) -> None:
        intensity = _safe_div(total_flops, moved)
        levels.append(
            RooflineLevel(
                level=level,
                bytes_moved=moved,
                intensity_flops_per_byte=intensity,
                bandwidth_bpc=bpc,
                ceiling_flops_per_cycle=(
                    intensity * bpc if moved > 0 else math.inf
                ),
            )
        )

    add("GM_MTE2", traffic["mte2_bytes"], mte2_bpc)
    add("L1_MTE1", mte1_bytes, arch.pipe("MTE1").bandwidth_bpc)
    add("L0C_FIXPIPE", fixpipe_bytes, arch.pipe("FIXPIPE").bandwidth_bpc)
    if options.model_hbm_contention and flow.cores > 1:
        share = arch.hbm_bandwidth_gbps * 1e9 / (arch.frequency_ghz * 1e9)
        add("DRAM", dram_bytes, share / flow.cores)
    else:
        add("DRAM", dram_bytes, arch.gm_bandwidth_bpc_per_core() * arch.cube_cores)

    attainable = peak_flops_per_cycle
    binding = "CUBE"
    for lvl in levels:
        if lvl.ceiling_flops_per_cycle < attainable:
            attainable = lvl.ceiling_flops_per_cycle
            binding = lvl.level
    return HierarchicalRoofline(
        total_flops=total_flops,
        peak_flops_per_cycle=peak_flops_per_cycle,
        levels=tuple(levels),
        binding_level=binding,
        attainable_flops_per_cycle=attainable,
    )


# ---------------------------------------------------------------------------
# Tier 3 -- max-plus pipeline schedule
# ---------------------------------------------------------------------------


_STAGE_TO_BOTTLENECK = {
    "MTE2": "MTE2_STARVATION",
    "MTE1": "MTE1_BOUND",
    "CUBE": "CUBE_COMPUTE_BOUND",
    "FIXPIPE": "FIXPIPE_BACKPRESSURE",
    "VECTOR": "VECTOR_BOUND",
    "MTE3": "MTE3_BOUND",
}


def _tier3_schedule(
    flow: AlgorithmFlow,
    arch: NpuArchitecture,
    work: _InnerWork,
    traffic: Mapping[str, float],
    fixpipe_bytes: float,
    vector_bytes: float,
    mte3_bytes: float,
    macs_per_cycle: float,
    mte2_bpc: float,
    mte2_derate: float,
) -> PipelineSchedule:
    """Steady-state ``T_iter = max_s tau_s`` with depth-aware serialisation.

    A stage whose destination buffer has depth >= 2 can overlap its consumer
    and therefore only contributes to the ``max``.  A depth-1 stage forces a
    barrier, so its service time *adds* to the iteration latency instead.
    """
    plan = flow.buffers
    steps = max(1, work.steps)
    cube = arch.cube

    issued_cube_cycles = work.issued_macs / macs_per_cycle
    ideal_cube_cycles = flow.loops.macs / macs_per_cycle

    stages: List[PipelineStage] = [
        PipelineStage(
            "MTE2",
            _safe_div(traffic["mte2_bytes"], steps * mte2_bpc * mte2_derate),
            traffic["mte2_bytes"],
            plan.l1_depth >= 2,
            plan.l1_depth,
        ),
        PipelineStage(
            "MTE1",
            _safe_div(work.mte1_bytes, steps * arch.pipe("MTE1").bandwidth_bpc),
            work.mte1_bytes,
            plan.l0_depth >= 2,
            plan.l0_depth,
        ),
        PipelineStage(
            "CUBE",
            issued_cube_cycles / steps + cube.issue_overhead_cycles,
            0.0,
            True,
            1,
        ),
        PipelineStage(
            "FIXPIPE",
            _safe_div(fixpipe_bytes, steps * arch.pipe("FIXPIPE").bandwidth_bpc),
            fixpipe_bytes,
            plan.l0c_depth >= 2 or flow.k_position is KPosition.INNER,
            plan.l0c_depth,
        ),
    ]
    if flow.uses_unified_buffer:
        vector_overlapped = arch.decoupled_vector_core and plan.ub_depth >= 2
        stages.append(
            PipelineStage(
                "VECTOR",
                _safe_div(vector_bytes, steps * arch.vector.bytes_per_cycle),
                vector_bytes,
                vector_overlapped,
                plan.ub_depth,
            )
        )
        stages.append(
            PipelineStage(
                "MTE3",
                _safe_div(mte3_bytes, steps * arch.pipe("MTE3").bandwidth_bpc),
                mte3_bytes,
                plan.ub_depth >= 2,
                plan.ub_depth,
            )
        )

    overlapped = [s for s in stages if s.overlapped]
    serialised = [s for s in stages if not s.overlapped]
    steady = max((s.cycles_per_step for s in overlapped), default=0.0)
    steady += sum(s.cycles_per_step for s in serialised)

    ranked = sorted(stages, key=lambda s: s.cycles_per_step, reverse=True)
    top = ranked[0]
    second = ranked[1] if len(ranked) > 1 else top
    limiting = _STAGE_TO_BOTTLENECK.get(top.name, top.name)
    # If fully overlapping every stage would materially shorten the step, the
    # real problem is the missing ping-pong, not whichever stage is widest.
    ideal_steady = max((s.cycles_per_step for s in stages), default=0.0)
    if serialised and steady > 1.2 * ideal_steady:
        limiting = "SERIALIZED_NO_DOUBLE_BUFFER"
    slack = 100.0 * (1.0 - _safe_div(second.cycles_per_step, top.cycles_per_step))

    # Pipeline fill: the first tile must traverse the whole chain unoverlapped.
    fill = (
        arch.pipe("MTE2").base_latency_cycles
        + arch.pipe("MTE1").base_latency_cycles
        + cube.pipeline_latency_cycles
    )
    drain = arch.pipe("FIXPIPE").base_latency_cycles
    if flow.uses_unified_buffer:
        drain += arch.pipe("MTE3").base_latency_cycles
    # Every outer tile pays the GM latency again unless depth hides it.
    outer_tiles = flow.tile.n_outer_tiles * flow.loops.batch
    if plan.l1_depth < 2:
        fill += outer_tiles * arch.pipe("MTE2").base_latency_cycles

    total = steps * steady + fill + drain
    serialized_total = (
        steps * sum(s.cycles_per_step for s in stages)
        + outer_tiles * (
            arch.pipe("MTE2").base_latency_cycles
            + arch.pipe("MTE1").base_latency_cycles
            + arch.pipe("FIXPIPE").base_latency_cycles
        )
    )

    return PipelineSchedule(
        stages=tuple(stages),
        steps=steps,
        steady_state_cycles_per_step=steady,
        fill_cycles=fill,
        drain_cycles=drain,
        total_cycles=total,
        serialized_cycles=serialized_total,
        ideal_cube_cycles=ideal_cube_cycles,
        issued_cube_cycles=issued_cube_cycles,
        limiting_stage=limiting,
        secondary_stage=second.name,
        slack_pct=slack,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _diagnostics(
    flow: AlgorithmFlow,
    arch: NpuArchitecture,
    schedule: PipelineSchedule,
    roofline: HierarchicalRoofline,
    footprints: Mapping[str, SramFootprint],
    fractal_efficiency: float,
    options: EvalOptions,
) -> List[Diagnostic]:
    """Turn the numbers into instructions an agent can act on."""
    out: List[Diagnostic] = []
    t = flow.tile
    util = schedule.cube_utilization_pct

    if schedule.limiting_stage == "MTE2_STARVATION":
        out.append(
            Diagnostic(
                "MTE2_STARVATION",
                "The Cube waits on GM->L1 loads: arithmetic intensity at the "
                f"MTE2 level is only "
                f"{roofline.level('GM_MTE2').intensity_flops_per_byte:.1f} "
                "FLOP/B.",
                f"Enlarge the outer tile (tile_m={t.tile_m}, tile_n={t.tile_n})"
                " so each loaded byte feeds more MACs; intensity grows as "
                "1/(1/tile_m + 1/tile_n). Raise l1_depth to hide GM latency.",
                impact="high",
            )
        )
    elif schedule.limiting_stage == "MTE1_BOUND":
        resident = flow.stationary_operand
        mte1_bpc = arch.pipe("MTE1").bandwidth_bpc
        # Balance point: 2*s*sub_k*itemsize / BW_MTE1 == s^2*sub_k / MACs_per_cyc
        itemsize = flow.compute_dtype.size_bytes
        balanced_s = 2 * arch.cube.profile(flow.compute_dtype).macs_per_cycle
        balanced_s = balanced_s * itemsize / mte1_bpc
        out.append(
            Diagnostic(
                "MTE1_BOUND",
                "L1->L0 fractal loads dominate: the inner subtile re-reads "
                "more from L1 than the Cube consumes. At the manifest's "
                f"{mte1_bpc:.0f} B/cycle MTE1 width, a square subtile only "
                f"balances the Cube at sub_m = sub_n = {balanced_s:.0f}, "
                f"which needs {balanced_s * balanced_s * 4 / 1024:.0f} KiB of "
                "L0C.",
                f"Grow sub_k (now {t.sub_k}) to deepen each Mmad, and grow "
                f"sub_{'n' if resident == 'A' else 'm'} so operand "
                f"{'B' if resident == 'A' else 'A'} is re-loaded less often "
                f"(operand {resident} is already stationary under "
                f"{flow.loop_order.value}). If the balance point exceeds L0C, "
                "the MTE1 width itself is the ceiling -- re-run with a "
                "measured value via --pipe-bw MTE1=<B/cycle>.",
                impact="high",
            )
        )
    elif schedule.limiting_stage == "FIXPIPE_BACKPRESSURE":
        out.append(
            Diagnostic(
                "FIXPIPE_BACKPRESSURE",
                "Draining L0C is the critical stage; the accumulator cannot "
                "be emptied as fast as it is filled.",
                "Increase tile_k so more MAC work happens per drain, or set "
                "k_position=inner so C is written exactly once, or raise "
                "l0c_depth to 2 to ping-pong the drain against the next tile.",
                impact="high",
            )
        )
    elif schedule.limiting_stage == "VECTOR_BOUND":
        out.append(
            Diagnostic(
                "VECTOR_BOUND",
                "The Vector epilogue is slower than the Cube feeding it.",
                "Fold the epilogue into the Fixpipe (unit='fixpipe') if it is "
                "dequant/quant/ReLU/bias -- that path is free. Otherwise "
                "shrink the UB tile and raise ub_depth.",
                impact="high",
            )
        )
    elif schedule.limiting_stage == "SERIALIZED_NO_DOUBLE_BUFFER":
        depth1 = [s.name for s in schedule.stages if not s.overlapped]
        out.append(
            Diagnostic(
                "SERIALIZED_NO_DOUBLE_BUFFER",
                f"Stages {depth1} run at depth 1, so their time adds to every "
                "iteration instead of overlapping.",
                "Set l1_depth=2 and l0_depth=2 (TQue<..., 2>) if the SRAM "
                "budget allows; this is usually the single largest win.",
                impact="high",
            )
        )
    elif schedule.limiting_stage == "CUBE_COMPUTE_BOUND":
        out.append(
            Diagnostic(
                "CUBE_COMPUTE_BOUND",
                f"The Cube is the critical stage at {util:.1f}% useful "
                "utilisation -- the tiling is balanced.",
                "Remaining headroom is fractal padding and pipeline fill; "
                "only larger problem extents or fewer tails will help.",
                impact="low",
            )
        )

    if fractal_efficiency < 0.95:
        out.append(
            Diagnostic(
                "FRACTAL_PADDING_WASTE",
                f"{100 * (1 - fractal_efficiency):.1f}% of issued MACs are "
                "fractal padding on ragged boundaries.",
                "Choose tiles that divide M, N and K, or pad the tensors to "
                "multiples of 16 (32 for INT8 along K) in the host allocator.",
                impact="medium" if fractal_efficiency < 0.85 else "low",
            )
        )

    fill_share = 100.0 * _safe_div(
        schedule.fill_cycles + schedule.drain_cycles, schedule.total_cycles
    )
    if fill_share > 15.0:
        out.append(
            Diagnostic(
                "LATENCY_BOUND_SHORT_PIPELINE",
                f"Pipeline fill and drain are {fill_share:.0f}% of total "
                "cycles: the steady state is too short to amortise them.",
                "Use larger tiles (fewer, longer pipeline runs) or fuse "
                "neighbouring operators so the pipeline stays full.",
                impact="medium",
            )
        )

    for name, fp in footprints.items():
        if fp.fits and 0 < fp.occupancy_pct < 35.0 and name in ("L1", "L0C"):
            out.append(
                Diagnostic(
                    f"{name}_UNDERUSED",
                    f"{name} is only {fp.occupancy_pct:.0f}% occupied.",
                    f"There is room to enlarge the tile feeding {name}; "
                    "bigger tiles raise arithmetic intensity at no SRAM cost.",
                    impact="medium",
                )
            )

    if flow.cores > 1 and options.model_hbm_contention:
        dram = roofline.level("DRAM")
        if dram.ceiling_flops_per_cycle < roofline.peak_flops_per_cycle:
            out.append(
                Diagnostic(
                    "HBM_CONTENTION",
                    f"With {flow.cores} cores streaming, the per-core HBM "
                    f"share is {dram.bandwidth_bpc:.0f} B/cycle against an "
                    f"MTE2 width of {arch.pipe('MTE2').bandwidth_bpc:.0f} "
                    "B/cycle.",
                    "Raise L2 reuse: order the core grid so neighbouring "
                    "cores share the streamed operand, or tile N across cores "
                    "and keep A resident.",
                    impact="high",
                )
            )

    if util < options.utilization_floor_pct and not out:
        out.append(
            Diagnostic(
                "LOW_UTILIZATION",
                f"Cube utilisation is {util:.1f}% with no single dominant "
                "stage.",
                "Run `npu-pipesim autotile` on this shape; the search "
                "explores the tile space this configuration is missing.",
                impact="medium",
            )
        )
    return out


# ---------------------------------------------------------------------------
# Code-generation hints
# ---------------------------------------------------------------------------


def _code_generation_hints(
    flow: AlgorithmFlow,
    arch: NpuArchitecture,
    work: _InnerWork,
    footprints: Mapping[str, SramFootprint],
    schedule: PipelineSchedule,
    k0: int,
    itemsize: int,
    out_itemsize: int,
) -> Dict[str, Any]:
    """Concrete Ascend C scaffolding derived from the validated tiling."""
    t = flow.tile
    plan = flow.buffers
    m_first = flow.loop_order is LoopOrder.M_FIRST
    outer = ["mo", "no"] if m_first else ["no", "mo"]
    if flow.k_position is KPosition.INNER:
        outer_nest = [*outer, "ko"]
    else:
        outer_nest = ["ko", *outer]
    inner_nest = ["sk", "sm", "sn"] if m_first else ["sk", "sn", "sm"]

    extents = {
        "mo": t.n_tiles_m, "no": t.n_tiles_n, "ko": t.n_tiles_k,
        "sm": t.n_subs_m, "sn": t.n_subs_n, "sk": t.n_subs_k,
    }
    pseudo = []
    indent = ""
    for name in outer_nest:
        pseudo.append(f"{indent}for {name} in range({extents[name]}):")
        indent += "  "
    pseudo.append(f"{indent}CopyIn A[{t.tile_m}x{t.tile_k}], B[{t.tile_k}x{t.tile_n}]  // MTE2, Nd2Nz")
    for name in inner_nest:
        pseudo.append(f"{indent}for {name} in range({extents[name]}):")
        indent += "  "
    pseudo.append(f"{indent}LoadData -> L0A/L0B  // MTE1")
    pseudo.append(f"{indent}Mmad({t.sub_m}, {t.sub_n}, {t.sub_k})  // Cube, accumulate in L0C")
    indent = indent[: -2 * len(inner_nest)]
    pseudo.append(f"{indent}Fixpipe L0C[{t.tile_m}x{t.tile_n}] -> "
                  f"{'UB then MTE3' if flow.uses_unified_buffer else 'GM'}")

    requires_tail = t.has_tail() or t.has_sub_tail()
    hints: Dict[str, Any] = {
        "block_dim": flow.cores,
        "loop_order": flow.loop_order.value,
        "k_position": flow.k_position.value,
        "outer_loop_nest": outer_nest,
        "inner_loop_nest": inner_nest,
        "loop_extents": extents,
        "pseudocode": pseudo,
        "fractal": {"m0": 16, "n0": 16, "k0": k0,
                    "note": f"{flow.compute_dtype.value} fractal rows are "
                            f"{k0 * itemsize} B"},
        "requires_set_tail": requires_tail,
        "tail_sizes": {
            "m": t.tail_m if flow.loops.m % t.tile_m else 0,
            "n": t.tail_n if flow.loops.n % t.tile_n else 0,
            "k": t.tail_k if flow.loops.k % t.tile_k else 0,
        },
        "queue_declarations": [
            f"TQue<TPosition::A1, {plan.l1_depth}> inQueueA1;  // "
            f"{footprints['L1'].breakdown.get('A_tile', 0)} B/buffer",
            f"TQue<TPosition::B1, {plan.l1_depth}> inQueueB1;  // "
            f"{footprints['L1'].breakdown.get('B_tile', 0)} B/buffer",
            f"TQue<TPosition::A2, {plan.l0_depth}> inQueueA2;  // "
            f"{footprints['L0A'].breakdown.get('A_subtile', 0)} B/buffer",
            f"TQue<TPosition::B2, {plan.l0_depth}> inQueueB2;  // "
            f"{footprints['L0B'].breakdown.get('B_subtile', 0)} B/buffer",
            f"TQue<TPosition::CO1, {plan.l0c_depth}> outQueueCO1;  // "
            f"{footprints['L0C'].breakdown.get('accumulator_fp32', 0)} B/buffer",
        ],
        "pipe_barriers": [
            "SetFlag/WaitFlag(MTE2 -> MTE1) around the L1 tile handoff",
            "SetFlag/WaitFlag(MTE1 -> M) around the L0A/L0B handoff",
            "SetFlag/WaitFlag(M -> FIX) around the L0C drain",
        ],
        "nd2nz": {
            "operand_a": {
                "ndNum": 1, "nValue": t.tile_m, "dValue": t.tile_k,
                "srcDValue": flow.loops.k,
                "dstNzC0Stride": _pad(t.tile_m, 16),
                "dstNzNStride": 1, "dstNzMatrixStride": 0,
            },
            "operand_b": {
                "ndNum": 1, "nValue": t.tile_k, "dValue": t.tile_n,
                "srcDValue": flow.loops.n,
                "dstNzC0Stride": _pad(t.tile_k, k0),
                "dstNzNStride": 1, "dstNzMatrixStride": 0,
            },
        },
        "epilogue_plan": [
            {
                "kind": op.kind,
                "unit": op.unit,
                "api": _epilogue_api(op.kind, op.unit),
            }
            for op in flow.epilogue
        ],
        "output_path": flow.output_path.value,
        "critical_stage": schedule.limiting_stage,
        "inner_steps_per_tile": t.n_inner_steps,
        "total_mac_steps": work.steps,
    }
    if requires_tail:
        hints["set_tail_call"] = (
            f"SetTail(tailM={hints['tail_sizes']['m'] or t.tile_m}, "
            f"tailN={hints['tail_sizes']['n'] or t.tile_n}, "
            f"tailK={hints['tail_sizes']['k'] or t.tile_k})"
        )
    return hints


def _epilogue_api(kind: str, unit: str) -> str:
    if unit == "fixpipe":
        return {
            "dequant": "Fixpipe<..., QuantMode_t::VDEQF16>(deqTensor)",
            "quant": "Fixpipe<..., QuantMode_t::VQF322B8_PRE>(quantTensor)",
            "relu": "FixpipeParams.reluEn = true",
            "bias": "SetFixpipeNz2ndFlag + bias table in L0C",
            "cast": "Fixpipe dst dtype conversion (free)",
        }.get(kind, f"Fixpipe inline {kind}")
    return {
        "dequant": "Mul(ubC, ubC, deqScaleUb, n)",
        "bias": "Add(ubC, ubC, biasUb, n)",
        "relu": "Relu(ubC, ubC, n)",
        "gelu": "Gelu(ubC, ubC, n)",
        "elemwise": "Vector op on UB tensors",
    }.get(kind, f"Vector {kind}")


# ---------------------------------------------------------------------------
# Elementwise (Vector-only) path
# ---------------------------------------------------------------------------


def _evaluate_elemwise(
    flow: AlgorithmFlow,
    arch: NpuArchitecture,
    options: EvalOptions,
) -> EvaluationReport:
    """GM -> UB -> Vector -> UB -> GM, with no Cube involvement."""
    itemsize = flow.a.dtype.size_bytes
    out_itemsize = flow.c.dtype.size_bytes
    n_in = flow.elemwise_inputs
    elements = flow.loops.m
    tile_elems = flow.tile.tile_m
    steps = _ceil_div(elements, tile_elems)
    plan = flow.buffers

    footprints, violations = _tier1_footprints_elemwise(
        flow, arch, itemsize, out_itemsize
    )
    in_bytes = float(elements * itemsize * n_in)
    out_bytes = float(elements * out_itemsize)
    ops_per_elem = sum(op.ops_per_element for op in flow.epilogue) or 1.0
    vector_bytes = (in_bytes + out_bytes) * ops_per_elem

    total_flops = int(elements * ops_per_elem)
    stages = (
        PipelineStage(
            "MTE2", _safe_div(in_bytes, steps * arch.pipe("MTE2_UB").bandwidth_bpc),
            in_bytes, plan.ub_depth >= 2, plan.ub_depth,
        ),
        PipelineStage(
            "VECTOR",
            _safe_div(vector_bytes, steps * arch.vector.bytes_per_cycle),
            vector_bytes, True, plan.ub_depth,
        ),
        PipelineStage(
            "MTE3", _safe_div(out_bytes, steps * arch.pipe("MTE3").bandwidth_bpc),
            out_bytes, plan.ub_depth >= 2, plan.ub_depth,
        ),
    )
    overlapped = [s for s in stages if s.overlapped]
    serialised = [s for s in stages if not s.overlapped]
    steady = max((s.cycles_per_step for s in overlapped), default=0.0)
    steady += sum(s.cycles_per_step for s in serialised)
    ranked = sorted(stages, key=lambda s: s.cycles_per_step, reverse=True)
    fill = arch.pipe("MTE2_UB").base_latency_cycles
    drain = arch.pipe("MTE3").base_latency_cycles
    total = steps * steady + fill + drain

    schedule = PipelineSchedule(
        stages=stages,
        steps=steps,
        steady_state_cycles_per_step=steady,
        fill_cycles=fill,
        drain_cycles=drain,
        total_cycles=total,
        serialized_cycles=steps * sum(s.cycles_per_step for s in stages)
        + steps * (fill + drain),
        ideal_cube_cycles=0.0,
        issued_cube_cycles=0.0,
        limiting_stage=_STAGE_TO_BOTTLENECK.get(ranked[0].name, ranked[0].name),
        secondary_stage=ranked[1].name,
        slack_pct=100.0
        * (1.0 - _safe_div(ranked[1].cycles_per_step, ranked[0].cycles_per_step)),
    )

    moved = in_bytes + out_bytes
    roofline = HierarchicalRoofline(
        total_flops=total_flops,
        peak_flops_per_cycle=arch.vector.bytes_per_cycle / max(1, out_itemsize),
        levels=(
            RooflineLevel(
                "GM_MTE2", moved, _safe_div(total_flops, moved),
                arch.pipe("MTE2_UB").bandwidth_bpc,
                _safe_div(total_flops, moved) * arch.pipe("MTE2_UB").bandwidth_bpc,
            ),
        ),
        binding_level="GM_MTE2",
        attainable_flops_per_cycle=_safe_div(total_flops, moved)
        * arch.pipe("MTE2_UB").bandwidth_bpc,
    )
    diagnostics = [
        Diagnostic(
            "MEMORY_BOUND_ELEMWISE",
            "Element-wise kernels are bandwidth bound by construction "
            f"({moved / 1024:.0f} KiB moved for {total_flops} ops).",
            "Fuse this op into the producer or consumer kernel so the UB "
            "round trip disappears.",
            impact="high",
        )
    ]
    us = arch.cycles_to_us(total)
    return EvaluationReport(
        flow_name=flow.name,
        arch_name=arch.name,
        summary=(
            f"{flow.name}: elementwise, {schedule.limiting_stage}, "
            f"{moved / max(us, 1e-9) / 1e3:.0f} GB/s single core"
        ),
        is_valid=not any(v.is_fatal for v in violations),
        violations=tuple(violations),
        diagnostics=tuple(diagnostics),
        sram_footprints=footprints,
        fractal_efficiency=1.0,
        roofline=roofline,
        schedule=schedule,
        t_min_cycles=steps * max(s.cycles_per_step for s in stages) + fill + drain,
        t_max_cycles=schedule.serialized_cycles,
        code_generation_hints={
            "block_dim": flow.cores,
            "tile_elements": tile_elems,
            "ub_depth": plan.ub_depth,
            "pseudocode": [
                f"for i in range({steps}):",
                f"  DataCopy(gm -> ub, {tile_elems} elems) x{n_in}  // MTE2",
                "  Vector op on UB",
                "  DataCopy(ub -> gm)  // MTE3",
            ],
            "requires_set_tail": elements % tile_elems != 0,
        },
        traffic={
            "mte2_bytes": in_bytes,
            "mte3_bytes": out_bytes,
            "vector_bytes": vector_bytes,
        },
        tile=flow.tile.to_dict(),
        timing_us={
            "total_us": us,
            "achieved_gbps_single_core": moved / max(us, 1e-9) / 1e3,
        },
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def evaluate(
    flow: AlgorithmFlow,
    arch: "str | NpuArchitecture" = "ascend910b",
    options: EvalOptions | None = None,
) -> EvaluationReport:
    """Run all three tiers and return a fully populated report.

    The call is side-effect free and allocation-light; it is designed to be
    invoked thousands of times by :mod:`ascend_pipesim.autotiler`.
    """
    arch = get_architecture(arch)
    options = options or EvalOptions()

    if flow.op_type is OpType.ELEMWISE:
        return _evaluate_elemwise(flow, arch, options)

    dtype = flow.compute_dtype
    if not arch.cube.supports(dtype):
        raise ValueError(
            f"{arch.name} Cube does not support {dtype.value}; "
            f"supported: {sorted(arch.cube.dtype_profiles)}"
        )
    profile = arch.cube.profile(dtype)
    k0 = profile.k0
    macs_per_cycle = float(profile.macs_per_cycle)
    itemsize = dtype.size_bytes
    out_itemsize = flow.c.dtype.size_bytes

    # -- Tier 1 ------------------------------------------------------------
    work = _inner_work(flow, k0, itemsize)
    footprints, violations = _tier1_footprints(
        flow, arch, work, k0, itemsize, out_itemsize
    )
    fractal_efficiency = _safe_div(flow.loops.macs, work.issued_macs)

    # -- traffic -----------------------------------------------------------
    traffic = dict(_outer_traffic(flow, itemsize, out_itemsize))
    traffic["mte1_bytes"] = work.mte1_bytes
    # The Fixpipe reads 4 B/element from L0C and writes out_itemsize to the
    # destination, so the wider of the two sides sets its service time.
    out_elements = _safe_div(traffic["fixpipe_out_bytes"], max(1, out_itemsize))
    if flow.k_position is KPosition.OUTER:
        out_elements = _safe_div(traffic["fixpipe_out_bytes"], 4)
    fixpipe_bytes = out_elements * max(4, out_itemsize)
    traffic["fixpipe_bytes"] = fixpipe_bytes

    vector_bytes = 0.0
    mte3_bytes = 0.0
    if flow.uses_unified_buffer:
        n_vec_ops = sum(
            op.ops_per_element for op in flow.vector_epilogue_ops()
        ) or 1.0
        vector_bytes = out_elements * (4 + out_itemsize) * n_vec_ops
        mte3_bytes = out_elements * out_itemsize
    traffic["vector_bytes"] = vector_bytes
    traffic["mte3_bytes"] = mte3_bytes

    # MTE2 engine efficiency: the L2 serves the engine in 32 B transactions,
    # so only a run that is not a multiple of 32 B wastes engine bandwidth.
    # Short-of-512 B DRAM bursts are charged to the DRAM roofline instead,
    # where they only bite for operands too large to stay resident in L2.
    txn = 32
    tile = flow.tile
    a_run = (tile.tile_m if flow.transpose_a else tile.tile_k) * itemsize
    b_run = (tile.tile_k if flow.transpose_b else tile.tile_n) * itemsize
    a_w, b_w = traffic["mte2_a_bytes"], traffic["mte2_b_bytes"]
    mte2_derate = _safe_div(
        a_w + b_w,
        a_w * (_pad(a_run, txn) / a_run) + b_w * (_pad(b_run, txn) / b_run),
    ) or 1.0
    mte2_bpc = arch.pipe("MTE2").bandwidth_bpc
    if options.model_hbm_contention and flow.cores > 1:
        hbm_bpc = arch.hbm_bandwidth_gbps * 1e9 / (arch.frequency_ghz * 1e9)
        mte2_bpc = min(mte2_bpc, hbm_bpc / flow.cores)

    dram_bytes = _dram_bytes(
        flow, arch, traffic, itemsize, out_itemsize, options
    )
    traffic["dram_bytes"] = dram_bytes
    traffic.pop("_c_drains", None)

    # -- Tier 2 ------------------------------------------------------------
    roofline = _tier2_roofline(
        flow, arch, traffic, work.mte1_bytes, fixpipe_bytes, dram_bytes,
        float(profile.flops_per_cycle), mte2_bpc, options,
    )

    # -- Tier 3 ------------------------------------------------------------
    schedule = _tier3_schedule(
        flow, arch, work, traffic, fixpipe_bytes, vector_bytes, mte3_bytes,
        macs_per_cycle, mte2_bpc, mte2_derate,
    )

    t_min = (
        schedule.steps * max(s.cycles_per_step for s in schedule.stages)
        + schedule.fill_cycles
        + schedule.drain_cycles
    )
    t_max = schedule.serialized_cycles

    diagnostics = _diagnostics(
        flow, arch, schedule, roofline, footprints, fractal_efficiency, options
    )
    hints = (
        _code_generation_hints(
            flow, arch, work, footprints, schedule, k0, itemsize, out_itemsize
        )
        if options.emit_hints
        else {}
    )

    is_valid = not any(v.is_fatal for v in violations)
    total_us = arch.cycles_to_us(schedule.total_cycles)
    achieved_tflops = _safe_div(flow.loops.flops, total_us * 1e-6) / 1e12
    timing = {
        "total_us": total_us,
        "t_min_us": arch.cycles_to_us(t_min),
        "t_max_us": arch.cycles_to_us(t_max),
        "achieved_tflops_single_core": achieved_tflops,
        "achieved_tflops_all_cores": achieved_tflops * flow.cores,
        "peak_tflops_single_core": arch.peak_tflops(dtype, all_cores=False),
    }
    summary = (
        f"{flow.name} on {arch.name}: "
        f"{'FEASIBLE' if is_valid else 'INFEASIBLE'}, "
        f"{schedule.cube_utilization_pct:.1f}% cube utilisation, "
        f"limited by {schedule.limiting_stage}, "
        f"{total_us:.1f} us single core"
    )

    return EvaluationReport(
        flow_name=flow.name,
        arch_name=arch.name,
        summary=summary,
        is_valid=is_valid,
        violations=tuple(violations),
        diagnostics=tuple(diagnostics),
        sram_footprints=footprints,
        fractal_efficiency=fractal_efficiency,
        roofline=roofline,
        schedule=schedule,
        t_min_cycles=t_min,
        t_max_cycles=t_max,
        code_generation_hints=hints,
        traffic=traffic,
        tile=flow.tile.to_dict(),
        timing_us=timing,
    )
