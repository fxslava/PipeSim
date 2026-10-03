"""Fast combinatorial search over the DaVinci tile space.

The search is a plain nested enumeration, but ordered so that the cheapest,
most selective constraint runs outermost:

1. ``(tile_m, tile_n)`` fixes the L0C accumulator footprint.  L0C is only
   128 KiB of FP32, so this single test eliminates the large majority of the
   grid before anything else is computed.
2. ``tile_k`` then fixes the L1 footprint.
3. ``(sub_m, sub_n, sub_k)`` fixes L0A/L0B.
4. Whatever survives goes to the full three-tier :func:`~evaluator.evaluate`.

With NumPy installed the stage-1 and stage-2 filters run vectorised over the
whole ``(tile_m, tile_n, tile_k)`` grid at once; without it the pure-Python
path produces identical results, just a little slower.  Either way a sweep of
a few thousand candidates finishes in well under a second.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .dsl import AlgorithmFlow, BufferPlan, LoopOrder, OpType, TileConfig
from .evaluator import EvalOptions, EvaluationReport, evaluate
from .topology import DType, NpuArchitecture, get_architecture

try:  # optional acceleration
    import numpy as _np
except ImportError:  # pragma: no cover - exercised only without NumPy
    _np = None

__all__ = [
    "AutotileOptions",
    "TileCandidate",
    "AutotileResult",
    "autotile",
    "autotile_gemm",
    "divisors",
    "candidate_tile_values",
]

HAS_NUMPY = _np is not None


def _ceil_div(a: int, b: int) -> int:
    return -(-int(a) // int(b))


def _pad(value: int, gran: int) -> int:
    return _ceil_div(value, gran) * gran


def divisors(n: int) -> List[int]:
    """All positive divisors of ``n``, ascending."""
    small, large = [], []
    i = 1
    while i * i <= n:
        if n % i == 0:
            small.append(i)
            if i != n // i:
                large.append(n // i)
        i += 1
    return small + large[::-1]


def candidate_tile_values(
    extent: int,
    granularity: int,
    *,
    max_value: int,
    include_divisors: bool = True,
    max_values: int = 10,
) -> List[int]:
    """Plausible tile lengths along one axis.

    Combines a geometric ladder (16, 32, 48, 64, 96, 128, ...) with the exact
    divisors of ``extent``, because a divisor removes the ragged tail that
    would otherwise force runtime ``SetTail`` handling.
    """
    limit = min(int(extent), int(max_value))
    if limit < granularity:
        return [min(extent, granularity)]

    values = set()
    step = granularity
    while step <= limit:
        values.add(step)
        half = step + step // 2
        half -= half % granularity
        if granularity <= half <= limit:
            values.add(half)
        step *= 2
    if include_divisors:
        values.update(
            d for d in divisors(int(extent))
            if d % granularity == 0 and d <= limit
        )
    values.add(_pad(limit, granularity) if limit % granularity == 0 else
               (limit - limit % granularity) or granularity)

    ordered = sorted(v for v in values if v >= granularity)
    if len(ordered) <= max_values:
        return ordered
    # Thin out the middle, always keeping the extremes and the divisors.
    keep = {ordered[0], ordered[-1]}
    keep.update(v for v in ordered if extent % v == 0)
    remaining = [v for v in ordered if v not in keep]
    budget = max(0, max_values - len(keep))
    if budget and remaining:
        stride = max(1, len(remaining) // budget)
        keep.update(remaining[::stride][:budget])
    return sorted(keep)


def _subtile_values(tile: int, granularity: int, max_values: int) -> List[int]:
    """Subtile lengths that divide ``tile`` exactly, largest first.

    A subtile that does not divide its tile adds a second level of fractal
    padding for no benefit, so the search never considers one.
    """
    vals = [d for d in divisors(tile) if d % granularity == 0]
    if not vals:
        vals = [min(tile, granularity)]
    return sorted(vals, reverse=True)[:max_values]


# ---------------------------------------------------------------------------
# Options and results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AutotileOptions:
    """Search extent, pruning limits and the ranking weights."""

    topk: int = 3
    max_tile: int = 512
    max_evaluations: int = 8000
    max_tile_values: int = 9
    max_subtile_values: int = 3
    include_divisors: bool = True
    require_feasible: bool = True
    #: Report at most one (best) subtiling per outer tile, so the ranking
    #: shows genuinely different strategies instead of near-duplicates.
    unique_outer_tiles: bool = True
    #: Reject configurations whose ragged tail needs runtime SetTail support.
    forbid_tails: bool = False
    buffer_plans: Tuple[BufferPlan, ...] = (BufferPlan.double(),)
    loop_orders: Tuple[LoopOrder, ...] = (LoopOrder.M_FIRST, LoopOrder.N_FIRST)
    #: Relative score cost of needing a SetTail epilogue (0.02 == 2%).
    tail_complexity_weight: float = 0.02
    #: Score points removed at 100% SRAM occupancy; a tiebreak, not a driver.
    footprint_weight: float = 1.0
    time_budget_s: Optional[float] = None
    eval_options: EvalOptions = field(default_factory=EvalOptions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topk": self.topk,
            "max_tile": self.max_tile,
            "max_evaluations": self.max_evaluations,
            "max_tile_values": self.max_tile_values,
            "max_subtile_values": self.max_subtile_values,
            "require_feasible": self.require_feasible,
            "unique_outer_tiles": self.unique_outer_tiles,
            "forbid_tails": self.forbid_tails,
            "loop_orders": [o.value for o in self.loop_orders],
            "buffer_plans": [p.to_dict() for p in self.buffer_plans],
            "tail_complexity_weight": self.tail_complexity_weight,
            "footprint_weight": self.footprint_weight,
            "time_budget_s": self.time_budget_s,
        }


@dataclass(frozen=True)
class TileCandidate:
    """One scored point of the search space."""

    flow: AlgorithmFlow
    report: EvaluationReport
    score: float
    rank: int = 0

    @property
    def tile(self) -> TileConfig:
        return self.flow.tile

    @property
    def cube_utilization_pct(self) -> float:
        return self.report.cube_utilization_pct

    @property
    def requires_set_tail(self) -> bool:
        return bool(
            self.report.code_generation_hints.get("requires_set_tail", False)
        )

    @property
    def peak_sram_occupancy_pct(self) -> float:
        return max(
            (fp.occupancy_pct for fp in self.report.sram_footprints.values()),
            default=0.0,
        )

    def to_dict(self, *, full_report: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "rank": self.rank,
            "score": round(self.score, 3),
            "tile": self.flow.tile.to_dict(),
            "buffers": self.flow.buffers.to_dict(),
            "loop_order": self.flow.loop_order.value,
            "cube_utilization_pct": round(self.cube_utilization_pct, 2),
            "limiting_stage": self.report.limiting_stage,
            "fractal_efficiency": round(self.report.fractal_efficiency, 4),
            "requires_set_tail": self.requires_set_tail,
            "peak_sram_occupancy_pct": round(self.peak_sram_occupancy_pct, 1),
            "total_us": round(self.report.timing_us["total_us"], 3),
            "achieved_tflops_single_core": round(
                self.report.timing_us["achieved_tflops_single_core"], 3
            ),
            "sram_footprints": {
                k: v.bytes_used
                for k, v in self.report.sram_footprints.items()
            },
            "is_valid": self.report.is_valid,
        }
        if full_report:
            payload["report"] = self.report.to_dict()
        return payload


@dataclass(frozen=True)
class AutotileResult:
    """Outcome of a sweep: the ranked survivors plus search statistics."""

    arch_name: str
    problem: Dict[str, Any]
    candidates: Tuple[TileCandidate, ...]
    evaluated: int
    pruned: int
    grid_size: int
    elapsed_s: float
    options: AutotileOptions
    truncated: bool = False

    @property
    def best(self) -> Optional[TileCandidate]:
        return self.candidates[0] if self.candidates else None

    def to_dict(self, *, full_report: bool = False) -> Dict[str, Any]:
        best = self.best
        return {
            "arch": self.arch_name,
            "problem": self.problem,
            "search": {
                "grid_size": self.grid_size,
                "evaluated": self.evaluated,
                "pruned": self.pruned,
                "elapsed_s": round(self.elapsed_s, 4),
                "us_per_evaluation": round(
                    1e6 * self.elapsed_s / self.evaluated, 1
                ) if self.evaluated else None,
                "truncated": self.truncated,
                "numpy_accelerated": HAS_NUMPY,
                "options": self.options.to_dict(),
            },
            "best": best.to_dict(full_report=full_report) if best else None,
            "candidates": [
                c.to_dict(full_report=full_report) for c in self.candidates
            ],
        }

    def table(self) -> str:
        """A compact fixed-width ranking, for terminals and logs."""
        header = (
            f"{'#':>2}  {'tile (M,N,K)':>16}  {'sub (m,n,k)':>14}  "
            f"{'order':>7}  {'util%':>6}  {'us':>9}  {'eta':>5}  "
            f"{'tail':>4}  {'limiter':<28}"
        )
        lines = [header, "-" * len(header)]
        for c in self.candidates:
            t = c.flow.tile
            lines.append(
                f"{c.rank:>2}  "
                f"{f'{t.tile_m},{t.tile_n},{t.tile_k}':>16}  "
                f"{f'{t.sub_m},{t.sub_n},{t.sub_k}':>14}  "
                f"{c.flow.loop_order.value:>7}  "
                f"{c.cube_utilization_pct:>6.1f}  "
                f"{c.report.timing_us['total_us']:>9.1f}  "
                f"{c.report.fractal_efficiency:>5.3f}  "
                f"{'yes' if c.requires_set_tail else 'no':>4}  "
                f"{c.report.limiting_stage:<28}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------


def _outer_grid(
    tile_ms: Sequence[int],
    tile_ns: Sequence[int],
    tile_ks: Sequence[int],
    *,
    k0: int,
    itemsize: int,
    l0c_capacity: int,
    l1_capacity: int,
    l1_depth: int,
    l0c_depth: int,
) -> Tuple[List[Tuple[int, int, int]], int]:
    """Enumerate ``(tile_m, tile_n, tile_k)`` triples that fit L0C and L1.

    Returns the survivors and the size of the grid before filtering.
    """
    grid_size = len(tile_ms) * len(tile_ns) * len(tile_ks)

    if _np is not None and grid_size > 64:
        tm = _np.asarray(tile_ms, dtype=_np.int64)
        tn = _np.asarray(tile_ns, dtype=_np.int64)
        tk = _np.asarray(tile_ks, dtype=_np.int64)
        pm = ((tm + 15) // 16) * 16
        pn = ((tn + 15) // 16) * 16
        pk = ((tk + k0 - 1) // k0) * k0
        # L0C depends only on (m, n): filter that plane first.
        acc = pm[:, None] * pn[None, :] * 4 * l0c_depth
        mn_ok = _np.argwhere(acc <= l0c_capacity)
        if mn_ok.size == 0:
            return [], grid_size
        pm_ok = pm[mn_ok[:, 0]]
        pn_ok = pn[mn_ok[:, 1]]
        l1 = (
            (pm_ok[:, None] * pk[None, :] + pk[None, :] * pn_ok[:, None])
            * itemsize * l1_depth
        )
        rows, cols = _np.nonzero(l1 <= l1_capacity)
        survivors = [
            (
                int(tm[mn_ok[r, 0]]),
                int(tn[mn_ok[r, 1]]),
                int(tk[c]),
            )
            for r, c in zip(rows.tolist(), cols.tolist())
        ]
        return survivors, grid_size

    survivors = []
    for m in tile_ms:
        pm = _pad(m, 16)
        for n in tile_ns:
            pn = _pad(n, 16)
            if pm * pn * 4 * l0c_depth > l0c_capacity:
                continue
            for k in tile_ks:
                pk = _pad(k, k0)
                if (pm * pk + pk * pn) * itemsize * l1_depth > l1_capacity:
                    continue
                survivors.append((m, n, k))
    return survivors, grid_size


def _promise(tile: Tuple[int, int, int]) -> Tuple[float, int, int]:
    """Cheap ordering heuristic: expected GM arithmetic intensity, descending.

    One outer tile loads ``tile_m*tile_k + tile_k*tile_n`` elements to perform
    ``tile_m*tile_n*tile_k`` MACs, so intensity scales with the harmonic mean
    ``1/(1/tile_m + 1/tile_n)``.  Searching in this order means a truncated
    sweep still covers the region where the answer lives.
    """
    m, n, k = tile
    return (-(m * n) / (m + n), -k, -(m * n))


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def _score(
    report: EvaluationReport, options: AutotileOptions, requires_tail: bool
) -> float:
    """Utilisation first, tail complexity second, footprint as a tiebreak."""
    util = report.cube_utilization_pct
    if requires_tail:
        util *= 1.0 - options.tail_complexity_weight
    peak_occupancy = max(
        (fp.occupancy_pct for fp in report.sram_footprints.values()),
        default=0.0,
    )
    return util - options.footprint_weight * peak_occupancy / 100.0


def autotile(
    template: AlgorithmFlow,
    arch: "str | NpuArchitecture" = "ascend910b",
    options: AutotileOptions | None = None,
) -> AutotileResult:
    """Search the tile space around ``template`` and rank the survivors.

    ``template`` supplies everything except the tiling: dtypes, epilogue,
    transposes, core count.  Its own ``tile`` is ignored except as a fallback
    when the search finds nothing feasible.
    """
    arch = get_architecture(arch)
    options = options or AutotileOptions()
    if template.op_type is OpType.ELEMWISE:
        raise ValueError(
            "autotile targets the GEMM family; element-wise kernels have a "
            "one-dimensional tile space -- sweep tile_elements directly"
        )

    started = time.perf_counter()
    dtype = template.compute_dtype
    profile = arch.cube.profile(dtype)
    k0 = profile.k0
    itemsize = dtype.size_bytes
    loops = template.loops

    tile_ms = candidate_tile_values(
        loops.m, 16, max_value=options.max_tile,
        include_divisors=options.include_divisors,
        max_values=options.max_tile_values,
    )
    tile_ns = candidate_tile_values(
        loops.n, 16, max_value=options.max_tile,
        include_divisors=options.include_divisors,
        max_values=options.max_tile_values,
    )
    tile_ks = candidate_tile_values(
        loops.k, k0, max_value=options.max_tile,
        include_divisors=options.include_divisors,
        max_values=options.max_tile_values,
    )

    l0c_cap = arch.pool("L0C").capacity_bytes
    l1_cap = arch.pool("L1").capacity_bytes
    l0a_cap = arch.pool("L0A").capacity_bytes
    l0b_cap = arch.pool("L0B").capacity_bytes

    candidates: List[TileCandidate] = []
    evaluated = 0
    pruned = 0
    grid_total = 0
    truncated = False
    deadline = (
        started + options.time_budget_s if options.time_budget_s else math.inf
    )
    # Ascend C hints cost about as much as the analysis itself and are only
    # wanted for the winners, so the sweep runs without them.
    sweep_options = replace(options.eval_options, emit_hints=False)

    for plan in options.buffer_plans:
        outer, grid_size = _outer_grid(
            tile_ms, tile_ns, tile_ks,
            k0=k0, itemsize=itemsize,
            l0c_capacity=l0c_cap, l1_capacity=l1_cap,
            l1_depth=plan.l1_depth, l0c_depth=plan.l0c_depth,
        )
        grid_total += grid_size
        pruned += grid_size - len(outer)
        outer.sort(key=_promise)

        for tile_m, tile_n, tile_k in outer:
            sub_ms = _subtile_values(tile_m, 16, options.max_subtile_values)
            sub_ns = _subtile_values(tile_n, 16, options.max_subtile_values)
            sub_ks = _subtile_values(tile_k, k0, options.max_subtile_values)
            for sub_m in sub_ms:
                for sub_k in sub_ks:
                    if _pad(sub_m, 16) * _pad(sub_k, k0) * itemsize \
                            * plan.l0_depth > l0a_cap:
                        pruned += len(sub_ns)
                        continue
                    for sub_n in sub_ns:
                        if _pad(sub_k, k0) * _pad(sub_n, 16) * itemsize \
                                * plan.l0_depth > l0b_cap:
                            pruned += 1
                            continue
                        for order in options.loop_orders:
                            if evaluated >= options.max_evaluations or \
                                    time.perf_counter() > deadline:
                                truncated = True
                                break
                            flow = template.evolve(
                                tile=TileConfig(
                                    tile_m, tile_n, tile_k,
                                    sub_m, sub_n, sub_k,
                                ),
                                buffers=plan,
                                loop_order=order,
                            )
                            report = evaluate(flow, arch, sweep_options)
                            evaluated += 1
                            if options.require_feasible and not report.is_valid:
                                continue
                            requires_tail = (
                                flow.tile.has_tail() or flow.tile.has_sub_tail()
                            )
                            if options.forbid_tails and requires_tail:
                                continue
                            candidates.append(
                                TileCandidate(
                                    flow=flow,
                                    report=report,
                                    score=_score(report, options, requires_tail),
                                )
                            )
                        if truncated:
                            break
                    if truncated:
                        break
                if truncated:
                    break
            if truncated:
                break
        if truncated:
            break

    candidates.sort(
        key=lambda c: (
            -c.score,
            c.report.timing_us["total_us"],
            c.peak_sram_occupancy_pct,
        )
    )
    if options.unique_outer_tiles:
        seen: set = set()
        deduped = []
        for cand in candidates:
            t = cand.flow.tile
            key = (t.tile_m, t.tile_n, t.tile_k)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(cand)
        candidates = deduped
    # Re-evaluate only the survivors with the caller's real options, so the
    # winners carry full Ascend C code-generation hints.
    ranked = tuple(
        replace(
            c,
            rank=i + 1,
            report=evaluate(c.flow, arch, options.eval_options),
        )
        for i, c in enumerate(candidates[: max(1, options.topk)])
    )

    return AutotileResult(
        arch_name=arch.name,
        problem={
            "op_type": template.op_type.value,
            "m": loops.m, "n": loops.n, "k": loops.k, "batch": loops.batch,
            "in_dtype": dtype.value,
            "out_dtype": template.c.dtype.value,
            "cores": template.cores,
            "epilogue": [op.kind for op in template.epilogue],
        },
        candidates=ranked,
        evaluated=evaluated,
        pruned=pruned,
        grid_size=grid_total,
        elapsed_s=time.perf_counter() - started,
        options=options,
        truncated=truncated,
    )


def autotile_gemm(
    *,
    m: int,
    n: int,
    k: int,
    arch: "str | NpuArchitecture" = "ascend910b",
    dtype: "str | DType" = DType.FP16,
    out_dtype: "str | DType | None" = None,
    batch: int = 1,
    cores: int = 1,
    dequant: bool = False,
    options: AutotileOptions | None = None,
    **flow_kwargs: Any,
) -> AutotileResult:
    """Convenience wrapper: build a template GEMM and search it.

    The seed tile is deliberately tiny (16x16x``k0``); it exists only so the
    template validates, and is replaced by every candidate the search tries.
    """
    arch = get_architecture(arch)
    k0 = arch.cube.profile(DType.parse(dtype)).k0
    seed = dict(tile=(16, 16, k0), subtile=(16, 16, k0))
    if dequant:
        template = AlgorithmFlow.matmul_dequant(
            m=m, n=n, k=k, in_dtype=dtype,
            out_dtype=out_dtype or DType.FP16,
            batch=batch, cores=cores, **seed, **flow_kwargs,
        )
    else:
        template = AlgorithmFlow.gemm(
            m=m, n=n, k=k, dtype=dtype, out_dtype=out_dtype,
            batch=batch, cores=cores, **seed, **flow_kwargs,
        )
    return autotile(template, arch, options)
