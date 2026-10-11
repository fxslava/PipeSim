"""Declarative operator schema for the PipeSim analytical engine.

An :class:`AlgorithmFlow` is a *complete, self-contained* description of what a
kernel computes **and** how it is staged through the DaVinci memory hierarchy.
It carries no hardware facts -- those live in :mod:`ascend_pipesim.topology` --
so the same flow can be re-evaluated against several manifests.

The canonical GEMM nest
-----------------------
PipeSim models a two-level tiling of ``C[M, N] = A[M, K] @ B[K, N]``:

* an **outer (L1) tile** ``(tile_m, tile_n, tile_k)`` staged GM -> L1 by MTE2;
* an **inner (L0) subtile** ``(sub_m, sub_n, sub_k)`` staged L1 -> L0A/L0B by
  MTE1 and consumed by the Cube engine.

``loop_order`` selects which spatial dimension is outermost, and
``k_position`` selects whether the reduction loop sits innermost (accumulate in
place in L0C -- the normal Ascend pattern) or outermost (re-materialise the
accumulator, which costs extra Fixpipe traffic but unlocks L1 reuse of the
stationary operand).

Example
-------
>>> flow = AlgorithmFlow.matmul_dequant(
...     m=2048, n=2048, k=4096,
...     tile=(128, 128, 64), subtile=(64, 64, 32),
...     in_dtype="int8", out_dtype="fp16",
... )
>>> flow.tile.n_tiles_m, flow.tile.n_tiles_n
(16, 16)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Sequence

from .topology import DType

__all__ = [
    "OpType",
    "Layout",
    "LoopOrder",
    "KPosition",
    "OutputPath",
    "TensorSpec",
    "LoopNest",
    "TileConfig",
    "BufferPlan",
    "EpilogueOp",
    "SimtWorkload",
    "ExecutionMode",
    "ExecutionGraph",
    "AlgorithmFlow",
    "FlowValidationError",
]


class FlowValidationError(ValueError):
    """Raised when an :class:`AlgorithmFlow` is structurally inconsistent.

    This covers errors that make the description meaningless (a zero tile, a
    subtile larger than its tile).  Hardware *capacity* violations are not
    errors here -- they are Tier-1 findings reported by the evaluator.
    """


@dataclass(frozen=True)
class SimtWorkload:
    """Warp prefill/packing work fused with a Vector or GEMV tile."""

    threads: int
    instructions_per_thread: int
    memory_ops_per_thread: int
    bytes_per_access: int
    stride_bytes: int
    working_set_bytes: int
    branch_efficiency: float = 1.0
    l2_hit_rate: float = 0.75
    per_outer_tile: bool = True
    registers_per_thread: int = 32

    def __post_init__(self) -> None:
        if (
            min(
                self.threads,
                self.instructions_per_thread,
                self.memory_ops_per_thread,
                self.bytes_per_access,
                self.stride_bytes,
                self.working_set_bytes,
            )
            <= 0
        ):
            raise FlowValidationError(
                "SIMT workload counts and byte sizes must be positive"
            )
        if not 0 < self.branch_efficiency <= 1:
            raise FlowValidationError("branch_efficiency must be in (0, 1]")
        if self.registers_per_thread <= 0:
            raise FlowValidationError("registers_per_thread must be positive")
        if not 0 <= self.l2_hit_rate <= 1:
            raise FlowValidationError("l2_hit_rate must be in [0, 1]")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "threads": self.threads,
            "instructions_per_thread": self.instructions_per_thread,
            "memory_ops_per_thread": self.memory_ops_per_thread,
            "bytes_per_access": self.bytes_per_access,
            "stride_bytes": self.stride_bytes,
            "working_set_bytes": self.working_set_bytes,
            "branch_efficiency": self.branch_efficiency,
            "l2_hit_rate": self.l2_hit_rate,
            "per_outer_tile": self.per_outer_tile,
            "registers_per_thread": self.registers_per_thread,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SimtWorkload":
        return cls(**data)


class OpType(str, Enum):
    """The computational shape of the kernel."""

    GEMM = "gemm"
    """Plain ``C = A @ B``, drained L0C -> GM by Fixpipe."""

    GEMM_DEQUANT = "gemm_dequant"
    """Quantised GEMM with an inline dequantisation epilogue."""

    BATCH_GEMM = "batch_gemm"
    """``batch`` independent GEMMs sharing one tiling strategy."""

    ELEMWISE = "elemwise"
    """Vector-only kernel: GM -> UB -> (Vector) -> UB -> GM."""

    VECTOR_GEMV = "vector_gemv"
    """Vector-core GEMV: weight rows and one input vector, reduced in UB."""


class ExecutionMode(str, Enum):
    STANDALONE_CUBE = "standalone_cube"
    MIX_SPLIT_M = "mix_split_m"
    MIX_SPLIT_N = "mix_split_n"
    HYBRID_SIMT_SIMD = "hybrid_simt_simd"


@dataclass(frozen=True)
class ExecutionGraph:
    """Lowered graph topology; ring slots are producer/consumer credits."""

    mode: ExecutionMode = ExecutionMode.STANDALONE_CUBE
    ring_slots: int = 1
    ring_space: str = "UB"
    vector_consumers: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", ExecutionMode(self.mode))
        object.__setattr__(self, "ring_space", self.ring_space.upper())
        if self.ring_slots < 1 or self.vector_consumers < 1:
            raise FlowValidationError(
                "ring_slots and vector_consumers must be positive"
            )
        if self.ring_space not in ("UB", "GM"):
            raise FlowValidationError("ring_space must be UB or GM")
        if self.mode in (ExecutionMode.MIX_SPLIT_M, ExecutionMode.MIX_SPLIT_N):
            if self.vector_consumers != 2:
                raise FlowValidationError("MIX mode requires two Vector consumers")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "ring_slots": self.ring_slots,
            "ring_space": self.ring_space,
            "vector_consumers": self.vector_consumers,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutionGraph":
        return cls(**data)


class Layout(str, Enum):
    """Tensor layout in memory."""

    ND = "ND"
    """Row-major n-dimensional, the normal layout of a GM tensor."""

    NZ = "NZ"
    """Fractal ``zN``/``nZ`` layout required by L1/L0 operands."""


class LoopOrder(str, Enum):
    """Which spatial dimension carries the outermost tile loop.

    ``M_FIRST`` (``for mo: for no: ...``) sweeps all of ``B`` for each row
    panel of ``A``, so ``B`` is the operand that must be captured by L2 to
    avoid repeated DRAM reads.  ``N_FIRST`` is the mirror image.
    """

    M_FIRST = "M_first"
    N_FIRST = "N_first"


class KPosition(str, Enum):
    """Where the reduction loop sits relative to the spatial tile loops."""

    INNER = "inner"
    """``for mo: for no: for ko:`` -- accumulate in L0C, drain C once."""

    OUTER = "outer"
    """``for ko: for mo: for no:`` -- C is re-materialised every k step."""


class OutputPath(str, Enum):
    """How the L0C accumulator reaches global memory."""

    FIXPIPE_GM = "fixpipe_gm"
    """L0C -> GM directly, using the Fixpipe quant/dequant hardware."""

    FIXPIPE_UB_MTE3 = "fixpipe_ub_mte3"
    """L0C -> UB (Fixpipe) -> Vector epilogue -> GM (MTE3)."""


# ---------------------------------------------------------------------------
# Leaf descriptors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorSpec:
    """A named tensor with a shape, an element type and a home memory space."""

    name: str
    shape: tuple[int, ...]
    dtype: DType
    space: str = "GM"
    layout: Layout = Layout.ND
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(int(d) for d in self.shape))
        object.__setattr__(self, "dtype", DType.parse(self.dtype))
        object.__setattr__(self, "layout", Layout(self.layout))
        if any(d <= 0 for d in self.shape):
            raise FlowValidationError(
                f"tensor {self.name!r}: every dimension must be positive, "
                f"got {self.shape}"
            )

    @property
    def num_elements(self) -> int:
        return math.prod(self.shape)

    @property
    def itemsize(self) -> float:
        return self.dtype.size_bytes

    @property
    def nbytes(self) -> float:
        return self.num_elements * self.itemsize

    @property
    def inner_stride_bytes(self) -> float:
        """Contiguous run length of the fastest-varying axis, in bytes."""
        return self.shape[-1] * self.itemsize

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype.value,
            "space": self.space,
            "layout": self.layout.value,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TensorSpec":
        return cls(
            name=data["name"],
            shape=tuple(data["shape"]),
            dtype=DType.parse(data["dtype"]),
            space=data.get("space", "GM"),
            layout=Layout(data.get("layout", "ND")),
            description=data.get("description", ""),
        )


@dataclass(frozen=True)
class LoopNest:
    """Problem extents of the GEMM iteration space."""

    m: int
    n: int
    k: int
    batch: int = 1

    def __post_init__(self) -> None:
        for name in ("m", "n", "k", "batch"):
            value = getattr(self, name)
            if int(value) <= 0:
                raise FlowValidationError(
                    f"loop extent {name}={value} must be positive"
                )
            object.__setattr__(self, name, int(value))

    @property
    def macs(self) -> int:
        return self.batch * self.m * self.n * self.k

    @property
    def flops(self) -> int:
        return 2 * self.macs

    def to_dict(self) -> Dict[str, Any]:
        return {"m": self.m, "n": self.n, "k": self.k, "batch": self.batch}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LoopNest":
        return cls(m=data["m"], n=data["n"], k=data["k"], batch=data.get("batch", 1))


def _ceil_div(a: int, b: int) -> int:
    return -(-int(a) // int(b))


def _tail(extent: int, tile: int) -> int:
    """Size of the final, possibly ragged tile along one axis."""
    rem = extent % tile
    return rem if rem else min(tile, extent)


@dataclass(frozen=True)
class TileConfig:
    """Two-level tiling: an L1 outer tile and an L0 inner subtile."""

    tile_m: int
    tile_n: int
    tile_k: int
    sub_m: int
    sub_n: int
    sub_k: int
    #: Problem extents, filled in by :class:`AlgorithmFlow`; needed for tails.
    loops: LoopNest | None = None

    def __post_init__(self) -> None:
        for name in ("tile_m", "tile_n", "tile_k", "sub_m", "sub_n", "sub_k"):
            value = int(getattr(self, name))
            if value <= 0:
                raise FlowValidationError(f"{name}={value} must be positive")
            object.__setattr__(self, name, value)
        for sub, tile in (
            ("sub_m", "tile_m"),
            ("sub_n", "tile_n"),
            ("sub_k", "tile_k"),
        ):
            if getattr(self, sub) > getattr(self, tile):
                raise FlowValidationError(
                    f"{sub}={getattr(self, sub)} exceeds {tile}={getattr(self, tile)}"
                )

    def bind(self, loops: LoopNest) -> "TileConfig":
        """Attach problem extents so tail sizes can be computed."""
        clipped = replace(
            self,
            tile_m=min(self.tile_m, loops.m),
            tile_n=min(self.tile_n, loops.n),
            tile_k=min(self.tile_k, loops.k),
            loops=loops,
        )
        return replace(
            clipped,
            sub_m=min(clipped.sub_m, clipped.tile_m),
            sub_n=min(clipped.sub_n, clipped.tile_n),
            sub_k=min(clipped.sub_k, clipped.tile_k),
        )

    def _loops(self) -> LoopNest:
        if self.loops is None:
            raise FlowValidationError(
                "TileConfig is not bound to a LoopNest; call .bind(loops) "
                "or build it through AlgorithmFlow"
            )
        return self.loops

    # -- outer tile counts -------------------------------------------------
    @property
    def n_tiles_m(self) -> int:
        return _ceil_div(self._loops().m, self.tile_m)

    @property
    def n_tiles_n(self) -> int:
        return _ceil_div(self._loops().n, self.tile_n)

    @property
    def n_tiles_k(self) -> int:
        return _ceil_div(self._loops().k, self.tile_k)

    @property
    def n_outer_tiles(self) -> int:
        return self.n_tiles_m * self.n_tiles_n * self.n_tiles_k

    # -- inner subtile counts ---------------------------------------------
    @property
    def n_subs_m(self) -> int:
        return _ceil_div(self.tile_m, self.sub_m)

    @property
    def n_subs_n(self) -> int:
        return _ceil_div(self.tile_n, self.sub_n)

    @property
    def n_subs_k(self) -> int:
        return _ceil_div(self.tile_k, self.sub_k)

    @property
    def n_inner_steps(self) -> int:
        """MAC steps inside one outer tile."""
        return self.n_subs_m * self.n_subs_n * self.n_subs_k

    # -- tails -------------------------------------------------------------
    @property
    def tail_m(self) -> int:
        return _tail(self._loops().m, self.tile_m)

    @property
    def tail_n(self) -> int:
        return _tail(self._loops().n, self.tile_n)

    @property
    def tail_k(self) -> int:
        return _tail(self._loops().k, self.tile_k)

    def has_tail(self) -> bool:
        loops = self._loops()
        return (
            loops.m % self.tile_m != 0
            or loops.n % self.tile_n != 0
            or loops.k % self.tile_k != 0
        )

    def has_sub_tail(self) -> bool:
        return (
            self.tile_m % self.sub_m != 0
            or self.tile_n % self.sub_n != 0
            or self.tile_k % self.sub_k != 0
        )

    def as_tuple(self) -> tuple[int, int, int, int, int, int]:
        return (
            self.tile_m,
            self.tile_n,
            self.tile_k,
            self.sub_m,
            self.sub_n,
            self.sub_k,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tile_m": self.tile_m,
            "tile_n": self.tile_n,
            "tile_k": self.tile_k,
            "sub_m": self.sub_m,
            "sub_n": self.sub_n,
            "sub_k": self.sub_k,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TileConfig":
        return cls(
            tile_m=data["tile_m"],
            tile_n=data["tile_n"],
            tile_k=data["tile_k"],
            sub_m=data["sub_m"],
            sub_n=data["sub_n"],
            sub_k=data["sub_k"],
        )


@dataclass(frozen=True)
class BufferPlan:
    """Multi-buffering depth per pool.

    Depth ``1`` means a single buffer (every stage serialises against its
    producer); depth ``2`` is the classic ping-pong that lets a transfer
    overlap the compute consuming the previous tile.  Depth ``>2`` buys
    tolerance against jittery GM latency at a linear SRAM cost.
    """

    l1_depth: int = 2
    l0_depth: int = 2
    l0c_depth: int = 1
    ub_depth: int = 2

    def __post_init__(self) -> None:
        for name in ("l1_depth", "l0_depth", "l0c_depth", "ub_depth"):
            value = int(getattr(self, name))
            if value < 1:
                raise FlowValidationError(f"{name}={value} must be >= 1")
            object.__setattr__(self, name, value)

    @classmethod
    def single(cls) -> "BufferPlan":
        return cls(l1_depth=1, l0_depth=1, l0c_depth=1, ub_depth=1)

    @classmethod
    def double(cls) -> "BufferPlan":
        return cls(l1_depth=2, l0_depth=2, l0c_depth=1, ub_depth=2)

    @classmethod
    def triple(cls) -> "BufferPlan":
        return cls(l1_depth=3, l0_depth=2, l0c_depth=2, ub_depth=2)

    @classmethod
    def named(cls, name: str) -> "BufferPlan":
        table = {
            "single": cls.single,
            "none": cls.single,
            "1": cls.single,
            "double": cls.double,
            "2": cls.double,
            "triple": cls.triple,
            "3": cls.triple,
        }
        key = str(name).strip().lower()
        if key not in table:
            raise FlowValidationError(
                f"unknown buffering scheme {name!r}; "
                f"expected one of {sorted(set(table))}"
            )
        return table[key]()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "l1_depth": self.l1_depth,
            "l0_depth": self.l0_depth,
            "l0c_depth": self.l0c_depth,
            "ub_depth": self.ub_depth,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BufferPlan":
        return cls(**{k: int(v) for k, v in data.items()})


@dataclass(frozen=True)
class EpilogueOp:
    """One post-accumulation operation applied to the L0C result.

    ``unit="fixpipe"`` ops ride the Fixpipe drain for free (the hardware has
    dedicated dequant/quant/ReLU stages).  ``unit="vector"`` ops force the
    result through the UB and cost real Vector cycles plus an MTE3 store.
    """

    kind: str
    unit: str = "fixpipe"
    operand: TensorSpec | None = None
    ops_per_element: float = 1.0

    FIXPIPE_CAPABLE = frozenset({"dequant", "quant", "relu", "bias", "cast"})

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", str(self.kind).strip().lower())
        object.__setattr__(self, "unit", str(self.unit).strip().lower())
        if self.unit not in ("fixpipe", "vector"):
            raise FlowValidationError(
                f"epilogue {self.kind!r}: unit must be 'fixpipe' or 'vector'"
            )

    @property
    def runs_on_fixpipe(self) -> bool:
        return self.unit == "fixpipe"

    def is_hardware_supported(self) -> bool:
        """True when the Fixpipe can actually absorb this op inline."""
        return self.kind in self.FIXPIPE_CAPABLE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "unit": self.unit,
            "operand": self.operand.to_dict() if self.operand else None,
            "ops_per_element": self.ops_per_element,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EpilogueOp":
        operand = data.get("operand")
        return cls(
            kind=data["kind"],
            unit=data.get("unit", "fixpipe"),
            operand=TensorSpec.from_dict(operand) if operand else None,
            ops_per_element=float(data.get("ops_per_element", 1.0)),
        )


# ---------------------------------------------------------------------------
# The flow
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlgorithmFlow:
    """A tiled operator ready for analytical evaluation."""

    name: str
    op_type: OpType
    loops: LoopNest
    tile: TileConfig
    a: TensorSpec
    b: TensorSpec
    c: TensorSpec
    buffers: BufferPlan = field(default_factory=BufferPlan.double)
    loop_order: LoopOrder = LoopOrder.M_FIRST
    k_position: KPosition = KPosition.INNER
    output_path: OutputPath = OutputPath.FIXPIPE_GM
    epilogue: tuple[EpilogueOp, ...] = ()
    transpose_a: bool = False
    transpose_b: bool = False
    #: Number of AI Cores the problem is split across (affects HBM share).
    cores: int = 1
    notes: str = ""
    simt: SimtWorkload | None = None
    execution_graph: ExecutionGraph = field(default_factory=ExecutionGraph)

    def __post_init__(self) -> None:
        object.__setattr__(self, "op_type", OpType(self.op_type))
        object.__setattr__(self, "loop_order", LoopOrder(self.loop_order))
        object.__setattr__(self, "k_position", KPosition(self.k_position))
        object.__setattr__(self, "output_path", OutputPath(self.output_path))
        object.__setattr__(self, "epilogue", tuple(self.epilogue))
        object.__setattr__(self, "tile", self.tile.bind(self.loops))
        if (
            self.execution_graph.mode is ExecutionMode.HYBRID_SIMT_SIMD
            and self.simt is None
        ):
            raise FlowValidationError("hybrid_simt_simd requires a SIMT workload")
        if self.execution_graph.mode in (
            ExecutionMode.MIX_SPLIT_M,
            ExecutionMode.MIX_SPLIT_N,
        ):
            if self.op_type in (OpType.ELEMWISE, OpType.VECTOR_GEMV):
                raise FlowValidationError("MIX split modes require a Cube operator")
        if self.execution_graph.mode is ExecutionMode.HYBRID_SIMT_SIMD:
            if self.op_type is not OpType.VECTOR_GEMV:
                raise FlowValidationError("hybrid_simt_simd requires vector_gemv")
        if self.cores < 1:
            raise FlowValidationError(f"cores={self.cores} must be >= 1")
        if self.op_type not in (OpType.ELEMWISE, OpType.VECTOR_GEMV):
            self._validate_gemm_shapes()

    def _validate_gemm_shapes(self) -> None:
        expect_a = (
            (self.loops.k, self.loops.m)
            if self.transpose_a
            else (self.loops.m, self.loops.k)
        )
        expect_b = (
            (self.loops.n, self.loops.k)
            if self.transpose_b
            else (self.loops.k, self.loops.n)
        )
        if self.a.shape[-2:] != expect_a:
            raise FlowValidationError(
                f"{self.name}: operand A has shape {self.a.shape}, expected "
                f"trailing dims {expect_a}"
            )
        if self.b.shape[-2:] != expect_b:
            raise FlowValidationError(
                f"{self.name}: operand B has shape {self.b.shape}, expected "
                f"trailing dims {expect_b}"
            )
        if self.c.shape[-2:] != (self.loops.m, self.loops.n):
            raise FlowValidationError(
                f"{self.name}: output C has shape {self.c.shape}, expected "
                f"trailing dims {(self.loops.m, self.loops.n)}"
            )

    # -- derived -----------------------------------------------------------
    @property
    def compute_dtype(self) -> DType:
        """The dtype the Cube engine sees (A and B must agree)."""
        return self.a.dtype

    @property
    def is_quantized(self) -> bool:
        return self.compute_dtype.is_quantized

    @property
    def uses_unified_buffer(self) -> bool:
        return (
            self.output_path is OutputPath.FIXPIPE_UB_MTE3
            or self.op_type is OpType.ELEMWISE
            or self.op_type is OpType.VECTOR_GEMV
            or any(not op.runs_on_fixpipe for op in self.epilogue)
        )

    @property
    def stationary_operand(self) -> str:
        """Operand whose panel is re-read by the inner spatial loop."""
        return "A" if self.loop_order is LoopOrder.M_FIRST else "B"

    @property
    def streamed_operand(self) -> str:
        """Operand swept in full by each outer iteration (the L2 candidate)."""
        return "B" if self.loop_order is LoopOrder.M_FIRST else "A"

    def vector_epilogue_ops(self) -> tuple[EpilogueOp, ...]:
        return tuple(op for op in self.epilogue if not op.runs_on_fixpipe)

    def fixpipe_epilogue_ops(self) -> tuple[EpilogueOp, ...]:
        return tuple(op for op in self.epilogue if op.runs_on_fixpipe)

    def with_tile(self, tile: TileConfig) -> "AlgorithmFlow":
        return replace(self, tile=tile.bind(self.loops))

    def with_buffers(self, buffers: BufferPlan) -> "AlgorithmFlow":
        return replace(self, buffers=buffers)

    def evolve(self, **changes: Any) -> "AlgorithmFlow":
        return replace(self, **changes)

    # -- factories ---------------------------------------------------------
    @classmethod
    def gemm(
        cls,
        *,
        m: int,
        n: int,
        k: int,
        tile: Sequence[int],
        subtile: Sequence[int],
        dtype: "str | DType" = DType.FP16,
        out_dtype: "str | DType | None" = None,
        batch: int = 1,
        buffers: BufferPlan | None = None,
        loop_order: "str | LoopOrder" = LoopOrder.M_FIRST,
        k_position: "str | KPosition" = KPosition.INNER,
        output_path: "str | OutputPath" = OutputPath.FIXPIPE_GM,
        epilogue: Iterable[EpilogueOp] = (),
        transpose_a: bool = False,
        transpose_b: bool = False,
        cores: int = 1,
        name: str = "gemm",
        simt: SimtWorkload | None = None,
    ) -> "AlgorithmFlow":
        """Build a (batched) GEMM flow from scalar parameters."""
        in_dt = DType.parse(dtype)
        out_dt = DType.parse(out_dtype) if out_dtype is not None else in_dt
        loops = LoopNest(m=m, n=n, k=k, batch=batch)
        bshape: tuple[int, ...] = (batch,) if batch > 1 else ()
        a_shape = bshape + ((k, m) if transpose_a else (m, k))
        b_shape = bshape + ((n, k) if transpose_b else (k, n))
        epilogue = tuple(epilogue)
        op_type = (
            OpType.BATCH_GEMM
            if batch > 1
            else OpType.GEMM_DEQUANT
            if any(op.kind == "dequant" for op in epilogue)
            else OpType.GEMM
        )
        if len(tuple(tile)) != 3 or len(tuple(subtile)) != 3:
            raise FlowValidationError(
                "tile and subtile must each be 3-tuples (M, N, K)"
            )
        return cls(
            name=name,
            op_type=op_type,
            loops=loops,
            tile=TileConfig(*tile, *subtile),
            a=TensorSpec("A", a_shape, in_dt),
            b=TensorSpec("B", b_shape, in_dt),
            c=TensorSpec("C", bshape + (m, n), out_dt),
            buffers=buffers or BufferPlan.double(),
            loop_order=LoopOrder(loop_order),
            k_position=KPosition(k_position),
            output_path=OutputPath(output_path),
            epilogue=epilogue,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            cores=cores,
            simt=simt,
        )

    @classmethod
    def matmul_dequant(
        cls,
        *,
        m: int,
        n: int,
        k: int,
        tile: Sequence[int],
        subtile: Sequence[int],
        in_dtype: "str | DType" = DType.INT8,
        out_dtype: "str | DType" = DType.FP16,
        per_channel: bool = True,
        bias: bool = False,
        relu: bool = False,
        on_vector: bool = False,
        name: str = "matmul_dequant",
        **kwargs: Any,
    ) -> "AlgorithmFlow":
        """Quantised matmul followed by a dequantisation epilogue.

        With ``on_vector=False`` (the default) the dequant rides the Fixpipe
        drain and costs no extra cycles; with ``on_vector=True`` the result is
        parked in the UB and scaled by the Vector unit, which adds a UB
        footprint, Vector cycles and an MTE3 store.
        """
        unit = "vector" if on_vector else "fixpipe"
        out_dt = DType.parse(out_dtype)
        scale_shape = (n,) if per_channel else (1,)
        ops: list[EpilogueOp] = [
            EpilogueOp(
                "dequant",
                unit,
                operand=TensorSpec("deq_scale", scale_shape, DType.FP32),
                ops_per_element=1.0,
            )
        ]
        if bias:
            ops.append(
                EpilogueOp(
                    "bias",
                    unit,
                    operand=TensorSpec("bias", (n,), DType.INT32),
                    ops_per_element=1.0,
                )
            )
        if relu:
            ops.append(EpilogueOp("relu", unit, ops_per_element=1.0))
        kwargs.setdefault(
            "output_path",
            OutputPath.FIXPIPE_UB_MTE3 if on_vector else OutputPath.FIXPIPE_GM,
        )
        return cls.gemm(
            m=m,
            n=n,
            k=k,
            tile=tile,
            subtile=subtile,
            dtype=in_dtype,
            out_dtype=out_dt,
            epilogue=ops,
            name=name,
            **kwargs,
        )

    @classmethod
    def elementwise(
        cls,
        *,
        elements: int,
        tile_elements: int,
        dtype: "str | DType" = DType.FP16,
        out_dtype: "str | DType | None" = None,
        n_inputs: int = 2,
        ops_per_element: float = 1.0,
        buffers: BufferPlan | None = None,
        name: str = "elemwise",
        simt: SimtWorkload | None = None,
    ) -> "AlgorithmFlow":
        """A Vector-only kernel streaming GM -> UB -> GM.

        ``n_inputs`` operands are loaded per output element; the loop nest is
        degenerate (``m=elements, n=1, k=1``) and only the UB/Vector/MTE3 path
        is exercised.
        """
        in_dt = DType.parse(dtype)
        out_dt = DType.parse(out_dtype) if out_dtype is not None else in_dt
        loops = LoopNest(m=elements, n=1, k=1)
        tile_elements = min(tile_elements, elements)
        return cls(
            name=name,
            op_type=OpType.ELEMWISE,
            simt=simt,
            loops=loops,
            tile=TileConfig(tile_elements, 1, 1, tile_elements, 1, 1),
            a=TensorSpec("X", (elements,), in_dt),
            b=TensorSpec("Y", (elements,), in_dt),
            c=TensorSpec("Z", (elements,), out_dt),
            buffers=buffers or BufferPlan.double(),
            output_path=OutputPath.FIXPIPE_UB_MTE3,
            epilogue=(
                EpilogueOp("elemwise", "vector", ops_per_element=ops_per_element),
            ),
            notes=f"n_inputs={n_inputs}",
        )

    @classmethod
    def vector_gemv(
        cls,
        *,
        rows: int,
        columns: int,
        tile_rows: int,
        dtype: "str | DType" = DType.FP16,
        out_dtype: "str | DType" = DType.FP32,
        buffers: BufferPlan | None = None,
        simt: SimtWorkload | None = None,
        name: str = "vector_gemv",
    ) -> "AlgorithmFlow":
        """Build a SIMD GEMV with an optional fused SIMT packing phase."""
        in_dt, out_dt = DType.parse(dtype), DType.parse(out_dtype)
        if not 1 <= tile_rows <= rows:
            raise FlowValidationError("tile_rows must be between 1 and rows")
        return cls(
            name=name,
            op_type=OpType.VECTOR_GEMV,
            loops=LoopNest(m=rows, n=1, k=columns),
            tile=TileConfig(tile_rows, 1, columns, tile_rows, 1, columns),
            a=TensorSpec("weights", (rows, columns), in_dt),
            b=TensorSpec("input", (columns,), in_dt),
            c=TensorSpec("output", (rows,), out_dt),
            buffers=buffers or BufferPlan.double(),
            simt=simt,
            output_path=OutputPath.FIXPIPE_UB_MTE3,
        )

    @property
    def elemwise_inputs(self) -> int:
        if self.op_type is not OpType.ELEMWISE:
            return 0
        for token in self.notes.split():
            if token.startswith("n_inputs="):
                return int(token.split("=", 1)[1])
        return 2

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "op_type": self.op_type.value,
            "loops": self.loops.to_dict(),
            "tile": self.tile.to_dict(),
            "a": self.a.to_dict(),
            "b": self.b.to_dict(),
            "c": self.c.to_dict(),
            "buffers": self.buffers.to_dict(),
            "loop_order": self.loop_order.value,
            "k_position": self.k_position.value,
            "output_path": self.output_path.value,
            "epilogue": [op.to_dict() for op in self.epilogue],
            "transpose_a": self.transpose_a,
            "transpose_b": self.transpose_b,
            "cores": self.cores,
            "notes": self.notes,
            "simt": self.simt.to_dict() if self.simt else None,
            "execution_graph": self.execution_graph.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AlgorithmFlow":
        return cls(
            name=data.get("name", "flow"),
            op_type=OpType(data["op_type"]),
            loops=LoopNest.from_dict(data["loops"]),
            tile=TileConfig.from_dict(data["tile"]),
            a=TensorSpec.from_dict(data["a"]),
            b=TensorSpec.from_dict(data["b"]),
            c=TensorSpec.from_dict(data["c"]),
            buffers=BufferPlan.from_dict(data.get("buffers", {})),
            loop_order=LoopOrder(data.get("loop_order", "M_first")),
            k_position=KPosition(data.get("k_position", "inner")),
            output_path=OutputPath(data.get("output_path", "fixpipe_gm")),
            epilogue=tuple(EpilogueOp.from_dict(op) for op in data.get("epilogue", [])),
            transpose_a=bool(data.get("transpose_a", False)),
            transpose_b=bool(data.get("transpose_b", False)),
            cores=int(data.get("cores", 1)),
            notes=data.get("notes", ""),
            simt=SimtWorkload.from_dict(data["simt"]) if data.get("simt") else None,
            execution_graph=ExecutionGraph.from_dict(data.get("execution_graph", {})),
        )

    def describe(self) -> str:
        """One-line human summary, handy in CLI tables and log lines."""
        t = self.tile
        return (
            f"{self.name}[{self.op_type.value}] "
            f"{self.loops.m}x{self.loops.n}x{self.loops.k}"
            f"{'' if self.loops.batch == 1 else f' b{self.loops.batch}'} "
            f"{self.compute_dtype.value}->{self.c.dtype.value} "
            f"tile=({t.tile_m},{t.tile_n},{t.tile_k}) "
            f"sub=({t.sub_m},{t.sub_n},{t.sub_k}) "
            f"L1d{self.buffers.l1_depth}/L0d{self.buffers.l0_depth} "
            f"{self.loop_order.value}"
        )
