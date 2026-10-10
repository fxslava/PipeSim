"""Hardware topology manifests for DaVinci-class Ascend AI Cores.

This module provides a declarative, *analytical* description of the DaVinci
micro-architecture: disjoint on-chip SRAM pools, the fixed-function transfer
engines (MTE1/MTE2/MTE3/Fixpipe) that move data between them, and the Cube and
Vector compute units.

Design notes
------------
The DaVinci core is a *spatial* accelerator: there is no hardware cache. Every
byte consumed by the Cube engine must be explicitly staged by the programmer
along the chain::

    GM --MTE2--> L1 --MTE1--> L0A / L0B --Cube--> L0C --Fixpipe--> GM / UB
      \\                                                             ^
       `--MTE2--> UB --Vector--> UB --MTE3--> GM                     |
                   ^-------------------------------------------------'

Because the chain is explicit and each engine has a fixed bandwidth, the
steady-state throughput of a kernel is a *max-plus* function of the per-stage
service times. That is exactly what :mod:`ascend_pipesim.evaluator` computes.

Cube throughput convention
--------------------------
The Cube engine consumes 16x16x16 *fractals*. A single fractal MAC step is
``16*16*16 = 4096`` multiply-accumulate operations. The engine sustains **one
fractal per cycle** (``4096 MACs/cycle`` == ``8192 FLOP/cycle`` for FP16/BF16)
behind a **16 cycle pipeline latency** -- the two commonly quoted numbers
("16 cycles per fractal", "4096 ops/cycle") describe the latency and the
throughput of the same pipelined unit respectively. Everywhere in this package
``ops`` means MACs and ``FLOPs`` means ``2 * MACs``.

All manifests below are *analytical approximations* intended for relative
ranking of tiling strategies, not datasheet reproductions. Every field is
user-overridable -- see :meth:`NpuArchitecture.evolve` and
:func:`architecture_from_dict`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Dict, Mapping

__all__ = [
    "DType",
    "PipeKind",
    "MemoryPool",
    "PipeChannel",
    "FractalShape",
    "CubeDTypeProfile",
    "ComputeUnit",
    "NpuArchitecture",
    "list_architectures",
    "get_architecture",
    "register_architecture",
    "architecture_from_dict",
    "KiB",
    "MiB",
]

KiB = 1024
MiB = 1024 * 1024


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class DType(str, Enum):
    """Element types understood by the DaVinci datapath."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    MXFP8 = "mxfp8"
    HIF8 = "hif8"
    MXFP4 = "mxfp4"
    INT32 = "int32"
    INT16 = "int16"
    INT8 = "int8"
    UINT8 = "uint8"

    @property
    def size_bytes(self) -> float:
        return _DTYPE_SIZE[self]

    @property
    def is_float(self) -> bool:
        return self in (DType.FP32, DType.FP16, DType.BF16, DType.FP8,
                        DType.MXFP8, DType.HIF8, DType.MXFP4)

    @property
    def is_quantized(self) -> bool:
        return self in (DType.INT8, DType.UINT8, DType.FP8,
                        DType.MXFP8, DType.HIF8, DType.MXFP4)

    @classmethod
    def parse(cls, value: "str | DType") -> "DType":
        if isinstance(value, DType):
            return value
        key = str(value).strip().lower().replace("-", "").replace("_", "")
        key = _DTYPE_ALIASES.get(key, key)
        try:
            return cls(key)
        except ValueError as exc:
            raise ValueError(
                f"unknown dtype {value!r}; expected one of "
                f"{[d.value for d in cls]}"
            ) from exc


_DTYPE_SIZE: Dict[DType, float] = {
    DType.FP32: 4,
    DType.FP16: 2,
    DType.BF16: 2,
    DType.FP8: 1,
    DType.MXFP8: 1,
    DType.HIF8: 1,
    DType.MXFP4: 0.5,
    DType.INT32: 4,
    DType.INT16: 2,
    DType.INT8: 1,
    DType.UINT8: 1,
}

_DTYPE_ALIASES: Dict[str, str] = {
    "float32": "fp32", "f32": "fp32", "float": "fp32",
    "float16": "fp16", "f16": "fp16", "half": "fp16",
    "bfloat16": "bf16",
    "float8": "fp8", "f8": "fp8", "e4m3": "fp8", "e5m2": "fp8",
    "microscalingfp8": "mxfp8", "mx8": "mxfp8",
    "huaweifp8": "hif8", "hi8": "hif8",
    "microscalingfp4": "mxfp4", "mx4": "mxfp4",
    "i32": "int32", "s32": "int32",
    "i16": "int16", "s16": "int16",
    "i8": "int8", "s8": "int8",
    "u8": "uint8",
}


class PipeKind(str, Enum):
    """The named hardware queues an Ascend C kernel synchronises against."""

    MTE1 = "MTE1"          # L1 -> L0A/L0B (fractal load, LoadData)
    MTE2 = "MTE2"          # GM -> L1 / UB (DataCopy, Nd2Nz)
    MTE3 = "MTE3"          # UB -> GM (DataCopy)
    FIXPIPE = "FIXPIPE"    # L0C -> GM / UB (Fixpipe; quant/dequant/ReLU)
    CUBE = "CUBE"          # L0A x L0B -> L0C (Mmad)
    VECTOR = "VECTOR"      # UB -> UB (Add, Mul, Cast, ...)
    SCALAR = "SCALAR"      # address / loop bookkeeping


# ---------------------------------------------------------------------------
# Structural elements
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryPool:
    """A physically disjoint storage pool with static capacity.

    Attributes
    ----------
    name:
        Pool identifier, e.g. ``"L1"``, ``"L0A"``, ``"UB"``.
    capacity_bytes:
        Hard capacity. Exceeding it is a Tier-1 feasibility violation, not a
        slowdown -- there is no spill path in hardware.
    alignment_bytes:
        Minimum allocation granularity. Every buffer is rounded up to this.
    banks, bank_stride_bytes:
        Banked pools (notably the Unified Buffer) suffer conflicts when the
        access stride maps successive elements onto the same bank.
    ports:
        Number of concurrent readers/writers the pool can service.
    """

    name: str
    capacity_bytes: int
    alignment_bytes: int = 32
    banks: int = 1
    bank_stride_bytes: int = 32
    ports: int = 1
    description: str = ""

    def __post_init__(self) -> None:
        if self.capacity_bytes <= 0:
            raise ValueError(f"{self.name}: capacity must be positive")
        if self.alignment_bytes <= 0:
            raise ValueError(f"{self.name}: alignment must be positive")

    def aligned_size(self, nbytes: float) -> int:
        """Round ``nbytes`` up to the allocation granularity of the pool."""
        n = int(math.ceil(nbytes))
        a = self.alignment_bytes
        return ((n + a - 1) // a) * a

    def is_aligned(self, nbytes: float) -> bool:
        return int(nbytes) % self.alignment_bytes == 0

    def fits(self, nbytes: float) -> bool:
        return self.aligned_size(nbytes) <= self.capacity_bytes

    def occupancy_pct(self, nbytes: float) -> float:
        return 100.0 * self.aligned_size(nbytes) / self.capacity_bytes

    def bank_conflict_factor(self, inner_stride_bytes: int) -> float:
        """Slowdown multiplier for a strided access pattern.

        A stride that is a multiple of ``banks * bank_stride_bytes`` funnels
        every access into one bank, so the worst case costs ``banks`` times the
        conflict-free time.
        """
        if self.banks <= 1 or inner_stride_bytes <= 0:
            return 1.0
        period = self.banks * self.bank_stride_bytes
        if inner_stride_bytes % period == 0:
            return float(self.banks)
        distinct = period // math.gcd(inner_stride_bytes, period)
        return max(1.0, self.banks / min(self.banks, distinct))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "capacity_bytes": self.capacity_bytes,
            "alignment_bytes": self.alignment_bytes,
            "banks": self.banks,
            "bank_stride_bytes": self.bank_stride_bytes,
            "ports": self.ports,
            "description": self.description,
        }


@dataclass(frozen=True)
class PipeChannel:
    """A fixed-function transfer engine connecting two pools.

    The cost model is the classic affine latency/bandwidth form::

        cycles(n) = base_latency + n / (bandwidth_bpc * efficiency * derate)

    ``efficiency`` captures static derating (for example an ND->NZ reformat on
    the MTE2 path), while callers supply a dynamic ``derate`` for ragged or
    misaligned transfers.
    """

    name: str
    kind: PipeKind
    src: str
    dst: str
    bandwidth_bpc: float
    base_latency_cycles: float
    max_outstanding: int = 4
    efficiency: float = 1.0
    min_burst_bytes: int = 32
    description: str = ""

    def __post_init__(self) -> None:
        if self.bandwidth_bpc <= 0:
            raise ValueError(f"{self.name}: bandwidth must be positive")
        if not 0.0 < self.efficiency <= 1.0:
            raise ValueError(f"{self.name}: efficiency must be in (0, 1]")

    def effective_bandwidth(self, derate: float = 1.0) -> float:
        return self.bandwidth_bpc * self.efficiency * max(1e-6, derate)

    def transfer_cycles(
        self,
        nbytes: float,
        *,
        include_latency: bool = True,
        derate: float = 1.0,
    ) -> float:
        """Cycles needed to move ``nbytes`` across this channel."""
        if nbytes <= 0:
            return float(self.base_latency_cycles) if include_latency else 0.0
        payload = max(float(nbytes), float(self.min_burst_bytes))
        cycles = payload / self.effective_bandwidth(derate)
        if include_latency:
            cycles += self.base_latency_cycles
        return cycles

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "src": self.src,
            "dst": self.dst,
            "bandwidth_bpc": self.bandwidth_bpc,
            "base_latency_cycles": self.base_latency_cycles,
            "max_outstanding": self.max_outstanding,
            "efficiency": self.efficiency,
            "min_burst_bytes": self.min_burst_bytes,
            "description": self.description,
        }


@dataclass(frozen=True)
class FractalShape:
    """The ``(m0, k0, n0)`` quantum consumed by one Cube MAC issue."""

    m0: int = 16
    n0: int = 16
    k0: int = 16

    @property
    def macs(self) -> int:
        return self.m0 * self.n0 * self.k0

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.m0, self.k0, self.n0)


@dataclass(frozen=True)
class CubeDTypeProfile:
    """Per-dtype Cube behaviour.

    ``k0`` is the reduction depth of one fractal; it widens as the element type
    narrows (FP32 -> 8, FP16/BF16 -> 16, INT8/FP8 -> 32) because the MAC array
    is byte-oriented. ``macs_per_cycle`` scales accordingly.
    """

    dtype: DType
    k0: int
    macs_per_cycle: int
    accumulator: DType = DType.FP32

    @property
    def fractal(self) -> FractalShape:
        return FractalShape(m0=16, n0=16, k0=self.k0)

    @property
    def flops_per_cycle(self) -> int:
        return 2 * self.macs_per_cycle


@dataclass(frozen=True)
class ComputeUnit:
    """A compute engine (Cube or Vector) attached to the core."""

    name: str
    kind: PipeKind
    pipeline_latency_cycles: float = 16.0
    #: Cube only: per-dtype fractal shape and MAC throughput.
    dtype_profiles: Mapping[str, CubeDTypeProfile] = field(default_factory=dict)
    #: Vector only: sustained element-wise throughput out of the UB.
    bytes_per_cycle: float = 0.0
    issue_overhead_cycles: float = 0.0
    description: str = ""

    def profile(self, dtype: DType | str) -> CubeDTypeProfile:
        dt = DType.parse(dtype)
        try:
            return self.dtype_profiles[dt.value]
        except KeyError as exc:
            raise ValueError(
                f"{self.name} does not support dtype {dt.value!r}; "
                f"supported: {sorted(self.dtype_profiles)}"
            ) from exc

    def supports(self, dtype: DType | str) -> bool:
        return DType.parse(dtype).value in self.dtype_profiles

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "pipeline_latency_cycles": self.pipeline_latency_cycles,
            "bytes_per_cycle": self.bytes_per_cycle,
            "issue_overhead_cycles": self.issue_overhead_cycles,
            "dtype_profiles": {
                k: {
                    "dtype": v.dtype.value,
                    "k0": v.k0,
                    "macs_per_cycle": v.macs_per_cycle,
                    "accumulator": v.accumulator.value,
                }
                for k, v in self.dtype_profiles.items()
            },
            "description": self.description,
        }


@dataclass(frozen=True)
class NpuArchitecture:
    """A complete analytical manifest for one AI Core plus its die context."""

    name: str
    family: str
    frequency_ghz: float
    cube_cores: int
    vector_cores: int
    pools: Mapping[str, MemoryPool]
    pipes: Mapping[str, PipeChannel]
    cube: ComputeUnit
    vector: ComputeUnit
    hbm_bandwidth_gbps: float
    l2_capacity_bytes: int = 0
    #: False on fused Cube+Vector cores, where a Vector epilogue blocks Cube.
    decoupled_vector_core: bool = True
    notes: str = ""

    # -- lookups ----------------------------------------------------------
    def pool(self, name: str) -> MemoryPool:
        try:
            return self.pools[name.upper()]
        except KeyError as exc:
            raise KeyError(
                f"{self.name}: no memory pool {name!r}; have {sorted(self.pools)}"
            ) from exc

    def pipe(self, name: str) -> PipeChannel:
        try:
            return self.pipes[name.upper()]
        except KeyError as exc:
            raise KeyError(
                f"{self.name}: no pipe {name!r}; have {sorted(self.pipes)}"
            ) from exc

    def has_pool(self, name: str) -> bool:
        return name.upper() in self.pools

    # -- derived performance figures --------------------------------------
    def cube_flops_per_cycle(self, dtype: DType | str) -> int:
        return self.cube.profile(dtype).flops_per_cycle

    def peak_tflops(self, dtype: DType | str, *, all_cores: bool = True) -> float:
        """Peak dense Cube throughput in TFLOP/s."""
        per_core = self.cube_flops_per_cycle(dtype) * self.frequency_ghz * 1e9
        cores = self.cube_cores if all_cores else 1
        return per_core * cores / 1e12

    def gm_bandwidth_bpc_per_core(self) -> float:
        """HBM bytes/cycle available to one core when every core streams."""
        if self.cube_cores <= 0:
            return 0.0
        total_bpc = self.hbm_bandwidth_gbps * 1e9 / (self.frequency_ghz * 1e9)
        return total_bpc / self.cube_cores

    def cycles_to_us(self, cycles: float) -> float:
        return cycles / (self.frequency_ghz * 1e3)

    # -- mutation ---------------------------------------------------------
    def evolve(self, **changes: Any) -> "NpuArchitecture":
        """Return a copy with ``changes`` applied (for what-if studies)."""
        return replace(self, **changes)

    def with_pipe_override(self, pipe_name: str, **changes: Any) -> "NpuArchitecture":
        pipes = dict(self.pipes)
        pipes[pipe_name.upper()] = replace(self.pipe(pipe_name), **changes)
        return replace(self, pipes=pipes)

    def with_pool_override(self, pool_name: str, **changes: Any) -> "NpuArchitecture":
        pools = dict(self.pools)
        pools[pool_name.upper()] = replace(self.pool(pool_name), **changes)
        return replace(self, pools=pools)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "frequency_ghz": self.frequency_ghz,
            "cube_cores": self.cube_cores,
            "vector_cores": self.vector_cores,
            "hbm_bandwidth_gbps": self.hbm_bandwidth_gbps,
            "l2_capacity_bytes": self.l2_capacity_bytes,
            "decoupled_vector_core": self.decoupled_vector_core,
            "pools": {k: v.to_dict() for k, v in self.pools.items()},
            "pipes": {k: v.to_dict() for k, v in self.pipes.items()},
            "cube": self.cube.to_dict(),
            "vector": self.vector.to_dict(),
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Built-in manifests
# ---------------------------------------------------------------------------


def _standard_pools(
    *,
    l1_bytes: int,
    ub_bytes: int,
    l0ab_bytes: int = 64 * KiB,
    l0c_bytes: int = 128 * KiB,
    ub_banks: int = 8,
) -> Dict[str, MemoryPool]:
    """The canonical DaVinci pool set.

    Sizes follow the invariants shared by every DaVinci generation modelled
    here: L0A/L0B are 64 KiB fractal operand buffers, L0C is a 128 KiB FP32
    accumulator (4 B/element regardless of the FP16/BF16/INT8 input type), and
    the UB is 8-way banked with a 32 B bank stride.
    """
    return {
        "GM": MemoryPool(
            "GM",
            capacity_bytes=64 * 1024 * MiB,
            alignment_bytes=512,
            ports=2,
            description="Off-chip HBM/LPDDR; 512 B burst alignment.",
        ),
        "L1": MemoryPool(
            "L1",
            capacity_bytes=l1_bytes,
            alignment_bytes=32,
            ports=2,
            description="Cube-local scratchpad staging GM tiles before MTE1.",
        ),
        "L0A": MemoryPool(
            "L0A",
            capacity_bytes=l0ab_bytes,
            alignment_bytes=32,
            description="Left (A) fractal operand buffer, Zz layout.",
        ),
        "L0B": MemoryPool(
            "L0B",
            capacity_bytes=l0ab_bytes,
            alignment_bytes=32,
            description="Right (B) fractal operand buffer, Nz layout.",
        ),
        "L0C": MemoryPool(
            "L0C",
            capacity_bytes=l0c_bytes,
            alignment_bytes=32,
            description="FP32/INT32 accumulator; always 4 B per element.",
        ),
        "UB": MemoryPool(
            "UB",
            capacity_bytes=ub_bytes,
            alignment_bytes=32,
            banks=ub_banks,
            bank_stride_bytes=32,
            ports=2,
            description="Unified Buffer: Vector scratchpad, 8 banks x 32 B.",
        ),
    }


def _standard_pipes(
    *,
    mte2_bpc: float = 64.0,
    mte2_latency: float = 200.0,
    mte1_bpc: float = 64.0,
    mte1_latency: float = 20.0,
    fixpipe_bpc: float = 64.0,
    fixpipe_latency: float = 80.0,
    mte3_bpc: float = 64.0,
    mte3_latency: float = 150.0,
) -> Dict[str, PipeChannel]:
    """The canonical DaVinci transfer-engine set."""
    return {
        "MTE2": PipeChannel(
            "MTE2", PipeKind.MTE2, "GM", "L1",
            bandwidth_bpc=mte2_bpc, base_latency_cycles=mte2_latency,
            max_outstanding=8, min_burst_bytes=512,
            description="GM -> L1 load, optional ND->NZ reformat.",
        ),
        "MTE2_UB": PipeChannel(
            "MTE2_UB", PipeKind.MTE2, "GM", "UB",
            bandwidth_bpc=mte2_bpc, base_latency_cycles=mte2_latency,
            max_outstanding=8, min_burst_bytes=512,
            description="GM -> UB load for Vector-side operands.",
        ),
        "MTE1": PipeChannel(
            "MTE1", PipeKind.MTE1, "L1", "L0A",
            bandwidth_bpc=mte1_bpc, base_latency_cycles=mte1_latency,
            max_outstanding=4, min_burst_bytes=32,
            description="L1 -> L0A/L0B fractal load (LoadData2d/3d).",
        ),
        "FIXPIPE": PipeChannel(
            "FIXPIPE", PipeKind.FIXPIPE, "L0C", "GM",
            bandwidth_bpc=fixpipe_bpc, base_latency_cycles=fixpipe_latency,
            max_outstanding=4, min_burst_bytes=32,
            description="L0C -> GM/UB drain with inline quant/dequant/ReLU.",
        ),
        "MTE3": PipeChannel(
            "MTE3", PipeKind.MTE3, "UB", "GM",
            bandwidth_bpc=mte3_bpc, base_latency_cycles=mte3_latency,
            max_outstanding=4, min_burst_bytes=512,
            description="UB -> GM store.",
        ),
    }


def _cube_unit(
    *,
    base_macs_per_cycle: int = 4096,
    int8_speedup: float = 2.0,
    fp32_slowdown: float = 4.0,
    fp8: bool = False,
    pipeline_latency: float = 16.0,
) -> ComputeUnit:
    """Build a Cube manifest from the FP16 baseline.

    One fractal (``16x16x16`` == 4096 MACs) retires per cycle at the FP16
    baseline; narrower types widen ``k0`` and raise throughput proportionally.
    """
    profiles: Dict[str, CubeDTypeProfile] = {
        DType.FP16.value: CubeDTypeProfile(
            DType.FP16, k0=16, macs_per_cycle=base_macs_per_cycle
        ),
        DType.BF16.value: CubeDTypeProfile(
            DType.BF16, k0=16, macs_per_cycle=base_macs_per_cycle
        ),
        DType.INT8.value: CubeDTypeProfile(
            DType.INT8, k0=32,
            macs_per_cycle=int(base_macs_per_cycle * int8_speedup),
            accumulator=DType.INT32,
        ),
        DType.UINT8.value: CubeDTypeProfile(
            DType.UINT8, k0=32,
            macs_per_cycle=int(base_macs_per_cycle * int8_speedup),
            accumulator=DType.INT32,
        ),
        DType.FP32.value: CubeDTypeProfile(
            DType.FP32, k0=8,
            macs_per_cycle=int(base_macs_per_cycle / fp32_slowdown),
        ),
    }
    if fp8:
        profiles[DType.FP8.value] = CubeDTypeProfile(
            DType.FP8, k0=32,
            macs_per_cycle=int(base_macs_per_cycle * int8_speedup),
        )
    return ComputeUnit(
        name="Cube",
        kind=PipeKind.CUBE,
        pipeline_latency_cycles=pipeline_latency,
        dtype_profiles=profiles,
        issue_overhead_cycles=1.0,
        description="16x16x16 fractal MAC array; 1 fractal/cycle throughput.",
    )


def _vector_unit(bytes_per_cycle: float) -> ComputeUnit:
    return ComputeUnit(
        name="Vector",
        kind=PipeKind.VECTOR,
        pipeline_latency_cycles=8.0,
        bytes_per_cycle=bytes_per_cycle,
        issue_overhead_cycles=4.0,
        description="SIMD unit operating in place on the Unified Buffer.",
    )


def _ascend310p() -> NpuArchitecture:
    return NpuArchitecture(
        name="ascend310p",
        family="DaVinci v1 (Ascend 310P / Atlas 300I Pro)",
        frequency_ghz=1.08,
        cube_cores=8,
        vector_cores=8,
        pools=_standard_pools(l1_bytes=512 * KiB, ub_bytes=192 * KiB),
        # LPDDR-class memory: same burst width, materially higher latency.
        pipes=_standard_pipes(mte2_latency=280.0, mte3_latency=200.0),
        cube=_cube_unit(base_macs_per_cycle=4096),
        vector=_vector_unit(bytes_per_cycle=128.0),
        hbm_bandwidth_gbps=204.8,
        l2_capacity_bytes=8 * MiB,
        decoupled_vector_core=False,
        notes=(
            "Inference-oriented part. Fused Cube+Vector core: a Vector "
            "epilogue does not overlap Cube work on the same core."
        ),
    )


def _ascend910b() -> NpuArchitecture:
    return NpuArchitecture(
        name="ascend910b",
        family="DaVinci v2 (Ascend 910B / Atlas 800T A2)",
        frequency_ghz=1.80,
        cube_cores=24,
        vector_cores=48,
        pools=_standard_pools(l1_bytes=512 * KiB, ub_bytes=192 * KiB),
        pipes=_standard_pipes(),
        cube=_cube_unit(base_macs_per_cycle=4096),
        vector=_vector_unit(bytes_per_cycle=256.0),
        hbm_bandwidth_gbps=1600.0,
        l2_capacity_bytes=192 * MiB,
        decoupled_vector_core=True,
        notes=(
            "Reference manifest for this package. Decoupled AI Cube / AI "
            "Vector cores (2 Vector per Cube), so UB epilogues overlap Cube."
        ),
    )


def _ascend910c() -> NpuArchitecture:
    return NpuArchitecture(
        name="ascend910c",
        family="DaVinci v2+ (Ascend 910C, dual-die)",
        frequency_ghz=1.80,
        cube_cores=32,
        vector_cores=64,
        pools=_standard_pools(l1_bytes=512 * KiB, ub_bytes=256 * KiB),
        pipes=_standard_pipes(mte2_latency=180.0),
        cube=_cube_unit(base_macs_per_cycle=4096, fp8=True),
        vector=_vector_unit(bytes_per_cycle=256.0),
        hbm_bandwidth_gbps=3200.0,
        l2_capacity_bytes=384 * MiB,
        decoupled_vector_core=True,
        notes="Dual-die package; per-die figures doubled. 256 KiB UB.",
    )


def _ascend351x() -> NpuArchitecture:
    return NpuArchitecture(
        name="ascend351x",
        family="DaVinci v3 (Ascend 351x, forward-looking manifest)",
        frequency_ghz=2.00,
        cube_cores=32,
        vector_cores=64,
        pools=_standard_pools(l1_bytes=1 * MiB, ub_bytes=256 * KiB),
        pipes=_standard_pipes(
            mte2_bpc=128.0, mte2_latency=160.0,
            mte1_bpc=128.0, fixpipe_bpc=128.0, mte3_bpc=128.0,
        ),
        cube=_cube_unit(base_macs_per_cycle=8192, fp8=True),
        vector=_vector_unit(bytes_per_cycle=512.0),
        hbm_bandwidth_gbps=4800.0,
        l2_capacity_bytes=512 * MiB,
        decoupled_vector_core=True,
        notes=(
            "Speculative next-gen manifest for sensitivity studies. Treat the "
            "absolute numbers as unvalidated; use relative trends only."
        ),
    )


def _ascend950pr() -> NpuArchitecture:
    """Default 32 Cube / 64 Vector 950PR bin from the 2026 whitepaper."""
    pools = _standard_pools(l1_bytes=512 * KiB, ub_bytes=512 * KiB,
                            l0c_bytes=256 * KiB)
    pools["GM"] = replace(
        pools["GM"], capacity_bytes=128 * 1024 * MiB,
        description="Package-attached high-speed memory in a two-AI-Die UMA space.",
    )
    cube = _cube_unit(base_macs_per_cycle=4096, fp8=True)
    profiles = dict(cube.dtype_profiles)
    for dtype, k0, macs in (
        (DType.MXFP8, 32, 8192), (DType.HIF8, 32, 8192),
        (DType.MXFP4, 64, 16384),
    ):
        profiles[dtype.value] = CubeDTypeProfile(dtype, k0=k0,
                                                 macs_per_cycle=macs)
    return NpuArchitecture(
        name="ascend950pr", family="DaVinci v3 (Ascend 950PR)",
        # Approximation inferred from 432 TFLOP/s Cube FP16 / 32 cores.
        frequency_ghz=1.65, cube_cores=32, vector_cores=64,
        pools=pools,
        pipes=_standard_pipes(mte2_bpc=128.0, mte1_bpc=128.0,
                              fixpipe_bpc=128.0, mte3_bpc=128.0),
        cube=replace(cube, dtype_profiles=profiles),
        vector=_vector_unit(bytes_per_cycle=512.0),
        hbm_bandwidth_gbps=1600.0, l2_capacity_bytes=128 * MiB,
        notes="950PR default bin; 28/56, 112 GB and 112 MB L2 variants also exist. "
              "Frequency and local DMA rates are analytical estimates.",
    )


def _ascend950dt() -> NpuArchitecture:
    """Default full 36 Cube / 72 Vector 950DT bin."""
    pr = _ascend950pr()
    return pr.evolve(
        name="ascend950dt", family="DaVinci v3 (Ascend 950DT)",
        cube_cores=36, vector_cores=72,
        pools={**pr.pools, "GM": replace(pr.pool("GM"),
                                         capacity_bytes=144 * 1024 * MiB)},
        hbm_bandwidth_gbps=4000.0,
        notes="950DT default bin; 32/64 and 28/56 variants also exist. "
              "Frequency and local DMA rates are analytical estimates.",
    )


_REGISTRY: Dict[str, Callable[[], NpuArchitecture]] = {
    "ascend310p": _ascend310p,
    "ascend910b": _ascend910b,
    "ascend910c": _ascend910c,
    "ascend351x": _ascend351x,
    "ascend950pr": _ascend950pr,
    "ascend950dt": _ascend950dt,
}

_ALIASES: Dict[str, str] = {
    "310p": "ascend310p",
    "ascend310p3": "ascend310p",
    "910b": "ascend910b",
    "ascend910b2": "ascend910b",
    "ascend910b4": "ascend910b",
    "910c": "ascend910c",
    "351x": "ascend351x",
    "950pr": "ascend950pr",
    "950dt": "ascend950dt",
    "ascend950": "ascend950dt",
    "default": "ascend910b",
}


def list_architectures() -> list[str]:
    """Names of all registered hardware manifests."""
    return sorted(_REGISTRY)


def get_architecture(name: "str | NpuArchitecture") -> NpuArchitecture:
    """Instantiate a built-in manifest by (case-insensitive) name."""
    if isinstance(name, NpuArchitecture):
        return name
    key = str(name).strip().lower().replace("-", "").replace("_", "")
    key = _ALIASES.get(key, key)
    try:
        return _REGISTRY[key]()
    except KeyError as exc:
        raise KeyError(
            f"unknown architecture {name!r}; available: {list_architectures()}"
        ) from exc


def register_architecture(name: str, factory: Callable[[], NpuArchitecture]) -> None:
    """Register a custom manifest factory under ``name``."""
    _REGISTRY[name.strip().lower()] = factory


def architecture_from_dict(data: Mapping[str, Any]) -> NpuArchitecture:
    """Rebuild a manifest from :meth:`NpuArchitecture.to_dict` output.

    A ``base`` key may be given instead of a full description, to patch a
    built-in manifest::

        {"base": "ascend910b", "frequency_ghz": 2.1,
         "pools": {"UB": {"capacity_bytes": 262144}}}
    """
    base_name = data.get("base")
    if base_name is not None:
        arch = get_architecture(base_name)
        scalar_keys = (
            "name", "family", "frequency_ghz", "cube_cores", "vector_cores",
            "hbm_bandwidth_gbps", "l2_capacity_bytes", "decoupled_vector_core",
            "notes",
        )
        patch = {k: data[k] for k in scalar_keys if k in data}
        if patch:
            arch = arch.evolve(**patch)
        for pool_name, changes in (data.get("pools") or {}).items():
            changes = {k: v for k, v in changes.items() if k != "name"}
            arch = arch.with_pool_override(pool_name, **changes)
        for pipe_name, changes in (data.get("pipes") or {}).items():
            changes = {
                k: v for k, v in changes.items() if k not in ("name", "kind")
            }
            arch = arch.with_pipe_override(pipe_name, **changes)
        return arch

    pools = {k.upper(): MemoryPool(**v) for k, v in (data.get("pools") or {}).items()}
    pipes: Dict[str, PipeChannel] = {}
    for k, v in (data.get("pipes") or {}).items():
        v = dict(v)
        v["kind"] = PipeKind(v.get("kind", "MTE2"))
        pipes[k.upper()] = PipeChannel(**v)
    return NpuArchitecture(
        name=data["name"],
        family=data.get("family", ""),
        frequency_ghz=float(data["frequency_ghz"]),
        cube_cores=int(data.get("cube_cores", 1)),
        vector_cores=int(data.get("vector_cores", 1)),
        pools=pools,
        pipes=pipes,
        cube=_compute_unit_from_dict(data["cube"]),
        vector=_compute_unit_from_dict(data["vector"]),
        hbm_bandwidth_gbps=float(data.get("hbm_bandwidth_gbps", 0.0)),
        l2_capacity_bytes=int(data.get("l2_capacity_bytes", 0)),
        decoupled_vector_core=bool(data.get("decoupled_vector_core", True)),
        notes=data.get("notes", ""),
    )


def _compute_unit_from_dict(data: Mapping[str, Any]) -> ComputeUnit:
    profiles = {}
    for k, v in (data.get("dtype_profiles") or {}).items():
        profiles[k] = CubeDTypeProfile(
            dtype=DType.parse(v["dtype"]),
            k0=int(v["k0"]),
            macs_per_cycle=int(v["macs_per_cycle"]),
            accumulator=DType.parse(v.get("accumulator", "fp32")),
        )
    return ComputeUnit(
        name=data["name"],
        kind=PipeKind(data.get("kind", "CUBE")),
        pipeline_latency_cycles=float(data.get("pipeline_latency_cycles", 16.0)),
        dtype_profiles=profiles,
        bytes_per_cycle=float(data.get("bytes_per_cycle", 0.0)),
        issue_overhead_cycles=float(data.get("issue_overhead_cycles", 0.0)),
        description=data.get("description", ""),
    )
