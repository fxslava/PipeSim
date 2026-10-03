"""Ascend C kernel synthesiser: evaluation result -> compilable kernel skeleton.

Given a validated :class:`~ascend_pipesim.evaluator.EvaluationReport` and the
:class:`~ascend_pipesim.dsl.AlgorithmFlow` that produced it, this module emits
two artefacts ready to drop into a CANN project:

``kernel_tiling.h``
    Every derived constant of the winning tiling as ``constexpr`` values --
    padded tile dimensions, loop trip counts, ragged tail widths, per-buffer
    byte lengths -- plus ``static_assert`` budget checks against the target
    manifest's pool capacities and the host-side tiling record.

``kernel.cpp``
    A single AI Core Ascend C kernel with the static staging pipeline the
    evaluator modelled: ``TQue`` layout (A1/B1 at ``l1_depth``, A2/B2 at
    ``l0_depth``, CO1 at ``l0c_depth``), the MTE2 (GM -> L1, Nd2Nz) copy-in,
    the MTE1 (L1 -> L0A/L0B, LoadData) subtile staging, the Cube iterate
    driven through a ``Matmul`` object, and the Fixpipe drain
    (``GetTensorC`` + the M->FIX flag handshake).  When the shape is ragged
    the synthesiser injects the runtime ``SetTail(tailM, tailN)`` boundary
    checks around the final tile row/column.

The byte lengths are taken from the evaluator's Tier-1 footprints (the single
source of truth), so the synthesised buffers match the simulated occupancy
exactly.

Example
-------
>>> from ascend_pipesim import AlgorithmFlow, evaluate
>>> from ascend_pipesim.codegen import synthesize_kernel
>>> flow = AlgorithmFlow.gemm(m=2048, n=2048, k=4096,
...                           tile=(128, 128, 64), subtile=(64, 64, 32))
>>> bundle = synthesize_kernel(evaluate(flow), flow)
>>> bundle.write("out/")          # kernel.cpp + kernel_tiling.h
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from .dsl import AlgorithmFlow, KPosition, LoopOrder, OpType
from .evaluator import EvaluationReport
from .topology import DType, NpuArchitecture, get_architecture

__all__ = [
    "BufferSpec",
    "KernelBundle",
    "synthesize_kernel",
    "synthesize",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ceil_div(a: int, b: int) -> int:
    return -(-int(a) // int(b))


def _pad(value: int, gran: int) -> int:
    return _ceil_div(value, gran) * gran


def _cbool(value: bool) -> str:
    return "true" if value else "false"


_ASCEND_C_TYPES: Dict[DType, str] = {
    DType.FP16: "half",
    DType.BF16: "bfloat16_t",
    DType.FP32: "float",
    DType.INT8: "int8_t",
    DType.UINT8: "uint8_t",
    DType.INT16: "int16_t",
    DType.INT32: "int32_t",
}


def _ascend_c_type(dtype: DType) -> str:
    try:
        return _ASCEND_C_TYPES[dtype]
    except KeyError as exc:
        raise ValueError(
            f"no Ascend C mapping for dtype {dtype.value!r}; synthesised "
            f"kernels support {sorted(d.value for d in _ASCEND_C_TYPES)}"
        ) from exc


def _sanitize(name: str, fallback: str = "gemm") -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z_]", "_", str(name)).strip("_")
    return cleaned or fallback


@dataclass(frozen=True)
class BufferSpec:
    """One synthesised TQue: position, depth and exact per-buffer bytes."""

    queue: str
    position: str
    depth: int
    per_buffer_bytes: int
    pool: str
    capacity_bytes: int
    note: str = ""

    @property
    def total_bytes(self) -> int:
        return self.per_buffer_bytes * self.depth

    @property
    def occupancy_pct(self) -> float:
        return 100.0 * self.total_bytes / self.capacity_bytes

    @property
    def declaration(self) -> str:
        # TPosition is the real Ascend C enum: the queue's template argument
        # is what downstream tools (and the CANN runtime) read the SRAM
        # domain from.
        return f"TQue<TPosition::{self.position}, {self.depth}> {self.queue};"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "queue": self.queue,
            "position": self.position,
            "depth": self.depth,
            "per_buffer_bytes": self.per_buffer_bytes,
            "total_bytes": self.total_bytes,
            "pool": self.pool,
            "capacity_bytes": self.capacity_bytes,
            "occupancy_pct": round(self.occupancy_pct, 2),
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# Synthesis context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Ctx:
    """Every scalar the emitters need, resolved once from flow + report."""

    flow: AlgorithmFlow
    arch: NpuArchitecture
    kernel_name: str
    guard_macro: str
    a_type: str
    b_type: str
    c_type: str
    k0: int
    itemsize: int
    out_itemsize: int
    m: int
    n: int
    k: int
    tile_m: int
    tile_n: int
    tile_k: int
    sub_m: int
    sub_n: int
    sub_k: int
    tile_m_pad: int
    tile_n_pad: int
    tile_k_pad: int
    sub_m_pad: int
    sub_n_pad: int
    sub_k_pad: int
    used_m: int
    used_n: int
    used_k: int
    sub_steps_m: int
    sub_steps_n: int
    sub_steps_k: int
    tail_m: int
    tail_n: int
    tail_k: int
    settail_required: bool
    outer_nest: List[str]
    inner_nest: List[str]
    buffers: Dict[str, BufferSpec]
    predicted: Dict[str, Any]
    transpose_a: bool
    transpose_b: bool
    epilogue: List[str]


def _footprint_bytes(
    report: EvaluationReport, pool: str, key: str, fallback: int, arch: NpuArchitecture
) -> int:
    fp = report.sram_footprints.get(pool)
    if fp is not None and key in fp.breakdown:
        return int(fp.breakdown[key])
    return arch.pool(pool).aligned_size(fallback)


def _outer_loop_nest(flow: AlgorithmFlow) -> List[str]:
    spatial = (
        ["mo", "no"] if flow.loop_order is LoopOrder.M_FIRST else ["no", "mo"]
    )
    if flow.k_position is KPosition.INNER:
        return [*spatial, "ko"]
    return ["ko", *spatial]


def _inner_loop_nest(flow: AlgorithmFlow) -> List[str]:
    return ["sk", "sm", "sn"] if flow.loop_order is LoopOrder.M_FIRST \
        else ["sk", "sn", "sm"]


def _build_ctx(
    report: EvaluationReport,
    flow: AlgorithmFlow,
    arch: NpuArchitecture,
) -> _Ctx:
    if flow.op_type is OpType.ELEMWISE:
        raise ValueError(
            "codegen synthesises the Cube GEMM pipeline (MTE2/MTE1/Cube/"
            "Fixpipe); element-wise Vector kernels have no TQue layout"
        )

    dtype = flow.compute_dtype
    k0 = arch.cube.profile(dtype).k0
    itemsize = dtype.size_bytes
    out_itemsize = flow.c.dtype.size_bytes
    t = flow.tile
    loops = flow.loops

    tile_m_pad, tile_n_pad = _pad(t.tile_m, 16), _pad(t.tile_n, 16)
    tile_k_pad = _pad(t.tile_k, k0)
    sub_m_pad, sub_n_pad = _pad(t.sub_m, 16), _pad(t.sub_n, 16)
    sub_k_pad = _pad(t.sub_k, k0)

    # Per-buffer byte lengths: Tier-1 footprints are authoritative; the
    # recomputed values are an identical fallback for hand-built reports.
    l1 = arch.pool("L1")
    a1_bytes = _footprint_bytes(
        report, "L1", "A_tile", tile_m_pad * tile_k_pad * itemsize, arch
    )
    b1_bytes = _footprint_bytes(
        report, "L1", "B_tile", tile_k_pad * tile_n_pad * itemsize, arch
    )
    a2_bytes = _footprint_bytes(
        report, "L0A", "A_subtile", sub_m_pad * sub_k_pad * itemsize, arch
    )
    b2_bytes = _footprint_bytes(
        report, "L0B", "B_subtile", sub_k_pad * sub_n_pad * itemsize, arch
    )
    co1_bytes = _footprint_bytes(
        report, "L0C", "accumulator_fp32", tile_m_pad * tile_n_pad * 4, arch
    )

    l1_depth = flow.buffers.l1_depth
    l0_depth = flow.buffers.l0_depth
    l0c_depth = flow.buffers.l0c_depth

    buffers: Dict[str, BufferSpec] = {
        "queA1_": BufferSpec(
            "queA1_", "A1", l1_depth, a1_bytes, "L1", l1.capacity_bytes,
            f"outer A panel {t.tile_m}x{t.tile_k} "
            f"({dtype.value}, zZ fractal layout)",
        ),
        "queB1_": BufferSpec(
            "queB1_", "B1", l1_depth, b1_bytes, "L1", l1.capacity_bytes,
            f"outer B panel {t.tile_k}x{t.tile_n} "
            f"({dtype.value}, nZ fractal layout)",
        ),
        "queA2_": BufferSpec(
            "queA2_", "A2", l0_depth, a2_bytes,
            "L0A", arch.pool("L0A").capacity_bytes,
            f"inner A subtile {t.sub_m}x{t.sub_k}",
        ),
        "queB2_": BufferSpec(
            "queB2_", "B2", l0_depth, b2_bytes,
            "L0B", arch.pool("L0B").capacity_bytes,
            f"inner B subtile {t.sub_k}x{t.sub_n}",
        ),
        "queCO1_": BufferSpec(
            "queCO1_", "CO1", l0c_depth, co1_bytes,
            "L0C", arch.pool("L0C").capacity_bytes,
            f"FP32 accumulator {t.tile_m}x{t.tile_n} (4 B/element)",
        ),
    }

    tail_m = loops.m % t.tile_m if loops.m % t.tile_m else 0
    tail_n = loops.n % t.tile_n if loops.n % t.tile_n else 0
    tail_k = loops.k % t.tile_k if loops.k % t.tile_k else 0
    settail_required = t.has_tail() or t.has_sub_tail()

    name = _sanitize(flow.name)
    return _Ctx(
        flow=flow,
        arch=arch,
        kernel_name=f"pipesim_{name}",
        guard_macro=f"PIPESIM_{name.upper()}_TILING_H_",
        a_type=_ascend_c_type(dtype),
        b_type=_ascend_c_type(dtype),
        c_type=_ascend_c_type(flow.c.dtype),
        k0=k0,
        itemsize=itemsize,
        out_itemsize=out_itemsize,
        m=loops.m,
        n=loops.n,
        k=loops.k,
        tile_m=t.tile_m,
        tile_n=t.tile_n,
        tile_k=t.tile_k,
        sub_m=t.sub_m,
        sub_n=t.sub_n,
        sub_k=t.sub_k,
        tile_m_pad=tile_m_pad,
        tile_n_pad=tile_n_pad,
        tile_k_pad=tile_k_pad,
        sub_m_pad=sub_m_pad,
        sub_n_pad=sub_n_pad,
        sub_k_pad=sub_k_pad,
        used_m=t.n_tiles_m,
        used_n=t.n_tiles_n,
        used_k=t.n_tiles_k,
        sub_steps_m=t.n_subs_m,
        sub_steps_n=t.n_subs_n,
        sub_steps_k=t.n_subs_k,
        tail_m=tail_m,
        tail_n=tail_n,
        tail_k=tail_k,
        settail_required=settail_required,
        outer_nest=_outer_loop_nest(flow),
        inner_nest=_inner_loop_nest(flow),
        buffers=buffers,
        predicted={
            "is_valid": report.is_valid,
            "cube_utilization_pct": round(report.cube_utilization_pct, 2),
            "limiting_stage": report.limiting_stage,
            "total_us": round(report.timing_us.get("total_us", 0.0), 2),
            "achieved_tflops_single_core": round(
                report.timing_us.get("achieved_tflops_single_core", 0.0), 3
            ),
        },
        transpose_a=flow.transpose_a,
        transpose_b=flow.transpose_b,
        epilogue=[op.kind for op in flow.epilogue],
    )


# ---------------------------------------------------------------------------
# kernel_tiling.h
# ---------------------------------------------------------------------------


def _emit_tiling_header(c: _Ctx) -> str:
    q = c.buffers
    L: List[str] = []
    L.append("// kernel_tiling.h")
    L.append("// Auto-generated by ascend-pipesim codegen -- DO NOT EDIT BY HAND.")
    L.append("//")
    L.append(
        f"// Problem   : C[M,N] = A[M,K] @ B[K,N]"
        f"   (M={c.m}, N={c.n}, K={c.k})"
    )
    L.append(
        f"// Data type : A/B {c.flow.compute_dtype.value} ({c.itemsize} B)"
        f"  ->  C {c.flow.c.dtype.value} ({c.out_itemsize} B),"
        f" L0C accumulator fp32 (4 B)"
    )
    L.append(
        f"// Tiling    : outer tile ({c.tile_m}, {c.tile_n}, {c.tile_k}), "
        f"subtile ({c.sub_m}, {c.sub_n}, {c.sub_k})"
    )
    L.append(
        f"// Buffers   : A1/B1 depth {q['queA1_'].depth} (L1), "
        f"A2/B2 depth {q['queA2_'].depth} (L0A/L0B), "
        f"CO1 depth {q['queCO1_'].depth} (L0C)"
    )
    L.append(
        "// Predicted : {util}% cube utilisation, limiting stage {stage}".format(
            util=c.predicted["cube_utilization_pct"],
            stage=c.predicted["limiting_stage"],
        )
    )
    L.append(f"#ifndef {c.guard_macro}")
    L.append(f"#define {c.guard_macro}")
    L.append("")
    L.append("#include <stdint.h>")
    L.append("")
    L.append("// --- problem extents --------------------------------------------------")
    L.append(f"constexpr int32_t M_EXTENT = {c.m};")
    L.append(f"constexpr int32_t N_EXTENT = {c.n};")
    L.append(f"constexpr int32_t K_EXTENT = {c.k};")
    L.append("")
    L.append("// --- outer (L1) tile and inner (L0) subtile ----------------------------")
    L.append(f"constexpr int32_t TILE_M = {c.tile_m};")
    L.append(f"constexpr int32_t TILE_N = {c.tile_n};")
    L.append(f"constexpr int32_t TILE_K = {c.tile_k};")
    L.append(f"constexpr int32_t SUB_M  = {c.sub_m};")
    L.append(f"constexpr int32_t SUB_N  = {c.sub_n};")
    L.append(f"constexpr int32_t SUB_K  = {c.sub_k};")
    L.append("")
    L.append("// Fractal-padded dimensions actually allocated in L1/L0.")
    L.append(f"constexpr int32_t TILE_M_PAD = {c.tile_m_pad};  // pad(TILE_M, 16)")
    L.append(f"constexpr int32_t TILE_N_PAD = {c.tile_n_pad};  // pad(TILE_N, 16)")
    L.append(f"constexpr int32_t TILE_K_PAD = {c.tile_k_pad};  // pad(TILE_K, FRACTAL_K0)")
    L.append(f"constexpr int32_t SUB_M_PAD  = {c.sub_m_pad};   // pad(SUB_M, 16)")
    L.append(f"constexpr int32_t SUB_N_PAD  = {c.sub_n_pad};   // pad(SUB_N, 16)")
    L.append(f"constexpr int32_t SUB_K_PAD  = {c.sub_k_pad};   // pad(SUB_K, FRACTAL_K0)")
    L.append("")
    L.append(
        "// Cube fractal geometry (m0 x n0 x k0; k0 widens for narrow types)."
    )
    L.append("constexpr int32_t FRACTAL_M0 = 16;")
    L.append("constexpr int32_t FRACTAL_N0 = 16;")
    L.append(f"constexpr int32_t FRACTAL_K0 = {c.k0};")
    L.append("")
    L.append("// --- loop trip counts ---------------------------------------------------")
    L.append(f"constexpr int32_t USED_M = {c.used_m};  // ceil(M_EXTENT / TILE_M)")
    L.append(f"constexpr int32_t USED_N = {c.used_n};  // ceil(N_EXTENT / TILE_N)")
    L.append(f"constexpr int32_t USED_K = {c.used_k};  // ceil(K_EXTENT / TILE_K)")
    L.append(f"constexpr int32_t SUB_STEPS_M = {c.sub_steps_m};  // ceil(TILE_M / SUB_M)")
    L.append(f"constexpr int32_t SUB_STEPS_N = {c.sub_steps_n};  // ceil(TILE_N / SUB_N)")
    L.append(f"constexpr int32_t SUB_STEPS_K = {c.sub_steps_k};  // ceil(TILE_K / SUB_K)")
    L.append(
        f"constexpr int32_t SUB_STEPS_TOTAL = {c.sub_steps_m * c.sub_steps_n * c.sub_steps_k};"
        f"  // flattened inner software-pipeline length"
    )
    L.append("")
    L.append("// --- ragged tails (0 when the axis divides exactly) ---------------------")
    L.append(f"constexpr int32_t TAIL_M = {c.tail_m};")
    L.append(f"constexpr int32_t TAIL_N = {c.tail_n};")
    L.append(f"constexpr int32_t TAIL_K = {c.tail_k};")
    L.append(
        f"constexpr bool   HAS_TAIL_M = "
        f"{_cbool(c.tail_m not in (0, c.tile_m))};"
    )
    L.append(
        f"constexpr bool   HAS_TAIL_N = "
        f"{_cbool(c.tail_n not in (0, c.tile_n))};"
    )
    L.append(
        f"constexpr bool   HAS_TAIL_K = "
        f"{_cbool(c.tail_k not in (0, c.tile_k))};"
    )
    L.append(
        f"constexpr bool   SETTAIL_REQUIRED = {_cbool(c.settail_required)};"
        if c.settail_required
        else "constexpr bool   SETTAIL_REQUIRED = false;"
    )
    L.append("")
    L.append("// --- TQue buffer byte lengths (per buffer; depth is the TQue template) --")
    for key, const in (
        ("queA1_", "A1_BUFFER_BYTES"),
        ("queB1_", "B1_BUFFER_BYTES"),
        ("queA2_", "A2_BUFFER_BYTES"),
        ("queB2_", "B2_BUFFER_BYTES"),
        ("queCO1_", "CO1_BUFFER_BYTES"),
    ):
        spec = c.buffers[key]
        L.append(
            f"constexpr uint32_t {const} = {spec.per_buffer_bytes};"
            f"   // {spec.note}"
        )
    L.append("")
    L.append(
        "// Pool budgets from the ascend-pipesim manifest; enforced at build time."
    )
    l1_total = q["queA1_"].total_bytes + q["queB1_"].total_bytes
    L.append(
        f"static_assert(A1_BUFFER_BYTES * {q['queA1_'].depth}"
        f" + B1_BUFFER_BYTES * {q['queB1_'].depth} <= "
        f"{q['queA1_'].capacity_bytes}, \"L1 overflow\");  // {l1_total} B"
    )
    L.append(
        f"static_assert(A2_BUFFER_BYTES * {q['queA2_'].depth} <= "
        f"{q['queA2_'].capacity_bytes}, \"L0A overflow\");"
    )
    L.append(
        f"static_assert(B2_BUFFER_BYTES * {q['queB2_'].depth} <= "
        f"{q['queB2_'].capacity_bytes}, \"L0B overflow\");"
    )
    L.append(
        f"static_assert(CO1_BUFFER_BYTES * {q['queCO1_'].depth} <= "
        f"{q['queCO1_'].capacity_bytes}, \"L0C overflow\");"
    )
    L.append("")
    L.append("// --- host-side tiling record (one instance per launch) -------------------")
    L.append("struct PipeSimTiling {")
    L.append("    int32_t usedM;  // = ceil(M / TILE_M), from the host")
    L.append("    int32_t usedN;")
    L.append("    int32_t usedK;")
    L.append("    int32_t tailM;  // ragged last-tile width along M (0 => none)")
    L.append("    int32_t tailN;")
    L.append("    int32_t tailK;")
    L.append("};")
    L.append("")
    L.append(f"#endif  // {c.guard_macro}")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# kernel.cpp
# ---------------------------------------------------------------------------


def _nd2nz_block(c: _Ctx, operand: str) -> List[str]:
    """CopyIn parameters for one operand panel, honouring transposes."""
    if operand == "A":
        if c.transpose_a:
            # A is stored [K, M]: panel = TILE_K rows x TILE_M contiguous cols.
            n_value, d_value = c.tile_k, c.tile_m
            src_d, c0 = c.m, c.tile_k_pad
            base = "ko * TILE_K * M_EXTENT + mo * TILE_M"
        else:
            n_value, d_value = c.tile_m, c.tile_k
            src_d, c0 = c.k, c.tile_m_pad
            base = "mo * TILE_M * K_EXTENT + ko * TILE_K"
        return [
            f"        Nd2NzParams nd2nzA;",
            f"        nd2nzA.ndNum = 1;",
            f"        nd2nzA.nValue = {n_value};          // rows of the A panel",
            f"        nd2nzA.dValue = {d_value};          // contiguous run per row",
            f"        nd2nzA.srcDValue = {src_d};         // GM row pitch of A",
            f"        nd2nzA.dstNzC0Stride = {c0};        // zZ fractal column height",
            f"        nd2nzA.dstNzNStride = 1;",
            f"        nd2nzA.dstNzMatrixStride = 0;",
            f"        DataCopy(a1, aGM_[{base}], nd2nzA);",
        ]
    if c.transpose_b:
        # B is stored [N, K]: panel = TILE_N rows x TILE_K contiguous cols.
        n_value, d_value = c.tile_n, c.tile_k
        src_d, c0 = c.k, c.tile_n_pad
        base = "no * TILE_N * K_EXTENT + ko * TILE_K"
    else:
        n_value, d_value = c.tile_k, c.tile_n
        src_d, c0 = c.n, c.tile_k_pad
        base = "ko * TILE_K * N_EXTENT + no * TILE_N"
    return [
        f"        Nd2NzParams nd2nzB;",
        f"        nd2nzB.ndNum = 1;",
        f"        nd2nzB.nValue = {n_value};          // rows of the B panel",
        f"        nd2nzB.dValue = {d_value};          // contiguous run per row",
        f"        nd2nzB.srcDValue = {src_d};         // GM row pitch of B",
        f"        nd2nzB.dstNzC0Stride = {c0};        // nZ fractal column height",
        f"        nd2nzB.dstNzNStride = 1;",
        f"        nd2nzB.dstNzMatrixStride = 0;",
        f"        DataCopy(b1, bGM_[{base}], nd2nzB);",
    ]


def _emit_kernel_cpp(c: _Ctx) -> str:
    q = c.buffers
    L: List[str] = []
    L.append("// kernel.cpp")
    L.append("// Auto-generated by ascend-pipesim codegen -- DO NOT EDIT BY HAND.")
    L.append("//")
    L.append(
        f"// C[{c.m},{c.n}] = A[{c.m},{c.k}] @ B[{c.k},{c.n}]"
        f"  {c.flow.compute_dtype.value} -> {c.flow.c.dtype.value},"
        f" loop order {c.flow.loop_order.value}, k {c.flow.k_position.value}."
    )
    L.append(
        f"// Outer tile ({c.tile_m},{c.tile_n},{c.tile_k}), "
        f"subtile ({c.sub_m},{c.sub_n},{c.sub_k})."
    )
    L.append(
        "// Pipeline: MTE2 (GM -> L1) -> MTE1 (L1 -> L0A/L0B) -> Cube iterate"
        " -> Fixpipe (L0C -> GM)."
    )
    L.append(
        "// Predicted: {util}% cube utilisation, limiting stage {stage}.".format(
            util=c.predicted["cube_utilization_pct"],
            stage=c.predicted["limiting_stage"],
        )
    )
    L.append("//")
    L.append("// Compile against the CANN Ascend C headers (kernel_operator.h and")
    L.append("// lib/matmul_intf.h) and launch with the host tiling record below.")
    L.append('#include "kernel_operator.h"')
    L.append('#include "lib/matmul_intf.h"')
    L.append('#include "kernel_tiling.h"')
    L.append("")
    L.append("using namespace AscendC;")
    L.append("")
    L.append(f"using A_T = {c.a_type};")
    L.append(f"using B_T = {c.b_type};")
    L.append(f"using C_T = {c.c_type};")
    L.append("using BiasT = int32_t;  // placeholder; bias epilogues load a table")
    L.append("")
    L.append(f"constexpr bool TRANSPOSE_A = {_cbool(c.transpose_a)};")
    L.append(f"constexpr bool TRANSPOSE_B = {_cbool(c.transpose_b)};")
    L.append(
        "constexpr MatmulConfig MM_CONFIG = "
        "GetNormConfig(TRANSPOSE_A, TRANSPOSE_B, false, false);"
    )
    L.append("")
    L.append("class PipeSimMatmulKernel {")
    L.append("public:")
    L.append("    __aicore__ inline PipeSimMatmulKernel(TPipe& pipe) : pipe_(pipe) {}")
    L.append("")
    L.append("    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c,")
    L.append("                                const __gm__ PipeSimTiling* tilingGm) {")
    L.append("        tiling_ = *tilingGm;")
    L.append(
        "        aGM_.SetGlobalBuffer(reinterpret_cast<__gm__ A_T*>(a),"
        " M_EXTENT * K_EXTENT);"
    )
    L.append(
        "        bGM_.SetGlobalBuffer(reinterpret_cast<__gm__ B_T*>(b),"
        " K_EXTENT * N_EXTENT);"
    )
    L.append(
        "        cGM_.SetGlobalBuffer(reinterpret_cast<__gm__ C_T*>(c),"
        " M_EXTENT * N_EXTENT);"
    )
    L.append("")
    L.append("        // ---- TPipe layout: exact byte lengths from the winning tiling ----")
    for key, const in (
        ("queA1_", "A1_BUFFER_BYTES"),
        ("queB1_", "B1_BUFFER_BYTES"),
        ("queA2_", "A2_BUFFER_BYTES"),
        ("queB2_", "B2_BUFFER_BYTES"),
        ("queCO1_", "CO1_BUFFER_BYTES"),
    ):
        spec = q[key]
        L.append(
            f"        pipe_.InitBuffer({spec.queue}, {const});"
            f"   // {spec.per_buffer_bytes} B x {spec.depth} buffer(s)"
            f" of {spec.pool} ({spec.occupancy_pct:.1f}%)"
        )
    L.append("        matmulObj.Init(&pipe_, nullptr);  // tiling via PipeSimTiling above")
    L.append("    }")
    L.append("")
    L.append("    __aicore__ inline void Process() {")

    indent = "        "
    for name in c.outer_nest[:-1]:
        axis = name[0].upper()  # mo -> usedM, no -> usedN, ko -> usedK
        L.append(f"{indent}for (int32_t {name} = 0; {name} < tiling_.used{axis}; ++{name}) {{")
        indent += "    "
    inner = c.outer_nest[-1]
    inner_axis = inner[0].upper()
    l1_depth = c.buffers["queA1_"].depth

    if l1_depth >= 2:
        prologue_args = ", ".join(
            "0" if name == inner else name for name in ("mo", "no", "ko")
        )
        prefetch_args = ", ".join(
            f"{inner} + 1" if name == inner else name
            for name in ("mo", "no", "ko")
        )
        L.append(f"{indent}// MTE2 prologue: fill L1 ping-pong bank 0 before any compute")
        L.append(f"{indent}// blocks, so the first Cube issue never waits on GM latency.")
        L.append(f"{indent}CopyIn({prologue_args});   // MTE2: GM -> L1, bank 0 (Nd2Nz)")
        L.append(
            f"{indent}for (int32_t {inner} = 0; {inner} < tiling_.used{inner_axis};"
            f" ++{inner}) {{"
        )
        indent += "    "
        L.append(f"{indent}// Prefetch the next tile into the alternate L1 bank BEFORE")
        L.append(f"{indent}// blocking on this tile's Cube work: MTE2 stays one token")
        L.append(f"{indent}// ahead of the MTE1/Cube consumer chain.")
        L.append(f"{indent}if ({inner} + 1 < tiling_.used{inner_axis}) {{")
        L.append(f"{indent}    CopyIn({prefetch_args});   // MTE2: GM -> L1, bank 1")
        L.append(f"{indent}}}")
        L.append(f"{indent}Compute(mo, no, ko);  // MTE1 -> L0 ping-pong + Cube iterate")
    else:
        L.append(f"{indent}// Single-buffered L1: no prefetch slack exists (one bank), so")
        L.append(f"{indent}// MTE2 and the consumer chain serialize -- exactly what the")
        L.append(f"{indent}// simulator reports as concurrency ratio Omega = 1.0.")
        L.append(
            f"{indent}for (int32_t {inner} = 0; {inner} < tiling_.used{inner_axis};"
            f" ++{inner}) {{"
        )
        indent += "    "
        L.append(f"{indent}CopyIn(mo, no, ko);   // MTE2: GM -> L1 (A1/B1, Nd2Nz)")
        L.append(f"{indent}Compute(mo, no, ko);  // MTE1 -> L0 + Cube iterate")

    # Close the innermost loop; the Fixpipe drain runs once per output tile
    # (or once per K step when k_position=outer re-materialises C).
    indent = indent[:-4]
    L.append(f"{indent}}}")
    L.append(f"{indent}CopyOut(mo, no);      // Fixpipe: L0C -> GM (GetTensorC + M->FIX wait)")

    # Close the remaining outer loops.
    for name in reversed(c.outer_nest[:-1]):
        indent = indent[:-4]
        L.append(f"{indent}}}")
    L.append("        matmulObj.End();")
    L.append("    }")
    L.append("")
    L.append("private:")
    L.append("    // ---- MTE2: GM -> L1 ---------------------------------------------------")
    L.append("    __aicore__ inline void CopyIn(int32_t mo, int32_t no, int32_t ko) {")
    L.append(
        f"        // One A panel [{c.tile_m} x {c.tile_k}] and one B panel "
        f"[{c.tile_k} x {c.tile_n}],"
    )
    if c.settail_required:
        L.append(
            "        // reformed ND -> NZ in flight. Ragged tails: the GM read"
        )
        L.append(
            "        // pads to the fractal; SetTail in Compute trims what the"
        )
        L.append("        // Cube multiplies.")
    else:
        L.append("        // reformed ND -> NZ in flight.")
    L.append("        LocalTensor<A_T> a1 = queA1_.AllocTensor<A_T>();")
    L.append("        LocalTensor<B_T> b1 = queB1_.AllocTensor<B_T>();")
    L.extend(_nd2nz_block(c, "A"))
    L.extend(_nd2nz_block(c, "B"))
    L.append("        queA1_.EnQue(a1);")
    L.append("        queB1_.EnQue(b1);")
    L.append("    }")
    L.append("")
    L.append("    // ---- MTE1 + Cube over the inner subtile grid --------------------------")
    L.append("    __aicore__ inline void Compute(int32_t mo, int32_t no, int32_t ko) {")
    L.append("        LocalTensor<A_T> a1 = queA1_.DeQue<A_T>();")
    L.append("        LocalTensor<B_T> b1 = queB1_.DeQue<B_T>();")
    L.append("")

    if c.settail_required:
        L.append("        // ---- Tail partitioning (SetTail) --------------------------------")
        L.append(
            "        // Ragged M/N boundary: reprogram the Cube with the live extent so"
        )
        L.append(
            "        // the final tile row/column does not multiply fractal padding."
        )
        L.append(
            "        const bool isTailRow = HAS_TAIL_M && (mo == tiling_.usedM - 1);"
        )
        L.append(
            "        const bool isTailCol = HAS_TAIL_N && (no == tiling_.usedN - 1);"
        )
        L.append("        if (isTailRow || isTailCol) {")
        L.append(
            "            const uint16_t tailM = static_cast<uint16_t>("
            "isTailRow ? tiling_.tailM : TILE_M);"
        )
        L.append(
            "            const uint16_t tailN = static_cast<uint16_t>("
            "isTailCol ? tiling_.tailN : TILE_N);"
        )
        L.append("            matmulObj.SetTail(tailM, tailN);")
        L.append("        }")
        L.append("")

    l0_depth = c.buffers["queA2_"].depth
    if l0_depth >= 2:
        L.append("        // ---- Software-pipeline prologue ------------------------------------")
        L.append("        // Prime BOTH L0 ping-pong banks before the first Cube issue so")
        L.append("        // the MTE1 latency of sub-tiles 0 and 1 hides behind Cube work.")
        L.append("        LoadSubtile(a1, b1, 0);")
        L.append("        if (SUB_STEPS_TOTAL > 1) {")
        L.append("            LoadSubtile(a1, b1, 1);")
        L.append("        }")
        L.append("")
        L.append("        // ---- Steady state: Cube on bank K, MTE1 on bank K+1 -----------------")
        L.append("        // The producer runs two tokens ahead of the consumer: DeQue is")
        L.append("        // delayed to the exact point of consumption, and each bank freed")
        L.append("        // by the Cube is immediately re-armed with sub-tile step+2, so")
        L.append("        // MTE1 issue never idles behind the Cube; no coarse all-pipe")
        L.append("        // barrier is ever issued inside the steady state.")
        L.append("        for (int32_t step = 0; step < SUB_STEPS_TOTAL; ++step) {")
        L.append("            LocalTensor<A_T> a2 = queA2_.DeQue<A_T>();   // bank step%2")
        L.append("            LocalTensor<B_T> b2 = queB2_.DeQue<B_T>();")
        L.append("            matmulObj.SetTensorA(a2);")
        L.append("            matmulObj.SetTensorB(b2);")
        L.append("            matmulObj.Iterate();   // Cube: L0A x L0B -> L0C, accumulates across sk")
        L.append("            queA2_.FreeTensor(a2);   // release this bank for sub-tile step+2")
        L.append("            queB2_.FreeTensor(b2);")
        L.append("            if (step + 2 < SUB_STEPS_TOTAL) {")
        L.append("                LoadSubtile(a1, b1, step + 2);   // MTE1: refill freed bank")
        L.append("            }")
        L.append("        }")
        L.append("        queA1_.FreeTensor(a1);")
        L.append("        queB1_.FreeTensor(b1);")
        L.append("    }")
        L.append("")
        L.append("    // ---- MTE1 producer step: Alloc -> LoadData -> EnQue -------------------")
        L.append("    // EnQue fires immediately after the load is scheduled; with depth-2")
        L.append("    // banks the consumer (DeQue in the steady-state loop) stays two")
        L.append("    // tokens behind, which is the maximum slack the queues can hold.")
        L.append("    __aicore__ inline void LoadSubtile(const LocalTensor<A_T>& a1,")
        L.append("                                       const LocalTensor<B_T>& b1,")
        L.append("                                       int32_t step) {")
        nest_names = c.inner_nest  # e.g. ["sk", "sm", "sn"]
        extents = {"sm": "SUB_STEPS_M", "sn": "SUB_STEPS_N", "sk": "SUB_STEPS_K"}
        mid_e, in_e = extents[nest_names[1]], extents[nest_names[2]]
        L.append(
            f"        // Decode the flattened step index into the "
            f"({', '.join(nest_names)}) sub-tile grid."
        )
        L.append(f"        const int32_t {nest_names[0]} = step / ({mid_e} * {in_e});")
        L.append(f"        const int32_t stepRem = step % ({mid_e} * {in_e});")
        L.append(f"        const int32_t {nest_names[1]} = stepRem / {in_e};")
        L.append(f"        const int32_t {nest_names[2]} = stepRem % {in_e};")
        L.append("        // MTE1: L1 -> L0A, zZ fractal load of the SUB_M x SUB_K block.")
        L.append("        LocalTensor<A_T> a2 = queA2_.AllocTensor<A_T>();")
        L.append("        LoadData2dParams ldA;")
        L.append("        ldA.repeat = (SUB_M_PAD / FRACTAL_M0) * (SUB_K_PAD / FRACTAL_K0);")
        L.append("        ldA.srcGap = 0;")
        L.append("        ldA.dstGap = 0;")
        L.append("        LoadData(a2, a1[A1BlockOffset(sm, sk)], ldA);")
        L.append("        queA2_.EnQue(a2);")
        L.append("        // MTE1: L1 -> L0B, nZ fractal load of the SUB_K x SUB_N block.")
        L.append("        LocalTensor<B_T> b2 = queB2_.AllocTensor<B_T>();")
        L.append("        LoadData2dParams ldB;")
        L.append("        ldB.repeat = (SUB_N_PAD / FRACTAL_N0) * (SUB_K_PAD / FRACTAL_K0);")
        L.append("        ldB.srcGap = 0;")
        L.append("        ldB.dstGap = 0;")
        L.append("        LoadData(b2, b1[B1BlockOffset(sk, sn)], ldB);")
        L.append("        queB2_.EnQue(b2);")
        L.append("    }")
    else:
        L.append("        // Single-buffered L0: one bank cannot hold producer/consumer")
        L.append("        // slack, so each sub-tile load synchronizes with its own")
        L.append("        // consumption (concurrency ratio Omega = 1.0 in the simulator).")
        L.append("        // Upgrade to --buffers double for overlapped issue.")
        sub_indent = "        "
        for name in c.inner_nest:
            extent = {"sm": "SUB_STEPS_M", "sn": "SUB_STEPS_N", "sk": "SUB_STEPS_K"}[name]
            L.append(
                f"{sub_indent}for (int32_t {name} = 0; {name} < {extent}; ++{name}) {{"
            )
            sub_indent += "    "
        L.append(f"{sub_indent}// MTE1: L1 -> L0A/L0B, then immediate synchronous consumption.")
        L.append(f"{sub_indent}LocalTensor<A_T> a2 = queA2_.AllocTensor<A_T>();")
        L.append(f"{sub_indent}LoadData2dParams ldA;")
        L.append(f"{sub_indent}ldA.repeat = (SUB_M_PAD / FRACTAL_M0) * (SUB_K_PAD / FRACTAL_K0);")
        L.append(f"{sub_indent}ldA.srcGap = 0;")
        L.append(f"{sub_indent}ldA.dstGap = 0;")
        L.append(f"{sub_indent}LoadData(a2, a1[A1BlockOffset(sm, sk)], ldA);")
        L.append(f"{sub_indent}queA2_.EnQue(a2);")
        L.append(f"{sub_indent}LocalTensor<B_T> b2 = queB2_.AllocTensor<B_T>();")
        L.append(f"{sub_indent}LoadData2dParams ldB;")
        L.append(f"{sub_indent}ldB.repeat = (SUB_N_PAD / FRACTAL_N0) * (SUB_K_PAD / FRACTAL_K0);")
        L.append(f"{sub_indent}ldB.srcGap = 0;")
        L.append(f"{sub_indent}ldB.dstGap = 0;")
        L.append(f"{sub_indent}LoadData(b2, b1[B1BlockOffset(sk, sn)], ldB);")
        L.append(f"{sub_indent}queB2_.EnQue(b2);")
        L.append(f"{sub_indent}matmulObj.SetTensorA(queA2_.DeQue<A_T>());")
        L.append(f"{sub_indent}matmulObj.SetTensorB(queB2_.DeQue<B_T>());")
        L.append(f"{sub_indent}matmulObj.Iterate();   // Cube: L0A x L0B -> L0C")
        L.append(f"{sub_indent}queA2_.FreeTensor(a2);")
        L.append(f"{sub_indent}queB2_.FreeTensor(b2);")
        for _ in c.inner_nest:
            sub_indent = sub_indent[:-4]
            L.append(f"{sub_indent}}}")
        L.append("        queA1_.FreeTensor(a1);")
        L.append("        queB1_.FreeTensor(b1);")
        L.append("    }")
    L.append("")
    L.append("    // ---- Fixpipe: L0C -> GM -----------------------------------------------")
    L.append("    __aicore__ inline void CopyOut(int32_t mo, int32_t no) {")
    L.append(
        "        // GetTensorC hands the accumulator to the Fixpipe (inline dtype"
    )
    L.append(
        "        // conversion); the M->FIX flag pair waits for the drain before"
    )
    L.append(
        "        // the Cube reuses L0C.  EVENT_ID0 is a literal so the"
    )
    L.append("        // handshake stays statically resolvable.")
    L.append("        SetFlag<HardEvent::M_FIX>(EVENT_ID0);")
    L.append("        LocalTensor<C_T> co1 = queCO1_.AllocTensor<C_T>();")
    L.append("        auto cOut = matmulObj.GetTensorC();")
    L.append("        DataCopyParams fixpipeCopy;")
    L.append("        fixpipeCopy.blockCount = TILE_M;   // output tile rows")
    L.append("        fixpipeCopy.blockLen  = TILE_N * sizeof(C_T) / 32;  // 32 B units")
    L.append("        fixpipeCopy.srcStride = 0;")
    L.append(
        "        fixpipeCopy.dstStride = (N_EXTENT - TILE_N) * sizeof(C_T) / 32;"
    )
    L.append("        DataCopy(cGM_[mo * TILE_M * N_EXTENT + no * TILE_N], cOut, fixpipeCopy);")
    L.append("        queCO1_.FreeTensor(co1);")
    L.append("        WaitFlag<HardEvent::M_FIX>(EVENT_ID0);")
    L.append("    }")
    L.append("")
    L.append("    // zZ (A) layout: each k0-fractal column block holds TILE_M_PAD rows")
    L.append("    // contiguously; sk advances whole fractal blocks, sm walks rows.")
    L.append("    static __aicore__ inline int32_t A1BlockOffset(int32_t sm, int32_t sk) {")
    L.append("        return sk * (SUB_K_PAD / FRACTAL_K0) * (TILE_M_PAD * FRACTAL_K0)")
    L.append("             + sm * SUB_M_PAD;")
    L.append("    }")
    L.append("    // nZ (B) layout: each n-fractal block holds TILE_K_PAD k-rows")
    L.append("    // contiguously; sn advances whole fractal blocks, sk walks k-rows.")
    L.append("    static __aicore__ inline int32_t B1BlockOffset(int32_t sk, int32_t sn) {")
    L.append("        return sn * (SUB_N_PAD / FRACTAL_N0) * (TILE_K_PAD * FRACTAL_N0)")
    L.append("             + sk * SUB_K_PAD;")
    L.append("    }")
    L.append("")
    L.append("    // ---- state --------------------------------------------------------------")
    L.append("    TPipe& pipe_;")
    for key in ("queA1_", "queB1_", "queA2_", "queB2_", "queCO1_"):
        spec = q[key]
        L.append(f"    {spec.declaration}  // {spec.note}")
    L.append(
        "    Matmul<A_T, B_T, C_T, BiasT, MM_CONFIG> matmulObj;"
    )
    L.append("    GlobalTensor<A_T> aGM_;")
    L.append("    GlobalTensor<B_T> bGM_;")
    L.append("    GlobalTensor<C_T> cGM_;")
    L.append("    PipeSimTiling tiling_ = {USED_M, USED_N, USED_K, TAIL_M, TAIL_N, TAIL_K};")
    L.append("};")
    L.append("")
    L.append(f"extern \"C\" __global__ __aicore__ void {c.kernel_name}(")
    L.append("        GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace,")
    L.append("        __gm__ PipeSimTiling* tiling) {")
    L.append(
        f"    // Analysed as a single AI Core; blockDim={c.flow.cores}"
        " in the simulator."
    )
    L.append("    TPipe pipe;")
    L.append("    PipeSimMatmulKernel kernel(pipe);")
    L.append("    kernel.Init(a, b, c, tiling);")
    L.append("    kernel.Process();")
    L.append("}")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelBundle:
    """The two synthesised artefacts plus the buffer layout they encode."""

    kernel_cpp: str
    kernel_tiling_h: str
    buffers: Dict[str, BufferSpec]
    meta: Dict[str, Any]

    def files(self) -> Dict[str, str]:
        return {
            "kernel.cpp": self.kernel_cpp,
            "kernel_tiling.h": self.kernel_tiling_h,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kernel.cpp": self.kernel_cpp,
            "kernel_tiling.h": self.kernel_tiling_h,
            "buffers": {k: v.to_dict() for k, v in self.buffers.items()},
            "meta": self.meta,
        }

    def write(self, outdir: "str | Path") -> List[Path]:
        """Write both artefacts into ``outdir`` and return their paths."""
        root = Path(outdir)
        root.mkdir(parents=True, exist_ok=True)
        paths = []
        for filename, text in self.files().items():
            path = root / filename
            path.write_text(text, encoding="utf-8")
            paths.append(path)
        return paths

    def summary(self) -> str:
        m = self.meta
        return (
            f"{m['kernel_name']}: tile {m['tile']['tile_m']}x"
            f"{m['tile']['tile_n']}x{m['tile']['tile_k']}, "
            f"SetTail={'yes' if m['settail_required'] else 'no'}, "
            f"predicted {m['predicted']['cube_utilization_pct']}% cube "
            f"utilisation ({m['predicted']['limiting_stage']})"
        )


def synthesize_kernel(
    report: EvaluationReport,
    flow: AlgorithmFlow,
    arch: "str | NpuArchitecture | None" = None,
) -> KernelBundle:
    """Convert an evaluation result and its tiling into Ascend C sources.

    ``report`` supplies the Tier-1 buffer footprints and the predicted
    pipeline numbers; ``flow`` (with its bound :class:`TileConfig`) supplies
    the structure.  The two must describe the same configuration -- the
    natural calling pattern is ``synthesize_kernel(evaluate(flow), flow)``.
    """
    arch = get_architecture(arch if arch is not None else report.arch_name)
    c = _build_ctx(report, flow, arch)
    meta = {
        "kernel_name": c.kernel_name,
        "arch": c.arch.name,
        "flow": c.flow.name,
        "dtype": c.flow.compute_dtype.value,
        "out_dtype": c.flow.c.dtype.value,
        "gemm": {"m": c.m, "n": c.n, "k": c.k},
        "tile": c.flow.tile.to_dict(),
        "loop_order": c.flow.loop_order.value,
        "k_position": c.flow.k_position.value,
        "pipeline_style": (
            "software_pipelined" if c.buffers["queA2_"].depth >= 2
            and c.buffers["queA1_"].depth >= 2 else "synchronous"
        ),
        "outer_loop_nest": list(c.outer_nest),
        "inner_loop_nest": list(c.inner_nest),
        "settail_required": c.settail_required,
        "tail_sizes": {"m": c.tail_m, "n": c.tail_n, "k": c.tail_k},
        "transposes": {"a": c.transpose_a, "b": c.transpose_b},
        "epilogue": c.epilogue,
        "predicted": c.predicted,
    }
    return KernelBundle(
        kernel_cpp=_emit_kernel_cpp(c),
        kernel_tiling_h=_emit_tiling_header(c),
        buffers=c.buffers,
        meta=meta,
    )


def synthesize(
    flow: AlgorithmFlow,
    arch: "str | NpuArchitecture" = "ascend910b",
    options: Any = None,
) -> KernelBundle:
    """Evaluate ``flow`` and synthesise its kernel in one step."""
    from .evaluator import evaluate as _evaluate

    report = _evaluate(flow, arch, options)
    return synthesize_kernel(report, flow, arch)
