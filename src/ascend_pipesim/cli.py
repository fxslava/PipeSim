"""Command line and JSON-RPC front end for ascend-pipesim.

Two ways in, one payload out.

Command line::

    npu-pipesim probe --arch ascend910b --gemm 2048,2048,4096 \\
        --tile 128,128,64 --subtile 64,64,32 --buffers double \\
        --dtype fp16 --stationary m
    npu-pipesim autotile --arch ascend910b --gemm 4096,4096,8192 \\
        --dtype fp16 --topk 3
    npu-pipesim codegen --arch ascend910b --gemm 1000,512,1024 \\
        --tile 128,128,64 --subtile 64,64,32 --outdir generated/

Sub-process JSON RPC (one request per line on stdin, one reply per line on
stdout, so an agent can keep a single warm process)::

    $ npu-pipesim rpc
    {"id": 1, "method": "probe", "params": {"gemm": [512, 512, 512]}}
    {"id": 1, "ok": true, "result": {...}}

Output defaults to JSON because the primary consumer is an LLM agent; pass
``--pretty`` for indented output, ``--compact`` for single-line JSON, and
``--format text`` for a human-readable rendering.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, TextIO

from .autotiler import AutotileOptions, AutotileResult, autotile
from .codegen import KernelBundle, synthesize_kernel
from .dsl import (
    AlgorithmFlow,
    BufferPlan,
    EpilogueOp,
    FlowValidationError,
    KPosition,
    LoopOrder,
    OutputPath,
    TensorSpec,
)
from .evaluator import EvalOptions, EvaluationReport, evaluate
from .topology import (
    DType,
    NpuArchitecture,
    architecture_from_dict,
    get_architecture,
    list_architectures,
)

__all__ = [
    "main",
    "build_parser",
    "run_probe",
    "run_autotile",
    "run_codegen",
    "serve_rpc",
]

EXIT_OK = 0
EXIT_INFEASIBLE = 1
EXIT_USAGE = 2


# ---------------------------------------------------------------------------
# Argument helpers
# ---------------------------------------------------------------------------


def _int_triple(text: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in str(text).replace("x", ",").split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"expected three comma-separated integers, got {text!r}"
        )
    try:
        values = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected integers in {text!r}"
        ) from exc
    if any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError(f"values must be positive: {text!r}")
    return values  # type: ignore[return-value]


def _key_value(text: str) -> tuple[str, float]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(
            f"expected NAME=VALUE, got {text!r}"
        )
    name, _, value = text.partition("=")
    try:
        return name.strip().upper(), float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{text!r}: value must be numeric"
        ) from exc


def _resolve_architecture(args: argparse.Namespace) -> NpuArchitecture:
    """Build the manifest, applying any file or inline overrides."""
    if getattr(args, "arch_file", None):
        with open(args.arch_file, "r", encoding="utf-8") as handle:
            arch = architecture_from_dict(json.load(handle))
    else:
        arch = get_architecture(args.arch)
    for name, value in getattr(args, "pipe_bw", None) or []:
        arch = arch.with_pipe_override(name, bandwidth_bpc=value)
    for name, value in getattr(args, "pipe_latency", None) or []:
        arch = arch.with_pipe_override(name, base_latency_cycles=value)
    for name, value in getattr(args, "pool_bytes", None) or []:
        arch = arch.with_pool_override(name, capacity_bytes=int(value))
    if getattr(args, "frequency", None):
        arch = arch.evolve(frequency_ghz=args.frequency)
    return arch


def _eval_options(args: argparse.Namespace) -> EvalOptions:
    return EvalOptions(
        model_hbm_contention=not getattr(args, "no_hbm_contention", False),
        model_l2_reuse=not getattr(args, "no_l2", False),
        emit_hints=not getattr(args, "no_hints", False),
    )


def _buffer_plan(args: argparse.Namespace) -> BufferPlan:
    plan = BufferPlan.named(getattr(args, "buffers", "double"))
    overrides = {
        key: getattr(args, key)
        for key in ("l1_depth", "l0_depth", "l0c_depth", "ub_depth")
        if getattr(args, key, None) is not None
    }
    return replace(plan, **overrides) if overrides else plan


def _epilogue(args: argparse.Namespace) -> tuple[EpilogueOp, ...]:
    unit = "vector" if getattr(args, "epilogue_on_vector", False) else "fixpipe"
    ops: List[EpilogueOp] = []
    n = args.gemm[1]
    if getattr(args, "dequant", False):
        ops.append(
            EpilogueOp(
                "dequant", unit,
                operand=TensorSpec("deq_scale", (n,), DType.FP32),
            )
        )
    if getattr(args, "bias", False):
        ops.append(
            EpilogueOp(
                "bias", unit, operand=TensorSpec("bias", (n,), DType.INT32)
            )
        )
    if getattr(args, "relu", False):
        ops.append(EpilogueOp("relu", unit))
    return tuple(ops)


def _stationary_loop_order(args: argparse.Namespace) -> Optional[LoopOrder]:
    """--stationary m keeps the M-side operand (A) resident: loop M first."""
    stationary = getattr(args, "stationary", None)
    if stationary == "m":
        return LoopOrder.M_FIRST
    if stationary == "n":
        return LoopOrder.N_FIRST
    return None


def _build_flow(args: argparse.Namespace) -> AlgorithmFlow:
    """Assemble an :class:`AlgorithmFlow` from parsed CLI arguments."""
    m, n, k = args.gemm
    epilogue = _epilogue(args)
    on_vector = getattr(args, "epilogue_on_vector", False)
    out_path = (
        OutputPath.FIXPIPE_UB_MTE3
        if on_vector and epilogue
        else OutputPath.FIXPIPE_GM
    )
    tile = args.tile or (128, 128, 64)
    subtile = args.subtile or (
        min(tile[0], 64), min(tile[1], 64), min(tile[2], 32)
    )
    loop_order = _stationary_loop_order(args) or LoopOrder(args.loop_order)
    return AlgorithmFlow.gemm(
        m=m, n=n, k=k,
        tile=tile, subtile=subtile,
        dtype=args.dtype,
        out_dtype=args.out_dtype,
        batch=args.batch,
        buffers=_buffer_plan(args),
        loop_order=loop_order,
        k_position=KPosition(args.k_position),
        output_path=out_path,
        epilogue=epilogue,
        transpose_a=args.transpose_a,
        transpose_b=args.transpose_b,
        cores=args.cores,
        name=args.name,
    )


def _json_payload(args: argparse.Namespace, payload: Any) -> str:
    """Serialise JSON: indented by default or with --pretty, tight with
    --compact alone."""
    compact = getattr(args, "compact", False) and not getattr(
        args, "pretty", False
    )
    return json.dumps(payload, indent=None if compact else 2)


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------


def _render_probe_text(report: EvaluationReport) -> str:
    lines: List[str] = [report.summary, ""]
    lines.append("Tier 1  feasibility")
    for name, fp in report.sram_footprints.items():
        flag = "ok " if fp.fits else "OVER"
        bar = "#" * int(min(100.0, fp.occupancy_pct) / 5)
        lines.append(
            f"  {flag} {name:<4} {fp.bytes_used:>8} / {fp.capacity_bytes:<8} B "
            f"{fp.occupancy_pct:>5.1f}%  {bar}"
        )
    lines.append(
        f"  fractal efficiency {report.fractal_efficiency:.4f} "
        f"(tail penalty {100 * (1 - report.fractal_efficiency):.2f}%)"
    )

    lines.append("")
    lines.append("Tier 2  hierarchical roofline")
    rf = report.roofline
    for level in rf.levels:
        marker = "<==" if level.level == rf.binding_level else "   "
        ceiling = level.ceiling_flops_per_cycle
        lines.append(
            f"  {marker} {level.level:<12} "
            f"I={level.intensity_flops_per_byte:>9.2f} FLOP/B  "
            f"BW={level.bandwidth_bpc:>7.1f} B/cyc  "
            f"ceiling={ceiling:>10.1f} FLOP/cyc"
        )
    lines.append(
        f"      peak {rf.peak_flops_per_cycle:.0f} FLOP/cyc, attainable "
        f"{rf.attainable_flops_per_cycle:.1f} "
        f"({rf.roofline_efficiency_pct:.1f}% of peak)"
    )

    lines.append("")
    lines.append("Tier 3  max-plus pipeline")
    sched = report.schedule
    worst = max((s.cycles_per_step for s in sched.stages), default=1.0) or 1.0
    for stage in sched.stages:
        bar = "#" * int(40 * stage.cycles_per_step / worst)
        mode = "overlap" if stage.overlapped else "SERIAL "
        lines.append(
            f"  {stage.name:<8} {stage.cycles_per_step:>9.2f} cyc/step "
            f"[{mode} d{stage.buffer_depth}] {bar}"
        )
    lines.append(
        f"  steady {sched.steady_state_cycles_per_step:.2f} cyc/step x "
        f"{sched.steps} steps + {sched.fill_cycles:.0f} fill + "
        f"{sched.drain_cycles:.0f} drain"
    )
    lines.append(
        f"  concurrency Omega={sched.concurrency_ratio:.2f} "
        f"(1.0 = serialized), latency hiding "
        f"{sched.latency_hiding_pct:.1f}%"
    )
    lines.append(
        f"  limiting stage : {sched.limiting_stage} "
        f"(next: {sched.secondary_stage}, slack {sched.slack_pct:.1f}%)"
    )
    lines.append(
        f"  cube utilisation: {sched.cube_utilization_pct:.2f}% useful, "
        f"{sched.cube_busy_pct:.2f}% busy"
    )
    lines.append(
        f"  T_min {report.t_min_cycles:,.0f} <= T {sched.total_cycles:,.0f} "
        f"<= T_max {report.t_max_cycles:,.0f} cycles "
        f"({report.timing_us['total_us']:.2f} us, "
        f"{report.timing_us['achieved_tflops_single_core']:.2f} TFLOP/s "
        f"on one core)"
    )

    if report.violations:
        lines.append("")
        lines.append("Violations")
        for v in report.violations:
            lines.append(f"  [{v.severity:<7}] {v.code}: {v.message}")
    if report.diagnostics:
        lines.append("")
        lines.append("Diagnostics")
        for d in report.diagnostics:
            lines.append(f"  [{d.impact:<6}] {d.code}: {d.message}")
            lines.append(f"           -> {d.suggestion}")
    hints = report.code_generation_hints
    if hints:
        lines.append("")
        lines.append("Code generation hints")
        for line in hints.get("pseudocode", []):
            lines.append(f"  {line}")
        if hints.get("requires_set_tail"):
            lines.append(f"  {hints.get('set_tail_call', 'SetTail required')}")
    return "\n".join(lines)


def _render_autotile_text(result: AutotileResult) -> str:
    p = result.problem
    lines = [
        f"autotile {p['m']}x{p['n']}x{p['k']} {p['in_dtype']}->"
        f"{p['out_dtype']} on {result.arch_name} "
        f"({result.evaluated} evaluated, {result.pruned} pruned, "
        f"{result.elapsed_s:.2f} s"
        f"{', truncated' if result.truncated else ''})",
        "",
        result.table(),
    ]
    best = result.best
    if best is not None:
        lines.append("")
        lines.append("Best candidate")
        lines.append(_render_probe_text(best.report))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def run_probe(args: argparse.Namespace) -> tuple[int, str]:
    """Evaluate one explicit tiling configuration."""
    arch = _resolve_architecture(args)
    flow = _build_flow(args)
    report = evaluate(flow, arch, _eval_options(args))
    if args.format == "text":
        payload = _render_probe_text(report)
    else:
        payload = _json_payload(args, report.to_dict())
    code = (
        EXIT_INFEASIBLE if (args.strict and not report.is_valid) else EXIT_OK
    )
    return code, payload


def run_autotile(args: argparse.Namespace) -> tuple[int, str]:
    """Sweep the tile space and rank the survivors."""
    arch = _resolve_architecture(args)
    template = _build_flow(args)
    stationary = _stationary_loop_order(args)
    if stationary is not None:
        orders: tuple[LoopOrder, ...] = (stationary,)
    elif args.loop_orders:
        orders = tuple(LoopOrder(o) for o in args.loop_orders)
    else:
        orders = (LoopOrder.M_FIRST, LoopOrder.N_FIRST)
    plans = tuple(BufferPlan.named(b) for b in (args.buffer_plans or ["double"]))
    options = AutotileOptions(
        topk=args.topk,
        max_tile=args.max_tile,
        max_evaluations=args.max_evaluations,
        forbid_tails=args.forbid_tails,
        require_feasible=not args.include_infeasible,
        loop_orders=orders,
        buffer_plans=plans,
        time_budget_s=args.time_budget,
        eval_options=_eval_options(args),
    )
    result = autotile(template, arch, options)
    if args.format == "text":
        payload = _render_autotile_text(result)
    else:
        payload = _json_payload(
            args, result.to_dict(full_report=args.full_report)
        )
    code = (
        EXIT_INFEASIBLE if (args.strict and result.best is None) else EXIT_OK
    )
    return code, payload


def run_codegen(args: argparse.Namespace) -> tuple[int, str]:
    """Evaluate a tiling and synthesise Ascend C kernel sources for it."""
    arch = _resolve_architecture(args)
    flow = _build_flow(args)
    report = evaluate(flow, arch, _eval_options(args))
    bundle = synthesize_kernel(report, flow, arch)

    if args.format == "text":
        payload = "\n".join(
            [
                bundle.summary(),
                "",
                "// ------------------- kernel_tiling.h -------------------",
                bundle.kernel_tiling_h,
                "// ---------------------- kernel.cpp ---------------------",
                bundle.kernel_cpp,
            ]
        )
    else:
        outdir = getattr(args, "outdir", None)
        if outdir:
            written = [str(p) for p in bundle.write(outdir)]
            payload = _json_payload(
                args,
                {
                    "written": written,
                    "outdir": str(Path(outdir)),
                    "meta": bundle.meta,
                    "buffers": {
                        k: v.to_dict() for k, v in bundle.buffers.items()
                    },
                },
            )
        else:
            payload = _json_payload(args, bundle.to_dict())

    code = (
        EXIT_INFEASIBLE if (args.strict and not report.is_valid) else EXIT_OK
    )
    return code, payload


def run_archs(args: argparse.Namespace) -> tuple[int, str]:
    """List the built-in manifests, or dump one in full."""
    if args.name:
        arch = get_architecture(args.name)
        if args.format == "text":
            lines = [
                f"{arch.name}  --  {arch.family}",
                f"  {arch.frequency_ghz} GHz, {arch.cube_cores} Cube cores, "
                f"{arch.vector_cores} Vector cores",
                f"  HBM {arch.hbm_bandwidth_gbps:.0f} GB/s "
                f"({arch.gm_bandwidth_bpc_per_core():.1f} B/cycle per core), "
                f"L2 {arch.l2_capacity_bytes // (1024 * 1024)} MiB",
                "  pools:",
            ]
            for name, pool in arch.pools.items():
                if name == "GM":
                    continue
                lines.append(
                    f"    {name:<4} {pool.capacity_bytes // 1024:>5} KiB  "
                    f"align {pool.alignment_bytes} B  banks {pool.banks}"
                )
            lines.append("  pipes:")
            for name, pipe in arch.pipes.items():
                lines.append(
                    f"    {name:<8} {pipe.src:>3} -> {pipe.dst:<3} "
                    f"{pipe.bandwidth_bpc:>6.0f} B/cyc  "
                    f"latency {pipe.base_latency_cycles:>5.0f} cyc"
                )
            lines.append("  peak throughput:")
            for dt in sorted(arch.cube.dtype_profiles):
                lines.append(
                    f"    {dt:<6} {arch.peak_tflops(dt):>8.1f} T/s all cores, "
                    f"{arch.peak_tflops(dt, all_cores=False):>6.2f} T/s one core"
                )
            if arch.notes:
                lines.append(f"  notes: {arch.notes}")
            return EXIT_OK, "\n".join(lines)
        return EXIT_OK, _json_payload(args, arch.to_dict())

    names = list_architectures()
    if args.format == "text":
        rows = []
        for name in names:
            arch = get_architecture(name)
            rows.append(
                f"  {name:<12} {arch.peak_tflops('fp16'):>7.1f} TFLOP/s fp16  "
                f"{arch.cube_cores:>3} cores  "
                f"L1 {arch.pool('L1').capacity_bytes // 1024:>4} KiB  "
                f"UB {arch.pool('UB').capacity_bytes // 1024:>4} KiB  "
                f"{arch.family}"
            )
        return EXIT_OK, "\n".join(["built-in manifests:", *rows])
    return EXIT_OK, _json_payload(
        args, {"architectures": [get_architecture(n).to_dict() for n in names]}
    )


# ---------------------------------------------------------------------------
# JSON RPC
# ---------------------------------------------------------------------------


def _rpc_namespace(params: Mapping[str, Any]) -> argparse.Namespace:
    """Turn an RPC params object into the namespace the commands expect."""
    defaults: Dict[str, Any] = {
        "arch": "ascend910b", "arch_file": None,
        "gemm": (512, 512, 512), "tile": None, "subtile": None,
        "dtype": "fp16", "out_dtype": None, "batch": 1, "cores": 1,
        "buffers": "double",
        "l1_depth": None, "l0_depth": None, "l0c_depth": None,
        "ub_depth": None,
        "loop_order": "M_first", "k_position": "inner", "stationary": None,
        "dequant": False, "bias": False, "relu": False,
        "epilogue_on_vector": False,
        "transpose_a": False, "transpose_b": False,
        "name": "flow", "format": "json", "compact": True, "pretty": False,
        "strict": False, "outdir": None,
        "no_l2": False, "no_hbm_contention": False, "no_hints": False,
        "pipe_bw": [], "pipe_latency": [], "pool_bytes": [],
        "frequency": None,
        "topk": 3, "max_tile": 512, "max_evaluations": 8000,
        "forbid_tails": False, "include_infeasible": False,
        "loop_orders": None, "buffer_plans": None, "time_budget": None,
        "full_report": False,
    }
    defaults.update(params)
    for key in ("gemm", "tile", "subtile"):
        value = defaults.get(key)
        if isinstance(value, str):
            defaults[key] = _int_triple(value)
        elif isinstance(value, (list, tuple)):
            defaults[key] = tuple(int(v) for v in value)
    for key in ("pipe_bw", "pipe_latency", "pool_bytes"):
        value = defaults.get(key) or []
        if isinstance(value, Mapping):
            defaults[key] = [(str(k).upper(), float(v)) for k, v in value.items()]
    return argparse.Namespace(**defaults)


def _rpc_dispatch(method: str, params: Mapping[str, Any]) -> Any:
    args = _rpc_namespace(params)
    args.format = "json"
    if method == "probe":
        arch = _resolve_architecture(args)
        return evaluate(_build_flow(args), arch, _eval_options(args)).to_dict()
    if method == "autotile":
        arch = _resolve_architecture(args)
        stationary = _stationary_loop_order(args)
        if stationary is not None:
            orders: tuple[LoopOrder, ...] = (stationary,)
        elif args.loop_orders:
            orders = tuple(LoopOrder(o) for o in args.loop_orders)
        else:
            orders = (LoopOrder.M_FIRST, LoopOrder.N_FIRST)
        plans = tuple(
            BufferPlan.named(b) for b in (args.buffer_plans or ["double"])
        )
        options = AutotileOptions(
            topk=args.topk,
            max_tile=args.max_tile,
            max_evaluations=args.max_evaluations,
            forbid_tails=args.forbid_tails,
            require_feasible=not args.include_infeasible,
            loop_orders=orders,
            buffer_plans=plans,
            time_budget_s=args.time_budget,
            eval_options=_eval_options(args),
        )
        return autotile(_build_flow(args), arch, options).to_dict(
            full_report=args.full_report
        )
    if method == "codegen":
        arch = _resolve_architecture(args)
        flow = _build_flow(args)
        report = evaluate(flow, arch, _eval_options(args))
        return synthesize_kernel(report, flow, arch).to_dict()
    if method == "architectures":
        return {"architectures": list_architectures()}
    if method == "describe_arch":
        return get_architecture(params.get("arch", "ascend910b")).to_dict()
    if method == "evaluate_flow":
        arch = _resolve_architecture(args)
        flow = AlgorithmFlow.from_dict(params["flow"])
        return evaluate(flow, arch, _eval_options(args)).to_dict()
    raise ValueError(
        f"unknown method {method!r}; expected one of probe, autotile, "
        f"codegen, architectures, describe_arch, evaluate_flow"
    )


def serve_rpc(stdin: TextIO | None = None, stdout: TextIO | None = None) -> int:
    """Serve newline-delimited JSON requests until stdin closes."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        request_id: Any = None
        try:
            request = json.loads(line)
            request_id = request.get("id")
            result = _rpc_dispatch(
                request.get("method", ""), request.get("params") or {}
            )
            response: Dict[str, Any] = {
                "id": request_id, "ok": True, "result": result
            }
        except Exception as exc:  # surfaced to the caller, never fatal here
            response = {
                "id": request_id,
                "ok": False,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        stdout.write(json.dumps(response) + "\n")
        stdout.flush()
    return EXIT_OK


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _add_arch_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("hardware")
    group.add_argument(
        "--arch", default="ascend910b",
        help=f"built-in manifest ({', '.join(list_architectures())})",
    )
    group.add_argument(
        "--arch-file", metavar="PATH",
        help="JSON manifest; may use {\"base\": \"ascend910b\", ...} to patch",
    )
    group.add_argument(
        "--pipe-bw", action="append", type=_key_value, metavar="NAME=BYTES",
        help="override a pipe bandwidth in B/cycle, e.g. --pipe-bw MTE1=512",
    )
    group.add_argument(
        "--pipe-latency", action="append", type=_key_value,
        metavar="NAME=CYCLES", help="override a pipe base latency",
    )
    group.add_argument(
        "--pool-bytes", action="append", type=_key_value, metavar="NAME=BYTES",
        help="override a pool capacity, e.g. --pool-bytes L1=1048576",
    )
    group.add_argument(
        "--frequency", type=float, metavar="GHZ", help="override core clock"
    )


def _add_problem_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("problem")
    group.add_argument(
        "--gemm", type=_int_triple, required=True, metavar="M,N,K",
        help="GEMM extents, e.g. --gemm 2048,2048,4096",
    )
    group.add_argument("--batch", type=int, default=1, help="batch count")
    group.add_argument(
        "--dtype", default="fp16",
        help="input element type (fp16, bf16, int8, fp32, fp8)",
    )
    group.add_argument(
        "--out-dtype", default=None, help="output element type (default: input)"
    )
    group.add_argument(
        "--cores", type=int, default=1,
        help="AI Cores the problem is split across; >1 models HBM contention",
    )
    group.add_argument("--transpose-a", action="store_true")
    group.add_argument("--transpose-b", action="store_true")
    group.add_argument("--dequant", action="store_true", help="dequant epilogue")
    group.add_argument("--bias", action="store_true", help="bias epilogue")
    group.add_argument("--relu", action="store_true", help="ReLU epilogue")
    group.add_argument(
        "--epilogue-on-vector", action="store_true",
        help="run the epilogue on the Vector unit via UB instead of Fixpipe",
    )
    group.add_argument("--name", default="flow", help="label for the report")


def _add_staging_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("staging")
    group.add_argument(
        "--buffers", default="double", choices=["single", "double", "triple"],
        help="multi-buffering preset (default: double)",
    )
    for name in ("l1-depth", "l0-depth", "l0c-depth", "ub-depth"):
        group.add_argument(
            f"--{name}", type=int, default=None,
            help=f"override {name.replace('-', '_')} from the preset",
        )
    group.add_argument(
        "--loop-order", default="M_first", choices=["M_first", "N_first"],
        help="which spatial dimension carries the outermost tile loop",
    )
    group.add_argument(
        "--stationary", choices=["m", "n"], default=None,
        help="operand that stays resident across the inner spatial sweep: "
             "m keeps the M-side operand (A) resident => M_first, n keeps B "
             "resident => N_first; overrides --loop-order",
    )
    group.add_argument(
        "--k-position", default="inner", choices=["inner", "outer"],
        help="inner keeps the accumulator in L0C; outer re-materialises C",
    )


def _add_output_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("output")
    group.add_argument(
        "--format", default="json", choices=["json", "text"],
        help="json (default, for agents) or text (for humans)",
    )
    group.add_argument(
        "--compact", action="store_true", help="single-line JSON"
    )
    group.add_argument(
        "--pretty", action="store_true",
        help="pretty-print JSON with indent=2 (the default; overrides --compact)",
    )
    group.add_argument(
        "-o", "--output", metavar="PATH", help="write to a file instead of stdout"
    )
    group.add_argument(
        "--strict", action="store_true",
        help=f"exit {EXIT_INFEASIBLE} when nothing feasible was found",
    )
    group.add_argument(
        "--no-l2", action="store_true", help="disable L2 reuse modelling"
    )
    group.add_argument(
        "--no-hbm-contention", action="store_true",
        help="ignore the per-core HBM share even when --cores > 1",
    )
    group.add_argument(
        "--no-hints", action="store_true",
        help="skip Ascend C code-generation hints",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="npu-pipesim",
        description=(
            "Analytical flow and tiling simulator for Huawei Ascend DaVinci "
            "AI Cores."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  npu-pipesim probe --arch ascend910b --gemm 2048,2048,4096 "
            "--tile 128,128,64 --subtile 64,64,32 --buffers double "
            "--stationary m\n"
            "  npu-pipesim autotile --arch ascend910b --gemm 4096,4096,8192 "
            "--dtype fp16 --topk 3\n"
            "  npu-pipesim codegen --arch ascend910b --gemm 1000,512,1024 "
            "--tile 128,128,64 --outdir generated/\n"
            "  npu-pipesim archs --format text\n"
            "  npu-pipesim rpc  < requests.jsonl\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    probe = sub.add_parser(
        "probe", help="evaluate one explicit tiling configuration"
    )
    _add_problem_arguments(probe)
    probe.add_argument(
        "--tile", type=_int_triple, metavar="M,N,K",
        help="outer L1 tile (default 128,128,64)",
    )
    probe.add_argument(
        "--subtile", type=_int_triple, metavar="m,n,k",
        help="inner L0 subtile (default: half the tile)",
    )
    _add_staging_arguments(probe)
    _add_arch_arguments(probe)
    _add_output_arguments(probe)
    probe.set_defaults(func=run_probe)

    auto = sub.add_parser("autotile", help="search and rank the tile space")
    _add_problem_arguments(auto)
    auto.add_argument("--topk", type=int, default=3, help="candidates to report")
    auto.add_argument(
        "--max-tile", type=int, default=512, help="largest tile edge to try"
    )
    auto.add_argument(
        "--max-evaluations", type=int, default=8000,
        help="cap on full evaluations (pruned candidates are free)",
    )
    auto.add_argument(
        "--time-budget", type=float, metavar="SECONDS",
        help="stop searching after this long and report what was found",
    )
    auto.add_argument(
        "--forbid-tails", action="store_true",
        help="only accept tiles that divide M, N and K exactly",
    )
    auto.add_argument(
        "--include-infeasible", action="store_true",
        help="keep candidates that violate a capacity constraint",
    )
    auto.add_argument(
        "--loop-orders", nargs="+", choices=["M_first", "N_first"],
        help="restrict the loop orders considered",
    )
    auto.add_argument(
        "--buffer-plans", nargs="+", choices=["single", "double", "triple"],
        help="buffering presets to consider (default: double)",
    )
    auto.add_argument(
        "--full-report", action="store_true",
        help="embed the complete evaluation report for every candidate",
    )
    auto.add_argument("--tile", type=_int_triple, help=argparse.SUPPRESS)
    auto.add_argument("--subtile", type=_int_triple, help=argparse.SUPPRESS)
    _add_staging_arguments(auto)
    _add_arch_arguments(auto)
    _add_output_arguments(auto)
    auto.set_defaults(func=run_autotile)

    gen = sub.add_parser(
        "codegen",
        help="evaluate a tiling and synthesise Ascend C kernel sources",
    )
    _add_problem_arguments(gen)
    gen.add_argument(
        "--tile", type=_int_triple, metavar="M,N,K",
        help="outer L1 tile (default 128,128,64)",
    )
    gen.add_argument(
        "--subtile", type=_int_triple, metavar="m,n,k",
        help="inner L0 subtile (default: half the tile)",
    )
    _add_staging_arguments(gen)
    _add_arch_arguments(gen)
    _add_output_arguments(gen)
    gen.add_argument(
        "--outdir", metavar="DIR",
        help="write kernel.cpp and kernel_tiling.h into DIR instead of "
             "emitting the sources as JSON",
    )
    gen.set_defaults(func=run_codegen)

    archs = sub.add_parser("archs", help="list or describe hardware manifests")
    archs.add_argument("name", nargs="?", help="manifest to describe in full")
    archs.add_argument(
        "--format", default="text", choices=["json", "text"],
    )
    archs.add_argument("--compact", action="store_true")
    archs.add_argument("-o", "--output", metavar="PATH")
    archs.set_defaults(func=run_archs)

    rpc = sub.add_parser(
        "rpc", help="serve newline-delimited JSON requests on stdin/stdout"
    )
    rpc.set_defaults(func=None)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "rpc":
        return serve_rpc()

    try:
        code, payload = args.func(args)
    except (FlowValidationError, ValueError, KeyError) as exc:
        parser.exit(EXIT_USAGE, f"npu-pipesim: {type(exc).__name__}: {exc}\n")
        return EXIT_USAGE  # pragma: no cover - parser.exit raises

    destination = getattr(args, "output", None)
    if destination:
        with open(destination, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    else:
        sys.stdout.write(payload + "\n")
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
