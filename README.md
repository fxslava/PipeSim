# Ascend PipeSim

**ascend-pipesim** is an analytical performance simulator, static verifier, and
autotiling engine for Huawei Ascend AI Cores based on the DaVinci architecture.
It models pipeline bottlenecks, memory footprints, and tiling efficiency without
requiring physical NPU hardware or cycle-accurate simulator runs. Results are
analytical estimates; architecture parameters and workload assumptions should
be calibrated against measurements when device-level accuracy is required.

PipeSim represents an operator and its staging strategy with the
`AlgorithmFlow` DSL, evaluates it against a hardware topology, explores tile
configurations, and can emit an Ascend C kernel skeleton. The same flow can be
evaluated against multiple architectures.

## Three-tier analytical evaluator

`evaluate()` runs three related analyses and returns an `EvaluationReport`:

1. **Tier 1 — feasibility.** Checks working-set footprints against the
   disjoint `L1`, `L0A`, `L0B`, `L0C`, and `UB` SRAM pools. It validates
   DaVinci fractal row and column dimensions (16×16 and dtype-specific `k0`
   alignment) as well as 512-byte GM burst constraints. Capacity violations
   are reported in the evaluation result.
2. **Tier 2 — hierarchical roofline.** Estimates arithmetic intensity
   (FLOP/Byte) and bandwidth ceilings at multiple levels of the hierarchy,
   including GM, L1, L0C, and DRAM/HBM.
3. **Tier 3 — max-plus pipeline schedule.** Models asynchronous MTE2, MTE1,
   Cube, Fixpipe, Vector ALU, and inter-core IPC work. The schedule accounts
   for prologue and drain effects, including exact small-loop bubble
   penalization when the tile count is two or fewer.

## Execution modes

`ExecutionMode` describes the execution graph that is evaluated:

| Mode | Behavior |
|---|---|
| `standalone_cube` | Single-core Cube GEMM with the accumulator drained through inline Fixpipe. |
| `mix_split_m` | MIX cluster mode with one AIC and two AIV consumers; spatially bisects the M dimension, slices local UB epilogue operands, and synchronizes cores through a UB or GM workspace ring. |
| `mix_split_n` | The corresponding two-AIV MIX mode with spatial bisection along N, local UB epilogue slicing, and UB- or GM-backed inter-core synchronization. |
| `hybrid_simt_simd` | DaVinci v3 SIMT/SIMD path modeling warp prefill and packing, DataCache hits and misses, branch divergence, register pressure/spilling, and shared UB-port arbitration. |

The MIX schedule includes producer/consumer synchronization and can overlap
handoffs with ring slots. The SIMT portion is enabled by workload metadata in
the flow; architecture cache and scheduling assumptions come from the selected
topology.

## Parametric uncertainty and Pareto frontier

Frontier search explores candidate execution graphs and tilings over a
four-dimensional uncertainty box, Θ:

- DMA latency jitter (`dma_latency_cycles`)
- HBM contention (`effective_hbm_bandwidth_fraction`)
- Cross-core synchronization overhead (`cross_core_sync_cycles`)
- SIMT DataCache miss-rate drift (`simt_cache_miss_rate_drift`)

The default `scenario` dominance policy compares the 25 evaluated scenarios
(box corners, center, and axial points). The more conservative
`certified_interval` policy prunes only when timing bounds prove dominance
across the continuous intervals. The result contains typed candidate flows and
predicted bounds; frontier search does not generate kernel source.

See [Frontier exploration](docs/FRONTIER_EXPLORATION.md) for the full API,
command-line, uncertainty-file, and scenario details.

## Supported hardware topologies

The built-in manifests are defined in
[`topology.py`](src/ascend_pipesim/topology.py). Core counts and bandwidths
below describe the analytical profiles in this repository, not a promise of
measured device performance.

| Name | Family / organization | Profile highlights |
|---|---|---|
| `ascend310p` | DaVinci v1; fused Cube and Vector cores | 8 Cube / 8 Vector cores; Vector epilogues do not overlap Cube work on the same fused core. |
| `ascend910b` | DaVinci v2; decoupled AIC/AIV | 24 Cube / 48 Vector cores; two Vector cores per Cube core. |
| `ascend910c` | DaVinci v2+; dual-die | 32 Cube / 64 Vector cores; FP8 profile and dual-die memory parameters. |
| `ascend950pr` | DaVinci v3 (`arch35`) | 1.65 GHz analytical profile; 32 Cube / 64 Vector cores; 216 KiB tensor UB budget, 32 KiB SIMT DataCache, and MXFP4/MXFP8 profiles. Frequency and local DMA rates are estimates. |
| `ascend950dt` | DaVinci v3 | 36 Cube / 72 Vector cores; 512 KiB aggregate UB model and 4,000 GB/s HBM profile. Frequency and local DMA rates are estimates. |

The model also recognizes aliases such as `arch35` for `ascend950pr`. Use
`npu-pipesim archs` to list profiles or `npu-pipesim archs ascend950pr` to
inspect a profile.

## Installation

From a checkout:

```bash
python -m pip install -e .
```

The `npu-pipesim` and `ascend-pipesim` console commands are installed by the
package. In a source checkout, the equivalent module entry point is
`python -m ascend_pipesim`.

## Command-line interface and JSON-RPC

All commands share architecture, staging, and output options where applicable.
Use `--help` on a command for its full option list. JSON is the default output
format; pass `--format text` for a human-readable report, or `--output PATH`
to write the result to a file.

### `probe`

Evaluate one explicit GEMM or flow. For example:

```bash
npu-pipesim probe --arch ascend910b --gemm 2048,2048,4096 \
  --tile 128,128,64 --subtile 64,64,32 --buffers double \
  --dtype fp16 --stationary m
```

Alternatively, supply a serialized `AlgorithmFlow` with `--flow-file`.

### `autotile`

Search and rank tile configurations for a GEMM:

```bash
npu-pipesim autotile --arch ascend910b --gemm 4096,4096,8192 \
  --dtype fp16 --topk 3 --max-evaluations 8000
```

### `frontier`

Explore execution graphs and tilings across uncertainty scenarios:

```bash
npu-pipesim frontier --arch ascend950pr --flow-file flow.json \
  --uncertainty uncertainty.json --topk 5 \
  --dominance-policy scenario
```

The `--top-k` spelling is also accepted as an alias for `--topk`.

### `codegen`

Evaluate a configuration and emit Ascend C kernel sources:

```bash
npu-pipesim codegen --arch ascend910b --gemm 2048,2048,4096 \
  --tile 128,128,64 --subtile 64,64,32 --outdir generated/
```

### `archs`

List built-in hardware manifests or describe one:

```bash
npu-pipesim archs
npu-pipesim archs ascend950pr --format text
```

### `rpc`

`rpc` serves newline-delimited JSON requests over stdin and writes one JSON
reply per line to stdout. It supports `probe`, `autotile`, `frontier`,
`codegen`, `architectures`, `describe_arch`, and `evaluate_flow`.

```text
$ npu-pipesim rpc
{"id":1,"method":"frontier","params":{"gemm":[2048,2048,4096],"arch":"ascend950pr","topk":3}}
```

The server responds with one JSON object per request, containing the same `id`,
an `ok` field, and the command result (or an error).

The module form of the CLI is equivalent:

```bash
python -m ascend_pipesim frontier --arch ascend950pr --flow-file flow.json
```

## Python API

This example constructs a GEMM flow, evaluates it under nominal conditions,
then searches its frontier over the default uncertainty space:

```python
from ascend_pipesim import (
    AlgorithmFlow,
    FrontierSearchOptions,
    UncertaintySpace,
    evaluate,
    find_optimal_frontiers,
)

flow = AlgorithmFlow.gemm(
    m=2048,
    n=2048,
    k=4096,
    tile=(128, 128, 64),
    subtile=(64, 64, 32),
    dtype="fp16",
    cores=32,
)

report = evaluate(flow, arch="ascend950pr")
print(report.is_valid, report.total_cycles, report.limiting_stage)

result = find_optimal_frontiers(
    flow,
    arch="ascend950pr",
    uncertainty_space=UncertaintySpace(),
    options=FrontierSearchOptions(top_k=3),
)
for candidate in result.candidates:
    print(candidate.frontier_rank, candidate.execution_mode.value)
```

For details on scenario modeling, candidate generation, dominance policies,
and JSON-RPC usage, see
[Frontier exploration](docs/FRONTIER_EXPLORATION.md). The proposed cross-tool
profile-unification design is documented in
[Unified SoC meta-language](docs/unified-soc-meta-language.md).
