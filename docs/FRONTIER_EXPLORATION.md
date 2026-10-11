# Parametric frontier exploration

> **Back to the project overview:** [PipeSim README](../README.md)

The frontier engine searches `AlgorithmFlow` objects. It emits typed flows in
JSON and does not generate kernel source. It reuses the three-tier evaluator for
SRAM feasibility, hierarchical rooflines, and the max-plus schedule.

```python
from ascend_pipesim import (
    AlgorithmFlow, EpilogueOp, FrontierSearchOptions, OutputPath,
    UncertaintySpace, find_optimal_frontiers,
)

flow = AlgorithmFlow.gemm(
    m=2048, n=2048, k=4096,
    tile=(128, 256, 64), subtile=(64, 64, 32), cores=32,
    epilogue=(EpilogueOp("silu", "vector", ops_per_element=4),),
    output_path=OutputPath.FIXPIPE_UB_MTE3,
)
result = find_optimal_frontiers(
    flow, "ascend950pr", UncertaintySpace(),
    FrontierSearchOptions(top_k=5, max_evaluations=240),
)
for candidate in result.candidates:
    print(candidate.execution_mode, candidate.predicted_bounds.to_dict())
```

To use JSON, write `flow.to_dict()` to a file and run:

```text
npu-pipesim frontier --flow-file flow.json --arch ascend950pr \
  --uncertainty uncertainty.json --topk 5 --output frontier.json
```

The same unified command is available as a Python module:

```text
python -m ascend_pipesim frontier --flow-file flow.json --arch ascend950pr \
  --uncertainty uncertainty.json --topk 5 --output frontier.json
```

An uncertainty file contains four `[low, high]` intervals:

```json
{
  "dma_latency_cycles": [180, 320],
  "effective_hbm_bandwidth_fraction": [0.6, 1.0],
  "cross_core_sync_cycles": [20, 80],
  "simt_cache_miss_rate_drift": [-0.05, 0.2]
}
```

The evaluator changes DMA startup and HBM share without changing Cube or
Vector throughput. SIMT drift is an absolute change to DataCache miss
probability, clamped to `[0, 1]`. MIX handoffs use one producer-ready and one
consumer-credit synchronization per streamed outer tile. A two-slot ring can
overlap handoffs; a one-slot ring serializes them. `mix_split_m` and
`mix_split_n` bisect the respective spatial tile across two Vector cores.
Per-core UB footprints pad the bisected dimensions to 16 elements before
charging capacity; Vector service reflects the same local tile fraction.
A GM ring incurs an extra workspace round trip. The output flow records the
graph topology and ring space in `execution_graph`.
For long tile streams, L1 depth bounds the number of outstanding DMA loads;
triple buffering can hide more launch latency than double buffering. A one or
two tile run still pays the sequential prologue and drain.
The tile sweep visits high arithmetic-intensity outer tiles first. For each
outer tile it enumerates aligned, exact-divisor subtiles and orders them by
the predicted MTE1/Cube service balance.

All 16 corners of the four-dimensional uncertainty box, its center, and eight
axial points are evaluated. The current model is monotone in the four
environmental costs, so corner timings bound the declared box. The default
`scenario` policy removes every candidate dominated on these 25 points.
Because an interior crossover could be missed, `--dominance-policy
certified_interval` provides conservative continuous-box pruning: it discards
a candidate only when another candidate's worst-case bound beats its best-case
bound. This may retain a dominated candidate when intervals overlap.
`truncated` reports when the configured evaluation budget ended the search.
Latencies, cache parameters, and IPC costs remain analytical estimates until
calibrated against measurements for a specific device and workload.

The long-lived JSON-RPC server accepts the same search through `method:
"frontier"`. Pass an `AlgorithmFlow.to_dict()` object in `params.flow`, an
uncertainty interval object in `params.uncertainty`, and search controls such
as `topk` and `max_evaluations` in `params`.

---

**Project overview:** [README.md](../README.md)
