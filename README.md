# Ascend PipeSim

PipeSim is an analytical simulator for Ascend tiled kernels. `AlgorithmFlow`
describes the operator, tiles, buffers, and execution graph. Architecture
profiles supply memory capacities, transfer engines, Cube and Vector rates,
and optional SIMT assumptions.

The parametric frontier engine evaluates flow alternatives across bounded DMA,
HBM, cross-core synchronization, and SIMT cache conditions. It returns JSON
flow specifications for later kernel engineering. See
[frontier exploration](docs/FRONTIER_EXPLORATION.md) for the API and CLI.

```text
python -m pytest
python -m ascend_pipesim frontier --flow-file flow.json --arch ascend950pr
```

Install the package with `pip install -e .`, or set `PYTHONPATH=src` when
running directly from the checkout.
