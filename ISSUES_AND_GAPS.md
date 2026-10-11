# PipeSim — issues and gaps

Found while doing pre-code modeling for the five DeepSeek-V4-Flash Stage-1
kernels on `--arch ascend950pr` (CANN 9.2.0-beta.2, `dav-3510`).

## 1. (worked as intended) Tier-1 feasibility rejected the obvious KER-01 tiling

Worth recording because it changed the design before any code existed. Modeling
the mHC projection as a `vector_gemv` over whole `phi` rows
(`rows=24, columns=16384, dtype=fp32`) is infeasible:

```
KER-01-rejected-wholerow
  valid=False  limiting_stage=MTE2_STARVATION
  violations: [{'code': 'UB_OVERFLOW', 'message':
    'UB needs 262208 B at depth 2 but only 221184 B exist'}]
```

Splitting each row into 4 chunks of 4096 FP32 elements is feasible at
65,600 B, and that is what the kernel implements. The `ascend950pr` pool
correctly carries the 216 KiB usable budget (221,184 B) rather than the
physical 256 KiB, which is what made the rejection meaningful.

## 2. (resolved) Per-row SIMT VF dispatch modeling

Resolved on 2026-10-11: `SimtWorkload.vf_dispatches_per_phase` counts
`asc_vf_call` boundaries per outer tile and `SimtHardware.
vf_mode_switch_penalty_cycles` (64, window 30..80) prices each one inside
the VECTOR stage of the max-plus schedule; `vector_gemv(..., simt=...)`
lowers to `hybrid_simt_simd` automatically. The original gap as written:

`SimtWorkload` attaches a SIMT phase to a `vector_gemv` or `elementwise` flow
with `per_outer_tile`, which models "warp prefill/packing fused with a tile".
Two of the Stage-1 kernels instead dispatch one `__simt_vf__` per output row,
with the dense reduction staying in SIMD, so the cost structure is
`rows x (VF switch + threads strided over cols)` plus `rows x` a SIMD
reduction.

There is no way to express that: `per_outer_tile=True` charges the SIMT work
once per outer tile, and the mode-switch drain (~30-80 cycles on a shared
Vector ALU, which at one switch per row over 2048 rows is not negligible) does
not appear in the max-plus schedule at all. The reported `limiting_stage` for
these flows is therefore driven only by the SIMD side.

*Suggested fix:* a `switches_per_outer_tile` (or `vf_dispatches`) field on
`SimtWorkload`, plus a per-switch drain penalty from the topology, so the
hybrid alternation cost is in the schedule. The topology already carries the
SIMT parameters needed (`warp_size`, `warp_schedulers`,
`ub_shared_bandwidth_bpc_estimate`).

## 3. (minor, API) Report accessors are easy to get wrong

`get_architecture("ascend950pr")` returns an object whose capacity lives at
`arch.pool("UB").capacity_bytes`; `soc_name`, `memory_domains` and
`pool(...).size_bytes` (the names used in `profiles/ascend950pr.json` and in
the README's prose) all raise `AttributeError`. Likewise
`report.to_dict()["sram_footprints"]["UB"]` is a dict whose byte figure is
under `bytes_used`, while `code_generation_hints` is a dict, not the list the
README's wording implies.

*Suggested fix:* either align the attribute names with the profile JSON keys or
document the accessor names in the README's example.

## 4. (minor) `limiting_stage` is `MTE2_STARVATION` for every decode-shape GEMV

All four GEMV-shaped Stage-1 flows report `MTE2_STARVATION`, which is correct
(they are weight-streaming decode GEMVs, arithmetic intensity well under 1
FLOP/byte) but not very actionable. A hint distinguishing "starved because the
weight stream is the whole problem" from "starved because the tiling is wrong"
would be more useful; the first is a property of decode, the second is fixable.
