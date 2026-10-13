"""Analytical Ascend 950PR pilot for one DeepSeek-V4 MoE expert."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ascend_pipesim import (
    AlgorithmFlow,
    BufferLifetime,
    BufferPlan,
    FrontierSearchOptions,
    UncertaintySpace,
    evaluate,
    evaluate_parametric,
    find_optimal_frontiers,
    get_architecture,
)


ARCH = "ascend950pr"
AIVS = 64
MXFP4_GROUP = 32
MXFP4_SCALE_BYTES = 1


@dataclass(frozen=True)
class StagePlan:
    name: str
    aiv_rows: int
    k: int
    note: str


STAGES = (
    StagePlan("W1_gate", 32, 4096, "32 output channels per AIV"),
    StagePlan("W3_up", 32, 4096, "32 output channels per AIV"),
    StagePlan("W2_down", 64, 2048, "64 output channels per AIV"),
)


def _gemv_template(stage: StagePlan) -> AlgorithmFlow:
    return AlgorithmFlow.vector_gemv(
        rows=stage.aiv_rows,
        columns=stage.k,
        tile_rows=stage.aiv_rows,
        dtype="mxfp4",
        input_dtype="bf16",
        out_dtype="bf16",
        weight_scale_bytes_per_group=MXFP4_SCALE_BYTES,
        weight_scale_group_size=MXFP4_GROUP,
        input_preloaded=stage.name == "W3_up",
        buffers=BufferPlan.double(),
        cores=AIVS,
        name=f"dsv4_{stage.name.lower()}",
    )


def _full_ub_lifetimes(
    w1_tile_rows: int, w2_tile_rows: int
) -> tuple[BufferLifetime, ...]:
    """One AIV's live ranges across the complete producer/consumer graph."""
    w1_weight_slot = w1_tile_rows * (
        4096 * 0.5 + 4096 // MXFP4_GROUP * MXFP4_SCALE_BYTES
    )
    w2_weight_slot = w2_tile_rows * (
        2048 * 0.5 + 2048 // MXFP4_GROUP * MXFP4_SCALE_BYTES
    )
    return (
        BufferLifetime("token_input", 4096 * 2, 0, 3),
        BufferLifetime("weight_ping", int(w1_weight_slot), 1, 3),
        BufferLifetime("weight_pong", int(w1_weight_slot), 1, 3),
        BufferLifetime("gate_bf16", 32 * 2, 1, 5),
        BufferLifetime("up_bf16", 32 * 2, 3, 5),
        BufferLifetime("swiglu_output", 32 * 2, 5, 6, alias_of="gate_bf16"),
        BufferLifetime("w2_weight_ping", int(w2_weight_slot), 7, 8),
        BufferLifetime("w2_weight_pong", int(w2_weight_slot), 7, 8),
        BufferLifetime("gathered_activation", 2048 * 2, 7, 8),
        BufferLifetime("down_accumulator", w2_tile_rows * 2, 8, 8),
    )


def _activation_flow() -> AlgorithmFlow:
    flow = AlgorithmFlow.elementwise(
        elements=32,
        tile_elements=32,
        dtype="bf16",
        out_dtype="bf16",
        n_inputs=2,
        ops_per_element=8,
        buffers=BufferPlan.double(),
        name="dsv4_clamped_swiglu",
    ).evolve(
        cores=AIVS,
        ub_lifetimes=(
            BufferLifetime("gate", 32 * 2, 0, 3),
            BufferLifetime("up", 32 * 2, 0, 2),
            BufferLifetime("clamped_gate", 32 * 2, 1, 2, alias_of="gate"),
            BufferLifetime("swish_gate", 32 * 2, 2, 2, alias_of="gate"),
            BufferLifetime("bf16_activation", 32 * 2, 3, 3, alias_of="gate"),
        ),
    )
    return flow


def run() -> dict[str, Any]:
    arch = get_architecture(ARCH)
    options = FrontierSearchOptions(
        top_k=3,
        max_evaluations=48,
        max_tile=64,
        max_tile_values=3,
        max_subtile_values=1,
        buffer_plans=(BufferPlan.double(),),
    )
    projections: dict[str, dict[str, Any]] = {}
    selected_flows: dict[str, AlgorithmFlow] = {}
    bank_conflicts: list[str] = []
    for stage in STAGES:
        frontier = find_optimal_frontiers(
            _gemv_template(stage), arch, UncertaintySpace(), options
        )
        if not frontier.candidates:
            raise RuntimeError(f"No feasible frontier candidate for {stage.name}")
        best = frontier.candidates[0]
        report = evaluate(best.flow, arch)
        if not report.is_valid:
            raise RuntimeError(f"Selected {stage.name} candidate is infeasible")
        bank_conflicts.extend(
            v.code for v in report.violations if v.code == "UB_BANK_CONFLICT"
        )
        selected_flows[stage.name] = best.flow
        projections[stage.name] = {
            "tile_rows": best.flow.tile.tile_m,
            "buffer_depth": best.flow.buffers.ub_depth,
            "steps": report.schedule.steps,
            "mode": best.flow.execution_graph.mode.value,
            "frontier_evaluated": frontier.evaluated,
            "frontier_feasible": frontier.feasible,
            "nominal_hbm_bandwidth_fraction": best.predicted_bounds.nominal.environment.effective_hbm_bandwidth_fraction,
            "cycle_bounds": {
                "best": round(
                    min(sample.cycles for sample in best.predicted_bounds.samples), 1
                ),
                "nominal": round(best.predicted_bounds.nominal.report.total_cycles, 1),
                "worst": round(
                    max(sample.cycles for sample in best.predicted_bounds.samples),
                    1,
                ),
            },
            "report": report.to_dict(),
        }

    activation_flow = _activation_flow()
    activation = evaluate(activation_flow, arch)
    activation_bounds = evaluate_parametric(
        activation_flow, arch, UncertaintySpace()
    )
    if not activation.is_valid or not activation_bounds.feasible:
        raise RuntimeError("SwiGLU activation stage is infeasible")
    bank_conflicts.extend(
        v.code for v in activation.violations if v.code == "UB_BANK_CONFLICT"
    )

    # The explicit ledger includes overlapping W1/W3 outputs and W2 staging.
    lifetime_flow = selected_flows["W1_gate"].evolve(
        ub_lifetimes=_full_ub_lifetimes(
            selected_flows["W1_gate"].tile.tile_m,
            selected_flows["W2_down"].tile.tile_m
        )
    )
    lifetime_report = evaluate(lifetime_flow, arch)
    if not lifetime_report.is_valid:
        raise RuntimeError("The full-flow UB lifetime ledger exceeds capacity")
    bank_conflicts.extend(
        v.code for v in lifetime_report.violations if v.code == "UB_BANK_CONFLICT"
    )
    if bank_conflicts:
        raise RuntimeError("The selected plan has UB bank conflicts")

    activation_cycles = activation_bounds.nominal.report.total_cycles
    sync_cycles = 30.0
    per_aiv_bytes = (
        projections["W1_gate"]["report"]["traffic_bytes"]["mte2_bytes"]
        + projections["W3_up"]["report"]["traffic_bytes"]["mte2_bytes"]
        + projections["W2_down"]["report"]["traffic_bytes"]["mte2_bytes"]
    )
    total_hbm_bytes = per_aiv_bytes * AIVS
    hbm_capacity_bytes_per_cycle = arch.hbm_bandwidth_bpc_total()
    nominal_hbm_fraction = projections["W1_gate"][
        "nominal_hbm_bandwidth_fraction"
    ]
    nominal_cycles = (
        projections["W1_gate"]["cycle_bounds"]["nominal"]
        + projections["W3_up"]["cycle_bounds"]["nominal"]
        + projections["W2_down"]["cycle_bounds"]["nominal"]
        + activation_bounds.nominal.report.total_cycles
        + sync_cycles
    )
    mte2_steady_cycles_per_aiv = sum(
        stage_report["report"]["pipeline"]["stages"]["MTE2"]["cycles_per_step"]
        * stage_report["report"]["pipeline"]["steps"]
        for stage_report in projections.values()
    )
    per_aiv_hbm_share = (
        hbm_capacity_bytes_per_cycle * nominal_hbm_fraction / AIVS
    )
    nominal_steady_mte2_cycles = (
        mte2_steady_cycles_per_aiv / nominal_hbm_fraction
    )

    return {
        "architecture": {
            "name": arch.name,
            "cube_to_aiv_ratio": f"1:{arch.vector_cores_per_cube:g}",
            "aiv_count": arch.vector_cores,
            "usable_ub_bytes_per_aiv": arch.pool("UB").capacity_bytes,
            "l2_line_bytes": arch.l2_line_bytes,
            "l2_sector_bytes": arch.l2_sector_bytes,
            "hbm_bandwidth_gbps": arch.hbm_bandwidth_gbps,
        },
        "execution_graph": [
            "MTE2 streams packed W1 and UE8M0 scales through a two-slot UB ring",
            "AIV SIMD computes W1; retain the 32 BF16 gate outputs in UB",
            "Reuse the same ring for W3; retain 32 BF16 up outputs",
            "In-place clamp [-10,10], swish, multiply by up, cast BF16",
            "MTE3 publishes each 32-element activation shard to a 64-shard GM buffer",
            "64-AIV barrier; each AIV gathers the 2048-element BF16 activation",
            "MTE2 streams W2/scale tiles; AIV SIMD computes 64 down channels",
            "MTE3 writes each output shard",
        ],
        "buffer_protocol": {
            "weight_ring_slots": 2,
            "ring_storage": "per-AIV UB; alternating BufferID 0/1",
            "producer_consumer": "MTE2 acquire/copy -> vector wait/consume -> release slot",
            "cross_aiv_activation": "GM shard store -> 64-AIV ready barrier -> gather -> release",
            "synchronization_cycles_assumed": sync_cycles,
        },
        "tiling": {
            "w1_w3": (
                f"32 output rows per AIV in "
                f"{selected_flows['W1_gate'].tile.tile_m}-row tiles x full K=4096"
            ),
            "w2": (
                f"64 output rows per AIV in "
                f"{selected_flows['W2_down'].tile.tile_m}-row tiles x full K=2048"
            ),
            "weight_buffer_depth": 2,
            "scale_buffer_depth": 2,
            "activation": "32 BF16 values local, overwrite gate buffer",
            "execution_mode": "AIV_ONLY (no Cube tail waste or SIMD/SIMT mode transition)",
        },
        "ub_ledger_bytes_per_aiv": {
            "peak": lifetime_report.sram_footprints["UB"].bytes_used,
            "capacity": lifetime_report.sram_footprints["UB"].capacity_bytes,
            "occupancy_pct": round(
                lifetime_report.sram_footprints["UB"].occupancy_pct, 2
            ),
            "peak_breakdown": lifetime_report.sram_footprints["UB"].breakdown,
            "intervals": [
                life.to_dict()
                for life in _full_ub_lifetimes(
                    selected_flows["W1_gate"].tile.tile_m,
                    selected_flows["W2_down"].tile.tile_m
                )
            ],
        },
        "validation": {
            "full_flow_tier1_valid": lifetime_report.is_valid,
            "ub_overflow": any(
                v.code == "UB_OVERFLOW" for v in lifetime_report.violations
            ),
            "ub_bank_conflict_violations": len(bank_conflicts),
        },
        "stages": projections,
        "activation": {
            "cycles": round(activation_bounds.nominal.report.total_cycles, 1),
            "cycle_bounds": {
                "best": round(
                    min(sample.cycles for sample in activation_bounds.samples), 1
                ),
                "nominal": round(activation_bounds.nominal.report.total_cycles, 1),
                "worst": round(
                    max(sample.cycles for sample in activation_bounds.samples), 1
                ),
            },
            "ub_bytes": activation.sram_footprints["UB"].bytes_used,
            "pipeline": activation.schedule.to_dict(),
        },
        "aggregate_prediction": {
            "cycle_bound_low": round(
                sum(
                    projections[name]["cycle_bounds"]["best"]
                    for name in ("W1_gate", "W3_up", "W2_down")
                )
                + min(sample.cycles for sample in activation_bounds.samples)
                + sync_cycles,
                1,
            ),
            "nominal_cycles": round(nominal_cycles, 1),
            "cycle_bound_high": round(
                sum(
                    projections[name]["cycle_bounds"]["worst"]
                    for name in ("W1_gate", "W3_up", "W2_down")
                )
                + max(sample.cycles for sample in activation_bounds.samples)
                + sync_cycles,
                1,
            ),
            "nominal_us": round(arch.cycles_to_us(nominal_cycles), 3),
            "mte2_hbm_bytes_per_invocation": round(total_hbm_bytes),
            "estimated_hbm_utilization_pct": round(
                100.0 * total_hbm_bytes
                / (nominal_cycles * hbm_capacity_bytes_per_cycle * nominal_hbm_fraction),
                1,
            ),
            "estimated_hbm_utilization_of_peak_pct": round(
                100.0
                * total_hbm_bytes
                / (nominal_cycles * hbm_capacity_bytes_per_cycle),
                1,
            ),
            "nominal_hbm_bandwidth_fraction": nominal_hbm_fraction,
            "steady_state_hbm_utilization_pct": round(
                100.0
                * per_aiv_bytes
                / (nominal_steady_mte2_cycles * per_aiv_hbm_share),
                1,
            ),
            "limiting_stage": "MTE2_BANDWIDTH_BOUND (HBM share)",
            "concurrency_ratio_omega": round(
                projections["W1_gate"]["report"]["concurrency_ratio"], 3
            ),
            "latency_hiding_eta_pct": projections["W1_gate"]["report"][
                "latency_hiding_pct"
            ],
        },
        "model_notes": [
            "Cycles are analytical bounds, not measured device latency.",
            "The per-AIV full-token input gather is charged as a GM-to-UB read in each W2 flow.",
            "The 30-cycle inter-AIV barrier is an explicit model assumption.",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
