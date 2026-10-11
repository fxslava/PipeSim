"""Regression checks for parametric DSL frontier exploration."""

import json
from io import StringIO

import pytest

from ascend_pipesim import (
    AlgorithmFlow,
    BufferPlan,
    DType,
    EpilogueOp,
    ExecutionGraph,
    ExecutionMode,
    FrontierSearchOptions,
    Interval,
    OutputPath,
    SimtWorkload,
    TensorSpec,
    UncertaintySpace,
    evaluate,
    evaluate_parametric,
    find_optimal_frontiers,
)
from ascend_pipesim import cli
from ascend_pipesim.cli_or_api import main
from ascend_pipesim.codegen import synthesize_kernel
from ascend_pipesim.frontier_search import _candidate_flows
from ascend_pipesim.parametric_evaluator import evaluate_scenario
from ascend_pipesim.topology import get_architecture


def _gemm():
    return AlgorithmFlow.gemm(
        m=128,
        n=128,
        k=128,
        tile=(64, 64, 64),
        subtile=(32, 32, 32),
        cores=4,
        output_path=OutputPath.FIXPIPE_UB_MTE3,
        epilogue=(EpilogueOp("silu", "vector", ops_per_element=5),),
    )


def test_uncertainty_envelope_and_small_tile_fill():
    flow = _gemm().with_buffers(BufferPlan.single())
    space = UncertaintySpace(
        Interval(180, 320),
        Interval(0.6, 1.0),
        Interval(20, 80),
        Interval(-0.1, 0.2),
    )
    bounded = evaluate_parametric(flow, "ascend950pr", space)
    assert bounded.feasible
    assert len(bounded.samples) == 25
    assert bounded.best_case_us <= bounded.nominal_us <= bounded.worst_case_us
    assert bounded.nominal.report.schedule.fill_cycles >= 250
    assert bounded.nominal.report.roofline.level("L1_MTE1")
    assert bounded.nominal.report.roofline.level("L0C_FIXPIPE")


def test_mix_credit_pipeline_and_sync_sensitivity():
    base = _gemm()
    mix = base.evolve(
        execution_graph=ExecutionGraph(
            ExecutionMode.MIX_SPLIT_M, ring_slots=2, vector_consumers=2
        )
    )
    assert AlgorithmFlow.from_dict(mix.to_dict()) == mix
    low = UncertaintySpace(cross_core_sync_cycles=Interval(20, 20)).nominal()
    high = UncertaintySpace(cross_core_sync_cycles=Interval(80, 80)).nominal()
    fast = evaluate_scenario(mix, "ascend950pr", low).report
    slow = evaluate_scenario(mix, "ascend950pr", high).report
    assert fast.is_valid
    assert any(stage.name == "IPC" for stage in fast.schedule.stages)
    assert slow.total_cycles > fast.total_cycles
    assert slow.schedule.fill_cycles > fast.schedule.fill_cycles


def test_mix_split_axis_and_ub_ring_capacity():
    flow = AlgorithmFlow.gemm(
        m=96,
        n=128,
        k=32,
        tile=(48, 64, 32),
        subtile=(16, 16, 16),
        output_path=OutputPath.FIXPIPE_UB_MTE3,
        epilogue=(
            EpilogueOp(
                "silu",
                "vector",
                TensorSpec("scale", (128,), DType.FP32),
                ops_per_element=20,
            ),
        ),
    )
    point = UncertaintySpace().nominal()
    split_m = flow.evolve(
        execution_graph=ExecutionGraph(ExecutionMode.MIX_SPLIT_M, 2, "UB", 2)
    )
    split_n = flow.evolve(
        execution_graph=ExecutionGraph(ExecutionMode.MIX_SPLIT_N, 2, "UB", 2)
    )
    m = evaluate_scenario(split_m, "ascend950pr", point).report
    n = evaluate_scenario(split_n, "ascend950pr", point).report
    assert m.schedule.stage("VECTOR").cycles_per_step == pytest.approx(
        (4 / 3) * n.schedule.stage("VECTOR").cycles_per_step
    )
    assert m.sram_footprints["UB"].breakdown["C_tile"] == 32 * 64 * 2
    assert n.sram_footprints["UB"].breakdown["C_tile"] == 48 * 32 * 2
    assert m.sram_footprints["UB"].breakdown["scale"] == 64 * 4
    assert n.sram_footprints["UB"].breakdown["scale"] == 32 * 4

    large = AlgorithmFlow.gemm(
        m=256,
        n=256,
        k=16,
        tile=(256, 256, 16),
        subtile=(16, 16, 16),
        buffers=BufferPlan.single(),
        output_path=OutputPath.FIXPIPE_UB_MTE3,
    ).evolve(execution_graph=ExecutionGraph(ExecutionMode.MIX_SPLIT_M, 2, "UB", 2))
    report = evaluate_scenario(large, "ascend950pr", point).report
    assert report.is_valid
    assert report.sram_footprints["UB"].bytes_used == 128 * 256 * 2 * 2
    larger_ring = large.evolve(
        execution_graph=ExecutionGraph(ExecutionMode.MIX_SPLIT_M, 4, "UB", 2)
    )
    overflow = evaluate_scenario(larger_ring, "ascend950pr", point).report
    assert not overflow.is_valid
    assert any(v.code == "UB_OVERFLOW" for v in overflow.violations)


def test_frontier_selects_mix_when_vector_epilogue_dominates():
    flow = AlgorithmFlow.gemm(
        m=256,
        n=256,
        k=64,
        tile=(64, 64, 64),
        subtile=(32, 32, 32),
        output_path=OutputPath.FIXPIPE_UB_MTE3,
        epilogue=(EpilogueOp("custom", "vector", ops_per_element=1000),),
    )
    result = find_optimal_frontiers(
        flow,
        "ascend950pr",
        UncertaintySpace(),
        FrontierSearchOptions(max_evaluations=36, max_tile_values=2),
    )
    assert result.candidates
    assert result.candidates[0].execution_mode in (
        ExecutionMode.MIX_SPLIT_M,
        ExecutionMode.MIX_SPLIT_N,
    )


def test_triple_l1_hides_repeated_dma_but_not_single_tile_startup():
    flow = AlgorithmFlow.gemm(
        m=1024,
        n=32,
        k=32,
        tile=(16, 32, 32),
        subtile=(16, 32, 32),
    )
    point = UncertaintySpace(dma_latency_cycles=Interval(320, 320)).nominal()
    double = evaluate_scenario(
        flow.with_buffers(BufferPlan.double()), "ascend950pr", point
    )
    triple = evaluate_scenario(
        flow.with_buffers(BufferPlan.triple()), "ascend950pr", point
    )
    assert triple.cycles < double.cycles
    one_tile = AlgorithmFlow.gemm(
        m=16,
        n=32,
        k=32,
        tile=(16, 32, 32),
        subtile=(16, 32, 32),
    )
    one_double = evaluate_scenario(
        one_tile.with_buffers(BufferPlan.double()), "ascend950pr", point
    )
    one_triple = evaluate_scenario(
        one_tile.with_buffers(BufferPlan.triple()), "ascend950pr", point
    )
    assert one_double.report.schedule.fill_cycles >= 320
    assert one_double.cycles == pytest.approx(one_triple.cycles)


def test_simt_cache_drift_and_hybrid_frontier():
    simt = SimtWorkload(
        threads=64,
        instructions_per_thread=12,
        memory_ops_per_thread=4,
        bytes_per_access=4,
        stride_bytes=16,
        working_set_bytes=64 * 1024,
    )
    flow = AlgorithmFlow.vector_gemv(
        rows=64, columns=128, tile_rows=16, simt=simt
    ).evolve(execution_graph=ExecutionGraph(ExecutionMode.HYBRID_SIMT_SIMD))
    low = UncertaintySpace(simt_cache_miss_rate_drift=Interval(0, 0)).nominal()
    high = UncertaintySpace(simt_cache_miss_rate_drift=Interval(0.2, 0.2)).nominal()
    a = evaluate_scenario(flow, "ascend950pr", low).report
    b = evaluate_scenario(flow, "ascend950pr", high).report
    assert b.simt.cache_miss_rate > a.simt.cache_miss_rate
    assert b.total_cycles > a.total_cycles
    result = find_optimal_frontiers(
        flow,
        "ascend950pr",
        UncertaintySpace(),
        FrontierSearchOptions(max_evaluations=9, max_tile_values=3),
    )
    assert result.candidates
    assert all(
        c.execution_mode is ExecutionMode.HYBRID_SIMT_SIMD for c in result.candidates
    )


def test_frontier_json_cli_roundtrip(tmp_path):
    template = _gemm()
    source = tmp_path / "flow.json"
    output = tmp_path / "frontier.json"
    source.write_text(json.dumps(template.to_dict()), encoding="utf-8")
    assert (
        main(
            [
                str(source),
                "--arch",
                "ascend950pr",
                "--top-k",
                "3",
                "--max-evaluations",
                "24",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["frontier"]
    assert data["dominance_policy"] == "scenario"
    assert len(data["frontier"]) <= 3
    for item in data["frontier"]:
        flow = AlgorithmFlow.from_dict(item["flow"])
        assert flow.execution_graph.mode.value == item["execution_mode"]
        assert (
            item["predicted_bounds"]["worst_case_us"]
            >= item["predicted_bounds"]["best_case_us"]
        )

    certified = find_optimal_frontiers(
        template,
        "ascend950pr",
        UncertaintySpace(),
        FrontierSearchOptions(
            max_evaluations=12, dominance_policy="certified_interval"
        ),
    )
    assert certified.dominance_policy == "certified_interval"
    assert certified.candidates


def test_invalid_uncertainty_rejected():
    with pytest.raises(ValueError):
        UncertaintySpace(effective_hbm_bandwidth_fraction=Interval(0, 1))


def test_search_starts_at_high_intensity_and_uses_dividing_subtiles():
    tiny_seed = AlgorithmFlow.gemm(
        m=256,
        n=256,
        k=256,
        tile=(16, 16, 16),
        subtile=(16, 16, 16),
    )
    options = FrontierSearchOptions(
        max_tile_values=3,
        max_subtile_values=3,
        buffer_plans=(BufferPlan.single(),),
    )
    flows = list(_candidate_flows(tiny_seed, get_architecture("ascend950pr"), options))
    assert flows[0].tile.tile_m > tiny_seed.tile.tile_m
    assert flows[0].tile.tile_n > tiny_seed.tile.tile_n
    assert len({flow.tile.as_tuple() for flow in flows}) > 1
    for flow in flows:
        tile = flow.tile
        assert tile.tile_m % tile.sub_m == 0 and tile.sub_m % 16 == 0
        assert tile.tile_n % tile.sub_n == 0 and tile.sub_n % 16 == 0
        assert tile.tile_k % tile.sub_k == 0 and tile.sub_k % 16 == 0


def test_frontier_main_cli_and_rpc(tmp_path, capsys):
    flow = _gemm()
    flow_path = tmp_path / "flow.json"
    space_path = tmp_path / "space.json"
    flow_path.write_text(json.dumps(flow.to_dict()), encoding="utf-8")
    space_path.write_text(json.dumps(UncertaintySpace().to_dict()), encoding="utf-8")
    assert (
        cli.main(
            [
                "frontier",
                "--flow-file",
                str(flow_path),
                "--arch",
                "ascend950pr",
                "--uncertainty",
                str(space_path),
                "--topk",
                "2",
                "--max-evaluations",
                "12",
                "--dominance-policy",
                "scenario",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["frontier"]
    assert len(payload["frontier"]) <= 2

    request = {
        "id": 7,
        "method": "frontier",
        "params": {
            "flow": flow.to_dict(),
            "arch": "ascend950pr",
            "uncertainty": UncertaintySpace().to_dict(),
            "topk": 2,
            "max_evaluations": 12,
        },
    }
    output = StringIO()
    cli.serve_rpc(StringIO(json.dumps(request) + "\n"), output)
    response = json.loads(output.getvalue())
    assert response["ok"] is True
    assert response["result"]["frontier"]

    assert (
        cli.main(
            [
                "frontier",
                "--gemm",
                "128,128,128",
                "--arch",
                "ascend950pr",
                "--topk",
                "1",
                "--max-evaluations",
                "6",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["frontier"]

    gemm_request = {
        "id": 8,
        "method": "frontier",
        "params": {
            "gemm": [128, 128, 128],
            "arch": "ascend950pr",
            "topk": 1,
            "max_evaluations": 6,
        },
    }
    output = StringIO()
    cli.serve_rpc(StringIO(json.dumps(gemm_request) + "\n"), output)
    assert json.loads(output.getvalue())["ok"] is True


def test_codegen_rejects_mix_and_simt_flows():
    mix = _gemm().evolve(
        execution_graph=ExecutionGraph(ExecutionMode.MIX_SPLIT_M, 2, "UB", 2)
    )
    with pytest.raises(NotImplementedError, match="multi-core MIX"):
        synthesize_kernel(evaluate(mix, "ascend950pr"), mix, "ascend950pr")

    simt = SimtWorkload(
        threads=32,
        instructions_per_thread=4,
        memory_ops_per_thread=2,
        bytes_per_access=4,
        stride_bytes=4,
        working_set_bytes=4096,
    )
    hybrid = AlgorithmFlow.vector_gemv(
        rows=32, columns=32, tile_rows=16, simt=simt
    ).evolve(execution_graph=ExecutionGraph(ExecutionMode.HYBRID_SIMT_SIMD))
    with pytest.raises(NotImplementedError, match="SIMT warp"):
        synthesize_kernel(evaluate(hybrid, "ascend950pr"), hybrid, "ascend950pr")
