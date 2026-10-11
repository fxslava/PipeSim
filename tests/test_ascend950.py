"""Ascend 950 manifest, packed dtype, and v3 capacity regressions."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from ascend_pipesim.autotiler import _outer_grid
from ascend_pipesim.cli import main
from ascend_pipesim.codegen import synthesize_kernel
from ascend_pipesim.dsl import AlgorithmFlow, BufferPlan, SimtWorkload
from ascend_pipesim.evaluator import evaluate
from ascend_pipesim.topology import DType, architecture_from_dict, get_architecture


def test_default_950_bins_and_precision_ratios():
    pr = get_architecture("950pr")
    dt = get_architecture("ascend950")
    assert (pr.cube_cores, pr.vector_cores, pr.hbm_bandwidth_gbps) == (32, 64, 1600.0)
    assert (dt.cube_cores, dt.vector_cores, dt.hbm_bandwidth_gbps) == (36, 72, 4000.0)
    for arch in (pr, dt):
        assert arch.pool("L0C").capacity_bytes == 262144
        assert arch.pool("UB").capacity_bytes == (221184 if arch is pr else 524288)
        assert arch.l2_capacity_bytes == 134217728
        assert (
            arch.cube.profile("mxfp4").macs_per_cycle
            == 4 * arch.cube.profile("fp16").macs_per_cycle
        )
        assert (
            arch.cube.profile("hif8").macs_per_cycle
            == 2 * arch.cube.profile("fp16").macs_per_cycle
        )


def test_packed_dtype_and_aliases():
    assert DType.parse("MX-FP4") is DType.MXFP4
    assert DType.parse("hi_f8") is DType.HIF8
    assert DType.parse("mx8") is DType.MXFP8
    assert DType.MXFP4.size_bytes == 0.5
    assert DType.MXFP8.size_bytes == DType.HIF8.size_bytes == 1


def test_l0c_pruning_uses_selected_architecture():
    def candidates(arch):
        return _outer_grid(
            [192],
            [192],
            [64],
            k0=16,
            itemsize=2,
            l0c_capacity=arch.pool("L0C").capacity_bytes,
            l1_capacity=arch.pool("L1").capacity_bytes,
            l1_depth=1,
            l0c_depth=1,
        )[0]

    assert candidates(get_architecture("ascend950dt")) == [(192, 192, 64)]
    assert candidates(get_architecture("ascend910b")) == []


def test_codegen_v3_budget_and_optional_bufferid():
    flow = AlgorithmFlow.gemm(
        m=192,
        n=192,
        k=64,
        tile=(192, 192, 64),
        subtile=(64, 64, 32),
        buffers=BufferPlan.single(),
    )
    report = evaluate(flow, "ascend950dt")
    assert report.is_valid
    bundle = synthesize_kernel(report, flow, bufferid_sync=True)
    assert "static_assert(CO1_BUFFER_BYTES * 1 <= 262144" in bundle.kernel_tiling_h
    assert "get_buf(PIPE_M, 0);" in bundle.kernel_cpp
    assert "rel_buf(PIPE_M, 0);" in bundle.kernel_cpp
    assert bundle.meta["bufferid_sync"] is True
    ub_flow = AlgorithmFlow.gemm(
        m=192,
        n=192,
        k=64,
        tile=(192, 192, 64),
        subtile=(64, 64, 32),
        buffers=BufferPlan.single(),
        output_path="fixpipe_ub_mte3",
    )
    ub_bundle = synthesize_kernel(evaluate(ub_flow, "ascend950dt"), ub_flow)
    assert "static_assert(UB_EPILOGUE_BYTES <= 524288" in ub_bundle.kernel_tiling_h
    with pytest.raises(ValueError, match="DaVinci v3"):
        synthesize_kernel(report, flow, "ascend910b", bufferid_sync=True)


def test_json_profiles_are_standalone_v3():
    profiles = Path(__file__).resolve().parents[1] / "profiles"
    for name, cores, vectors, bandwidth in (
        ("ascend950pr", 32, 64, 1600),
        ("ascend950dt", 36, 72, 4000),
    ):
        p = json.loads((profiles / f"{name}.json").read_text(encoding="utf-8"))
        assert p["isa_generation"] == "davinci_v3"
        assert "base_profile" not in p and not p.get("commercial_label")
        assert p["system"]["cube_cores"] == cores
        assert p["system"]["vector_cores"] == vectors
        assert p["system"]["hbm_bandwidth_gbps"] == bandwidth
        assert p["memory_domains"]["L0C"]["size_bytes"] == 262144
        assert p["memory_domains"]["UB"]["size_bytes"] == (
            221184 if name == "ascend950pr" else 524288
        )
        assert p["transfer_channels"]["MTE3"]["min_burst_bytes"] == 512
        assert {"NDDMA", "CV_BYPASS"} <= p["transfer_channels"].keys()


def test_950pr_arch35_transfer_and_ub_budget():
    pr = get_architecture("ascend950pr")
    assert pr.hbm_bandwidth_gbps == 1600.0
    assert pr.hbm_bandwidth_bpc_total() == pytest.approx(969.697, rel=1e-5)
    assert pr.pipe("MTE2").base_latency_cycles == 200
    assert pr.pipe("MTE3").min_burst_bytes == 512
    assert pr.pool("UB").capacity_bytes == 216 * 1024
    assert architecture_from_dict(pr.to_dict()).simt == pr.simt
    tuned = architecture_from_dict(
        {"base": "ascend950pr", "simt": {**pr.simt.to_dict(), "cache_hit_cycles": 12.0}}
    )
    assert tuned.simt.cache_hit_cycles == 12.0


def test_mte3_short_store_pays_minimum_burst_per_tile():
    flow = AlgorithmFlow.elementwise(elements=32, tile_elements=16, dtype="fp16")
    report = evaluate(flow, "ascend950pr")
    assert report.traffic["mte3_bytes"] == 2 * 512
    store = next(s for s in report.schedule.stages if s.name == "MTE3")
    assert store.total_bytes == 1024
    tail = AlgorithmFlow.elementwise(elements=272, tile_elements=256, dtype="fp16")
    assert evaluate(tail, "ascend950pr").traffic["mte3_bytes"] == 1024


@pytest.mark.parametrize("m, steps", [(64, 1), (128, 2)])
def test_short_runs_pay_pipeline_startup_service(m, steps):
    flow = AlgorithmFlow.gemm(
        m=m,
        n=64,
        k=32,
        tile=(64, 64, 32),
        subtile=(64, 64, 32),
        buffers=BufferPlan.double(),
    )
    report = evaluate(flow, "ascend950pr")
    schedule = report.schedule
    assert schedule.steps == steps
    parallel = [s.cycles_per_step for s in schedule.stages if s.overlapped]
    startup = sum(parallel) - max(parallel)
    arch = get_architecture("ascend950pr")
    assert schedule.fill_cycles == pytest.approx(
        arch.pipe("MTE2").base_latency_cycles
        + arch.pipe("MTE1").base_latency_cycles
        + arch.cube.pipeline_latency_cycles
        + startup
    )
    assert schedule.total_cycles >= sum(s.cycles_per_step for s in schedule.stages)
    assert schedule.serialized_cycles >= schedule.total_cycles


def _mhc_simt_workload() -> SimtWorkload:
    return SimtWorkload(
        threads=256,
        instructions_per_thread=24,
        memory_ops_per_thread=2,
        bytes_per_access=4,
        stride_bytes=4,
        working_set_bytes=64 * 1024,
        branch_efficiency=0.75,
    )


def test_fused_simt_prefill_and_simd_gemv_reports_shared_resources():
    # Header constants: 24 Phi rows and 1024 features per core.
    flow = AlgorithmFlow.vector_gemv(
        rows=24,
        columns=1024,
        tile_rows=4,
        dtype="fp32",
        simt=_mhc_simt_workload(),
        name="mhc_fused_sinkhorn_gemv",
    )
    report = evaluate(flow, "ascend950pr")
    assert report.is_valid
    assert report.simt is not None
    assert report.simt.cache_miss_rate > 0
    assert report.simt.coalescing_factor >= 1
    assert report.schedule.vector_alu_utilization_pct > 0
    assert report.to_dict()["vector_alu_utilization_pct"] > 0
    assert report.traffic["simt_datacache_fill_bytes"] > 0
    assert report.to_dict()["simt"]["cache_miss_rate"] > 0
    assert AlgorithmFlow.from_dict(flow.to_dict()).simt == flow.simt


def test_mhc_gemv_tiling_sweep_retains_short_tile_bubbles():
    reports = {}
    for rows_per_tile in (4, 12, 24):
        for plan_name, plan in (
            ("single", BufferPlan.single()),
            ("double", BufferPlan.double()),
        ):
            flow = AlgorithmFlow.vector_gemv(
                rows=24,
                columns=1024,
                tile_rows=rows_per_tile,
                dtype="fp32",
                buffers=plan,
                simt=_mhc_simt_workload(),
                name="mhc_fused_sinkhorn_gemv",
            )
            reports[(rows_per_tile, plan_name)] = evaluate(flow, "ascend950pr")
    arch = get_architecture("ascend950pr")
    assert reports[(24, "double")].schedule.steps == 1
    assert reports[(12, "double")].schedule.steps == 2
    assert (
        reports[(24, "double")].schedule.fill_cycles
        >= arch.pipe("MTE2").base_latency_cycles
    )
    assert (
        reports[(12, "double")].schedule.fill_cycles
        >= arch.pipe("MTE2").base_latency_cycles
    )
    assert reports[(4, "single")].total_cycles > reports[(4, "double")].total_cycles
    assert all(r.simt.cache_miss_rate > 0 for r in reports.values())


def test_simt_divergence_coalescing_and_ub_port_contention():
    base = _mhc_simt_workload()
    flow = AlgorithmFlow.vector_gemv(
        rows=24, columns=1024, tile_rows=4, dtype="fp32", simt=base
    )
    arch = get_architecture("arch35")
    coalesced = evaluate(flow, arch)
    strided = evaluate(flow.evolve(simt=replace(base, stride_bytes=128)), arch)
    divergent = evaluate(flow.evolve(simt=replace(base, branch_efficiency=0.5)), arch)
    narrow_ub = evaluate(
        flow,
        arch.evolve(simt=replace(arch.simt, ub_shared_bandwidth_bpc=32.0)),
    )
    assert strided.simt.cache_transactions > coalesced.simt.cache_transactions
    assert strided.simt.coalescing_factor > coalesced.simt.coalescing_factor
    assert divergent.simt.vector_alu_cycles > coalesced.simt.vector_alu_cycles
    assert (
        narrow_ub.schedule.ub_port_cycles_per_step
        > coalesced.schedule.ub_port_cycles_per_step
    )
    assert narrow_ub.total_cycles > coalesced.total_cycles


def test_simt_register_occupancy_limit_and_spill_cost():
    base = _mhc_simt_workload()
    no_spill = AlgorithmFlow.vector_gemv(
        rows=24,
        columns=1024,
        tile_rows=4,
        dtype="fp32",
        simt=replace(base, threads=2048, registers_per_thread=16),
    )
    spill = no_spill.evolve(simt=replace(no_spill.simt, registers_per_thread=40))
    lean = evaluate(no_spill, "ascend950pr")
    heavy = evaluate(spill, "ascend950pr")
    assert lean.simt.register_limit_per_thread == 16
    assert lean.simt.spill_bytes == 0
    assert heavy.simt.spill_bytes > 0
    assert heavy.simt.vector_alu_cycles > lean.simt.vector_alu_cycles
    with pytest.raises(ValueError, match="exceeds"):
        evaluate(spill.evolve(simt=replace(spill.simt, threads=2049)), "ascend950pr")


def test_probe_cli_accepts_fused_simt_gemv_flow_file(tmp_path, capsys):
    flow = AlgorithmFlow.vector_gemv(
        rows=24,
        columns=1024,
        tile_rows=4,
        dtype="fp32",
        simt=_mhc_simt_workload(),
    )
    path = tmp_path / "mhc_flow.json"
    path.write_text(json.dumps(flow.to_dict()), encoding="utf-8")
    assert main(["probe", "--flow-file", str(path), "--arch", "ascend950pr"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["simt"]["cache_miss_rate"] > 0
    assert payload["vector_alu_utilization_pct"] > 0
    assert (
        main(
            ["probe", "--flow-file", str(path), "--arch", "arch35", "--format", "text"]
        )
        == 0
    )
    assert "SIMT DataCache miss rate" in capsys.readouterr().out
