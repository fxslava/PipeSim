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


# ---------------------------------------------------------------------------
# DaVinci v3 hybrid VF mode-switch, butterfly shuffle and staging models
# ---------------------------------------------------------------------------


def _ker05_simt_workload(**overrides):
    """KER-05 mxfp4_expert_gemv: one VF dispatch per expert row."""
    base = dict(
        threads=128,
        instructions_per_thread=40,
        memory_ops_per_thread=8,
        bytes_per_access=0.5,  # two threads share one packed MXFP4 byte
        stride_bytes=0.5,
        working_set_bytes=512 * (1024 + 64),
        branch_efficiency=1.0,
        registers_per_thread=40,
        vf_dispatches_per_phase=512,  # one asc_vf_call per expert row
        shuffle_ops_per_thread=5,  # log2(32) butterfly steps
    )
    base.update(overrides)
    return SimtWorkload(**base)


def _ker05_flow(**workload_overrides):
    return AlgorithmFlow.vector_gemv(
        rows=512,
        columns=2048,
        tile_rows=256,
        dtype="fp16",
        simt=_ker05_simt_workload(**workload_overrides),
    )


def test_vf_mode_switch_penalty_field_default_and_window():
    pr = get_architecture("ascend950pr")
    assert pr.simt.vf_mode_switch_penalty_cycles == 64.0
    assert architecture_from_dict(pr.to_dict()).simt == pr.simt
    for bad in (29.9, 80.1):
        with pytest.raises(ValueError, match="30..80"):
            architecture_from_dict(
                {"base": "ascend950pr",
                 "simt": {"vf_mode_switch_penalty_cycles": bad}}
            )
    for edge in (30.0, 80.0):
        tuned = architecture_from_dict(
            {"base": "ascend950pr",
             "simt": {"vf_mode_switch_penalty_cycles": edge}}
        )
        assert tuned.simt.vf_mode_switch_penalty_cycles == edge


def test_hybrid_gemv_pays_vf_mode_switch_per_dispatch():
    with_switches = evaluate(_ker05_flow(vf_dispatches_per_phase=512), "ascend950pr")
    assert with_switches.simt.vf_mode_switch_cycles == pytest.approx(
        with_switches.simt.phases * 512 * 64.0
    )
    without = evaluate(_ker05_flow(vf_dispatches_per_phase=0), "ascend950pr")

    def vector_tau(report):
        return next(s for s in report.schedule.stages if s.name == "VECTOR")

    switch_per_step = (
        with_switches.simt.vf_mode_switch_cycles / with_switches.schedule.steps
    )
    assert vector_tau(with_switches).cycles_per_step == pytest.approx(
        vector_tau(without).cycles_per_step + switch_per_step
    )
    assert with_switches.total_cycles > without.total_cycles
    assert with_switches.traffic["simt_vf_switch_cycles"] == pytest.approx(
        with_switches.simt.vf_mode_switch_cycles
    )


def test_vf_mode_switches_only_charged_in_hybrid_mode():
    from ascend_pipesim.dsl import ExecutionMode

    hybrid = _ker05_flow()
    assert hybrid.execution_graph.mode is ExecutionMode.HYBRID_SIMT_SIMD
    assert evaluate(hybrid, "ascend950pr").simt.vf_mode_switch_cycles > 0

    plain = AlgorithmFlow(
        name="plain",
        op_type="vector_gemv",
        loops=hybrid.loops,
        tile=hybrid.tile,
        a=hybrid.a,
        b=hybrid.b,
        c=hybrid.c,
        simt=_ker05_simt_workload(),
    )
    assert plain.execution_graph.mode is ExecutionMode.STANDALONE_CUBE
    assert evaluate(plain, "ascend950pr").simt.vf_mode_switch_cycles == 0.0


def test_warp_butterfly_shuffles_are_register_resident():
    with_shuffles = evaluate(_ker05_flow(shuffle_ops_per_thread=5), "ascend950pr")
    without = evaluate(_ker05_flow(shuffle_ops_per_thread=0), "ascend950pr")
    simt = with_shuffles.simt
    assert simt.warp_shuffle_cycles == pytest.approx(
        simt.warps * 5 * simt.phases / 4.0
    )
    # Register-resident: identical DataCache traffic with and without.
    assert simt.cache_transactions == pytest.approx(without.simt.cache_transactions)
    assert simt.cache_fill_bytes == pytest.approx(without.simt.cache_fill_bytes)
    assert simt.hbm_fill_bytes == pytest.approx(without.simt.hbm_fill_bytes)


def test_mxfp4_packed_pairs_coalesce_into_single_cache_lines():
    report = evaluate(_ker05_flow(), "ascend950pr")
    simt = report.simt
    # 32 lanes x 0.5 B on a 0.5 B stride span 16 B of one 128 B line, so a
    # warp's slice resolves to one transaction per memory op.
    assert simt.warps == 4
    assert simt.cache_transactions == pytest.approx(
        simt.warps * 8 * simt.phases, rel=1e-6
    )
    assert simt.coalescing_factor > 1.0


def test_mte2_staged_simt_loads_bypass_datacache():
    direct = evaluate(_ker05_flow(direct_gm_loads=True), "ascend950pr")
    staged = evaluate(_ker05_flow(direct_gm_loads=False), "ascend950pr")
    assert staged.simt.direct_gm_loads is False
    assert staged.simt.cache_fill_bytes == 0.0
    assert staged.simt.cache_transactions == 0.0
    assert staged.simt.mte2_staged_bytes == pytest.approx(
        512 * (1024 + 64) * staged.simt.phases
    )
    mte2_direct = next(s for s in direct.schedule.stages if s.name == "MTE2")
    mte2_staged = next(s for s in staged.schedule.stages if s.name == "MTE2")
    assert mte2_staged.total_bytes > mte2_direct.total_bytes
    assert staged.traffic["simt_mte2_staged_bytes"] == pytest.approx(
        staged.simt.mte2_staged_bytes
    )


@pytest.mark.parametrize("tile_rows, steps", [(512, 1), (256, 2)])
def test_hybrid_short_tile_runs_keep_fill_drain_and_switch_bubbles(tile_rows, steps):
    """Small tile counts must not assume 100% steady-state overlap (2.2)."""
    arch = get_architecture("ascend950pr")
    report = evaluate(
        AlgorithmFlow.vector_gemv(
            rows=512,
            columns=2048,
            tile_rows=tile_rows,
            dtype="fp16",
            simt=_ker05_simt_workload(),
        ),
        "ascend950pr",
    )
    schedule = report.schedule
    assert schedule.steps == steps
    # Prologue: the first tile pays the MTE2 DMA startup latency unhidden.
    assert schedule.fill_cycles >= arch.pipe("MTE2").base_latency_cycles
    # Epilogue: the last tile drains through MTE3 after the final switch.
    assert schedule.drain_cycles >= arch.pipe("MTE3").base_latency_cycles
    assert schedule.total_cycles == pytest.approx(
        steps * schedule.steady_state_cycles_per_step
        + schedule.fill_cycles
        + schedule.drain_cycles
    )
    # The VF switches ride the VECTOR stage, so a 1-2 tile run carries them
    # at full weight inside the makespan.
    assert report.simt.vf_mode_switch_cycles == pytest.approx(
        steps * 512 * 64.0
    )
    # The fixed prologue/epilogue bubbles amortize: at 8 tiles their share
    # per tile is an eighth of the 1-tile run's.
    wide = evaluate(
        AlgorithmFlow.vector_gemv(
            rows=2048,
            columns=2048,
            tile_rows=256,
            dtype="fp16",
            simt=_ker05_simt_workload(),
        ),
        "ascend950pr",
    )
    short_bubble = (schedule.fill_cycles + schedule.drain_cycles) / steps
    wide_bubble = (
        wide.schedule.fill_cycles + wide.schedule.drain_cycles
    ) / wide.schedule.steps
    assert wide_bubble < short_bubble


# ---------------------------------------------------------------------------
# Stage-1 fleet acceptance probes (KER-01 / KER-05, hybrid_simt_simd)
# ---------------------------------------------------------------------------

_FLOW_DIR = Path(__file__).parent / "flows"
_UB_BUDGET_KER05 = 34944  # kernel-side static UB reservation: 34.1 KiB


def _probe_payload(capsys, path, *extra):
    argv = ["probe", "--flow-file", str(path), "--arch", "ascend950pr", *extra]
    assert main(argv) == 0
    return json.loads(capsys.readouterr().out)


def test_ker01_and_ker05_flow_files_probe_hybrid(capsys):
    for name, dispatches in (
        ("ker01_mhc_pre_norm_sinkhorn_3", 4),
        ("ker05_mxfp4_expert_gemv_3", 3),
    ):
        path = _FLOW_DIR / f"{name}.json"
        flow = json.loads(path.read_text(encoding="utf-8"))
        assert flow["execution_graph"]["mode"] == "hybrid_simt_simd"
        assert flow["simt"]["direct_gm_loads"] is True

        payload = _probe_payload(capsys, path)
        assert payload["is_valid"] is True
        simt = payload["simt"]
        assert simt["vf_mode_switch_cycles"] == pytest.approx(
            simt["phases"] * dispatches * 64.0
        )
        assert payload["traffic_bytes"]["simt_vf_switch_cycles"] == pytest.approx(
            simt["vf_mode_switch_cycles"]
        )
        # The mode transition cost is priced into the makespan itself.
        vector = payload["pipeline"]["stages"]["VECTOR"]
        assert vector["cycles_per_step"] > 0


def test_ker05_probe_text_shows_switch_and_reduced_ub(capsys):
    assert (
        main(
            [
                "probe",
                "--flow-file",
                str(_FLOW_DIR / "ker05_mxfp4_expert_gemv_3.json"),
                "--arch",
                "ascend950pr",
                "--format",
                "text",
            ]
        )
        == 0
    )
    text = capsys.readouterr().out
    assert "VF mode switches 32,832 cyc" in text
    assert "ok  UB" in text


def test_ker05_reduced_ub_footprint_and_whole_row_infeasibility(capsys):
    """The DataCache-resident weight path is what keeps KER-05's UB small.

    The flow's 3-row chunking stages 32832 B - inside the kernel's 34944 B
    (34.1 KiB) static budget - while the whole-row tile the hybrid design
    replaced would overflow physical UB outright.
    """
    payload = _probe_payload(capsys, _FLOW_DIR / "ker05_mxfp4_expert_gemv_3.json")
    ub = payload["sram_footprints"]["UB"]
    assert ub["bytes_used"] == 32832
    assert ub["bytes_used"] <= _UB_BUDGET_KER05

    from ascend_pipesim.dsl import SimtWorkload

    whole_row = AlgorithmFlow.vector_gemv(
        rows=512,
        columns=2048,
        tile_rows=512,
        dtype="fp16",
        out_dtype="fp16",
        simt=SimtWorkload(
            threads=128,
            instructions_per_thread=40,
            memory_ops_per_thread=8,
            bytes_per_access=0.5,
            stride_bytes=0.5,
            working_set_bytes=512 * (1024 + 64),
            vf_dispatches_per_phase=512,
            shuffle_ops_per_thread=5,
        ),
    )
    report = evaluate(whole_row, "ascend950pr")
    assert not report.is_valid
    assert any(v.code == "UB_OVERFLOW" for v in report.violations)
