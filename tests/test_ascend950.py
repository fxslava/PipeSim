"""Ascend 950 manifest, packed dtype, and v3 capacity regressions."""

import json
from pathlib import Path

import pytest

from ascend_pipesim.autotiler import _outer_grid
from ascend_pipesim.codegen import synthesize_kernel
from ascend_pipesim.dsl import AlgorithmFlow, BufferPlan
from ascend_pipesim.evaluator import evaluate
from ascend_pipesim.topology import DType, get_architecture


def test_default_950_bins_and_precision_ratios():
    pr = get_architecture("950pr")
    dt = get_architecture("ascend950")
    assert (pr.cube_cores, pr.vector_cores, pr.hbm_bandwidth_gbps) == (32, 64, 1600.0)
    assert (dt.cube_cores, dt.vector_cores, dt.hbm_bandwidth_gbps) == (36, 72, 4000.0)
    for arch in (pr, dt):
        assert arch.pool("L0C").capacity_bytes == 262144
        assert arch.pool("UB").capacity_bytes == 524288
        assert arch.l2_capacity_bytes == 134217728
        assert arch.cube.profile("mxfp4").macs_per_cycle == 4 * arch.cube.profile("fp16").macs_per_cycle
        assert arch.cube.profile("hif8").macs_per_cycle == 2 * arch.cube.profile("fp16").macs_per_cycle


def test_packed_dtype_and_aliases():
    assert DType.parse("MX-FP4") is DType.MXFP4
    assert DType.parse("hi_f8") is DType.HIF8
    assert DType.parse("mx8") is DType.MXFP8
    assert DType.MXFP4.size_bytes == 0.5
    assert DType.MXFP8.size_bytes == DType.HIF8.size_bytes == 1


def test_l0c_pruning_uses_selected_architecture():
    def candidates(arch):
        return _outer_grid([192], [192], [64], k0=16, itemsize=2,
                           l0c_capacity=arch.pool("L0C").capacity_bytes,
                           l1_capacity=arch.pool("L1").capacity_bytes,
                           l1_depth=1, l0c_depth=1)[0]
    assert candidates(get_architecture("ascend950dt")) == [(192, 192, 64)]
    assert candidates(get_architecture("ascend910b")) == []


def test_codegen_v3_budget_and_optional_bufferid():
    flow = AlgorithmFlow.gemm(m=192, n=192, k=64,
                              tile=(192, 192, 64), subtile=(64, 64, 32),
                              buffers=BufferPlan.single())
    report = evaluate(flow, "ascend950dt")
    assert report.is_valid
    bundle = synthesize_kernel(report, flow, bufferid_sync=True)
    assert 'static_assert(CO1_BUFFER_BYTES * 1 <= 262144' in bundle.kernel_tiling_h
    assert "get_buf(PIPE_M, 0);" in bundle.kernel_cpp
    assert "rel_buf(PIPE_M, 0);" in bundle.kernel_cpp
    assert bundle.meta["bufferid_sync"] is True
    ub_flow = AlgorithmFlow.gemm(m=192, n=192, k=64,
                                 tile=(192, 192, 64), subtile=(64, 64, 32),
                                 buffers=BufferPlan.single(),
                                 output_path="fixpipe_ub_mte3")
    ub_bundle = synthesize_kernel(evaluate(ub_flow, "ascend950dt"), ub_flow)
    assert 'static_assert(UB_EPILOGUE_BYTES <= 524288' in ub_bundle.kernel_tiling_h
    with pytest.raises(ValueError, match="DaVinci v3"):
        synthesize_kernel(report, flow, "ascend910b", bufferid_sync=True)


def test_json_profiles_are_standalone_v3():
    profiles = Path(__file__).resolve().parents[1] / "profiles"
    for name, cores, vectors, bandwidth in (
        ("ascend950pr", 32, 64, 1600), ("ascend950dt", 36, 72, 4000),
    ):
        p = json.loads((profiles / f"{name}.json").read_text(encoding="utf-8"))
        assert p["isa_generation"] == "davinci_v3"
        assert "base_profile" not in p and not p.get("commercial_label")
        assert p["system"]["cube_cores"] == cores
        assert p["system"]["vector_cores"] == vectors
        assert p["system"]["hbm_bandwidth_gbps"] == bandwidth
        assert p["memory_domains"]["L0C"]["size_bytes"] == 262144
        assert p["memory_domains"]["UB"]["size_bytes"] == 524288
        assert {"NDDMA", "CV_BYPASS"} <= p["transfer_channels"].keys()
