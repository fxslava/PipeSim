"""Phase 2 acceptance tests: CLI (probe / autotile / codegen / rpc) and the
Ascend C kernel synthesiser.

The golden configuration used throughout: fp16 GEMM on ascend910b with outer
tile (128, 128, 64) and subtile (64, 64, 32), which pins every synthesised
byte length:

    A1 = pad(128,16) * pad(64,16) * 2 = 16384   (zZ A panel)
    B1 = pad(64,16) * pad(128,16) * 2 = 16384   (nZ B panel)
    A2 = 64 * 32 * 2               =  4096      (L0A subtile)
    B2 = 32 * 64 * 2               =  4096      (L0B subtile)
    CO1 = 128 * 128 * 4            = 65536      (fp32 accumulator)
"""

from __future__ import annotations

import json
import re
from io import StringIO
from pathlib import Path

import pytest

from ascend_pipesim import cli
from ascend_pipesim.codegen import KernelBundle, synthesize, synthesize_kernel
from ascend_pipesim.dsl import AlgorithmFlow, BufferPlan, LoopOrder
from ascend_pipesim.evaluator import evaluate

GEMM = "2048,2048,4096"
TILE = "128,128,64"
SUBTILE = "64,64,32"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def run_cli(argv, capsys):
    code = cli.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out


def run_probe(extra, capsys):
    argv = [
        "probe", "--arch", "ascend910b",
        "--gemm", GEMM, "--tile", TILE, "--subtile", SUBTILE,
        "--buffers", "double", "--dtype", "fp16", "--compact",
        *extra,
    ]
    code, out = run_cli(argv, capsys)
    assert code == 0
    return json.loads(out)


def make_flow(m=2048, n=2048, k=4096, dtype="fp16", buffers=None, **kwargs):
    return AlgorithmFlow.gemm(
        m=m, n=n, k=k,
        tile=(128, 128, 64), subtile=(64, 64, 32),
        dtype=dtype, buffers=buffers or BufferPlan.double(), **kwargs,
    )


def make_bundle(flow):
    report = evaluate(flow, "ascend910b")
    return synthesize_kernel(report, flow)


def header_const(header: str, name: str) -> int:
    match = re.search(rf"{name}\s*=\s*(-?\d+);", header)
    assert match, f"constant {name} missing from kernel_tiling.h"
    return int(match.group(1))


# ---------------------------------------------------------------------------
# CLI probe
# ---------------------------------------------------------------------------


class TestProbeCommand:
    def test_probe_json_schema(self, capsys):
        payload = run_probe([], capsys)
        for key in (
            "is_valid", "cube_utilization_pct", "limiting_stage",
            "hierarchical_roofline", "sram_footprints", "violations",
            "code_generation_hints",
        ):
            assert key in payload, key
        assert payload["is_valid"] is True
        assert 0.0 < payload["cube_utilization_pct"] <= 100.0
        assert isinstance(payload["limiting_stage"], str)
        assert payload["limiting_stage"]

        roofline = payload["hierarchical_roofline"]
        assert roofline["binding_level"]
        for level in ("GM_MTE2", "L1_MTE1", "L0C_FIXPIPE", "DRAM"):
            assert level in roofline["levels"]

        footprints = payload["sram_footprints"]
        for pool in ("L1", "L0A", "L0B", "L0C", "UB"):
            assert pool in footprints
            assert footprints[pool]["fits"] is True

        assert isinstance(payload["violations"], list)
        assert all(v["severity"] != "error" for v in payload["violations"])

        hints = payload["code_generation_hints"]
        assert hints["requires_set_tail"] is False  # aligned shape
        assert len(hints["queue_declarations"]) == 5
        assert hints["loop_order"] == "M_first"

    def test_probe_stationary_selects_loop_order(self, capsys):
        payload = run_probe(["--stationary", "n"], capsys)
        assert payload["code_generation_hints"]["loop_order"] == "N_first"
        payload = run_probe(["--stationary", "m"], capsys)
        assert payload["code_generation_hints"]["loop_order"] == "M_first"

    @pytest.mark.parametrize("dtype,k0", [("fp16", 16), ("bf16", 16), ("int8", 32)])
    def test_probe_dtypes(self, capsys, dtype, k0):
        payload = run_probe([f"--dtype", dtype], capsys)
        assert payload["is_valid"] is True
        assert payload["code_generation_hints"]["fractal"]["k0"] == k0

    def test_probe_infeasible_tile_reports_violations(self, capsys):
        code, out = run_cli(
            [
                "probe", "--arch", "ascend910b", "--gemm", GEMM,
                "--tile", "512,512,64", "--subtile", "64,64,32",
                "--buffers", "double", "--dtype", "fp16", "--compact",
            ],
            capsys,
        )
        assert code == 0  # analysis succeeded even though the tiling cannot
        payload = json.loads(out)
        assert payload["is_valid"] is False
        assert any(
            v["code"] == "L0C_OVERFLOW" for v in payload["violations"]
        )
        assert payload["sram_footprints"]["L0C"]["fits"] is False

        code, _ = run_cli(
            [
                "probe", "--arch", "ascend910b", "--gemm", GEMM,
                "--tile", "512,512,64", "--subtile", "64,64,32",
                "--dtype", "fp16", "--strict", "--compact",
            ],
            capsys,
        )
        assert code == 1  # --strict turns infeasibility into exit 1

    def test_probe_pretty_flag(self, capsys):
        _, out = run_cli(
            [
                "probe", "--gemm", "256,256,256", "--compact",
            ],
            capsys,
        )
        assert "\n" not in out.strip()  # compact: one line
        _, out = run_cli(
            [
                "probe", "--gemm", "256,256,256", "--compact", "--pretty",
            ],
            capsys,
        )
        assert out.startswith("{\n")  # --pretty overrides --compact
        assert len(out.splitlines()) > 10

    def test_exit_code_2_on_parse_error(self):
        with pytest.raises(SystemExit) as exc:
            cli.main(["probe", "--gemm", "1,2"])  # not a triple
        assert exc.value.code == 2

    def test_exit_code_2_on_unknown_arch(self):
        with pytest.raises(SystemExit) as exc:
            cli.main(["probe", "--gemm", "128,128,64", "--arch", "ascend9999"])
        assert exc.value.code == 2

    def test_exit_code_2_on_bad_dtype(self):
        with pytest.raises(SystemExit) as exc:
            cli.main(["probe", "--gemm", "128,128,64", "--dtype", "fp7"])
        assert exc.value.code == 2


# ---------------------------------------------------------------------------
# CLI autotile
# ---------------------------------------------------------------------------


class TestAutotileCommand:
    def test_autotile_topk_without_duplicate_outer_tiles(self, capsys):
        code, out = run_cli(
            [
                "autotile", "--arch", "ascend910b",
                "--gemm", "512,512,512", "--dtype", "fp16",
                "--buffers", "double", "--topk", "3",
                "--max-tile", "128", "--compact",
            ],
            capsys,
        )
        assert code == 0
        data = json.loads(out)

        candidates = data["candidates"]
        assert len(candidates) == 3
        assert [c["rank"] for c in candidates] == [1, 2, 3]
        scores = [c["score"] for c in candidates]
        assert scores == sorted(scores, reverse=True)

        outer_tiles = [
            (c["tile"]["tile_m"], c["tile"]["tile_n"], c["tile"]["tile_k"])
            for c in candidates
        ]
        assert len(set(outer_tiles)) == len(outer_tiles)  # deduplicated

        for cand in candidates:
            for key in (
                "tile", "buffers", "loop_order", "cube_utilization_pct",
                "limiting_stage", "fractal_efficiency", "requires_set_tail",
                "peak_sram_occupancy_pct", "total_us", "sram_footprints",
                "score", "is_valid",
            ):
                assert key in cand, key
            assert cand["is_valid"] is True
            assert 0.0 <= cand["cube_utilization_pct"] <= 100.0
            assert cand["limiting_stage"]
            assert cand["tile"]["sub_m"] <= cand["tile"]["tile_m"]
            assert cand["tile"]["sub_n"] <= cand["tile"]["tile_n"]
            assert cand["tile"]["sub_k"] <= cand["tile"]["tile_k"]

        assert data["best"]["rank"] == 1
        assert data["search"]["evaluated"] > 0
        assert data["search"]["grid_size"] > 0

    def test_autotile_stationary_restricts_loop_order(self, capsys):
        _, out = run_cli(
            [
                "autotile", "--arch", "ascend910b",
                "--gemm", "512,512,512", "--dtype", "fp16",
                "--topk", "2", "--max-tile", "128",
                "--stationary", "n", "--compact",
            ],
            capsys,
        )
        data = json.loads(out)
        assert data["candidates"]
        assert all(
            c["loop_order"] == "N_first" for c in data["candidates"]
        )


# ---------------------------------------------------------------------------
# Codegen: API level
# ---------------------------------------------------------------------------


class TestSynthesizeKernel:
    def test_tiling_header_constants(self):
        bundle = make_bundle(make_flow())
        header = bundle.kernel_tiling_h
        assert header_const(header, "M_EXTENT") == 2048
        assert header_const(header, "N_EXTENT") == 2048
        assert header_const(header, "K_EXTENT") == 4096
        assert header_const(header, "TILE_M") == 128
        assert header_const(header, "TILE_N") == 128
        assert header_const(header, "TILE_K") == 64
        assert header_const(header, "SUB_M") == 64
        assert header_const(header, "SUB_K") == 32
        assert header_const(header, "USED_M") == 16
        assert header_const(header, "USED_N") == 16
        assert header_const(header, "USED_K") == 64
        assert header_const(header, "SUB_STEPS_M") == 2
        assert header_const(header, "SUB_STEPS_K") == 2
        # exact byte lengths for the golden fp16 configuration
        assert header_const(header, "A1_BUFFER_BYTES") == 16384
        assert header_const(header, "B1_BUFFER_BYTES") == 16384
        assert header_const(header, "A2_BUFFER_BYTES") == 4096
        assert header_const(header, "B2_BUFFER_BYTES") == 4096
        assert header_const(header, "CO1_BUFFER_BYTES") == 65536
        assert header_const(header, "FRACTAL_K0") == 16
        assert "SETTAIL_REQUIRED = false;" in header
        assert header.count("static_assert(") == 4

    def test_kernel_tque_layout_and_pipeline(self):
        bundle = make_bundle(make_flow())
        cpp = bundle.kernel_cpp

        for position, depth in (
            ("A1", 2), ("B1", 2), ("A2", 2), ("B2", 2), ("CO1", 1),
        ):
            assert re.search(
                rf"TQue<TPosition::{position},\s*{depth}> que{position}_;",
                cpp,
            ), f"missing TQue<TPosition::{position}, {depth}>"

        assert cpp.count("pipe_.InitBuffer(") == 5
        assert "pipe_.InitBuffer(queA1_, A1_BUFFER_BYTES);" in cpp
        assert "pipe_.InitBuffer(queCO1_, CO1_BUFFER_BYTES);" in cpp
        assert "using A_T = half;" in cpp
        assert "using C_T = half;" in cpp

        # static staging order: MTE2 -> MTE1 -> Cube Iterate -> Fixpipe
        markers = [
            "CopyIn(mo, no, ko);   // MTE2",
            "LoadData(a2",          # MTE1: L1 -> L0A
            "LoadData(b2",          # MTE1: L1 -> L0B
            "matmulObj.Iterate()",  # Cube
            "matmulObj.GetTensorC()",  # Fixpipe drain
            "WaitFlag<HardEvent::M_FIX>(EVENT_ID0)",
        ]
        positions = [cpp.index(marker) for marker in markers]
        assert positions == sorted(positions)

        # the L0C drain handshake uses a literal, statically resolvable id
        assert "SetFlag<HardEvent::M_FIX>(EVENT_ID0);" in cpp

        # every outer loop iterates against its own trip count
        for name, bound in (("mo", "usedM"), ("no", "usedN"), ("ko", "usedK")):
            assert re.search(
                rf"for \(int32_t {name} = 0; {name} < tiling_\.{bound};", cpp
            )

        # structurally sane C++: balanced braces, entry point present
        assert cpp.count("{") == cpp.count("}")
        assert 'extern "C" __global__ __aicore__ void pipesim_gemm(' in cpp

    def test_aligned_shape_has_no_settail(self):
        bundle = make_bundle(make_flow())
        assert "SetTail" not in bundle.kernel_cpp
        assert bundle.meta["settail_required"] is False

    def test_ragged_shape_injects_settail(self):
        flow = make_flow(m=1000, n=512, k=1024)  # 1000 % 128 == 104
        report = evaluate(flow, "ascend910b")
        assert report.code_generation_hints["requires_set_tail"] is True

        bundle = synthesize_kernel(report, flow)
        cpp = bundle.kernel_cpp
        assert "matmulObj.SetTail(tailM, tailN);" in cpp
        assert "tiling_.tailM" in cpp and "tiling_.tailN" in cpp
        assert "isTailRow" in cpp and "isTailCol" in cpp
        header = bundle.kernel_tiling_h
        assert header_const(header, "TAIL_M") == 104
        assert header_const(header, "TAIL_N") == 0
        assert header_const(header, "USED_M") == 8  # ceil(1000 / 128)
        assert "SETTAIL_REQUIRED = true;" in header
        assert bundle.meta["settail_required"] is True
        assert bundle.meta["tail_sizes"] == {"m": 104, "n": 0, "k": 0}

    def test_single_buffers_use_depth_one(self):
        bundle = make_bundle(
            make_flow(buffers=BufferPlan.single())
        )
        cpp = bundle.kernel_cpp
        for position in ("A1", "B1", "A2", "B2", "CO1"):
            assert re.search(
                rf"TQue<TPosition::{position},\s*1> que{position}_;", cpp
            )
        assert bundle.buffers["queA1_"].depth == 1
        assert bundle.buffers["queCO1_"].total_bytes == 65536

    def test_int8_dtype_widens_fractal(self):
        bundle = make_bundle(make_flow(dtype="int8"))
        cpp, header = bundle.kernel_cpp, bundle.kernel_tiling_h
        assert "using A_T = int8_t;" in cpp
        assert "using B_T = int8_t;" in cpp
        assert header_const(header, "FRACTAL_K0") == 32
        # k0=32: pad(64,32)=64, itemsize=1
        assert header_const(header, "A1_BUFFER_BYTES") == 128 * 64
        assert header_const(header, "B1_BUFFER_BYTES") == 64 * 128
        # accumulator is always 4 B/element
        assert header_const(header, "CO1_BUFFER_BYTES") == 65536

    def test_bytes_match_probe_footprints(self, capsys):
        payload = run_probe([], capsys)
        bundle = make_bundle(make_flow())
        header = bundle.kernel_tiling_h
        breakdown = payload["sram_footprints"]
        assert (
            breakdown["L1"]["breakdown"]["A_tile"]
            == header_const(header, "A1_BUFFER_BYTES")
        )
        assert (
            breakdown["L1"]["breakdown"]["B_tile"]
            == header_const(header, "B1_BUFFER_BYTES")
        )
        assert (
            breakdown["L0A"]["breakdown"]["A_subtile"]
            == header_const(header, "A2_BUFFER_BYTES")
        )
        assert (
            breakdown["L0B"]["breakdown"]["B_subtile"]
            == header_const(header, "B2_BUFFER_BYTES")
        )
        assert (
            breakdown["L0C"]["breakdown"]["accumulator_fp32"]
            == header_const(header, "CO1_BUFFER_BYTES")
        )

    def test_bundle_write(self, tmp_path):
        bundle = make_bundle(make_flow())
        paths = bundle.write(tmp_path)
        assert [p.name for p in paths] == ["kernel.cpp", "kernel_tiling.h"]
        for path in paths:
            assert path.exists() and path.stat().st_size > 0
        assert (tmp_path / "kernel.cpp").read_text(encoding="utf-8") == \
            bundle.kernel_cpp
        assert (tmp_path / "kernel_tiling.h").read_text(encoding="utf-8") == \
            bundle.kernel_tiling_h

    def test_synthesize_convenience(self):
        bundle = synthesize(make_flow())
        assert isinstance(bundle, KernelBundle)
        assert bundle.meta["arch"] == "ascend910b"
        assert bundle.summary().startswith("pipesim_gemm:")

    def test_transposed_operands(self):
        flow = make_flow(transpose_a=True, transpose_b=True)
        bundle = make_bundle(flow)
        cpp = bundle.kernel_cpp
        assert "constexpr bool TRANSPOSE_A = true;" in cpp
        assert "constexpr bool TRANSPOSE_B = true;" in cpp
        assert "GetNormConfig(TRANSPOSE_A, TRANSPOSE_B, false, false);" in cpp

    def test_k_outer_rematerialises_per_k_step(self):
        bundle = make_bundle(make_flow(k_position="outer"))
        cpp = bundle.kernel_cpp
        assert bundle.meta["outer_loop_nest"] == ["ko", "mo", "no"]
        # ko is outermost; the Fixpipe drain closes the innermost spatial
        # loop, re-materialising C once per K step.
        assert cpp.index("for (int32_t ko") < cpp.index("for (int32_t mo")
        assert cpp.index("for (int32_t mo") < cpp.index("for (int32_t no")
        assert cpp.index("for (int32_t no") < cpp.index("CopyOut(mo, no);")


# ---------------------------------------------------------------------------
# Codegen: CLI level
# ---------------------------------------------------------------------------


class TestCodegenCommand:
    def test_codegen_json_to_stdout(self, capsys):
        code, out = run_cli(
            [
                "codegen", "--arch", "ascend910b",
                "--gemm", "1000,512,1024",
                "--tile", TILE, "--subtile", SUBTILE,
                "--buffers", "double", "--dtype", "fp16",
                "--compact",
            ],
            capsys,
        )
        assert code == 0
        payload = json.loads(out)
        for key in ("kernel.cpp", "kernel_tiling.h", "buffers", "meta"):
            assert key in payload
        assert "matmulObj.SetTail(tailM, tailN);" in payload["kernel.cpp"]
        assert payload["meta"]["settail_required"] is True
        assert payload["meta"]["tail_sizes"]["m"] == 104
        assert payload["buffers"]["queA1_"]["per_buffer_bytes"] == 16384
        assert payload["buffers"]["queCO1_"]["depth"] == 1

    def test_codegen_writes_files(self, capsys, tmp_path):
        outdir = tmp_path / "generated"
        code, out = run_cli(
            [
                "codegen", "--arch", "ascend910b",
                "--gemm", "2048,2048,4096",
                "--tile", TILE, "--subtile", SUBTILE,
                "--outdir", str(outdir), "--compact",
            ],
            capsys,
        )
        assert code == 0
        payload = json.loads(out)
        assert set(payload["written"]) == {
            str(outdir / "kernel.cpp"), str(outdir / "kernel_tiling.h"),
        }
        for name in ("kernel.cpp", "kernel_tiling.h"):
            path = outdir / name
            assert path.exists()
            assert path.stat().st_size > 0
        assert "SetTail" not in (outdir / "kernel.cpp").read_text(
            encoding="utf-8"
        )

    def test_codegen_respects_stationary(self, capsys):
        _, out = run_cli(
            [
                "codegen", "--gemm", "512,512,512",
                "--tile", TILE, "--subtile", SUBTILE,
                "--stationary", "n", "--compact",
            ],
            capsys,
        )
        payload = json.loads(out)
        assert payload["meta"]["loop_order"] == "N_first"
        # N_first walks the spatial loops in the mirrored order
        assert payload["meta"]["outer_loop_nest"][0] == "no"


# ---------------------------------------------------------------------------
# JSON RPC
# ---------------------------------------------------------------------------


class TestJsonRpc:
    def test_rpc_probe_and_codegen(self):
        requests = "\n".join(
            [
                json.dumps({"id": 1, "method": "probe",
                            "params": {"gemm": [2048, 2048, 4096],
                                       "tile": [128, 128, 64],
                                       "subtile": [64, 64, 32]}}),
                json.dumps({"id": 2, "method": "codegen",
                            "params": {"gemm": "1000,512,1024",
                                       "tile": TILE, "subtile": SUBTILE}}),
                json.dumps({"id": 3, "method": "no_such_method",
                            "params": {}}),
            ]
        )
        stdout = StringIO()
        cli.serve_rpc(stdin=StringIO(requests), stdout=stdout)
        replies = [json.loads(line) for line in stdout.getvalue().splitlines()]

        assert len(replies) == 3
        assert replies[0]["ok"] is True
        assert replies[0]["id"] == 1
        assert replies[0]["result"]["is_valid"] is True
        assert replies[0]["result"]["code_generation_hints"][
            "requires_set_tail"
        ] is False

        assert replies[1]["ok"] is True
        assert "matmulObj.SetTail(tailM, tailN);" in replies[1]["result"][
            "kernel.cpp"
        ]
        assert replies[1]["result"]["meta"]["tail_sizes"]["m"] == 104

        assert replies[2]["ok"] is False
        assert "unknown method" in replies[2]["error"]["message"]
