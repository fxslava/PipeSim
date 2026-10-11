# Ascend 950 architecture reference

Source: `昇腾950 NPU架构白皮书.pdf` (Huawei, 2026), especially sections 3, 4.1–4.7 and tables 3-1 and 4-2. Capacities below use the whitepaper's KB/MB/GB labels; byte values in PipeSim use binary KiB/MiB. The whitepaper gives aggregate peak performance and relative precision ratios, but does **not** publish per-cycle MTE throughput, core clock, or a CANN spelling for BufferID intrinsics. Those values in the simulator are identified as modeling assumptions.

## Package, bins, and system hierarchy

The package contains **two AI Dies, two IO Dies**, and high-speed memory modules: eight for 950PR or four for 950DT. D2D Clink and memory interfaces form one **unified memory address (UMA)** space. A full configuration has **36 AI subsystems**, each with **one Cube and two Vector cores** (36 Cube / 72 Vector). The shipped bins in table 3-1 are:

| Product | Cube / Vector cores | Memory capacity | Peak memory bandwidth | L2 |
| --- | ---: | ---: | ---: | ---: |
| 950PR default | 32 / 64 | 128 GB | 1.6 TB/s | 128 MB |
| 950PR reduced | 28 / 56 | 112 GB | 1.4 TB/s | 112 MB |
| 950DT default | 36 / 72 | 144 GB | 4.0 TB/s | 128 MB |
| 950DT other bins | 32 / 64 or 28 / 56 | 96 GB option | 4.0 TB/s listed for family | 128 MB |

The 950DT table does not map every memory option uniquely to every core-count bin. Do not assume that 28/56 necessarily means 96 GB. The simulator defaults to the first row for each product and supports overrides for other bins.

Four AI CPU clusters each contain **two** Linx816 ARMv8-A cores with two hardware threads per core and **4 MB L3** per cluster. The full package is therefore **8 CPU cores / 16 threads**, not eight cores per cluster. Each CPU core has 64 KB L1 and 1 MB L2. Four DVPP subsystems provide up to **4 VPC, 4 JPEGE, and 8 JPEGD** cores; some product bins reduce VPC/JPEGE to two. STARS 2.0 schedules AIC, AIV, CPU, DVPP, SDMA, Unified Bus, and CCU tasks over the dedicated **HSCB** control bus. The whitepaper lists 2048 resident device task streams, up to 32 concurrent CCU tasks, and up to eight resource groups (section 4.4).

## Memory hierarchy and data movement

| Resource | Capacity and organization | Programming consequence |
| --- | --- | --- |
| Cube L1 buffer | **512 KiB per AI subsystem/core** | Stage reusable A/B panels before MTE1 loads. |
| L0A / L0B | **64 KiB each** | Cube operand fractals, conventionally Zz for A and Nz for B. |
| L0C | **256 KiB**, FP32/INT32 accumulator | Each output element consumes 4 B; account for queue depth. This doubles the 128 KiB older-generation budget. |
| Unified Buffer | **512 KiB per AI subsystem** | Shared aggregate of the two Vector cores: **256 KiB physical per AIV**. For the 950PR arch35 preset, the usable tensor ceiling is **216 KiB per AIV** after 8 KiB compiler reserve and 32 KiB minimum SIMT DataCache. Eight-bank, 32 B interleave is the kernel-model bank rule. |
| L2 cache | **Up to 128 MB** globally addressable UMA cache | Distributed banks, die-local affinity, hardware coherence across AI Dies, **512 B cache lines** with **four 128 B sectors**. A reduced 950PR bin has 112 MB. |
| Package memory | 950PR up to **128 GB / 1.6 TB/s**; 950DT up to **144 GB / 4.0 TB/s** | Package-attached high-speed memory, not an ordinary external DIMM. Supports online ECC, patrol, spare rows, and dynamic row isolation. |

L2 software controls include **allocate/non-allocate L2 hints** for reuse and eviction behavior and SDMA **CMO** operations: Prefetch, Writeback, Invalidate (called “Invalid” in the paper), and Flush. L2 traffic can cross AI Dies within the UMA address space (section 4.3.2). Local SRAM is explicitly staged; L2 is a cache, not a substitute for local allocation checks.

## DaVinci v3 compute

The Cube supports **TF32, FP16, BF16, FP8 E5M2/E4M3, MXFP8, HiF8, MXFP4, and INT8**. At the same frequency, FP8/MXFP8/HiF8 tensor peak is **2× FP16**, and MXFP4 is **4× FP16** (section 4.1.1). HiF8 uses tapered precision, spans exponent positions **−22 through +15** (38 powers of two), and does not require the separate 8-bit MX scale of MXFP8 (section 4.1.1). Cube result conversion can run inline from FP32/INT32 L0C to BF16/FP16/FP8/INT8 in UB, with NZ-to-ND/DN layout conversion. For mixed-precision kernels, budget scale metadata and quantization intermediates in addition to packed operand bytes.

The Vector core adds a register file between UB and ALU, dual-issue SIMD with out-of-order execution, plus SIMT for irregular control flow. A **Vector Function (VF)** can use SIMD or SIMT and switch between them. The paper reports **100% higher per-core FP16 and FP32 Vector peak** than the prior generation, as well as native BF16 and conversion instructions (sections 4.1.2–4.1.3). Model the 1:2 Cube-to-Vector ratio when partitioning work: each Cube subsystem feeds two Vector cores.

`AlgorithmFlow.vector_gemv(..., simt=SimtWorkload(...))` models a fused SIMT
packing phase and SIMD GEMV on an AIV. The report exposes warp count, branch
divergence factor, cache transaction and miss estimates, coalescing factor,
DataCache fill traffic, and shared Vector ALU utilization. The 950PR preset
uses the documented 32-thread warp, four warp schedulers and 128-byte GM
transaction granularity. Cache hit and fill latencies are configurable
analytical estimates in `SimtHardware`; they are not measured instruction
latencies. MTE2 and SIMT cache fills share bandwidth in the schedule, while
SIMT and SIMD instructions contribute to one Vector stage. The minimum 200
cycle MTE2 startup and pipeline fill/drain terms remain visible for one- and
two-tile sweeps.
The configurable shared UB-port throughput imposes a further lower bound on
simultaneous DMA/cache fills and SIMD reads; the report includes
`ub_port_cycles_per_step` so this contention can be inspected directly.
`SimtWorkload.registers_per_thread` is checked against the 128 KiB SIMT
register file and the documented launch-size limits; estimated spills add
Vector issue work and UB-port traffic.

Hybrid VF dispatch accounting: every `asc_vf_call` boundary reconfigures the
physically shared Vector ALU between SIMD and SIMT warp issue.
`SimtHardware.vf_mode_switch_penalty_cycles` (default 64, validated inside
the measured 30..80-cycle window) prices each boundary, and
`SimtWorkload.vf_dispatches_per_phase` counts them per outer tile, so a
per-row dispatch GEMV carries `rows x 64` cycles inside the VECTOR stage.
`SimtWorkload.shuffle_ops_per_thread` models warp butterfly shuffles
(`asc_shfl_xor`) as register-resident reductions - shared-ALU issue cycles
with zero SRAM traffic - and `direct_gm_loads=False` swaps the DataCache
path for a classical MTE2 DMA staging burst (`simt_mte2_staged_bytes`).
Fractional `bytes_per_access`/`stride_bytes` (0.5 B for packed MXFP4 pairs)
let the coalescing model share one cache line across adjacent lanes.
`AlgorithmFlow.vector_gemv(..., simt=...)` lowers to the
`hybrid_simt_simd` execution graph, which is the only mode charged for
switches.

The CLI accepts a serialized `AlgorithmFlow` for this path:
`npu-pipesim probe --arch ascend950pr --flow-file mhc_flow.json`.

| PipeSim analytical dtype | Element storage | Cube model | Fractal K0 |
| --- | ---: | ---: | ---: |
| FP16 / BF16 | 2 B | 4096 MAC/cycle | 16 |
| FP8 / MXFP8 / HiF8 | 1 B | 8192 MAC/cycle | 32 |
| MXFP4 | 0.5 B, packed | 16384 MAC/cycle | 64 |

These **per-cycle** rows preserve the whitepaper's relative ratios and the simulator's 4096-MAC FP16 convention. They are **not measured chip timing**. The 1.65 GHz simulator frequency is inferred approximately from the published 32-core FP16 Cube total (432 TFLOP/s). The paper's aggregate totals in table 3-1 should be used for product peak claims instead.

## Fusion, DMA, and synchronization

**CV fusion:** a direct pathway links **Cube L1** to **Vector UB**, with on-path precision and layout conversion, avoiding an L2 round trip. Use it to stream Cube outputs into online Softmax/FlashAttention or a fused epilogue while preserving vector-local reuse (section 4.1.4). The separate L0C-to-UB result conversion remains available.

**NDDMA:** an asynchronous engine performs up to **five-dimensional** GM-to-UB rearrangement or transpose. Hardware address generation and an internal cache coalesce element-size reads into **128 B reads**. It is well suited to NCHW/NHWC changes, ND-to-NZ-like staging, and regular irregular-stride gathers, provided the chosen layout is supported by the actual CANN release (section 4.1.5).

**BufferID:** the whitepaper describes `get_buf(pipe, #id)` as acquisition and `rel_buf(pipe, #id)` as release, with buffer ownership replacing explicit `set_flag`/`wait_flag` pair management (section 4.1.6). This is a **semantic notation from the paper**, not a verified Ascend C header signature. Bind it to the target CANN version, preserve one acquire/release per live buffer, and ensure neither Cube nor Vector overwrites data still consumed by the other. PipeSim codegen can emit a gated BufferID branch while retaining a legacy flag branch for builds without that API.

The 950 JSON profiles use **128 B/cycle** for MTE1/MTE2/MTE3/FIXPIPE, NDDMA, and CV bypass as an **analytical service-rate assumption requested for v3 modeling**. The whitepaper specifies 128 B NDDMA read coalescing, not 128 B/cycle throughput. Calibrate these channel rates and startup latencies against measured kernels before using absolute time predictions.

## Interconnect and collectives

**Unified Bus 2.0** supplies **72 HiLink SerDes lanes** arranged as **18 x4 ports**, each lane up to **112 Gb/s**. Table 3-1 lists **2016 GB/s bidirectional** aggregate port bandwidth, subject to optical module limits. Two ports can be reused for **2 × 400 Gb/s UBoE**; four can be reused for **PCIe 5.0 x16**, listed as 128 GB/s bidirectional. URMA offers asynchronous queue access; UB Memory offers synchronous Load/Store/Atomic addressing. IO Dies can forward traffic between their ports without consuming compute-die or memory bandwidth (section 4.6).

The **CCU** offloads programmed collective data movement, synchronization, and reductions, including Broadcast, ReduceScatter, AllGather, AllReduce, All2All, and All2Allv. Unified Bus supports **up to 8192 cards per supernode** with Clos, Full Mesh + Clos, or nD-Mesh arrangements; larger clusters can exceed 128K cards through network expansion (sections 4.6.4 and 4.7).
