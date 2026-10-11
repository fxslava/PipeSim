# Unified SoC Architecture Meta-Language & Dynamic Manifest Registration

> **Note on Implementation Status:** This document is an architectural design proposal for cross-tool manifest unification with `ascend-kernel-analyzer`. Built-in PipeSim hardware topologies currently reside in `topology.py`.

**Status:** Proposal; the shared-manifest design described below is not the
source of PipeSim's currently active built-in topologies.
**Scope:** `D:\Projects\PipeSim` (analytical simulator / auto-tiler) and
`D:\Projects\ascend-kernel-analyzer` (static AST / deadlock / memory auditor).
**Goal:** adding a chip target (a commercial binning, a custom 910B SKU, or an
edge variant) becomes "drop one JSON file into a shared `profiles/` directory"
— zero Python/C++ source changes in either tool.

---

## 1. Comparative codebase audit

### 1.1 Where each tool defines hardware today

| Concern | PipeSim (`src/ascend_pipesim/topology.py`) | ascend_analyzer (`ascend_analyzer/hardware.py`, `checkers/`) |
|---|---|---|
| Chip registry | `_REGISTRY` (factory per chip) + `_ALIASES`, queried by `get_architecture()`; extensible at runtime via `register_architecture()` | `CHIP_PROFILES: Dict[str, ChipSpec]` + per-spec `aliases`, queried by `resolve_chip()`; **no runtime registration hook** |
| Memory pools | `MemoryPool(capacity_bytes, alignment_bytes, banks, bank_stride_bytes, ports)` — one alignment per pool, banks/stride as data | `DomainSpec(capacity_bytes, base_alignment, size_alignment, stride_alignment)` — three alignments, **no banks field** (`banks = 8` hardcoded in the AKA3006 checker, `checkers/memory.py:588`) |
| Extra domains | GM (64 GiB symbolic, 512 B burst align) | BT (1 KiB/64 B), FB (2 KiB/128 B); no GM domain object |
| Transfer engines | `PipeChannel` per engine incl. `MTE2_UB` and `MTE3`: bandwidth, **base latency**, min_burst, max_outstanding, efficiency | Flat `ChipSpec` ints: `mte2/mte1/fixpipe_bytes_per_cycle` only (no latencies, no MTE3, no burst model) |
| Compute | `ComputeUnit` with per-dtype `CubeDTypeProfile(k0, macs_per_cycle, accumulator)` (int8 k0=32 @8192, fp32 k0=8 @1024), fractal shape, pipeline + issue latencies; Vector `bytes_per_cycle` | Single `cube_macs_per_cycle=4096` (fp16-only view), hardcoded fractal 16 in `cube_contraction_cycles()`, `cube_read_bytes_per_cycle`, `vector_bytes=256` (register footprint), `sync_handoff_cycles=30` |
| System level | `frequency_ghz`, `cube_cores`, `vector_cores`, `hbm_bandwidth_gbps`, `l2_capacity_bytes`, `decoupled_vector_core` | none (single-core static view) |
| Synchronization | not modeled (implicit in pipeline schedule) | `reserved_event_ids={6,7}`, `max_event_id=7`, HardEvent route grammar (`SRC_DST` from the `Pipe` enum) |
| Instruction gating | none | `apis.py` API table with `requires_feature` (e.g. `mx_cube`), checked against `ChipSpec.features` |
| UB partitioning | none | 351x: `ub_total/compiler_reserved/min_datacache/max_usable_ub` bytes (AKA1010) |
| Overrides | `--arch-file` JSON patch (`architecture_from_dict`, supports `{"base": ...}`), `--pipe-bw/--pipe-latency/--pool-bytes/--frequency` | `--chip-profile` JSON, `--ub-bytes/--l1-bytes` |

### 1.2 Shared parameters (same concept, same value on 910B)

Pool capacities L1 512 KiB, L0A/L0B 64 KiB, L0C 128 KiB; UB 8-bank / 32 B
interleave; access block 32 B; MTE1/MTE2/Fixpipe 64 B/cycle; Cube 4096
MACs/cycle with 16×16×16 fractal; L1 base alignment 32 B; UB base alignment 32 B;
alias resolution by normalized name. Both tools agree on 910B in every shared
quantity they implement — consolidation is viable with **no semantic
re-negotiation** for the reference chip. (UB capacity is the one deliberate
exception: see D1.)

### 1.3 Unique to PipeSim (performance-simulation layer)

Discrete per-channel base latencies (MTE2 200 / MTE1 20 / Fixpipe 80 / MTE3 150
cycles), min-burst and outstanding-request models, per-dtype Cube profiles
(k0 widening, INT8×2, FP32÷4, accumulator dtype), core frequency / core counts /
HBM GB/s / L2 reuse, `decoupled_vector_core`, pool `ports`, efficiency derates,
fractal m0/n0 as data.

### 1.4 Unique to ascend_analyzer (correctness-audit layer)

Three-way per-domain alignment (base/size/stride) with fractal-derived L0A/B 512 B
and L0C 1024 B rules; BT/FB domains; reserved/max EVENT_IDs; `sync_handoff_cycles`;
`vector_bytes` (register footprint) vs PipeSim's `bytes_per_cycle` (UB bandwidth —
same number 256, different meaning, both kept); instruction whitelist with
per-feature gating; UB partitioning quadruple; provisional-capacity flags.

### 1.5 Drift register (found or threatened)

| # | Item | PipeSim | Analyzer | Resolution in the unified profile |
|---|---|---|---|---|
| D1 | **910B UB capacity vs. spec layer** | 192 KiB | 192 KiB | **RESOLVED 2026-10-03 (ground truth applied):** the manifest records the physical DaVinci v2 capacity **256 KiB** per the standing frozen-constants directive; the 192 KiB in both tools' built-in code is a legacy conservative bound. Phase B migrates both built-ins to the manifest value; until then the golden-equivalence test must expect exactly one intended delta (UB capacity/occupancy) and nothing else. |
| D2 | **910C L0C capacity** | 128 KiB | **256 KiB** | **RESOLVED 2026-10-03 (ground truth applied):** manifest records the physical 256 KiB FP32 accumulator (`element_bytes: 4`). PipeSim's 128 KiB built-in is the superseded bound and migrates in Phase B. |
| D3 | **351x L1 / MTE widths / Cube rate** | L1 1 MiB, 128 B/cyc, 8192 MACs | L1 512 KiB, 64 B/cyc, 4096 MACs | Both self-declared speculative. The manifest records mixed provenance *explicitly* (v3 engine widths and 8192-MAC Cube from PipeSim; enforcement capacities, UB partitioning and `mx_cube` from the analyzer; L1 divergence stated in `notes`). `provisional: true`; disagreement is now visible data instead of hidden code. |
| D4 | **Pool alignment granularity** | 32 B everywhere | base 512/512/1024 B on L0A/B/C | Canonical = analyzer's stricter fractal rules (physically derived). Verified: does not change PipeSim's footprints for fractal-sized buffers (all golden-config buffers are multiples). |
| D4b | **910C L0C base alignment** | 32 | 1024 (fractal fp32) | Ground-truth block for the corrected 256 KiB 910C accumulator specifies base alignment 32 B. Recorded as delivered; the fractal size-alignment rule for the enlarged accumulator is to be re-confirmed against the datasheet in Phase B. |
| D5 | **UB bank count** | data field on the pool | hardcoded `banks = 8` in AKA3006 | Manifest field `memory_domains.UB.banks`; Phase C removes the hardcode. |
| D6 | **310P coverage** | present | absent | 310P manifest exists (UB explicitly 192 KiB per ground truth); analyzer simply has no entry until someone adds `isa_generation: davinci_v15` audit rules — acceptable, the loader degrades to "unknown chip" explicitly. Audit-layer fields the analyzer would need (event-id reservations) are omitted rather than invented. |
| D7 | **Aliases** | `910b`, `default`, … | `910b`, `atlas-a2`, … | Union in the manifest; each tool's resolver reads the same list. |
| D8 | **950PR topology** | Built-in `ascend950pr` DaVinci v3 (`arch35`) profile in `topology.py`: 1.65 GHz, 32 Cube / 64 Vector cores, 216 KiB tensor UB budget, 32 KiB SIMT DataCache, and MXFP4/MXFP8 profiles. | Not evaluated here | This implemented PipeSim profile is not a legacy 20-core 910B binning. Any equivalent shared-manifest representation remains part of this proposal. |

---

## 2. The meta-language

Proposed normative contract: `profiles/ascend_soc_profile.schema.json` (JSON
Schema draft-07, self-contained). The profile paths and inheritance rules in
this section describe the proposed cross-tool design; PipeSim's current
built-in profiles are implemented in `src/ascend_pipesim/topology.py`.
Examples considered by the proposal include an `ascend910b` profile and an
`ascend950pr` profile. Design decisions:

* **Tool-neutral union.** The schema is the superset; each consumer validates
  and reads its subset and ignores the rest (`additionalProperties` at the
  profile root is closed to catch typos, but consumers must tolerate unknown
  *sections* they don't implement by ignoring them at load time).
* **Three layers of truth:** memory/system facts (both tools), simulation
  layer (latencies, bursts, per-dtype rates), audit layer (alignments ×3,
  event IDs, features, UB partitioning). Layers live beside each other in one
  file because they describe one die.
* **Commercial labels are derivations, not chips.** In the proposed manifest
  model, `base_profile` +
  `commercial_label: true` inherits everything (deep merge: objects per-key,
  arrays/scalars replace) and may only override `system` binning metadata and
  identity fields. A commercial profile that edits pools/channels/dtype
  throughputs fails validation. This is a general design rule for future
  manifest derivations; it does not describe the active `ascend950pr`
  implementation, which is the DaVinci v3 (`arch35`) profile recorded in
  §1.5.
* **Versioned.** `schema_version: 1`; loaders reject unknown major versions.

Minimal custom chip example (all defaults inherited from `davinci_v2`):

```json
{
  "schema_version": 1, "soc_name": "ascend910b_edge_16c",
  "isa_generation": "davinci_v2", "base_profile": "ascend910b",
  "system": {"ai_cores": 16, "cube_cores": 16, "vector_cores": 32}
}
```

---

## 3. Loader & integration mechanics

### 3.1 `ascend_soc_manifest` (one stdlib-only file, shared verbatim)

```python
# ascend_soc_manifest.py  -- zero dependencies beyond json/pathlib/dataclasses
PROFILE_ENV   = "ASCEND_CHIP_PROFILES_DIR"
DEFAULT_DIRS  = ("<shared checkout>/profiles",)   # Phase A: each repo's profiles/
SCHEMA_NAME   = "ascend_soc_profile.schema.json"

@dataclass(frozen=True)
class SocProfile:
    name: str; raw: dict            # fully resolved (inheritance applied)
    def __getitem__(self, key): return self.raw[key]

def load_dir(path: Path) -> dict[str, SocProfile]:
    """Parse + validate every *.json in `path` (except the schema itself),
    resolve base_profile chains (cycle-checked, depth-capped), return
    {soc_name: profile}. Duplicate soc_name = load error, not shadowing."""

def resolve(name: str, profiles: dict[str, SocProfile]) -> SocProfile:
    """Alias-tolerant lookup: lowercase, strip '-_' (both tools' convention)."""

def available(profiles: dict[str, SocProfile]) -> list[str]:
    """Canonical names for CLI help / error messages."""
```

Validation is structural (required keys, enum/pattern/bounds checks mirroring
the schema) implemented by hand so neither repo gains a dependency; repos that
already have `jsonschema` may validate against the normative file instead. The
two implementations must agree — a conformance test feeds the same directory
to both.

Resolution order for the profiles directory:

1. `ASCEND_CHIP_PROFILES_DIR` (explicit, wins)
2. shared default (Phase A: per-repo `<repo>/profiles`; Phase D proposal: a
   standalone `D:\Projects\ascend-profiles` checkout both tools point at)
3. built-in Python profiles shipped in each tool (fallback, so both tools stay
   functional with the profiles directory deleted — the migration never makes
   JSON load-bearing for existing chips until Phase B's golden tests lock it in).

### 3.2 PipeSim consumption (`topology.py`)

* New: `architecture_from_profile(p: SocProfile) -> NpuArchitecture` maps the
  manifest onto the **existing** `architecture_from_dict` payload shape
  (pools, pipes, cube dtype_profiles, vector, system scalars). No new model
  classes — the manifest is just another producer of the current dict API.
* `_REGISTRY` stays; at import (or first `get_architecture`) `topology`
  registers every manifest profile via the existing
  `register_architecture(name, factory)` hook. Manifest names **add** to the
  registry; a manifest may shadow a built-in only when its content is
  golden-test-identical or explicitly versioned newer.
* The CLI's `--arch-file` / `--pipe-bw` / … overrides keep working unchanged
  on top of any profile-built architecture.
* The proposal would map a profile to `NpuArchitecture`. It does not change the
  active built-in `ascend950pr` topology described in §1.5; in particular,
  that topology is not 910B-identical.

### 3.3 ascend_analyzer consumption (`hardware.py`, CLI, checkers)

* New: `chipspec_from_profile(p: SocProfile) -> ChipSpec` — domains (three
  alignments), `block_bytes = access_block_bytes`, `vector_bytes`, event IDs,
  `features`, `ub_partitioning`, and the perf fields (mte1/mte2/fixpipe bpc,
  cube macs, read bpc, sync hand-off).
* `CHIP_PROFILES` becomes a lazily-populated registry: built-in Python specs
  first, then `load_dir()` results merged (manifest wins on collision so
  curated JSON supersedes provisional Python). `resolve_chip()` and the CLI
  error text already derive from the registry, so `--chip test_chip_custom`
  works with **zero** CLI changes once registration is dynamic; `--list-chips`
  learns `--arch`-style help from `available()`.
* Checker fixes in Phase C (hardcode removal, all now manifest-fed):
  * **AKA3006** `banks = 8` → `chip.domains[UB].banks`; interleave block →
    `access_block_bytes`.
  * **AKA1002** already reads `DomainSpec.base_alignment` — manifest-driven
    automatically once ChipSpecs are built from profiles.
  * **AKA1009** is *not* chip data (it is the static
    `TPosition → PhysicalDomain` mapping) and stays in code; the manifest
    only decides whether a position's domain exists on this part.

---

## 4. Phased migration plan

**Phase A — schema & shared assets (proposal; implementation pending).**
The proposed deliverables are `profiles/ascend_soc_profile.schema.json`,
`profiles/ascend910b.json`, `profiles/ascend950pr.json`, and this document.
Add remaining manifests (`ascend310p`, `ascend910c`, `ascend351x`) plus a cross-tool
`scripts/check_profiles.py` that asserts manifest→objects equivalence against
each tool's built-ins (surfacing D2/D3 as explicit failures until decided).
*Gate:* schema validates all manifests; equivalence green for 910B.

**Phase B — PipeSim integration.**
Add `ascend_soc_manifest.py` + `architecture_from_profile` +
registry wiring behind the resolution order in §3.1; keep built-ins as
fallback; new tests: manifest-built 910B probe JSON ≡ built-in probe JSON
(golden), the active DaVinci v3 `ascend950pr` topology remains distinct from
910B, and unknown-chip errors list manifest names.
*Gate:* all 27 existing tests pass unmodified + new profile tests.

**Phase C — analyzer integration.**
Port `ascend_soc_manifest.py` (identical file), `chipspec_from_profile`,
registry merge; de-hardcode AKA3006 banks; audit the codebase for remaining
chip literals (grep `CHIP_PROFILES`, `banks =`, `4096`, `6, 7` outside the
built-in fallback module). Analyzer's own `tests/` suite must pass.
*Gate:* zero hardcoded chip lists outside the built-in fallback; analyzer
regression suite green.

---

**Phase D — validation gate (the acceptance demo).**
Write a dummy `test_chip_custom.json` (910B-derived, 8 cores, odd UB size)
into a temp dir; `ASCEND_CHIP_PROFILES_DIR=<tmp>`; then
`npu-pipesim probe --arch test_chip_custom --gemm 512,512,512` and
`python -m ascend_analyzer kernel.cpp --chip test_chip_custom` must both run
with **no source changes and no rebuild**. Profile loading must preserve each
tool's current built-ins, including PipeSim's DaVinci v3 `ascend950pr`;
deleting the profiles dir restores built-in behaviour. Each phase ends with
both suites green; nothing is removed, only superseded, until its replacement
is proven equivalent.

---

**PipeSim project overview:** [README.md](../README.md)
· [Frontier exploration](FRONTIER_EXPLORATION.md)
