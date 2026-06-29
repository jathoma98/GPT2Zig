# Slang — Master Reference for GPT2Zig

*The golden source of truth for using **Slang 2026.12** as the shader language and **compile-time interface authority** for this Zig 0.16 + Vulkan 1.3 (SPIR-V 1.6) inference engine.*

Everything in this document was **empirically validated** against the actual `slangc 2026.12` binary, `zig 0.16.0`, `spirv-val`/`spirv-dis`, and the real reflection-JSON output — not reconstructed from memory. Where a claim was checked and found wrong (several were), the correction is called out inline. Test shaders, generated artifacts, and the working codegen prototype live under `tmp/slang-playground/work/`.

---

## 0. Thesis: one source of truth, derived everything

A `.slang` kernel is the **only** place shader/host shared knowledge is authored. Everything the Zig host needs to talk to that kernel is *derived* and **cannot drift**, with the deepest guarantees enforced as Zig **compile errors**:

```
kernel.slang ──slangc──▶ kernel.spv            (binary, @embedFile'd into the exe)
             ├─────────▶ kernel.reflect.json   (bindings / layout / dispatch facts)
             └─────────▶ kernel.d              (imported-module deps, for cache invalidation)
                               │
                        reflect_codegen        (host Zig tool, build-time)
                               │
                               ▼
                         bindings.zig          (generated: enums, structs, comptime asserts, helpers)
                               │
                       @import("…_bindings")
                               ▼
                           host Zig            (never hardcodes a binding index, offset, or workgroup size)
```

The graph is wired by **data dependencies** in `build.zig`, so editing the shader re-runs `slangc` → re-runs codegen → recompiles the host. Nothing is manual; nothing is committed-and-stale. This mirrors the project's existing `gen_bpe` / golden-codegen patterns exactly.

**Validation status: the full chain works end-to-end, proven for real.** The GEMM shader compiles + passes `spirv-val --target-env vulkan1.3`; the Zig codegen tool builds and runs; the generated `bindings.zig` imports into host Zig; and the comptime layout asserts both **pass** for a correct layout and **fail the build** for a corrupted offset or a `@Vector` regression (the intended safety net firing).

---

## 1. Toolchain & version pinning

| Component | Version | Notes |
|---|---|---|
| slangc | **2026.12** | Already auto-downloaded by `build.zig` to a temp cache (`slangcPath(b)`); `SLANG_VERSION = "2026.12"`. |
| Zig | 0.16.0 | Per CLAUDE.md. |
| SPIR-V | **1.6** | = Vulkan 1.3. **Not the default — must be requested (see §2).** |
| Validators | `spirv-val`, `spirv-dis` | Optional but recommended as a CI gate on every emitted `.spv`. |

**Pin the language version and the compiler together.** Slang's ABI is explicitly unstable (library major version still 0) and `-std latest` floats to whatever the newest rules become. Pin the concrete `-std 2026`, and pin the slangc that defines those rules (already done via `SLANG_VERSION`). A compiler bump then becomes an explicit, reviewable change rather than a silent semantic shift.

---

## 2. CLI flags — the definitive reference

All 18 flags surveyed exist in 2026.12. The corrections below matter.

### The four traps (read these first)

1. **`-capability spirv_1_6` is mandatory.** The default SPIR-V header is **1.5**, but Vulkan 1.3 expects 1.6. Nothing else bumps the header. *Verified: header flips `Version: 1.5` → `1.6` only with this flag.* This is the single most important and most silently-omittable flag.
2. **Omitting `-std` selects the legacy 2018 language.** `include/slang.h`: `DEFAULT == LEGACY == 2018`. Always pass **`-std 2026`** (not `latest` — it's a moving target).
   - *Correction to prior notes:* `-std` does **not** control enum scoping or parameter mutability. Enums are scoped in all versions (`-unscoped-enum` is the real toggle); `in` params are mutable copies in all versions. The trap (unset = 2018) is real; its previously-claimed *symptoms* were wrong.
3. **`-separate-debug-info` writes a content-hash-named sidecar** `<40hex>.dbg.spv`, **not** a name derived from `-o`. A build system must glob for it, not predict it. (We avoid this flag in the default profiles.)
4. **Default optimization is `-O1`, not `-O0`.** `0=none 1=default 2=high 3=maximal`. On the GEMM: O1=2492 B, O2=2152 B, O3=2152 B (no further win). **O2 is the release sweet spot**; re-measure O3 per-kernel only if a shader is large.

### Correctness flags — turn runtime GPU bugs into build failures

| Flag | What it buys | Use |
|---|---|---|
| `-restrictive-capability-check` | Escalates the late require-capability pass from warnings → **errors** (`E41013`). A coopmat op leaking into the portable build fails the build instead of trapping on a device. | **Always on.** Pairs with in-shader `[require(...)]`. |
| `-warnings-as-errors all` | No warning rots in the log (uninitialized-var `E41016` → error). | Always on. |
| *(keep)* SPIR-V validation | On by default; opt-out is `-skip-spirv-validation`. | **Never pass the opt-out.** |
| *(keep)* uninitialized-var check | On by default; `-disable-non-essential-validations` would silence it. | **Never pass it.** |
| `-validate-uniformity` *(experimental)* | Flags non-uniform control flow around wave ops (UB that silently corrupts subgroup reductions). | CI lint that *can* catch issues — needs specific patterns to fire, so treat as advisory, not a guarantee. |
| `-no-codegen` | Type-check + layout only, ~80 ms. Skips real SPIR-V emission. | Fast pre-commit/on-save lint. **Caveat: skips `static_assert` evaluation** (those need full codegen). |
| `-zero-initialize` | Deprecated, accepted-but-ignored. | Don't rely on it; initialize explicitly. |

### Target / layout / numerics

| Flag | Purpose |
|---|---|
| `-target spirv` + `-emit-spirv-directly` | Direct path is the **default**; the legacy `-emit-spirv-via-glsl` supports fewer features — never use it. |
| `-fvk-use-scalar-layout` (alias `-force-glsl-scalar-layout`) | **Always on.** Packs buffers/push-constants tightly so they match a Zig `extern struct` (see §4). Demonstrably changes emitted bytes. |
| `-matrix-layout-row-major` | Pin it to match the row-major host mental model. |
| `-fp-mode precise\|fast\|default` | `precise` disables result-changing FP reassociation/contraction — develop & validate against the CPU oracle with `precise` so a mismatch means a real bug; relax proven-correct hot kernels to `fast`. |
| `-denorm-mode-fp32 / -denorm-mode-fp16` (`any\|preserve\|ftz`) | Default `any` is implementation-defined → cross-vendor divergence on near-zero fp16. Set explicitly (requesting beats leaving it undefined; not every driver honors it). |
| `-preserve-params` | Keeps unused resource params so binding indices stay stable across kernel variants that share a descriptor-set layout. Costs dead descriptors, buys a uniform layout. |
| `-reflection-json <path>` | **The flag that drives codegen.** Works in one invocation alongside `-o out.spv`. |
| `-fspv-reflect` | *Distinct* — embeds reflection decorations *into* the `.spv` (grows it; JSON is byte-identical). Only add if you also do runtime reflection. We don't need it. |
| `-depfile <path>` | Emits Makefile-syntax deps **including `import`ed modules** — wired into the Zig build graph so editing an imported `.slang` busts the cache (§5). |

### Capability atoms (all accepted; `+`-joined)

- **Baseline (portable):** `spirv_1_6` + subgroup atoms `spvGroupNonUniformArithmetic`, `spvGroupNonUniformShuffle`, `spvGroupNonUniformBallot` + `spvVulkanMemoryModelKHR`.
- **Coopmat fast-path variant:** baseline `+ spvCooperativeMatrixKHR` (pulls in `VulkanMemoryModel`). Add `+spvBFloat16KHR` / `+spvFloat8EXT` only if the shader uses those dtypes.
- **Avoid on the portable path:** `SPV_EXT_physical_storage_buffer` (buffer device address — MoltenVK-fragile) and **64-bit atomics** (MoltenVK on Apple Silicon: `shaderBufferInt64Atomics = false` → dead on Apple; keep atomics 32-bit).

Both baseline and coopmat outputs pass `spirv-val --target-env vulkan1.3`. **Keep the coopmat variant strictly behind a runtime `VkPhysicalDevice` capability check** — MoltenVK lacks coopmat.

### The three finalized profiles

```bash
# (1) Fast lint / pre-commit  (~80 ms, no codegen)
slangc <in>.slang -target spirv -std 2026 \
  -no-codegen -warnings-as-errors all -restrictive-capability-check

# (2) Dev build  (debuggable, correctness-maximal, reflection + depfile)
slangc <in>.slang -target spirv -std 2026 -capability spirv_1_6 \
  -O0 -g3 \
  -warnings-as-errors all -restrictive-capability-check \
  -matrix-layout-row-major -fvk-use-scalar-layout \
  -reflection-json <in>.refl.json -depfile <in>.d -o <in>.spv
#   (do NOT add -disable-non-essential-validations; SPIR-V validation stays on)

# (3a) Release baseline  (portable subgroup path)
slangc <in>.slang -target spirv -std 2026 -capability spirv_1_6 \
  -O2 \
  -warnings-as-errors all -restrictive-capability-check \
  -matrix-layout-row-major -fvk-use-scalar-layout \
  -fp-mode precise -denorm-mode-fp32 preserve \
  -reflection-json <in>.refl.json -depfile <in>.d -o <in>.spv

# (3b) Release coopmat variant  (same source; runtime-gated)
#   add: -capability spirv_1_6+spvCooperativeMatrixKHR+spvVulkanMemoryModelKHR+spvGroupNonUniformArithmetic+spvGroupNonUniformShuffle+spvGroupNonUniformBallot
#   →  -o <in>.coopmat.spv
```

---

## 3. The reflection JSON schema (empirical map)

Emitter ground truth: `slang-2026.12/source/slang/slang-reflection-json.cpp`. The schema is **stable in shape**; the codegen must key off field *presence* defensively because several fields are conditional.

### Top-level document
```jsonc
{ "parameters": [ <Param> ],   // global params across ALL sets/spaces
  "entryPoints": [ <EntryPoint> ],
  "bindlessSpaceIndex": 1 }     // always present; the space reserved for bindless
```

### `<Binding>` — two mutually-exclusive shapes, keyed by `"kind"`
```jsonc
// 1. uniform shape — struct fields inside constant/push-constant buffers & StructuredBuffer<struct>
{ "kind": "uniform", "offset": 0, "size": 4, "elementStride": 0 }
//   offset/size/elementStride ALWAYS present; elementStride = per-element stride (0 for scalars/matrices)

// 2. resource/slot shape — everything else
{ "kind": "descriptorTableSlot", "space": 1, "index": 0, "count": 4 }
//   space  omitted ⇒ 0   (present ⇔ nonzero)
//   count  omitted ⇒ 1   (present ⇔ ≠1; does NOT give resource-array length — use type.elementCount)
//   index  always present
```

For a **compute** engine the only `kind`s seen are: `descriptorTableSlot` (SRV/UAV/CBV), `pushConstantBuffer`, `specializationConstant`, and `uniform` (struct fields).

> **String-or-int gotcha:** `offset`/`size`/`index`/`count` go through `emitReflectionSize`, which emits the **string** `"unbounded"` or `"unknown"` for runtime-sized/opaque cases instead of an int. Parse defensively. (Not hit by static compute kernels, but unbounded arrays would.)

### `<Type>` payloads (keyed by `"kind"`)
```jsonc
{ "kind": "scalar", "scalarType": "float32" }
{ "kind": "vector", "elementCount": 3, "elementType": <Type> }
{ "kind": "matrix", "rowCount": 4, "columnCount": 4, "elementType": <Type> }
{ "kind": "array",  "elementCount": 8, "elementType": <Type>, "uniformStride": 4 }

// resource
{ "kind": "resource", "baseShape": "structuredBuffer",
  "access": "readWrite",        // OPTIONAL — absent ⇒ read-only (SRV)
  "resultType": <Type> }        // element type; ABSENT for byteAddressBuffer (untyped)

// constantBuffer — BOTH ConstantBuffer<T> (UBO) AND push-constant blocks
{ "kind": "constantBuffer",
  "elementType": <struct with fields>,                 // ← struct layout lives HERE for CB/PC
  "elementVarLayout": { "binding": {"kind":"uniform","size":N} } }   // ← N = TOTAL block size

{ "kind": "struct", "name": "Dims", "fields": [ {"name","type","binding":<uniform>} ] }
```

### `<EntryPoint>`
```jsonc
{ "name": "main", "stage": "compute",
  "threadGroupSize": [16, 16, 1],                       // numthreads
  "parameters": [ {"name","semanticName","type"} ],     // OMITTED when no varying inputs
  "bindings": [ {"name", "binding"} ] }                 // which global params THIS entry uses (no type)
```

### Codegen robustness checklist (absent ⇒ default)

| Field | Rule |
|---|---|
| `binding.space` | absent ⇒ **0** |
| `binding.count` | absent ⇒ **1**; don't use for array length (use `type.elementCount`) |
| `type.access` | absent ⇒ **read-only SRV**; else `write` / `readWrite` / `rasterOrdered` / `append` / `consume` |
| `type.resultType` | absent for **byteAddressBuffer** |
| struct layout | under `type.elementType` for **ConstantBuffer/push-constant**, but under `type.resultType` for **StructuredBuffer<struct>** |
| total block size | read `type.elementVarLayout.binding.size` — **not** derivable from fields alone |
| `entryPoint.parameters` | omitted when no varying params |
| push-constant vs UBO | the **only** JSON difference is `param.binding.kind`: `pushConstantBuffer` vs `descriptorTableSlot` (both have `type.kind=="constantBuffer"`) |
| spec constants | top-level params with `binding.kind=="specializationConstant"`, id = `binding.index`, scalar type. **Default value is NOT in the JSON** — read SPIR-V `OpSpecConstant` or the `.slang` source if you need it |
| ints | may be the strings `"unbounded"`/`"unknown"` |

**Switch exhaustively on every `kind` enum** (no `else`-default) so a new Slang kind surfaces as a Zig compile error rather than silent mishandling — this is the project's exhaustive-switch discipline applied to the codegen.

### `scalarType` → Zig type, with byte sizes
`void`(0) · `bool`(spec-const = 32-bit id) · `int8`/`uint8`(1) · `int16`/`uint16`(2) · `int32`/`uint32`(4) · `int64`/`uint64`(8) · `float16`(2) · `float32`(4) · `float64`(8) · `bfloat16`(2) · `float_e4m3`(1, fp8) · `float_e5m2`(1, fp8). An `"unknown"` scalarType is an emitter error case — treat as hard error.

---

## 4. Layout / ABI — when Slang and a Zig `extern struct` coincide

This is the crown jewel: the codegen pins the Zig struct's byte layout to the shader's reflected layout via `comptime` asserts. The rule for when they match was determined empirically against real Zig `@offsetOf`/`@sizeOf`/`@alignOf`.

### The coincidence rule

**Under `-fvk-use-scalar-layout`, Slang packs every field at natural-size alignment with zero vec/array/struct padding — byte-for-byte identical to a Zig C-ABI `extern struct`, *provided the Zig side uses `[N]f32`/`[N]f16` arrays for vectors/arrays, never `@Vector`.***

Confirmed exact matches (scalar layout ↔ Zig `extern struct`): `{u32,u32,u32}`=12 B, `{f32,float2,float3,float4}`=40 B, `float4x4`↔`[16]f32`=64 B, `float[4]`=16 B (4 B stride), `{half,half}`=4 B, nested struct=16 B. `-fvk-use-c-layout` produced byte-identical results to scalar in every case — but prefer `-fvk-use-scalar-layout` (the established, validated Vulkan path).

### The traps

1. **`@Vector` is the real trap, not `vec3` per se.** Zig's `@Vector(3,f32)` is 16 B/align-16 (SIMD), `@Vector(2,f32)` is align-8 — so a `@Vector`-based struct mismatches scalar layout. With `[3]f32`, vec3 *does* match scalar (12 B). **Rule: never `@Vector` in a shared struct; always `[N]f32`.**
2. **std140 (default UBO) array stride balloons to 16 B/element.** `float[4]` in a default `ConstantBuffer` becomes 64 B. Only scalar (and std430 / StructuredBuffer) keep 4 B stride to match Zig `[4]f32`. std140 also aligns vec2→8, vec3/vec4→16, nested struct→16, and pads block totals to 16. **The scalar-layout flag eliminates all of this.**
3. **`[[vk::offset(N)]]` is silently ignored** — no error, no effect. Field **order is the only layout knob**. The codegen must not depend on manual offsets.
4. **StructuredBuffer element stride is not emitted** in the JSON. For a struct of scalars/scalar-arrays under scalar layout it equals the tight size (= Zig `@sizeOf(T)`). Compile any host-indexed `StructuredBuffer<T>` shader with scalar layout so indexing matches.

### Safe-authoring rules (put these in the shader style guide)

1. Always compile shared shaders with `-fvk-use-scalar-layout` (+ `-matrix-layout-row-major` if matrices are shared).
2. In shared push-constant / ConstantBuffer / StructuredBuffer structs, use **only** scalars (`uint`/`int`/`float`/`float16_t`) and **explicit scalar arrays** (`float[N]`).
3. Mirror on the Zig side with `extern struct`, using `[N]f32` / `[N]f16`. **Never `@Vector`.**
4. **Avoid `float3`/`vec3`** in shared structs even though scalar `float3` matches `[3]f32` — it's the single most error-prone type. Prefer `float[3]` or pad to `float4`.
5. Field order matters and is the only control. Let the comptime asserts verify everything.

### What the codegen asserts

The JSON reports per-field `offset` + `size` and the total block size, but **no per-field alignment**. So emit, for every shared struct:

```zig
comptime {
    std.debug.assert(@offsetOf(T, "field") == <uniform.offset>);   // every field
    std.debug.assert(@sizeOf(T) == <elementVarLayout.binding.size>); // total block size
}
```

Offset-of-every-field + total size **fully pins layout** — any padding/stride drift shifts a later offset or the total. Do **not** emit `@alignOf` asserts (alignment isn't in the JSON; the offset asserts already catch alignment-induced shifts). The asserts correctly **fail** if someone uses `@Vector` or drops the scalar flag — the intended safety net.

---

## 5. The codegen pipeline — end-to-end, working

### 5.1 The codegen tool (`tools/reflect_codegen.zig`)

A **pure, unit-testable core** `generate(gpa, json_bytes) ![]u8` plus a thin IO shell that matches the project's `tools/gen_bpe.zig` conventions exactly. This compiles, runs, and its unit test passes under Zig 0.16. (Full source in `tmp/slang-playground/work/pipeline/reflect_codegen.zig`; abridged here to the shape.)

```zig
//! Build-time codegen: Slang reflection JSON -> Zig bindings module.
const std = @import("std");

// --- Pure core (unit-testable, no IO) ---
pub fn generate(gpa: std.mem.Allocator, json_bytes: []const u8) ![]u8 {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var parsed = try std.json.parseFromSlice(std.json.Value, arena, json_bytes, .{});
    defer parsed.deinit();
    var bindings = std.ArrayList(Binding).empty;       // 0.16: unmanaged
    var push_fields = std.ArrayList(PushField).empty;
    const pipe = try parsePipeline(arena, parsed.value, &bindings, &push_fields);
    const hash = std.hash.Wyhash.hash(0, json_bytes);  // provenance hash
    var aw = std.Io.Writer.Allocating.init(gpa);        // 0.16: growable writer
    errdefer aw.deinit();
    try emit(&aw.writer, pipe, hash);
    return aw.toOwnedSlice();
}

// --- IO shell (matches tools/gen_bpe.zig) ---
pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const gpa = init.gpa;
    var args = init.minimal.args.iterate();
    _ = args.skip();
    const in_path = args.next() orelse return error.MissingInputArg;
    const out_path = args.next() orelse return error.MissingOutputArg;
    const cwd = std.Io.Dir.cwd();
    const json_bytes = try cwd.readFileAlloc(io, in_path, gpa, .unlimited);
    defer gpa.free(json_bytes);
    const out = try generate(gpa, json_bytes);
    defer gpa.free(out);
    try cwd.writeFile(io, .{ .sub_path = out_path, .data = out });
}
```

**Zig 0.16 API gotchas hit & fixed (all verified by compiling):**
- `std.ArrayList(T)` is **unmanaged**: `.empty`, `list.append(gpa, x)`, allocator per call.
- Growable string builder is `std.Io.Writer.Allocating.init(gpa)`; write via `&aw.writer` (a `*std.Io.Writer`), finish with `aw.toOwnedSlice()`. (No `ArrayList(u8).writer()`.)
- `std.json.parseFromSlice(std.json.Value, …)` returns a `Parsed` wrapper (`.value`, `defer .deinit()`); accessors `.object` (`std.json.ObjectMap`, `.get`→`?Value`), `.array.items`, `.string`, `.integer` (i64 → `@intCast`). The dynamic `Value` tree beats typed parsing given the two binding shapes + conditional fields.
- Tool entry: `pub fn main(init: std.process.Init)`, `init.io`/`init.gpa`, `init.minimal.args.iterate()`, `std.Io.Dir.cwd()` for IO — identical to `gen_bpe.zig`.

### 5.2 The generated `bindings.zig` (verbatim, for the GEMM)

```zig
//! GENERATED by reflect_codegen.zig from Slang reflection JSON. DO NOT EDIT.
const std = @import("std");

pub const reflection_hash: u64 = 0xa30bbb6edfb53eb5;
pub const descriptor_set: u32 = 0;

pub const Binding = enum(u32) { A = 0, B = 1, C = 2 };

pub const binding_access = [_]enum { read_only, write, read_write }{
    .read_only, // A
    .read_only, // B
    .read_write, // C
};
pub const binding_elem = [_][]const u8{ "float32", "float32", "float32" };

pub const PushConstants = extern struct { M: u32, K: u32, N: u32 };
pub const push_constant_size: u32 = 12;
comptime {
    std.debug.assert(@offsetOf(PushConstants, "M") == 0);
    std.debug.assert(@offsetOf(PushConstants, "K") == 4);
    std.debug.assert(@offsetOf(PushConstants, "N") == 8);
    std.debug.assert(@sizeOf(PushConstants) == 12);
}

// Per-entry-point namespace (one struct per [shader] function in the file).
pub const main = struct {
    pub const entry_point = "main";
    pub const local_size = [3]u32{ 16, 16, 1 };
    pub fn dispatchGroups(global: [3]u32) [3]u32 {
        const ls = @This().local_size; // qualify: avoids clash with the module-level flat alias
        return .{
            (global[0] + ls[0] - 1) / ls[0],
            (global[1] + ls[1] - 1) / ls[1],
            (global[2] + ls[2] - 1) / ls[2],
        };
    }
};

// Single-entry convenience: flat aliases so the host can write `bind.dispatchGroups(...)`.
// (Emitted ONLY when the file has exactly one entry point; see §5.3.)
pub const entry_point = main.entry_point;
pub const local_size = main.local_size;
pub const dispatchGroups = main.dispatchGroups;
```

**Safety net proven both directions:** the correct module builds + runs (exit 0); corrupting `"K" == 4` to `== 5` fails at compile time (`reached unreachable code` at the assert); swapping the three scalar fields for `@Vector(3,u32)` (simulating a dropped scalar-layout flag) fails with `no field named 'M'`. Both classes of drift are caught at `zig build`, never on the GPU.

### 5.3 Multi-entry-point kernels — the definitive approach

This was driven to certainty empirically (compiled four candidate layouts; see `tmp/slang-playground/work/pipeline/`). **The decision rule has two regimes, and which one applies is determined by a single question: do the kernels share the exact same binding set and push-constant block?**

#### The hard constraint that forces the rule

In a **whole-module compile**, every module-global resource declaration is reflected for **every** entry point, regardless of which entry actually uses it — and `-entry <name>` does **not** prune it (verified: selecting one entry still reflects all module globals). Worse, two kernels with different `[[vk::push_constant]]` structs produce **two** `pushConstantBuffer` params (`index` 0 and 1) in one module — and Vulkan permits only **one** push-constant block per pipeline layout. So:

- If kernels **share** the same buffers + one push-constant → whole-module reflection is clean and correct (one shared binding set, one push-constant, per-entry `threadGroupSize`).
- If kernels **diverge** → whole-module reflection is a polluted union with multiple push-constants. Unusable.

> **Ruled out: resources as entry-point `uniform` parameters.** It *does* scope bindings per kernel (each entry re-indexes from 0 under `entryPoints[i].parameters`), but mixing a `StructuredBuffer` and a uniform/push struct in one entry's parameter group emits `E31106`/`E31107` ("members … moved into another binding slot") — which become **hard errors under our `-warnings-as-errors all` profile** (verified). Don't use this pattern.

#### Regime 1 — kernels share bindings (e.g. multi-pass softmax, reduce→apply)

One `.slang` file, multiple `[shader("compute")]` functions, **whole-module compile** → one `.spv`, one reflection JSON. `parameters[]` is the single shared binding list; each `entryPoints[i]` contributes only its `name` + `threadGroupSize`. The codegen emits the shared decls **once** and one namespaced struct per entry point. **This is implemented and validated** — the tool loops over `entryPoints[]`:

```zig
// multi_bindings.zig (generated from a 2-kernel file sharing X, Y, and the `scale` push-constant)
pub const Binding = enum(u32) { X = 0, Y = 1 };          // shared, emitted ONCE
pub const PushConstants = extern struct { k: f32, n: u32 };
comptime { /* @offsetOf/@sizeOf asserts — shared */ }

pub const scaleKernel = struct {                          // one struct per entry point
    pub const entry_point = "scaleKernel";
    pub const local_size = [3]u32{ 64, 1, 1 };
    pub fn dispatchGroups(global: [3]u32) [3]u32 { ... }
};
pub const reluKernel = struct {
    pub const entry_point = "reluKernel";
    pub const local_size = [3]u32{ 32, 1, 1 };
    pub fn dispatchGroups(global: [3]u32) [3]u32 { ... }
};
```
Host: `bind.scaleKernel.dispatchGroups(.{n,1,1})` and `bind.reluKernel.dispatchGroups(...)` — different workgroup sizes, shared `bind.Binding` / `bind.PushConstants`. *(Validated: scale@64 → 2 groups, relu@32 → 4 groups for n=100; both consumers compile and run.)* When the file has **exactly one** entry point the tool also emits the flat aliases (`bind.dispatchGroups`, `bind.entry_point`) shown in §5.2, so single-kernel files stay terse.

#### Regime 2 — divergent kernels (the GPT-2 norm: layernorm ≠ gemv ≠ softmax)

**One kernel per `.slang` file**, entry named `main`, module-global resources; compile each separately; share code via `import helpers;`. Each file → its own `.spv` + reflection + namespaced bindings module. Reflection is clean and identical to the validated GEMM baseline (indices `0..N`, exactly one push-constant, **no warnings** even under `-warnings-as-errors all`). This is the default for the engine's kernel set; one slangc Run step + one codegen Run step per file in `build.zig`.

#### Decision rule (put this in the kernel style guide)

> Put kernels in the **same file** only when they share the **identical** buffer set and a single push-constant block (genuine multi-pass over the same data). Otherwise give each kernel **its own file**. When unsure, separate — Regime 2 is always correct; Regime 1 is an optimization for the shared-buffer case.

### 5.4 `build.zig` wiring (using this project's real patterns)

The project already: downloads slangc to a cache (`slangcPath(b)`), builds host tools with `.target = b.graph.host`, uses `addRunArtifact` + `addOutputFileArg` for codegen outputs, exposes build outputs via `mod.addAnonymousImport`, and mirrors generated files into `src/generated/` via `addUpdateSourceFiles`. The shader pipeline drops straight into that.

```zig
// 1) Host codegen tool.
const codegen_exe = b.addExecutable(.{
    .name = "reflect_codegen",
    .root_module = b.createModule(.{
        .root_source_file = b.path("tools/reflect_codegen.zig"),
        .target = b.graph.host,
        .optimize = .Debug,
    }),
});

// 2) Per shader: slangc Run -> .spv + .reflect.json (+ depfile for imports).
const slang_run = std.Build.Step.Run.create(b, "slangc gemm");
slang_run.addArg(slangcPath(b));
slang_run.addFileArg(b.path("shaders/gemm.slang"));     // tracks the MAIN .slang
slang_run.addArgs(&.{
    "-target", "spirv", "-std", "2026", "-capability", "spirv_1_6",
    "-fvk-use-scalar-layout", "-matrix-layout-row-major",
    "-warnings-as-errors", "all", "-restrictive-capability-check", "-O2",
});
slang_run.addArg("-reflection-json");
const reflect_json = slang_run.addOutputFileArg("gemm.reflect.json");
slang_run.addArg("-o");
const spv = slang_run.addOutputFileArg("gemm.spv");
slang_run.addArg("-depfile");
_ = slang_run.addDepFileOutputArg("gemm.d");  // ← std parses it & registers imported .slang deps

// 3) Codegen Run: reflect.json -> bindings.zig (this addFileArg wires slang_run -> gen_run).
const gen_run = b.addRunArtifact(codegen_exe);
gen_run.addFileArg(reflect_json);
const bindings_zig = gen_run.addOutputFileArg("bindings.zig");

// 4) Make both importable by the engine module.
exe_mod.addAnonymousImport("gemm_bindings", .{ .root_source_file = bindings_zig });
exe_mod.addAnonymousImport("gemm_spv",      .{ .root_source_file = spv });
// host: const gemm = @import("gemm_bindings");
//       const spirv align(4) = @embedFile("gemm_spv").*;   // SPIR-V is a u32 stream → align(4)

// 5) Mirror bindings into src/generated/ so a bare `zig test src/foo.zig` resolves it.
const update = b.addUpdateSourceFiles();
update.addCopyFileToSource(bindings_zig, "src/generated/gemm_bindings.zig");
```

**Cache invalidation — the one subtlety:**
- Editing the **main** `gemm.slang`: tracked by `addFileArg` → re-runs. ✅
- Editing an **imported** `helpers.slang`: *not* covered by the main `addFileArg`. **Use `addDepFileOutputArg`** (verified present in Zig 0.16 `std/Build/Step/Run.zig`): the child writes its discovered deps into the `.d`, and the Run step parses them and registers the imported `.slang` files as additional inputs. *(Confirmed empirically: `slangc -depfile` lists both `multi.slang` and the `import`ed `helpers.slang`.)* ✅
- `reflect.json → bindings.zig`: dependency auto-established by passing the JSON LazyPath into `gen_run.addFileArg`. ✅

---

## 6. The type-safety ladder

Six rungs, each a strictly stronger guarantee. The pipeline implements 1–4 and 6; 5 is a worthwhile add.

| Level | Guarantee | Mechanism |
|---|---|---|
| **1** Generated constants | binding indices / set / dispatch dims can't drift | `Binding` enum, `local_size` |
| **2** Generated push-constant struct | field *set* can't drift | `PushConstants extern struct` emitted from reflected fields |
| **3** Comptime layout asserts ★ | byte *layout* can't drift — the deepest static guarantee | `@offsetOf`/`@sizeOf` pinned to reflected offsets; mismatch = `zig build` error |
| **4** Typed handles + generated descriptor layout + dispatch helper | can't bind wrong slot / wrong descriptor type / wrong workgroup shape | `Binding` enum + `binding_access` + `dispatchGroups` |
| **5** Phantom-typed buffer elements | can't bind a `u32` buffer to an `f32` slot | wrap device buffers in `TypedBuffer(elemType)`; `binding_elem` drives the check |
| **6** Provenance hash | a stale `.spv` paired with fresh bindings is caught | `reflection_hash` (Wyhash of the JSON) baked into `bindings.zig` |

**What's left that reflection can't see:** runtime facts — actual buffer sizes vs `M·K`, device SPIR-V support, NaN-free inputs. Those stay as **host-side init asserts** (the project's aggressive-init-assert rule) and numerical validation against the CPU oracle. Reflection pins the *interface*; the oracle pins the *math*.

---

## 7. Slang language idioms & style (corrected & validated)

The mental shift: a kernel runs **per-lane in lockstep across a subgroup**; divergent branches serialize. **Uniform control flow is king**, and compile-time specialization replaces per-item branching/state-machines. The project's Zig style maps over almost verbatim, with the corrections below.

### Verified type-safety idioms, ranked (lean on these)

1. **Interface + struct conformance** — the crown jewel. `interface IWeightFormat { static const uint block_size; float dequant(...); }` + `struct Q4 : IWeightFormat {...}`. Omitting a required member is a **hard error `E38100`** on the struct declaration. This replaces Zig's duck-typed comptime structs with a *checked contract* — add a requirement and every non-conforming format fails to compile.
2. **Generics over interfaces, statically specialized** — `float dot<F : IWeightFormat>(...)`. Calling an op outside the contract → `E30027`. Concrete `<Q4>` is monomorphized (zero-cost, no dynamic dispatch). *Slang type-checks the generic once at definition* — no C++-template cascade.
3. **Generic `let` value params** — `void load<let TILE : int>(...)`. `int`/`uint`/`bool`/`enum` allowed; **`float` rejected** (`E30624`). Tile/head-dim/unroll counts as compile-time params, like Zig comptime ints.
4. **`static_assert(cond, "msg")`** — exact spelling; failure → `E41400`. **The correct tool for numeric invariants** (`TILE > 0`, vec4-alignment). ⚠️ Only fires under **full codegen, not `-no-codegen`**.
5. **Scoped enums** (default in `-std 2026`) — `Phase.load` required; bare `load` → `E30015`. Use as state-machine tags like Zig tagged-union enums.
6. **`[require(cap)]` + `-restrictive-capability-check`** — lowercase `[require(...)]`, comma-separated atoms. Makes a missing capability a build error (`E41013`) instead of a silent profile upgrade. The compile-time enforcement of the portable/coopmat split.
7. **`let` immutable bindings** — reassignment → `E30011`. (Do **not** rely on `in` params for immutability — see corrections.)
8. **`[vk::constant_id(N)]` spec constants** — canonical for explicit Vulkan pipeline wiring (emits `OpDecorate SpecId N` + `OpSpecConstant`).

### Corrections to the prior style guide (verified wrong)

- **`where D > 0` does NOT exist.** `where` does *type* constraints only (`where T : IFace`, `where T == U`, `where countof(Pack) == N`); numeric `>` fails to parse (`E20001`). **Use `static_assert` for numeric bounds.**
- **`in` params are NOT immutable in 2026.** Reassigning an `in`/default param compiles (copy-in value semantics; mutation stays local). The immutability win comes from **`let`**, not `in`.
- **`extension` swizzles must be qualified** — `this.x`, not bare `x` (bare → `E30015`).
- **`static_assert` requires full codegen** — it won't fire in the `-no-codegen` lint profile.
- **Spec constants:** both `[vk::constant_id(N)]` and `[SpecializationConstant]` work; prefer `[vk::constant_id(N)]` for explicit SpecId control.

### Confirmed-as-written

`__init` constructors, `[mutating]` methods, `groupshared` + `GroupMemoryBarrierWithGroupSync()`, `[unroll]`/`[loop]`/`[ForceInline]`, wave intrinsics (`WaveActiveSum/Max`, `WavePrefixSum`, `WaveGetLaneIndex/Count`, `WaveReadLaneFirst`), `module`/`import`, `printf`, `let`/`var`.

### Capability auto-inference (disassembled SPIR-V)

- `WaveActiveSum` → `GroupNonUniform` + `GroupNonUniformBallot` + `GroupNonUniformArithmetic`.
- Cooperative matrix lives in the **`linalg` namespace** (`using namespace linalg;`). Type `CoopMat<T, MemoryScope.Subgroup, R, C, Use>`; load via free fn `coopMatLoad<...>(buf, off, stride)`; store via **method** `mat.Store<Layout>(buf, off, stride)`. Pulls in `CooperativeMatrixKHR` + `VulkanMemoryModel`.

So in-shader `[require(...)]` is usually for documenting/locking intent and for the **coopmat split** — Slang auto-infers caps for well-formed intrinsics, and `-restrictive-capability-check` turns a profile shortfall into an error.

---

## 8. Anti-patterns & SIMT traps

- **Runtime interface existentials / dynamic dispatch** — storing "some `IMatmul`" and calling through it at runtime emulates a tagged union with a type-tag switch + memory copies (divergent, slow). Keep the type compile-time-known so it specializes. (Slang has **no tagged unions**; the host owns the state machine, kernels are pure leaves.)
- **Data-dependent divergent branches in hot loops** — serialize the subgroup. Predicate with `select`/`lerp`, or hoist the branch to a uniform spec constant. Reserve real branches for wave-uniform conditions.
- **`Optional<T>` / sentinels in hot paths** — an unwrap is a per-lane branch. Keep optionals at host boundaries.
- **Hardcoding subgroup width (32/64)** — breaks across NVIDIA/AMD/Apple. Always `WaveGetLaneCount()`; pair with `-validate-uniformity`.
- **Trusting `switch` exhaustiveness** — Slang's `switch` is C-like and **does not** check enum exhaustiveness (unlike Zig). Recover the guarantee by specializing over an interface (a missing method *is* a compile error), self-imposing enumerate-all-cases, or codegen-ing the arms.
- **Oversized register arrays** (`float reg[256]` per lane) spill and collapse occupancy — size with a generic value param.
- **`@Vector` in shared host structs** (§4) and **non-scalar layout flags** — silent layout corruption; the comptime asserts are the backstop, not a license to ignore the rule.

---

## 9. Verify-before-trust checklist (version-sensitive spots)

- **Reflection schema** is empirically mapped here against 2026.12; if the slangc version bumps, re-dump one reflection JSON and re-confirm field names. The codegen's defensive keying (§3) absorbs absent/conditional fields, but a *renamed* key needs a parser update.
- **`static_assert` needs full codegen** — the fast lint profile won't catch failing asserts; the dev/release builds will.
- **`addDepFileOutputArg`** confirmed in Zig 0.16; if it's ever absent, fall back to post-processing the `.d` into `slang_run.addFileInput(...)` calls.
- **SPIR-V version** — confirm the header is 1.6 (`spirv-dis | head`) after adding `-capability spirv_1_6`; the default is 1.5.
- **Coopmat is runtime-gated** — never let the coopmat `.spv` reach a MoltenVK device; check `VkPhysicalDevice` features first.
- **Run `spirv-val --target-env vulkan1.3`** on every emitted `.spv` in CI.
- **Multi-entry-point codegen** — implemented and validated (§5.3). Keep divergent kernels in separate files (Regime 2); only co-locate kernels that share the identical binding set + one push-constant (Regime 1). Never use entry-point `uniform` resource params — they fail `-warnings-as-errors all`.

---

*Throughline:* one source of truth (the `.slang`), a derived bridge (reflection JSON + depfile), generated Zig that can't drift, and comptime assertions that pin the shader's byte-level layout into Zig's type system. `-std 2026` + `-capability spirv_1_6` + `-fvk-use-scalar-layout` + `-restrictive-capability-check` are the four flags that do the real work; the boilerplate disappears and the bugs that remain are the ones reflection can't see — exactly where host-side init asserts and the CPU oracle take over.
