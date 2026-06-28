# Model-Parallel Cross-Machine Inference for GPT-2

*A systems-engineer's guide to splitting our naive forward pass across machines.*

Scope: how **layer-level model parallelism** works, taken concretely from the forward
pass we already have in [`src/core/model.zig`](../src/core/model.zig). We split the 12
transformer layers across N machines so the model's weights need not fit in one device's
memory. No batching, no KV cache, no latency hiding yet — just the bare mechanics of
*what data crosses the wire, and when*. Those come later and are noted where they bolt on.

---

## 0. TL;DR

A transformer forward pass is a **pipe**. The only state that flows from one layer to the
next is a single matrix — the *residual stream* `x : [S, n_embd]`. For GPT-2 small that's
`[S, 768]` f32. Everything else (the ~28 MB of weights per layer) is *resident local
state* that never moves.

That one fact is the whole trick:

- **Cut the layer loop** ([`model.zig:205`](../src/core/model.zig#L205)) at any layer
  boundary. Put layers `0..k` on machine A, `k..m` on machine B, etc.
- At each cut, machine A ships the residual stream `x` to machine B. That's it. ~3 KB per
  token per boundary.
- The wire payload is a raw little-endian f32 blob — a `memcpy` of a Zig slice, no
  serialization library.

Because the comms is just "send a slice of floats, receive a slice of floats," the same
Zig binary cross-compiles to every machine and the networking logic is shared and unit-
testable over loopback. The heterogeneity is purely a *deployment-time* decision about how
many layers each machine owns.

---

## 1. Three flavors of parallelism (and which one we're doing)

People say "model parallelism" to mean several different things. Pin down the taxonomy
first, because the comms pattern is completely different for each.

| Flavor | What gets split | What crosses the wire | Chattiness |
|---|---|---|---|
| **Data parallel** | nothing — every machine has the *full* model, different inputs | gradients (training only) | N/A for inference |
| **Tensor parallel** | each weight *matrix* is sliced column/row-wise across machines | partial sums, **every sub-layer**, via all-reduce | very high — multiple syncs *per layer* |
| **Pipeline / layer parallel** | *contiguous groups of whole layers* | the residual stream, **once per cut** | low — one send per boundary |

We are doing the third one. "Model-level parallelism = split layers across devices" in
your task description is **pipeline parallelism** (a.k.a. layer parallelism). It is by far
the friendliest to a heterogeneous, loosely-coupled cluster:

- **Tensor parallel** needs an all-reduce *inside every attention and every MLP* — i.e.
  ~24 synchronization barriers per token for GPT-2. That only makes sense over NVLink /
  InfiniBand where latency is sub-microsecond. Over Ethernet between a laptop and a Pi it
  would be catastrophic.
- **Pipeline parallel** needs *one* message per machine boundary per forward pass. Over a
  LAN that's fine. This is the right primitive for "highly portable heterogeneous PoC."

So: we keep each layer *whole and intact* (the kernels in `op.zig` are untouched), and we
cut *between* layers.

> **Noob note — what is an "activation"?** In systems terms: the residual stream `x` is the
> pipe's *message buffer*. Each layer reads it, does math against its local weights, and
> writes an update back into it (literally `x[idx] += proj[idx]` at
> [`model.zig:248`](../src/core/model.zig#L248) and `x[idx] += mlp[idx]` at
> [`model.zig:264`](../src/core/model.zig#L264)). The weights are the *program*; `x` is the
> *data flowing through it*. "Activation" = the value of `x` (or any intermediate) at some
> point in the pipe.

---

## 2. The one fact that makes it work: the residual stream is the only inter-layer state

Look at our layer loop. Stripped to its skeleton:

```zig
// model.zig:205
for (0..cfg.n_layer) |layer| {
    const L = self.w.layers[layer];          // <- this layer's weights (local, resident)

    layernormRows(x, L.ln_1_w, ...);          // reads x
    op.matmul(ln, L.attn_w, ...);             //   |
    // ...attention over heads...             //   |  all scratch (ln, qkv, attn, proj,
    op.matmul(attn, L.attn_proj_w, ...);      //   |  hidden, mlp) is INTERNAL to the layer
    for (...) |idx| x[idx] += proj[idx];      // writes x   <- residual add

    layernormRows(x, L.ln_2_w, ...);          // reads x
    op.matmul(ln2, L.fc_w, ...);              //   |
    op.gelu(hidden, hidden);                  //   |
    op.matmul(hidden, L.mlp_proj_w, ...);     //   |
    for (...) |idx| x[idx] += mlp[idx];       // writes x   <- residual add
}
```

Two invariants, both visible in the code:

1. **The only thing read at the *start* of a layer and written at the *end* is `x`.**
   Every other buffer — `ln`, `qkv`, `attn`, `proj`, `hidden`, `mlp`, `scores` — is
   `Scratch` ([`model.zig:64`](../src/core/model.zig#L64)), allocated once and reused
   *within* a layer. None of it survives across the loop iteration. So nothing scratch-
   related ever needs to cross a machine boundary.

2. **Each layer's weights are independent.** `self.w.layers[layer]` is a self-contained
   `Layer` struct ([`model.zig:23`](../src/core/model.zig#L23)). Layer 7 needs *only*
   layer 7's weights to run, given `x`. So machine B can hold *only* layers `4..8`'s weights
   and nothing else.

Therefore the cut between layer `k-1` and layer `k` is clean: the entire coupling between
the two halves is the value of `x : [S, n_embd]` at that point. Hand B that matrix and B
can continue as if it were a local loop.

This is *not* true of tensor parallelism, where a single matmul's output is spread across
machines and must be glued back together mid-layer. Here the seam is already at a natural
joint.

---

## 3. The full pass, as a pipeline of stages

The complete GPT-2 forward pass has three structural parts. Mapping them to pipeline
stages:

```
                  ┌─────────────────────────────────────────────────────────────┐
   token ids ───► │ HEAD  : embed (wte+wpe)         model.zig:192-198             │
   [S] u32        │         → x:[S,768]                                           │
                  └───────────────────────────────┬─────────────────────────────┘
                                                  │  x:[S,768]  (residual stream)
                  ┌───────────────────────────────▼─────────────────────────────┐
                  │ BODY  : transformer layers 0..n_layer   model.zig:205-269    │
                  │         (this is what we partition across machines)          │
                  └───────────────────────────────┬─────────────────────────────┘
                                                  │  x:[S,768]
                  ┌───────────────────────────────▼─────────────────────────────┐
   logits   ◄──── │ TAIL  : final layernorm + tied logit matmul                  │
   [S,50257]      │         ln_f → x @ wteᵀ          model.zig:273-278           │
                  └─────────────────────────────────────────────────────────────┘
```

- **HEAD** is two table lookups and an add. Trivial compute, but it *owns the big embedding
  table* `wte : [50257, 768]` (~154 MB at f32 — the single largest tensor in the model).
- **BODY** is 12 identical layers. This is ~85% of the FLOPs and the part we slice.
- **TAIL** is one layernorm plus the output projection `x @ wteᵀ`
  ([`op.matmulBT`](../src/core/op.zig#L37)). Note: GPT-2 **ties** the output embedding to
  the input embedding — the TAIL reuses the *same* `wte` table as the HEAD
  ([`model.zig:278`](../src/core/model.zig#L278)). Hold that thought; it creates a wrinkle
  in §5.

A concrete 3-machine BODY split (12 layers ÷ 3):

```
  Machine A:  HEAD + layers 0,1,2,3
  Machine B:         layers 4,5,6,7
  Machine C:         layers 8,9,10,11 + TAIL
```

The cuts are between layer 3→4 (A→B) and 7→8 (B→C). At each cut, the upstream machine
sends `x` and the downstream machine resumes the loop with its own layer weights. The loop
index is just a global counter; B's `for` runs `4..8`, C's runs `8..12`.

---

## 4. What actually crosses the wire, and when

This is the heart of your question. Walk one **prefill** (encode the prompt) and then one
**generation step**, naming every message and its size. GPT-2 small: `n_embd=768`,
`vocab=50257`, f32 = 4 bytes.

### 4a. Prefill — the prompt "Hello, I am" (S = 4 tokens)

```
 time   machine A                machine B                machine C
 ────   ──────────────────────   ──────────────────────   ──────────────────────
  t0    encode prompt → ids[4]
        embed → x:[4,768]
        run layers 0..3
  t1    ── send x (12 KB) ─────►  receive x:[4,768]
  t2                              run layers 4..7
  t3                              ── send x (12 KB) ────►  receive x:[4,768]
  t4                                                       run layers 8..11
                                                           ln_f, x @ wteᵀ → logits[4,50257]
                                                           argmax(last row) → next_id
  t5    receive next_id (4 B) ◄──────────────────────────  ── send next_id (4 B) ──
        append to ids → ids[5]
        ... repeat for next token ...
```

Message inventory for prefill:

| Edge | Payload | Shape | Bytes (S=4) |
|---|---|---|---|
| A → B | residual stream | `[4, 768]` f32 | 12,288 (~12 KB) |
| B → C | residual stream | `[4, 768]` f32 | 12,288 (~12 KB) |
| C → A | next token id | `[1]` u32 | 4 |

**Key design decision — sample on the TAIL, return only the token.** The naive thing would
be for C to send the full `logits : [S, 50257]` back to A. That's `S × 50257 × 4` ≈ **196 KB
per row** — and *all but the last row are useless* during generation (greedy decoding only
looks at the last position, [`main.zig:66`](../src/main.zig#L66)). So C should run the
argmax locally and send back **4 bytes**. The decode/sampling logic lives on whichever
machine owns the TAIL. This collapses the return hop from ~200 KB to nothing.

### 4b. Each subsequent generation step (no KV cache)

Our current loop ([`main.zig:63`](../src/main.zig#L63)) has **no KV cache**: every step
re-runs `forward()` over the *entire* sequence so far. Distributed, that means the *whole
growing* residual stream flows through the whole pipe every step:

| Step | Seq len S | A→B and B→C payload each | C→A |
|---|---|---|---|
| prefill | 4 | 12 KB | 4 B |
| +1 | 5 | 15 KB | 4 B |
| +2 | 6 | 18 KB | 4 B |
| ... | ... | `S × 768 × 4` | 4 B |
| +40 | 44 | 132 KB | 4 B |

The per-boundary payload is `S × n_embd × 4` bytes and grows linearly with the sequence.
Forty tokens of generation = forty full pipe traversals, each re-sending an ever-longer `x`.
Total wire traffic is modest (low single-digit MB for a 40-token generation) — the cost
here is *latency and redundant compute*, not bandwidth (see §7).

> **Why does the whole sequence flow, not just the new token?** Because of *attention*. Token
> `i`'s output depends on the keys/values of *all* tokens `j ≤ i`
> ([`model.zig:221-240`](../src/core/model.zig#L221)). Without a KV cache, machine B doesn't
> remember token 0..3's keys from the previous step — so A must resend the full prefix's `x`
> for B to recompute them. §6 explains how a per-stage KV cache fixes exactly this, and why
> layer-parallelism makes it clean.

---

## 5. Where to cut, and the tied-embedding wrinkle

### Heterogeneous partitioning — cut by capacity, not evenly

The whole point of splitting is *memory*: fit a model no single device can hold. So
partition the layers **proportional to each machine's free RAM**, not into equal counts.

Per-layer resident weight cost (GPT-2 small, f32):

| Tensor | Shape | Params | Bytes |
|---|---|---|---|
| `ln_1` (w+b) | `2×768` | 1,536 | 6 KB |
| `attn.c_attn` (w+b) | `768×2304 + 2304` | 1,771,776 | 6.76 MB |
| `attn.c_proj` (w+b) | `768×768 + 768` | 590,592 | 2.25 MB |
| `ln_2` (w+b) | `2×768` | 1,536 | 6 KB |
| `mlp.c_fc` (w+b) | `768×3072 + 3072` | 2,362,368 | 9.01 MB |
| `mlp.c_proj` (w+b) | `3072×768 + 768` | 2,360,064 | 9.00 MB |
| **per layer** | | **~7.09 M** | **~27.0 MB** |

Plus the non-layer tensors:

| Tensor | Shape | Bytes |
|---|---|---|
| `wte` (token embedding, **tied**) | `50257×768` | ~154 MB |
| `wpe` (position embedding) | `1024×768` | ~3 MB |
| `ln_f` (w+b) | `2×768` | 6 KB |

So a machine assigned `L` layers needs `L × 27 MB` of weight RAM, plus activations
(`x` and scratch, a few MB at most for `n_ctx=1024`). A beefy box takes 8 layers (~216 MB),
a Raspberry Pi takes 2 (~54 MB). The runtime cost of a layer is also ~constant, so layer
count is a decent proxy for both the memory *and* compute a machine signs up for.

**Each machine loads only its shard.** Today `Model.init` ([`model.zig:107`](../src/core/model.zig#L107))
loads *all* layers via `loadAligned`. The distributed version parameterizes that: machine B
calls `loadAligned` only for `h.4.*` … `h.7.*`. The safetensors file is the same on every
box (memory-mapped, [`safetensors.zig`]); each machine just touches a different *subset* of
named tensors. The mmap means unused tensors are never paged in — so holding the full file
on disk but only mapping your shard's pages is essentially free. (Long-term, you'd ship each
machine only its shard, but for a PoC "same file, load your slice" is simplest.)

### The tied-embedding wrinkle

`wte` is used at **both ends**: the HEAD embeds with it
([`model.zig:193`](../src/core/model.zig#L193)) and the TAIL scores with its transpose
([`model.zig:278`](../src/core/model.zig#L278)). That 154 MB table is the single biggest
chunk of the model and we'd rather not store it twice.

Three options:

1. **Replicate `wte` on HEAD and TAIL machines.** Simple, costs +154 MB on one extra machine.
2. **Ring topology — make HEAD and TAIL the *same* machine.** A owns `wte`, does the embed,
   runs layers 0..3, and *also* owns the TAIL. The pipe is a ring `A → B → C → A`, and the
   loop closes naturally: C sends its final `x` back to A, A does `ln_f` + `x @ wteᵀ` +
   argmax locally (it already has `wte` resident), appends the token, and starts the next
   step. **One copy of `wte`, natural loop closure.** ← recommended for the PoC.
3. **TAIL sends `ln_f(x)` back to HEAD for the logit matmul.** Keeps `wte` single-copy but
   adds a `[S,768]` hop *and* puts the 50257-wide output matmul on the HEAD machine.
   Strictly worse than (2).

Recommendation: **ring**. Machine A = HEAD + first layers + TAIL. It's the only machine
holding `wte`, the loop literally closes back to where the token ids live, and the
return-hop payload is the 4-byte token id from §4. The ring also matches the autoregressive
loop's shape: generation is inherently `A → … → A → A → …`.

```
   ┌──────────────────────────────────────────────────────┐
   │  A: wte/wpe, embed, layers 0..3,  AND  ln_f + logits  │
   └───┬──────────────────────────────────────────▲───────┘
       │ x:[S,768]                    next_id:[1]  │ (4 B)
       ▼                                           │
   ┌────────────────┐   x:[S,768]   ┌──────────────┴───────┐
   │ B: layers 4..7 │ ────────────► │ C: layers 8..11      │
   └────────────────┘               └──────────────────────┘
        ▲    │ x:[S,768]
        └────┘  (B→C)
```

---

## 6. Where state lives: the KV cache (the natural next step)

You said skip latency hiding — and a KV cache *is* partly a latency optimization — but its
*placement* is a structural fact about distributed inference worth stating now, because it
determines *where state lives* and dramatically changes the comms pattern.

**What it is.** Attention at token `i` needs the keys `K` and values `V` of every prior
token ([`model.zig:221-240`](../src/core/model.zig#L221)). Those depend only on the prior
tokens' `x` *at that layer* — they don't change once computed. A KV cache stores, per layer,
the `K` and `V` rows computed so far, so a new token only computes *its own* K/V and reuses
the rest. Per layer the cache is `[S, n_embd]` for K and the same for V.

**Why layer-parallelism makes it clean.** The KV cache for layer `ℓ` is needed *only* by the
machine that owns layer `ℓ`. There is **no cross-machine KV movement** — each stage keeps the
KV for *its own* layers, locally, forever. The cut is still just the residual stream.

This flips the per-step comms from §4b:

| | No KV cache (today) | Per-stage KV cache (future) |
|---|---|---|
| What HEAD sends per step | the **whole** prefix `x : [S,768]` | just the **new token's** `x : [1,768]` |
| A→B payload at step S | `S × 768 × 4` (grows) | `768 × 4` = 3 KB (constant) |
| Each stage recomputes | all S tokens, all its layers | only the 1 new token |
| State held per stage | none between steps | `K,V : [S,768]×2` per owned layer |

So with per-stage KV caches, generation becomes: A embeds *one* new token → 3 KB down the
pipe → each stage attends it against its locally-cached K/V → C/A argmax → 4 B token back.
Constant 3 KB per hop regardless of how long the sequence gets. That's the elegant target
architecture; the no-cache version in §4 is the honest naive baseline we build first.

---

## 7. The cost of "naive": the pipeline bubble

With a single sequence and no overlap, **exactly one machine is busy at a time**. While A
runs layers 0..3, B and C idle; when B runs, A and C idle. The total latency per forward
pass is:

```
  T_pass ≈ T_compute(A) + T_net(A→B) + T_compute(B) + T_net(B→C)
                        + T_compute(C) + T_net(C→A)
```

i.e. the *sum* of every stage's compute plus every hop — no parallel speedup at all on a
single stream. This idle time is the **pipeline bubble**. We accept it for the PoC because:

- The goal is *capacity* (run a model too big for one box), not *throughput*.
- Hiding the bubble needs **micro-batching / multiple in-flight sequences** (stage A works on
  sequence 2 while B works on sequence 1) — that's the "latency hiding" explicitly out of
  scope here. The point to internalize: the layer-parallel *structure* is what later makes
  micro-batching possible; it's a strict extension, not a rewrite.

For bandwidth, note the traffic is tiny (KBs per hop) — the bottleneck is **per-hop latency
and serialized compute**, which is exactly what KV-cache (§6) and micro-batching (out of
scope) attack. Don't reach for compression or fancy transports; a plain TCP stream is
oversized for 3 KB messages.

---

## 8. The wire protocol — shared, cross-compiled, testable Zig

The comms layer is deliberately boring, which is what makes it portable and testable.

### Payload format: raw little-endian f32, no serialization library

The residual stream is a contiguous `[]f32`. On every target we care about (x86-64, arm64,
both little-endian, IEEE-754 f32), the wire format is **the bytes of the slice, verbatim**.
Sending is `std.mem.sliceAsBytes(x)`; receiving is `@memcpy` into a pre-sized buffer then
`bytesAsF32` (we already have [`tensor.bytesAsF32`](../src/core/tensor.zig)). No protobuf, no
JSON, no per-element encoding.

- **Assert endianness at startup**, don't convert: `comptime assert(native_endian == .little)`.
  If we ever target a big-endian box, that's a loud compile-time failure, not silent
  corruption — consistent with the project's "move errors to compile time" rule.
- **Float determinism caveat.** Different CPUs/compilers may contract `a*b+c` into a fused
  multiply-add or reorder a SIMD reduction, producing bit-different (not wrong) results.
  Our kernels are naive scalar `ijk` ([`op.zig:13`](../src/core/op.zig#L13)) so today they're
  deterministic across machines, but once we SIMD-ize per machine, cross-arch results may
  diverge in the last bits. The existing `1e-3` activation tolerance and the *tolerance-free
  argmax* check ([`model.zig:399`](../src/core/model.zig#L399)) are exactly the right
  acceptance bar: greedy decoding is robust to sub-`1e-3` drift, so a heterogeneous cluster
  should still pick the same tokens.

### Framing

A minimal length-prefixed frame is enough. One header struct + payload:

```zig
const Frame = enum(u8) {
    activations,  // payload: S * n_embd f32  — residual stream for a stage boundary
    token,        // payload: 1 u32           — sampled next-token id (loop closure)
    _,
};

const Header = extern struct {
    kind: Frame,        // what the payload is
    seq_len: u32,       // S — lets the receiver size its buffer & resume the layer loop
    req_id: u32,        // generation step #, for sanity-asserting ordering
    payload_bytes: u32, // = seq_len * n_embd * 4 for .activations
};
// send: writev([Header bytes][payload bytes]); recv: read Header, then read payload_bytes.
```

Transport: a plain TCP stream per directed edge (A→B, B→C, C→A). Connection bring-up — DNS/
dial → handshake (exchange `n_embd`, layer ranges, assert agreement) → ready — is a textbook
fit for the project's **reduce → decide → transition** state-machine pattern (the same shape
as `ensureVenvReady`'s FSM in `build.zig`). Each stage's "am I HEAD / BODY / TAIL, who's
upstream, who's downstream" is decided once from config, then the per-step loop is a straight
line.

### Why this is shared and testable

Because a stage is *just* "read a `[S,768]` from upstream, run my layer range, write a
`[S,768]` to downstream," the entire distributed pass can run **in one process over a
loopback socket** (or even an in-memory pipe) with the layer split set to `[0..12]` on a
single "machine." That means:

- **One binary, N roles.** Cross-compile the same executable for every target
  (`zig build -Dtarget=aarch64-linux` etc.). A runtime flag / config says which layers this
  instance owns and who its neighbors are. The networking and the math are identical
  everywhere.
- **A loopback integration test** wires HEAD→BODY→TAIL stages over `127.0.0.1` in one test
  binary and asserts the distributed logits match the single-process `forward()` goldens we
  already have. The split is invisible to correctness: same residual stream in, same logits
  out. This is the cross-machine analog of the existing bisection test
  ([`model.zig:315`](../src/core/model.zig#L315)) and reuses the same activation goldens.

---

## 9. Putting it together — a PoC milestone sketch

Mapping the above onto incremental, testable steps (each one keeps the single-process
`forward()` as the oracle):

1. **Parameterize the layer range.** Make `Model.init` load a `[lo, hi)` shard of layers and
   `forward()` accept an entry `x` and produce an exit `x` for that range. Single process,
   call three shards back-to-back, assert it equals the whole-model `forward()`. *No
   networking yet — this proves the cut is clean.*
2. **Stage struct + loopback transport.** Wrap each shard in a stage that reads/writes `x`
   over a stream. Run all stages in one process over loopback; assert logits match goldens.
3. **Ring loop closure + sample-on-tail.** TAIL does argmax, sends the 4-byte token back to
   HEAD; HEAD appends and drives the autoregressive loop (§5 ring). Reproduce the
   `main.zig` generation output ("Hello, I am sorry, …") but with the model split across
   three in-process stages.
4. **Cross-compile + real machines.** Same binary to a second box; config sets the layer
   split and neighbor addresses. Heterogeneous partition by RAM (§5).
5. **(Later, out of scope here)** per-stage KV cache (§6), then micro-batching to fill the
   pipeline bubble (§7).

The invariant that makes every step verifiable: **the distributed pass is bit-for-bit (to
`1e-3`) the same computation as the local one** — only the residual stream's journey changes.

---

## Appendix: quick reference

GPT-2 small dimensions ([`config.zig`](../src/core/config.zig)): `n_layer=12`, `n_head=12`,
`n_embd=768`, `head_dim=64`, `n_ctx=1024`, `vocab=50257`, f32.

| Quantity | Formula | Value |
|---|---|---|
| Residual stream, 1 token | `n_embd × 4` | 3,072 B (~3 KB) |
| Residual stream, S tokens | `S × n_embd × 4` | S × 3 KB |
| Per-layer weights | — | ~27 MB |
| `wte` (tied embed/unembed) | `50257 × 768 × 4` | ~154 MB |
| Full logits row | `vocab × 4` | ~196 KB |
| **Loop-closure token id** | `1 × 4` | **4 B** |
| Full model (f32) | 124 M params × 4 | ~497 MB |

The two numbers that matter: **~3 KB per token crosses each cut** (cheap, this is why
layer-parallel works over a LAN), and **4 B closes the loop** (sample on the tail, never ship
logits). Everything else is resident local state that never moves.
