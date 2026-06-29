# Compute Shaders & LLM Kernels — a guide for the systems programmer who's new to LLMs

You know threads, caches, DRAM bandwidth, SIMD, and ABIs. You do **not** (yet) know what a
"layernorm" is or why a transformer multiplies so many matrices. This guide bridges the two: for
every kernel in this engine it shows the **CPU algorithm** we deleted, the **Slang compute shader**
that replaced it, the **intuition** for what the op actually means in an LLM, and the **optimizations**
a real engine would apply. The CPU source lived in `src/core/op.zig` (the math) and
`src/core/model.zig` (the forward pass); the shaders live in `src/shaders/`.

---

## Part 0 — The mental model (read this first)

### A compute shader is a parallel `for` loop

Here is the single most important translation. A CPU kernel is a serial loop:

```zig
for (0..N) |i| out[i] = f(in[i]);   // one core walks i = 0, 1, 2, ...
```

A GPU compute shader is the **body** of that loop, launched once per `i` across thousands of
hardware threads simultaneously:

```hlsl
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= N) return;     // the loop bound becomes a guard
    out[i] = f(in[i]);      // the loop body, run by one thread for its own i
}
```

There is no loop over `i` in the shader — **the hardware is the loop**. Your job writing a kernel is:
"given my thread's global index, which one output element do I produce, and how?" In this engine every
kernel uses a **flat 1-D index**: the host computes the total number of output elements, launches that
many threads, and each thread decodes its `(row, col)` or `(head, query, dim)` from `tid.x` with
divides and mods. (Uniform and boring on purpose — correctness first.)

### The execution hierarchy (maps onto things you know)

| GPU concept | Closest systems analogy |
|---|---|
| **Thread / invocation** | One iteration of the parallel for-loop |
| **Subgroup / warp** (32 threads) | A SIMD lane-group executing in lockstep — *one* program counter for 32 lanes |
| **Workgroup** (`[numthreads(64,1,1)]`) | A cooperating thread block that shares fast scratch memory and can barrier among itself |
| **Dispatch / grid** | The whole `vkCmdDispatch(gx,gy,gz)` launch — all the workgroups |
| **`groupshared` memory** | A software-managed L1 scratchpad (you load/evict it by hand) |
| **Global / device memory** | VRAM — high latency, high bandwidth, the thing you must not thrash |
| **Registers** | Per-thread registers; spilling them kills occupancy like stack spills kill a hot loop |

**SIMT is the gotcha.** The 32 threads of a subgroup share one instruction pointer. If they take
different branches (`if (j > i)`), the hardware runs *both* sides with the inactive lanes masked off —
divergence serializes. Uniform control flow (all lanes branch the same way) is free; data-dependent
per-lane branches in a hot loop are the GPU equivalent of a mispredicted branch in your innermost loop,
except worse.

**Occupancy** is the GPU's hyperthreading: each core keeps many subgroups in flight and switches
between them to hide memory latency. More live threads = more latency hiding — *unless* each thread
uses too many registers or too much shared memory, which caps how many fit. It's the classic
throughput-vs-resource tradeoff.

### How a thread finds its data: bindings & push constants

Two ways the host hands data to a kernel — and they mirror calling conventions you already know:

- **Storage buffers (descriptors)** = pointers to big arrays in VRAM. The shader declares
  `StructuredBuffer<float> X;` (read-only) or `RWStructuredBuffer<float> Y;` (read-write). The host
  binds an actual `(buffer, offset, length)` to each slot before the launch. Think "an array of args
  passed by reference."
- **Push constants** = a tiny struct (≤128 bytes) of scalars baked into the command, delivered
  basically in registers. We pass dimensions here: `struct Push { uint M, K, N; }`. Think "the
  by-value scalar args" — `M`, `N`, a learning-rate, etc.

The clever part of this codebase: the push-constant struct's **byte layout is verified at compile
time** against what the shader compiler reports (see `research/slang-overview.md`). If the Zig struct
and the shader struct ever disagree on a field offset, the build fails instead of corrupting a
dispatch. That's the kind of compile-time guarantee a systems programmer appreciates.

### Our deliberately-dumb execution model

This engine runs **one kernel at a time, fully to completion, with a full queue-wait-idle between
every single op, and no barriers**. It is the GPU equivalent of running each parallel-for, joining all
threads, then starting the next — totally serial at the op level. It is slow and it is *correct*, which
is exactly the right starting point. Almost every optimization in this guide is about **breaking that
serialization**: fusing ops so intermediate results never hit VRAM, and reusing data so you don't
re-read it from VRAM a thousand times.

### What a transformer forward pass actually does (the 60-second version)

You feed in a list of **token IDs** (integers, ~50k vocabulary). You want a probability distribution
over the *next* token. The pipeline:

1. **Embed**: turn each token ID into a vector of `n_embd=768` floats (a learned "meaning" vector),
   plus a position vector. Now you have an `[S, 768]` matrix — `S` tokens, each a 768-d row. This
   matrix is the **residual stream**, the central bus that flows through everything.
2. **12 transformer layers**, each of which does two things and *adds* its result back into the
   residual stream:
   - **Attention**: let each token look at earlier tokens and pull in relevant information (this is
     how "it" finds its antecedent, how context propagates).
   - **MLP**: per-token nonlinear processing (a 2-layer feed-forward net), the "thinking" applied to
     each position independently.
3. **Unembed**: project the final `[S, 768]` back up to `[S, 50257]` — a score for every vocabulary
   word at every position. The last row's argmax is your predicted next token.

Crucial fact for optimization: **~99% of the floating-point work is matrix multiplies** (steps 1's
projections, the MLP, and the unembed). Everything else — layernorm, softmax, gelu, the adds — is
cheap arithmetic but **memory-bound**: dominated by shuffling the `[S,768]` stream in and out of VRAM.
That split (compute-bound matmuls vs memory-bound everything-else) drives every optimization decision
below.

---

## Part 1 — The matmul (the kernel that matters)

Everything else is a rounding error next to this. Get matmul right and you've captured ~99% of the
FLOPs.

### What it means intuitively

A linear layer `y = x @ W + b` takes each input vector (a row of `x`) and produces an output vector by
taking **weighted sums** of its components — `W` is the table of weights, learned during training.
Geometrically: a rotation/projection/scaling of the input vectors into a new space. In the transformer
this is how you go from "768-d meaning vector" to "Q, K, V vectors," or from "768-d" to the MLP's
"3072-d hidden," etc. It's the workhorse; the model's knowledge lives almost entirely in these `W`
matrices.

### CPU version (`op.zig`)

```zig
// y[i,j] = b[j] + Σ_k x[i,k] * w[k,j].  x:[M,K], w:[K,N], b:[N], y:[M,N]
pub fn matmul(x, w, b, y, m, k, n) void {
    for (0..m) |row| {
        for (0..n) |col| {
            var acc: f32 = b[col];
            for (0..k) |i| acc += x[row*k + i] * w[i*n + col];  // dot product
            y[row*n + col] = acc;
        }
    }
}
```

The classic triple loop: for every output cell `(row, col)`, dot the `row`-th input row with the
`col`-th weight column.

### Shader version (`matmul.slang`)

```hlsl
void main(uint3 tid : SV_DispatchThreadID) {
    uint t = tid.x;
    if (t >= p.M * p.N) return;
    uint row = t / p.N, col = t % p.N;     // recover (row,col) from flat index
    float acc = B[col];
    for (uint k = 0; k < p.K; ++k)
        acc += X[row*p.K + k] * W[k*p.N + col];
    Y[t] = acc;
}
```

**The translation is exact:** the outer two CPU loops (`row`, `col`) *vanished* — they became the
launch grid, one thread per `(row,col)`. The inner `k` loop survives inside each thread, because a dot
product is inherently sequential (you're accumulating). So `M*N` threads each run a `K`-length loop.
For the GPT-2 QKV projection that's `4 tokens × 2304 outputs = 9216` threads each looping 768 times —
versus a CPU doing all 7 million MACs on one core. That's the win, for free, from the dumb version.

### Why the naive version is slow, and how real matmul kernels fix it

The naive kernel is **compute-bound in theory but bandwidth-bound in practice** because of *redundant
global memory traffic*. Look at the access pattern: thread `(row, col)` reads the entire `row` of `X`
and the entire `col` of `W` from VRAM. The thread next to it, `(row, col+1)`, reads *the same row of
`X` all over again*. Across the whole launch, each element of `X` is read from VRAM `N` times and each
element of `W` is read `M` times. You're paying DRAM bandwidth for data you already had.

The fixes, in rough order of impact:

1. **Shared-memory tiling.** Have a workgroup cooperatively load a `tile × tile` block of `X` and of
   `W` into `groupshared` (your manual L1), then every thread in the group computes its partial products
   from the fast scratchpad before loading the next tile. This turns `O(N)` redundant global reads into
   `O(N/tile)`. This is *the* classic GPU matmul optimization — directly analogous to cache blocking a
   CPU GEMM so the working set fits in L1.
2. **Register blocking (micro-tiles).** Each thread computes a small `4×4` or `8×8` block of outputs
   instead of one, reusing each loaded value across multiple accumulators held in registers. Raises
   arithmetic intensity (FLOPs per byte loaded) — same idea as unrolling a CPU GEMM to keep the FMA
   units fed.
3. **Coalesced access.** When 32 threads of a subgroup read `W[k*N + col]` for consecutive `col`, they
   hit 32 *consecutive* addresses — the hardware fuses that into one wide memory transaction
   (coalescing). The naive kernel actually gets this right for `W` and `Y`; the read of `X[row*K+k]` is
   the same address for all of them (a broadcast, also fine). Coalescing is the GPU's version of
   "sequential DRAM access is 10× faster than random" — and getting your indexing wrong here silently
   tanks bandwidth.
4. **Cooperative matrix / tensor cores.** Modern GPUs have dedicated hardware that multiplies small
   matrix tiles (e.g. 16×16) in a single instruction, often in fp16/bf16. Slang exposes these as
   `CoopMat` (see `slang-overview.md` §7). This is where the real FLOPS are — often 8–16× over hand-rolled
   FMA loops. The catch: not portable (MoltenVK on Apple lacks it), so it lives behind a runtime
   capability check.
5. **Mixed precision.** Store/multiply weights in fp16 or int8, accumulate in fp32. Halves or quarters
   the bandwidth (the binding constraint) and unlocks the tensor-core path. The accumulation stays fp32
   to preserve accuracy.

For intuition: a well-tuned GPU GEMM is ~99% shared-memory tiling + register blocking + tensor cores.
Our naive one leaves easily 50–100× on the table — but it's *correct*, and correctness is what the
goldens prove.

---

## Part 2 — The "transpose" matmul (`matmul_bt`)

### What it means

The output projection at the very end: `logits = lnf @ Wᵀ`. GPT-2 **ties** its weights — the same
`wte` table that maps token-ID → vector is reused (transposed) to map vector → token-scores. So
`logits[i,j]` = how well token `i`'s final vector matches vocabulary word `j`'s embedding. It's a
giant similarity search against all 50,257 words.

### CPU vs shader

```zig
// y[i,j] = Σ_k x[i,k] * w[j,k]   — note w indexed [j,k] not [k,j]: that's the transpose, no copy needed
for (0..m) |row| for (0..n) |col| {
    var acc: f32 = 0;
    for (0..k) |i| acc += x[row*k + i] * w[col*k + i];
    y[row*n + col] = acc;
}
```

```hlsl
uint row = t / p.N, col = t % p.N;
float acc = 0.0;
for (uint k = 0; k < p.K; ++k) acc += X[row*p.K + k] * W[col*p.K + k];
Y[t] = acc;
```

Identical structure to `matmul`; the only change is `W[col*K + k]` instead of `W[k*N + col]`. That
indexing *is* the transpose — we read `W`'s rows instead of its columns, so no physical transpose is
needed.

### Optimizations specific to it

- **This is the single biggest matmul in the model** (`N = vocab = 50257`). All the matmul tricks
  above apply, but the transposed access pattern hurts coalescing: consecutive `col` threads now read
  *different rows* of `W` (stride `K` apart), which is the bad, uncoalesced pattern. A tuned kernel
  pre-transposes `W` into a coalescing-friendly layout, or tiles to hide it.
- **The killer optimization: only compute the last row.** During greedy generation you only need the
  *next* token, which depends only on `logits[S-1]` — the last token's row. Computing logits for all `S`
  positions (as we do) is `S×` wasted work in the most expensive kernel. Real inference computes one
  row: `logits = lnf[last] @ Wᵀ`. Combined with a KV cache (Part 9) this is the difference between
  O(S²) and O(S) per generated token.

---

## Part 3 — Layernorm

### What it means

Before each attention/MLP block, the residual stream is **normalized**: each token's 768-d vector is
re-centered to mean 0 and scaled to variance 1, then re-scaled/shifted by learned `gamma`/`beta`. Why?
Deep networks are numerically fragile — activations drift to huge or tiny magnitudes and training
diverges. Layernorm keeps every vector on a comparable scale so the next matmul behaves. Think of it as
**automatic gain control** on each token vector. (It normalizes *across the 768 features of one token*,
independently per token — not across tokens.)

### CPU version

```zig
pub fn layernorm(row, gamma, beta, eps, out) void {
    var sum: f32 = 0; for (row) |v| sum += v;
    const mean = sum / d;                          // pass 1: mean
    var sq: f32 = 0; for (row) |v| sq += (v-mean)*(v-mean);
    const inv_std = 1.0 / @sqrt(sq/d + eps);       // pass 2: variance
    for (0..d) |i| out[i] = (row[i]-mean)*inv_std*gamma[i] + beta[i];  // pass 3: apply
}
```

### Shader version (`layernorm.slang`) — note the design choice

```hlsl
void main(uint3 tid) {
    uint r = tid.x;
    if (r >= p.rows) return;
    uint base = r * p.cols;
    // ONE THREAD does the whole row: the two reduction passes + the apply pass, sequentially.
    float sum = 0; for (k) sum += In[base+k];           float mean = sum/fcols;
    float sq  = 0; for (k) { float d=In[base+k]-mean; sq += d*d; }
    float inv_std = 1.0/sqrt(sq/fcols + p.eps);
    for (k) Out[base+k] = (In[base+k]-mean)*inv_std*Gamma[k] + Beta[k];
}
```

Here the parallelism is **one thread per row** (per token), and the inner work over the 768 features
stays a sequential loop inside the thread. This is the simplest correct mapping but it's leaving
parallelism on the floor: a *reduction* (sum over 768 elements) is exactly the thing GPUs parallelize
well, and we're doing it serially.

### Optimizations

- **One workgroup per row + parallel reduction.** Assign 256 threads to a row; each sums a slice into
  `groupshared`, then a log-step tree reduction (or a single `WaveActiveSum` subgroup intrinsic)
  combines them. Turns the 768-long serial sum into ~log₂(768) steps. This is the textbook parallel
  reduction — same pattern as a parallel `std::accumulate`.
- **Welford / one-pass mean+variance.** The CPU does two passes over the row (mean, then variance).
  Welford's online algorithm computes both in a single pass, halving the memory reads. Layernorm is
  **memory-bound** (it reads each element ~3× here), so fewer passes ≈ proportional speedup.
- **Fusion.** Layernorm reads the residual stream from VRAM and writes a normalized copy back, only for
  the next matmul to read it again. A fused kernel computes the normalization in the matmul's *prologue*
  so the normalized vector never round-trips through VRAM. This is the recurring theme: the bottleneck
  isn't the math, it's the VRAM traffic between dumb separate kernels.

---

## Part 4 — Softmax

### What it means

Softmax turns a vector of arbitrary scores into a **probability distribution** (positive, sums to 1).
`softmax(x)_i = e^{x_i} / Σ e^{x_j}`. In attention it converts raw query·key "relevance scores" into
attention weights ("pay 70% attention to token 3, 20% to token 1, ..."). The exponential makes it
"soft argmax" — it sharply favors the largest score but keeps everything differentiable.

### CPU vs shader — and the numerical-stability trick

```zig
pub fn softmax(row, out) void {
    var max = row[0]; for (row[1..]) |v| max = @max(max, v);   // ← subtract the max
    var sum: f32 = 0;
    for (0..d) |i| { const e = @exp(row[i]-max); out[i]=e; sum+=e; }
    for (out) |*v| v.* *= 1.0/sum;
}
```

```hlsl
float mx = In[base]; for (k>=1) mx = max(mx, In[base+k]);
float sum = 0; for (k) { float e = exp(In[base+k]-mx); Out[base+k]=e; sum+=e; }
for (k) Out[base+k] *= 1.0/sum;
```

Why subtract the max first? `e^{x}` overflows to `inf` for `x` around 89 (fp32). Subtracting the row
max makes the largest exponent `e^0 = 1` and everything else `≤ 1` — mathematically identical (the max
cancels in the ratio), numerically safe. This is a stability trick worth internalizing; it shows up
everywhere exponentials do. Same structure as layernorm: **two reductions** (max, then sum) plus an
apply pass, done serially per row in our naive kernel.

### Optimizations

- **Parallel reduction**, exactly as layernorm (max-reduce and sum-reduce are both tree reductions /
  `WaveActiveMax` + `WaveActiveSum`).
- **Online (one-pass) softmax.** You can compute the running max and running sum in a *single* pass,
  rescaling the partial sum each time you discover a new max. This is the mathematical heart of
  **FlashAttention** (Part 8) and cuts the passes over the data.
- **Fuse it into attention** — see Part 8. As a standalone op, softmax is pure memory traffic.

---

## Part 5 — GELU (the activation function)

### What it means

The MLP needs a **nonlinearity** between its two matmuls — without it, two stacked linear layers
collapse into one linear layer and the network can't learn anything interesting. GELU ("Gaussian Error
Linear Unit") is a smooth gate: it passes large positive values through, squashes negatives toward
zero, with a soft S-curve transition. Intuitively it's a "soft ReLU" that lets the network decide how
much of each hidden feature to keep. GPT-2 uses the `tanh` approximation specifically (matching it
exactly matters for parity with the reference).

### CPU vs shader — a trivial 1:1 map

```zig
const inner = 0.7978845608 * (v + 0.044715*v*v*v);
out[i] = 0.5 * v * (1.0 + std.math.tanh(inner));
```

```hlsl
float inner = GELU_C * (v + GELU_A * v*v*v);
Out[i] = 0.5 * v * (1.0 + tanh(inner));
```

Pure elementwise: one thread per element, no loop, no reduction. The most "embarrassingly parallel"
kernel in the set. (The magic constant `0.7978... = √(2/π)`.)

### Optimizations

GELU is **100% memory-bound**: it does a handful of flops but one read and one write per element of a
big `[S, 3072]` buffer. So:

- **Fuse it into the preceding matmul's epilogue.** The MLP's first matmul produces the hidden vector;
  apply GELU right there before writing out, so the `4×n_embd` hidden never makes a separate VRAM
  round-trip. This is the single biggest win and is standard in every real inference engine.
- **Vectorize** (`float4` loads/stores) to use the full memory bus width.
- A lookup table or cheaper polynomial approximation is rarely worth it — `tanh` is fast and the op is
  bandwidth-bound, not ALU-bound.

The lesson: a kernel this cheap should almost never exist as its own dispatch. It exists here only
because our model is deliberately un-fused.

---

## Part 6 — Embedding lookup

### What it means

The very first step: token ID → vector. `wte` is a `[50257, 768]` table; token `id`'s vector is just
`wte[id]` — a **gather**. We also add `wpe[position]` so the model knows token *order* (attention
itself is order-blind; position embeddings inject "this is the 3rd token"). 

### CPU vs shader

```zig
for (0..S) |s| for (0..n_embd) |d|
    x[s*E + d] = wte[ids[s]*E + d] + wpe[s*E + d];
```

```hlsl
uint s = t / p.E, d = t % p.E;
uint id = Ids[s];
X[t] = Wte[id*p.E + d] + Wpe[s*p.E + d];
```

One thread per output element `(token, feature)`. The interesting wrinkle: `Ids[s]` is a **data-
dependent index** into `Wte` — a scattered read (the token IDs are arbitrary). 

### Optimizations

- It's a tiny fraction of runtime, so rarely worth tuning, but: the gather is **memory-bound and
  scattered**. Within one token's row the `d` accesses are consecutive (coalesced); across tokens the
  base addresses jump around (each token grabs a different `wte` row). That's fine — each row is
  contiguous.
- The `wpe` add is already fused in (good). In a leaner engine, embedding for a *single* new token per
  step (with a KV cache) is one tiny gather, essentially free.

---

## Part 7 — Residual add

### What it means

The "residual stream" is an accumulator bus. Each block computes a *delta* and adds it back:
`x = x + attention(x)`, then `x = x + mlp(x)`. This is the **residual connection** — the reason very
deep nets train at all: gradients (and information) flow straight down the `+` path without being
mangled by every layer. Architecturally it means each layer *edits* the running representation rather
than *replacing* it.

### CPU vs shader

```zig
for (0..S*n_embd) |idx| x[idx] += proj[idx];
```

```hlsl
uint i = tid.x;
if (i >= p.n) return;
X[i] += Y[i];     // X is RWStructuredBuffer (read+write), Y read-only
```

The most trivial kernel possible: elementwise `a += b`. One thread per element.

### Optimizations

Same story as GELU — **pure bandwidth, should be fused**. The add is the natural epilogue of the matmul
that produced `proj`/`mlp`: accumulate directly into the residual buffer instead of writing a temp and
re-reading it. As its own dispatch it's 2 reads + 1 write of the whole stream for one flop each.

---

## Part 8 — Attention (the three-kernel set, and the heart of the transformer)

This is the conceptually rich one. We split it into three kernels (`attn_scores`, `attn_softmax`,
`attn_weighted_sum`) because that's the simplest correct decomposition; a real engine fuses all three.

### What attention means intuitively

Each token needs to gather context from other tokens. The mechanism is a **soft, content-addressed
lookup** — associative memory, basically:

- Each token emits a **query** ("what am I looking for?"), a **key** ("what do I offer?"), and a
  **value** ("what I'll hand over if you pick me"). These are three different linear projections of the
  token's vector (computed by the QKV matmul before attention).
- Token `i` scores every earlier token `j` by `query_i · key_j` — a dot product measuring relevance.
  High dot product = "token j is what I'm looking for."
- Those scores go through **softmax** → attention weights (a probability distribution over earlier
  tokens).
- Token `i`'s output is the weighted average of all the **values**: `Σ_j weight_{ij} · value_j`. It
  has pulled in a blend of information from the tokens it found relevant.

"Causal" / "masked" attention means token `i` can only look at `j ≤ i` (the past), because the model is
predicting the future and mustn't peek. That's the `j <= i` bound in our kernels. **Multi-head**:
this happens `n_head=12` times in parallel with different projections, so the model can attend "by
syntax" in one head, "by topic" in another, etc.

This is *the* operation that lets "it" resolve to the right noun, lets a fact stated early inform a
token generated late, etc. It's also the one with `O(S²)` cost (every token scores every other), which
makes it the scaling bottleneck for long sequences.

### Kernel 1 — scores (`attn_scores.slang`)

CPU (from `model.zig`'s `applyLayer`):
```zig
for (0..i+1) |j| {                       // causal: only j <= i
    var dot: f32 = 0;
    for (0..head_dim) |d| dot += q[d] * k[d];
    scores[j] = dot * scale;             // scale = 1/sqrt(head_dim)
}
```
Shader: one thread per `(head, query i, key j)`; each does the `head_dim`-long dot product. The
`1/√head_dim` scale keeps the dot products from growing with dimension (which would saturate softmax).
Threads with `j > i` write 0 and bail (the causal mask). The full scores tensor is `[n_head, S, S]` —
**this is the `O(S²)` memory blowup**, 50 MB at `S=1024`.

### Kernel 2 — softmax over keys (`attn_softmax.slang`)

Per `(head, query)`, softmax the `i+1` valid scores — identical to Part 4, applied to each causal row.
The row length is `i+1` (varies per query — the causal triangle).

### Kernel 3 — weighted sum of values (`attn_weighted_sum.slang`)

```zig
for (0..i+1) |j| { const p = scores[j]; for (d) out[d] += p * v[d]; }
```
One thread per `(head, query, dim)`; sum the value vectors weighted by the attention probabilities.
Output is `[S, n_embd]` with the heads concatenated.

### Why this is slow, and FlashAttention

The three-kernel version commits the cardinal GPU sin: it **materializes the `[n_head, S, S]` scores
matrix to VRAM**, writes it, reads it back for softmax, writes it, reads it back for the weighted sum.
For long sequences that's the dominant cost, and the memory itself becomes the limit (quadratic in `S`).

**FlashAttention** is the famous fix and it's worth understanding the trick: fuse all three kernels
into **one** that never writes the scores matrix to VRAM at all. It tiles over keys/values, keeping a
running **online softmax** (Part 4) — a running max and running weighted-value-sum that it rescales as
it streams through key/value tiles held in shared memory. The `S²` scores live only in registers/shared
memory, never VRAM. Result: same math, but memory traffic drops from `O(S²)` to `O(S)` and it runs many
times faster. This is *the* attention optimization; everything in production uses some descendant of it.

Other wins:
- **Don't launch the upper triangle.** Half our `attn_scores` threads compute `j > i` and immediately
  return — wasted. Launch only the causal lower triangle.
- **KV cache** (next section) makes attention incremental: a new token only computes *its* query
  against *cached* keys/values, turning per-step attention from `O(S²)` to `O(S)`.
- **Multi-query / grouped-query attention** (architectural): share keys/values across heads to shrink
  the KV cache — a memory optimization newer models use.

---

## Part 9 — The optimizations that dwarf all the others

Per-kernel tuning matters, but for *inference* (generating tokens one at a time) two structural changes
beat everything above:

### KV cache — stop recomputing the past

This engine has **no KV cache**, and it's the biggest inefficiency by far. Watch what the generation
loop does: to produce token 41, it runs the *entire* forward pass over all 41 tokens — recomputing the
keys and values for tokens 1–40 that haven't changed since last step. Generation is `O(S²)` total work
when it should be `O(S)`.

The fix: **cache** each layer's keys and values. A previously-seen token's K and V never change, so
compute them once and keep them in a GPU buffer. Each new step only: embeds 1 token, computes its Q/K/V,
appends K/V to the cache, attends against the cached K/V, and unembeds 1 row. Per-step cost drops from
"reprocess everything" to "process one token." This is *the* reason real LLM serving is fast, and it's
the highest-leverage change you could make to this codebase.

### Kernel fusion — stop round-tripping VRAM

Our model dispatches ~15 kernels per layer with a full `vkQueueWaitIdle` between each, and every
intermediate (`ln`, `qkv`, `attn`, `hidden`, ...) is written to VRAM and read back by the next kernel.
The memory-bound ops (layernorm, softmax, gelu, add, attention) spend most of their time on this
traffic, not math. Real engines **fuse**: layernorm into the next matmul's prologue, gelu and the
residual add into the previous matmul's epilogue, the three attention kernels into one FlashAttention
pass. Fewer kernels, fewer VRAM round-trips, fewer pipeline barriers. The naive one-op-at-a-time model
here is the thing you'd dismantle first for speed.

### The supporting cast

- **Batching.** Process many sequences at once so the big matmuls have more rows — turns
  bandwidth-bound matmuls compute-bound and amortizes weight loads. (Less relevant for single-stream
  greedy decode, central to serving.)
- **Quantization.** Store weights in int8/int4/fp8. Weights *are* the memory bottleneck during decode
  (you stream all ~500 MB of them per token); halving their size nearly doubles throughput.
- **Tensor cores / cooperative matrix.** Dedicated matrix-multiply hardware in fp16/bf16 — the
  difference between "GPU" and "fast GPU" for matmul.
- **Overlap & async.** Drop the per-op `vkQueueWaitIdle`; use barriers/semaphores so the GPU pipelines
  consecutive ops and overlaps compute with memory transfers. Our serial model exists purely for
  debuggability.

---

## TL;DR cheat sheet

| Kernel | What it does (LLM intuition) | Parallelism | Bound by | Top optimization |
|---|---|---|---|---|
| `matmul` | linear layer; the model's knowledge | thread per output cell | compute (huge) | tiling + register blocking + tensor cores |
| `matmul_bt` | unembed: vector → vocab scores | thread per output cell | compute (biggest matmul) | tiling; **only compute last token's row** |
| `layernorm` | per-token automatic gain control | thread per row (naive) | memory | parallel reduction + fuse into next matmul |
| `softmax` | scores → probabilities | thread per row (naive) | memory | online one-pass + fuse into attention |
| `gelu` | MLP nonlinearity (soft gate) | thread per element | memory | fuse into matmul epilogue |
| `embed` | token ID → vector (gather) | thread per element | memory (scattered) | trivial; fuse `wpe` add (done) |
| `add` | residual stream accumulate | thread per element | memory | fuse into matmul epilogue |
| `attn_scores` | query·key relevance, causal | thread per (head,q,k) | memory (`S²` blowup) | **FlashAttention** (fuse all 3) |
| `attn_softmax` | relevance → attention weights | thread per (head,q) | memory | FlashAttention online softmax |
| `attn_weighted_sum` | gather values by weight | thread per (head,q,dim) | memory | FlashAttention |

**The two structural wins above all kernel tuning: a KV cache (stop recomputing the past) and fusion
(stop round-tripping VRAM).** This engine deliberately does neither — it's the correct, legible
baseline the goldens pin down, and every item in this guide is a knob you can now turn from a position
of "I know exactly what the slow-but-correct version does."
