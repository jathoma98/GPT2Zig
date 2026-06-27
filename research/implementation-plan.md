# GPT-2 124M Zig Inference Engine — Implementation Plan

This is the *executable* companion to [`gpt2-inference-engine-guide.md`](gpt2-inference-engine-guide.md).
The guide tells you the architecture and the gotchas; this doc tells you **the order to build in,
what to prove green before moving on, and why each piece of math exists** for someone fluent in
linear algebra but new to LLMs.

## How to read this

Each milestone is a **build-then-prove** unit:

1. Build the milestone's code.
2. Run its golden/unit test until green.
3. *Only then* move to the next milestone.

The whole point is that you never debug a 12-layer black box. At every step you can answer
"am I numerically correct so far?" with a hard pass/fail against a Python oracle, not vibes.

Correctness milestones (M0–M5) get the full treatment: overview, why-it-exists (n00b math),
pseudocode, unit tests, integration tests, footguns. Perf milestones (M6–M10) are lighter:
overview, why, footguns, and how to measure.

### Dependency graph / critical path

```
M0  Scaffolding + goldens
 │
 ├──────────────┐
 ▼              ▼
M1 Loader     M4 Tokenizer        ← parallel track, decoupled from forward-pass validation
 │            (validated alone     (uses tiktoken golden; needs NO model math)
 ▼             vs tiktoken)
M2 Kernels        │
 │                │
 ▼                │
M3 Forward pass   │   ← validated with HARDCODED ids [15496,11,314,1101], no tokenizer needed
 │                │
 └───────┬────────┘
         ▼
        M5 Sampling + generation loop + detokenize  ← end-to-end, capstone test
         │
         ▼
   ┌─────┴─────┬──────┬──────┬──────┐
  M6 KV     M7 Thread M8 SIMD M9 Quant M10 Tile/Layout/Fuse   ← perf
  cache
```

**Key insight:** the forward pass (M3) is validated with *hardcoded token IDs* — the same ones
`gen_ref_logits.py` uses. So the tokenizer (M4) is a fully **parallel track**. You can get a
numerically-correct forward pass with no working tokenizer at all, and you can build the tokenizer
with no working model. They only meet at M5. This decoupling is the single biggest reason this
plan converges fast: the two hardest-to-debug subsystems never confound each other.

### Golden-first discipline

The repo already has a Python venv at `python/.venv` and two oracles. This plan adds two more
(per-kernel goldens and per-stage activation goldens). **Generate the golden, then match it.**
Never advance a milestone whose golden is red — a bug carried forward costs 10× to find later
because it hides inside a stack of plausible-but-wrong outputs.

A note on the **golden file format**: the existing scripts dump `.npy`. NumPy's `.npy` format has
a small ASCII header you'd have to parse in Zig. Recommended: for the *new* goldens, dump **raw
little-endian f32** plus a tiny `.shape.json` sidecar (`{"shape":[4,768]}`). Zig then reads the
shape, `@memcpy`/`@bitCast`es the bytes, done — zero parsing dependency. (Keep `ref_logits.npy`
as-is and write a ~40-line `.npy` reader, or re-dump it raw; either is fine. Pick one convention
and use it everywhere.)

---

# M0 — Scaffolding & golden harness

### 1. Overview
Get the validation loop runnable *before* writing any engine code: download the model, lay out the
`src/` modules, wire `zig build test`, and extend the Python oracles. End state: `zig build test`
runs (with empty/stub tests) and all four Python golden generators produce files.

### 2. Why this is needed
You can't validate what you can't compare against. Inference debugging is *entirely* a
"my number vs the reference number" loop. M0 builds the rig that makes every later milestone
falsifiable. Skipping it means you write 500 lines, get garbage tokens, and have no idea which of
ten suspects is wrong.

### 3. Pseudocode (setup actions, not a data transform)
```
# Model files → repo (gitignored; they're ~500MB)
download model.safetensors, config.json, tokenizer.json, vocab.json, merges.txt
    from https://huggingface.co/openai-community/gpt2/resolve/main/<file>

# Module skeleton (guide §6)
src/
  main.zig         # CLI/REPL + generation loop        (M5)
  safetensors.zig  # mmap + header parse               (M1)
  config.zig       # parse config.json                 (M1)
  tokenizer.zig    # BPE encode/decode                 (M4)
  model.zig        # weight structs + forward pass     (M3)
  math.zig         # matmul/layernorm/softmax/gelu      (M2)  ← you own this
  testdata.zig     # shared golden-reader helper (raw-f32 + shape sidecar)
  tests/
    kernel_test.zig
    forward_test.zig
    tokenizer_test.zig

# Golden generators (python/)
gen_tokenizer_golden.py      # EXISTS — tiktoken exact ids
gen_ref_logits.py            # EXISTS — final HF logits → ref_logits.npy
gen_kernel_goldens.py        # NEW (see M2)
gen_activation_goldens.py    # NEW (see M3)
```

### 4. Unit tests / golden comparisons
- `zig build test` exits 0 with stub tests.
- Each Python script runs clean and writes its output files.
- Write `testdata.zig`'s reader and unit-test it on a 2×3 array you dump from Python and
  hand-verify. (Prove the *harness* before trusting it on real goldens.)

### 5. Integration tests
- `git status` shows the model files are gitignored (add `*.safetensors`, `*.json` model files,
  `merges.txt`, `vocab.json`, `*.npy`, `*.bin` golden dumps to `.gitignore`). Don't commit 500MB.

### 6. Tips / footguns
- **Pin your Zig version.** Repo is on **0.16.0**. Zig is pre-1.0 and breaks `std` APIs between
  releases; the guide assumes 0.13+, so some `std` calls in examples may have moved. Pin and don't
  upgrade mid-build.
- mmap the safetensors; don't read it into a heap buffer (it's hundreds of MB, and per CLAUDE.md
  heap allocation is a code smell — mmap gives you the bytes for free, OS-paged).
- Decide raw-f32-vs-`.npy` *now* and write `testdata.zig` once. Re-deciding later means rewriting
  every test.

---

# M1 — Safetensors loader + config

### 1. Overview
Parse `config.json` into a `Config` struct, and turn `model.safetensors` into a map from tensor
name → typed slice over the mmap'd bytes. No math yet — this is the data layer everything stands on.

### 2. Why this is needed
The model is ~124M numbers organized into ~150 named tensors. Before you can multiply by a weight
matrix you need to (a) find it by name, (b) know its shape, (c) point at the right bytes. The one
conceptual landmine lives here, so internalize it now:

**Row-major + the Conv1D transpose.** A weight matrix `W` of logical shape `[rows, cols]` is stored
as `rows` contiguous runs of `cols` floats. A normal PyTorch `nn.Linear(in, out)` stores its weight
as `[out, in]`. **But GPT-2 doesn't use `nn.Linear` — it uses `Conv1D`, which stores `[in, out]`**
(transposed!). So `c_attn.weight` is `[768, 2304]` meaning `[in=768, out=2304]`. This is *already*
the orientation a `y = x @ W` matmul wants (`x` is `[S, 768]`, `W` is `[768, 2304]`, out is
`[S, 2304]`) — so for GPT-2 you actually do **not** transpose; you index it as-is. The bug the guide
warns about is assuming `nn.Linear` `[out,in]` layout and transposing when you shouldn't (or vice
versa). Decide your convention here, write it down, and every matmul downstream is consistent.

(The genuinely-transposed case is the **tied output projection** in M3: logits reuse `wte.weight`
`[50257, 768]` and you need `x @ wteᵀ`. That one transpose is real.)

### 3. Pseudocode
```
# safetensors layout: [8-byte LE u64 header_len][header_len bytes UTF-8 JSON][raw tensor data]
fn load(path):
    bytes   = mmap(path)
    N       = read_u64_le(bytes[0..8])
    header  = json_parse(bytes[8 .. 8+N])      # name -> {dtype, shape, data_offsets:[begin,end]}
    base    = 8 + N
    for (name, meta) in header:
        if name == "__metadata__": continue
        slice = bytes[base + meta.data_offsets[0] .. base + meta.data_offsets[1]]
        store name -> Tensor{ dtype: meta.dtype, shape: meta.shape, data: slice }
    return TensorMap

fn loadConfig(path):
    j = json_parse(read(path))
    return Config{ n_layer:12, n_head:12, n_embd:768, n_ctx:1024,
                   vocab:50257, ln_eps:1e-5 }   # read from j; values shown are gpt2-small
```
Input: file paths. Output: `Config` + `name → Tensor{dtype, shape, []const f32}`.

### 4. Unit tests / golden comparisons
- New tiny script (or extend a generator) dumps a **manifest**: `[(name, dtype, shape), ...]` for
  every tensor, from Python `safetensors`/numpy. Assert your Zig loader produces the identical set
  (same count — ~149 tensors, same shapes, same dtype).
- **Spot-check values:** dump the first 5 floats of `wte.weight` and of `h.0.attn.c_attn.bias` from
  numpy; assert bit-or-near-equal in Zig. This catches off-by-`base` slicing and endianness.

### 5. Integration tests
- Load all tensors, sum every weight's absolute values, print it. Run twice — deterministic. A NaN
  or wildly different magnitude means a slicing/dtype bug. (Cheap canary before M3.)

### 6. Tips / footguns
- **dtype:** HF gpt2 is usually **F32**, but some mirrors ship F16/BF16 — *check `meta.dtype`*,
  don't assume. BF16→F32 widening is just `@as(f32, @bitCast(@as(u32, bf16_bits) << 16))`.
- Represent tensor handles per CLAUDE.md style: an `enum(u32){_}` index into a flat array of
  `Tensor` structs beats passing raw slices around, and lets you name tensors by handle.
- `__metadata__` is a real key in the header JSON and is **not** a tensor — skip it.
- Validate `data_offsets` are within bounds and contiguous; a corrupt download fails loudly here,
  not as silent garbage in layer 7.

---

# M2 — Math kernels (`math.zig`)

### 1. Overview
The four hand-written numerical primitives every layer is built from: `matmul` (with bias),
`layernorm`, `softmax`, `gelu_tanh`. **This is the file you own** — per the guide, the kernels and
the bug-hunt are the actual experiment. Get each one matching a numpy golden in isolation before
they're ever composed in M3.

### 2. Why this is needed (the math, intuitively)
- **matmul (`y = x @ W + b`)** — a learned *linear mixing*. Each output feature is a weighted sum
  of all input features; the weights `W` are what training learned. This is ~99% of the FLOPs in
  the model. Everything else just reshapes or gently bends the output of a matmul.
- **LayerNorm** — *re-centering and re-scaling each token's vector* so its values have mean 0 and
  variance 1, then applying a learned per-feature scale (`weight`/γ) and shift (`bias`/β). Why:
  after many residual additions the vector's magnitude drifts; without renormalization later layers
  see wildly different scales and training/inference destabilizes. Think "AGC (automatic gain
  control) on the signal before each block." It is **not** RMSNorm — GPT-2 subtracts the mean *and*
  adds β. Variance is **biased** (divide by N, not N−1).
- **Softmax** — turns a vector of arbitrary real "scores" into a probability distribution
  (positive, sums to 1) by exponentiating then normalizing. In attention it converts
  similarity-scores into "how much should I attend to each previous token." The max-subtraction is
  pure numerical hygiene: `exp(large)` overflows to `inf`; subtracting the row max makes the largest
  exponent `exp(0)=1`, and the result is mathematically identical.
- **GELU (tanh approx, `gelu_new`)** — a smooth gate: it passes large positive values through,
  squashes negatives toward 0, and does so *smoothly* (unlike ReLU's hard kink). It's the MLP's
  nonlinearity — without a nonlinearity, stacked matmuls collapse into a single matmul and the
  network can't learn anything beyond a linear map. **Use the tanh approximation, not exact erf** —
  GPT-2 was trained with `gelu_new`, and erf-GELU produces tiny logit drift that fails tight parity.

### 3. Pseudocode (exact formulas — match or fail validation)
```
matmul(x:[M,K], W:[K,N], b:[N]) -> y:[M,N]:
    for m in 0..M, n in 0..N:
        acc = b[n]
        for k in 0..K: acc += x[m,k] * W[k,n]     # W indexed [in,out], see M1
        y[m,n] = acc

layernorm(row:[D], gamma:[D], beta:[D], eps=1e-5) -> [D]:
    mu  = mean(row)
    var = mean((row - mu)^2)                       # BIASED: / D, not /(D-1)
    return (row - mu) / sqrt(var + eps) * gamma + beta

softmax(row) -> [.]:
    m = max(row)
    e = exp(row - m)
    return e / sum(e)

gelu_tanh(x) -> .:
    return 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
```

### 4. Unit tests / golden comparisons
Add **`python/gen_kernel_goldens.py`**: with a fixed seed, generate small random inputs and their
numpy reference outputs, dumped as raw-f32 + shape sidecar:
- `layernorm`: input `[4, 768]`, γ/β random → output `[4, 768]`. Use the *biased* variance
  (`np.var(x, axis=-1)` defaults to biased — good).
- `softmax`: input `[4, 16]` → row-wise softmax.
- `gelu_tanh`: input `[256]` → `0.5*x*(1+tanh(...))` (write the tanh formula explicitly in numpy;
  do **not** use `scipy.special.erf`).
- `matmul`: `x[3,5] @ W[5,7] + b[7]` → `[3,7]`.

Zig unit tests assert max-abs-diff ≤ `1e-5` per kernel. Also add a couple of **hand-computed** cases
(e.g. `softmax([0,0])==[0.5,0.5]`, `layernorm` of a constant row → all β) — these catch sign/axis
bugs a random golden might mask.

### 5. Integration tests
- None yet — kernels integrate in M3. But do a quick `gelu_tanh` vs erf-`gelu` diff print so you
  *see* the drift the guide warns about and remember why you chose tanh.

### 6. Tips / footguns (the §5 gotcha table, kernel-resident ones)
- **GELU flavor:** tanh, not erf. #1 "compiles, runs, fails tight parity" bug.
- **LayerNorm ≠ RMSNorm:** subtract mean *and* add β; biased variance; eps `1e-5` *inside* the sqrt.
- **Softmax stability:** subtract row max; operate row-wise (last axis), not over the whole matrix.
- **matmul indexing:** `W[k,n]` (i.e. `[in,out]`) for GPT-2 Conv1D weights — see M1.
- Keep kernels allocation-free: take output slices as parameters (caller owns scratch). Per
  CLAUDE.md, statically size scratch to known upper bounds (`n_embd`, `n_ctx`).
- Write kernels straightforwardly first (naive `ijk`). **Do not** SIMD/thread yet — that's M7/M8,
  and you want a known-correct baseline to diff the fast versions against.

---

# M3 — Forward pass (`model.zig`)

### 1. Overview
Compose M1's weights and M2's kernels into the full GPT-2 forward pass: token IDs → logits.
Validated with **hardcoded IDs** `[15496, 11, 314, 1101]` (the same ones `gen_ref_logits.py` uses),
so this milestone needs M1 + M2 but **not** the tokenizer. This is the numerical heart of the
project and where most bugs live — which is exactly why we bisect it with per-stage goldens.

### 2. Why this is needed (the math, intuitively)
The forward pass maps a sequence of tokens to, for each position, a score over all 50257 possible
next tokens. The mechanism:

- **Embedding (`wte[token] + wpe[pos]`):** look up a 768-vector for the token's *identity* and add a
  768-vector for its *position*. GPT-2 learns position as a plain lookup table (no RoPE) — position 0
  has its own vector, position 1 its own, etc. The sum is the token's starting point in the
  "residual stream."
- **The residual stream:** think of `x` (`[S, 768]`) as a **shared scratchpad**, one 768-dim vector
  per token. Every block *reads* it, computes something, and *adds* its result back (`x = x + ...`).
  Nothing overwrites; everyone contributes. This is why gradients flow and why you can stack 12
  blocks.
- **Attention = content-addressed lookup.** Each token emits three projections of its vector:
  a **Query** ("what am I looking for?"), a **Key** ("what do I offer?"), and a **Value** ("what I'll
  hand over if you pick me"). Token `i`'s query is dotted with every token's key → a similarity
  score per token. Softmax those into weights, then take the weighted sum of Values. So each token
  pulls in a *blend of information from the tokens it finds relevant.* "Multi-head" = do this 12
  times in parallel on 64-dim slices, each head free to specialize (one tracks syntax, one tracks
  the subject noun, etc.), then concatenate.
  - **Causal mask:** at generation time token `i` must not see the future, so scores for `j > i` are
    set to `-inf` (→ 0 after softmax). Token `i` *can* see itself (`j == i`).
  - **Scale by `1/sqrt(head_dim) = 1/8`:** dot products of 64-dim vectors grow ~`sqrt(64)` in
    magnitude; without down-scaling, softmax saturates (one weight ≈ 1, rest ≈ 0) and gradients
    vanish. Dividing by `sqrt(head_dim)` keeps the score variance ~1 so softmax stays soft. Note
    it's `sqrt(head_dim)=8`, **not** `head_dim=64` — a classic drift bug.
- **MLP (`c_fc` up to 3072, GELU, `c_proj` back to 768`):** a per-token feature transform. Attention
  *moves information between tokens*; the MLP *processes each token's vector independently*, using a
  4× wider hidden layer and a nonlinearity to compute richer features. The two alternate: mix across
  tokens, then think per token, repeat.
- **Final norm + tied logits:** one last LayerNorm, then project each token's 768-vector against the
  *transpose of the token embedding* (`x @ wteᵀ`) to score all 50257 tokens. GPT-2 **ties** input and
  output embeddings — there is no separate `lm_head`. Intuition: the same vector that *represents* a
  token is reused to *recognize* it.

### 3. Pseudocode (prev stage `token_ids:[S]` → this stage `logits:[S,50257]`)
```
forward(ids:[S]) -> logits:[S, 50257]:
    # --- embedding ---
    x = [S,768]
    for s in 0..S: x[s] = wte[ids[s]] + wpe[s]

    for layer in 0..12:
        # --- attention block ---
        a   = layernorm_rows(x, ln_1.w, ln_1.b)                 # [S,768]
        qkv = matmul(a, c_attn.w, c_attn.b)                     # [S,2304]
        q,k,v = qkv[:, 0:768], qkv[:, 768:1536], qkv[:, 1536:2304]
        # view each as [n_head=12, S, head_dim=64]
        attn = [S,768]
        for h in 0..12:
            scores = (q_h @ k_hᵀ) * (1/8)                       # [S,S]
            for i in 0..S, j in 0..S: if j>i: scores[i,j] = -inf  # causal mask
            p = softmax_rows(scores)                            # [S,S]
            out_h = p @ v_h                                     # [S,64]
            write out_h into attn[:, h*64 : (h+1)*64]
        attn = matmul(attn, c_proj.w, c_proj.b)                 # [S,768]
        x = x + attn                                            # residual

        # --- mlp block ---
        m = layernorm_rows(x, ln_2.w, ln_2.b)                   # [S,768]
        hdn = matmul(m, c_fc.w, c_fc.b)                         # [S,3072]
        hdn = gelu_tanh(hdn)
        m = matmul(hdn, c_proj_mlp.w, c_proj_mlp.b)             # [S,768]
        x = x + m                                               # residual

    # --- final norm + tied logits ---
    x = layernorm_rows(x, ln_f.w, ln_f.b)                       # [S,768]
    logits = matmul_transposed(x, wte)                          # x @ wteᵀ -> [S,50257]
    return logits
```

### 4. Unit tests / golden comparisons
Add **`python/gen_activation_goldens.py`**: run HF `GPT2LMHeadModel` on the fixed prompt with
forward hooks, dumping intermediate tensors as raw-f32 + shape sidecars:
- `act_embed.bin` — `x` right after `wte+wpe` (`[4,768]`)
- `act_l0_ln1.bin` — after layer 0's `ln_1`
- `act_l0_attn.bin` — layer 0 attention output (post `c_proj`, *pre* residual add)
- `act_l0_resid1.bin` — `x` after the attention residual
- `act_l0_mlp.bin` — layer 0 MLP output (pre residual)
- `act_l0_out.bin` — `x` after layer 0 fully
- `act_l5_out.bin` — a deeper layer, to catch errors that only accumulate
- `act_lnf.bin` — after `ln_f`
- final: reuse `ref_logits.npy`

Then **bisect**: compare your Zig activation at each tap, in order. The *first* tap that diverges
localizes the bug to one sub-operation. Target max-abs-diff ≤ ~`1e-3` (F32; tighter is better, but
small accumulation differences across 12 layers are expected). Without these taps you only know
"logits wrong" and face ten suspects; with them you know "layer 0 attention output wrong" and face
one.

> HF hook note: HF's `GPT2Attention` returns the post-`c_proj` output; the `mlp` submodule's output
> is pre-residual. Register `forward_hook`s on `transformer.wte`/`wpe` (or capture their sum),
> `h[0].ln_1`, `h[0].attn`, `h[0].mlp`, `h[5]`, and `transformer.ln_f`. Verify each captured shape
> is `[1,4,768]` before trusting it.

### 5. Integration tests
- Full `forward([15496,11,314,1101])` → assert final logits match `ref_logits.npy` ≤ `1e-3`.
- `argmax(logits[last])` should equal HF's greedy next token for that prompt (print both). This is
  the first time you see the model "say" something — a strong correctness signal even before M5.

### 6. Tips / footguns
- **Fused QKV split:** `c_attn` outputs 2304 = 3×768. Q is `[:, 0:768]`, K `[:, 768:1536]`,
  V `[:, 1536:2304]`. Getting the order or boundaries wrong silently corrupts attention.
- **Head reshape layout:** within each token's 768-vector, head `h` owns dims `[h*64:(h+1)*64]`. Be
  consistent between how you slice Q/K/V and where you write `out_h` back into `attn`.
- **Attention scale `1/8`, not `1/64`.** Re-verify against `act_l0_attn.bin`.
- **Causal mask `j > i` only** (strictly future). `j == i` must stay — a token attends to itself.
  Off-by-one here "looks fine" but leaks future info and fails the deep-layer golden.
- **Tied embedding:** there is no `lm_head` tensor; reuse `wte.weight` transposed. If you're hunting
  for a missing output-projection weight, this is why — stop hunting.
- **Two different `c_proj`s:** attention has one (`[768,768]`) and MLP has another (`[3072,768]`).
  Don't cross them.
- Per CLAUDE.md: preallocate all scratch (`x`, `qkv`, `scores`, `hdn`) once at known upper bounds
  (`n_ctx=1024`, `n_embd=768`, `4*n_embd=3072`) — no per-layer or per-token allocation.
- Structure the layer as the "multiple passes" style: compute the full attention block (its own
  source of truth) then the full MLP block, rather than interleaving — easier to tap and reason about.

---

# M4 — Tokenizer (`tokenizer.zig`) — *parallel track*

### 1. Overview
String ↔ token IDs. Byte-level BPE: split off special tokens, pre-tokenize with the GPT-2 regex,
merge byte-pairs by `merges.txt` rank, map bytes through GPT-2's byte↔unicode table. Validated
against the existing `gen_tokenizer_golden.py` table — **completely independent of the model**, so
build it whenever (in parallel with M1–M3). It's the most unit-testable thing in the project:
deterministic, integer output, exact oracle, lossless roundtrip.

### 2. Why this is needed (intuition)
The model speaks integers, not text. BPE is the translation layer.

- **BPE (Byte-Pair Encoding):** start from individual bytes; repeatedly merge the adjacent pair with
  the *highest-priority merge rule* (rank = line number in `merges.txt`, lower = merge earlier).
  Frequent sequences ("the", " of") collapse into single tokens; rare ones stay fragmented. It's a
  greedy compression learned from the training corpus. Vocab is 50257 = 256 base bytes + 50000
  learned merges + 1 special token.
- **Byte-level + the byte↔unicode remap:** GPT-2 operates on raw *bytes* so it can encode *any* text
  (no UNK). But the BPE tables are stored as printable text, and control bytes (space, newline, tab)
  aren't printable. So GPT-2 defines a reversible map from all 256 bytes to safe printable Unicode
  codepoints — famously space (`0x20`) → `Ġ`, newline → `Ċ`. That `Ġ` you see prefixing words is
  just "a space lived here." You must implement this map *and its inverse* exactly.
- **Pre-tokenization regex:** before BPE, text is split into chunks (words, contractions, digit
  runs, whitespace) by a fixed regex. This stops merges from crossing word boundaries and is why
  `" hello"` (leading space attaches to the word) tokenizes differently from `"hello"`.

### 3. Pseudocode
```
encode(text) -> [ids]:
    parts = split_on_special_tokens(text)         # "<|endoftext|>" -> atomic id 50256
    ids = []
    for part in parts:
        if part is special: ids.append(special_id); continue
        for chunk in gpt2_regex_split(part):      # words/contractions/digits/spaces
            bytes  = utf8_bytes(chunk)
            symbols = [ byte2unicode[b] for b in bytes ]   # each byte -> printable codepoint
            while true:
                pair = lowest_rank_adjacent_pair(symbols, merges)   # min rank in merges.txt
                if none: break
                symbols = merge(symbols, pair)
            for sym in symbols: ids.append(vocab[sym])     # vocab.json: token-string -> id
    return ids

decode(ids) -> text:
    s = ""
    for id in ids: s += id2token[id]              # token-string (in remapped unicode)
    bytes = [ unicode2byte[c] for c in s ]        # inverse map
    return utf8_decode(bytes)
```

### 4. Unit tests / golden comparisons
- **Exact integer equality** against `gen_tokenizer_golden.py` (already covers the edge cases from
  guide §7): `"hello world"`, `" hello"`, `"don't"`, `"they're"`, `"2024"`, `"café"`, `"\n"`,
  `"\t"`, `"   "`, `"aaaaaaaa"`, `""`→`[]`, `"<|endoftext|>"`→`[50256]`. Slice-equality, no fuzz.
- **Fuzzed roundtrip:** for random strings (include multibyte/emoji/CJK and a few invalid-UTF-8
  byte sequences), assert `decode(encode(s)) == s`.

### 5. Integration tests
- In M5, the tokenizer feeds the model. As an early check, `encode("Hello, I am")` should produce
  `[15496, 11, 314, 1101]` — the exact hardcoded IDs M3 validates against. That single assertion
  proves the two parallel tracks agree before you wire the loop.

### 6. Tips / footguns
- **Roundtrip is directional:** `decode(encode(s)) == s` *always* holds; `encode(decode(ids)) == ids`
  does **not** for arbitrary id sequences. Test only the first direction.
- **Leading space matters:** `" hello"` ≠ `"hello"`. The regex attaches the space to the next word.
  Test both.
- **Special-token atomicity:** `"<|endoftext|>"` must become `[50256]` as one unit, never
  byte-split. This is the bug most likely to silently corrupt things later.
- **Byte↔unicode map is the #1 BPE bug.** Implement it from the canonical 256-entry table and
  unit-test the map alone (`map[0x20] == 'Ġ'`, inverse round-trips all 256).
- **Pick one oracle and match it exactly.** tiktoken `gpt2`, HF `tokenizer.json`, and llama.cpp BPE
  diverge on edge cases (special tokens especially). The repo's golden uses **tiktoken** — match
  tiktoken, since that removes a confound with whatever the logit reference used.
- The GPT-2 pre-tokenization regex is gnarly (Unicode categories, contraction special-cases). A
  hand-written ~100-line BPE validated against the golden is often less pain than vendoring a
  Zig-version-drifting library — but if you vendor, still run it against the golden table.

---

# M5 — Sampling + generation loop + detokenize (end-to-end)

### 1. Overview
Join the tracks: tokenize a prompt, run M3's forward repeatedly, greedily pick the next token,
detokenize, print. End state: `zig build run -- "Hello, I am"` streams text that matches the
reference token-for-token. This is the guide §10 "done by Sunday" bar.

### 2. Why this is needed (intuition)
The forward pass scores *all possible next tokens* at every position. Generation is: take the
scores at the *last* position, pick a token, append it, and run again — autoregression. **Greedy
sampling** = always take the argmax (highest-scoring token). It's deterministic, which is the whole
point right now: a deterministic decode means "do my generated tokens exactly match the reference's"
is a clean pass/fail, not a fuzzy distribution comparison. Temperature/top-k/top-p add randomness
and are deferred until correctness is locked.

### 3. Pseudocode
```
generate(prompt, max_new) -> text:
    ids = tokenize(prompt)                  # M4
    out = ""
    for step in 0..max_new:
        logits = forward(ids)               # M3 — recomputes WHOLE sequence each step (O(S^2))
        next   = argmax(logits[ids.len - 1])# greedy: last position only
        if next == 50256: break             # <|endoftext|>
        ids.append(next)
        out += decode([next])               # stream one token
    return out
```
Input: prompt string. Output: generated text (and the token stream, for the test).

### 4. Unit tests / golden comparisons
- `argmax` helper: trivial unit test (ties → lowest index, to match numpy/torch convention).
- Greedy next-token after the fixed prompt equals HF's greedy choice (already provable from
  `ref_logits.npy` argmax — assert your loop reproduces it).

### 5. Integration tests
- **Capstone:** generate N tokens greedily from `"Hello, I am"`. Produce the reference token
  sequence by running HF greedy in Python for the same N (add a tiny `gen_greedy_golden.py` or
  extend `gen_ref_logits.py`), and assert **token-for-token equality**. This single test proves the
  entire correctness stack (loader + kernels + forward + tokenizer + loop) at once.
- Print tokens/sec — your perf **baseline** for M6+.

### 6. Tips / footguns
- **EOS:** stop on `50256`. Also cap at `n_ctx=1024` — `wpe` only has 1024 positions; indexing past
  it is an out-of-bounds bug, not a graceful truncation.
- **Stream decoding caveat:** decoding one token at a time can split a multibyte UTF-8 character
  across token boundaries (a token may be a partial byte sequence). For correctness of the *test*,
  compare token IDs, not printed strings. For pretty printing, buffer bytes and only flush complete
  UTF-8 sequences.
- Keep `ids` in a preallocated `[1024]u32` with a length, not a growing heap list (CLAUDE.md).
- Don't add temperature/top-k yet. Every bit of nondeterminism you add now is a bit of correctness
  signal you lose.

---

# Perf milestones (M6–M10)

First, the **mental model** that makes the ranking make sense. Generation has two phases with
*different bottlenecks*:

- **Prefill** (process the whole prompt at once): compute-bound **GEMM** (matrix×matrix). Wins from
  blocking, SIMD, threading.
- **Decode** (generate one token at a time): memory-bandwidth-bound **GEMV** (matrix×vector). Each
  step streams the *entire weight set* once to produce one token. Wins most from *moving fewer
  bytes* (quantization) and threading.

For realistic generation lengths **decode dominates wall-clock and decode is bandwidth-bound** —
that ordering drives the list. **Record tokens/sec before and after every change, form a bandwidth
hypothesis, apply one change, watch the number move (or stubbornly not).** That loop *is* the job.

Discipline: **keep the M3/M5 F32 logit-diff and greedy-match tests green through M6–M8.** They must
not change outputs. M9 (quantization) is the one exception — it changes outputs, so it gets its own
accuracy pass.

### M6 — KV cache (do this first; the only complexity-class win)
- **What:** cache each layer's K and V (`[layer, head, seq, head_dim]`). Each step computes Q for
  *only the new token*, attends against cached K/V, and appends the new K/V. Per-step work drops from
  "whole growing sequence" to "one token."
- **Why (n00b):** the naive loop recomputes K and V for every previous token at every step — pure
  redundancy, since past tokens' K/V never change (causal mask means they don't depend on the
  future). Caching turns total work from O(N²) to O(N). Often 10–100× for multi-token generation.
  It also *changes the bottleneck* to GEMV, which sets up everything below.
- **Footguns:** prefill (fill cache for the whole prompt at once) vs decode (one token) are now two
  code paths — share the kernels, not the loop. Preallocate the cache as one arena
  (`[12][12][1024][64]f32`, ~75MB) — never grow it. The causal mask becomes implicit (new token
  attends to all cached positions ≤ itself), so don't *also* apply a square mask.
- **Measure:** tokens/sec at N=10, 50, 200 — the speedup should grow with N (that's the O(N²)→O(N)
  signature). Re-run the greedy-match test; output must be identical.

### M7 — Multithread the matmuls (biggest, easiest constant-factor win)
- **What:** split the output dimension across a `std.Thread.Pool`; each thread computes a row/column
  band. Near-linear with physical cores → 4–16×.
- **Why (n00b):** every output element of a matmul is independent — embarrassingly parallel. Lowest
  effort-to-payoff on the list.
- **Footguns:** parallelize the **final logits projection** hardest — `[768]×[768,50257]` is the
  largest single GEMV per decode step (~38M MACs), bigger than any one attention/MLP matmul. Watch
  false sharing (have threads write disjoint, cache-line-aligned output ranges). Create the pool
  once, reuse it; don't spawn per token.
- **Measure:** tokens/sec vs thread count (1,2,4,8…) — expect near-linear until you hit physical
  cores, then flatten (you've become bandwidth-bound — exactly the transition that motivates M9).

### M8 — SIMD the inner loop (`@Vector`)
- **What:** vectorize the dot-product inner loop with `@Vector(16, f32)` (AVX-512: 16 f32/lane) +
  horizontal reduce. Typically 4–8× on the hot loop. Pairs with M7 (thread across rows, SIMD within).
- **Why (n00b):** one CPU instruction multiplies 16 pairs at once instead of 1. The dot product —
  the kernel of every matmul — is the ideal target.
- **Footguns:** handle the tail (dims not a multiple of 16). Let Zig's backend lower `@Vector`; only
  drop to intrinsics if it won't. Keep the naive kernel around as the correctness oracle to diff
  against — SIMD reassociation changes float rounding slightly (should stay within `1e-3`).
- **Measure:** tokens/sec before/after; profile to confirm the hot loop actually vectorized
  (disassemble or check it's not falling back to scalar).

### M9 — Quantization Q8 → Q4 (the most inference-representative optimization)
- **What:** store weights as int8 (or int4) with per-block f32 scales; dequantize in the hot loop
  (or use integer dot products with the scales). Halving/quartering bytes-moved-per-weight nearly
  directly multiplies *decode* throughput (it's bandwidth-bound) and shrinks the footprint.
- **Why (n00b):** decode's wall-clock is dominated by *streaming weights from RAM*, not by the
  arithmetic. Fewer bytes per weight → proportionally faster decode. This is the optimization that
  most resembles frontier-lab inference work.
- **Footguns:** **this changes outputs** — it gets its own correctness pass. Don't reuse the `1e-3`
  F32 threshold; instead check that **greedy token output still matches** (or top-1 agreement /
  perplexity stays within tolerance) against the F32 reference. Choose block size (e.g. 32 or 64
  weights per scale) — smaller blocks = better accuracy, more scale overhead. Pack each block's
  scale adjacent to its quants (sets up M10's layout).
- **Measure:** tokens/sec *and* an accuracy metric, plotted together — the throughput-vs-accuracy
  trade is the whole exercise. If tuning that knob is fun, that's strong signal about the day job.

### M10 — Tiling / layout / weight pre-packing / fusion
- **What:** cache-block (tile) the prefill GEMM for L1/L2; pre-transpose/pack weights at load into
  the exact layout the inner loop reads contiguously (and quant blocks with adjacent scales); fuse
  bias-add + activation into the matmul epilogue; reuse scratch and keep the KV cache in one arena
  (kill per-token allocations).
- **Why (n00b):** naive `ijk` matmul thrashes cache (re-reads data that fell out of L1). Tiling
  keeps a working block resident. Layout/packing is *foundational* — it's what lets M8 (SIMD) and M9
  (quant) actually hit their ceilings — even though on its own it's a small headline number.
- **Footguns:** highest value when prompts are long (prefill-heavy); lower when decode dominates, so
  weigh it against your actual workload. Fusion removes intermediate buffers — make sure the fused
  path still matches the unfused one numerically before deleting the slow path.
- **Measure:** prefill tokens/sec on a long prompt specifically (this is where tiling shows up);
  decode tokens/sec for the fusion/alloc wins.

---

# Appendix

### "Done" checklist (guide §10)
- [ ] Greedy generation **matches the reference token-for-token** on a fixed prompt (M5 capstone).
- [ ] Tokenizer: green golden table + fuzzed `decode(encode(s))==s` (M4).
- [ ] Forward pass: per-stage activations + final logits within `1e-3` of HF (M3).
- [ ] A tokens/sec number moved by **at least** KV cache + threading + SIMD, profiler open, a
      hypothesis behind each change (M6–M8).

### Golden generators (python/)
| Script | Status | Produces |
|---|---|---|
| `gen_tokenizer_golden.py` | exists | `(string, ids)` pairs from tiktoken (M4) |
| `gen_ref_logits.py` | exists | `ref_logits.npy` — final HF logits, fixed prompt (M3) |
| `gen_kernel_goldens.py` | **new (M2)** | per-kernel input/output raw-f32 (layernorm, softmax, gelu_tanh, matmul) |
| `gen_activation_goldens.py` | **new (M3)** | per-stage activation taps for bisecting the forward pass |
| `gen_greedy_golden.py` | **new (M5)** | reference greedy token sequence for the capstone test |

Run any with: `python/.venv/bin/python python/<script>.py`

### Cross-reference
For the full weight-tensor table, the silent-failure gotcha table, and the reference link index,
see [`gpt2-inference-engine-guide.md`](gpt2-inference-engine-guide.md). This plan is the *order and
the why*; the guide is the *spec and the warnings*.
