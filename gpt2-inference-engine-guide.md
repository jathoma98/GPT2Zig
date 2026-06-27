# Building a Naive GPT-2 Inference Engine (Zig)

A weekend-scoped guide to writing a from-scratch CPU forward pass for GPT-2 124M, validating it against a reference, then optimizing it until you can watch tokens/sec climb. The point of the build is the *texture* of inference work: numerical-correctness debugging, hot-loop profiling, and memory-bandwidth thinking.

---

## 0. Scope, and one honesty caveat

**What this builds:** a single-file-ish Zig program that loads GPT-2 124M, tokenizes a prompt, runs a correct forward pass, greedy-decodes tokens, and prints text — then a ranked set of optimizations to make it fast.

**The caveat that matters:** GPT-2 124M is a *base completion* model, not an instruct model. It has no chat template and no notion of turns; it just continues text. So "chat engine" here means a completion REPL, not an assistant. That's fine — the thing you're actually testing (correctness loop + perf loop) is architecture- and instruct-independent. If you later want something that *feels* like chat, swap in an instruct-tuned small model (SmolLM2-135M-Instruct, Qwen3-0.6B), which costs you RoPE + GQA + SwiGLU + RMSNorm on top of this.

**Why GPT-2 for the experiment:** it has neither RoPE nor GQA — the two highest-bug-surface components of a modern decode. Position is a learned lookup; attention is full multi-head. That strips out the insidious "looks fine, fails validation" bugs and gets you to a correct pass fastest, so you spend your hours on the perf work that actually tests whether you enjoy this.

---

## 1. Prerequisites & downloads

### Toolchain
- **Zig** — a recent release (0.13+ or current master). Note Zig is pre-1.0 and breaks `std` APIs between versions; pin one version for the weekend.
- **Python** (for the reference oracle only): `pip install torch transformers tiktoken numpy safetensors`

### Model files
Pull from the canonical HF repo: **https://huggingface.co/openai-community/gpt2**

Minimum set:
| File | Purpose |
|---|---|
| `model.safetensors` | the weights |
| `config.json` | hyperparameters |
| `tokenizer.json` | BPE vocab + merges + pre-tokenization regex (single-file) |
| `vocab.json` + `merges.txt` | same data, split form (use if your tokenizer wants it) |

Direct download (raw):
- `https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors`
- `https://huggingface.co/openai-community/gpt2/resolve/main/config.json`
- `https://huggingface.co/openai-community/gpt2/resolve/main/tokenizer.json`

### Reference implementations (your oracles)
- **nanoGPT** — https://github.com/karpathy/nanoGPT — ~300-line clean GPT-2; diff your logits against it.
- **minGPT** — https://github.com/karpathy/minGPT — more annotated, same architecture.
- **tiktoken** — https://github.com/openai/tiktoken — ground-truth GPT-2 tokenizer (`get_encoding("gpt2")`).
- **safetensors format spec** — https://github.com/huggingface/safetensors

### Optional Zig-native tokenizer (if you don't hand-write BPE)
- `https://github.com/jaco-bro/tokenizer` (more complete, `zig build run`)
- `https://github.com/alvarobartt/tokeni.zig` (minimal, loads gpt2 `tokenizer.json`)

Caveat: both were last touched ~early 2025; expect Zig-version drift. For GPT-2 specifically, an AI-written ~100-line BPE encoder validated against tiktoken is often *less* integration pain than vendoring. See §6.

---

## 2. Model architecture reference — GPT-2 124M ("small")

```
n_layer   = 12
n_head    = 12
n_embd    = 768
head_dim  = 64          (768 / 12)
vocab     = 50257
n_ctx     = 1024        (max sequence length)
ln_eps    = 1e-5
```

### Weight tensors (HF naming)

| Tensor | Shape | Notes |
|---|---|---|
| `wte.weight` | `[50257, 768]` | token embedding; **also** the output projection (tied) |
| `wpe.weight` | `[1024, 768]` | learned position embedding |
| `h.{i}.ln_1.weight/bias` | `[768]` | pre-attention LayerNorm |
| `h.{i}.attn.c_attn.weight` | `[768, 2304]` | **Conv1D** → stored `[in, out]`; fused QKV (3×768) |
| `h.{i}.attn.c_attn.bias` | `[2304]` | |
| `h.{i}.attn.c_proj.weight` | `[768, 768]` | **Conv1D** |
| `h.{i}.attn.c_proj.bias` | `[768]` | |
| `h.{i}.ln_2.weight/bias` | `[768]` | pre-MLP LayerNorm |
| `h.{i}.mlp.c_fc.weight` | `[768, 3072]` | **Conv1D**; up-projection (4×) |
| `h.{i}.mlp.c_fc.bias` | `[3072]` | |
| `h.{i}.mlp.c_proj.weight` | `[3072, 768]` | **Conv1D**; down-projection |
| `h.{i}.mlp.c_proj.bias` | `[768]` | |
| `ln_f.weight/bias` | `[768]` | final LayerNorm before logits |

**The Conv1D transpose is the single most common GPT-2 load bug.** HF stores GPT-2's linear layers as `Conv1D`, which is `[in, out]` — transposed relative to the `[out, in]` of `nn.Linear`. If a matmul outputs garbage, this is almost always why. Transpose once at load (or index accordingly) and never think about it again.

---

## 3. The inference pipeline (bare minimum to get tokens)

### Pipeline overview

```
prompt string
   │
   ▼  [tokenizer]  byte-level BPE
token ids  (e.g. [15496, 11, 314, 1101])
   │
   ▼  [forward pass]  repeated each step
logits over vocab  [seq, 50257]
   │
   ▼  [sample]  greedy = argmax of last position
next token id
   │
   ├──► append to sequence, loop back into forward pass
   │
   ▼  [detokenizer]  ids → bytes → UTF-8
output text
```

### 3.1 Load weights (safetensors)

safetensors layout: first 8 bytes are a little-endian `u64` header length `N`; next `N` bytes are a UTF-8 JSON object mapping tensor name → `{dtype, shape, data_offsets:[begin,end]}`; everything after is raw, contiguous, row-major tensor data.

Loader = mmap the file, read `N`, `std.json` the header, hand back each tensor as a slice into the mapping at `8 + N + begin`.

GPT-2's weights are typically **F32** in the HF safetensors (some mirrors ship F16/BF16 — check `dtype`). If BF16: widening to F32 is just the high 16 bits — `@as(f32, @bitCast(@as(u32, bits) << 16))`.

### 3.2 Tokenize (byte-level BPE)

Three stages: (1) split off special tokens (`<|endoftext|>` = 50256), (2) pre-tokenize with the GPT-2 regex (splits on word boundaries, contractions, digit runs), (3) byte-level BPE-merge each chunk using `merges.txt` rank order. See §6 for build-vs-vendor and §7 for the byte↔unicode remap gotcha.

### 3.3 Forward pass (the core)

Given token ids `t[0..S]` and positions `0..S`:

**Embedding**
```
x[s] = wte[t[s]] + wpe[s]        →  x : [S, 768]
```

**For each layer i in 0..12:**

```
# --- Attention block ---
a = layernorm(x, ln_1.weight, ln_1.bias)          # [S, 768]
qkv = a @ c_attn.weight + c_attn.bias             # [S, 2304]   (Conv1D weight already [in,out])
q, k, v = split(qkv, 768)                          # each [S, 768]
reshape each to [n_head=12, S, head_dim=64]

for each head:
    scores = (q @ kᵀ) / sqrt(64)                  # [S, S],  scale = 1/8
    apply causal mask: scores[i][j] = -inf for j > i
    p = softmax(scores, axis=-1)                   # row-wise
    head_out = p @ v                               # [S, 64]

attn = concat heads → [S, 768]
attn = attn @ c_proj.weight + c_proj.bias          # [S, 768]
x = x + attn                                       # residual

# --- MLP block ---
m = layernorm(x, ln_2.weight, ln_2.bias)           # [S, 768]
h = m @ c_fc.weight + c_fc.bias                    # [S, 3072]
h = gelu_tanh(h)                                   # gelu_new
m = h @ c_proj.weight + c_proj.bias                # [S, 768]
x = x + m                                          # residual
```

**Final norm + logits (tied embedding)**
```
x = layernorm(x, ln_f.weight, ln_f.bias)           # [S, 768]
logits = x @ wte.weightᵀ                            # [S, 50257]   ← reuse token embedding, no separate lm_head
```

### Exact formulas (match these or fail validation)

**LayerNorm** (per row, biased variance — divide by N not N−1):
```
μ  = mean(x)
σ² = mean((x − μ)²)
y  = (x − μ) / sqrt(σ² + 1e-5) * weight + bias
```

**GELU — tanh approximation (`gelu_new`), NOT exact erf:**
```
gelu(x) = 0.5 * x * (1 + tanh( sqrt(2/π) * (x + 0.044715 * x³) ))
```
Using exact erf-GELU here produces small logit drift that fails tight reference parity. This is the one GPT-2 gotcha in the "looks fine, fails validation" class — match `gelu_new` deliberately.

**Softmax** (numerically stable, row-wise):
```
m = max(row)
e = exp(row − m)
softmax = e / sum(e)
```

### 3.4 Sample

Start with **greedy**: `next = argmax(logits[last_position])`. Deterministic, so it's diffable against the reference. Add temperature / top-k / top-p later once correctness is proven.

### 3.5 Detokenize

`ids → byte strings (via the byte↔unicode inverse map) → concatenate → interpret as UTF-8`.

### 3.6 Generation loop (naive)

```
ids = tokenize(prompt)
for step in 0..max_new_tokens:
    logits = forward(ids)            # recomputes the WHOLE sequence every step
    next = argmax(logits[ids.len-1])
    if next == 50256: break          # <|endoftext|>
    ids.append(next)
    print(detokenize([next]))
```

This is correct but O(S²) in total work — it reprocesses the entire prefix every step. That's deliberate: get it correct first, then §8 #1 (KV cache) fixes exactly this.

---

## 4. Validation strategy (do this before any optimization)

Two independent oracles, two precision regimes:

**Tokenizer — exact integer equality.** Generate `(string, ids)` pairs from tiktoken and assert slice equality:
```python
import tiktoken, json
enc = tiktoken.get_encoding("gpt2")
cases = ["hello world", " hello", "don't", "2024", "café", "\n\t", "<|endoftext|>"]
print(json.dumps([{"in": c, "out": enc.encode(c, allowed_special="all")} for c in cases]))
```
Your Zig `encode` must match exactly. (See §7 for which edge cases matter and why.)

**Forward pass — float diff against nanoGPT/HF.** Feed the *same* token ids to HF transformers and dump logits:
```python
from transformers import GPT2LMHeadModel
import torch
m = GPT2LMHeadModel.from_pretrained("gpt2").eval()
ids = torch.tensor([[15496, 11, 314, 1101]])
with torch.no_grad():
    logits = m(ids).logits          # [1, 4, 50257]
torch.save(logits, "ref_logits.pt") # or dump to .npy / raw f32
```
Compare your logits to these. Target: max abs diff on the order of `1e-3` or tighter in F32. If they diverge, the diff *location* tells you the bug — biggest offenders in order: Conv1D transpose, GELU flavor, LayerNorm (mean/variance/eps), attention scale (1/8), causal mask off-by-one, position-embedding indexing.

**Greedy is your friend here:** deterministic argmax means "does my generated token sequence match the reference's" is a clean pass/fail, not a fuzzy comparison.

---

## 5. The silent-failure gotcha table

These compile and run and produce *plausible* tokens while being wrong. Each is worth an explicit test.

| Gotcha | Symptom | Fix |
|---|---|---|
| **Conv1D transpose** | garbage matmul output | weights are `[in, out]`; transpose once at load |
| **Fused QKV** | wrong attention | `c_attn` → 2304; slice into Q,K,V at 768 boundaries |
| **GELU flavor** | small logit drift, fails tight parity | use tanh approx (`gelu_new`), not erf |
| **LayerNorm not RMSNorm** | drift | subtract mean *and* add β bias; biased variance; eps 1e-5 |
| **Attention scale** | drift | divide scores by `sqrt(head_dim)=8`, not by head_dim |
| **Causal mask off-by-one** | subtle, attends to future | mask `j > i` (strictly future), keep `j == i` |
| **Tied embeddings** | hunting for missing `lm_head` | there is none; reuse `wte.weight` transposed |
| **Byte↔unicode map** | breaks on spaces/newlines/tabs | GPT-2 remaps 256 bytes to printable codepoints (`Ġ`=space); implement the reversible map |
| **Leading space** | wrong tokens in context | `"hello"` ≠ `" hello"`; test both |
| **Special tokens** | corrupted chat formatting | `encode("<|endoftext|>")` → `[50256]` atomic, not byte-split |

---

## 6. Zig module layout

```
src/
  main.zig            # CLI / REPL, generation loop
  safetensors.zig     # mmap + header parse → name→tensor slices
  config.zig          # parse config.json
  tokenizer.zig       # BPE encode/decode  (hand-write or vendor)
  model.zig           # weight structs, layer forward, full forward
  math.zig            # matmul, layernorm, softmax, gelu  ← the part you hand-write & optimize
  tests/
    tokenizer_test.zig  # golden table vs tiktoken + fuzzed roundtrip
    forward_test.zig    # logit diff vs HF dump
```

**Discipline that makes this a valid enjoyment experiment:** let the AI write `safetensors.zig`, `config.zig`, `tokenizer.zig`, the CLI, and the comparison harnesses — the boilerplate it's good at. **You** write `math.zig` and own the correctness loop. The kernels and the bug hunt are the parts you're actually evaluating; if the AI writes those, you learn whether *it* enjoys inference work, not whether you do.

**On the tokenizer specifically:** it's the most unit-testable thing in the build (deterministic, integer output, exact oracle, lossless). Drive it to provably correct — golden table + fuzzed `decode(encode(s)) == s` — then treat it as a closed black box. Note the roundtrip is directional: `decode(encode(s)) == s` always holds; `encode(decode(ids)) == ids` does **not** for arbitrary id sequences. Test the first direction only.

---

## 7. Tokenizer edge cases worth pinning

Put all of these in the golden table; each targets a real bug:

- **Leading space:** `" hello"` ≠ `"hello"` — different IDs. Pre-tokenization attaches the space.
- **Whitespace / byte map:** `"\n"`, `"\t"`, runs of spaces — exercises the byte↔unicode remap, the #1 BPE bug.
- **Contractions:** `"don't"`, `"they're"` — the regex special-cases `'s 't 're 've 'm 'll 'd`.
- **Digit runs:** `"2024"` — GPT-2 splits digits differently from GPT-4.
- **Multibyte / invalid UTF-8:** `"café"`, emoji, CJK, a malformed byte — mostly a roundtrip-property concern.
- **Repeated chars:** `"aaaaaaaa"` — exercises merge ordering.
- **Empty string** → `[]`.
- **Special token:** `"<|endoftext|>"` → `[50256]` atomic. This is the one most likely to silently corrupt formatting later.

Pick **one** oracle and match it exactly — tiktoken `gpt2`, HF `tokenizer.json`, and llama.cpp's BPE diverge on edge cases (special tokens especially). Match whatever your *logit* reference uses to remove a confound.

---

## 8. Performance optimizations, ranked

First, the mental model that makes the ranking make sense. Generation has two phases with **different bottlenecks**:

- **Prefill** (process the prompt, all positions at once): compute-bound **GEMM** (matrix–matrix). Benefits from blocking, SIMD, threading.
- **Decode** (generate one token at a time): memory-bandwidth-bound **GEMV** (matrix–vector). Each step streams the entire weight set once to produce one token. Benefits most from *moving fewer bytes* — i.e. quantization — and from threading.

For any realistic generation length, **decode dominates wall-clock**, and decode is bandwidth-bound. That ordering drives the list. Record tokens/sec after every change — forming a bandwidth hypothesis, applying one change, and watching the number move (or stubbornly not) is the core loop of the job.

### Ranked

**1. KV cache — the only algorithmic (complexity-class) win.**
Without it, generating N tokens reprocesses the whole growing prefix every step: O(N²) total work. With it, you cache each layer's K and V `[layer, head, seq, head_dim]` and each step computes Q for *only the new token*, attends against cached K/V, and appends. Per-step work goes from "whole sequence" to "one token." For multi-token generation this is the single biggest win, often 10–100× depending on length. It also *changes the bottleneck*: post-cache, decode is GEMV, which sets up everything below. Do this first.

**2. Multithread the matmuls — biggest, easiest constant-factor win.**
The output dimension is embarrassingly parallel: split rows/columns across a thread pool (`std.Thread.Pool`). Near-linear scaling with physical cores → 4–16×. Lowest complexity-to-payoff ratio on the list. The big per-token matmul to parallelize hardest is the **final logits projection**: `[768] × [768, 50257]` is the largest single GEMV per decode step (~38M MACs), bigger than any one attention or MLP matmul — it alone is a large fraction of per-token cost.

**3. SIMD the inner loop (`@Vector` / AVX-512).**
Dot products vectorize trivially. AVX-512 processes 16 f32/lane; realized speedups are typically 4–8× on the hot loop. Use Zig's `@Vector(16, f32)` and a horizontal reduce; let the backend lower it, or drop to intrinsics where it won't. Pairs with #2 (thread across rows, SIMD within each).

**4. Quantization (Q8 → Q4) — the most *inference-representative* optimization.**
Since decode is bandwidth-bound, halving (Q8) or quartering (Q4) the bytes moved per weight near-directly multiplies decode throughput, and shrinks the memory footprint. Cost: dequantize in the hot loop (or use integer dot products with per-block scales) and *validate accuracy* against the F32 logits — quantization is the one optimization that can change outputs, so it gets its own correctness pass. This is the optimization that most resembles the day job at a frontier lab; if the accuracy-vs-throughput tuning is fun, that's strong signal.

**5. Cache blocking / tiling (matters most for prefill GEMM).**
Naive `ijk` matmul thrashes cache. Tile for L1/L2 (block the M/N/K loops), keep the inner kernel's working set resident. High value when prompts are long (prefill-heavy); lower when decode dominates. Standard HPC technique — good profiler practice.

**6. Memory layout / weight pre-packing.**
Pre-transpose the Conv1D weights once at load (you're doing this for correctness anyway — do it in the layout the matmul wants). Store weights so the inner loop reads contiguously; pack quantized blocks with their scales adjacent. Foundational — it's what makes #3 and #4 actually hit their ceilings — but low *headline* number on its own.

**7. Fuse ops & kill per-token allocations.**
Fuse bias-add + activation into the matmul epilogue; reuse scratch buffers across steps instead of allocating per token; keep the KV cache in one preallocated arena. Smaller multiplier, but removes overhead that otherwise caps the gains above.

### Suggested order of attack
Get correct (greedy, F32, single-thread, no cache) → **KV cache** → **thread** → **SIMD** → **quantize** → tile/layout/fuse. Re-measure tokens/sec at each step; keep the F32 logit-diff test green through #1–#3, add a quant-accuracy test at #4.

---

## 9. Reference link index

**Model:** https://huggingface.co/openai-community/gpt2
**nanoGPT:** https://github.com/karpathy/nanoGPT
**minGPT:** https://github.com/karpathy/minGPT
**tiktoken:** https://github.com/openai/tiktoken
**safetensors:** https://github.com/huggingface/safetensors
**GPT-2 paper (Radford et al., "Language Models are Unsupervised Multitask Learners"):** the byte-level BPE + architecture source
**Zig BPE (optional vendor):** https://github.com/jaco-bro/tokenizer · https://github.com/alvarobartt/tokeni.zig

---

## 10. What "done" looks like by Sunday night

- Greedy generation that **matches the reference token-for-token** on a fixed prompt (correctness proven, not vibes).
- A tokenizer with a green golden table + fuzzed roundtrip.
- A tokens/sec number you've moved by **at least** KV cache + threading + SIMD, with a profiler open and a hypothesis behind each change.

If the correctness bug-hunt energized you rather than drained you, and you reached for the profiler instinctively rather than avoiding it — that's your signal. The scale and codebase-archaeology of a real lab differ, but the inner-loop texture transfers cleanly.
