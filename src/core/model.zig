//! M3 forward pass: token IDs → logits. Composes the M1 loader (safetensors weights) and the M2
//! kernels (op.zig) into the full GPT-2 stack. Validated against per-stage HuggingFace activation
//! goldens (see python/gen_activation_goldens.py) with the hardcoded prompt "Hello, I am".
const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;

const op = @import("op.zig");
const tensor = @import("tensor.zig");
const Config = @import("config.zig").Config;
const safetensors = @import("../safetensors/safetensors.zig");
const SafeTensors = safetensors.SafeTensors;

const MAX_LAYERS = 12; // gpt2-small; init asserts cfg.n_layer fits.

// =================
// === Weights ===
//
// Aligned copies of every tensor, owned for the program lifetime. The mmap'd source is unaligned
// (safetensors data_base isn't 4-aligned), so we @memcpy each tensor into a page_allocator buffer
// rather than relax the M2 kernels to align(1). Never freed — params outlive everything.

const Layer = struct {
    ln_1_w: []const f32, // [n_embd]
    ln_1_b: []const f32,
    attn_w: []const f32, // c_attn.weight [n_embd, 3*n_embd]
    attn_b: []const f32, // c_attn.bias  [3*n_embd]
    attn_proj_w: []const f32, // c_proj.weight [n_embd, n_embd]
    attn_proj_b: []const f32, // [n_embd]
    ln_2_w: []const f32,
    ln_2_b: []const f32,
    fc_w: []const f32, // mlp.c_fc.weight [n_embd, 4*n_embd]
    fc_b: []const f32, // [4*n_embd]
    mlp_proj_w: []const f32, // mlp.c_proj.weight [4*n_embd, n_embd]
    mlp_proj_b: []const f32, // [n_embd]
};

const Weights = struct {
    wte: []const f32, // [vocab, n_embd]
    wpe: []const f32, // [n_ctx, n_embd]
    ln_f_w: []const f32, // [n_embd]
    ln_f_b: []const f32,
    layers: [MAX_LAYERS]Layer,
};

// Copy a named tensor out of the mmap into a fresh aligned buffer. Panics if the tensor is absent
// or the wrong size — both are data-integrity failures we want surfaced at load, not in layer 7.
fn loadAligned(st: *const SafeTensors, name: []const u8, expected_len: usize) ![]const f32 {
    const h = st.find(name) orelse std.debug.panic("missing tensor: {s}", .{name});
    const src = tensor.bytesAsF32(st.data_bytes(h));
    assert(src.len == expected_len);
    const dst = try std.heap.page_allocator.alloc(f32, expected_len);
    @memcpy(dst, src);
    return dst;
}

// =================
// === Scratch ===
//
// Bounded activation buffers, allocated once at init at the n_ctx upper bound — no per-token or
// per-layer allocation. `scores` is a single row reused per (head, query), so attention never
// materializes an [S,S] matrix.

const Scratch = struct {
    x: []f32, // residual stream [n_ctx, n_embd]
    ln: []f32, // layernorm output [n_ctx, n_embd]
    qkv: []f32, // fused QKV [n_ctx, 3*n_embd]
    attn: []f32, // concatenated heads [n_ctx, n_embd]
    proj: []f32, // attention/mlp projection out [n_ctx, n_embd]
    hidden: []f32, // mlp hidden [n_ctx, 4*n_embd]
    mlp: []f32, // mlp output [n_ctx, n_embd]
    scores: []f32, // one attention row [n_ctx]
};

// =================
// === Taps ===
//
// Opt-in activation captures for the bisection test. forward() copies the live activation into any
// non-null field at the matching point; production passes `.{}` and pays nothing.

pub const Taps = struct {
    embed: ?[]f32 = null,
    l0_ln1: ?[]f32 = null,
    l0_attn: ?[]f32 = null,
    l0_resid1: ?[]f32 = null,
    l0_mlp: ?[]f32 = null,
    l0_out: ?[]f32 = null,
    l5_out: ?[]f32 = null,
    lnf: ?[]f32 = null,
};

fn tap(dst: ?[]f32, src: []const f32) void {
    if (dst) |d| {
        assert(d.len == src.len);
        @memcpy(d, src);
    }
}

// =================
// === Model ===

pub const Model = struct {
    cfg: Config,
    w: Weights,
    s: Scratch,

    pub fn init(st: *const SafeTensors, cfg: Config) !Model {
        assert(cfg.n_layer <= MAX_LAYERS);
        assert(cfg.n_embd % cfg.n_head == 0);

        const n_embd: usize = cfg.n_embd;
        const n_ctx: usize = cfg.n_ctx;
        const vocab: usize = cfg.vocab_size;

        var w: Weights = .{
            .wte = try loadAligned(st, "wte.weight", vocab * n_embd),
            .wpe = try loadAligned(st, "wpe.weight", n_ctx * n_embd),
            .ln_f_w = try loadAligned(st, "ln_f.weight", n_embd),
            .ln_f_b = try loadAligned(st, "ln_f.bias", n_embd),
            .layers = undefined,
        };

        var name_buf: [64]u8 = undefined;
        for (0..cfg.n_layer) |i| {
            const N = struct {
                fn f(buf: []u8, layer: usize, comptime suffix: []const u8) []const u8 {
                    return std.fmt.bufPrint(buf, "h.{d}." ++ suffix, .{layer}) catch unreachable;
                }
            }.f;
            w.layers[i] = .{
                .ln_1_w = try loadAligned(st, N(&name_buf, i, "ln_1.weight"), n_embd),
                .ln_1_b = try loadAligned(st, N(&name_buf, i, "ln_1.bias"), n_embd),
                .attn_w = try loadAligned(st, N(&name_buf, i, "attn.c_attn.weight"), n_embd * 3 * n_embd),
                .attn_b = try loadAligned(st, N(&name_buf, i, "attn.c_attn.bias"), 3 * n_embd),
                .attn_proj_w = try loadAligned(st, N(&name_buf, i, "attn.c_proj.weight"), n_embd * n_embd),
                .attn_proj_b = try loadAligned(st, N(&name_buf, i, "attn.c_proj.bias"), n_embd),
                .ln_2_w = try loadAligned(st, N(&name_buf, i, "ln_2.weight"), n_embd),
                .ln_2_b = try loadAligned(st, N(&name_buf, i, "ln_2.bias"), n_embd),
                .fc_w = try loadAligned(st, N(&name_buf, i, "mlp.c_fc.weight"), n_embd * 4 * n_embd),
                .fc_b = try loadAligned(st, N(&name_buf, i, "mlp.c_fc.bias"), 4 * n_embd),
                .mlp_proj_w = try loadAligned(st, N(&name_buf, i, "mlp.c_proj.weight"), 4 * n_embd * n_embd),
                .mlp_proj_b = try loadAligned(st, N(&name_buf, i, "mlp.c_proj.bias"), n_embd),
            };
        }

        const a = std.heap.page_allocator;
        const s: Scratch = .{
            .x = try a.alloc(f32, n_ctx * n_embd),
            .ln = try a.alloc(f32, n_ctx * n_embd),
            .qkv = try a.alloc(f32, n_ctx * 3 * n_embd),
            .attn = try a.alloc(f32, n_ctx * n_embd),
            .proj = try a.alloc(f32, n_ctx * n_embd),
            .hidden = try a.alloc(f32, n_ctx * 4 * n_embd),
            .mlp = try a.alloc(f32, n_ctx * n_embd),
            .scores = try a.alloc(f32, n_ctx),
        };

        return .{ .cfg = cfg, .w = w, .s = s };
    }

    pub fn deinit(self: *Model) void {
        const a = std.heap.page_allocator;
        a.free(self.s.x);
        a.free(self.s.ln);
        a.free(self.s.qkv);
        a.free(self.s.attn);
        a.free(self.s.proj);
        a.free(self.s.hidden);
        a.free(self.s.mlp);
        a.free(self.s.scores);
        // Weights are intentionally leaked: program-lifetime params, page-allocated.
    }

    // ids:[S] → logits_out:[S, vocab]. taps capture intermediates for bisection (pass .{} in prod).
    pub fn forward(self: *Model, ids: []const u32, logits_out: []f32, taps: Taps) void {
        const cfg = self.cfg;
        const n_embd: usize = cfg.n_embd;
        const n_head: usize = cfg.n_head;
        const head_dim: usize = n_embd / n_head;
        const vocab: usize = cfg.vocab_size;
        const S = ids.len;

        assert(S > 0 and S <= cfg.n_ctx);
        for (ids) |id| assert(id < cfg.vocab_size);
        assert(logits_out.len == S * vocab);

        const x = self.s.x[0 .. S * n_embd];

        // =================
        // === Embedding ===
        // x[s] = wte[ids[s]] + wpe[s]
        for (0..S) |si| {
            const wte_row = self.w.wte[ids[si] * n_embd ..][0..n_embd];
            const wpe_row = self.w.wpe[si * n_embd ..][0..n_embd];
            const dst = x[si * n_embd ..][0..n_embd];
            for (0..n_embd) |d| dst[d] = wte_row[d] + wpe_row[d];
        }
        tap(taps.embed, x);

        // =================
        // === Transformer layers ===
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
        const qkv_stride = 3 * n_embd;

        for (0..cfg.n_layer) |layer| {
            const L = self.w.layers[layer];

            // --- attention block ---
            const ln = self.s.ln[0 .. S * n_embd];
            layernormRows(x, L.ln_1_w, L.ln_1_b, cfg.ln_eps, ln, S, n_embd);
            if (layer == 0) tap(taps.l0_ln1, ln);

            const qkv = self.s.qkv[0 .. S * qkv_stride];
            op.matmul(ln, L.attn_w, L.attn_b, qkv, S, n_embd, qkv_stride);

            const attn = self.s.attn[0 .. S * n_embd];
            for (0..n_head) |h| {
                const q_off = h * head_dim; // Q block starts at col 0
                const k_off = n_embd + h * head_dim; // K block at col n_embd
                const v_off = 2 * n_embd + h * head_dim; // V block at col 2*n_embd
                for (0..S) |i| {
                    // scores[j] = (q_i · k_j) * scale, causal: only j <= i.
                    const q = qkv[i * qkv_stride + q_off ..][0..head_dim];
                    const scores = self.s.scores[0 .. i + 1];
                    for (0..i + 1) |j| {
                        const k = qkv[j * qkv_stride + k_off ..][0..head_dim];
                        var dot: f32 = 0;
                        for (0..head_dim) |d| dot += q[d] * k[d];
                        scores[j] = dot * scale;
                    }
                    op.softmax(scores, scores);

                    // out_i = Σ_{j<=i} p[j] · v_j, written into attn[i, h*head_dim:]
                    const out = attn[i * n_embd + h * head_dim ..][0..head_dim];
                    @memset(out, 0);
                    for (0..i + 1) |j| {
                        const v = qkv[j * qkv_stride + v_off ..][0..head_dim];
                        const p = scores[j];
                        for (0..head_dim) |d| out[d] += p * v[d];
                    }
                }
            }

            const proj = self.s.proj[0 .. S * n_embd];
            op.matmul(attn, L.attn_proj_w, L.attn_proj_b, proj, S, n_embd, n_embd);
            if (layer == 0) tap(taps.l0_attn, proj);

            for (0..S * n_embd) |idx| x[idx] += proj[idx];
            if (layer == 0) tap(taps.l0_resid1, x);

            // --- mlp block ---
            const hdim = 4 * n_embd;
            const ln2 = self.s.ln[0 .. S * n_embd];
            layernormRows(x, L.ln_2_w, L.ln_2_b, cfg.ln_eps, ln2, S, n_embd);

            const hidden = self.s.hidden[0 .. S * hdim];
            op.matmul(ln2, L.fc_w, L.fc_b, hidden, S, n_embd, hdim);
            op.gelu(hidden, hidden);

            const mlp = self.s.mlp[0 .. S * n_embd];
            op.matmul(hidden, L.mlp_proj_w, L.mlp_proj_b, mlp, S, hdim, n_embd);
            if (layer == 0) tap(taps.l0_mlp, mlp);

            for (0..S * n_embd) |idx| x[idx] += mlp[idx];
            if (layer == 0) tap(taps.l0_out, x);
            if (layer == 5) tap(taps.l5_out, x);

            assertFinite(x);
        }

        // =================
        // === Final norm + tied logits ===
        const lnf = self.s.ln[0 .. S * n_embd];
        layernormRows(x, self.w.ln_f_w, self.w.ln_f_b, cfg.ln_eps, lnf, S, n_embd);
        tap(taps.lnf, lnf);

        // logits = lnf @ wteᵀ — tied output embedding, no bias.
        op.matmulBT(lnf, self.w.wte, logits_out, S, n_embd, vocab);
    }
};

// Apply layernorm per row of a [rows, cols] row-major matrix.
fn layernormRows(in: []const f32, gamma: []const f32, beta: []const f32, eps: f32, out: []f32, rows: usize, cols: usize) void {
    assert(in.len == rows * cols and out.len == rows * cols);
    for (0..rows) |r| {
        const lo = r * cols;
        op.layernorm(in[lo .. lo + cols], gamma, beta, eps, out[lo .. lo + cols]);
    }
}

fn assertFinite(buf: []const f32) void {
    if (builtin.mode != .Debug) return;
    for (buf) |v| assert(std.math.isFinite(v));
}

// =================
// === Tests ===

const testdata = @import("testdata.zig");
const Golden = testdata.Golden;
const asset = @import("asset/asset.zig");

// Hardcoded prompt "Hello, I am" — the same ids gen_ref_logits.py / gen_activation_goldens.py use.
const PROMPT = [_]u32{ 15496, 11, 314, 1101 };

fn argmaxLastRow(logits: []align(1) const f32, s: usize, vocab: usize) usize {
    const row = logits[(s - 1) * vocab ..][0..vocab];
    var best: usize = 0;
    for (row, 0..) |v, i| {
        if (v > row[best]) best = i;
    }
    return best;
}

test "forward bisection vs activation goldens" {
    const io = std.testing.io;
    const alloc = std.testing.allocator;

    var st = try SafeTensors.init(io, asset.model_safetensors_path);
    const cfg = try Config.fromBytes(asset.config_json);
    var model = try Model.init(&st, cfg);
    st.deinit(); // weights copied; mmap no longer referenced
    defer model.deinit();

    const S = PROMPT.len;
    const n_embd: usize = cfg.n_embd;
    const vocab: usize = cfg.vocab_size;

    // Goldens are all-or-nothing; if absent (not generated), skip the whole bisection test.
    const gld = asset.act_goldens orelse {
        std.debug.print("\nactivation goldens missing — run `zig build gen-goldens` to enable the M3 bisection test\n", .{});
        return error.SkipZigTest;
    };
    var g_logits = Golden.fromBytes(gld.logits);
    defer g_logits.deinit();
    var g_embed = Golden.fromBytes(gld.embed);
    defer g_embed.deinit();
    var g_l0_ln1 = Golden.fromBytes(gld.l0_ln1);
    defer g_l0_ln1.deinit();
    var g_l0_attn = Golden.fromBytes(gld.l0_attn);
    defer g_l0_attn.deinit();
    var g_l0_resid1 = Golden.fromBytes(gld.l0_resid1);
    defer g_l0_resid1.deinit();
    var g_l0_mlp = Golden.fromBytes(gld.l0_mlp);
    defer g_l0_mlp.deinit();
    var g_l0_out = Golden.fromBytes(gld.l0_out);
    defer g_l0_out.deinit();
    var g_l5_out = Golden.fromBytes(gld.l5_out);
    defer g_l5_out.deinit();
    var g_lnf = Golden.fromBytes(gld.lnf);
    defer g_lnf.deinit();

    // Tap buffers.
    const embed = try alloc.alloc(f32, S * n_embd);
    defer alloc.free(embed);
    const l0_ln1 = try alloc.alloc(f32, S * n_embd);
    defer alloc.free(l0_ln1);
    const l0_attn = try alloc.alloc(f32, S * n_embd);
    defer alloc.free(l0_attn);
    const l0_resid1 = try alloc.alloc(f32, S * n_embd);
    defer alloc.free(l0_resid1);
    const l0_mlp = try alloc.alloc(f32, S * n_embd);
    defer alloc.free(l0_mlp);
    const l0_out = try alloc.alloc(f32, S * n_embd);
    defer alloc.free(l0_out);
    const l5_out = try alloc.alloc(f32, S * n_embd);
    defer alloc.free(l5_out);
    const lnf = try alloc.alloc(f32, S * n_embd);
    defer alloc.free(lnf);
    const logits = try alloc.alloc(f32, S * vocab);
    defer alloc.free(logits);

    model.forward(&PROMPT, logits, .{
        .embed = embed,
        .l0_ln1 = l0_ln1,
        .l0_attn = l0_attn,
        .l0_resid1 = l0_resid1,
        .l0_mlp = l0_mlp,
        .l0_out = l0_out,
        .l5_out = l5_out,
        .lnf = lnf,
    });

    // Bisect in forward order: the first failing tap localizes the bug to one sub-op.
    // std.debug.print("\nM3 activation bisection:\n", .{});
    const act_tol: f32 = 1e-3;
    const logit_tol: f32 = 1e-2;
    try testdata.expectClose("embed", g_embed.data(), embed, act_tol);
    try testdata.expectClose("l0_ln1", g_l0_ln1.data(), l0_ln1, act_tol);
    try testdata.expectClose("l0_attn", g_l0_attn.data(), l0_attn, act_tol);
    try testdata.expectClose("l0_resid1", g_l0_resid1.data(), l0_resid1, act_tol);
    try testdata.expectClose("l0_mlp", g_l0_mlp.data(), l0_mlp, act_tol);
    try testdata.expectClose("l0_out", g_l0_out.data(), l0_out, act_tol);
    try testdata.expectClose("l5_out", g_l5_out.data(), l5_out, act_tol);
    try testdata.expectClose("lnf", g_lnf.data(), lnf, act_tol);
    try testdata.expectClose("logits", g_logits.data(), logits, logit_tol);

    // Tolerance-free signal: greedy next token must match HF's.
    const got = argmaxLastRow(logits, S, vocab);
    const want = argmaxLastRow(g_logits.data(), S, vocab);
    // std.debug.print("  argmax(last): got {d}, want {d}\n", .{ got, want });
    try std.testing.expectEqual(want, got);
}
