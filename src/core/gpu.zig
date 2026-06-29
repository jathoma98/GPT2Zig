//! GPU GPT-2 forward pass: composes the pure `vk` compute leaf (device, buffers, pipelines,
//! dispatch) into the transformer stack, dispatching one Slang kernel per layer op. This is the
//! business layer — it owns all GPT-2 knowledge (weight layout, the kernel sequence, the residual
//! stream) and keeps it OUT of the `vk` module. It mirrors the old CPU `Model` (src/core/model.zig)
//! op-for-op so the same HuggingFace goldens validate it.
//!
//! Execution is deliberately serial and CPU-like (correctness/simplicity over speed): the whole
//! model is uploaded to one GPU buffer once, every op is a separate kernel dispatch, and `vk`
//! `vkQueueWaitIdle`s after each. No barriers, no streaming.
const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;

const vk = @import("vk");
const tensor = @import("tensor.zig");
const Config = @import("config.zig").Config;
const safetensors = @import("../safetensors/safetensors.zig");
const SafeTensors = safetensors.SafeTensors;

const log = std.log.scoped(.gpu);

const MAX_LAYERS = 12; // gpt2-small; init asserts cfg.n_layer fits.

// =========================
// === Kernel bindings + SPIR-V (build-graph imports; see build.zig addSlangKernel) ===

const k_matmul = @import("matmul_bindings");
const k_matmul_bt = @import("matmul_bt_bindings");
const k_layernorm = @import("layernorm_bindings");
const k_softmax = @import("softmax_bindings");
const k_gelu = @import("gelu_bindings");
const k_embed = @import("embed_bindings");
const k_add = @import("add_bindings");
const k_scores = @import("attn_scores_bindings");
const k_attn_softmax = @import("attn_softmax_bindings");
const k_wsum = @import("attn_weighted_sum_bindings");

// SPIR-V blobs forced to 4-byte alignment (a u32 stream) via a comptime rodata copy.
const spv_matmul align(4) = @embedFile("matmul_spv").*;
const spv_matmul_bt align(4) = @embedFile("matmul_bt_spv").*;
const spv_layernorm align(4) = @embedFile("layernorm_spv").*;
const spv_softmax align(4) = @embedFile("softmax_spv").*;
const spv_gelu align(4) = @embedFile("gelu_spv").*;
const spv_embed align(4) = @embedFile("embed_spv").*;
const spv_add align(4) = @embedFile("add_spv").*;
const spv_scores align(4) = @embedFile("attn_scores_spv").*;
const spv_attn_softmax align(4) = @embedFile("attn_softmax_spv").*;
const spv_wsum align(4) = @embedFile("attn_weighted_sum_spv").*;

// Every kernel uses [64,1,1] and flat 1-D dispatch; the host passes the total thread count as the
// x extent. Pin that assumption to the codegen — change a shader's numthreads and this fails.
const WG: u32 = 64;
comptime {
    const sizes = [_][3]u32{
        k_matmul.local_size,  k_matmul_bt.local_size,    k_layernorm.local_size,
        k_softmax.local_size, k_gelu.local_size,         k_embed.local_size,
        k_add.local_size,     k_scores.local_size,       k_attn_softmax.local_size,
        k_wsum.local_size,
    };
    for (sizes) |ls| assert(ls[0] == WG and ls[1] == 1 and ls[2] == 1);
}
fn groups1d(total: u32) [3]u32 {
    return .{ (total + WG - 1) / WG, 1, 1 };
}

// =========================
// === Weight layout ===
//
// All weights live in one big GPU buffer, each tensor at an offset aligned to
// minStorageBufferOffsetAlignment so it can be bound as a descriptor sub-range. A TRef is that
// (byte offset, element count) view; `bytes()` is range = len*4.

const TRef = struct {
    off: u64,
    len: usize,
    fn bytes(self: TRef) u64 {
        return self.len * 4;
    }
};

const LayerRef = struct {
    ln_1_w: TRef,
    ln_1_b: TRef,
    attn_w: TRef,
    attn_b: TRef,
    attn_proj_w: TRef,
    attn_proj_b: TRef,
    ln_2_w: TRef,
    ln_2_b: TRef,
    fc_w: TRef,
    fc_b: TRef,
    mlp_proj_w: TRef,
    mlp_proj_b: TRef,
};

const WeightRefs = struct {
    wte: TRef,
    wpe: TRef,
    ln_f_w: TRef,
    ln_f_b: TRef,
    layers: [MAX_LAYERS]LayerRef,
};

// =========================
// === Scratch + Taps ===

const Scratch = struct {
    x: vk.Buffer, // residual stream [n_ctx, E]
    ln: vk.Buffer, // layernorm output [n_ctx, E] (reused by ln_1 / ln_2 / ln_f)
    qkv: vk.Buffer, // fused QKV [n_ctx, 3E]
    attn: vk.Buffer, // concatenated heads [n_ctx, E]
    proj: vk.Buffer, // attn/mlp projection out [n_ctx, E]
    hidden: vk.Buffer, // mlp hidden [n_ctx, 4E]
    mlp: vk.Buffer, // mlp output [n_ctx, E]
    scores: vk.Buffer, // attention scores [n_head, n_ctx, n_ctx]
    logits: vk.Buffer, // output logits [n_ctx, vocab]
    ids: vk.Buffer, // token ids [n_ctx] (u32)
};

// Mirror of model.zig's Taps: opt-in CPU read-backs for the bisection test.
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

// =========================
// === Pipelines ===

const Pipelines = struct {
    matmul: vk.Pipeline,
    matmul_bt: vk.Pipeline,
    layernorm: vk.Pipeline,
    gelu: vk.Pipeline,
    embed: vk.Pipeline,
    add: vk.Pipeline,
    scores: vk.Pipeline,
    attn_softmax: vk.Pipeline,
    wsum: vk.Pipeline,
    // `softmax` (standalone) is only used by its kernel golden test, not the forward pass, so it is
    // built lazily there rather than kept resident here.
};

// =========================
// === Gpu ===

// Instance options for production (master/slave). In Debug builds validation is on with the
// panic-on-error callback (a validation message aborts immediately); Release runs without it.
pub fn defaultInstanceOpts() vk.InitOptions {
    return if (builtin.mode == .Debug)
        .{ .enable_validation = true, .debug_callback = vk.panicCallback }
    else
        .{};
}

pub const Gpu = struct {
    vtbl: vk.VTbl,
    dev: vk.Device,
    cfg: Config,
    model: vk.Buffer,
    w: WeightRefs,
    s: Scratch,
    p: Pipelines,

    // Returns null when Vulkan is unavailable (no loader / ICD / compute device) — callers map that
    // to a test skip or a production fatal. A genuine device-creation or resource failure is a hard
    // error. `inst` carries the debug callback (panicCallback in production, captureCallback in tests).
    pub fn init(io: std.Io, gpa: std.mem.Allocator, model_path: []const u8, inst: vk.InitOptions) !?Gpu {
        var vtbl = vk.instanceOrSkip(inst) catch |e| switch (e) {
            error.SkipZigTest => return null,
            error.VkInstanceCreateFailed => return error.VkInstanceCreateFailed,
        };
        errdefer vtbl.deinit();

        var dev = vk.initDevice(&vtbl, gpa, .{}) catch |e| switch (e) {
            error.NoComputeDevice => {
                vtbl.deinit();
                return null;
            },
            else => return e,
        };
        errdefer dev.deinit();

        // --- model + config ---
        var st = try SafeTensors.init(io, model_path);
        const cfg = try Config.fromBytes(@import("asset/asset.zig").config_json);
        assert(cfg.n_layer <= MAX_LAYERS);
        assert(cfg.n_embd % cfg.n_head == 0);
        log.info("gpu: loading model (n_layer={d}, n_embd={d})", .{ cfg.n_layer, cfg.n_embd });

        // --- lay out + upload all weights into one big GPU buffer ---
        var w: WeightRefs = undefined;
        const model = try uploadWeights(&dev, &st, cfg, &w);
        st.deinit(); // weights now resident on the GPU; mmap no longer referenced

        var self: Gpu = .{
            .vtbl = vtbl,
            .dev = dev,
            .cfg = cfg,
            .model = model,
            .w = w,
            .s = undefined,
            .p = undefined,
        };
        try self.allocScratch();
        try self.buildPipelines();
        return self;
    }

    pub fn deinit(self: *Gpu) void {
        self.p.matmul.deinit(&self.dev);
        self.p.matmul_bt.deinit(&self.dev);
        self.p.layernorm.deinit(&self.dev);
        self.p.gelu.deinit(&self.dev);
        self.p.embed.deinit(&self.dev);
        self.p.add.deinit(&self.dev);
        self.p.scores.deinit(&self.dev);
        self.p.attn_softmax.deinit(&self.dev);
        self.p.wsum.deinit(&self.dev);
        inline for (std.meta.fields(Scratch)) |f| @field(self.s, f.name).deinit(&self.dev);
        self.model.deinit(&self.dev);
        self.dev.deinit();
        self.vtbl.deinit();
        self.* = undefined;
    }

    fn allocScratch(self: *Gpu) !void {
        const E = self.cfg.n_embd;
        const ctx = self.cfg.n_ctx;
        const H = self.cfg.n_head;
        const vocab = self.cfg.vocab_size;
        self.s = .{
            .x = try vk.createBuffer(&self.dev, ctx * E * 4),
            .ln = try vk.createBuffer(&self.dev, ctx * E * 4),
            .qkv = try vk.createBuffer(&self.dev, ctx * 3 * E * 4),
            .attn = try vk.createBuffer(&self.dev, ctx * E * 4),
            .proj = try vk.createBuffer(&self.dev, ctx * E * 4),
            .hidden = try vk.createBuffer(&self.dev, ctx * 4 * E * 4),
            .mlp = try vk.createBuffer(&self.dev, ctx * E * 4),
            .scores = try vk.createBuffer(&self.dev, H * ctx * ctx * 4),
            .logits = try vk.createBuffer(&self.dev, ctx * vocab * 4),
            .ids = try vk.createBuffer(&self.dev, ctx * 4),
        };
    }

    fn buildPipelines(self: *Gpu) !void {
        const d = &self.dev;
        self.p = .{
            .matmul = try vk.createComputePipeline(d, &spv_matmul, k_matmul.binding_count, k_matmul.push_constant_size, k_matmul.entry_point),
            .matmul_bt = try vk.createComputePipeline(d, &spv_matmul_bt, k_matmul_bt.binding_count, k_matmul_bt.push_constant_size, k_matmul_bt.entry_point),
            .layernorm = try vk.createComputePipeline(d, &spv_layernorm, k_layernorm.binding_count, k_layernorm.push_constant_size, k_layernorm.entry_point),
            .gelu = try vk.createComputePipeline(d, &spv_gelu, k_gelu.binding_count, k_gelu.push_constant_size, k_gelu.entry_point),
            .embed = try vk.createComputePipeline(d, &spv_embed, k_embed.binding_count, k_embed.push_constant_size, k_embed.entry_point),
            .add = try vk.createComputePipeline(d, &spv_add, k_add.binding_count, k_add.push_constant_size, k_add.entry_point),
            .scores = try vk.createComputePipeline(d, &spv_scores, k_scores.binding_count, k_scores.push_constant_size, k_scores.entry_point),
            .attn_softmax = try vk.createComputePipeline(d, &spv_attn_softmax, k_attn_softmax.binding_count, k_attn_softmax.push_constant_size, k_attn_softmax.entry_point),
            .wsum = try vk.createComputePipeline(d, &spv_wsum, k_wsum.binding_count, k_wsum.push_constant_size, k_wsum.entry_point),
        };
    }

    // =====================
    // === Binding helpers ===

    fn modelBind(self: *Gpu, ref: TRef) vk.BufferBinding {
        return .{ .buffer = self.model.buffer, .offset = ref.off, .range = ref.bytes() };
    }
    fn scratchBind(buf: *vk.Buffer, n_floats: usize) vk.BufferBinding {
        return .{ .buffer = buf.buffer, .offset = 0, .range = n_floats * 4 };
    }

    // A flat 1-D dispatch with push constants by value.
    fn disp(self: *Gpu, pipe: *vk.Pipeline, binds: []const vk.BufferBinding, push: anytype, total: u32) void {
        const bytes = if (@TypeOf(push) == void) &[_]u8{} else std.mem.asBytes(&push);
        vk.dispatch(&self.dev, pipe, binds, bytes, groups1d(total));
    }

    fn tapRead(self: *Gpu, dst: ?[]f32, buf: *vk.Buffer, n: usize) !void {
        if (dst) |d| {
            assert(d.len == n);
            try vk.readBuffer(&self.dev, buf, std.mem.sliceAsBytes(d));
        }
    }

    // =====================
    // === Forward-pass stages (mirror src/core/model.zig op-for-op) ===

    // ids:[S] -> residual stream in self.s.x. Uploads ids, then dispatches the embed kernel.
    pub fn embed(self: *Gpu, ids: []const u32, taps: Taps) !void {
        const E = self.cfg.n_embd;
        const S: u32 = @intCast(ids.len);
        assert(ids.len > 0 and ids.len <= self.cfg.n_ctx);
        try vk.writeBuffer(&self.dev, &self.s.ids, std.mem.sliceAsBytes(ids));

        const binds = [_]vk.BufferBinding{
            self.modelBind(self.w.wte),
            self.modelBind(self.w.wpe),
            scratchBind(&self.s.ids, ids.len),
            scratchBind(&self.s.x, ids.len * E),
        };
        self.disp(&self.p.embed, &binds, k_embed.PushConstants{ .S = S, .E = @intCast(E) }, S * @as(u32, @intCast(E)));
        try self.tapRead(taps.embed, &self.s.x, ids.len * E);
    }

    pub fn applyLayer(self: *Gpu, S: usize, layer_idx: usize, taps: Taps) !void {
        const cfg = self.cfg;
        const E: u32 = @intCast(cfg.n_embd);
        const H: u32 = @intCast(cfg.n_head);
        const hd: u32 = E / H;
        const Su: u32 = @intCast(S);
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        const L = self.w.layers[layer_idx];

        // --- attention block ---
        // ln = layernorm(x; ln_1) -> self.s.ln
        self.layernorm(&self.s.x, L.ln_1_w, L.ln_1_b, &self.s.ln, Su, E);
        if (layer_idx == 0) try self.tapRead(taps.l0_ln1, &self.s.ln, S * cfg.n_embd);

        // qkv = ln @ attn_w + attn_b  (M=S, K=E, N=3E)
        self.matmul(&self.s.ln, L.attn_w, L.attn_b, &self.s.qkv, Su, E, 3 * E);

        // attention: scores -> softmax -> weighted sum
        {
            const sc = k_scores.PushConstants{ .S = Su, .n_head = H, .head_dim = hd, .n_embd = E, .scale = scale };
            const b0 = [_]vk.BufferBinding{ scratchBind(&self.s.qkv, S * 3 * cfg.n_embd), scratchBind(&self.s.scores, @as(usize, H) * S * S) };
            self.disp(&self.p.scores, &b0, sc, H * Su * Su);

            const sm = k_attn_softmax.PushConstants{ .S = Su, .n_head = H };
            const b1 = [_]vk.BufferBinding{scratchBind(&self.s.scores, @as(usize, H) * S * S)};
            self.disp(&self.p.attn_softmax, &b1, sm, H * Su);

            const ws = k_wsum.PushConstants{ .S = Su, .n_head = H, .head_dim = hd, .n_embd = E };
            const b2 = [_]vk.BufferBinding{
                scratchBind(&self.s.qkv, S * 3 * cfg.n_embd),
                scratchBind(&self.s.scores, @as(usize, H) * S * S),
                scratchBind(&self.s.attn, S * cfg.n_embd),
            };
            self.disp(&self.p.wsum, &b2, ws, H * Su * hd);
        }

        // proj = attn @ attn_proj_w + attn_proj_b  (M=S, K=E, N=E)
        self.matmul(&self.s.attn, L.attn_proj_w, L.attn_proj_b, &self.s.proj, Su, E, E);
        if (layer_idx == 0) try self.tapRead(taps.l0_attn, &self.s.proj, S * cfg.n_embd);

        // x += proj
        self.add(&self.s.x, &self.s.proj, Su * E);
        if (layer_idx == 0) try self.tapRead(taps.l0_resid1, &self.s.x, S * cfg.n_embd);

        // --- mlp block ---
        // ln2 = layernorm(x; ln_2) -> self.s.ln
        self.layernorm(&self.s.x, L.ln_2_w, L.ln_2_b, &self.s.ln, Su, E);

        // hidden = ln2 @ fc_w + fc_b  (M=S, K=E, N=4E)
        self.matmul(&self.s.ln, L.fc_w, L.fc_b, &self.s.hidden, Su, E, 4 * E);
        // gelu(hidden) in place
        self.disp(&self.p.gelu, &[_]vk.BufferBinding{
            scratchBind(&self.s.hidden, S * 4 * cfg.n_embd),
            scratchBind(&self.s.hidden, S * 4 * cfg.n_embd),
        }, k_gelu.PushConstants{ .n = Su * 4 * E }, Su * 4 * E);

        // mlp = hidden @ mlp_proj_w + mlp_proj_b  (M=S, K=4E, N=E)
        self.matmul(&self.s.hidden, L.mlp_proj_w, L.mlp_proj_b, &self.s.mlp, Su, 4 * E, E);
        if (layer_idx == 0) try self.tapRead(taps.l0_mlp, &self.s.mlp, S * cfg.n_embd);

        // x += mlp
        self.add(&self.s.x, &self.s.mlp, Su * E);
        if (layer_idx == 0) try self.tapRead(taps.l0_out, &self.s.x, S * cfg.n_embd);
        if (layer_idx == 5) try self.tapRead(taps.l5_out, &self.s.x, S * cfg.n_embd);
    }

    pub fn runLayers(self: *Gpu, S: usize, lo: usize, hi: usize) !void {
        assert(lo <= hi and hi <= self.cfg.n_layer);
        for (lo..hi) |layer| try self.applyLayer(S, layer, .{});
    }

    // x:[S,E] -> logits_out:[S, vocab]. Final layernorm + tied output projection, then read back.
    pub fn tail(self: *Gpu, S: usize, logits_out: []f32, taps: Taps) !void {
        const cfg = self.cfg;
        const E: u32 = @intCast(cfg.n_embd);
        const vocab: u32 = @intCast(cfg.vocab_size);
        const Su: u32 = @intCast(S);
        assert(logits_out.len == S * cfg.vocab_size);

        // lnf = layernorm(x; ln_f) -> self.s.ln
        self.layernorm(&self.s.x, self.w.ln_f_w, self.w.ln_f_b, &self.s.ln, Su, E);
        try self.tapRead(taps.lnf, &self.s.ln, S * cfg.n_embd);

        // logits = lnf @ wteᵀ  (M=S, K=E, N=vocab), no bias
        const binds = [_]vk.BufferBinding{
            scratchBind(&self.s.ln, S * cfg.n_embd),
            self.modelBind(self.w.wte),
            scratchBind(&self.s.logits, S * cfg.vocab_size),
        };
        self.disp(&self.p.matmul_bt, &binds, k_matmul_bt.PushConstants{ .M = Su, .K = E, .N = vocab }, Su * vocab);
        try vk.readBuffer(&self.dev, &self.s.logits, std.mem.sliceAsBytes(logits_out));
    }

    pub fn forward(self: *Gpu, ids: []const u32, logits_out: []f32, taps: Taps) !void {
        const S = ids.len;
        try self.embed(ids, taps);
        for (0..self.cfg.n_layer) |layer| try self.applyLayer(S, layer, taps);
        try self.tail(S, logits_out, taps);
    }

    // === residual stream read/write for the distributed broadcast (master/slave) ===
    // Read the [S,E] residual prefix out of GPU memory (after a deviceWaitIdle by the caller).
    pub fn readResidual(self: *Gpu, S: usize, out: []f32) !void {
        assert(out.len == S * self.cfg.n_embd);
        try vk.readBuffer(&self.dev, &self.s.x, std.mem.sliceAsBytes(out));
    }
    // Upload a received [S,E] residual into GPU memory before running this shard's layers.
    pub fn writeResidual(self: *Gpu, S: usize, in: []const f32) !void {
        assert(in.len == S * self.cfg.n_embd);
        try vk.writeBuffer(&self.dev, &self.s.x, std.mem.sliceAsBytes(in));
    }

    pub fn deviceWaitIdle(self: *Gpu) void {
        self.dev.waitIdle();
    }

    // === op wrappers (bind the right buffers + push constants for the shared kernels) ===
    fn matmul(self: *Gpu, x: *vk.Buffer, w_ref: TRef, b_ref: TRef, y: *vk.Buffer, m: u32, k: u32, n: u32) void {
        const binds = [_]vk.BufferBinding{
            scratchBind(x, @as(usize, m) * k),
            self.modelBind(w_ref),
            self.modelBind(b_ref),
            scratchBind(y, @as(usize, m) * n),
        };
        self.disp(&self.p.matmul, &binds, k_matmul.PushConstants{ .M = m, .K = k, .N = n }, m * n);
    }

    fn layernorm(self: *Gpu, in: *vk.Buffer, gamma: TRef, beta: TRef, out: *vk.Buffer, rows: u32, cols: u32) void {
        const binds = [_]vk.BufferBinding{
            scratchBind(in, @as(usize, rows) * cols),
            self.modelBind(gamma),
            self.modelBind(beta),
            scratchBind(out, @as(usize, rows) * cols),
        };
        self.disp(&self.p.layernorm, &binds, k_layernorm.PushConstants{ .rows = rows, .cols = cols, .eps = self.cfg.ln_eps }, rows);
    }

    fn add(self: *Gpu, x: *vk.Buffer, y: *vk.Buffer, n: u32) void {
        const binds = [_]vk.BufferBinding{ scratchBind(x, n), scratchBind(y, n) };
        self.disp(&self.p.add, &binds, k_add.PushConstants{ .n = n }, n);
    }
};

// =========================
// === Weight upload ===

fn alignUp(x: usize, a: usize) usize {
    return (x + a - 1) / a * a;
}

// Concatenate every weight tensor into one host blob (each aligned to the storage-offset
// requirement), create a single GPU buffer, and upload it. Returns the model buffer; fills `refs`.
fn uploadWeights(dev: *vk.Device, st: *const SafeTensors, cfg: Config, refs: *WeightRefs) !vk.Buffer {
    const blob_alloc = std.heap.page_allocator;
    const E = cfg.n_embd;
    const al: usize = @intCast(dev.min_storage_offset_align);

    var blob = std.ArrayList(u8).empty;
    defer blob.deinit(blob_alloc);
    // Generous upper bound so appends never reallocate (148 tensors, ~497 MB for gpt2-small).
    const approx = (cfg.vocab_size + cfg.n_ctx + 2) * E + cfg.n_layer * (12 * E * E + 13 * E);
    try blob.ensureTotalCapacity(blob_alloc, approx * 4 + 148 * al + 4096);

    const put = struct {
        fn f(b: *std.ArrayList(u8), a: usize, ga: std.mem.Allocator, s: *const SafeTensors, name: []const u8, len: usize) !TRef {
            const pad = alignUp(b.items.len, a) - b.items.len;
            try b.appendNTimes(ga, 0, pad);
            const off = b.items.len;
            const h = s.find(name) orelse std.debug.panic("missing tensor: {s}", .{name});
            const bytes = s.data_bytes(h);
            assert(bytes.len == len * 4);
            try b.appendSlice(ga, bytes);
            return .{ .off = off, .len = len };
        }
    }.f;

    refs.wte = try put(&blob, al, blob_alloc, st, "wte.weight", cfg.vocab_size * E);
    refs.wpe = try put(&blob, al, blob_alloc, st, "wpe.weight", cfg.n_ctx * E);
    refs.ln_f_w = try put(&blob, al, blob_alloc, st, "ln_f.weight", E);
    refs.ln_f_b = try put(&blob, al, blob_alloc, st, "ln_f.bias", E);

    var name_buf: [64]u8 = undefined;
    for (0..cfg.n_layer) |i| {
        const N = struct {
            fn f(buf: []u8, layer: usize, comptime suffix: []const u8) []const u8 {
                return std.fmt.bufPrint(buf, "h.{d}." ++ suffix, .{layer}) catch unreachable;
            }
        }.f;
        refs.layers[i] = .{
            .ln_1_w = try put(&blob, al, blob_alloc, st, N(&name_buf, i, "ln_1.weight"), E),
            .ln_1_b = try put(&blob, al, blob_alloc, st, N(&name_buf, i, "ln_1.bias"), E),
            .attn_w = try put(&blob, al, blob_alloc, st, N(&name_buf, i, "attn.c_attn.weight"), E * 3 * E),
            .attn_b = try put(&blob, al, blob_alloc, st, N(&name_buf, i, "attn.c_attn.bias"), 3 * E),
            .attn_proj_w = try put(&blob, al, blob_alloc, st, N(&name_buf, i, "attn.c_proj.weight"), E * E),
            .attn_proj_b = try put(&blob, al, blob_alloc, st, N(&name_buf, i, "attn.c_proj.bias"), E),
            .ln_2_w = try put(&blob, al, blob_alloc, st, N(&name_buf, i, "ln_2.weight"), E),
            .ln_2_b = try put(&blob, al, blob_alloc, st, N(&name_buf, i, "ln_2.bias"), E),
            .fc_w = try put(&blob, al, blob_alloc, st, N(&name_buf, i, "mlp.c_fc.weight"), E * 4 * E),
            .fc_b = try put(&blob, al, blob_alloc, st, N(&name_buf, i, "mlp.c_fc.bias"), 4 * E),
            .mlp_proj_w = try put(&blob, al, blob_alloc, st, N(&name_buf, i, "mlp.c_proj.weight"), 4 * E * E),
            .mlp_proj_b = try put(&blob, al, blob_alloc, st, N(&name_buf, i, "mlp.c_proj.bias"), E),
        };
    }

    var model = try vk.createBuffer(dev, blob.items.len);
    errdefer model.deinit(dev);
    try vk.writeBuffer(dev, &model, blob.items);
    log.info("gpu: uploaded {d} MB of weights", .{blob.items.len / (1024 * 1024)});
    return model;
}

// =========================
// === Tests ===

const testdata = @import("testdata.zig");
const golden = @import("../generated/kernel_golden.zig");
const asset = @import("asset/asset.zig");

// File-scope so the messenger's pUserData pointer (set in instanceOrSkip) outlives the VTbl.
var test_cap: vk.ValidationCapture = .{};

fn testInstanceOpts() vk.InitOptions {
    return .{ .enable_validation = true, .debug_callback = vk.captureCallback, .debug_user_data = &test_cap };
}

const Harness = struct {
    vt: vk.VTbl,
    dev: vk.Device,

    fn init() !Harness {
        test_cap = .{};
        var vt = vk.instanceOrSkip(testInstanceOpts()) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        errdefer vt.deinit();
        const dev = vk.initDevice(&vt, std.testing.allocator, .{}) catch |e| switch (e) {
            error.NoComputeDevice => {
                vt.deinit();
                return error.SkipZigTest;
            },
            else => return e,
        };
        return .{ .vt = vt, .dev = dev };
    }
    fn deinit(self: *Harness) void {
        self.dev.deinit();
        self.vt.deinit();
        test_cap.assertNoValidationErrors();
    }

    fn upload(self: *Harness, data: []const f32) !vk.Buffer {
        var buf = try vk.createBuffer(&self.dev, data.len * 4);
        errdefer buf.deinit(&self.dev);
        try vk.writeBuffer(&self.dev, &buf, std.mem.sliceAsBytes(data));
        return buf;
    }
};

// Convert a comptime array of u32 bit patterns (the golden encoding) to runtime f32.
fn bits(comptime arr: anytype) [arr.len]f32 {
    var out: [arr.len]f32 = undefined;
    for (arr, 0..) |b, i| out[i] = @bitCast(b);
    return out;
}

// Mixed abs+rel tolerance (testdata.expectClose): GPU accumulation order won't hold a tight 1e-5
// absolute bound that the CPU did, but real kernel bugs diverge by orders of magnitude more.
const kernel_atol: f32 = 1e-3;

test "gpu matmul golden" {
    var h = Harness.init() catch |e| return e;
    defer h.deinit();

    const x = bits(golden.matmul.x);
    const w = bits(golden.matmul.w);
    const b = bits(golden.matmul.b);
    const expected = bits(golden.matmul.y);
    const m = golden.matmul.m;
    const k = golden.matmul.k;
    const n = golden.matmul.n;

    var pipe = try vk.createComputePipeline(&h.dev, &spv_matmul, k_matmul.binding_count, k_matmul.push_constant_size, k_matmul.entry_point);
    defer pipe.deinit(&h.dev);

    var bx = try h.upload(&x);
    defer bx.deinit(&h.dev);
    var bw = try h.upload(&w);
    defer bw.deinit(&h.dev);
    var bb = try h.upload(&b);
    defer bb.deinit(&h.dev);
    var by = try vk.createBuffer(&h.dev, expected.len * 4);
    defer by.deinit(&h.dev);

    const binds = [_]vk.BufferBinding{
        .{ .buffer = bx.buffer, .range = x.len * 4 },
        .{ .buffer = bw.buffer, .range = w.len * 4 },
        .{ .buffer = bb.buffer, .range = b.len * 4 },
        .{ .buffer = by.buffer, .range = expected.len * 4 },
    };
    const pc = k_matmul.PushConstants{ .M = m, .K = k, .N = n };
    vk.dispatch(&h.dev, &pipe, &binds, std.mem.asBytes(&pc), groups1d(m * n));

    var got: [golden.matmul.m * golden.matmul.n]f32 = undefined;
    try vk.readBuffer(&h.dev, &by, std.mem.sliceAsBytes(&got));
    try testdata.expectClose("matmul", &expected, &got, kernel_atol);
}

test "gpu matmulBT hand-computed" {
    var h = Harness.init() catch |e| return e;
    defer h.deinit();

    // Same case as op.zig: x:[2,3], W:[2,3] (Wᵀ is [3,2]); y = x @ Wᵀ.
    const x = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const w = [_]f32{ 1, 0, -1, 2, 2, 2 };
    const expected = [_]f32{ -2, 12, -2, 30 };

    var pipe = try vk.createComputePipeline(&h.dev, &spv_matmul_bt, k_matmul_bt.binding_count, k_matmul_bt.push_constant_size, k_matmul_bt.entry_point);
    defer pipe.deinit(&h.dev);

    var bx = try h.upload(&x);
    defer bx.deinit(&h.dev);
    var bw = try h.upload(&w);
    defer bw.deinit(&h.dev);
    var by = try vk.createBuffer(&h.dev, expected.len * 4);
    defer by.deinit(&h.dev);

    const binds = [_]vk.BufferBinding{
        .{ .buffer = bx.buffer, .range = x.len * 4 },
        .{ .buffer = bw.buffer, .range = w.len * 4 },
        .{ .buffer = by.buffer, .range = expected.len * 4 },
    };
    const pc = k_matmul_bt.PushConstants{ .M = 2, .K = 3, .N = 2 };
    vk.dispatch(&h.dev, &pipe, &binds, std.mem.asBytes(&pc), groups1d(2 * 2));

    var got: [4]f32 = undefined;
    try vk.readBuffer(&h.dev, &by, std.mem.sliceAsBytes(&got));
    try testdata.expectClose("matmulBT", &expected, &got, kernel_atol);
}

test "gpu layernorm golden" {
    var h = Harness.init() catch |e| return e;
    defer h.deinit();

    const x = bits(golden.layernorm.x);
    const gamma = bits(golden.layernorm.gamma);
    const beta = bits(golden.layernorm.beta);
    const expected = bits(golden.layernorm.out);
    const eps: f32 = @bitCast(golden.layernorm.eps_bits);
    const rows = golden.layernorm.rows;
    const cols = golden.layernorm.cols;

    var pipe = try vk.createComputePipeline(&h.dev, &spv_layernorm, k_layernorm.binding_count, k_layernorm.push_constant_size, k_layernorm.entry_point);
    defer pipe.deinit(&h.dev);

    var bx = try h.upload(&x);
    defer bx.deinit(&h.dev);
    var bg = try h.upload(&gamma);
    defer bg.deinit(&h.dev);
    var bbeta = try h.upload(&beta);
    defer bbeta.deinit(&h.dev);
    var bo = try vk.createBuffer(&h.dev, expected.len * 4);
    defer bo.deinit(&h.dev);

    const binds = [_]vk.BufferBinding{
        .{ .buffer = bx.buffer, .range = x.len * 4 },
        .{ .buffer = bg.buffer, .range = gamma.len * 4 },
        .{ .buffer = bbeta.buffer, .range = beta.len * 4 },
        .{ .buffer = bo.buffer, .range = expected.len * 4 },
    };
    const pc = k_layernorm.PushConstants{ .rows = rows, .cols = cols, .eps = eps };
    vk.dispatch(&h.dev, &pipe, &binds, std.mem.asBytes(&pc), groups1d(rows));

    var got: [golden.layernorm.rows * golden.layernorm.cols]f32 = undefined;
    try vk.readBuffer(&h.dev, &bo, std.mem.sliceAsBytes(&got));
    try testdata.expectClose("layernorm", &expected, &got, kernel_atol);
}

test "gpu softmax golden" {
    var h = Harness.init() catch |e| return e;
    defer h.deinit();

    const x = bits(golden.softmax.x);
    const expected = bits(golden.softmax.out);
    const rows = golden.softmax.rows;
    const cols = golden.softmax.cols;

    var pipe = try vk.createComputePipeline(&h.dev, &spv_softmax, k_softmax.binding_count, k_softmax.push_constant_size, k_softmax.entry_point);
    defer pipe.deinit(&h.dev);

    var bx = try h.upload(&x);
    defer bx.deinit(&h.dev);
    var bo = try vk.createBuffer(&h.dev, expected.len * 4);
    defer bo.deinit(&h.dev);

    const binds = [_]vk.BufferBinding{
        .{ .buffer = bx.buffer, .range = x.len * 4 },
        .{ .buffer = bo.buffer, .range = expected.len * 4 },
    };
    const pc = k_softmax.PushConstants{ .rows = rows, .cols = cols };
    vk.dispatch(&h.dev, &pipe, &binds, std.mem.asBytes(&pc), groups1d(rows));

    var got: [golden.softmax.rows * golden.softmax.cols]f32 = undefined;
    try vk.readBuffer(&h.dev, &bo, std.mem.sliceAsBytes(&got));
    try testdata.expectClose("softmax", &expected, &got, kernel_atol);
}

test "gpu gelu golden" {
    var h = Harness.init() catch |e| return e;
    defer h.deinit();

    const x = bits(golden.gelu.x);
    const expected = bits(golden.gelu.out);

    var pipe = try vk.createComputePipeline(&h.dev, &spv_gelu, k_gelu.binding_count, k_gelu.push_constant_size, k_gelu.entry_point);
    defer pipe.deinit(&h.dev);

    var bx = try h.upload(&x);
    defer bx.deinit(&h.dev);
    var bo = try vk.createBuffer(&h.dev, expected.len * 4);
    defer bo.deinit(&h.dev);

    const binds = [_]vk.BufferBinding{
        .{ .buffer = bx.buffer, .range = x.len * 4 },
        .{ .buffer = bo.buffer, .range = expected.len * 4 },
    };
    const pc = k_gelu.PushConstants{ .n = golden.gelu.len };
    vk.dispatch(&h.dev, &pipe, &binds, std.mem.asBytes(&pc), groups1d(golden.gelu.len));

    var got: [golden.gelu.len]f32 = undefined;
    try vk.readBuffer(&h.dev, &bo, std.mem.sliceAsBytes(&got));
    try testdata.expectClose("gelu", &expected, &got, kernel_atol);
}

// Hardcoded prompt "Hello, I am" — same ids the Python oracle uses.
const PROMPT = [_]u32{ 15496, 11, 314, 1101 };

fn argmaxLastRow(logits: []align(1) const f32, s: usize, vocab: usize) usize {
    const row = logits[(s - 1) * vocab ..][0..vocab];
    var best: usize = 0;
    for (row, 0..) |v, i| if (v > row[best]) {
        best = i;
    };
    return best;
}

test "gpu forward bisection vs activation goldens" {
    const io = std.testing.io;
    const alloc = std.testing.allocator;

    const gld = asset.act_goldens orelse {
        std.debug.print("\nactivation goldens missing — run `zig build gen-goldens`\n", .{});
        return error.SkipZigTest;
    };

    var gpu = (Gpu.init(io, alloc, asset.model_safetensors_path, testInstanceOpts()) catch |e| return e) orelse
        return error.SkipZigTest;
    defer gpu.deinit();
    defer test_cap.assertNoValidationErrors();

    const S = PROMPT.len;
    const n_embd = gpu.cfg.n_embd;
    const vocab = gpu.cfg.vocab_size;

    var g_logits = testdata.Golden.fromBytes(gld.logits);
    var g_embed = testdata.Golden.fromBytes(gld.embed);
    var g_l0_ln1 = testdata.Golden.fromBytes(gld.l0_ln1);
    var g_l0_attn = testdata.Golden.fromBytes(gld.l0_attn);
    var g_l0_resid1 = testdata.Golden.fromBytes(gld.l0_resid1);
    var g_l0_mlp = testdata.Golden.fromBytes(gld.l0_mlp);
    var g_l0_out = testdata.Golden.fromBytes(gld.l0_out);
    var g_l5_out = testdata.Golden.fromBytes(gld.l5_out);
    var g_lnf = testdata.Golden.fromBytes(gld.lnf);

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

    try gpu.forward(&PROMPT, logits, .{
        .embed = embed,
        .l0_ln1 = l0_ln1,
        .l0_attn = l0_attn,
        .l0_resid1 = l0_resid1,
        .l0_mlp = l0_mlp,
        .l0_out = l0_out,
        .l5_out = l5_out,
        .lnf = lnf,
    });

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

    const got = argmaxLastRow(logits, S, vocab);
    const want = argmaxLastRow(g_logits.data(), S, vocab);
    try std.testing.expectEqual(want, got);
}
