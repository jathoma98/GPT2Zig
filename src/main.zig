const std = @import("std");
const Io = std.Io;
const assert = std.debug.assert;

const GPT2Zig = @import("GPT2Zig");
const Model = GPT2Zig.model.Model;
const Config = GPT2Zig.config.Config;
const SafeTensors = GPT2Zig.safetensors.SafeTensors;
const Tokenizer = GPT2Zig.token.Tokenizer;

const MODEL = "models/gpt2/model.safetensors";
const CONFIG = "models/gpt2/config.json";
const BPE_BIN = GPT2Zig.token.BPE_BIN;

const DEFAULT_PROMPT = "Hello, I am";
// Cap generated tokens. Kept modest because there's no KV cache yet (M6): each step recomputes the
// whole sequence with naive matmul, so wall-clock grows fast. Also bounded by n_ctx below.
const MAX_NEW = 40;

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const gpa = init.gpa;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const out = &stdout_file_writer.interface;

    // === prompt: argv[1..] joined with single spaces, one token sequence ===
    var prompt_buf = std.ArrayList(u8).empty;
    defer prompt_buf.deinit(gpa);
    var args = init.minimal.args.iterate();
    _ = args.skip(); // argv[0]
    while (args.next()) |a| {
        if (prompt_buf.items.len != 0) try prompt_buf.append(gpa, ' ');
        try prompt_buf.appendSlice(gpa, a);
    }
    const prompt: []const u8 = if (prompt_buf.items.len == 0) DEFAULT_PROMPT else prompt_buf.items;

    // === load model + tokenizer ===
    var st = try SafeTensors.init(io, MODEL);
    const cfg = try Config.fromFile(io, CONFIG);
    var model = try Model.init(&st, cfg);
    st.deinit(); // weights copied; mmap no longer referenced
    defer model.deinit();

    var tok = try Tokenizer.init(io, BPE_BIN);
    defer tok.deinit();

    const n_ctx: usize = cfg.n_ctx;
    const vocab: usize = cfg.vocab_size;

    const ids = try gpa.alloc(u32, n_ctx);
    defer gpa.free(ids);
    var len = tok.encode(prompt, ids);
    assert(len > 0 and len <= n_ctx);

    // One logits buffer at the upper bound; each step uses the live [len*vocab] prefix.
    const logits = try std.heap.page_allocator.alloc(f32, n_ctx * vocab);
    defer std.heap.page_allocator.free(logits);

    try out.writeAll(prompt);
    try out.flush();

    // === greedy generation loop (no KV cache yet — recomputes the whole sequence each step) ===
    var produced: usize = 0;
    while (produced < MAX_NEW and len < n_ctx) : (produced += 1) {
        model.forward(ids[0..len], logits[0 .. len * vocab], .{});

        const last = logits[(len - 1) * vocab ..][0..vocab];
        var next: usize = 0;
        for (last, 0..) |v, i| {
            if (v > last[next]) next = i;
        }
        if (next == cfg.eos_token_id) break;

        ids[len] = @intCast(next);
        len += 1;
        try out.writeAll(tok.decodeToken(@intCast(next)));
        try out.flush();
    }

    try out.writeByte('\n');
    try out.flush();
}
