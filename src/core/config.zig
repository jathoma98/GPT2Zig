const std = @import("std");
const assert = std.debug.assert;

pub const Config = struct {
    n_layer: u32, // 12
    n_head: u32, // 12
    n_embd: u32, // 768
    n_ctx: u32, // 1024  (wpe row count / max positions)
    vocab_size: u32, // 50257
    ln_eps: f32, // 1e-5  (layer_norm_epsilon)
    eos_token_id: u32, // 50256

    pub fn fromFile(io: std.Io, path: []const u8) !Config {
        // config.json is ~700 bytes; read into a stack buffer (no mmap needed).
        var io_buf: [512]u8 = undefined;
        var json_buf: [4096]u8 = undefined;
        const file = try std.Io.Dir.cwd().openFile(io, path, .{});
        defer file.close(io);
        const size = (try file.stat(io)).size;
        assert(size <= json_buf.len);
        var reader = file.reader(io, &io_buf);
        try reader.interface.readSliceAll(json_buf[0..size]);

        // Map only the fields we use; parseFromSliceLeaky + ignore_unknown_fields skips
        // the nested objects/arrays (task_specific_params, architectures) and unused scalars.
        const Raw = struct {
            n_layer: u32,
            n_head: u32,
            n_embd: u32,
            n_ctx: u32,
            vocab_size: u32,
            layer_norm_epsilon: f32,
            eos_token_id: u32,
            activation_function: []const u8,
        };
        var fba_buf: [1024]u8 = undefined;
        var fba = std.heap.FixedBufferAllocator.init(&fba_buf);
        const raw = try std.json.parseFromSliceLeaky(
            Raw,
            fba.allocator(),
            json_buf[0..size],
            .{ .ignore_unknown_fields = true },
        );
        // GPT-2 small must be gelu_new (tanh GELU); fail loudly if a mirror differs (see M2).
        assert(std.mem.eql(u8, raw.activation_function, "gelu_new"));

        return .{
            .n_layer = raw.n_layer,
            .n_head = raw.n_head,
            .n_embd = raw.n_embd,
            .n_ctx = raw.n_ctx,
            .vocab_size = raw.vocab_size,
            .ln_eps = raw.layer_norm_epsilon,
            .eos_token_id = raw.eos_token_id,
        };
    }
};

// =================
// === Tests ===

test "config parse bit-equiv" {
    const golden = @import("../generated/safetensors_golden.zig");
    const cfg = try Config.fromFile(std.testing.io, "models/gpt2/config.json");

    try std.testing.expectEqual(golden.config.n_layer, cfg.n_layer);
    try std.testing.expectEqual(golden.config.n_head, cfg.n_head);
    try std.testing.expectEqual(golden.config.n_embd, cfg.n_embd);
    try std.testing.expectEqual(golden.config.n_ctx, cfg.n_ctx);
    try std.testing.expectEqual(golden.config.vocab_size, cfg.vocab_size);
    try std.testing.expectEqual(golden.config.eos_token_id, cfg.eos_token_id);
    // bit-compare validates correct 1e-5 float parsing
    try std.testing.expectEqual(golden.config.ln_eps_bits, @as(u32, @bitCast(cfg.ln_eps)));
}
