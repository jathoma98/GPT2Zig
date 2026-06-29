//! By convention, root.zig is the root source file when making a package.
const std = @import("std");
const Io = std.Io;

pub const safetensors = @import("safetensors/safetensors.zig");
pub const tensor = @import("core/tensor.zig");
pub const config = @import("core/config.zig");
pub const gpu = @import("core/gpu.zig");
pub const token = @import("core/token.zig");
pub const asset = @import("core/asset/asset.zig");
pub const platform = @import("core/platform/platform.zig");
pub const vk = @import("vk");

pub const dist = struct {
    pub const runconfig = @import("dist/runconfig.zig");
    pub const master = @import("dist/master.zig");
    pub const slave = @import("dist/slave.zig");
    pub const wire = @import("dist/wire.zig");
    pub const partition = @import("dist/partition.zig");
};

test {
    _ = @import("safetensors/safetensors.zig");
    _ = @import("core/tensor.zig");
    _ = @import("core/config.zig");
    _ = @import("core/gpu.zig");
    _ = @import("core/testdata.zig");
    _ = @import("core/token.zig");
    _ = @import("core/asset/asset.zig");
    _ = @import("core/platform/platform.zig");
    _ = @import("dist/partition.zig");
    _ = @import("dist/wire.zig");
    _ = @import("dist/runconfig.zig");
    _ = @import("dist/master.zig");
    _ = @import("dist/slave.zig");
}

pub fn printAnotherMessage(writer: *Io.Writer) Io.Writer.Error!void {
    try writer.print("Run `zig build test` to run the tests.\n", .{});
}

test "tokenizer golden cases" {
    const g = @import("generated/tokenizer_golden.zig");
    try std.testing.expect(g.cases.len >= 12);
    try std.testing.expectEqualStrings("hello world", g.cases[0].in);
    // "<|endoftext|>" must encode atomically to the single EOS token 50256 — find that case.
    var saw_eos = false;
    for (g.cases) |c| {
        if (std.mem.eql(u8, c.in, "<|endoftext|>")) {
            try std.testing.expectEqual(@as(usize, 1), c.out.len);
            try std.testing.expectEqual(@as(u32, 50256), c.out[0]);
            saw_eos = true;
        }
    }
    try std.testing.expect(saw_eos);
}
