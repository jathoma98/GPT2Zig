//! By convention, root.zig is the root source file when making a package.
const std = @import("std");
const Io = std.Io;

pub const safetensors = @import("safetensors/safetensors.zig");
pub const tensor = @import("core/tensor.zig");
pub const config = @import("core/config.zig");
pub const op = @import("core/op.zig");
pub const model = @import("core/model.zig");
pub const token = @import("core/token.zig");

test {
    _ = @import("safetensors/safetensors.zig");
    _ = @import("core/tensor.zig");
    _ = @import("core/config.zig");
    _ = @import("core/op.zig");
    _ = @import("core/model.zig");
    _ = @import("core/testdata.zig");
    _ = @import("core/token.zig");
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
