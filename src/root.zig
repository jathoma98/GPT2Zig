//! By convention, root.zig is the root source file when making a package.
const std = @import("std");
const Io = std.Io;

/// This is a documentation comment to explain the `printAnotherMessage` function below.
///
/// Accepting an `Io.Writer` instance is a handy way to write reusable code.
pub fn printAnotherMessage(writer: *Io.Writer) Io.Writer.Error!void {
    try writer.print("Run `zig build test` to run the tests.\n", .{});
}

pub fn add(a: i32, b: i32) i32 {
    return a + b;
}

test "basic add functionality" {
    try std.testing.expect(add(3, 7) == 10);
}

test "tokenizer golden cases" {
    const g = @import("tokenizer_golden");
    try std.testing.expectEqual(@as(usize, 12), g.cases.len);
    try std.testing.expectEqualStrings("hello world", g.cases[0].in);
    // "<|endoftext|>" must encode atomically to the single EOS token 50256
    try std.testing.expectEqual(@as(usize, 1), g.cases[11].out.len);
    try std.testing.expectEqual(@as(u32, 50256), g.cases[11].out[0]);
}
