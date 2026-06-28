//! Reader for the self-describing golden files the M3 activation oracle dumps (embedded via
//! asset.zig). Format (little-endian): [u32 ndim][u32 dim0]…[u32 dim_{ndim-1}][f32 data…]. The
//! data section begins at byte 4*(1+ndim); f32 payload is read with align(1) loads, so no
//! alignment guarantee on the backing bytes is required.
const std = @import("std");
const assert = std.debug.assert;
const tensor = @import("tensor.zig");

const MAX_DIMS = 2;

pub const Golden = struct {
    bytes: []align(1) const u8,
    ndim: u32,
    dims: [MAX_DIMS]u32,
    data_offset: u32,

    // Bytes are embedded (see asset.zig); parse the self-describing header straight from the slice.
    pub fn fromBytes(raw: []align(1) const u8) Golden {
        assert(raw.len >= 8); // at least ndim + one dim

        const ndim = std.mem.readInt(u32, raw[0..4], .little);
        assert(ndim >= 1 and ndim <= MAX_DIMS);

        var dims: [MAX_DIMS]u32 = .{ 0, 0 };
        var count: u64 = 1;
        for (0..ndim) |i| {
            const off = 4 + i * 4;
            dims[i] = std.mem.readInt(u32, raw[off..][0..4], .little);
            count *= dims[i];
        }

        const data_offset: u32 = @intCast(4 * (1 + ndim));
        // Byte length must match the declared shape exactly — catches a python/zig shape mismatch.
        assert(raw.len == data_offset + count * 4);

        return .{ .bytes = raw, .ndim = ndim, .dims = dims, .data_offset = data_offset };
    }

    pub fn deinit(self: *Golden) void {
        _ = self; // embedded rodata — nothing to free (kept so callsites can `defer g.deinit()`)
    }

    pub fn data(self: *const Golden) []align(1) const f32 {
        return tensor.bytesAsF32(self.bytes[self.data_offset..]);
    }

    pub fn n_elements(self: *const Golden) usize {
        var n: usize = 1;
        for (0..self.ndim) |i| n *= self.dims[i];
        return n;
    }
};

// Compare an actual activation against a golden, reporting the worst element before asserting so a
// failing tap tells you HOW far off it is (and where), not just that it diverged.
pub fn expectClose(label: []const u8, golden: []align(1) const f32, actual: []const f32, tol: f32) !void {
    assert(golden.len == actual.len);
    var max_diff: f32 = 0;
    var max_idx: usize = 0;
    for (golden, actual, 0..) |g, a, i| {
        const d = @abs(g - a);
        if (d > max_diff) {
            max_diff = d;
            max_idx = i;
        }
    }
    _ = label;
    // std.debug.print("  {s}: max-abs-diff {e:.4} at [{d}] (golden {e:.6} vs actual {e:.6})\n", .{
    //     label, max_diff, max_idx, golden[max_idx], actual[max_idx],
    // });
    if (max_diff > tol) return error.GoldenMismatch;
}
