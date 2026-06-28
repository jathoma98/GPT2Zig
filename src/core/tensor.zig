const std = @import("std");
const assert = std.debug.assert;
const safetensors = @import("../safetensors/safetensors.zig");

// The Tensor API is consolidated on row-major: dims are [rows, cols] and at(r, c)
// is row-major logical. Layout lets the PHYSICAL storage be either orientation —
// at() switches the index math — so e.g. wte.weight can be viewed as wteᵀ zero-copy
// (a col_major view) for the M3 tied-output projection.
pub const Layout = enum { row_major, col_major };

pub const Tensor = struct {
    // Zero-copy view into the mmap. align(1): data_base (8 + header_len) is not 4-byte
    // aligned, so element addresses are misaligned; indexing lowers to unaligned loads
    // (correct everywhere, free on x86_64/ARM64).
    data: []align(1) const f32,
    rows: u32,
    cols: u32,
    layout: Layout,

    pub fn view(data: []align(1) const f32, rows: u32, cols: u32, layout: Layout) Tensor {
        assert(data.len == @as(usize, rows) * cols);
        return .{ .data = data, .rows = rows, .cols = cols, .layout = layout };
    }

    // GPT-2 tensors are rank 1 (bias [N]) or rank 2 (weight [R, C]); safetensors stores
    // them C-contiguous → row_major. Rank-1 is modeled as a [1, N] row vector.
    pub fn fromHandle(st: *const safetensors.SafeTensors, h: safetensors.Handle) Tensor {
        const m = st.meta(h);
        assert(m.rank == 1 or m.rank == 2);
        const rows: u32 = if (m.rank == 2) m.shape[0] else 1;
        const cols: u32 = if (m.rank == 2) m.shape[1] else m.shape[0];
        return view(bytesAsF32(st.data_bytes(h)), rows, cols, .row_major);
    }

    pub fn at(self: *const Tensor, r: u32, c: u32) f32 {
        assert(r < self.rows and c < self.cols);
        const idx = switch (self.layout) {
            .row_major => @as(usize, r) * self.cols + c,
            .col_major => @as(usize, c) * self.rows + r,
        };
        return self.data[idx];
    }
};

pub fn bytesAsF32(bytes: []align(1) const u8) []align(1) const f32 {
    assert(bytes.len % 4 == 0);
    const ptr: [*]align(1) const f32 = @ptrCast(bytes.ptr);
    return ptr[0 .. bytes.len / 4];
}

// =================
// === Tests ===

const MODEL = "models/gpt2/model.safetensors";

test "tensor row-major bit-equiv" {
    const golden = @import("../generated/safetensors_golden.zig");
    var st = try safetensors.SafeTensors.init(std.testing.io, MODEL);
    defer st.deinit();

    for (golden.element_samples) |s| {
        const t = Tensor.fromHandle(&st, st.find(s.name).?);
        const actual_bits: u32 = @bitCast(t.at(s.r, s.c));
        try std.testing.expectEqual(s.bits, actual_bits);
    }
}

test "tensor col-major transpose view" {
    const golden = @import("../generated/safetensors_golden.zig");
    var st = try safetensors.SafeTensors.init(std.testing.io, MODEL);
    defer st.deinit();

    // wte.weight [50257, 768] row-major, viewed as wteᵀ [768, 50257] col-major over
    // the SAME bytes. at(k, n) must equal wte[n, k] with no transpose copy.
    const wte = Tensor.fromHandle(&st, st.find("wte.weight").?);
    const wte_t = Tensor.view(wte.data, 768, 50257, .col_major);

    for (golden.wte_t_samples) |s| {
        const actual_bits: u32 = @bitCast(wte_t.at(s.r, s.c));
        try std.testing.expectEqual(s.bits, actual_bits);
    }
}

test "tensor layout index math" {
    // Same 6 floats, indexed two ways. Hand-computed expectations catch axis-swap
    // bugs a random golden could mask.
    const buf = [6]f32{ 0, 1, 2, 3, 4, 5 };
    const data: []align(1) const f32 = &buf;

    // row_major [2, 3]: at(r, c) = buf[r*3 + c]
    const rm = Tensor.view(data, 2, 3, .row_major);
    try std.testing.expectEqual(@as(f32, 0), rm.at(0, 0));
    try std.testing.expectEqual(@as(f32, 2), rm.at(0, 2));
    try std.testing.expectEqual(@as(f32, 3), rm.at(1, 0));
    try std.testing.expectEqual(@as(f32, 5), rm.at(1, 2));

    // col_major [2, 3]: at(r, c) = buf[c*2 + r]
    const cm = Tensor.view(data, 2, 3, .col_major);
    try std.testing.expectEqual(@as(f32, 0), cm.at(0, 0));
    try std.testing.expectEqual(@as(f32, 1), cm.at(1, 0));
    try std.testing.expectEqual(@as(f32, 2), cm.at(0, 1));
    try std.testing.expectEqual(@as(f32, 5), cm.at(1, 2));
}
