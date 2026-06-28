//! Layer partitioning for model-parallel inference. The master is participant 0 (it owns the front
//! layer chunk plus HEAD/TAIL); each slave is participant 1..N in pipeline order. The split is a
//! pure function of (participant index, participant count, layer count) so the master can compute
//! every stage's range and hand each slave its [lo, hi) in the handshake.
const std = @import("std");
const assert = std.debug.assert;

pub const Range = struct { lo: usize, hi: usize };

// Even contiguous split of n_layer across `participants` stages. The first `n_layer % participants`
// stages take one extra layer, so the ranges tile [0, n_layer) with no gap or overlap.
pub fn layerRange(idx: usize, participants: usize, n_layer: usize) Range {
    assert(participants > 0);
    assert(idx < participants);
    const base = n_layer / participants;
    const rem = n_layer % participants;
    const lo = idx * base + @min(idx, rem);
    const extra: usize = if (idx < rem) 1 else 0;
    const hi = lo + base + extra;
    assert(hi <= n_layer);
    return .{ .lo = lo, .hi = hi };
}

// =================
// === Tests ===

test "layerRange tiles all layers contiguously" {
    const cases = [_]struct { participants: usize, n_layer: usize }{
        .{ .participants = 1, .n_layer = 12 },
        .{ .participants = 2, .n_layer = 12 },
        .{ .participants = 3, .n_layer = 12 },
        .{ .participants = 5, .n_layer = 12 }, // uneven: sizes 3,3,2,2,2
    };
    for (cases) |c| {
        var prev_hi: usize = 0;
        for (0..c.participants) |idx| {
            const r = layerRange(idx, c.participants, c.n_layer);
            try std.testing.expectEqual(prev_hi, r.lo); // contiguous: no gap, no overlap
            try std.testing.expect(r.hi >= r.lo);
            prev_hi = r.hi;
        }
        try std.testing.expectEqual(c.n_layer, prev_hi); // full coverage
    }
}

test "layerRange 2-way matches the PoC split" {
    try std.testing.expectEqual(Range{ .lo = 0, .hi = 6 }, layerRange(0, 2, 12)); // master front
    try std.testing.expectEqual(Range{ .lo = 6, .hi = 12 }, layerRange(1, 2, 12)); // slave
}
