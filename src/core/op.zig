//! M2 math kernels: the four numerical primitives every transformer layer is built from.
//! Naive, single-threaded, allocation-free — caller owns every buffer and passes the output
//! slice. This is the known-correct baseline the SIMD/threaded versions (M7/M8) diff against.
const std = @import("std");
const assert = std.debug.assert;

// =================
// === matmul ===
//
// y = x @ W + b.  x:[M,K] row-major, W:[K,N] row-major, b:[N], y:[M,N].
// W is indexed [in, out] — GPT-2's Conv1D weight is already stored in this orientation, so no
// transpose. Naive ijk; acc seeded with the bias.
pub fn matmul(x: []const f32, w: []const f32, b: []const f32, y: []f32, m: usize, k: usize, n: usize) void {
    assert(x.len == m * k);
    assert(w.len == k * n);
    assert(b.len == n);
    assert(y.len == m * n);

    for (0..m) |row| {
        for (0..n) |col| {
            var acc: f32 = b[col];
            for (0..k) |i| {
                acc += x[row * k + i] * w[i * n + col];
            }
            y[row * n + col] = acc;
        }
    }
}

// =================
// === layernorm ===
//
// (row - mean) / sqrt(var + eps) * gamma + beta. Per-row; variance is BIASED (/D, not /(D-1)).
// Not RMSNorm: the mean is subtracted and beta is added. mean/var are read from `row` before any
// write, so `out` may alias `row`.
pub fn layernorm(row: []const f32, gamma: []const f32, beta: []const f32, eps: f32, out: []f32) void {
    const d = row.len;
    assert(d > 0);
    assert(gamma.len == d and beta.len == d and out.len == d);

    var sum: f32 = 0;
    for (row) |v| sum += v;
    const mean = sum / @as(f32, @floatFromInt(d));

    var sq: f32 = 0;
    for (row) |v| {
        const diff = v - mean;
        sq += diff * diff;
    }
    const variance = sq / @as(f32, @floatFromInt(d));
    const inv_std = 1.0 / @sqrt(variance + eps);

    for (0..d) |i| {
        out[i] = (row[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

// =================
// === softmax ===
//
// Row-wise softmax. The max-subtraction is numerical hygiene — exp(large) overflows to inf;
// subtracting the row max makes the largest exponent exp(0)=1 with identical math. Reads `row`
// for the max before writing, so `out` may alias `row`.
pub fn softmax(row: []const f32, out: []f32) void {
    const d = row.len;
    assert(d > 0);
    assert(out.len == d);

    var max: f32 = row[0];
    for (row[1..]) |v| max = @max(max, v);

    var sum: f32 = 0;
    for (0..d) |i| {
        const e = @exp(row[i] - max);
        out[i] = e;
        sum += e;
    }

    const inv = 1.0 / sum;
    for (out) |*v| v.* *= inv;
}

// =================
// === gelu ===
//
// GELU tanh approximation (gelu_new) — the flavor GPT-2 was trained with. Exact erf-GELU
// produces tiny logit drift that fails tight parity. Elementwise; `out` may alias `x`.
const gelu_c: f32 = 0.7978845608028654; // sqrt(2/pi)
const gelu_a: f32 = 0.044715;

pub fn gelu(x: []const f32, out: []f32) void {
    assert(out.len == x.len);
    for (0..x.len) |i| {
        const v = x[i];
        const inner = gelu_c * (v + gelu_a * v * v * v);
        out[i] = 0.5 * v * (1.0 + std.math.tanh(inner));
    }
}

// =================
// === Tests ===

const golden = @import("../generated/kernel_golden.zig");
const tol: f32 = 1e-5;

fn bits(comptime arr: anytype) [arr.len]f32 {
    var out: [arr.len]f32 = undefined;
    for (arr, 0..) |b, i| out[i] = @bitCast(b);
    return out;
}

test "matmul golden" {
    const x = bits(golden.matmul.x);
    const w = bits(golden.matmul.w);
    const b = bits(golden.matmul.b);
    const expected = bits(golden.matmul.y);

    var y: [golden.matmul.m * golden.matmul.n]f32 = undefined;
    matmul(&x, &w, &b, &y, golden.matmul.m, golden.matmul.k, golden.matmul.n);

    for (expected, y) |e, a| try std.testing.expectApproxEqAbs(e, a, tol);
}

test "layernorm golden" {
    const x = bits(golden.layernorm.x);
    const gamma = bits(golden.layernorm.gamma);
    const beta = bits(golden.layernorm.beta);
    const expected = bits(golden.layernorm.out);
    const eps: f32 = @bitCast(golden.layernorm.eps_bits);
    const cols = golden.layernorm.cols;

    var out: [cols]f32 = undefined;
    for (0..golden.layernorm.rows) |r| {
        const lo = r * cols;
        layernorm(x[lo .. lo + cols], &gamma, &beta, eps, &out);
        for (expected[lo .. lo + cols], out) |e, a| try std.testing.expectApproxEqAbs(e, a, tol);
    }
}

test "softmax golden" {
    const x = bits(golden.softmax.x);
    const expected = bits(golden.softmax.out);
    const cols = golden.softmax.cols;

    var out: [cols]f32 = undefined;
    for (0..golden.softmax.rows) |r| {
        const lo = r * cols;
        softmax(x[lo .. lo + cols], &out);
        for (expected[lo .. lo + cols], out) |e, a| try std.testing.expectApproxEqAbs(e, a, tol);
    }
}

test "gelu golden" {
    const x = bits(golden.gelu.x);
    const expected = bits(golden.gelu.out);

    var out: [golden.gelu.len]f32 = undefined;
    gelu(&x, &out);

    for (expected, out) |e, a| try std.testing.expectApproxEqAbs(e, a, tol);
}

// Hand-computed cases catch sign/axis bugs a random golden could mask.

test "softmax uniform" {
    var out: [2]f32 = undefined;
    softmax(&.{ 0, 0 }, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), out[0], tol);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), out[1], tol);
}

test "layernorm of constant row is beta" {
    // A constant row has zero variance → the normalized term is ~0, so out == beta.
    const row = [_]f32{ 3, 3, 3, 3 };
    const gamma = [_]f32{ 2, 2, 2, 2 };
    const beta = [_]f32{ -1, 0, 1, 2 };
    var out: [4]f32 = undefined;
    layernorm(&row, &gamma, &beta, 1e-5, &out);
    for (beta, out) |e, a| try std.testing.expectApproxEqAbs(e, a, 1e-3);
}

test "out may alias input" {
    var buf = [_]f32{ 1, 2, 3, 4 };
    softmax(&buf, &buf);
    var sum: f32 = 0;
    for (buf) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, tol);
}
