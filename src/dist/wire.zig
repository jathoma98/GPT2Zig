//! The model-parallel wire protocol. Deliberately boring: a fixed length-prefixed header followed
//! by a raw payload. The residual stream is a contiguous []f32 and every target we care about is
//! little-endian IEEE-754, so the payload is the bytes of the slice verbatim — no serialization
//! library, no per-element encoding. Helpers operate on the standard Io.Reader/Io.Writer that
//! std.Io.net.Stream exposes, so the same code runs over a TCP socket or an in-memory buffer (test).
const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;
const Io = std.Io;

comptime {
    // A big-endian target would silently corrupt the f32 payload. Fail loudly at compile time.
    assert(builtin.cpu.arch.endian() == .little);
}

// One fixed port for the whole fleet. The master binds it on its chosen interface (loopback or all
// LAN, per the config's listenAddr); every slave dials this same port on the master's address.
pub const PORT: u16 = 9876;

pub const FrameKind = enum(u8) {
    handshake, // payload: Handshake — master → slave, once, at bring-up
    activations, // payload: seq_len * n_embd f32 — the residual stream at a stage boundary
    _,
};

pub const Header = extern struct {
    kind: FrameKind,
    _pad: [3]u8 = .{ 0, 0, 0 }, // keep seq_len u32-aligned; extern layout is fixed across the wire
    seq_len: u32,
    payload_bytes: u32,
};

pub const Handshake = extern struct {
    layer_lo: u32,
    layer_hi: u32,
};

// =================
// === Send ===

fn writeHeader(w: *Io.Writer, h: Header) Io.Writer.Error!void {
    try w.writeAll(std.mem.asBytes(&h));
}

pub fn sendHandshake(w: *Io.Writer, hs: Handshake) Io.Writer.Error!void {
    try writeHeader(w, .{ .kind = .handshake, .seq_len = 0, .payload_bytes = @sizeOf(Handshake) });
    try w.writeAll(std.mem.asBytes(&hs));
    try w.flush();
}

pub fn sendActivations(w: *Io.Writer, x: []const f32, seq_len: u32) Io.Writer.Error!void {
    const payload = std.mem.sliceAsBytes(x);
    try writeHeader(w, .{ .kind = .activations, .seq_len = seq_len, .payload_bytes = @intCast(payload.len) });
    try w.writeAll(payload);
    try w.flush();
}

// =================
// === Receive ===

fn readHeader(r: *Io.Reader) Io.Reader.Error!Header {
    var h: Header = undefined;
    try r.readSliceAll(std.mem.asBytes(&h));
    return h;
}

pub fn recvHandshake(r: *Io.Reader) Io.Reader.Error!Handshake {
    const h = try readHeader(r);
    assert(h.kind == .handshake);
    assert(h.payload_bytes == @sizeOf(Handshake));
    var hs: Handshake = undefined;
    try r.readSliceAll(std.mem.asBytes(&hs));
    return hs;
}

// Reads one activations frame into x_buf (sized at the n_ctx upper bound) and returns seq_len.
// n_embd is the receiver's own model dim; payload_bytes asserts the sender agreed (same model).
// Returns error.EndOfStream when the peer closes the stream — the slave's clean-shutdown signal.
pub fn recvActivations(r: *Io.Reader, x_buf: []f32, n_embd: u32) Io.Reader.Error!u32 {
    const h = try readHeader(r);
    assert(h.kind == .activations);
    const n_floats = @as(usize, h.seq_len) * n_embd;
    assert(h.payload_bytes == n_floats * @sizeOf(f32));
    assert(n_floats <= x_buf.len);
    try r.readSliceAll(std.mem.sliceAsBytes(x_buf[0..n_floats]));
    return h.seq_len;
}

// =================
// === Tests ===

test "activations frame round-trips over an in-memory buffer" {
    const n_embd: u32 = 4;
    const seq_len: u32 = 3;
    const sent = [_]f32{ 1.0, -2.5, 3.25, 0.0, 4.0, 5.5, -6.0, 7.0, 8.0, 9.5, 10.0, -11.0 };

    var buf: [256]u8 = undefined;
    var w = Io.Writer.fixed(&buf);
    try sendActivations(&w, &sent, seq_len);

    var r = Io.Reader.fixed(buf[0..w.end]);
    var recv: [16]f32 = undefined;
    const got_len = try recvActivations(&r, &recv, n_embd);

    try std.testing.expectEqual(seq_len, got_len);
    try std.testing.expectEqualSlices(f32, &sent, recv[0 .. seq_len * n_embd]);
}

test "handshake frame round-trips" {
    var buf: [64]u8 = undefined;
    var w = Io.Writer.fixed(&buf);
    try sendHandshake(&w, .{ .layer_lo = 6, .layer_hi = 12 });

    var r = Io.Reader.fixed(buf[0..w.end]);
    const hs = try recvHandshake(&r);
    try std.testing.expectEqual(@as(u32, 6), hs.layer_lo);
    try std.testing.expectEqual(@as(u32, 12), hs.layer_hi);
}
