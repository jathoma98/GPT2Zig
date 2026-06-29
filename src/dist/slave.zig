//! Slave role: own one contiguous BODY shard of the transformer. A slave is dead simple — it
//! connects to the master, learns its layer range from the handshake, then loops forever:
//! receive the residual stream, run its layers in place, send it back. It never talks to another
//! slave and never samples; the master orchestrates everything.
//!
//! Everything network-related logs under the `.net` scope (grep stderr for `(net)`): bring-up at
//! info, per-frame traffic at debug, failures at err with the failing stage named.
const std = @import("std");
const assert = std.debug.assert;
const net = std.Io.net;
const log = std.log.scoped(.net);

const gpu_mod = @import("../core/gpu.zig");
const Gpu = gpu_mod.Gpu;
const wire = @import("wire.zig");
const partition = @import("partition.zig");

// Per-connection stream buffers. Activations stream through these in chunks; they need not hold a
// whole frame, so a modest size is plenty (max frame is n_ctx*n_embd*4 ≈ 3 MB, sent incrementally).
const CONN_BUF = 64 * 1024;

// The master must be listening before we dial; retry briefly so manual start-order is forgiving.
const CONNECT_ATTEMPTS = 40;
const CONNECT_RETRY_MS = 250;

pub fn run(io: std.Io, gpa: std.mem.Allocator, model_path: []const u8, master_addr: []const u8) !void {
    // === load the full model onto the GPU (we run only our shard, but loading all is simplest) ===
    log.info("slave: loading model from {s}", .{model_path});
    var g = (try Gpu.init(io, gpa, model_path, gpu_mod.defaultInstanceOpts())) orelse {
        log.err("slave: Vulkan unavailable — cannot run GPU inference", .{});
        return error.VulkanUnavailable;
    };
    defer g.deinit();
    const cfg = g.cfg;
    const n_embd: u32 = cfg.n_embd;
    log.info("slave: model loaded (n_layer={d}, n_embd={d})", .{ cfg.n_layer, n_embd });

    // === bring-up: connect (with retry) → directive handshake → ready ===
    const addr = try net.IpAddress.parse(master_addr, wire.PORT);
    log.info("slave: will connect to {s}:{d}", .{ master_addr, wire.PORT });
    var rbuf: [CONN_BUF]u8 = undefined;
    var reader: net.Stream.Reader = undefined; // initialized by the bring-up FSM
    const range = try bringUpSlave(io, addr, &reader, &rbuf);
    defer {
        log.info("slave: closing connection", .{});
        reader.stream.close(io);
    }

    var wbuf: [CONN_BUF]u8 = undefined;
    var writer = net.Stream.Writer.init(reader.stream, io, &wbuf);
    log.info("slave: ready, serving layers [{d}, {d})", .{ range.lo, range.hi });

    // CPU staging for the residual: wire → here → GPU → run → here → wire.
    const x_cpu = try gpa.alloc(f32, cfg.n_ctx * n_embd);
    defer gpa.free(x_cpu);

    // === serve loop: recv x → upload → runLayers → readback → send x. EndOfStream = clean shutdown. ===
    var frame: usize = 0;
    while (true) : (frame += 1) {
        const S = wire.recvActivations(&reader.interface, x_cpu, n_embd) catch |e| switch (e) {
            error.EndOfStream => {
                log.info("slave: master closed the stream after {d} frame(s); shutting down", .{frame});
                break;
            },
            else => {
                log.err("slave: recv failed on frame {d}: {s}", .{ frame, @errorName(e) });
                return e;
            },
        };
        const n: usize = @as(usize, S) * n_embd;
        log.debug("slave: frame {d}: recv S={d}", .{ frame, S });

        try g.writeResidual(S, x_cpu[0..n]); // CPU → GPU
        try g.runLayers(S, range.lo, range.hi); // this shard's layers (on GPU)
        g.deviceWaitIdle(); // finish all kernels before reading the residual back for the reply
        try g.readResidual(S, x_cpu[0..n]); // GPU → CPU

        wire.sendActivations(&writer.interface, x_cpu[0..n], S) catch |e| {
            log.err("slave: send failed on frame {d}: {s}", .{ frame, @errorName(e) });
            return e;
        };
        log.debug("slave: frame {d}: ran layers [{d},{d}), sent S={d}", .{ frame, range.lo, range.hi, S });
    }
}

// =================
// === Bring-up FSM (reduce → decide → transition) ===

const BringUp = union(enum) {
    connect: struct { attempts_left: u32 },
    await_directive: struct { stream: net.Stream },
    ready: partition.Range,
    failed: anyerror,
};

// reader_out is caller-owned; transitionAwaitDirective initializes it over rbuf so the SAME reader
// (with any bytes it buffered past the handshake) is reused by the serve loop.
fn bringUpSlave(io: std.Io, addr: net.IpAddress, reader_out: *net.Stream.Reader, rbuf: []u8) !partition.Range {
    state: switch (BringUp{ .connect = .{ .attempts_left = CONNECT_ATTEMPTS } }) {
        .connect => |p| continue :state transitionConnect(io, addr, p.attempts_left),
        .await_directive => |p| continue :state transitionAwaitDirective(io, p.stream, reader_out, rbuf),
        .ready => |range| return range,
        .failed => |e| {
            log.err("slave: bring-up failed: {s}", .{@errorName(e)});
            return e;
        },
    }
}

fn transitionConnect(io: std.Io, addr: net.IpAddress, attempts_left: u32) BringUp {
    const attempt = CONNECT_ATTEMPTS - attempts_left + 1;
    log.info("slave: connecting to 127.0.0.1:{d} (attempt {d}/{d})", .{ wire.PORT, attempt, CONNECT_ATTEMPTS });
    const stream = addr.connect(io, .{ .mode = .stream }) catch |e| {
        if (attempts_left == 0) {
            log.err("slave: connect gave up after {d} attempts: {s}", .{ CONNECT_ATTEMPTS, @errorName(e) });
            return .{ .failed = e };
        }
        log.warn("slave: connect failed ({s}), retrying in {d}ms ({d} attempt(s) left)", .{ @errorName(e), CONNECT_RETRY_MS, attempts_left });
        io.sleep(std.Io.Duration.fromMilliseconds(CONNECT_RETRY_MS), .awake) catch {};
        return .{ .connect = .{ .attempts_left = attempts_left - 1 } };
    };
    log.info("slave: connected to master", .{});
    return .{ .await_directive = .{ .stream = stream } };
}

fn transitionAwaitDirective(io: std.Io, stream: net.Stream, reader_out: *net.Stream.Reader, rbuf: []u8) BringUp {
    log.info("slave: awaiting layer directive from master", .{});
    reader_out.* = net.Stream.Reader.init(stream, io, rbuf);
    const hs = wire.recvHandshake(&reader_out.interface) catch |e| return .{ .failed = e };
    log.info("slave: received directive → layers [{d}, {d})", .{ hs.layer_lo, hs.layer_hi });
    return .{ .ready = .{ .lo = hs.layer_lo, .hi = hs.layer_hi } };
}
