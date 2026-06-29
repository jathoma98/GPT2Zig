//! Master role: the hub. It owns the HEAD (embedding, `wte`) and the TAIL (final norm + tied logits
//! + sampling), runs the first contiguous layer chunk, and routes the residual stream through each
//! slave in pipeline order — every slave's reply comes back to the master, which forwards it to the
//! next slave (star topology, logical pipeline). `expected_slaves == 0` is pure local mode: the
//! layer range becomes [0, n_layer) and the whole forward pass runs here with no sockets.
//!
//! Everything network-related logs under the `.net` scope (grep stderr for `(net)`): bring-up at
//! info, per-frame traffic at debug, failures at err with the failing stage named.
const std = @import("std");
const assert = std.debug.assert;
const Io = std.Io;
const net = std.Io.net;
const log = std.log.scoped(.net);

const gpu_mod = @import("../core/gpu.zig");
const Gpu = gpu_mod.Gpu;
const Tokenizer = @import("../core/token.zig").Tokenizer;
const asset = @import("../core/asset/asset.zig");
const wire = @import("wire.zig");
const partition = @import("partition.zig");

// Cap generated tokens. Modest because there's no KV cache: each step recomputes the whole sequence.
const MAX_NEW = 40;
const CONN_BUF = 64 * 1024;

const ConnBufs = struct {
    rbuf: [CONN_BUF]u8,
    wbuf: [CONN_BUF]u8,
};

// One connected slave: its layer range plus the persistent reader/writer over its socket.
const SlaveConn = struct {
    range: partition.Range,
    reader: net.Stream.Reader,
    writer: net.Stream.Writer,
};

pub fn run(io: std.Io, gpa: std.mem.Allocator, model_path: []const u8, prompt: []const u8, expected_slaves: u32) !void {
    var stdout_buffer: [4096]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const out = &stdout_file_writer.interface;

    // === load model (onto the GPU) + tokenizer ===
    log.info("master: loading model from {s} ({d} expected slave(s))", .{ model_path, expected_slaves });
    var g = (try Gpu.init(io, gpa, model_path, gpu_mod.defaultInstanceOpts())) orelse {
        log.err("master: Vulkan unavailable — cannot run GPU inference", .{});
        return error.VulkanUnavailable;
    };
    defer g.deinit();
    const cfg = g.cfg;

    var tok = try Tokenizer.fromBytes(asset.bpe);
    defer tok.deinit();
    log.info("master: model loaded (n_layer={d}, n_embd={d})", .{ cfg.n_layer, cfg.n_embd });

    const n_embd: u32 = cfg.n_embd;
    const vocab: usize = cfg.vocab_size;
    const n_ctx: usize = cfg.n_ctx;
    const participants = 1 + @as(usize, expected_slaves);

    // === bring up slaves (FSM): listen → accept N → send each its layer directive ===
    const slaves = try gpa.alloc(SlaveConn, expected_slaves);
    defer gpa.free(slaves);
    const bufs = try gpa.alloc(ConnBufs, expected_slaves);
    defer gpa.free(bufs);
    if (expected_slaves > 0) {
        const addr = try net.IpAddress.parse("127.0.0.1", wire.PORT);
        try bringUpMaster(io, addr, participants, cfg.n_layer, slaves, bufs);
    } else {
        log.info("master: local mode (0 slaves) — running the whole model in-process, no sockets", .{});
    }
    defer { // closing each stream → slaves see EndOfStream → clean exit
        for (slaves, 0..) |*sc, i| {
            log.info("master: closing connection to slave {d}", .{i});
            sc.reader.stream.close(io);
        }
    }

    const master_range = partition.layerRange(0, participants, cfg.n_layer);
    log.info("master: own range = layers [{d}, {d}) + HEAD + TAIL", .{ master_range.lo, master_range.hi });

    // === encode prompt ===
    const ids = try gpa.alloc(u32, n_ctx);
    defer gpa.free(ids);
    var len = tok.encode(prompt, ids);
    assert(len > 0 and len <= n_ctx);

    const logits = try std.heap.page_allocator.alloc(f32, n_ctx * vocab);
    defer std.heap.page_allocator.free(logits);

    // CPU staging for the residual stream during a broadcast: GPU → here → wire → here → GPU.
    const x_cpu = try gpa.alloc(f32, n_ctx * n_embd);
    defer gpa.free(x_cpu);

    try out.writeAll(prompt);
    try out.flush();

    // === greedy generation loop (no KV cache — recomputes the whole sequence each step) ===
    var produced: usize = 0;
    while (produced < MAX_NEW and len < n_ctx) : (produced += 1) {
        try g.embed(ids[0..len], .{}); // HEAD (on GPU)
        try g.runLayers(len, master_range.lo, master_range.hi); // master front chunk (on GPU)
        log.debug("master: step {d}: S={d}, ran [{d},{d}), routing through {d} slave(s)", .{ produced, len, master_range.lo, master_range.hi, slaves.len });

        if (slaves.len > 0) {
            // The broadcast boundary: finish all kernels, copy the residual to CPU, route it through
            // each slave in pipeline order (reply overwrites x_cpu), then upload it back for the TAIL.
            g.deviceWaitIdle();
            try g.readResidual(len, x_cpu[0 .. len * n_embd]);
            for (slaves, 0..) |*sc, i| {
                wire.sendActivations(&sc.writer.interface, x_cpu[0 .. len * n_embd], @intCast(len)) catch |e| {
                    log.err("master: step {d}: send to slave {d} failed: {s}", .{ produced, i, @errorName(e) });
                    return e;
                };
                const got = wire.recvActivations(&sc.reader.interface, x_cpu, n_embd) catch |e| {
                    log.err("master: step {d}: recv from slave {d} failed: {s}", .{ produced, i, @errorName(e) });
                    return e;
                };
                log.debug("master: step {d}: ↔ slave {d} reply S={d}", .{ produced, i, got });
                assert(got == len);
            }
            try g.writeResidual(len, x_cpu[0 .. len * n_embd]);
        }

        try g.tail(len, logits[0 .. len * vocab], .{}); // TAIL (master owns wte; reads logits back to CPU)

        const last = logits[(len - 1) * vocab ..][0..vocab];
        var next: usize = 0;
        for (last, 0..) |v, i| {
            if (v > last[next]) next = i;
        }
        if (next == cfg.eos_token_id) break;

        ids[len] = @intCast(next);
        len += 1;
        try out.writeAll(tok.decodeToken(@intCast(next)));
        try out.flush();
    }

    log.info("master: generation complete ({d} token(s), seq_len={d})", .{ produced, len });
    try out.writeByte('\n');
    try out.flush();
}

// =================
// === Bring-up FSM (reduce → decide → transition) ===

const BringUp = union(enum) {
    listen,
    accept: struct { server: net.Server, next: usize },
    ready,
    failed: anyerror,
};

fn bringUpMaster(
    io: std.Io,
    addr: net.IpAddress,
    participants: usize,
    n_layer: usize,
    slaves: []SlaveConn,
    bufs: []ConnBufs,
) !void {
    state: switch (BringUp{ .listen = {} }) {
        .listen => continue :state transitionListen(io, addr),
        .accept => |s| continue :state transitionAccept(io, s.server, s.next, participants, n_layer, slaves, bufs),
        .ready => return,
        .failed => |e| {
            log.err("master: bring-up failed: {s}", .{@errorName(e)});
            return e;
        },
    }
}

fn transitionListen(io: std.Io, addr: net.IpAddress) BringUp {
    const server = addr.listen(io, .{ .reuse_address = true }) catch |e| {
        log.err("master: listen on 127.0.0.1:{d} failed: {s}", .{ wire.PORT, @errorName(e) });
        return .{ .failed = e };
    };
    log.info("master: listening on 127.0.0.1:{d}", .{wire.PORT});
    return .{ .accept = .{ .server = server, .next = 0 } };
}

fn transitionAccept(
    io: std.Io,
    server: net.Server,
    next: usize,
    participants: usize,
    n_layer: usize,
    slaves: []SlaveConn,
    bufs: []ConnBufs,
) BringUp {
    var srv = server;
    if (next == slaves.len) {
        srv.deinit(io); // stop listening; accepted streams stay open
        log.info("master: all {d} slave(s) connected, done accepting", .{slaves.len});
        return .ready;
    }

    log.info("master: waiting for slave {d}/{d} to connect...", .{ next + 1, slaves.len });
    const stream = srv.accept(io) catch |e| {
        log.err("master: accept of slave {d} failed: {s}", .{ next, @errorName(e) });
        return .{ .failed = e };
    };
    log.info("master: slave {d} connected", .{next});

    const range = partition.layerRange(next + 1, participants, n_layer); // slaves are participants 1..N
    slaves[next] = .{
        .range = range,
        .reader = net.Stream.Reader.init(stream, io, &bufs[next].rbuf),
        .writer = net.Stream.Writer.init(stream, io, &bufs[next].wbuf),
    };
    wire.sendHandshake(&slaves[next].writer.interface, .{
        .layer_lo = @intCast(range.lo),
        .layer_hi = @intCast(range.hi),
    }) catch |e| {
        log.err("master: handshake to slave {d} failed: {s}", .{ next, @errorName(e) });
        return .{ .failed = e };
    };
    log.info("master: slave {d} ← directive: layers [{d}, {d})", .{ next, range.lo, range.hi });

    return .{ .accept = .{ .server = srv, .next = next + 1 } };
}
