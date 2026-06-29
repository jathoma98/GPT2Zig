const std = @import("std");
const GPT2Zig = @import("GPT2Zig");
const dist = GPT2Zig.dist;

// Surface all log levels (incl. .debug) on stderr — the distributed path logs per-frame wire
// traffic under the `.net` scope at debug, and full passes run in ReleaseSafe (where debug is
// otherwise dropped). Generated text goes to stdout, so logs never corrupt the output.
pub const std_options: std.Options = .{ .log_level = .debug };

// The binary's sole CLI arg is a path to a run-config JSON (see src/dist/runconfig.zig). The parsed
// role decides whether this process is the master (hub + HEAD/TAIL) or a slave (one BODY shard).
pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const gpa = init.gpa;
    const arena = init.arena.allocator();

    // iterateAllocator (not iterate) is the cross-platform form: Windows has no flat argv, so the
    // command line is parsed into a heap buffer here (no-op deinit on POSIX). cfg_path points into
    // that buffer, so the iterator must outlive its use below — hence deinit is deferred to end of main.
    var args = try init.minimal.args.iterateAllocator(gpa);
    defer args.deinit();
    _ = args.skip(); // argv[0]
    const cfg_path = args.next() orelse {
        std.log.err("usage: GPT2Zig <config.json>", .{});
        return error.MissingConfigPath;
    };

    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, cfg_path, arena, .unlimited);
    const cfg = try dist.runconfig.fromBytes(arena, bytes);

    switch (cfg) {
        .master => |m| try dist.master.run(io, gpa, m.model_path, m.prompt, m.expected_slaves, m.listen_addr),
        .slave => |s| try dist.slave.run(io, gpa, s.model_path, s.master_addr),
    }
}
