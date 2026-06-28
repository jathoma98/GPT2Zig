//! Runtime role config. The binary's sole CLI arg is a path to one of these JSON files; the parsed
//! `type` decides whether this process is the master (hub + HEAD/TAIL) or a slave (one BODY shard).
//! Parse follows the reduce→decide shape: read a flat Raw struct, then reduce it to the RunConfig
//! tagged union so the rest of the program can't reach master-only fields on a slave (or vice versa).
const std = @import("std");

pub const RunConfig = union(enum) {
    master: struct {
        model_path: []const u8,
        prompt: []const u8,
        expected_slaves: u32, // 0 ⇒ pure local mode (no sockets, single-process forward)
    },
    slave: struct {
        model_path: []const u8,
    },
};

// JSON: { "model_path": "...", "serverParams": { "type": "master"|"slave", "prompt"?, "expectedSlaves"? } }
const Raw = struct {
    model_path: []const u8,
    serverParams: struct {
        type: []const u8,
        prompt: ?[]const u8 = null,
        expectedSlaves: ?u32 = null,
    },
};

// `arena` must outlive the returned RunConfig: parsed strings point into allocations made here.
pub fn fromBytes(arena: std.mem.Allocator, bytes: []const u8) !RunConfig {
    const raw = try std.json.parseFromSliceLeaky(Raw, arena, bytes, .{ .ignore_unknown_fields = true });
    const sp = raw.serverParams;

    if (std.mem.eql(u8, sp.type, "master")) {
        return .{ .master = .{
            .model_path = raw.model_path,
            .prompt = sp.prompt orelse return error.MasterMissingPrompt,
            .expected_slaves = sp.expectedSlaves orelse return error.MasterMissingExpectedSlaves,
        } };
    }
    if (std.mem.eql(u8, sp.type, "slave")) {
        return .{ .slave = .{ .model_path = raw.model_path } };
    }
    return error.InvalidServerType;
}

// =================
// === Tests ===

test "parse master config" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const json =
        \\{ "model_path": "models/gpt2/model.safetensors",
        \\  "serverParams": { "type": "master", "prompt": "Hello, I am", "expectedSlaves": 1 } }
    ;
    const cfg = try fromBytes(arena.allocator(), json);
    try std.testing.expect(cfg == .master);
    try std.testing.expectEqualStrings("models/gpt2/model.safetensors", cfg.master.model_path);
    try std.testing.expectEqualStrings("Hello, I am", cfg.master.prompt);
    try std.testing.expectEqual(@as(u32, 1), cfg.master.expected_slaves);
}

test "parse slave config" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const json =
        \\{ "model_path": "m.safetensors", "serverParams": { "type": "slave" } }
    ;
    const cfg = try fromBytes(arena.allocator(), json);
    try std.testing.expect(cfg == .slave);
    try std.testing.expectEqualStrings("m.safetensors", cfg.slave.model_path);
}

test "master missing fields is an error" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const json =
        \\{ "model_path": "m", "serverParams": { "type": "master" } }
    ;
    try std.testing.expectError(error.MasterMissingPrompt, fromBytes(arena.allocator(), json));
}
