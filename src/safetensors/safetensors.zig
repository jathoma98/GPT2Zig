const std = @import("std");
const assert = std.debug.assert;
const platform = @import("../core/platform/platform.zig");

const MAX_TENSORS = 200;
const MAX_NAME = 64;

pub const Dtype = enum { f32 };

pub const TensorMeta = struct {
    name_buf: [MAX_NAME]u8,
    name_len: u8,
    dtype: Dtype,
    rank: u8,
    shape: [4]u32,
    data_begin: u64,
    data_end: u64,

    pub fn name(self: *const TensorMeta) []const u8 {
        return self.name_buf[0..self.name_len];
    }

    pub fn n_elements(self: *const TensorMeta) u64 {
        var n: u64 = 1;
        for (self.shape[0..self.rank]) |d| n *= d;
        return n;
    }

    pub fn byte_count(self: *const TensorMeta) u64 {
        return self.n_elements() * 4;
    }
};

pub const Handle = enum(u32) { _ };

pub const SafeTensors = struct {
    map: platform.FileMap,
    tensors: [MAX_TENSORS]TensorMeta,
    count: u32,
    data_base: u64,

    pub fn init(io: std.Io, path: []const u8) !SafeTensors {
        var map = try platform.FileMap.open(io, path);
        errdefer map.close();
        assert(map.bytes.len > 8);
        const hdr = try parseHeader(map.bytes);
        return .{
            .map = map,
            .data_base = hdr.data_base,
            .count = hdr.count,
            .tensors = hdr.tensors,
        };
    }

    pub fn deinit(self: *SafeTensors) void {
        self.map.close();
    }

    pub fn find(self: *const SafeTensors, n: []const u8) ?Handle {
        for (self.tensors[0..self.count], 0..) |*t, i| {
            if (std.mem.eql(u8, t.name(), n)) return @enumFromInt(i);
        }
        return null;
    }

    pub fn meta(self: *const SafeTensors, h: Handle) *const TensorMeta {
        const i = @intFromEnum(h);
        assert(i < self.count);
        return &self.tensors[i];
    }

    // Callers must NOT @alignCast this slice to [*]f32: data_base (8 + header_len = 14291)
    // is not 4-byte aligned, so absolute tensor addresses are misaligned regardless of
    // data_offsets. Use std.mem.readInt(u32, ...) for element access.
    pub fn data_bytes(self: *const SafeTensors, h: Handle) []const u8 {
        const m = self.meta(h);
        const begin = self.data_base + m.data_begin;
        const end = self.data_base + m.data_end;
        return self.map.bytes[begin..end];
    }
};

const ParsedHeader = struct {
    data_base: u64,
    count: u32,
    tensors: [MAX_TENSORS]TensorMeta,
};

fn parseHeader(bytes: []const u8) !ParsedHeader {
    assert(bytes.len >= 8);

    const header_len = std.mem.readInt(u64, bytes[0..8], .little);
    assert(header_len > 0);
    assert(8 + header_len <= bytes.len);

    const data_base = 8 + header_len;
    const header_json = bytes[8 .. 8 + header_len];

    // Tensor names and dtype values are plain ASCII — no JSON escape sequences.
    // next() returns slices directly into the mmap'd bytes; zero allocations needed.
    // The allocator arg is only used by nextAlloc() for escaped strings; we never call it.
    var scanner = std.json.Scanner.initCompleteInput(std.heap.page_allocator, header_json);
    defer scanner.deinit();

    var tensors: [MAX_TENSORS]TensorMeta = undefined;
    var count: u32 = 0;

    // =======================
    // === outer: root object
    //
    // {
    //   "__metadata__": { ... },        <- skipped
    //   "wte.weight": { ... },          <- tensor entry; key = tensor name
    //   "h.0.attn.c_attn.weight": { ... },
    //   ...
    // }
    expectTag(try scanner.next(), .object_begin);

    outer: while (true) {
        switch (try scanner.next()) {
            .object_end => break :outer,
            .string => |key| {
                if (std.mem.eql(u8, key, "__metadata__")) {
                    try skipJsonValue(&scanner);
                    continue;
                }

                assert(key.len <= MAX_NAME);
                assert(count < MAX_TENSORS);

                expectTag(try scanner.next(), .object_begin);

                var name_buf: [MAX_NAME]u8 = .{0} ** MAX_NAME;
                @memcpy(name_buf[0..key.len], key);
                var rank: u8 = 0;
                var shape: [4]u32 = .{ 0, 0, 0, 0 };
                var data_begin: u64 = 0;
                var data_end: u64 = 0;
                const TensorField = enum { dtype, shape, data_offsets };
                var fields_seen = std.EnumSet(TensorField).initEmpty();

                // =======================
                // === inner: tensor info object
                //
                // "wte.weight": {
                //   "dtype": "F32",
                //   "shape": [ 50257, 768 ],
                //   "data_offsets": [ 287209472, 441598976 ]
                // }
                inner: while (true) {
                    switch (try scanner.next()) {
                        .object_end => break :inner,
                        .string => |field| {
                            if (std.mem.eql(u8, field, "dtype")) {
                                const s = (try scanner.next()).string;
                                // GPT2 is F32-only. Panic loudly if we ever see anything else.
                                assert(std.mem.eql(u8, s, "F32"));
                                fields_seen.insert(.dtype);
                            } else if (std.mem.eql(u8, field, "shape")) {
                                expectTag(try scanner.next(), .array_begin);
                                // === dim: shape array  [ D0, D1, ... ]  (up to rank 4)
                                dim: while (true) {
                                    switch (try scanner.next()) {
                                        .array_end => break :dim,
                                        .number => |s| {
                                            assert(rank < 4);
                                            shape[rank] = @intCast(try std.fmt.parseInt(u64, s, 10));
                                            rank += 1;
                                        },
                                        else => unreachable,
                                    }
                                }
                                fields_seen.insert(.shape);
                            } else if (std.mem.eql(u8, field, "data_offsets")) {
                                expectTag(try scanner.next(), .array_begin);
                                // === exactly [ begin, end ] — byte range within the data section
                                data_begin = @intCast(try std.fmt.parseInt(u64, (try scanner.next()).number, 10));
                                data_end = @intCast(try std.fmt.parseInt(u64, (try scanner.next()).number, 10));
                                expectTag(try scanner.next(), .array_end);
                                fields_seen.insert(.data_offsets);
                            } else {
                                try skipJsonValue(&scanner);
                            }
                        },
                        else => unreachable,
                    }
                }

                assert(fields_seen.eql(std.EnumSet(TensorField).initFull()));
                assert(data_begin < data_end);
                assert(data_base + data_end <= bytes.len);

                tensors[count] = .{
                    .name_buf = name_buf,
                    .name_len = @intCast(key.len),
                    .dtype = .f32,
                    .rank = rank,
                    .shape = shape,
                    .data_begin = data_begin,
                    .data_end = data_end,
                };
                assert(tensors[count].byte_count() == data_end - data_begin);
                count += 1;
            },
            else => unreachable,
        }
    }

    return .{
        .data_base = data_base,
        .count = count,
        .tensors = tensors,
    };
}

fn expectTag(tok: std.json.Token, expected: std.meta.Tag(std.json.Token)) void {
    assert(std.meta.activeTag(tok) == expected);
}

fn skipJsonValue(scanner: *std.json.Scanner) !void {
    var depth: i32 = 0;
    while (true) {
        switch (try scanner.next()) {
            .object_begin, .array_begin => depth += 1,
            .object_end, .array_end => {
                depth -= 1;
                if (depth == 0) return;
            },
            .string, .number, .true, .false, .null, .partial_string, .allocated_string, .allocated_number, .partial_number, .partial_string_escaped_1, .partial_string_escaped_2, .partial_string_escaped_3, .partial_string_escaped_4 => {
                if (depth == 0) return;
            },
            .end_of_document => return error.UnexpectedEof,
        }
    }
}

// =================
// === Tests ===

test "safetensors dump" {
    var st = try SafeTensors.init(std.testing.io, "models/gpt2/model.safetensors");
    defer st.deinit();
}

test "safetensors manifest" {
    const golden = @import("../generated/safetensors_golden.zig");
    var st = try SafeTensors.init(std.testing.io, "models/gpt2/model.safetensors");
    defer st.deinit();

    try std.testing.expectEqual(@as(u32, golden.manifest.len), st.count);

    for (golden.manifest) |entry| {
        const h = st.find(entry.name) orelse {
            std.debug.print("missing tensor: {s}\n", .{entry.name});
            return error.MissingTensor;
        };
        const m = st.meta(h);
        try std.testing.expectEqual(entry.rank, m.rank);
        for (0..entry.rank) |i| {
            try std.testing.expectEqual(entry.shape[i], m.shape[i]);
        }
        try std.testing.expectEqual(entry.n_elements, m.n_elements());
    }
}

test "safetensors spot-check" {
    const golden = @import("../generated/safetensors_golden.zig");
    var st = try SafeTensors.init(std.testing.io, "models/gpt2/model.safetensors");
    defer st.deinit();

    const wte_bytes = st.data_bytes(st.find("wte.weight").?);
    for (golden.wte_weight_first5, 0..) |expected_bits, i| {
        const actual_bits = std.mem.readInt(u32, wte_bytes[i * 4 ..][0..4], .little);
        try std.testing.expectEqual(expected_bits, actual_bits);
    }

    const bias_bytes = st.data_bytes(st.find("h.0.attn.c_attn.bias").?);
    for (golden.h_0_attn_c_attn_bias_first5, 0..) |expected_bits, i| {
        const actual_bits = std.mem.readInt(u32, bias_bytes[i * 4 ..][0..4], .little);
        try std.testing.expectEqual(expected_bits, actual_bits);
    }
}
