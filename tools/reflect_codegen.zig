//! Build-time codegen: Slang reflection JSON -> Zig bindings module.
//!
//! Run by build.zig:  reflect_codegen <kernel.reflect.json> <bindings.zig>
//!
//! The pure core `generate(gpa, json_bytes) -> []u8` parses the reflection JSON Slang emits
//! (`slangc -reflection-json`) and emits a Zig source module describing one compute pipeline:
//! the descriptor bindings (StructuredBuffer slots), their access (read-only / write / readWrite)
//! and element scalar type, the push-constant block as an `extern struct` with comptime
//! `@offsetOf`/`@sizeOf` asserts pinning it to the layout slangc reported, plus a per-entry-point
//! namespace with name, local workgroup size and a `dispatchGroups` helper.
//!
//! Why a build tool and not comptime JSON parsing: the JSON is a build *input artifact* produced
//! by an external compiler; doing the transform here keeps the layout asserts as the single
//! safety net and lets the engine import a plain, readable .zig module with zero comptime cost.
//!
//! This is the compile-time interface authority for the whole shader set, so it is deliberately
//! strict: anything it cannot represent faithfully is a hard error (see ParseError) rather than a
//! silent default. The big offenders it refuses are (a) two push-constant blocks in one module —
//! Vulkan permits one per pipeline layout — and (b) the string-or-int reflection sizes
//! ("unbounded"/"unknown") that would otherwise crash the dynamic-Value `.integer` access.
const std = @import("std");
const assert = std.debug.assert;

// =====================================================================
// === Reflection model: the subset of the JSON schema we consume ===
//
// Mirrors the verified schema (research/slang-overview.md §3). We model exactly what codegen
// needs; unknown fields are ignored by the dynamic Value walk. `space` is omitted in JSON when 0
// and `count` when 1 (we default accordingly); `access` absent ⇒ read-only.

const Access = enum {
    read_only, // JSON: type.access absent
    write,
    read_write,

    fn zigName(self: Access) []const u8 {
        return switch (self) {
            .read_only => "read_only",
            .write => "write",
            .read_write => "read_write",
        };
    }
};

const Binding = struct {
    name: []const u8,
    index: u32,
    access: Access,
    elem_scalar: []const u8, // scalarType string, e.g. "float32" ("" when untyped, e.g. byteAddressBuffer)
};

const PushField = struct {
    name: []const u8,
    zig_type: []const u8,
    offset: u32,
    size: u32,
};

const PushBlock = struct {
    fields: []const PushField,
    total_size: u32,
};

// One compute entry point. A single .slang file may declare several when they share the same
// module-global bindings + push constant (e.g. a two-pass reduce/apply); each contributes only
// its own name + workgroup size. Divergent kernels go in separate files (one Entry each).
const Entry = struct {
    name: []const u8,
    local_size: [3]u32,
};

const Pipeline = struct {
    bindings: []const Binding,
    push: ?PushBlock,
    entries: []const Entry,
};

// =====================================================================
// === scalarType -> Zig type ===
//
// The full set slangc 2026.12 can emit (slang-overview §3). bfloat16/fp8 have no native Zig
// scalar, so they map to the same-width unsigned integer — the host treats them as opaque bytes,
// and the @sizeOf assert still pins the block layout. `void`/`unknown` are emitter error cases.

fn scalarToZig(s: []const u8) ?[]const u8 {
    const map = [_]struct { []const u8, []const u8 }{
        .{ "float16", "f16" },     .{ "float32", "f32" },    .{ "float64", "f64" },
        .{ "int8", "i8" },         .{ "uint8", "u8" },       .{ "int16", "i16" },
        .{ "uint16", "u16" },      .{ "int32", "i32" },      .{ "uint32", "u32" },
        .{ "int64", "i64" },       .{ "uint64", "u64" },     .{ "bool", "u32" },
        .{ "bfloat16", "u16" },    .{ "float_e4m3", "u8" },  .{ "float_e5m2", "u8" },
    };
    for (map) |kv| if (std.mem.eql(u8, kv[0], s)) return kv[1];
    return null;
}

// =====================================================================
// === Parse: JSON Value tree -> Pipeline ===
//
// We use std.json.Value (dynamic tree) rather than a typed parse: the binding object has two
// shapes keyed by "kind", and several fields are conditionally present, so a hand-walk over the
// dynamic tree is simpler and more explicit than wrestling the typed parser's optionals.

const ParseError = error{
    MissingParameters,
    MissingEntryPoints,
    NoEntryPoint,
    BadBinding,
    UnknownScalarType,
    MissingPushSize,
    MultiplePushConstants, // two [[vk::push_constant]] blocks — illegal in one pipeline layout
    NonIntegerSize, // a reflection size came back as a string ("unbounded"/"unknown") or overflowed
    OutOfMemory,
};

// Reflection sizes/offsets/indices go through slangc's emitReflectionSize, which emits the STRING
// "unbounded"/"unknown" for runtime-sized/opaque cases instead of an int (slang-overview §3). The
// dynamic Value would then be a `.string`/`.number_string`, and a bare `.integer` access panics.
// Funnel every such read through here so those cases surface as a clean build error.
fn jsonU32(v: std.json.Value) ParseError!u32 {
    return switch (v) {
        .integer => |i| if (i >= 0 and i <= std.math.maxInt(u32)) @intCast(i) else error.NonIntegerSize,
        .string, .number_string => error.NonIntegerSize,
        else => error.NonIntegerSize,
    };
}

fn parsePipeline(
    arena: std.mem.Allocator,
    root: std.json.Value,
    bindings_out: *std.ArrayList(Binding),
    push_fields_out: *std.ArrayList(PushField),
    entries_out: *std.ArrayList(Entry),
) ParseError!Pipeline {
    const obj = root.object;

    // --- global parameters: descriptor buffers + push constant block ---
    const params = (obj.get("parameters") orelse return error.MissingParameters).array;
    var push: ?PushBlock = null;

    for (params.items) |p| {
        const po = p.object;
        const name = (po.get("name") orelse return error.BadBinding).string;
        const binding = (po.get("binding") orelse return error.BadBinding).object;
        const kind = (binding.get("kind") orelse return error.BadBinding).string;
        const ptype = (po.get("type") orelse return error.BadBinding).object;

        if (std.mem.eql(u8, kind, "descriptorTableSlot")) {
            const index = try jsonU32(binding.get("index") orelse return error.BadBinding);
            const access = blk: {
                const a = ptype.get("access") orelse break :blk Access.read_only;
                const as = a.string;
                if (std.mem.eql(u8, as, "write")) break :blk Access.write;
                if (std.mem.eql(u8, as, "readWrite")) break :blk Access.read_write;
                break :blk Access.read_only;
            };
            const elem = blk: {
                const rt = ptype.get("resultType") orelse break :blk ""; // untyped, e.g. byteAddressBuffer
                break :blk (rt.object.get("scalarType") orelse break :blk "").string;
            };
            try bindings_out.append(arena, .{
                .name = name,
                .index = index,
                .access = access,
                .elem_scalar = elem,
            });
        } else if (std.mem.eql(u8, kind, "pushConstantBuffer")) {
            if (push != null) return error.MultiplePushConstants;
            push = try parsePushBlock(arena, ptype, push_fields_out);
        }
        // Other kinds (descriptorTableSlot UBO, specializationConstant, etc.) are out of scope here.
    }

    // --- entry points: every [shader("compute")] in the module, each with its own name + size ---
    const eps = (obj.get("entryPoints") orelse return error.MissingEntryPoints).array;
    if (eps.items.len == 0) return error.NoEntryPoint;
    for (eps.items) |ep_val| {
        const ep = ep_val.object;
        const name = (ep.get("name") orelse return error.NoEntryPoint).string;
        var local_size: [3]u32 = .{ 1, 1, 1 };
        if (ep.get("threadGroupSize")) |tgs| {
            for (tgs.array.items, 0..) |v, i| {
                if (i >= 3) break;
                local_size[i] = try jsonU32(v);
            }
        }
        try entries_out.append(arena, .{ .name = name, .local_size = local_size });
    }

    return .{
        .bindings = bindings_out.items,
        .push = push,
        .entries = entries_out.items,
    };
}

fn parsePushBlock(
    arena: std.mem.Allocator,
    ptype: std.json.ObjectMap,
    out: *std.ArrayList(PushField),
) ParseError!PushBlock {
    const elem_type = (ptype.get("elementType") orelse return error.BadBinding).object;
    const fields = (elem_type.get("fields") orelse return error.BadBinding).array;
    for (fields.items) |f| {
        const fo = f.object;
        const fname = (fo.get("name") orelse return error.BadBinding).string;
        const ftype = (fo.get("type") orelse return error.BadBinding).object;
        const scalar = (ftype.get("scalarType") orelse return error.UnknownScalarType).string;
        const zt = scalarToZig(scalar) orelse return error.UnknownScalarType;
        const fb = (fo.get("binding") orelse return error.BadBinding).object;
        const offset = try jsonU32(fb.get("offset") orelse return error.BadBinding);
        const size = try jsonU32(fb.get("size") orelse return error.BadBinding);
        try out.append(arena, .{ .name = fname, .zig_type = zt, .offset = offset, .size = size });
    }

    // Total block size lives in elementVarLayout.binding.size (NOT derivable from fields alone).
    const evl = (ptype.get("elementVarLayout") orelse return error.MissingPushSize).object;
    const evl_binding = (evl.get("binding") orelse return error.MissingPushSize).object;
    const total = try jsonU32(evl_binding.get("size") orelse return error.MissingPushSize);

    return .{ .fields = out.items, .total_size = total };
}

// =====================================================================
// === Emit: Pipeline -> Zig source text ===

fn emit(w: *std.Io.Writer, pipe: Pipeline, hash: u64) !void {
    try w.print(
        \\//! GENERATED by reflect_codegen.zig from Slang reflection JSON. DO NOT EDIT.
        \\const std = @import("std");
        \\
        \\pub const reflection_hash: u64 = 0x{x};
        \\pub const descriptor_set: u32 = 0;
        \\pub const binding_count: u32 = {d};
        \\
        \\
    , .{ hash, pipe.bindings.len });

    // --- Binding enum ---
    try w.writeAll("pub const Binding = enum(u32) {\n");
    for (pipe.bindings) |b| {
        try w.print("    {s} = {d},\n", .{ b.name, b.index });
    }
    try w.writeAll("};\n\n");

    // --- binding_access / binding_elem parallel arrays, indexed by Binding ---
    try w.print("pub const binding_access = [_]enum {{ read_only, write, read_write }}{{\n", .{});
    for (pipe.bindings) |b| {
        try w.print("    .{s}, // {s}\n", .{ b.access.zigName(), b.name });
    }
    try w.writeAll("};\n\n");

    try w.writeAll("pub const binding_elem = [_][]const u8{\n");
    for (pipe.bindings) |b| {
        try w.print("    \"{s}\", // {s}\n", .{ b.elem_scalar, b.name });
    }
    try w.writeAll("};\n\n");

    // --- Push constants extern struct + layout asserts ---
    if (pipe.push) |push| {
        try w.writeAll("pub const PushConstants = extern struct {\n");
        for (push.fields) |f| {
            try w.print("    {s}: {s},\n", .{ f.name, f.zig_type });
        }
        try w.writeAll("};\n\n");
        try w.print("pub const push_constant_size: u32 = {d};\n\n", .{push.total_size});

        // The safety net: pin every field offset and the total size to what slangc reported.
        // Fails to compile if someone drops -fvk-use-scalar-layout or swaps to @Vector fields.
        try w.writeAll("comptime {\n");
        for (push.fields) |f| {
            try w.print(
                "    std.debug.assert(@offsetOf(PushConstants, \"{s}\") == {d});\n",
                .{ f.name, f.offset },
            );
        }
        try w.print("    std.debug.assert(@sizeOf(PushConstants) == {d});\n", .{push.total_size});
        try w.writeAll("}\n\n");
    } else {
        try w.writeAll("pub const PushConstants = void;\n");
        try w.writeAll("pub const push_constant_size: u32 = 0;\n\n");
    }

    // --- Entry points + dispatch ---
    // One namespaced struct per entry point. The bindings/push above are module-global (shared by
    // all entries), so they stay at module scope; only name + workgroup size + dispatch differ.
    for (pipe.entries) |e| try emitEntry(w, e);

    // For the common single-kernel file, also expose the entry's members flat at module scope so
    // the host can write `bind.dispatchGroups(...)` instead of `bind.<name>.dispatchGroups(...)`.
    if (pipe.entries.len == 1) {
        const e = pipe.entries[0];
        try w.print(
            \\
            \\pub const entry_point = {s}.entry_point;
            \\pub const local_size = {s}.local_size;
            \\pub const dispatchGroups = {s}.dispatchGroups;
            \\
        , .{ e.name, e.name, e.name });
    }
}

fn emitEntry(w: *std.Io.Writer, e: Entry) !void {
    try w.print(
        \\pub const {s} = struct {{
        \\    pub const entry_point = "{s}";
        \\    pub const local_size = [3]u32{{ {d}, {d}, {d} }};
        \\    /// Workgroup counts to cover an (x,y,z) global thread extent, rounding up per dim.
        \\    pub fn dispatchGroups(global: [3]u32) [3]u32 {{
        \\        const ls = @This().local_size; // qualify: avoids clash with module-level flat alias
        \\        return .{{
        \\            (global[0] + ls[0] - 1) / ls[0],
        \\            (global[1] + ls[1] - 1) / ls[1],
        \\            (global[2] + ls[2] - 1) / ls[2],
        \\        }};
        \\    }}
        \\}};
        \\
    , .{ e.name, e.name, e.local_size[0], e.local_size[1], e.local_size[2] });
}

// =====================================================================
// === Pure core: JSON bytes -> bindings.zig bytes ===

pub fn generate(gpa: std.mem.Allocator, json_bytes: []const u8) ![]u8 {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var parsed = try std.json.parseFromSlice(std.json.Value, arena, json_bytes, .{});
    defer parsed.deinit();

    var bindings = std.ArrayList(Binding).empty;
    var push_fields = std.ArrayList(PushField).empty;
    var entries = std.ArrayList(Entry).empty;
    const pipe = try parsePipeline(arena, parsed.value, &bindings, &push_fields, &entries);

    // Content hash of the JSON, so a stale binding can be detected at runtime against the .spv.
    const hash = std.hash.Wyhash.hash(0, json_bytes);

    var aw = std.Io.Writer.Allocating.init(gpa);
    errdefer aw.deinit();
    try emit(&aw.writer, pipe, hash);
    return aw.toOwnedSlice();
}

// =====================================================================
// === IO shell (matches tools/gen_bpe.zig) ===

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const gpa = init.gpa;

    var args = try init.minimal.args.iterateAllocator(gpa);
    defer args.deinit();
    _ = args.skip();
    const in_path = args.next() orelse return error.MissingInputArg;
    const out_path = args.next() orelse return error.MissingOutputArg;

    const cwd = std.Io.Dir.cwd();
    const json_bytes = try cwd.readFileAlloc(io, in_path, gpa, .unlimited);
    defer gpa.free(json_bytes);

    const out = try generate(gpa, json_bytes);
    defer gpa.free(out);

    try cwd.writeFile(io, .{ .sub_path = out_path, .data = out });
}

// =====================================================================
// === Tests ===

fn expectContainsAll(out: []const u8, needles: []const []const u8) !void {
    for (needles) |needle| {
        std.testing.expect(std.mem.indexOf(u8, out, needle) != null) catch |e| {
            std.debug.print("MISSING: {s}\n", .{needle});
            return e;
        };
    }
}

const gemm_json =
    \\{
    \\  "parameters": [
    \\    {"name":"A","binding":{"kind":"descriptorTableSlot","index":0},
    \\     "type":{"kind":"resource","baseShape":"structuredBuffer","resultType":{"kind":"scalar","scalarType":"float32"}}},
    \\    {"name":"C","binding":{"kind":"descriptorTableSlot","index":1},
    \\     "type":{"kind":"resource","baseShape":"structuredBuffer","access":"readWrite","resultType":{"kind":"scalar","scalarType":"float32"}}},
    \\    {"name":"dims","binding":{"kind":"pushConstantBuffer","index":0},
    \\     "type":{"kind":"constantBuffer","elementType":{"kind":"struct","name":"Dims","fields":[
    \\       {"name":"M","type":{"kind":"scalar","scalarType":"uint32"},"binding":{"kind":"uniform","offset":0,"size":4,"elementStride":0}},
    \\       {"name":"K","type":{"kind":"scalar","scalarType":"uint32"},"binding":{"kind":"uniform","offset":4,"size":4,"elementStride":0}},
    \\       {"name":"N","type":{"kind":"scalar","scalarType":"uint32"},"binding":{"kind":"uniform","offset":8,"size":4,"elementStride":0}}
    \\     ]},"elementVarLayout":{"binding":{"kind":"uniform","offset":0,"size":12,"elementStride":0}}}}
    \\  ],
    \\  "entryPoints": [
    \\    {"name":"main","stage":"compute","threadGroupSize":[16,16,1],"bindings":[]}
    \\  ],
    \\  "bindlessSpaceIndex": 1
    \\}
;

test "single-entry: enum, access, asserts, push struct, dispatch, flat alias" {
    const gpa = std.testing.allocator;
    const out = try generate(gpa, gemm_json);
    defer gpa.free(out);

    try expectContainsAll(out, &.{
        "pub const binding_count: u32 = 2;",
        "pub const Binding = enum(u32) {",
        "A = 0,",
        "C = 1,",
        ".read_only, // A",
        ".read_write, // C",
        "pub const PushConstants = extern struct {",
        "M: u32,",
        "K: u32,",
        "N: u32,",
        "std.debug.assert(@offsetOf(PushConstants, \"K\") == 4)",
        "std.debug.assert(@offsetOf(PushConstants, \"N\") == 8)",
        "std.debug.assert(@sizeOf(PushConstants) == 12)",
        "pub const push_constant_size: u32 = 12;",
        "pub const main = struct {",
        "pub const entry_point = \"main\";",
        "pub const local_size = [3]u32{ 16, 16, 1 };",
        "pub fn dispatchGroups(global: [3]u32) [3]u32 {",
        // single-entry flat aliases
        "pub const entry_point = main.entry_point;",
        "pub const dispatchGroups = main.dispatchGroups;",
    });
}

// Multi-entry-point: two kernels sharing the same module-global bindings + push constant.
const multi_json =
    \\{
    \\  "parameters": [
    \\    {"name":"X","binding":{"kind":"descriptorTableSlot","index":0},
    \\     "type":{"kind":"resource","baseShape":"structuredBuffer","resultType":{"kind":"scalar","scalarType":"float32"}}},
    \\    {"name":"Y","binding":{"kind":"descriptorTableSlot","index":1},
    \\     "type":{"kind":"resource","baseShape":"structuredBuffer","access":"readWrite","resultType":{"kind":"scalar","scalarType":"float32"}}},
    \\    {"name":"scale","binding":{"kind":"pushConstantBuffer","index":0},
    \\     "type":{"kind":"constantBuffer","elementType":{"kind":"struct","name":"Scale","fields":[
    \\       {"name":"k","type":{"kind":"scalar","scalarType":"float32"},"binding":{"kind":"uniform","offset":0,"size":4,"elementStride":0}},
    \\       {"name":"n","type":{"kind":"scalar","scalarType":"uint32"},"binding":{"kind":"uniform","offset":4,"size":4,"elementStride":0}}
    \\     ]},"elementVarLayout":{"binding":{"kind":"uniform","offset":0,"size":8,"elementStride":0}}}}
    \\  ],
    \\  "entryPoints": [
    \\    {"name":"scaleKernel","stage":"compute","threadGroupSize":[64,1,1],"bindings":[]},
    \\    {"name":"reluKernel","stage":"compute","threadGroupSize":[32,1,1],"bindings":[]}
    \\  ],
    \\  "bindlessSpaceIndex": 1
    \\}
;

test "multi-entry: per-entry structs, shared bindings, no flat alias" {
    const gpa = std.testing.allocator;
    const out = try generate(gpa, multi_json);
    defer gpa.free(out);

    try expectContainsAll(out, &.{
        "pub const Binding = enum(u32) {", "X = 0,", "Y = 1,",
        "pub const PushConstants = extern struct {",
        "k: f32,",
        "n: u32,",
        "pub const scaleKernel = struct {",
        "pub const local_size = [3]u32{ 64, 1, 1 };",
        "pub const reluKernel = struct {",
        "pub const local_size = [3]u32{ 32, 1, 1 };",
    });
    // With >1 entry point there must be NO module-level flat alias (would be ambiguous).
    try std.testing.expect(std.mem.indexOf(u8, out, "pub const dispatchGroups = ") == null);
}

test "no push constant: PushConstants = void, size 0" {
    const json =
        \\{
        \\  "parameters": [
        \\    {"name":"In","binding":{"kind":"descriptorTableSlot","index":0},
        \\     "type":{"kind":"resource","baseShape":"structuredBuffer","resultType":{"kind":"scalar","scalarType":"float32"}}}
        \\  ],
        \\  "entryPoints": [ {"name":"main","stage":"compute","threadGroupSize":[256,1,1]} ],
        \\  "bindlessSpaceIndex": 1
        \\}
    ;
    const gpa = std.testing.allocator;
    const out = try generate(gpa, json);
    defer gpa.free(out);
    try expectContainsAll(out, &.{
        "pub const PushConstants = void;",
        "pub const push_constant_size: u32 = 0;",
        "pub const binding_count: u32 = 1;",
    });
}

test "write access + untyped buffer + default index/space" {
    // `Out` is write-only; `Raw` is a byteAddressBuffer (no resultType ⇒ empty elem string).
    const json =
        \\{
        \\  "parameters": [
        \\    {"name":"Out","binding":{"kind":"descriptorTableSlot","index":0},
        \\     "type":{"kind":"resource","baseShape":"structuredBuffer","access":"write","resultType":{"kind":"scalar","scalarType":"float32"}}},
        \\    {"name":"Raw","binding":{"kind":"descriptorTableSlot","index":1},
        \\     "type":{"kind":"resource","baseShape":"byteAddressBuffer","access":"readWrite"}}
        \\  ],
        \\  "entryPoints": [ {"name":"main","stage":"compute","threadGroupSize":[1,1,1]} ],
        \\  "bindlessSpaceIndex": 1
        \\}
    ;
    const gpa = std.testing.allocator;
    const out = try generate(gpa, json);
    defer gpa.free(out);
    try expectContainsAll(out, &.{
        ".write, // Out",
        ".read_write, // Raw",
        "\"float32\", // Out",
        "\"\", // Raw", // untyped buffer ⇒ empty element scalar string
    });
}

test "every scalarType maps to the right Zig type in a push block" {
    // One field per supported scalar, offsets/total are dummy (we only assert the type spellings).
    const json =
        \\{
        \\  "parameters": [
        \\    {"name":"p","binding":{"kind":"pushConstantBuffer","index":0},
        \\     "type":{"kind":"constantBuffer","elementType":{"kind":"struct","name":"P","fields":[
        \\       {"name":"a","type":{"kind":"scalar","scalarType":"float16"},"binding":{"kind":"uniform","offset":0,"size":2}},
        \\       {"name":"b","type":{"kind":"scalar","scalarType":"float64"},"binding":{"kind":"uniform","offset":8,"size":8}},
        \\       {"name":"c","type":{"kind":"scalar","scalarType":"int8"},"binding":{"kind":"uniform","offset":16,"size":1}},
        \\       {"name":"d","type":{"kind":"scalar","scalarType":"uint64"},"binding":{"kind":"uniform","offset":24,"size":8}},
        \\       {"name":"e","type":{"kind":"scalar","scalarType":"bfloat16"},"binding":{"kind":"uniform","offset":32,"size":2}},
        \\       {"name":"f","type":{"kind":"scalar","scalarType":"float_e4m3"},"binding":{"kind":"uniform","offset":34,"size":1}},
        \\       {"name":"g","type":{"kind":"scalar","scalarType":"bool"},"binding":{"kind":"uniform","offset":36,"size":4}}
        \\     ]},"elementVarLayout":{"binding":{"kind":"uniform","offset":0,"size":40}}}}
        \\  ],
        \\  "entryPoints": [ {"name":"main","stage":"compute","threadGroupSize":[1,1,1]} ],
        \\  "bindlessSpaceIndex": 1
        \\}
    ;
    const gpa = std.testing.allocator;
    const out = try generate(gpa, json);
    defer gpa.free(out);
    try expectContainsAll(out, &.{
        "a: f16,", "b: f64,", "c: i8,", "d: u64,", "e: u16,", "f: u8,", "g: u32,",
    });
}

test "unknown scalarType is a hard error" {
    const json =
        \\{
        \\  "parameters": [
        \\    {"name":"p","binding":{"kind":"pushConstantBuffer","index":0},
        \\     "type":{"kind":"constantBuffer","elementType":{"kind":"struct","name":"P","fields":[
        \\       {"name":"a","type":{"kind":"scalar","scalarType":"quaternion16"},"binding":{"kind":"uniform","offset":0,"size":2}}
        \\     ]},"elementVarLayout":{"binding":{"kind":"uniform","offset":0,"size":4}}}}
        \\  ],
        \\  "entryPoints": [ {"name":"main","stage":"compute","threadGroupSize":[1,1,1]} ],
        \\  "bindlessSpaceIndex": 1
        \\}
    ;
    try std.testing.expectError(error.UnknownScalarType, generate(std.testing.allocator, json));
}

test "two push-constant blocks is a hard error (one per pipeline layout)" {
    const json =
        \\{
        \\  "parameters": [
        \\    {"name":"p","binding":{"kind":"pushConstantBuffer","index":0},
        \\     "type":{"kind":"constantBuffer","elementType":{"kind":"struct","name":"P","fields":[
        \\       {"name":"a","type":{"kind":"scalar","scalarType":"uint32"},"binding":{"kind":"uniform","offset":0,"size":4}}
        \\     ]},"elementVarLayout":{"binding":{"kind":"uniform","offset":0,"size":4}}}},
        \\    {"name":"q","binding":{"kind":"pushConstantBuffer","index":1},
        \\     "type":{"kind":"constantBuffer","elementType":{"kind":"struct","name":"Q","fields":[
        \\       {"name":"b","type":{"kind":"scalar","scalarType":"uint32"},"binding":{"kind":"uniform","offset":0,"size":4}}
        \\     ]},"elementVarLayout":{"binding":{"kind":"uniform","offset":0,"size":4}}}}
        \\  ],
        \\  "entryPoints": [ {"name":"main","stage":"compute","threadGroupSize":[1,1,1]} ],
        \\  "bindlessSpaceIndex": 1
        \\}
    ;
    try std.testing.expectError(error.MultiplePushConstants, generate(std.testing.allocator, json));
}

test "string-valued reflection size (unbounded) is a hard error, not a panic" {
    // emitReflectionSize stringifies runtime-sized cases; a bare `.integer` access would panic.
    const json =
        \\{
        \\  "parameters": [
        \\    {"name":"p","binding":{"kind":"pushConstantBuffer","index":0},
        \\     "type":{"kind":"constantBuffer","elementType":{"kind":"struct","name":"P","fields":[
        \\       {"name":"a","type":{"kind":"scalar","scalarType":"uint32"},"binding":{"kind":"uniform","offset":"unbounded","size":4}}
        \\     ]},"elementVarLayout":{"binding":{"kind":"uniform","offset":0,"size":4}}}}
        \\  ],
        \\  "entryPoints": [ {"name":"main","stage":"compute","threadGroupSize":[1,1,1]} ],
        \\  "bindlessSpaceIndex": 1
        \\}
    ;
    try std.testing.expectError(error.NonIntegerSize, generate(std.testing.allocator, json));
}

test "missing parameters / no entry points are hard errors" {
    try std.testing.expectError(
        error.MissingParameters,
        generate(std.testing.allocator, "{\"entryPoints\":[]}"),
    );
    try std.testing.expectError(
        error.NoEntryPoint,
        generate(std.testing.allocator, "{\"parameters\":[],\"entryPoints\":[]}"),
    );
}
