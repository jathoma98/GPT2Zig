//! Build-time tokenizer table generator. Reads GPT-2's `merges.txt` and emits a packed,
//! mmap-ready binary the inference engine consumes with zero runtime construction (see
//! src/core/token.zig). Run by build.zig:  gen_bpe <merges.txt> <out.bin>
//!
//! Why a build tool and not @embedFile+comptime: `StaticStringMap.initComptime` on 50k entries
//! OOM-kills the compiler (comptime sort). Doing the transform here — with native hashmaps and a
//! native sort — keeps compiles fast and makes this file the single source of truth for the
//! bytes_to_unicode ordering (the engine just slices what we emit).
//!
//! Key fact this exploits: GPT-2 token ids are fully derivable from merges.txt alone. The 256 base
//! byte-tokens take ids 0..255 in bytes_to_unicode() order; merge rank i → id 256+i with token
//! string a++b; id 50256 is the <|endoftext|> special. No vocab.json needed.
//!
//! Output format (little-endian), all sections back-to-back, offsets stored in the header:
//!   header: [9]u32 = { magic, n_merges, n_tokens, off_byte2id, off_keys, off_vals, off_doff,
//!                      off_blob, blob_len }
//!   byte2id:  [256]u32        raw byte → base token id
//!   keys:     [n_merges]u64   (id_a<<32 | id_b), sorted ascending  ← binary-searched at runtime
//!   vals:     [n_merges]u32   merged_id for the key at the same index
//!   doff:     [n_tokens+1]u32 prefix offsets into blob; token id's bytes = blob[doff[id]..doff[id+1]]
//!   blob:     [blob_len]u8    concatenated raw decoded bytes of every token, in id order
const std = @import("std");
const assert = std.debug.assert;

const MAGIC: u32 = 0x42504531; // "BPE1"
const N_MERGES_EXPECTED = 50000;
const EOS_ID: u32 = 50256;
const EOS_TEXT = "<|endoftext|>";

const Entry = struct { key: u64, val: u32 };

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const gpa = init.gpa;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // === args: <exe> <merges_path> <out_path> ===
    var args = init.minimal.args.iterate();
    _ = args.skip();
    const merges_path = args.next() orelse return error.MissingMergesArg;
    const out_path = args.next() orelse return error.MissingOutArg;

    const cwd = std.Io.Dir.cwd();

    // ===========================================
    // === bytes_to_unicode: base ids + decode ===
    //
    // bs lists the 256 bytes in the order GPT-2 assigns base ids: the printable ranges first (each
    // mapping to its own codepoint), then the remaining bytes appended (mapped to codepoints 256+n).
    // cs[k] is base id k's remapped codepoint; bs[k] is its raw byte.
    var bs: [256]u8 = undefined;
    var cs: [256]u21 = undefined;
    var byte_to_id: [256]u32 = undefined;
    {
        var present = [_]bool{false} ** 256;
        var n: usize = 0;
        const ranges = [_][2]u16{ .{ 33, 126 }, .{ 161, 172 }, .{ 174, 255 } };
        for (ranges) |r| {
            var b: u16 = r[0];
            while (b <= r[1]) : (b += 1) {
                bs[n] = @intCast(b);
                cs[n] = @intCast(b);
                byte_to_id[b] = @intCast(n);
                present[b] = true;
                n += 1;
            }
        }
        var extra: u21 = 0;
        var b: u16 = 0;
        while (b < 256) : (b += 1) {
            if (!present[b]) {
                bs[n] = @intCast(b);
                cs[n] = 256 + extra;
                byte_to_id[b] = @intCast(n);
                extra += 1;
                n += 1;
            }
        }
        assert(n == 256);
    }

    // === encoder: remapped token string → id (used only here, to resolve merge pairs) ===
    // === decode_bytes: id → raw bytes (becomes the blob) ===
    var encoder = std.StringHashMap(u32).init(gpa);
    defer encoder.deinit();

    const data = try cwd.readFileAlloc(io, merges_path, gpa, .unlimited);
    defer gpa.free(data);

    // n_tokens isn't known until we count merges, but the file has a fixed count; size generously.
    var decode_bytes = std.ArrayList([]const u8).empty;
    defer decode_bytes.deinit(gpa);

    // Seed the 256 base tokens: string = UTF-8 of the remapped codepoint, decoded bytes = the byte.
    for (0..256) |k| {
        var buf: [4]u8 = undefined;
        const len = std.unicode.utf8Encode(cs[k], &buf) catch unreachable;
        const key = try arena.dupe(u8, buf[0..len]);
        try encoder.put(key, @intCast(k));
        try decode_bytes.append(gpa, try arena.dupe(u8, &[_]u8{bs[k]}));
    }

    // === parse merges in rank order, assigning id = 256 + rank ===
    var entries = std.ArrayList(Entry).empty;
    defer entries.deinit(gpa);

    var lines = std.mem.splitScalar(u8, data, '\n');
    if (lines.next()) |first| assert(std.mem.startsWith(u8, first, "#version")); // header line
    var rank: u32 = 0;
    while (lines.next()) |line_raw| {
        const line = std.mem.trimEnd(u8, line_raw, "\r");
        if (line.len == 0) continue;
        const sp = std.mem.indexOfScalar(u8, line, ' ') orelse return error.MalformedMergeLine;
        const a = line[0..sp];
        const b = line[sp + 1 ..];
        const ia = encoder.get(a) orelse return error.MergeRefersToUnknownToken;
        const ib = encoder.get(b) orelse return error.MergeRefersToUnknownToken;
        const mid: u32 = 256 + rank;

        const merged = try std.mem.concat(arena, u8, &.{ a, b });
        try encoder.put(merged, mid);
        try entries.append(gpa, .{ .key = (@as(u64, ia) << 32) | ib, .val = mid });

        const db = try std.mem.concat(arena, u8, &.{ decode_bytes.items[ia], decode_bytes.items[ib] });
        assert(decode_bytes.items.len == mid); // ids are dense and monotonic
        try decode_bytes.append(gpa, db);
        rank += 1;
    }
    assert(rank == N_MERGES_EXPECTED);

    // === special token: id 50256, decodes back to its literal text ===
    assert(decode_bytes.items.len == EOS_ID);
    try decode_bytes.append(gpa, EOS_TEXT);

    const n_merges: u32 = rank;
    const n_tokens: u32 = @intCast(decode_bytes.items.len); // 256 + 50000 + 1 = 50257

    // === sort merge entries by key for runtime binary search ===
    std.mem.sort(Entry, entries.items, {}, struct {
        fn lt(_: void, x: Entry, y: Entry) bool {
            return x.key < y.key;
        }
    }.lt);

    // ============================
    // === lay out + write file ===
    var blob_len: u32 = 0;
    for (decode_bytes.items) |db| blob_len += @intCast(db.len);

    const hdr_bytes: u32 = 9 * 4;
    const off_byte2id: u32 = hdr_bytes;
    const after_b2id = off_byte2id + 256 * 4;
    const off_keys: u32 = std.mem.alignForward(u32, after_b2id, 8); // u64 keys need 8-alignment
    const off_vals: u32 = off_keys + n_merges * 8;
    const off_doff: u32 = off_vals + n_merges * 4;
    const off_blob: u32 = off_doff + (n_tokens + 1) * 4;
    const total: u32 = off_blob + blob_len;

    const buf = try gpa.alloc(u8, total);
    defer gpa.free(buf);
    @memset(buf, 0);

    const hdr = [9]u32{ MAGIC, n_merges, n_tokens, off_byte2id, off_keys, off_vals, off_doff, off_blob, blob_len };
    for (hdr, 0..) |v, i| std.mem.writeInt(u32, buf[i * 4 ..][0..4], v, .little);

    for (byte_to_id, 0..) |v, i| std.mem.writeInt(u32, buf[off_byte2id + i * 4 ..][0..4], v, .little);
    for (entries.items, 0..) |e, i| std.mem.writeInt(u64, buf[off_keys + i * 8 ..][0..8], e.key, .little);
    for (entries.items, 0..) |e, i| std.mem.writeInt(u32, buf[off_vals + i * 4 ..][0..4], e.val, .little);

    var cursor: u32 = 0;
    for (decode_bytes.items, 0..) |db, id| {
        std.mem.writeInt(u32, buf[off_doff + id * 4 ..][0..4], cursor, .little);
        @memcpy(buf[off_blob + cursor ..][0..db.len], db);
        cursor += @intCast(db.len);
    }
    std.mem.writeInt(u32, buf[off_doff + n_tokens * 4 ..][0..4], cursor, .little); // sentinel end offset
    assert(cursor == blob_len);

    try cwd.writeFile(io, .{ .sub_path = out_path, .data = buf });
}
