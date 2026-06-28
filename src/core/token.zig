//! M4 GPT-2 byte-level BPE tokenizer. Consumes the packed table that tools/gen_bpe.zig emits
//! (src/generated/bpe_tokenizer.bin) by mmap — runtime does zero construction: slice-cast the
//! sections, binary-search for merges, index for decode. See gen_bpe.zig for the format and the
//! id-derivation facts.
//!
//! encode pipeline:  special-split → pre-tokenize (GPT-2 regex) → per-chunk BPE in id-space.
//! BPE works on u32 ids (not strings): the merge table is keyed on (id_a,id_b) so no remapped-string
//! reconstruction is ever needed; the merged id IS 256+rank, and lower id == lower rank == merge
//! first.
//!
//! Unicode caveat: the GPT-2 pre-tokenizer regex uses \p{L}/\p{N}. Rather than vendor megabytes of
//! Unicode category tables, we treat ASCII letters/digits exactly and ANY non-ASCII codepoint
//! (>= 0x80) as a letter. Exact for all ASCII and for the goldens (incl. "café"); round-trip holds
//! for all input regardless. Can diverge from tiktoken only on exotic non-ASCII symbols/digits/
//! whitespace (emoji, CJK punctuation, superscripts) — none of which appear in the golden set.
const std = @import("std");
const assert = std.debug.assert;

const page = std.heap.page_size_min;

pub const BPE_BIN = "src/generated/bpe_tokenizer.bin";

const MAGIC: u32 = 0x42504531;
const SPECIAL = "<|endoftext|>";

// A single pre-token chunk's initial byte count bounds the BPE working set. 64 KiB comfortably
// covers any realistic word/whitespace run; a longer single chunk asserts rather than corrupts.
const MAX_CHUNK = 1 << 16;

pub const Tokenizer = struct {
    bytes: []align(page) const u8, // the mmap; munmap'd in deinit
    byte_to_id: []align(4) const u32, // [256]
    merge_keys: []align(8) const u64, // [n_merges] sorted ascending
    merge_vals: []align(4) const u32, // [n_merges] parallel
    decode_offsets: []align(4) const u32, // [n_tokens+1] prefix offsets into decode_blob
    decode_blob: []const u8,
    n_tokens: u32,
    work: []u32, // reused per chunk; len == MAX_CHUNK

    pub fn init(io: std.Io, path: []const u8) !Tokenizer {
        const file = try std.Io.Dir.cwd().openFile(io, path, .{});
        defer file.close(io);
        const size = (try file.stat(io)).size;
        assert(size >= 36);
        const raw = try std.posix.mmap(null, size, .{ .READ = true }, .{ .TYPE = .PRIVATE }, file.handle, 0);

        var hdr: [9]u32 = undefined;
        for (&hdr, 0..) |*h, i| h.* = std.mem.readInt(u32, raw[i * 4 ..][0..4], .little);
        assert(hdr[0] == MAGIC);
        const n_merges = hdr[1];
        const n_tokens = hdr[2];
        const off_b2id = hdr[3];
        const off_keys = hdr[4];
        const off_vals = hdr[5];
        const off_doff = hdr[6];
        const off_blob = hdr[7];
        const blob_len = hdr[8];
        assert(off_keys % 8 == 0); // u64 section must be 8-aligned within the page-aligned mmap
        assert(off_blob + blob_len == size);

        const work = try std.heap.page_allocator.alloc(u32, MAX_CHUNK);

        return .{
            .bytes = raw,
            .byte_to_id = sliceT(u32, raw, off_b2id, 256),
            .merge_keys = sliceT(u64, raw, off_keys, n_merges),
            .merge_vals = sliceT(u32, raw, off_vals, n_merges),
            .decode_offsets = sliceT(u32, raw, off_doff, n_tokens + 1),
            .decode_blob = raw[off_blob..][0..blob_len],
            .n_tokens = n_tokens,
            .work = work,
        };
    }

    pub fn deinit(self: *Tokenizer) void {
        std.heap.page_allocator.free(self.work);
        std.posix.munmap(self.bytes);
    }

    // Reinterpret a section of the page-aligned mmap as []T. Sound because every section offset the
    // tool emits is a multiple of @alignOf(T) and the mmap base is page-aligned.
    fn sliceT(comptime T: type, raw: []align(page) const u8, off: u32, n: u32) []align(@alignOf(T)) const T {
        const want = @as(usize, n) * @sizeOf(T);
        const seg: []align(@alignOf(T)) const u8 = @alignCast(raw[off..][0..want]);
        return std.mem.bytesAsSlice(T, seg);
    }

    pub fn decodeToken(self: *const Tokenizer, id: u32) []const u8 {
        assert(id < self.n_tokens);
        return self.decode_blob[self.decode_offsets[id]..self.decode_offsets[id + 1]];
    }

    // Binary search the sorted merge table. Returns the merged id (== 256+rank) or null.
    fn mergeId(self: *const Tokenizer, key: u64) ?u32 {
        var lo: usize = 0;
        var hi: usize = self.merge_keys.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            const k = self.merge_keys[mid];
            if (k < key) lo = mid + 1 else if (k > key) hi = mid else return self.merge_vals[mid];
        }
        return null;
    }

    // text → token ids written into `out` (caller-owned, sized to the model's n_ctx). Returns count.
    pub fn encode(self: *Tokenizer, text: []const u8, out: []u32) usize {
        var count: usize = 0;
        var i: usize = 0;
        while (i < text.len) {
            if (std.mem.indexOfPos(u8, text, i, SPECIAL)) |sp| {
                count = self.encodeSpan(text[i..sp], out, count);
                assert(count < out.len);
                out[count] = 50256; // <|endoftext|>, emitted atomically
                count += 1;
                i = sp + SPECIAL.len;
            } else {
                return self.encodeSpan(text[i..], out, count);
            }
        }
        return count;
    }

    fn encodeSpan(self: *Tokenizer, span: []const u8, out: []u32, count_in: usize) usize {
        var count = count_in;
        var it = PreTok{ .text = span, .i = 0 };
        while (it.next()) |chunk| count = self.bpeChunk(chunk, out, count);
        return count;
    }

    // Pure id-space BPE on one pre-token chunk. Greedily merges the lowest-rank adjacent bigram
    // (lowest merged id) until none remain, then appends the surviving ids to `out`.
    fn bpeChunk(self: *Tokenizer, chunk: []const u8, out: []u32, count_in: usize) usize {
        assert(chunk.len <= self.work.len);
        var len: usize = 0;
        for (chunk) |b| {
            self.work[len] = self.byte_to_id[b];
            len += 1;
        }

        while (len >= 2) {
            var best: u32 = std.math.maxInt(u32);
            var best_i: usize = 0;
            var found = false;
            var k: usize = 0;
            while (k + 1 < len) : (k += 1) {
                const key = (@as(u64, self.work[k]) << 32) | self.work[k + 1];
                if (self.mergeId(key)) |mid| {
                    if (mid < best) {
                        best = mid;
                        best_i = k;
                        found = true;
                    }
                }
            }
            if (!found) break;
            self.work[best_i] = best;
            var j = best_i + 1;
            while (j + 1 < len) : (j += 1) self.work[j] = self.work[j + 1];
            len -= 1;
        }

        var count = count_in;
        for (self.work[0..len]) |id| {
            assert(count < out.len);
            out[count] = id;
            count += 1;
        }
        return count;
    }
};

// =================================
// === Pre-tokenizer (GPT-2 regex) ===
//
// Reproduces the priority of  's|'t|'re|'ve|'m|'ll|'d | ?\p{L}+ | ?\p{N}+ | ?[^\s\p{L}\p{N}]+ |
// \s+(?!\S) | \s+  as a hand-coded scanner over UTF-8 codepoints. The optional leading space in the
// letter/number/other arms is a literal 0x20 only (not any \s) — that's why " word" attaches but
// "\tword" does not. The \s+(?!\S) rule hands the LAST space of a multi-space run to the following
// token (so "a  b" → "a", " ", "Ġb").

const Cat = enum { letter, number, other };

fn isWs(cp: u21) bool {
    return cp == ' ' or cp == '\t' or cp == '\n' or cp == '\r' or cp == 0x0b or cp == 0x0c;
}

// Approximation (see file header): ASCII letters exactly; any non-ASCII codepoint is a "letter".
fn catOf(cp: u21) ?Cat {
    if ((cp >= 'a' and cp <= 'z') or (cp >= 'A' and cp <= 'Z') or cp >= 0x80) return .letter;
    if (cp >= '0' and cp <= '9') return .number;
    if (isWs(cp)) return null; // whitespace is handled by its own arm, never a run category
    return .other;
}

const Decoded = struct { cp: u21, len: usize };

fn cpAt(text: []const u8, i: usize) Decoded {
    const b = text[i];
    const l = std.unicode.utf8ByteSequenceLength(b) catch return .{ .cp = b, .len = 1 };
    if (i + l > text.len) return .{ .cp = b, .len = 1 };
    const cp = std.unicode.utf8Decode(text[i .. i + l]) catch return .{ .cp = b, .len = 1 };
    return .{ .cp = cp, .len = l };
}

const PreTok = struct {
    text: []const u8,
    i: usize,

    fn next(self: *PreTok) ?[]const u8 {
        const t = self.text;
        const start = self.i;
        if (start >= t.len) return null;
        const first = cpAt(t, start);
        const c0 = first.cp;

        // --- contractions: 's 't 're 've 'm 'll 'd (lowercase, in regex order) ---
        if (c0 == '\'') {
            const rem = t[start + 1 ..];
            if (startsWith(rem, "re") or startsWith(rem, "ve") or startsWith(rem, "ll")) {
                self.i = start + 3;
                return t[start..self.i];
            }
            if (rem.len >= 1 and (rem[0] == 's' or rem[0] == 't' or rem[0] == 'm' or rem[0] == 'd')) {
                self.i = start + 2;
                return t[start..self.i];
            }
            // bare apostrophe: falls through to the "other" arm
        }

        // --- ?\p{L}+ | ?\p{N}+ | ?[^\s\p{L}\p{N}]+  (one optional leading 0x20) ---
        {
            var j = start;
            var has_space = false;
            if (c0 == ' ' and start + 1 < t.len) {
                const nx = cpAt(t, start + 1);
                if (catOf(nx.cp) != null) {
                    has_space = true;
                    j = start + 1;
                }
            }
            if (has_space or c0 != ' ') {
                const head = if (has_space) cpAt(t, j) else first;
                if (catOf(head.cp)) |category| {
                    var k = j;
                    while (k < t.len) {
                        const w = cpAt(t, k);
                        const wc = catOf(w.cp);
                        if (wc == null or wc.? != category) break;
                        k += w.len;
                    }
                    self.i = k;
                    return t[start..k];
                }
            }
        }

        // --- \s+(?!\S) | \s+  (c0 is whitespace here) ---
        var k = start;
        while (k < t.len) {
            const w = cpAt(t, k);
            if (!isWs(w.cp)) break;
            k += w.len;
        }
        // ws chars are all single-byte ASCII, so (k - start) counts them. If a non-ws char follows a
        // run of >=2, leave the last space for the next token's optional leading-0x20.
        var end = k;
        if (k < t.len and (k - start) >= 2) end = k - 1;
        self.i = end;
        return t[start..end];
    }
};

fn startsWith(s: []const u8, prefix: []const u8) bool {
    return std.mem.startsWith(u8, s, prefix);
}

// =================
// === Tests ===

const testing = std.testing;

fn openTok() !Tokenizer {
    return Tokenizer.init(testing.io, BPE_BIN) catch |e| switch (e) {
        error.FileNotFound => {
            std.debug.print("missing {s} — run `zig build` to generate it\n", .{BPE_BIN});
            return error.SkipZigTest;
        },
        else => e,
    };
}

test "encode matches tiktoken goldens" {
    const golden = @import("../generated/tokenizer_golden.zig");
    var tok = try openTok();
    defer tok.deinit();

    var buf: [256]u32 = undefined;
    for (golden.cases) |c| {
        const n = tok.encode(c.in, &buf);
        testing.expectEqualSlices(u32, c.out, buf[0..n]) catch |e| {
            std.debug.print("case in={s}\n  want={any}\n  got ={any}\n", .{ c.in, c.out, buf[0..n] });
            return e;
        };
    }
}

test "encode anchor: Hello, I'm == M3 prompt" {
    // The M3 forward pass was validated against ids [15496, 11, 314, 1101]; those decode to
    // "Hello, I'm" (1101 = "'m"), NOT "Hello, I am" (which is [15496, 11, 314, 716]). The M3
    // generator scripts mislabel the prompt — this anchor pins the actual text↔id agreement.
    var tok = try openTok();
    defer tok.deinit();
    var buf: [16]u32 = undefined;
    const n = tok.encode("Hello, I'm", &buf);
    try testing.expectEqualSlices(u32, &.{ 15496, 11, 314, 1101 }, buf[0..n]);
}

test "special token is atomic" {
    var tok = try openTok();
    defer tok.deinit();
    var buf: [16]u32 = undefined;

    try testing.expectEqual(@as(usize, 1), tok.encode("<|endoftext|>", &buf));
    try testing.expectEqual(@as(u32, 50256), buf[0]);

    const n = tok.encode("a<|endoftext|>b", &buf);
    var saw_eos = false;
    for (buf[0..n]) |id| if (id == 50256) {
        saw_eos = true;
    };
    try testing.expect(saw_eos);
}

test "decode(encode(s)) round-trips" {
    var tok = try openTok();
    defer tok.deinit();

    const cases = [_][]const u8{
        "hello world",
        " leading space",
        "trailing space ",
        "a  b   c",
        "tab\tand\nnewline",
        "don't they're I'm",
        "café résumé naïve",
        "emoji 🚀 and 中文 mixed",
        "2024-06-27T12:00",
        "!!!??? ...",
        "",
        "<|endoftext|>",
        "before<|endoftext|>after",
    };
    var ids: [1024]u32 = undefined;
    var out: [4096]u8 = undefined;
    for (cases) |s| {
        const n = tok.encode(s, &ids);
        var len: usize = 0;
        for (ids[0..n]) |id| {
            const b = tok.decodeToken(id);
            @memcpy(out[len..][0..b.len], b);
            len += b.len;
        }
        try testing.expectEqualStrings(s, out[0..len]);
    }
}

test "decodeToken known ids" {
    var tok = try openTok();
    defer tok.deinit();
    try testing.expectEqualStrings("Hello", tok.decodeToken(15496));
    try testing.expectEqualStrings(" I", tok.decodeToken(314));
    try testing.expectEqualStrings("<|endoftext|>", tok.decodeToken(50256));
}
