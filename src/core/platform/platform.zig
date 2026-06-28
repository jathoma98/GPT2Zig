//! Cross-platform shims for the handful of OS primitives we touch. Today that's exactly one thing:
//! mapping a whole file read-only into memory (the safetensors model loader). POSIX uses mmap;
//! Windows uses CreateFileMapping + MapViewOfFile. The platform split is a comptime-known `is_windows`
//! const so the untaken branch is never analyzed — that's what lets a Windows target compile without
//! tripping over `std.posix.mmap`, and lets a native build avoid referencing the kernel32 externs.
const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;
const windows = std.os.windows;

const is_windows = builtin.os.tag == .windows;

// Zig 0.16's std.os.windows.kernel32 wraps CloseHandle but NOT the file-mapping calls, so declare
// them. These are referenced only inside the `is_windows` branch of FileMap, so non-Windows builds
// neither analyze nor link them.
extern "kernel32" fn CreateFileMappingW(
    hFile: windows.HANDLE,
    lpFileMappingAttributes: ?*anyopaque,
    flProtect: windows.DWORD,
    dwMaximumSizeHigh: windows.DWORD,
    dwMaximumSizeLow: windows.DWORD,
    lpName: ?windows.LPCWSTR,
) callconv(.winapi) ?windows.HANDLE;
extern "kernel32" fn MapViewOfFile(
    hFileMappingObject: windows.HANDLE,
    dwDesiredAccess: windows.DWORD,
    dwFileOffsetHigh: windows.DWORD,
    dwFileOffsetLow: windows.DWORD,
    dwNumberOfBytesToMap: windows.SIZE_T,
) callconv(.winapi) ?windows.LPVOID;
extern "kernel32" fn UnmapViewOfFile(lpBaseAddress: ?*const anyopaque) callconv(.winapi) windows.BOOL;

const PAGE_READONLY: windows.DWORD = 0x02;
const FILE_MAP_READ: windows.DWORD = 0x0004;

/// A read-only memory map of an entire file. `bytes` is an unaligned view of the whole file; the
/// page alignment mmap happens to provide is not relied upon by any consumer (safetensors reads
/// everything through unaligned std.mem.readInt), so the public view is plain `[]const u8`.
pub const FileMap = struct {
    bytes: []const u8,
    // Kept solely for teardown: POSIX munmap wants the page-aligned mmap slice; Windows
    // UnmapViewOfFile wants the view base pointer.
    raw: if (is_windows) []const u8 else []align(std.heap.page_size_min) const u8,

    pub fn open(io: std.Io, path: []const u8) !FileMap {
        const file = try std.Io.Dir.cwd().openFile(io, path, .{});
        defer file.close(io); // the mapping keeps the file's pages alive after the fd/handle closes
        const size = (try file.stat(io)).size;
        assert(size > 0);

        if (is_windows) {
            // dwMaximumSize* = 0 ⇒ map exactly the file's current size.
            const mapping = CreateFileMappingW(file.handle, null, PAGE_READONLY, 0, 0, null) orelse
                return error.FileMapFailed;
            defer windows.CloseHandle(mapping); // the view survives closing the mapping handle
            const base = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0) orelse
                return error.FileMapFailed;
            const bytes = @as([*]const u8, @ptrCast(base))[0..size];
            return .{ .bytes = bytes, .raw = bytes };
        } else {
            const raw = try std.posix.mmap(null, size, .{ .READ = true }, .{ .TYPE = .PRIVATE }, file.handle, 0);
            return .{ .bytes = raw, .raw = raw };
        }
    }

    pub fn close(self: *FileMap) void {
        if (is_windows) {
            _ = UnmapViewOfFile(self.raw.ptr);
        } else {
            std.posix.munmap(self.raw);
        }
    }
};

// =================
// === Tests ===

test "FileMap round-trips a file's bytes" {
    const io = std.testing.io;
    const dir = std.Io.Dir.cwd();
    const path = "platform_filemap_test.bin";

    const payload = "the quick brown fox jumps over the lazy dog" ** 100;
    try dir.writeFile(io, .{ .sub_path = path, .data = payload });
    defer dir.deleteFile(io, path) catch {};

    var map = try FileMap.open(io, path);
    defer map.close();

    try std.testing.expectEqual(payload.len, map.bytes.len);
    try std.testing.expectEqualSlices(u8, payload, map.bytes);
}
