const std = @import("std");
const Io = std.Io;

const GPT2Zig = @import("GPT2Zig");

pub fn main(init: std.process.Init) !void {
    const io = init.io;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const stdout_writer = &stdout_file_writer.interface;

    try GPT2Zig.printAnotherMessage(stdout_writer);

    try stdout_writer.flush();
}
