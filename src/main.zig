const std = @import("std");
const Io = std.Io;

const GPT2Zig = @import("GPT2Zig");
const Model = GPT2Zig.model.Model;
const Config = GPT2Zig.config.Config;
const SafeTensors = GPT2Zig.safetensors.SafeTensors;

const MODEL = "models/gpt2/model.safetensors";
const CONFIG = "models/gpt2/config.json";

// Hardcoded "Hello, I'm" — M4 (tokenizer) isn't wired yet, so we feed ids directly and print the
// predicted next token id. Reconstructed text waits on the tokenizer.
const PROMPT = [_]u32{ 15496, 11, 314, 1101 };

pub fn main(init: std.process.Init) !void {
    const io = init.io;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const out = &stdout_file_writer.interface;

    var st = try SafeTensors.init(io, MODEL);
    const cfg = try Config.fromFile(io, CONFIG);
    var model = try Model.init(&st, cfg);
    st.deinit(); // weights copied into the model; mmap no longer referenced
    defer model.deinit();

    const vocab: usize = cfg.vocab_size;
    const logits = try std.heap.page_allocator.alloc(f32, PROMPT.len * vocab);
    defer std.heap.page_allocator.free(logits);

    model.forward(&PROMPT, logits, .{});

    // Greedy: argmax over the last position's logits is the predicted next token.
    const last = logits[(PROMPT.len - 1) * vocab ..][0..vocab];
    var next: usize = 0;
    for (last, 0..) |v, i| {
        if (v > last[next]) next = i;
    }

    try out.print("prompt ids: {any}\n", .{PROMPT});
    try out.print("predicted next token id: {d}\n", .{next});
    try out.flush();
}
