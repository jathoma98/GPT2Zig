//! Single source of truth for the binary's embedded assets. Every runtime/test asset is reached
//! through here via @embedFile. The files are wired as build-graph anonymous imports in build.zig
//! (`mod.addAnonymousImport("asset_xxx", ...)`) because @embedFile cannot reference a path that
//! escapes the module root (`src/`) with `..`, and because some assets are build-step outputs.
//!
//! Exceptions / non-members:
//! - model.safetensors (523 MB) is NOT embedded for now — too large to bake into every build; it's
//!   mmap'd from `model_safetensors_path` at runtime and will be supplied differently later. This
//!   module still owns the path so there's one place that names it.
//! - The generated .zig goldens (kernel/safetensors/tokenizer) are compiled-in *source code*, not
//!   binary assets, and keep their own relative @import. They don't belong here.
const build_options = @import("build_options");

// Model weights: runtime path (the single embed exception).
pub const model_safetensors_path: []const u8 = "models/gpt2/model.safetensors";

// config.json (~700 B) — embedded; parsed by Config.fromBytes.
pub const config_json: []const u8 = @embedFile("asset_config");

// BPE tokenizer table (tools/gen_bpe.zig output, ~1.1 MB). Its packed format has a u64 section that
// needs 8-byte alignment; @embedFile is alignment-1, so force an 8-aligned rodata copy at comptime.
const bpe_raw align(8) = @embedFile("asset_bpe").*;
pub const bpe: []align(8) const u8 = &bpe_raw;

// Activation goldens (M3 forward-pass oracle). All-or-nothing: they're produced by the slow
// `zig build gen-goldens` step, so build.zig sets goldens_embedded=true only when ALL act_*.bin
// exist on disk. The whole set is one nullable struct — callers null-check once, then dereference
// members freely — rather than a `?` per field. When absent the @embedFile branch isn't compiled
// (so the imports need not exist) and the forward-pass test skips.
pub const Goldens = struct {
    embed: []const u8,
    l0_ln1: []const u8,
    l0_attn: []const u8,
    l0_resid1: []const u8,
    l0_mlp: []const u8,
    l0_out: []const u8,
    l5_out: []const u8,
    lnf: []const u8,
    logits: []const u8,
};

pub const act_goldens: ?Goldens = if (build_options.goldens_embedded) .{
    .embed = @embedFile("asset_act_embed"),
    .l0_ln1 = @embedFile("asset_act_l0_ln1"),
    .l0_attn = @embedFile("asset_act_l0_attn"),
    .l0_resid1 = @embedFile("asset_act_l0_resid1"),
    .l0_mlp = @embedFile("asset_act_l0_mlp"),
    .l0_out = @embedFile("asset_act_l0_out"),
    .l5_out = @embedFile("asset_act_l5_out"),
    .lnf = @embedFile("asset_act_lnf"),
    .logits = @embedFile("asset_act_logits"),
} else null;
