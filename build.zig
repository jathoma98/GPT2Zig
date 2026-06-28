const std = @import("std");

// ============================
// === Python Venv Bootstrap ===

const VENV_PYTHON = "python/.venv/bin/python";
const VENV_SENTINEL = "python/.venv/.deps_installed";

// Each variant owns exactly the data its stage needs; illegal combinations
// (e.g. install_deps with no venv present) are unreachable by construction.
const PythonVenvState = union(enum) {
    create_venv: struct { system_python: []const u8 },
    install_deps,
    ready,
    failed: []const u8,
};

fn reducePythonState(b: *std.Build) PythonVenvState {
    const sys_python = b.findProgram(&.{"python3"}, &.{}) catch
        return .{ .failed = "python3 not found in PATH; install Python 3 to continue" };

    const io = b.graph.io;
    const cwd = std.Io.Dir.cwd();

    cwd.access(io, VENV_PYTHON, .{}) catch
        return .{ .create_venv = .{ .system_python = sys_python } };

    const sen = cwd.statFile(io, VENV_SENTINEL, .{}) catch return .install_deps;
    const req = cwd.statFile(io, "python/requirements.txt", .{}) catch return .ready;
    return if (sen.mtime.nanoseconds < req.mtime.nanoseconds) .install_deps else .ready;
}

fn ensureVenvReady(b: *std.Build) []const u8 {
    const io = b.graph.io;
    state: switch (reducePythonState(b)) {
        .create_venv => |p| continue :state transitionToInstallDeps(io, p.system_python),
        .install_deps => continue :state transitionToReady(io),
        .ready => return VENV_PYTHON,
        .failed => |msg| {
            std.log.err("{s}", .{msg});
            std.process.exit(1);
        },
    }
}

fn transitionToInstallDeps(io: std.Io, system_python: []const u8) PythonVenvState {
    std.log.info("python: creating venv at python/.venv", .{});
    runCommand(io, &.{ system_python, "-m", "venv", "python/.venv" }) catch
        return .{ .failed = "failed to create Python venv" };
    return .install_deps;
}

fn transitionToReady(io: std.Io) PythonVenvState {
    std.log.info("python: installing deps from python/requirements.txt", .{});
    runCommand(io, &.{ VENV_PYTHON, "-m", "pip", "install", "-r", "python/requirements.txt" }) catch
        return .{ .failed = "pip install failed" };
    const f = std.Io.Dir.cwd().createFile(io, VENV_SENTINEL, .{}) catch
        return .{ .failed = "failed to write venv sentinel" };
    f.close(io);
    return .ready;
}

fn runCommand(io: std.Io, argv: []const []const u8) !void {
    var child = try std.process.spawn(io, .{ .argv = argv });
    const term = try child.wait(io);
    switch (term) {
        .exited => |code| if (code != 0) return error.CommandFailed,
        else => return error.CommandFailed,
    }
}

// ============
// === Build ===

pub fn build(b: *std.Build) void {
    const venv_python = ensureVenvReady(b);

    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mod = b.addModule("GPT2Zig", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
    });

    const exe = b.addExecutable(.{
        .name = "GPT2Zig",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "GPT2Zig", .module = mod },
            },
        }),
    });

    b.installArtifact(exe);

    const run_step = b.step("run", "Run the app");
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    // -Dtest-filter lets the VSCode debug launcher build a binary containing only the
    // test being debugged (filter is baked in at compile time). Empty → all tests.
    const test_filter = b.option([]const u8, "test-filter", "Only build/run tests matching this substring");
    const test_filters: []const []const u8 = if (test_filter) |f| &.{f} else &.{};

    const mod_tests = b.addTest(.{
        .name = "unit-tests",
        .root_module = mod,
        .filters = test_filters,
    });
    const run_mod_tests = b.addRunArtifact(mod_tests);

    // =================================
    // === Codegen: tokenizer golden ===

    // Runs gen_tokenizer_golden.py and writes tokenizer_golden.zig into the build cache.
    // Fast (~1s), cached on its inputs. The cache LazyPath is copied into src/generated/
    // below so a bare `zig test <file>` (IDE inline debugger) can @import it by path.
    const gen_tok_cmd = b.addSystemCommand(&.{
        venv_python,
        "python/gen_tokenizer_golden.py",
    });
    const tok_golden_zig = gen_tok_cmd.addOutputFileArg("tokenizer_golden.zig");

    // =====================================
    // === Codegen: safetensors golden ===

    // Reads the model header + two spot-check tensors; fast (~1s) on every `zig build test`.
    const gen_st_cmd = b.addSystemCommand(&.{
        venv_python,
        "python/gen_safetensors_golden.py",
    });
    // Register the model + config files as tracked inputs: build cache invalidates if they change.
    gen_st_cmd.addFileArg(b.path("models/gpt2/model.safetensors"));
    gen_st_cmd.addFileArg(b.path("models/gpt2/config.json"));
    const st_golden_zig = gen_st_cmd.addOutputFileArg("safetensors_golden.zig");

    // =================================
    // === Codegen: kernel golden ===

    // Fixed-seed numpy reference outputs for the M2 math kernels. No tracked file inputs —
    // inputs are RNG-seeded in the script, so the output is purely a function of the script.
    const gen_kernel_cmd = b.addSystemCommand(&.{
        venv_python,
        "python/gen_kernel_goldens.py",
    });
    const kernel_golden_zig = gen_kernel_cmd.addOutputFileArg("kernel_golden.zig");

    // Slow oracle refresh — also generates ref_logits.npy. Run manually when needed:
    //   zig build gen-goldens
    const gen_ref_cmd = b.addSystemCommand(&.{
        venv_python,
        "python/gen_ref_logits.py",
    });
    // M3 activation goldens: per-stage taps + final logits as self-describing raw-f32 .bin files,
    // written straight into src/generated/ (gitignored) and mmap'd by the forward-pass test.
    // Slow (loads HF gpt2), so it lives only in the manual gen-goldens step, not the test path.
    const gen_act_cmd = b.addSystemCommand(&.{
        venv_python,
        "python/gen_activation_goldens.py",
        "src/generated",
    });
    const gen_goldens_step = b.step("gen-goldens", "Regenerate all Python oracle files");
    gen_goldens_step.dependOn(&gen_tok_cmd.step);
    gen_goldens_step.dependOn(&gen_st_cmd.step);
    gen_goldens_step.dependOn(&gen_kernel_cmd.step);
    gen_goldens_step.dependOn(&gen_ref_cmd.step);
    gen_goldens_step.dependOn(&gen_act_cmd.step);

    // Copy the cached golden outputs to a fixed, gitignored source path. Tests @import them
    // by relative path ("generated/…") instead of as build-graph named modules, so the same
    // imports resolve under both `zig build test` and a standalone `zig test <file>`.
    const sync_goldens = b.addUpdateSourceFiles();
    sync_goldens.addCopyFileToSource(tok_golden_zig, "src/generated/tokenizer_golden.zig");
    sync_goldens.addCopyFileToSource(st_golden_zig, "src/generated/safetensors_golden.zig");
    sync_goldens.addCopyFileToSource(kernel_golden_zig, "src/generated/kernel_golden.zig");

    // Only the test build references the goldens (all @imports are inside test blocks), so
    // only the test compile needs them on disk first; the exe build does not.
    mod_tests.step.dependOn(&sync_goldens.step);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);

    // `zig build test-debug -Dtest-filter=<name>` builds (but does not run) the unit-test
    // binary to a stable path for the VSCode debugger. Subdir test files (core/*.zig) can't
    // be compiled by a bare `zig test <file>` because their `../` imports escape the
    // standalone module root — the full module graph here resolves them.
    const install_test = b.addInstallArtifact(mod_tests, .{ .dest_dir = .{ .override = .bin } });
    install_test.step.dependOn(&sync_goldens.step);
    const test_debug_step = b.step("test-debug", "Build the unit-test binary for debugging");
    test_debug_step.dependOn(&install_test.step);
}
