const std = @import("std");
const builtin = @import("builtin");

// ============================
// === Python Venv Bootstrap ===

// venv layout is OS-dependent: POSIX puts the interpreter in bin/, Windows in Scripts/ with a .exe.
const VENV_PYTHON = if (builtin.os.tag == .windows) "python/.venv/Scripts/python.exe" else "python/.venv/bin/python";
// System interpreter name also differs: python.org/Store installers expose `python`/`py` on Windows
// (and only sometimes `python3`), whereas POSIX canonically has `python3`. findProgram takes the
// whole list and returns the first that resolves on PATH.
const SYSTEM_PYTHON_NAMES: []const []const u8 = if (builtin.os.tag == .windows)
    &.{ "python", "python3", "py" }
else
    &.{ "python3", "python" };
const VENV_SENTINEL = "python/.venv/.deps_installed";
// download_model.py materializes this link (symlink on POSIX, directory junction on Windows) into
// the venv-local HF cache. Its presence is what distinguishes "deps ready" from "deps ready AND
// model present".
const MODEL_PATH = "models/gpt2/model.safetensors";

// Each variant owns exactly the data its stage needs; illegal combinations
// (e.g. install_deps with no venv present) are unreachable by construction.
const PythonVenvState = union(enum) {
    create_venv: struct { system_python: []const u8 },
    install_deps,
    download_model,
    ready,
    failed: []const u8,
};

fn reducePythonState(b: *std.Build) PythonVenvState {
    const sys_python = b.findProgram(SYSTEM_PYTHON_NAMES, &.{}) catch
        return .{ .failed = "Python 3 not found in PATH; install Python 3 to continue" };

    const io = b.graph.io;
    const cwd = std.Io.Dir.cwd();

    cwd.access(io, VENV_PYTHON, .{}) catch
        return .{ .create_venv = .{ .system_python = sys_python } };

    const sen = cwd.statFile(io, VENV_SENTINEL, .{}) catch return .install_deps;
    const req = cwd.statFile(io, "python/requirements.txt", .{}) catch return readyOrDownload(io, cwd);
    if (sen.mtime.nanoseconds < req.mtime.nanoseconds) return .install_deps;
    return readyOrDownload(io, cwd);
}

// Deps are installed; the only thing left to decide is whether the model artifact exists. A missing
// or dangling `models/gpt2` symlink (access follows links) drops us into the download stage — which
// must run before the graph compiles, since asset.zig @embedFile's config.json from that dir.
fn readyOrDownload(io: std.Io, cwd: std.Io.Dir) PythonVenvState {
    cwd.access(io, MODEL_PATH, .{}) catch return .download_model;
    return .ready;
}

fn ensureVenvReady(b: *std.Build) []const u8 {
    const io = b.graph.io;
    state: switch (reducePythonState(b)) {
        .create_venv => |p| continue :state transitionToInstallDeps(io, p.system_python),
        .install_deps => continue :state transitionToDownloadModel(io),
        .download_model => continue :state transitionToReady(io),
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

fn transitionToDownloadModel(io: std.Io) PythonVenvState {
    std.log.info("python: installing deps from python/requirements.txt", .{});
    runCommand(io, &.{ VENV_PYTHON, "-m", "pip", "install", "-r", "python/requirements.txt" }) catch
        return .{ .failed = "pip install failed" };
    const f = std.Io.Dir.cwd().createFile(io, VENV_SENTINEL, .{}) catch
        return .{ .failed = "failed to write venv sentinel" };
    f.close(io);
    return .download_model;
}

fn transitionToReady(io: std.Io) PythonVenvState {
    std.log.info("python: materializing gpt2 into the venv cache (python/.venv/hf_cache)", .{});
    runCommand(io, &.{ VENV_PYTHON, "python/download_model.py" }) catch
        return .{ .failed = "model download failed" };
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

// ================================
// === Slang Compiler Bootstrap ===
//
// slangc is downloaded prebuilt (matching the BUILD machine, not the Zig target) from GitHub
// releases and cached in the OS temp dir. The vendor/slang source tree is a backup only — we
// never compile it. This wires host-detection -> download (curl) -> extract (native std.zip) ->
// presence check; nothing consumes slangc yet. Mirrors the venv state machine above.

const SLANG_VERSION = "2026.12";
// slangc dynamically links libslang, so it only runs from a full extraction (bin/ + lib/). The
// presence of this binary is what distinguishes "needs download" from "ready".
const SLANG_EXE_NAME = if (builtin.os.tag == .windows) "slangc.exe" else "slangc";

const SlangState = union(enum) {
    download: struct { url: []const u8, archive_path: []const u8, dest_dir: []const u8 },
    extract: struct { archive_path: []const u8, dest_dir: []const u8 },
    ready,
    failed: []const u8,
};

// Persistent cross-build cache root. std.Build.tmpPath exists but is auto-cleaned on build
// success, which would force a redownload every run — so we resolve the OS temp dir from the
// environment instead (the portable equivalent of a hardcoded /tmp).
fn slangTempBase(b: *std.Build) []const u8 {
    const env = b.graph.environ_map;
    const base = if (builtin.os.tag == .windows)
        (env.get("TEMP") orelse env.get("TMP") orelse "C:\\Windows\\Temp")
    else
        (env.get("TMPDIR") orelse "/tmp");
    return b.pathJoin(&.{ base, "gpt-slang" });
}

fn slangcPath(b: *std.Build) []const u8 {
    return b.pathJoin(&.{ slangTempBase(b), "bin", SLANG_EXE_NAME });
}

fn reduceSlangState(b: *std.Build) SlangState {
    // builtin refers to the build machine here (build.zig runs on the host), so these map the
    // host onto the release-archive naming. os.tag/cpu.arch are large external enums whose other
    // members are all unsupported, so an else arm is justified over enumerating dozens of tags.
    const os: []const u8 = switch (builtin.os.tag) {
        .macos => "macos",
        .linux => "linux",
        .windows => "windows",
        else => return .{ .failed = "unsupported host OS for slang (need macos/linux/windows)" },
    };
    const arch: []const u8 = switch (builtin.cpu.arch) {
        .x86_64 => "x86_64",
        .aarch64 => "aarch64",
        else => return .{ .failed = "unsupported host arch for slang (need x86_64/aarch64)" },
    };

    const dest_dir = slangTempBase(b);

    const io = b.graph.io;
    std.Io.Dir.cwd().access(io, slangcPath(b), .{}) catch {
        const archive_name = b.fmt("slang-{s}-{s}-{s}.zip", .{ SLANG_VERSION, os, arch });
        return .{ .download = .{
            .url = b.fmt("https://github.com/shader-slang/slang/releases/download/v{s}/{s}", .{ SLANG_VERSION, archive_name }),
            .archive_path = b.pathJoin(&.{ dest_dir, archive_name }),
            .dest_dir = dest_dir,
        } };
    };
    return .ready;
}

fn ensureSlangReady(b: *std.Build) []const u8 {
    state: switch (reduceSlangState(b)) {
        .download => |p| continue :state transitionToSlangExtract(b, p.url, p.archive_path, p.dest_dir),
        .extract => |p| continue :state transitionToSlangReady(b, p.archive_path, p.dest_dir),
        .ready => return slangcPath(b),
        .failed => |msg| {
            std.log.err("{s}", .{msg});
            std.process.exit(1);
        },
    }
}

fn transitionToSlangExtract(b: *std.Build, url: []const u8, archive_path: []const u8, dest_dir: []const u8) SlangState {
    const io = b.graph.io;
    std.log.info("slang: downloading {s}", .{url});

    std.Io.Dir.cwd().createDirPath(io, dest_dir) catch return .{ .failed = "failed to create slang cache dir" };

    // The build runner compiles std.http.Client with TLS disabled (std.options.http_disable_tls),
    // so a native https fetch hits `unreachable` here. Shell out to curl instead — present on
    // macOS, Win10+, and modern Linux — which handles TLS and the GitHub -> asset-CDN redirect
    // (-L) and fails with a nonzero exit on HTTP errors (-f). Extraction stays native (std.zip).
    runCommand(io, &.{ "curl", "-fL", "-o", archive_path, url }) catch return .{ .failed = "slang download failed (curl)" };

    return .{ .extract = .{ .archive_path = archive_path, .dest_dir = dest_dir } };
}

fn transitionToSlangReady(b: *std.Build, archive_path: []const u8, dest_dir: []const u8) SlangState {
    const io = b.graph.io;
    const cwd = std.Io.Dir.cwd();
    std.log.info("slang: extracting to {s}", .{dest_dir});

    const archive = cwd.openFile(io, archive_path, .{}) catch return .{ .failed = "failed to open slang archive" };
    defer archive.close(io);
    var rbuf: [64 * 1024]u8 = undefined;
    var fr = archive.reader(io, &rbuf);

    var dest = cwd.openDir(io, dest_dir, .{}) catch return .{ .failed = "failed to open slang cache dir" };
    defer dest.close(io);

    std.zip.extract(dest, &fr, .{}) catch return .{ .failed = "slang extract failed" };

    // std.zip doesn't carry the Unix exec bit through (entries land 0644), so slangc comes out
    // non-executable. Restore it where an exec bit exists; on Windows execution is by extension
    // (has_executable_bit == false) so this whole block compiles out.
    if (std.Io.File.Permissions.has_executable_bit) {
        const exe = dest.openFile(io, b.pathJoin(&.{ "bin", SLANG_EXE_NAME }), .{}) catch
            return .{ .failed = "failed to open slangc to set exec bit" };
        defer exe.close(io);
        exe.setPermissions(io, std.Io.File.Permissions.fromMode(0o755)) catch
            return .{ .failed = "failed to set slangc exec bit" };
    }
    return .ready;
}

// ==========================
// === Slang shader pipeline ===
//
// The compute kernel set. Each is one .slang file in src/shaders/ (Regime 2: divergent kernels in
// separate files; see research/slang-overview.md §5.3). The order is irrelevant — every kernel is
// wired identically by addSlangKernel.
const SHADER_KERNELS = [_][]const u8{
    "matmul", "matmul_bt", "layernorm",   "softmax",      "gelu",
    "embed",  "add",       "attn_scores", "attn_softmax", "attn_weighted_sum",
};

// Wire one kernel into the build graph: slangc -> {.spv, .reflect.json}, then reflect_codegen
// -> <name>_bindings.zig, then expose both the SPIR-V blob and the generated bindings to `mod` as
// named anonymous imports (`@embedFile("<name>_spv")` / `@import("<name>_bindings")`). The data
// dependencies make editing a .slang re-run slangc -> re-run codegen -> recompile the host.
//
// `all_sources` is every .slang file: each is registered as a tracked input so editing ANY shader
// recompiles ALL kernels. This is coarser than slangc's -depfile (which tracks only the modules a
// kernel actually `import`s) but slangc emits Make-escaped Windows paths (`C\:\\...`) that Zig's
// depfile parser mis-reads as bad path names — so we trade precision for a portable, correct
// invalidation rule. Flags are the dev profile from §2: spirv_1_6 + scalar layout + row-major +
// warnings-as-errors + restrictive caps + precise FP (so GPU results track the CPU/numpy oracle).
fn addSlangKernel(
    b: *std.Build,
    mod: *std.Build.Module,
    codegen_exe: *std.Build.Step.Compile,
    slangc: []const u8,
    name: []const u8,
    all_sources: []const std.Build.LazyPath,
) void {
    const slang = b.addSystemCommand(&.{slangc});
    slang.addFileArg(b.path(b.fmt("src/shaders/{s}.slang", .{name}))); // the file being compiled
    for (all_sources) |src| slang.addFileInput(src); // any shader change -> recompile this kernel
    slang.addArgs(&.{
        "-target",                       "spirv",
        "-std",                          "2026",
        "-capability",                   "spirv_1_6",
        "-fvk-use-scalar-layout",        "-matrix-layout-row-major",
        "-warnings-as-errors",           "all",
        "-restrictive-capability-check", "-fp-mode",
        "precise",                       "-O0",
        "-g3",
    });
    slang.addArg("-reflection-json");
    const reflect_json = slang.addOutputFileArg(b.fmt("{s}.reflect.json", .{name}));
    slang.addArg("-o");
    const spv = slang.addOutputFileArg(b.fmt("{s}.spv", .{name}));

    const gen = b.addRunArtifact(codegen_exe);
    gen.addFileArg(reflect_json); // LazyPath input wires slang -> gen
    const bindings = gen.addOutputFileArg(b.fmt("{s}_bindings.zig", .{name}));

    mod.addAnonymousImport(b.fmt("{s}_spv", .{name}), .{ .root_source_file = spv });
    mod.addAnonymousImport(b.fmt("{s}_bindings", .{name}), .{ .root_source_file = bindings });
}

// ============
// === Build ===

pub fn build(b: *std.Build) void {
    const venv_python = ensureVenvReady(b);
    const slangc = ensureSlangReady(b);

    // Vulkan ICDs/MoltenVK are loaded into the process via dlopen and link libc themselves; the
    // host process must therefore be a libc build. On Linux the musl default would also produce a
    // statically-linked binary that can't host a glibc driver, so default the abi to gnu (a
    // user-supplied -Dtarget still overrides). macOS/Windows ignore the abi hint.
    const target = b.standardTargetOptions(.{
        .default_target = if (builtin.os.tag == .linux) .{ .abi = .gnu } else .{},
    });
    const optimize = b.standardOptimizeOption(.{});

    // ==========================
    // === Vulkan bindings ===
    //
    // translate-c the vendored Vulkan headers into a Zig module against the TARGET ABI (not the
    // host) so struct layouts/enum widths match the binary we ship. VK_NO_PROTOTYPES drops the
    // extern vk* function decls, leaving only PFN_* typedefs + types — we load every function
    // pointer ourselves (see src/vk/vtbl.zig), so nothing links against a system libvulkan.
    const vk_tc = b.addTranslateC(.{
        .root_source_file = b.path("src/vk/include/vulkan/vulkan.h"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    vk_tc.addIncludePath(b.path("src/vk/include"));
    vk_tc.defineCMacro("VK_NO_PROTOTYPES", "1");
    const vulkan_c = vk_tc.addModule("vulkan_c");

    // Dedicated module whose ONLY import is the bindings: vtbl.zig is used for early init-time
    // sanity checks, so it must stay free of any project dependency. A separate module makes that
    // structurally enforced — nothing but `vulkan_c` and std is in its import graph.
    const vk_mod = b.createModule(.{
        .root_source_file = b.path("src/vk/vk.zig"),
        .target = target,
        .link_libc = true,
        .imports = &.{
            .{ .name = "vulkan_c", .module = vulkan_c },
        },
    });

    const mod = b.addModule("GPT2Zig", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .imports = &.{
            .{ .name = "vk", .module = vk_mod },
        },
    });

    // === Slang kernels: slangc + reflect_codegen, exposed to `mod` as spv/bindings imports ===
    // The codegen tool runs on the BUILD host (b.graph.host), like gen_bpe. Each kernel's SPIR-V and
    // generated bindings become @embedFile/@import targets consumed by src/core/gpu.zig.
    const codegen_exe = b.addExecutable(.{
        .name = "reflect_codegen",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/reflect_codegen.zig"),
            .target = b.graph.host,
            .optimize = .Debug,
        }),
    });
    var shader_sources: [SHADER_KERNELS.len]std.Build.LazyPath = undefined;
    for (SHADER_KERNELS, 0..) |k, i| shader_sources[i] = b.path(b.fmt("src/shaders/{s}.slang", .{k}));
    for (SHADER_KERNELS) |k| addSlangKernel(b, mod, codegen_exe, slangc, k, &shader_sources);

    const exe = b.addExecutable(.{
        .name = "GPT2Zig",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "GPT2Zig", .module = mod },
            },
        }),
    });

    // Android (bionic) cross builds: Zig can't synthesize bionic libc, so the exe link needs the
    // NDK's crt objects + libc.so. Scope it to the exe via a libc file — a global --sysroot would
    // wrongly redirect the native host codegen tools (reflect_codegen, gen_bpe) at this same sysroot.
    if (target.result.abi == .android) {
        const ndk_sysroot = b.option([]const u8, "ndk-sysroot", "Android NDK toolchain sysroot (required for android targets)") orelse
            @panic("android target requires -Dndk-sysroot=<NDK>/toolchains/llvm/prebuilt/<host>/sysroot");
        const api = b.option(u32, "android-api", "Android API level for the NDK lib dir (default 29)") orelse 29;
        const triple = b.fmt("{s}-linux-android", .{@tagName(target.result.cpu.arch)});
        const libc_conf = b.fmt(
            \\include_dir={s}/usr/include
            \\sys_include_dir={s}/usr/include/{s}
            \\crt_dir={s}/usr/lib/{s}/{d}
            \\gcc_dir=
            \\msvc_lib_dir=
            \\kernel32_lib_dir=
            \\
        , .{ ndk_sysroot, ndk_sysroot, triple, ndk_sysroot, triple, api });
        const wf = b.addWriteFiles();
        exe.setLibCFile(wf.add("android-libc.conf", libc_conf));
        exe.step.dependOn(&wf.step);
        // crt_dir gives Zig the crt objects, but -lc/-lm/-ldl still need this on the link search path.
        exe.root_module.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/usr/lib/{s}/{d}", .{ ndk_sysroot, triple, api }) });
        // The NDK ships only shared libc/libm/libdl (.so, no .a); force a dynamic PIE so lld links
        // against those (and emits interpreter /system/bin/linker64) instead of hunting for .a files.
        exe.linkage = .dynamic;
    }

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

    // ==========================================
    // === Build-time tool: BPE table codegen ===
    //
    // A host-target Zig program transforms models/gpt2/merges.txt into a packed binary that
    // token.zig consumes with zero runtime construction. Compiling this is fast; doing the same
    // transform at comptime OOM-kills the compiler. The .bin is embedded straight from this Run
    // step's output (see asset_bpe below), so its LazyPath alone wires the compile→tool dependency.
    const bpe_tool = b.addExecutable(.{
        .name = "gen_bpe",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/gen_bpe.zig"),
            .target = b.graph.host,
            .optimize = .ReleaseFast,
        }),
    });
    const gen_bpe_run = b.addRunArtifact(bpe_tool);
    gen_bpe_run.addFileArg(b.path("models/gpt2/merges.txt")); // tracked input → cache-invalidates
    const bpe_bin = gen_bpe_run.addOutputFileArg("bpe_tokenizer.bin");

    // ===========================================================
    // === Embedded assets (single source of truth: asset.zig) ===
    //
    // @embedFile can't reference paths outside the module root (src/), and the bpe bin is a build
    // output, so each asset is exposed to asset.zig as a named build-graph import. Adding them to
    // `mod` makes them visible to both the exe (imports GPT2Zig) and the test build (which IS mod);
    // the LazyPath edges make every compile depend on the model download / bpe tool automatically.
    mod.addAnonymousImport("asset_config", .{ .root_source_file = b.path("models/gpt2/config.json") });
    mod.addAnonymousImport("asset_bpe", .{ .root_source_file = bpe_bin });

    // Activation goldens come from the slow `gen-goldens` step and may be absent. Embed them only
    // when all are present on disk; asset.zig reads `goldens_embedded` to expose them as optionals
    // (and the forward-pass test skips when null). Checking presence here is a configure-phase
    // decision, so a later `gen-goldens` run + rebuild flips the option on.
    const golden_names = [_][]const u8{ "embed", "l0_ln1", "l0_attn", "l0_resid1", "l0_mlp", "l0_out", "l5_out", "lnf", "logits" };
    var goldens_present = true;
    for (golden_names) |name| {
        std.Io.Dir.cwd().access(b.graph.io, b.fmt("src/generated/act_{s}.bin", .{name}), .{}) catch {
            goldens_present = false;
        };
    }
    const build_opts = b.addOptions();
    build_opts.addOption(bool, "goldens_embedded", goldens_present);
    mod.addOptions("build_options", build_opts);
    if (goldens_present) {
        for (golden_names) |name| {
            mod.addAnonymousImport(
                b.fmt("asset_act_{s}", .{name}),
                .{ .root_source_file = b.path(b.fmt("src/generated/act_{s}.bin", .{name})) },
            );
        }
    }

    // =================================
    // === Codegen: tokenizer golden ===

    // Runs gen_tokenizer_golden.py and writes tokenizer_golden.zig into the build cache.
    // Fast (~1s), cached on its inputs. The cache LazyPath is copied into src/generated/
    // below so a bare `zig test <file>` (IDE inline debugger) can @import it by path.
    const gen_tok_cmd = b.addSystemCommand(&.{
        venv_python,
        "python/gen_tokenizer_golden.py",
    });
    // Track the script itself so editing the case list re-runs codegen (argv alone won't bust it).
    gen_tok_cmd.addFileInput(b.path("python/gen_tokenizer_golden.py"));
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

    // The golden values are only *used* in tests, but `@import` loads its target file eagerly
    // during AstGen (even for imports inside test blocks), so any compile of these modules — exe
    // included — needs the generated files present on disk to load, regardless of whether the
    // contents are ever analyzed. Both the exe and the test build therefore depend on the sync.
    exe.step.dependOn(&sync_goldens.step);
    mod_tests.step.dependOn(&sync_goldens.step);

    // The vk module is separate from GPT2Zig, so its tests are invisible to mod_tests — give it
    // its own test artifact. It needs no generated goldens, only the translate-c output.
    const vk_tests = b.addTest(.{
        .name = "vk-tests",
        .root_module = vk_mod,
        .filters = test_filters,
    });
    const run_vk_tests = b.addRunArtifact(vk_tests);

    // The reflect_codegen host tool has its own unit tests (the compile-time-safety codegen is the
    // shader interface authority, so it's hardened against malformed/edge-case reflection JSON).
    const codegen_tests = b.addTest(.{
        .name = "reflect-codegen-tests",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/reflect_codegen.zig"),
            .target = b.graph.host,
            .optimize = .Debug,
        }),
        .filters = test_filters,
    });
    const run_codegen_tests = b.addRunArtifact(codegen_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_vk_tests.step);
    test_step.dependOn(&run_codegen_tests.step);

    // `zig build test-debug -Dtest-filter=<name>` builds (but does not run) the unit-test
    // binary to a stable path for the VSCode debugger. Subdir test files (core/*.zig) can't
    // be compiled by a bare `zig test <file>` because their `../` imports escape the
    // standalone module root — the full module graph here resolves them.
    const install_test = b.addInstallArtifact(mod_tests, .{ .dest_dir = .{ .override = .bin } });
    install_test.step.dependOn(&sync_goldens.step);
    const test_debug_step = b.step("test-debug", "Build the unit-test binary for debugging");
    test_debug_step.dependOn(&install_test.step);
}
