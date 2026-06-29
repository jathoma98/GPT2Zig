//! Dynamic loading of libvulkan/MoltenVK + the instance-level function-pointer table. This file
//! deliberately depends on NOTHING but std and the translate-c'd Vulkan headers (`vulkan_c`) — it
//! is meant to run during early init-time sanity checks, so it must not pull in any project state.
//!
//! We don't use a generic loader (volk). Instead: dlopen the platform's Vulkan library, resolve
//! vkGetInstanceProcAddr from it, then load exactly the handful of entry points we use via that
//! bootstrap pointer. Library-name candidate lists mirror volk's loader order.

const std = @import("std");
const builtin = @import("builtin");
const c = @import("vulkan_c");

const log = std.log.scoped(.vk);

// =========================
// === Library loading ===

// volk's per-OS candidate order. We try each in turn and take the first that loads.
fn openVulkanLib() ?std.DynLib {
    const candidates: []const []const u8 = switch (builtin.os.tag) {
        // The explicit /usr/local/lib path matters: modern macOS doesn't search it for a bare
        // dlopen name, but the Vulkan SDK installs the loader there (volk hits the same case).
        .macos => &.{ "libvulkan.dylib", "libvulkan.1.dylib", "/usr/local/lib/libvulkan.dylib", "libMoltenVK.dylib" },
        .windows => &.{"vulkan-1.dll"},
        else => &.{ "libvulkan.so.1", "libvulkan.so" },
    };
    for (candidates) |name| {
        if (std.DynLib.open(name)) |lib| return lib else |_| {}
    }
    return null;
}

// Resolve a typed entry point through vkGetInstanceProcAddr. `instance` is null for the global
// commands (vkCreateInstance et al.) and the real handle for instance-level commands. Returns null
// (via the optional PFN type) when the loader doesn't know the name.
fn loadProc(comptime T: type, gipa: c.PFN_vkGetInstanceProcAddr, instance: c.VkInstance, name: [*:0]const u8) T {
    const p = gipa.?(instance, name);
    return @ptrCast(p);
}

// =====================
// === Init result ===

pub const InitOptions = struct {
    app_name: [*:0]const u8 = "GPT2Zig",
    enable_validation: bool = false,
    // When set, VK_EXT_debug_utils is enabled and a messenger using this callback is installed both
    // for instance create/destroy (via pNext) and persistently (vkCreateDebugUtilsMessengerEXT).
    debug_callback: c.PFN_vkDebugUtilsMessengerCallbackEXT = null,
    debug_user_data: ?*anyopaque = null,
};

// Outcome of an init attempt. Each variant carries exactly what the caller needs to react.
pub const Init = union(enum) {
    library_not_found,
    vk_error: c.VkResult, // a required-layer/extension miss or vkCreateInstance failure
    success: VTbl,
};

// =========================
// === Function table ===

pub const VTbl = struct {
    lib: std.DynLib,
    instance: c.VkInstance,
    messenger: c.VkDebugUtilsMessengerEXT,

    get_instance_proc_addr: c.PFN_vkGetInstanceProcAddr,

    // global (instance == null)
    createInstance: c.PFN_vkCreateInstance,
    enumerateInstanceLayerProperties: c.PFN_vkEnumerateInstanceLayerProperties,
    enumerateInstanceExtensionProperties: c.PFN_vkEnumerateInstanceExtensionProperties,
    enumerateInstanceVersion: c.PFN_vkEnumerateInstanceVersion,

    // instance-level
    destroyInstance: c.PFN_vkDestroyInstance,
    createDebugUtilsMessengerEXT: c.PFN_vkCreateDebugUtilsMessengerEXT,
    destroyDebugUtilsMessengerEXT: c.PFN_vkDestroyDebugUtilsMessengerEXT,

    pub fn deinit(self: *VTbl) void {
        if (self.messenger != null) {
            if (self.destroyDebugUtilsMessengerEXT) |f| f(self.instance, self.messenger, null);
        }
        if (self.destroyInstance) |f| f(self.instance, null);
        self.lib.close();
    }
};

// =================
// === Init ===

// Severity/type masks are c_int constants in the headers; the create-info fields are u32 flag
// words. Build the masks once at comptime.
const dbg_severity_all: u32 = @intCast(c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
    c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
    c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
    c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT);
const dbg_type_all: u32 = @intCast(c.VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
    c.VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
    c.VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT);

pub fn init(opts: InitOptions) Init {
    // --- load the library + bootstrap pointer ---
    var lib = openVulkanLib() orelse return .library_not_found;
    const gipa = lib.lookup(c.PFN_vkGetInstanceProcAddr, "vkGetInstanceProcAddr") orelse {
        lib.close();
        return .library_not_found;
    };

    // --- global commands ---
    const create_instance = loadProc(c.PFN_vkCreateInstance, gipa, null, "vkCreateInstance");
    const enum_layers = loadProc(c.PFN_vkEnumerateInstanceLayerProperties, gipa, null, "vkEnumerateInstanceLayerProperties");
    const enum_exts = loadProc(c.PFN_vkEnumerateInstanceExtensionProperties, gipa, null, "vkEnumerateInstanceExtensionProperties");
    const enum_version = loadProc(c.PFN_vkEnumerateInstanceVersion, gipa, null, "vkEnumerateInstanceVersion");

    if (builtin.mode == .Debug) logInstanceCapabilities(enum_layers, enum_exts);

    // --- enabled layers + extensions ---
    var layers: [1][*c]const u8 = undefined;
    var layer_count: u32 = 0;
    if (opts.enable_validation) {
        layers[layer_count] = "VK_LAYER_KHRONOS_validation";
        layer_count += 1;
    }

    var exts: [2][*c]const u8 = undefined;
    var ext_count: u32 = 0;
    if (opts.debug_callback != null) {
        exts[ext_count] = c.VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
        ext_count += 1;
    }
    // MoltenVK is a portability driver, so 1.3 instance creation must opt into enumerating it.
    var flags: c.VkInstanceCreateFlags = 0;
    if (builtin.os.tag == .macos) {
        exts[ext_count] = c.VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME;
        ext_count += 1;
        flags |= @intCast(c.VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR);
    }

    // --- create info (messenger chained into pNext so create/destroy is also covered) ---
    const app_info: c.VkApplicationInfo = .{
        .sType = @intCast(c.VK_STRUCTURE_TYPE_APPLICATION_INFO),
        .pApplicationName = opts.app_name,
        .pEngineName = "GPT2Zig",
        .apiVersion = c.VK_API_VERSION_1_3,
    };

    var dbg_ci: c.VkDebugUtilsMessengerCreateInfoEXT = .{
        .sType = @intCast(c.VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT),
        .messageSeverity = dbg_severity_all,
        .messageType = dbg_type_all,
        .pfnUserCallback = opts.debug_callback,
        .pUserData = opts.debug_user_data,
    };

    const inst_ci: c.VkInstanceCreateInfo = .{
        .sType = @intCast(c.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO),
        .pNext = if (opts.debug_callback != null) &dbg_ci else null,
        .flags = flags,
        .pApplicationInfo = &app_info,
        .enabledLayerCount = layer_count,
        .ppEnabledLayerNames = if (layer_count > 0) &layers else null,
        .enabledExtensionCount = ext_count,
        .ppEnabledExtensionNames = if (ext_count > 0) &exts else null,
    };

    var instance: c.VkInstance = null;
    const res = create_instance.?(&inst_ci, null, &instance);
    if (res != c.VK_SUCCESS) {
        lib.close();
        return .{ .vk_error = res };
    }

    // --- instance-level commands ---
    const destroy_instance = loadProc(c.PFN_vkDestroyInstance, gipa, instance, "vkDestroyInstance");
    const create_msgr = loadProc(c.PFN_vkCreateDebugUtilsMessengerEXT, gipa, instance, "vkCreateDebugUtilsMessengerEXT");
    const destroy_msgr = loadProc(c.PFN_vkDestroyDebugUtilsMessengerEXT, gipa, instance, "vkDestroyDebugUtilsMessengerEXT");

    var messenger: c.VkDebugUtilsMessengerEXT = null;
    if (opts.debug_callback != null) {
        if (create_msgr) |f| _ = f(instance, &dbg_ci, null, &messenger);
    }

    return .{ .success = .{
        .lib = lib,
        .instance = instance,
        .messenger = messenger,
        .get_instance_proc_addr = gipa,
        .createInstance = create_instance,
        .enumerateInstanceLayerProperties = enum_layers,
        .enumerateInstanceExtensionProperties = enum_exts,
        .enumerateInstanceVersion = enum_version,
        .destroyInstance = destroy_instance,
        .createDebugUtilsMessengerEXT = create_msgr,
        .destroyDebugUtilsMessengerEXT = destroy_msgr,
    } };
}

// The generic "skip if Vulkan/VVL unavailable" gate, lifted out of the instance test so every GPU
// golden test (here and in src/core/gpu.zig) shares one policy: a missing loader / ICD / validation
// layer / debug-utils extension is an environment skip; any other instance-create failure is a real
// test failure. Device creation is NOT gated here — once the instance exists we expect a device, so
// initDevice failure is a hard error (see device.zig).
pub fn instanceOrSkip(opts: InitOptions) error{ SkipZigTest, VkInstanceCreateFailed }!VTbl {
    switch (init(opts)) {
        .library_not_found => return error.SkipZigTest,
        .vk_error => |code| switch (code) {
            c.VK_ERROR_LAYER_NOT_PRESENT,
            c.VK_ERROR_EXTENSION_NOT_PRESENT,
            c.VK_ERROR_INCOMPATIBLE_DRIVER,
            => return error.SkipZigTest,
            else => {
                log.err("vkCreateInstance failed: {d}", .{code});
                return error.VkInstanceCreateFailed;
            },
        },
        .success => |vtbl| return vtbl,
    }
}

// Debug-only: dump the layers/extensions the loader advertises. Fixed stack buffers cap the count
// we materialize (the total is still logged); no allocation in the init path.
fn logInstanceCapabilities(
    enum_layers: c.PFN_vkEnumerateInstanceLayerProperties,
    enum_exts: c.PFN_vkEnumerateInstanceExtensionProperties,
) void {
    if (enum_layers) |f| {
        var n: u32 = 0;
        _ = f(&n, null);
        var buf: [64]c.VkLayerProperties = undefined;
        const take: u32 = @min(n, buf.len);
        var got: u32 = take;
        _ = f(&got, &buf);
        log.debug("{d} instance layer(s):", .{n});
        for (buf[0..take]) |p| log.debug("  layer: {s}", .{std.mem.sliceTo(&p.layerName, 0)});
    }
    if (enum_exts) |f| {
        var n: u32 = 0;
        _ = f(null, &n, null);
        var buf: [256]c.VkExtensionProperties = undefined;
        const take: u32 = @min(n, buf.len);
        var got: u32 = take;
        _ = f(null, &got, &buf);
        log.debug("{d} instance extension(s):", .{n});
        for (buf[0..take]) |p| log.debug("  ext: {s}", .{std.mem.sliceTo(&p.extensionName, 0)});
    }
}

// ====================================================
// === Test validation capture (shared mechanism) ===

// Tests pass a pointer to one of these as pUserData. The messenger sets `.fired` on any
// warning/error-severity message, and the test asserts it stayed false. No globals — state is
// threaded through pUserData so concurrent tests don't clobber each other.
pub const ValidationCapture = struct {
    fired: bool = false,

    // Defer-able assertion for tests: `defer cap.assertNoValidationErrors();`. A defer can't
    // propagate an error, so a fired validation message panics (which fails the running test) —
    // the detailed message was already logged at .err by captureCallback.
    pub fn assertNoValidationErrors(self: *const ValidationCapture) void {
        if (self.fired) @panic("Vulkan validation error(s) fired — see the validation log above");
    }
};

const severity_warn_or_err: u32 = @intCast(c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
    c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT);

// Runtime debug callback (production, when validation is enabled in Debug builds): a validation
// warning/error is a programming bug, so log it and abort immediately rather than let the engine
// limp on with undefined GPU behavior. Tests use captureCallback instead (collect + assert).
pub fn panicCallback(
    severity: c.VkDebugUtilsMessageSeverityFlagBitsEXT,
    types: c.VkDebugUtilsMessageTypeFlagsEXT,
    data: [*c]const c.VkDebugUtilsMessengerCallbackDataEXT,
    user: ?*anyopaque,
) callconv(.c) c.VkBool32 {
    _ = types;
    _ = user;
    const sev: u32 = @intCast(severity);
    const msg: [*c]const u8 = if (data != null) data.*.pMessage else null;
    const text = if (msg != null) std.mem.sliceTo(msg, 0) else "(no message)";
    if (sev & severity_warn_or_err != 0) {
        log.err("validation: {s}", .{text});
        @panic("Vulkan validation error fired at runtime");
    } else {
        log.debug("validation: {s}", .{text});
    }
    return 0;
}

pub fn captureCallback(
    severity: c.VkDebugUtilsMessageSeverityFlagBitsEXT,
    types: c.VkDebugUtilsMessageTypeFlagsEXT,
    data: [*c]const c.VkDebugUtilsMessengerCallbackDataEXT,
    user: ?*anyopaque,
) callconv(.c) c.VkBool32 {
    _ = types;
    const sev: u32 = @intCast(severity);
    const msg: [*c]const u8 = if (data != null) data.*.pMessage else null;
    const text = if (msg != null) std.mem.sliceTo(msg, 0) else "(no message)";

    if (sev & severity_warn_or_err != 0) {
        log.err("validation: {s}", .{text});
        if (user) |u| {
            const cap: *ValidationCapture = @ptrCast(@alignCast(u));
            cap.fired = true;
        }
    } else {
        log.debug("validation: {s}", .{text});
    }
    return 0; // VK_FALSE: never abort the triggering call
}

// =============
// === Tests ===

test "instance init + debug messenger sees no validation errors" {
    var cap: ValidationCapture = .{};
    var vtbl = instanceOrSkip(.{
        .enable_validation = true,
        .debug_callback = captureCallback,
        .debug_user_data = &cap,
    }) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        error.VkInstanceCreateFailed => return error.VkInstanceCreateFailed,
    };
    defer vtbl.deinit();
    defer cap.assertNoValidationErrors();
    try std.testing.expect(vtbl.instance != null);
}
