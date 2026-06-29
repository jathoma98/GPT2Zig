//! Public surface of the Vulkan module. The translate-c'd headers are exposed as `vk.c`; runtime
//! loading + the instance-level function table live in vtbl.zig. This module's only dependency is
//! the generated bindings — keep it that way (init-time sanity checks rely on it being standalone).

pub const c = @import("vulkan_c");
pub const vtbl = @import("vtbl.zig");

pub const Init = vtbl.Init;
pub const VTbl = vtbl.VTbl;
pub const InitOptions = vtbl.InitOptions;
pub const init = vtbl.init;

test {
    _ = vtbl;
}
