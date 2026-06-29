//! Public surface of the Vulkan module. The translate-c'd headers are exposed as `vk.c`; runtime
//! loading + the instance-level function table live in vtbl.zig. This module's only dependency is
//! the generated bindings — keep it that way (init-time sanity checks rely on it being standalone).

pub const c = @import("vulkan_c");
pub const vtbl = @import("vtbl.zig");
pub const device = @import("device.zig");

// Instance-level (vtbl.zig).
pub const Init = vtbl.Init;
pub const VTbl = vtbl.VTbl;
pub const InitOptions = vtbl.InitOptions;
pub const init = vtbl.init;
pub const instanceOrSkip = vtbl.instanceOrSkip;
pub const ValidationCapture = vtbl.ValidationCapture;
pub const captureCallback = vtbl.captureCallback;
pub const panicCallback = vtbl.panicCallback;

// Device-level compute (device.zig).
pub const Device = device.Device;
pub const Buffer = device.Buffer;
pub const BufferBinding = device.BufferBinding;
pub const Pipeline = device.Pipeline;
pub const initDevice = device.initDevice;
pub const createBuffer = device.createBuffer;
pub const writeBuffer = device.writeBuffer;
pub const readBuffer = device.readBuffer;
pub const createComputePipeline = device.createComputePipeline;
pub const dispatch = device.dispatch;

test {
    _ = vtbl;
    _ = device;
}
