//! Generic Vulkan compute device layer. Like vtbl.zig this is a PURE Vulkan leaf — it knows
//! nothing about GPT-2, weights, layers, or kernels. It provides exactly the mechanisms the
//! business layer (src/core/gpu.zig) composes: pick a compute device, allocate buffers (preferring
//! DEVICE_LOCAL memory, tracking host-visibility/coherence so reads pick map-vs-copy), build a
//! compute pipeline from SPIR-V bytes + a binding count + a push-constant size, and dispatch it.
//!
//! The execution model is the simplest possible (performance is a non-goal): one command buffer,
//! reset + re-recorded per op, submitted and `vkQueueWaitIdle`'d immediately — serial, CPU-like,
//! no barriers. All synchronization is full-queue idle.
const std = @import("std");
const builtin = @import("builtin");
const c = @import("vulkan_c");
const vtbl = @import("vtbl.zig");

const log = std.log.scoped(.vk);
const VTbl = vtbl.VTbl;

// Re-exported so callers have one import surface (vk.instanceOrSkip / vk.initDevice / ...).
pub const instanceOrSkip = vtbl.instanceOrSkip;

// =========================
// === Flag constants ===
// Bit consts are c_int in the headers; the create-info flag fields are u32. Build the masks once.

const MEM_DEVICE_LOCAL: u32 = @intCast(c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
const MEM_HOST_VISIBLE: u32 = @intCast(c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
const MEM_HOST_COHERENT: u32 = @intCast(c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

const USAGE_WORKING: u32 = @intCast(c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
    c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | c.VK_BUFFER_USAGE_TRANSFER_DST_BIT);
const USAGE_STAGING: u32 = @intCast(c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
    c.VK_BUFFER_USAGE_TRANSFER_DST_BIT);

const STAGE_COMPUTE: u32 = @intCast(c.VK_SHADER_STAGE_COMPUTE_BIT);
const DTYPE_STORAGE: u32 = @intCast(c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

const MAX_BINDINGS = 8; // every kernel in this engine uses ≤4 storage buffers; 8 is generous slack.

const DeviceError = error{
    NoComputeDevice, // instance created but no physical device exposes a compute queue (skip-able)
    VkDeviceCreateFailed,
    VkCommandPoolFailed,
    VkOutOfMemory, // a Vulkan allocation/creation call returned non-success
    NoSuitableMemoryType,
    OutOfMemory, // Zig allocator
};

// =========================
// === Function tables ===

fn loadDev(comptime T: type, gdpa: c.PFN_vkGetDeviceProcAddr, device: c.VkDevice, name: [*:0]const u8) T {
    return @ptrCast(gdpa.?(device, name));
}

fn loadInst(comptime T: type, gipa: c.PFN_vkGetInstanceProcAddr, instance: c.VkInstance, name: [*:0]const u8) T {
    return @ptrCast(gipa.?(instance, name));
}

// Device-level entry points, loaded via vkGetDeviceProcAddr. One flat table; verbose but explicit.
const DeviceTbl = struct {
    getDeviceQueue: c.PFN_vkGetDeviceQueue,
    createCommandPool: c.PFN_vkCreateCommandPool,
    destroyCommandPool: c.PFN_vkDestroyCommandPool,
    allocateCommandBuffers: c.PFN_vkAllocateCommandBuffers,
    resetCommandBuffer: c.PFN_vkResetCommandBuffer,
    beginCommandBuffer: c.PFN_vkBeginCommandBuffer,
    endCommandBuffer: c.PFN_vkEndCommandBuffer,
    cmdBindPipeline: c.PFN_vkCmdBindPipeline,
    cmdBindDescriptorSets: c.PFN_vkCmdBindDescriptorSets,
    cmdPushConstants: c.PFN_vkCmdPushConstants,
    cmdDispatch: c.PFN_vkCmdDispatch,
    cmdCopyBuffer: c.PFN_vkCmdCopyBuffer,
    queueSubmit: c.PFN_vkQueueSubmit,
    queueWaitIdle: c.PFN_vkQueueWaitIdle,
    deviceWaitIdle: c.PFN_vkDeviceWaitIdle,
    createBuffer: c.PFN_vkCreateBuffer,
    destroyBuffer: c.PFN_vkDestroyBuffer,
    getBufferMemoryRequirements: c.PFN_vkGetBufferMemoryRequirements,
    allocateMemory: c.PFN_vkAllocateMemory,
    freeMemory: c.PFN_vkFreeMemory,
    bindBufferMemory: c.PFN_vkBindBufferMemory,
    mapMemory: c.PFN_vkMapMemory,
    unmapMemory: c.PFN_vkUnmapMemory,
    flushMappedMemoryRanges: c.PFN_vkFlushMappedMemoryRanges,
    invalidateMappedMemoryRanges: c.PFN_vkInvalidateMappedMemoryRanges,
    createShaderModule: c.PFN_vkCreateShaderModule,
    destroyShaderModule: c.PFN_vkDestroyShaderModule,
    createDescriptorSetLayout: c.PFN_vkCreateDescriptorSetLayout,
    destroyDescriptorSetLayout: c.PFN_vkDestroyDescriptorSetLayout,
    createPipelineLayout: c.PFN_vkCreatePipelineLayout,
    destroyPipelineLayout: c.PFN_vkDestroyPipelineLayout,
    createComputePipelines: c.PFN_vkCreateComputePipelines,
    destroyPipeline: c.PFN_vkDestroyPipeline,
    createDescriptorPool: c.PFN_vkCreateDescriptorPool,
    destroyDescriptorPool: c.PFN_vkDestroyDescriptorPool,
    allocateDescriptorSets: c.PFN_vkAllocateDescriptorSets,
    updateDescriptorSets: c.PFN_vkUpdateDescriptorSets,
    destroyDevice: c.PFN_vkDestroyDevice,

    fn load(gdpa: c.PFN_vkGetDeviceProcAddr, device: c.VkDevice) DeviceTbl {
        const L = struct {
            fn f(comptime T: type, g: c.PFN_vkGetDeviceProcAddr, d: c.VkDevice, n: [*:0]const u8) T {
                return loadDev(T, g, d, n);
            }
        }.f;
        return .{
            .getDeviceQueue = L(c.PFN_vkGetDeviceQueue, gdpa, device, "vkGetDeviceQueue"),
            .createCommandPool = L(c.PFN_vkCreateCommandPool, gdpa, device, "vkCreateCommandPool"),
            .destroyCommandPool = L(c.PFN_vkDestroyCommandPool, gdpa, device, "vkDestroyCommandPool"),
            .allocateCommandBuffers = L(c.PFN_vkAllocateCommandBuffers, gdpa, device, "vkAllocateCommandBuffers"),
            .resetCommandBuffer = L(c.PFN_vkResetCommandBuffer, gdpa, device, "vkResetCommandBuffer"),
            .beginCommandBuffer = L(c.PFN_vkBeginCommandBuffer, gdpa, device, "vkBeginCommandBuffer"),
            .endCommandBuffer = L(c.PFN_vkEndCommandBuffer, gdpa, device, "vkEndCommandBuffer"),
            .cmdBindPipeline = L(c.PFN_vkCmdBindPipeline, gdpa, device, "vkCmdBindPipeline"),
            .cmdBindDescriptorSets = L(c.PFN_vkCmdBindDescriptorSets, gdpa, device, "vkCmdBindDescriptorSets"),
            .cmdPushConstants = L(c.PFN_vkCmdPushConstants, gdpa, device, "vkCmdPushConstants"),
            .cmdDispatch = L(c.PFN_vkCmdDispatch, gdpa, device, "vkCmdDispatch"),
            .cmdCopyBuffer = L(c.PFN_vkCmdCopyBuffer, gdpa, device, "vkCmdCopyBuffer"),
            .queueSubmit = L(c.PFN_vkQueueSubmit, gdpa, device, "vkQueueSubmit"),
            .queueWaitIdle = L(c.PFN_vkQueueWaitIdle, gdpa, device, "vkQueueWaitIdle"),
            .deviceWaitIdle = L(c.PFN_vkDeviceWaitIdle, gdpa, device, "vkDeviceWaitIdle"),
            .createBuffer = L(c.PFN_vkCreateBuffer, gdpa, device, "vkCreateBuffer"),
            .destroyBuffer = L(c.PFN_vkDestroyBuffer, gdpa, device, "vkDestroyBuffer"),
            .getBufferMemoryRequirements = L(c.PFN_vkGetBufferMemoryRequirements, gdpa, device, "vkGetBufferMemoryRequirements"),
            .allocateMemory = L(c.PFN_vkAllocateMemory, gdpa, device, "vkAllocateMemory"),
            .freeMemory = L(c.PFN_vkFreeMemory, gdpa, device, "vkFreeMemory"),
            .bindBufferMemory = L(c.PFN_vkBindBufferMemory, gdpa, device, "vkBindBufferMemory"),
            .mapMemory = L(c.PFN_vkMapMemory, gdpa, device, "vkMapMemory"),
            .unmapMemory = L(c.PFN_vkUnmapMemory, gdpa, device, "vkUnmapMemory"),
            .flushMappedMemoryRanges = L(c.PFN_vkFlushMappedMemoryRanges, gdpa, device, "vkFlushMappedMemoryRanges"),
            .invalidateMappedMemoryRanges = L(c.PFN_vkInvalidateMappedMemoryRanges, gdpa, device, "vkInvalidateMappedMemoryRanges"),
            .createShaderModule = L(c.PFN_vkCreateShaderModule, gdpa, device, "vkCreateShaderModule"),
            .destroyShaderModule = L(c.PFN_vkDestroyShaderModule, gdpa, device, "vkDestroyShaderModule"),
            .createDescriptorSetLayout = L(c.PFN_vkCreateDescriptorSetLayout, gdpa, device, "vkCreateDescriptorSetLayout"),
            .destroyDescriptorSetLayout = L(c.PFN_vkDestroyDescriptorSetLayout, gdpa, device, "vkDestroyDescriptorSetLayout"),
            .createPipelineLayout = L(c.PFN_vkCreatePipelineLayout, gdpa, device, "vkCreatePipelineLayout"),
            .destroyPipelineLayout = L(c.PFN_vkDestroyPipelineLayout, gdpa, device, "vkDestroyPipelineLayout"),
            .createComputePipelines = L(c.PFN_vkCreateComputePipelines, gdpa, device, "vkCreateComputePipelines"),
            .destroyPipeline = L(c.PFN_vkDestroyPipeline, gdpa, device, "vkDestroyPipeline"),
            .createDescriptorPool = L(c.PFN_vkCreateDescriptorPool, gdpa, device, "vkCreateDescriptorPool"),
            .destroyDescriptorPool = L(c.PFN_vkDestroyDescriptorPool, gdpa, device, "vkDestroyDescriptorPool"),
            .allocateDescriptorSets = L(c.PFN_vkAllocateDescriptorSets, gdpa, device, "vkAllocateDescriptorSets"),
            .updateDescriptorSets = L(c.PFN_vkUpdateDescriptorSets, gdpa, device, "vkUpdateDescriptorSets"),
            .destroyDevice = L(c.PFN_vkDestroyDevice, gdpa, device, "vkDestroyDevice"),
        };
    }
};

// =========================
// === Buffer ===

pub const Buffer = struct {
    buffer: c.VkBuffer,
    memory: c.VkDeviceMemory,
    size: u64,
    host_visible: bool, // mappable → read/write without staging
    coherent: bool, // HOST_COHERENT → no explicit flush/invalidate around maps

    pub fn deinit(self: *Buffer, dev: *Device) void {
        if (self.buffer != null) dev.dt.destroyBuffer.?(dev.device, self.buffer, null);
        if (self.memory != null) dev.dt.freeMemory.?(dev.device, self.memory, null);
        self.* = undefined;
    }
};

// A (buffer, offset, range) view bound to one descriptor slot. The model buffer is one big buffer
// whose weight sub-ranges are addressed by (offset, range); scratch buffers bind whole (offset 0).
pub const BufferBinding = struct {
    buffer: c.VkBuffer,
    offset: u64 = 0,
    range: u64, // bytes; never VK_WHOLE_SIZE so the shader's array length is exact
};

// =========================
// === Pipeline ===

pub const Pipeline = struct {
    pipeline: c.VkPipeline,
    layout: c.VkPipelineLayout,
    set_layout: c.VkDescriptorSetLayout,
    desc_pool: c.VkDescriptorPool,
    desc_set: c.VkDescriptorSet,
    n_bindings: u32,

    pub fn deinit(self: *Pipeline, dev: *Device) void {
        const dt = dev.dt;
        if (self.pipeline != null) dt.destroyPipeline.?(dev.device, self.pipeline, null);
        if (self.layout != null) dt.destroyPipelineLayout.?(dev.device, self.layout, null);
        if (self.desc_pool != null) dt.destroyDescriptorPool.?(dev.device, self.desc_pool, null);
        if (self.set_layout != null) dt.destroyDescriptorSetLayout.?(dev.device, self.set_layout, null);
        self.* = undefined;
    }
};

// =========================
// === Device ===

pub const DeviceOptions = struct {};

pub const Device = struct {
    physical: c.VkPhysicalDevice,
    device: c.VkDevice,
    queue: c.VkQueue,
    queue_family: u32,
    cmd_pool: c.VkCommandPool,
    cmd: c.VkCommandBuffer,
    mem_props: c.VkPhysicalDeviceMemoryProperties,
    min_storage_offset_align: u64, // limits.minStorageBufferOffsetAlignment — descriptor sub-range offsets must be a multiple
    dt: DeviceTbl,
    // Lazily-grown HOST_VISIBLE scratch for transfers to/from non-host-visible (discrete) buffers.
    staging: ?Buffer = null,

    pub fn deinit(self: *Device) void {
        if (self.staging) |*s| s.deinit(self);
        if (self.cmd_pool != null) self.dt.destroyCommandPool.?(self.device, self.cmd_pool, null);
        if (self.device != null) self.dt.destroyDevice.?(self.device, null);
        self.* = undefined;
    }

    pub fn waitIdle(self: *Device) void {
        _ = self.dt.deviceWaitIdle.?(self.device);
    }
};

// =========================
// === Device init ===

// Create a logical device + compute queue + command pool. Instance must already exist. Any failure
// past "a compute device exists" is a hard error — the caller assumes a device is creatable; only
// the genuinely environmental "no compute-capable physical device" returns NoComputeDevice (which
// callers map to a test skip).
pub fn initDevice(vt: *VTbl, gpa: std.mem.Allocator, opts: DeviceOptions) DeviceError!Device {
    _ = opts;
    const gipa = vt.get_instance_proc_addr;
    const instance = vt.instance;

    const enumPhys = loadInst(c.PFN_vkEnumeratePhysicalDevices, gipa, instance, "vkEnumeratePhysicalDevices");
    const getProps = loadInst(c.PFN_vkGetPhysicalDeviceProperties, gipa, instance, "vkGetPhysicalDeviceProperties");
    const getQF = loadInst(c.PFN_vkGetPhysicalDeviceQueueFamilyProperties, gipa, instance, "vkGetPhysicalDeviceQueueFamilyProperties");
    const getMemProps = loadInst(c.PFN_vkGetPhysicalDeviceMemoryProperties, gipa, instance, "vkGetPhysicalDeviceMemoryProperties");
    const enumDevExt = loadInst(c.PFN_vkEnumerateDeviceExtensionProperties, gipa, instance, "vkEnumerateDeviceExtensionProperties");
    const createDevice = loadInst(c.PFN_vkCreateDevice, gipa, instance, "vkCreateDevice");
    const gdpa = loadInst(c.PFN_vkGetDeviceProcAddr, gipa, instance, "vkGetDeviceProcAddr");

    // --- enumerate physical devices ---
    var phys_count: u32 = 0;
    _ = enumPhys.?(instance, &phys_count, null);
    if (phys_count == 0) return error.NoComputeDevice;
    var phys_buf: [8]c.VkPhysicalDevice = undefined;
    var got: u32 = @min(phys_count, @as(u32, phys_buf.len));
    _ = enumPhys.?(instance, &got, &phys_buf);

    // --- pick a (device, queue family) with a compute queue; prefer a discrete GPU ---
    var chosen: ?struct { phys: c.VkPhysicalDevice, qf: u32, discrete: bool } = null;
    for (phys_buf[0..got]) |pd| {
        var qf_count: u32 = 0;
        getQF.?(pd, &qf_count, null);
        var qf_buf: [16]c.VkQueueFamilyProperties = undefined;
        var qf_got: u32 = @min(qf_count, @as(u32, qf_buf.len));
        getQF.?(pd, &qf_got, &qf_buf);
        const compute_bit: u32 = @intCast(c.VK_QUEUE_COMPUTE_BIT);
        for (qf_buf[0..qf_got], 0..) |qf, i| {
            if (qf.queueFlags & compute_bit != 0) {
                var props: c.VkPhysicalDeviceProperties = undefined;
                getProps.?(pd, &props);
                const discrete = props.deviceType == c.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
                if (chosen == null or (discrete and !chosen.?.discrete)) {
                    chosen = .{ .phys = pd, .qf = @intCast(i), .discrete = discrete };
                }
                break; // one compute family per device is enough
            }
        }
    }
    const pick = chosen orelse return error.NoComputeDevice;

    var chosen_props: c.VkPhysicalDeviceProperties = undefined;
    getProps.?(pick.phys, &chosen_props);
    log.info("vk: using device '{s}' (queue family {d})", .{ std.mem.sliceTo(&chosen_props.deviceName, 0), pick.qf });

    // --- portability subset: a portability ICD (MoltenVK) REQUIRES this extension be enabled if it
    //     advertises it, else device creation/validation errors. Detect + enable it. ---
    var enable_portability = false;
    {
        var ext_count: u32 = 0;
        _ = enumDevExt.?(pick.phys, null, &ext_count, null);
        const exts = try gpa.alloc(c.VkExtensionProperties, ext_count);
        defer gpa.free(exts);
        var ext_got: u32 = ext_count;
        _ = enumDevExt.?(pick.phys, null, &ext_got, exts.ptr);
        for (exts[0..ext_got]) |e| {
            if (std.mem.eql(u8, std.mem.sliceTo(&e.extensionName, 0), "VK_KHR_portability_subset")) {
                enable_portability = true;
                break;
            }
        }
    }
    var dev_exts: [1][*c]const u8 = .{"VK_KHR_portability_subset"};
    const dev_ext_count: u32 = if (enable_portability) 1 else 0;

    // --- create the logical device with one compute queue ---
    const priority: f32 = 1.0;
    const qci: c.VkDeviceQueueCreateInfo = .{
        .sType = @intCast(c.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO),
        .queueFamilyIndex = pick.qf,
        .queueCount = 1,
        .pQueuePriorities = &priority,
    };
    const dci: c.VkDeviceCreateInfo = .{
        .sType = @intCast(c.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO),
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qci,
        .enabledExtensionCount = dev_ext_count,
        .ppEnabledExtensionNames = if (dev_ext_count > 0) &dev_exts else null,
    };
    var device: c.VkDevice = null;
    if (createDevice.?(pick.phys, &dci, null, &device) != c.VK_SUCCESS) return error.VkDeviceCreateFailed;
    const dt = DeviceTbl.load(gdpa, device);
    errdefer dt.destroyDevice.?(device, null);

    var queue: c.VkQueue = null;
    dt.getDeviceQueue.?(device, pick.qf, 0, &queue);

    // --- command pool (RESET flag: we reset+re-record the single buffer each op) + one buffer ---
    const pool_ci: c.VkCommandPoolCreateInfo = .{
        .sType = @intCast(c.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO),
        .flags = @intCast(c.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT),
        .queueFamilyIndex = pick.qf,
    };
    var cmd_pool: c.VkCommandPool = null;
    if (dt.createCommandPool.?(device, &pool_ci, null, &cmd_pool) != c.VK_SUCCESS) return error.VkCommandPoolFailed;
    errdefer dt.destroyCommandPool.?(device, cmd_pool, null);

    const cb_ai: c.VkCommandBufferAllocateInfo = .{
        .sType = @intCast(c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO),
        .commandPool = cmd_pool,
        .level = @intCast(c.VK_COMMAND_BUFFER_LEVEL_PRIMARY),
        .commandBufferCount = 1,
    };
    var cmd: c.VkCommandBuffer = null;
    if (dt.allocateCommandBuffers.?(device, &cb_ai, &cmd) != c.VK_SUCCESS) return error.VkCommandPoolFailed;

    var mem_props: c.VkPhysicalDeviceMemoryProperties = undefined;
    getMemProps.?(pick.phys, &mem_props);

    return .{
        .physical = pick.phys,
        .device = device,
        .queue = queue,
        .queue_family = pick.qf,
        .cmd_pool = cmd_pool,
        .cmd = cmd,
        .mem_props = mem_props,
        .min_storage_offset_align = chosen_props.limits.minStorageBufferOffsetAlignment,
        .dt = dt,
    };
}

// =========================
// === Memory + buffers ===

// Find a memory type satisfying `type_bits` with all of `required`, preferring one that also has
// `preferred`. Returns the index. Working buffers prefer DEVICE_LOCAL (required=0); staging
// requires HOST_VISIBLE|HOST_COHERENT.
fn findMemoryType(mp: *const c.VkPhysicalDeviceMemoryProperties, type_bits: u32, required: u32, preferred: u32) ?u32 {
    // Pass 1: required + preferred.
    var i: u32 = 0;
    while (i < mp.memoryTypeCount) : (i += 1) {
        const flags = mp.memoryTypes[i].propertyFlags;
        if (type_bits & (@as(u32, 1) << @intCast(i)) != 0 and
            flags & required == required and flags & preferred == preferred)
            return i;
    }
    // Pass 2: required only.
    i = 0;
    while (i < mp.memoryTypeCount) : (i += 1) {
        const flags = mp.memoryTypes[i].propertyFlags;
        if (type_bits & (@as(u32, 1) << @intCast(i)) != 0 and flags & required == required)
            return i;
    }
    return null;
}

fn allocBuffer(dev: *Device, size: u64, usage: u32, required: u32, preferred: u32) DeviceError!Buffer {
    const dt = dev.dt;
    const bci: c.VkBufferCreateInfo = .{
        .sType = @intCast(c.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO),
        .size = size,
        .usage = usage,
        .sharingMode = @intCast(c.VK_SHARING_MODE_EXCLUSIVE),
    };
    var buffer: c.VkBuffer = null;
    if (dt.createBuffer.?(dev.device, &bci, null, &buffer) != c.VK_SUCCESS) return error.VkOutOfMemory;
    errdefer dt.destroyBuffer.?(dev.device, buffer, null);

    var reqs: c.VkMemoryRequirements = undefined;
    dt.getBufferMemoryRequirements.?(dev.device, buffer, &reqs);
    const type_index = findMemoryType(&dev.mem_props, reqs.memoryTypeBits, required, preferred) orelse
        return error.NoSuitableMemoryType;
    const flags = dev.mem_props.memoryTypes[type_index].propertyFlags;

    const mai: c.VkMemoryAllocateInfo = .{
        .sType = @intCast(c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO),
        .allocationSize = reqs.size,
        .memoryTypeIndex = type_index,
    };
    var memory: c.VkDeviceMemory = null;
    if (dt.allocateMemory.?(dev.device, &mai, null, &memory) != c.VK_SUCCESS) return error.VkOutOfMemory;
    errdefer dt.freeMemory.?(dev.device, memory, null);

    if (dt.bindBufferMemory.?(dev.device, buffer, memory, 0) != c.VK_SUCCESS) return error.VkOutOfMemory;

    return .{
        .buffer = buffer,
        .memory = memory,
        .size = size,
        .host_visible = flags & MEM_HOST_VISIBLE != 0,
        .coherent = flags & MEM_HOST_COHERENT != 0,
    };
}

// A working buffer: storage + transfer src/dst, preferring DEVICE_LOCAL memory. On unified-memory
// devices (MoltenVK) the chosen type is also host-visible → writeBuffer/readBuffer map directly; on
// a discrete GPU it isn't → they route through a staging buffer.
pub fn createBuffer(dev: *Device, size: u64) DeviceError!Buffer {
    return allocBuffer(dev, size, USAGE_WORKING, 0, MEM_DEVICE_LOCAL);
}

fn ensureStaging(dev: *Device, size: u64) DeviceError!*Buffer {
    if (dev.staging) |*s| {
        if (s.size >= size) return s;
        s.deinit(dev);
        dev.staging = null;
    }
    dev.staging = try allocBuffer(dev, size, USAGE_STAGING, MEM_HOST_VISIBLE | MEM_HOST_COHERENT, 0);
    return &dev.staging.?;
}

// Record a single buffer→buffer copy, submit it, and wait for the queue to idle (serial model).
fn copyBufferSync(dev: *Device, src: c.VkBuffer, dst: c.VkBuffer, size: u64) void {
    const dt = dev.dt;
    _ = dt.resetCommandBuffer.?(dev.cmd, 0);
    const begin: c.VkCommandBufferBeginInfo = .{
        .sType = @intCast(c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO),
        .flags = @intCast(c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT),
    };
    _ = dt.beginCommandBuffer.?(dev.cmd, &begin);
    const region: c.VkBufferCopy = .{ .srcOffset = 0, .dstOffset = 0, .size = size };
    dt.cmdCopyBuffer.?(dev.cmd, src, dst, 1, &region);
    _ = dt.endCommandBuffer.?(dev.cmd);
    submitAndWait(dev);
}

fn submitAndWait(dev: *Device) void {
    const dt = dev.dt;
    var cmd = dev.cmd;
    const submit: c.VkSubmitInfo = .{
        .sType = @intCast(c.VK_STRUCTURE_TYPE_SUBMIT_INFO),
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
    };
    _ = dt.queueSubmit.?(dev.queue, 1, &submit, null);
    _ = dt.queueWaitIdle.?(dev.queue);
}

// === the "copy-if-on-CPU-else-map" abstraction (write half) ===
// host-visible → map + memcpy (+ flush when not coherent). Otherwise stage + GPU copy + wait idle.
pub fn writeBuffer(dev: *Device, buf: *Buffer, bytes: []const u8) DeviceError!void {
    std.debug.assert(bytes.len <= buf.size);
    if (bytes.len == 0) return;
    if (buf.host_visible) {
        try mapWrite(dev, buf, bytes);
    } else {
        const stg = try ensureStaging(dev, bytes.len);
        try mapWrite(dev, stg, bytes);
        copyBufferSync(dev, stg.buffer, buf.buffer, bytes.len);
    }
}

// === the "copy-if-on-CPU-else-map" abstraction (read half) ===
// host-visible → (invalidate when not coherent) map + memcpy out. Otherwise GPU copy → staging,
// wait idle, then map the staging buffer out. Callers vkQueueWaitIdle the producing dispatch first.
pub fn readBuffer(dev: *Device, buf: *Buffer, out: []u8) DeviceError!void {
    std.debug.assert(out.len <= buf.size);
    if (out.len == 0) return;
    if (buf.host_visible) {
        try mapRead(dev, buf, out);
    } else {
        const stg = try ensureStaging(dev, out.len);
        copyBufferSync(dev, buf.buffer, stg.buffer, out.len);
        try mapRead(dev, stg, out);
    }
}

fn mapWrite(dev: *Device, buf: *Buffer, bytes: []const u8) DeviceError!void {
    const dt = dev.dt;
    var ptr: ?*anyopaque = null;
    if (dt.mapMemory.?(dev.device, buf.memory, 0, bytes.len, 0, &ptr) != c.VK_SUCCESS) return error.VkOutOfMemory;
    const dst: [*]u8 = @ptrCast(ptr.?);
    @memcpy(dst[0..bytes.len], bytes);
    if (!buf.coherent) flushRange(dev, buf.memory, bytes.len, true);
    dt.unmapMemory.?(dev.device, buf.memory);
}

fn mapRead(dev: *Device, buf: *Buffer, out: []u8) DeviceError!void {
    const dt = dev.dt;
    var ptr: ?*anyopaque = null;
    if (dt.mapMemory.?(dev.device, buf.memory, 0, out.len, 0, &ptr) != c.VK_SUCCESS) return error.VkOutOfMemory;
    if (!buf.coherent) flushRange(dev, buf.memory, out.len, false);
    const src: [*]const u8 = @ptrCast(ptr.?);
    @memcpy(out, src[0..out.len]);
    dt.unmapMemory.?(dev.device, buf.memory);
}

// Flush (host writes → device) or invalidate (device writes → host) a mapped range on non-coherent
// memory. MoltenVK exposes non-coherent host-visible memory, where skipping this silently corrupts
// the transferred bytes — the subtle correctness bug this whole helper exists to prevent.
fn flushRange(dev: *Device, memory: c.VkDeviceMemory, size: u64, is_flush: bool) void {
    const range: c.VkMappedMemoryRange = .{
        .sType = @intCast(c.VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE),
        .memory = memory,
        .offset = 0,
        .size = @intCast(c.VK_WHOLE_SIZE),
    };
    _ = size;
    if (is_flush) {
        _ = dev.dt.flushMappedMemoryRanges.?(dev.device, 1, &range);
    } else {
        _ = dev.dt.invalidateMappedMemoryRanges.?(dev.device, 1, &range);
    }
}

// =========================
// === Compute pipeline ===

// Build a compute pipeline from SPIR-V bytes: a set-0 layout of `n_bindings` storage buffers, a
// push-constant range of `push_constant_size` (0 ⇒ none), a 1-set descriptor pool + set. The set is
// re-written before each dispatch (safe: the prior use completed at the last queue-wait-idle).
pub fn createComputePipeline(
    dev: *Device,
    spirv: []align(4) const u8,
    n_bindings: u32,
    push_constant_size: u32,
    entry_point: [*:0]const u8,
) DeviceError!Pipeline {
    std.debug.assert(n_bindings <= MAX_BINDINGS);
    const dt = dev.dt;

    // --- shader module ---
    const smci: c.VkShaderModuleCreateInfo = .{
        .sType = @intCast(c.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO),
        .codeSize = spirv.len,
        .pCode = @ptrCast(spirv.ptr),
    };
    var module: c.VkShaderModule = null;
    if (dt.createShaderModule.?(dev.device, &smci, null, &module) != c.VK_SUCCESS) return error.VkOutOfMemory;
    defer dt.destroyShaderModule.?(dev.device, module, null); // not needed after pipeline creation

    // --- descriptor set layout: n storage buffers at bindings 0..n ---
    var slb: [MAX_BINDINGS]c.VkDescriptorSetLayoutBinding = undefined;
    for (0..n_bindings) |i| slb[i] = .{
        .binding = @intCast(i),
        .descriptorType = DTYPE_STORAGE,
        .descriptorCount = 1,
        .stageFlags = STAGE_COMPUTE,
    };
    const dslci: c.VkDescriptorSetLayoutCreateInfo = .{
        .sType = @intCast(c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO),
        .bindingCount = n_bindings,
        .pBindings = &slb,
    };
    var set_layout: c.VkDescriptorSetLayout = null;
    if (dt.createDescriptorSetLayout.?(dev.device, &dslci, null, &set_layout) != c.VK_SUCCESS) return error.VkOutOfMemory;
    errdefer dt.destroyDescriptorSetLayout.?(dev.device, set_layout, null);

    // --- pipeline layout (+ optional push range) ---
    const pc_range: c.VkPushConstantRange = .{ .stageFlags = STAGE_COMPUTE, .offset = 0, .size = push_constant_size };
    const plci: c.VkPipelineLayoutCreateInfo = .{
        .sType = @intCast(c.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO),
        .setLayoutCount = 1,
        .pSetLayouts = &set_layout,
        .pushConstantRangeCount = if (push_constant_size > 0) 1 else 0,
        .pPushConstantRanges = if (push_constant_size > 0) &pc_range else null,
    };
    var layout: c.VkPipelineLayout = null;
    if (dt.createPipelineLayout.?(dev.device, &plci, null, &layout) != c.VK_SUCCESS) return error.VkOutOfMemory;
    errdefer dt.destroyPipelineLayout.?(dev.device, layout, null);

    // --- descriptor pool + set ---
    const pool_size: c.VkDescriptorPoolSize = .{ .type = DTYPE_STORAGE, .descriptorCount = @max(n_bindings, 1) };
    const dpci: c.VkDescriptorPoolCreateInfo = .{
        .sType = @intCast(c.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO),
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size,
    };
    var desc_pool: c.VkDescriptorPool = null;
    if (dt.createDescriptorPool.?(dev.device, &dpci, null, &desc_pool) != c.VK_SUCCESS) return error.VkOutOfMemory;
    errdefer dt.destroyDescriptorPool.?(dev.device, desc_pool, null);

    var sl = set_layout;
    const dsai: c.VkDescriptorSetAllocateInfo = .{
        .sType = @intCast(c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO),
        .descriptorPool = desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &sl,
    };
    var desc_set: c.VkDescriptorSet = null;
    if (dt.allocateDescriptorSets.?(dev.device, &dsai, &desc_set) != c.VK_SUCCESS) return error.VkOutOfMemory;

    // --- compute pipeline ---
    const stage: c.VkPipelineShaderStageCreateInfo = .{
        .sType = @intCast(c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO),
        .stage = @intCast(c.VK_SHADER_STAGE_COMPUTE_BIT),
        .module = module,
        .pName = entry_point,
    };
    const cpci: c.VkComputePipelineCreateInfo = .{
        .sType = @intCast(c.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO),
        .stage = stage,
        .layout = layout,
    };
    var pipeline: c.VkPipeline = null;
    if (dt.createComputePipelines.?(dev.device, null, 1, &cpci, null, &pipeline) != c.VK_SUCCESS) return error.VkOutOfMemory;

    return .{
        .pipeline = pipeline,
        .layout = layout,
        .set_layout = set_layout,
        .desc_pool = desc_pool,
        .desc_set = desc_set,
        .n_bindings = n_bindings,
    };
}

// Bind `bindings` to the pipeline's set, push `push_bytes`, dispatch `groups` workgroups, submit,
// and wait for the queue to idle. The whole op is one reset+record+submit+idle — no barriers.
pub fn dispatch(dev: *Device, pipe: *Pipeline, bindings: []const BufferBinding, push_bytes: []const u8, groups: [3]u32) void {
    std.debug.assert(bindings.len == pipe.n_bindings);
    std.debug.assert(bindings.len <= MAX_BINDINGS);
    const dt = dev.dt;

    // --- update the descriptor set (prior use finished at the last queue-wait-idle) ---
    var infos: [MAX_BINDINGS]c.VkDescriptorBufferInfo = undefined;
    var writes: [MAX_BINDINGS]c.VkWriteDescriptorSet = undefined;
    for (bindings, 0..) |b, i| {
        infos[i] = .{ .buffer = b.buffer, .offset = b.offset, .range = b.range };
        writes[i] = .{
            .sType = @intCast(c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET),
            .dstSet = pipe.desc_set,
            .dstBinding = @intCast(i),
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = DTYPE_STORAGE,
            .pBufferInfo = &infos[i],
        };
    }
    dt.updateDescriptorSets.?(dev.device, @intCast(bindings.len), &writes, 0, null);

    // --- record ---
    _ = dt.resetCommandBuffer.?(dev.cmd, 0);
    const begin: c.VkCommandBufferBeginInfo = .{
        .sType = @intCast(c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO),
        .flags = @intCast(c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT),
    };
    _ = dt.beginCommandBuffer.?(dev.cmd, &begin);
    dt.cmdBindPipeline.?(dev.cmd, @intCast(c.VK_PIPELINE_BIND_POINT_COMPUTE), pipe.pipeline);
    var set = pipe.desc_set;
    dt.cmdBindDescriptorSets.?(dev.cmd, @intCast(c.VK_PIPELINE_BIND_POINT_COMPUTE), pipe.layout, 0, 1, &set, 0, null);
    if (push_bytes.len > 0) {
        dt.cmdPushConstants.?(dev.cmd, pipe.layout, STAGE_COMPUTE, 0, @intCast(push_bytes.len), push_bytes.ptr);
    }
    dt.cmdDispatch.?(dev.cmd, groups[0], groups[1], groups[2]);
    _ = dt.endCommandBuffer.?(dev.cmd);

    // --- submit + serialize ---
    submitAndWait(dev);
}

// =========================
// === Tests ===

test "device init + buffer write/read round-trip" {
    var vt = vtbl.instanceOrSkip(.{
        .enable_validation = true,
        .debug_callback = vtbl.captureCallback,
        .debug_user_data = &cap,
    }) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        error.VkInstanceCreateFailed => return error.VkInstanceCreateFailed,
    };
    defer vt.deinit();
    defer cap.assertNoValidationErrors();

    var dev = initDevice(&vt, std.testing.allocator, .{}) catch |e| switch (e) {
        // No compute-capable physical device is environmental, like a missing loader: skip.
        error.NoComputeDevice => return error.SkipZigTest,
        else => return e,
    };
    defer dev.deinit();

    // Round-trip a known byte pattern through a working buffer — exercises map-vs-stage + (in)validate.
    var src: [256]u8 = undefined;
    for (&src, 0..) |*b, i| b.* = @intCast(i & 0xff);
    var buf = try createBuffer(&dev, src.len);
    defer buf.deinit(&dev);

    try writeBuffer(&dev, &buf, &src);
    var dst: [256]u8 = undefined;
    try readBuffer(&dev, &buf, &dst);
    try std.testing.expectEqualSlices(u8, &src, &dst);
}

// File-scope capture so the defer in the test can reference it after `vt` is moved into scope.
var cap: vtbl.ValidationCapture = .{};
