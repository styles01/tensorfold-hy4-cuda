//! Hy4-flash's CUDA family over the maintained runtime, the core and the lane core.
const std = @import("std");

pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{
        .root_source_file = b.path("zig/src/families/hy4_flash/cuda_family.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "cuda", .module = cuda },
            .{ .name = "core", .module = core },
            .{ .name = "lanes", .module = lanes },
            .{ .name = "engram_hash_core", .module = engramHashCore(b, target, optimize) },
        },
    });
}

fn engramHashCore(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/engram_hash.zig"), .target = target, .optimize = optimize });
}
