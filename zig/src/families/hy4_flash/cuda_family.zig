//! Hy4-flash's CUDA family: the native registry's entry (native/cuda.zig openWith over cuda_native.zig).
pub const native = @import("cuda_native.zig");

test {
    _ = native;
}
