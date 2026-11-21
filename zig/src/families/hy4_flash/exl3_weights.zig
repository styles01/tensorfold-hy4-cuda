//! Split routed experts from the standard Hy4-flash EXL3 checkpoint.
const std = @import("std");
const core = @import("core");
const Tensor = core.safetensors.Tensor;

pub const Projection = struct {
    parts: core.exl3_load.Parts,
    geometry: core.exl3_layout.Group,
};

pub const Expert = struct { gate: Projection, up: Projection, down: Projection };

pub fn readExpert(source: anytype, layer: usize, expert: usize, latent: usize, intermediate: usize) !Expert {
    std.debug.print("tf-re {d} {d}\n", .{ layer, expert });
    var buf: [160]u8 = undefined;
    const legacy = try std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.experts.{d}.w1.trellis", .{ layer, expert });
    if (!source.has(legacy)) return .{
        .gate = try batched(source, layer, expert, "gate_trellis", "w13", latent, intermediate, 0),
        .up = try batched(source, layer, expert, "up_trellis", "w13", latent, intermediate, 1),
        .down = try batched(source, layer, expert, "down_trellis", "w2", intermediate, latent, 0),
    };
    return .{
        .gate = try read(source, layer, expert, "gate", latent, intermediate),
        .up = try read(source, layer, expert, "up", latent, intermediate),
        .down = try read(source, layer, expert, "down", intermediate, latent),
    };
}

/// The released pack's batched routed experts: per layer, `routed_experts.{gate,up}_trellis.{e}`,
/// packed gate+up (w13) and down (w2) f16 scales [E, 2, k] / [E, k]; the MCG codebook rides in
/// w13_mul1, so no per-expert marker exists there.
fn batched(source: anytype, layer: usize, expert: usize, trellis_name: []const u8, packed_name: []const u8, k: usize, n: usize, which: usize) !Projection {
    std.debug.print("tf-enter {d} {d} {s}\n", .{ layer, expert, trellis_name });
    var tbuf: [220]u8 = undefined;
    const trellis = blk: {
        break :blk source.get(try std.fmt.bufPrint(&tbuf, "language_model.model.layers.{d}.mlp.experts.routed_experts.{s}.{d}", .{ layer, trellis_name, expert })) catch |err| {
            std.debug.print("tf-trellis-get-failed {s}\n", .{@errorName(err)});
            return err;
        };
    };
    if (trellis.dtype != .i16 or trellis.rank != 3 or k % 128 != 0 or n % 128 != 0)
        return error.BadHy4Trellis;
    const g = try core.exl3_layout.Group.fromShape(trellis.shape[0..trellis.rank]);
    if (g.k() != k or g.n() != n or trellis.bytes.len != g.trellisBytes())
        return error.BadHy4Trellis;
    var sbuf: [220]u8 = undefined;
    const suh_big = try source.get(try std.fmt.bufPrint(&sbuf, "language_model.model.layers.{d}.mlp.experts.routed_experts.{s}_suh", .{ layer, packed_name }));
    var vbuf: [220]u8 = undefined;
    const svh_big = try source.get(try std.fmt.bufPrint(&vbuf, "language_model.model.layers.{d}.mlp.experts.routed_experts.{s}_svh", .{ layer, packed_name }));
    std.debug.print("tf-batched layer {d} exp {d} {s}: trellis {any} {s} k={d} n={d} suh {any} svh {any}\n", .{ layer, expert, trellis_name, trellis.shape[0..2], packed_name, k, n, suh_big.shape[0..3], svh_big.shape[0..3] });
    if (suh_big.dtype != .f16 or svh_big.dtype != .f16 or suh_big.rank < 2 or svh_big.rank < 2 or suh_big.rank > 3 or svh_big.rank > 3) {
        std.debug.print("check A fail {s} {any} {any}\n", .{ packed_name, suh_big.shape[0..3], svh_big.shape[0..3] });
        return error.BadHy4ExpertScales;
    }
    // w13 packs gate and up as [E, 2, k]; w2 is [E, k, 1]
    const flat = suh_big.shape[2] == 1 and svh_big.shape[2] == 1;
    const suh_row: usize = if (flat) suh_big.shape[1] else suh_big.shape[2];
    const svh_row: usize = if (flat) svh_big.shape[1] else svh_big.shape[2];
    if (suh_row != k or svh_row != n) { std.debug.print("check B fail {s} k={d} suh_row={d} n={d} svh_row={d}\n", .{ packed_name, k, suh_row, n, svh_row }); return error.BadHy4ExpertScales; }
    const slots: usize = if (flat) 1 else suh_big.shape[1];
    const svh_slots: usize = if (flat) 1 else svh_big.shape[1];
    if (which >= slots or which >= svh_slots) { std.debug.print("check C fail {s} slots={d}\n", .{ packed_name, slots }); return error.BadHy4ExpertScales; }
    const suh_at = (expert * slots + which) * suh_row * 2;
    const svh_at = (expert * svh_slots + which) * svh_row * 2;
    if (suh_at + suh_row * 2 > suh_big.bytes.len or svh_at + svh_row * 2 > svh_big.bytes.len) {
        std.debug.print("check D fail {s} at={d} len={d}\n", .{ packed_name, suh_at, suh_big.bytes.len });
        return error.BadHy4ExpertScales;
    }
    return .{
        .parts = .{
            .trellis = trellis,
            .suh = fixed(.f16, &.{k}, suh_big.bytes[suh_at .. suh_at + k * 2]),
            .svh = fixed(.f16, &.{n}, svh_big.bytes[svh_at .. svh_at + n * 2]),
        },
        .geometry = g,
    };
}

fn read(source: anytype, layer: usize, expert: usize, name: []const u8, k: usize, n: usize) !Projection {
    var buf: [160]u8 = undefined;
    // the released pack names the routed experts w1 (gate), w2 (down) and w3 (up)
    const stored = if (std.mem.eql(u8, name, "gate")) "w1" else if (std.mem.eql(u8, name, "up")) "w3" else "w2";
    const prefix = try std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.experts.{d}.{s}", .{ layer, expert, stored });
    for ([_][]const u8{ "mcg", "bias", "weight", "su", "sv" }) |suffix| {
        var full: [192]u8 = undefined;
        if (std.mem.eql(u8, suffix, "weight") and source.has(try std.fmt.bufPrint(&full, "{s}.{s}", .{ prefix, suffix })))
            return error.UnsupportedHy4Expert;
    }
    return validate(
        try tensor(source, prefix, "trellis"),
        try tensor(source, prefix, "suh"),
        try tensor(source, prefix, "svh"),
        try tensor(source, prefix, "mul1"),
        k,
        n,
    );
}

fn tensor(source: anytype, prefix: []const u8, suffix: []const u8) !Tensor {
    var buf: [192]u8 = undefined;
    return source.get(try std.fmt.bufPrint(&buf, "{s}.{s}", .{ prefix, suffix }));
}

fn validate(trellis: Tensor, suh: Tensor, svh: Tensor, marker: Tensor, k: usize, n: usize) !Projection {
    if (trellis.dtype != .i16 or trellis.rank != 3 or k % 128 != 0 or n % 128 != 0)
        return error.BadHy4Trellis;
    const g = try core.exl3_layout.Group.fromShape(trellis.shape[0..trellis.rank]);
    if (g.k() != k or g.n() != n or trellis.bytes.len != g.trellisBytes())
        return error.BadHy4Trellis;
    if (!suh.is(.f16, &.{k}) or !svh.is(.f16, &.{n}) or suh.bytes.len != 2 * k or svh.bytes.len != 2 * n)
        return error.BadHy4ExpertScales;
    if ((!marker.is(.i32, &.{}) and !marker.is(.i32, &.{1})) or marker.bytes.len != 4)
        return error.BadHy4ExpertMarker;
    if (try core.exl3_layout.fromMarker(std.mem.readInt(u32, marker.bytes[0..4], .little)) != .mul1)
        return error.BadHy4ExpertMarker;
    return .{ .parts = .{ .trellis = trellis, .suh = suh, .svh = svh }, .geometry = g };
}

fn fixed(dtype: core.safetensors.DType, shape: []const usize, bytes: []const u8) Tensor {
    var out: Tensor = .{ .dtype = dtype, .rank = @intCast(shape.len), .shape = @splat(1), .bytes = bytes };
    @memcpy(out.shape[0..shape.len], shape);
    return out;
}

test "Hy4 orders EXL3 axes as input then output and requires fp16 scales and mul1" {
    const t = std.testing;
    const words: [8 * 16 * 64 * 2]u8 = @splat(0);
    const input: [128 * 2]u8 = @splat(0);
    const output: [256 * 2]u8 = @splat(0);
    const magic = &.{ 0x2d, 0xd1, 0xdc, 0x83 };
    const trellis = fixed(.i16, &.{ 8, 16, 64 }, &words);
    const suh = fixed(.f16, &.{128}, &input);
    const svh = fixed(.f16, &.{256}, &output);
    const marker = fixed(.i32, &.{1}, magic);
    const p = try validate(trellis, suh, svh, marker, 128, 256);
    try t.expectEqual(@as(u32, 8), p.geometry.k2());
    _ = try validate(trellis, suh, svh, fixed(.i32, &.{}, magic), 128, 256);
    try t.expectError(error.BadHy4Trellis, validate(trellis, suh, svh, marker, 256, 128));
    try t.expectError(error.BadHy4ExpertScales, validate(trellis, suh, suh, marker, 128, 256));
    try t.expectError(error.BadHy4ExpertMarker, validate(trellis, suh, svh, fixed(.u32, &.{1}, magic), 128, 256));
    try t.expectError(error.BadHy4ExpertMarker, validate(trellis, suh, svh, fixed(.i32, &.{}, &.{ 0xed, 0x1f, 0xac, 0xcb }), 128, 256));
}
