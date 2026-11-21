//! Hy4-flash on CUDA behind the native registry, over the family's own model (one sequence, its own caches).
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const fwd = @import("cuda_model.zig");
const mla = @import("mla_weights.zig");

const be = lanes.backend;
const Stream = lanes.Stream;

pub const model_type = "hy_v4_flash";
pub const formats: []const []const u8 = &.{"exl3"};
pub const default_context: i64 = 2048;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = 512; // a prompt chunk's rows (the model's pass limit is 1,024)
pub const snap_lookahead: u32 = 0;
pub const explains_open = true;
/// No DFly2 head on CUDA yet: drafts come from the host's copy paths alone (copy drafting is off too).
pub const takes_drafter = false;
pub const drafter_refusal = "--drafter: Hy4-flash on CUDA loads no drafter; DFly2 stays on the Metal engine";

/// The server's reading of the room left after the weights; null if unreadable.
pub const Room = struct { ctx: *anyopaque, read: *const fn (*anyopaque) ?u64 };
pub const Options = struct { context: usize, drafts: bool, segments: usize = 1, slide: bool = false, lanes: u32 = 1, lanes_fixed: bool = false, room: ?Room = null };

pub const Loaded = struct {
    backend: be.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    free_streams: u32,
    prompt_rows: u32,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
};

const capacity_most: usize = 2048; // the sparse attention's per-row key list holds every position

const Serve = struct {
    gpa: std.mem.Allocator,
    s: cuda.Stream,
    k: *fwd.Kernels,
    m: *fwd.Model,
    capacity: usize, // the sparse caches' slots: the window
    vocab: usize,
    raw: []f32, // the last pass's last-row logits (the model's own download)
    logits: []u16, // the same row as bf16 (the host draw's input)
    kept: bool = false, // the released stream's caches and history, for a prompt that shares their prefix
    active: ?*Stream = null,
    len: u32 = 0, // the active stream's cache length (the model's history length)
    stream_bytes: usize,

    /// The weights, the kernels and one warm pass over the window's slots (UMA pages touched once).
    fn init(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, capacity: usize) !Serve {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var s = try cuda.Stream.init(ctx.d, false);
        errdefer s.deinit();
        const k = try gpa.create(fwd.Kernels);
        errdefer gpa.destroy(k);
        k.* = try fwd.Kernels.load(gpa, ctx.d);
        errdefer k.deinit();
        const m = try fwd.Model.open(gpa, io, ctx.d, s, k, dir, capacity, prompt_rows);
        var x = Serve{
            .gpa = gpa,
            .s = s,
            .k = k,
            .m = m,
            .capacity = capacity,
            .vocab = m.text.vocab_size,
            .raw = undefined,
            .logits = undefined,
            .stream_bytes = m.layers.len * capacity * mla.absorbed_stride * 2,
        };
        x.raw = try gpa.alloc(f32, x.vocab);
        errdefer gpa.free(x.raw);
        x.logits = try gpa.alloc(u16, x.vocab);
        errdefer gpa.free(x.logits);
        // one pass over the whole window: every cache slot and every weight page touched once
        var row: [prompt_rows]u32 = undefined;
        for (&row, 0..) |*t, i| t.* = @intCast(1000 + i * 37 % 20000);
        var at: usize = 0;
        while (at < capacity) : (at += prompt_rows) try x.m.pass(&row, x.raw);
        x.m.history.items.len = 0;
        {
            var lo: f32 = std.math.floatMax(f32);
            var hi: f32 = -std.math.floatMax(f32);
            var bad: usize = 0;
            for (x.raw) |v| {
                if (!std.math.isFinite(v)) bad += 1;
                if (std.math.isFinite(v)) {
                    lo = @min(lo, v);
                    hi = @max(hi, v);
                }
            }
            std.debug.print("tf-warmup lo={d} hi={d} nonfinite={d}\n", .{ lo, hi, bad });
        }
        if (std.c.getenv("TF_HY4_PROBE") != null) {
            const n = 8;
            var ids: [n]u32 = undefined;
            for (&ids, 0..) |*t, i| t.* = @intCast(1000 + i * 37 % 20000);
            x.m.history.items.len = 0;
            std.debug.print("tf-probe run=1 rows=7\n", .{});
            try x.m.pass(ids[0 .. n - 1], x.raw);
            std.debug.print("tf-probe run=2 rows=1\n", .{});
            x.m.history.items.len = n - 1;
            try x.m.pass(ids[n - 1 .. n], x.raw);
            std.debug.print("tf-probe run=2 done\n", .{});
        }
        return x;
    }

    fn deinit(x: *Serve, g: std.mem.Allocator) void {
        x.m.deinit();
        x.k.deinit();
        g.destroy(x.k);
        g.free(x.logits);
        g.free(x.raw);
        x.s.deinit();
    }

    fn draw(x: *Serve, sampling: ?lanes.Sampling, position: u64, mask: []const u32) !u32 {
        if (std.c.getenv("TF_HY4_TOP") != null) {
            var ids: [5]usize = .{ 0, 0, 0, 0, 0 };
            var vals: [5]f32 = .{ -std.math.floatMax(f32), -std.math.floatMax(f32), -std.math.floatMax(f32), -std.math.floatMax(f32), -std.math.floatMax(f32) };
            for (x.raw, 0..) |v, i| {
                var s: usize = 0;
                for (1..5) |j| { if (vals[j] < vals[s]) { s = j; } }
                if (v > vals[s]) { vals[s] = v; ids[s] = i; }
            }
            std.debug.print("tf-top pos={d} ids={any} vals={any}\n", .{ position, ids, vals });
        }
        var lo: f32 = std.math.floatMax(f32);
        var hi: f32 = -std.math.floatMax(f32);
        var bad: usize = 0;
        for (x.raw) |v| {
            if (!std.math.isFinite(v)) bad += 1;
            if (std.math.isFinite(v)) {
                lo = @min(lo, v);
                hi = @max(hi, v);
            }
        }
        std.debug.print("tf-draw pos={d} lo={d} hi={d} nonfinite={d}\n", .{ position, lo, hi, bad });
        for (x.raw, 0..) |v, i| x.logits[i] = bf16(v);
        return lanes.sampling_row.draw(x.gpa, x.logits, sampling, position, mask);
    }

    fn self(ptr: *anyopaque) *Serve {
        return @ptrCast(@alignCast(ptr));
    }

    fn mine(x: *Serve, s: *Stream) !void {
        if (x.active != s) return error.NoFreeSlot;
    }

    fn prefill(ptr: *anyopaque, s: *Stream) anyerror!void {
        const x = self(ptr);
        const ids = s.prompt();
        if (ids.len == 0 or (x.active != null and x.active != s)) return error.InvalidHy4Prompt;
        if (ids.len + s.max_new + 1 > x.capacity) return error.PastContext;
        if (x.active != s) {
            if (x.active != null) return error.NoFreeSlot;
            // the model's caches and history, when this prompt starts with the released stream's tokens
            var at: usize = if (x.kept) @min(x.m.history.items.len, ids.len - 1) else 0;
            const bound = at;
            at = 0;
            while (at < bound and x.m.history.items[at] == ids[at]) at += 1;
            if (at < 64) at = 0;
            x.m.history.items.len = at;
            x.kept = false;
            x.active = s;
            x.len = @intCast(at);
            s.cached = @intCast(at);
        }
        var at: usize = x.len;
        while (at < ids.len) {
            if (s.isCancelled()) return error.Cancelled;
            const end = @min(ids.len, at + prompt_rows);
            try x.m.pass(ids[at..end], x.raw);
            if (end == ids.len) {
                x.len = @intCast(x.m.history.items.len);
                return lanes.prefill_widths.completed(x.gpa, s, end - at);
            }
            at = end;
        }
        x.len = @intCast(x.m.history.items.len);
    }

    fn first(ptr: *anyopaque, s: *Stream, position: u64) anyerror!u64 {
        const x = self(ptr);
        try x.mine(s);
        if (position != x.len) return error.InvalidHy4Position;
        return try x.draw(s.sampling, position, &.{});
    }

    fn firstMasked(ptr: *anyopaque, s: *Stream, position: u64, mask: []const u32) anyerror!u64 {
        const x = self(ptr);
        try x.mine(s);
        if (position != x.len) return error.InvalidHy4Position;
        return try x.draw(s.sampling, position, mask);
    }

    fn queue(ptr: *anyopaque, s: *Stream, f: be.Feed, position: u64) anyerror!u64 {
        const x = self(ptr);
        try x.mine(s);
        if (position != x.len + 1) return error.InvalidHy4Position;
        const token: u32 = switch (f) {
            .value => |v| v,
            .handle => |h| if (h >= x.vocab) return error.InvalidHy4Token else @intCast(h),
        };
        try x.m.pass(&.{token}, x.raw);
        x.len += 1;
        return try x.draw(s.sampling, position, &.{});
    }

    fn read(ptr: *anyopaque, handle: u64) anyerror!u32 {
        const x = self(ptr);
        if (handle >= x.vocab) return error.InvalidHy4Token;
        return @intCast(handle);
    }

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const x = self(ptr);
        if (windows.len != out.len) return error.InvalidHy4Windows;
        const words = (x.vocab + 31) / 32;
        for (windows, out) |w, o| {
            try x.mine(w.stream);
            if (w.rows() != 1 or w.held != 0 or w.parents != null) return error.NoDraftHead;
            if (o.sampled.len != w.rows() or o.drafts.len != 0) return error.InvalidHy4Window;
            if (w.positions[0] != x.len + 1) return error.InvalidHy4Position;
            try x.m.pass(&.{w.pending}, x.raw);
            x.len += 1;
            const mask = if (w.masks.len >= words) w.masks[0..words] else &.{};
            o.sampled[0] = try x.draw(w.stream.sampling, w.positions[0], mask);
        }
    }

    fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        _ = windows; // one stream: the model's history rolls back to the kept prefix
        const x = self(ptr);
        const path = paths[paths.len - 1];
        for (path, 0..) |t, at| if (t != at) return error.InvalidHy4Tree;
        x.len -= 1;
        x.len += @intCast(path.len);
        x.m.history.items.len = x.len;
    }

    fn draft(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const x = self(ptr);
        for (requests) |r| {
            try x.mine(r.stream);
            if (r.depth != 0 or r.lanes != null) return error.NoDraftHead;
        }
    }

    fn release(ptr: *anyopaque, s: *Stream) void {
        const x = self(ptr);
        if (x.active != s) return;
        x.active = null;
        x.kept = true;
    }

    fn facts(_: *const Serve) lanes.Model {
        return .{ .exact_width = 1, .mtp = false, .speculate = false, .copy_drafts = false, .streams_exact = true, .max_streams = 1 };
    }

    const vtable: be.Backend.VTable = .{ .prefill = prefill, .first = first, .first_masked = firstMasked, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release };
};

fn gib(b: u64) f64 {
    return @as(f64, @floatFromInt(b)) / (1 << 30);
}

/// f32 as bf16 bits, to nearest even: the model's fp32 head rows into the host draw's bf16 rows.
fn bf16(v: f32) u16 {
    const b: u32 = @bitCast(v);
    if (!std.math.isFinite(v)) return @truncate(b >> 16);
    const rounded: u32 = (b +% 0x7FFF +% ((b >> 16) & 1)) >> 16;
    return @truncate(rounded);
}

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    if (kernels != null) return error.CapturedKernels;
    if (o.room) |r| if (r.read(r.ctx)) |room| {
        if (room < 8 << 30) return error.Hy4RoomShort; // the weights and the window's caches below it
    };
    if (o.context > capacity_most) return error.PastContext;
    const capacity = o.context;
    const x = try gpa.create(Serve);
    errdefer gpa.destroy(x);
    x.* = try Serve.init(gpa, io, ctx, dir, capacity);
    errdefer x.deinit(gpa);
    std.log.info("hy4 CUDA: {d}-token window, {d:.2} GiB of caches, prompts in chunks of {d} rows", .{ capacity, gib(x.stream_bytes), prompt_rows });
    return .{
        .backend = .{ .ptr = x, .vtable = &Serve.vtable },
        .facts = x.facts(),
        .rows = 1,
        .stream_bytes = x.stream_bytes,
        .free_streams = 1,
        .prompt_rows = prompt_rows,
        .ctx = x,
        .deinit = shutdown,
    };
}

fn shutdown(p: *anyopaque) void {
    const x: *Serve = @ptrCast(@alignCast(p));
    const g = x.gpa;
    x.deinit(g);
    g.destroy(x);
}



/// The folder's weight bytes: every shard's file size, once (a stand-in for the engine's own
/// weight-file pricing, which this module cannot reach).
fn wholeWeightBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8) !u64 {
    _ = a;
    var d = try std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true });
    defer d.close(io);
    var total: u64 = 0;
    var it = d.iterate();
    while (try it.next(io)) |e| {
        if (!std.mem.endsWith(u8, e.name, ".safetensors")) continue;
        const st = try d.statFile(io, e.name, .{});
        total = std.math.add(u64, total, st.size) catch return error.WeightSizeOverflow;
    }
    if (total == 0) return error.NoWeightFiles;
    return total;
}

/// The weight bytes the device must hold during the pre-open budget check: the checkpoint's weight
/// files less the Engram `multi_head_embedding` tables, which the model streams from disk
/// (O_DIRECT row gathers through bf16_rows.openDirect) and never uploads. A checkpoint whose
/// config doesn't price the tables keeps the plain weights.
pub fn textWeightBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8) !u64 {
    const whole = try wholeWeightBytes(a, io, dir);
    const tables = engramTableBytes(a, io, dir) catch |e| {
        std.debug.print("tf-debug: engramTableBytes err={s}\n", .{@errorName(e)});
        return whole;
    };
    std.debug.print("tf-debug: whole={d} tables={d}\n", .{ whole, tables });
    return if (tables > 0 and tables < whole) whole - tables else whole;
}


/// The Engram tables' bytes from config.json: every engram layer's rows (its per-head vocabulary
/// sizes, summed) times 96 BF16 columns per row.
fn engramTableBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8) !u64 {
    const fwd_cfg = @import("config.zig");
    const config_text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "config.json" }), a, .limited(1 << 20));
    const cfg = try fwd_cfg.parse(a, config_text);
    const t = cfg.value.text_config;
    var rows: u64 = 0;
    for (t.engram_layer_ids) |layer| rows += try t.engramRows(layer);
    const cols: u64 = 96;
    return rows * cols * 2;
}

/// A request or load this engine refuses, in words; null: none of its own.
pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.PastContext, error.BadHy4RunnerCapacity => "the prompt and its reply exceed the native Hy4-flash CUDA engine's 2,048-token window (the sparse attention's per-row key list): lower --context (the default 2,048 stands)",
        error.Hy4NotExl3 => "the checkpoint is not an EXL3 pack (its config.json carries no quantization_config with quant_method exl3): the CUDA engine loads only EXL3 routed experts with BF16 elsewhere",
        error.CapturedKernels => "Hy4-flash runs on the engine's own CUDA kernels: unset TENSORFOLD_CUDA_KERNELS",
        error.BuiltWithoutKernels => "this build carries no CUDA kernel images: build with -Dnvcc",
        error.BadHy4CheckpointTensor => "the checkpoint's tensors are not the layout the CUDA engine loads (EXL3 routed experts, BF16 elsewhere)",
        error.Hy4MemoryFloor => "the checkpoint's weights fit the device but free host memory fell below TF_HY4_MEM_FLOOR_GIB: free memory or lower that floor",
        error.NoDraftHead => "Hy4-flash on CUDA has no DFly2 head yet: serve without --drafter",
        error.Hy4RoomShort => "the CUDA memory budget leaves too little for Hy4-flash's weights and caches: free device memory or adjust TENSORFOLD_MEMORY_RESERVE_GIB / TENSORFOLD_CUDA_MEMORY_LIMIT_GB",
        else => null,
    };
}

test "bf16 rows at the extremes" {
    try std.testing.expectEqual(@as(u16, 0x3F80), bf16(1.0));
}
