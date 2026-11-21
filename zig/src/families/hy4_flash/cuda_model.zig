//! Hy4-flash text model on CUDA for EXL3 routed experts with BF16 elsewhere; one sequence, causal or windowed keys.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const config = @import("config.zig");
const ihc = @import("ihc.zig");
const ihc_cuda = @import("ihc_cuda.zig");
const mla = @import("mla_cuda.zig");
const m = @import("mla_weights.zig");
const attn = @import("mla_attention.zig");
const rope = @import("rope.zig");
const exl3 = @import("moe_exl3_cuda.zig");
const expert_weights = @import("exl3_weights.zig");
const engram = @import("engram.zig");
const engram_cuda = @import("engram_cuda.zig");
const direct = @import("bf16_rows.zig");

const H = 3072;
const S = ihc.streams;
const L = 1536;
const D = 13824;
const E = 512;
const K = 16;
const Q = 1024;
const sparse_stride = 2048;
const sliding_stride = 512;

var dump_rows: usize = 0;

pub const Kernels = struct {
    /// Every module image the model loaded, before its weights go.
    pub fn deinit(k: *Kernels) void {
        k.gemm.deinit();
        k.norms.unload();
        k.project.deinit();
        k.attention.deinit();
        k.ihc.deinit();
        k.experts.unload();
        k.act.unload();
        k.glue.unload();
        k.router.unload();
        k.engram.deinit();
    }
    gemm: cuda.vit.Gemms,
    norms: cuda.norms.Norms,
    project: mla.Projections,
    attention: mla.Attention,
    ihc: ihc_cuda.Kernels,
    experts: cuda.exl3_experts.Experts,
    act: cuda.act_rows.ActRows,
    glue: cuda.moe_glue.MoeGlue,
    router: cuda.route_rows.RouteRows,
    engram: engram_cuda.Kernels,

    /// The kernel images embedded in this build (`-Dnvcc` or `-Dprebuilt`).
    pub fn load(a: std.mem.Allocator, d: *const cuda.Driver) !Kernels {
        return .{
            .gemm = try cuda.vit.Gemms.init(a, d),
            .norms = try cuda.norms.Norms.load(d),
            .project = try mla.Projections.load(d),
            .attention = try mla.Attention.load(d),
            .ihc = try ihc_cuda.Kernels.load(d),
            .experts = try cuda.exl3_experts.Experts.load(d),
            .act = try cuda.act_rows.ActRows.load(d),
            .glue = try cuda.moe_glue.MoeGlue.load(d),
            .router = try cuda.route_rows.RouteRows.load(d),
            .engram = try engram_cuda.Kernels.load(d),
        };
    }
};

const Hc = struct { project: u64, bias: u64, scale: u64 };

const Attention = struct {
    sliding: bool,
    heads: u32,
    hc: Hc,
    input_norm: u64,
    q_a: u64,
    q_norm: u64,
    q_b: u64,
    kv_a: u64,
    kv_norm: u64,
    key: u64,
    value: u64,
    gate: u64,
    output: u64,
    output_norm: u64,
    sinks: u64,
    cache: cuda.DeviceBuffer,
};

const Dense = struct { gate: u64, up: u64, down: u64 };
const Sparse = struct {
    latent_in: u64,
    latent_out: u64,
    shared_gate: u64,
    shared_up: u64,
    shared_down: u64,
    router: u64,
    router_bias: u64,
    routed: exl3.Routed,
};

const Feed = struct {
    hc: Hc,
    input_norm: u64,
    output_norm: u64,
    mlp: union(enum) { dense: Dense, sparse: Sparse },
};

const Engram = struct {
    w: engram_cuda.Weights,
    table: direct.Table,
};

const Layer = struct { engram: ?Engram, attention: Attention, feed: Feed };

/// Per-pass device scratch sized for `max_rows`.
const Scratch = struct {
    streams: [3]cuda.DeviceBuffer, // [rows, 4, H], rotated through the layer halves
    collapsed: u64,
    post: u64,
    hidden: u64,
    norm_tmp: u64,
    q_raw: u64,
    q_tmp: u64,
    q_resid: u64,
    q: u64,
    kv: u64,
    gate: u64,
    absorb: u64,
    latent_out: u64,
    values: u64,
    branch: u64,
    branch_norm: u64,
    attn_scratch: u64,
    attn_scratch_bytes: usize,
    positions: cuda.DeviceBuffer,
    table: u64,
    slots: cuda.DeviceBuffer,
    counts: cuda.DeviceBuffer,
    write_slots: cuda.DeviceBuffer,
    media: u64,
    latent: u64,
    logits: u64,
    picks: u64,
    probabilities: u64,
    mix: u64,
    route_out: u64,
    ff_gate: u64,
    ff_up: u64,
    ff_inner: u64,
    shared: u64,
    e_raw: cuda.DeviceBuffer,
    e_normalized: u64,
    e_keys: u64,
    e_value: u64,
    head_hidden: u64,
    head_norm: u64,
    lm_logits: u64,
};

pub const Model = struct {
    a: std.mem.Allocator,
    d: *const cuda.Driver,
    s: cuda.Stream,
    k: *Kernels,
    text: config.Text,
    own: cuda.exl3_upload.Uploader,
    layers: []Layer,
    embed: u64,
    head_hc: Hc,
    final_norm: u64,
    lm_head: u64,
    scratch: Scratch,
    max_rows: usize,
    capacity: usize,
    history: std.ArrayList(u32) = .empty,
    weight_bytes: usize = 0,

    /// quantization_config { quant_method: "exl3" }: what this loader's routed-expert path reads.
    fn exl3Pack(a: std.mem.Allocator, text: []const u8) !bool {
        var parsed = std.json.parseFromSlice(std.json.Value, a, text, .{}) catch return error.BadHy4CheckpointTensor;
        defer parsed.deinit();
        if (parsed.value != .object) return false;
        const q = parsed.value.object.get("quantization_config") orelse return false;
        if (q != .object) return false;
        const method = q.object.get("quant_method") orelse return false;
        return method == .string and std.mem.eql(u8, method.string, "exl3");
    }

    fn tensor(ck: *core.Checkpoint, name: []const u8, dtype: anytype, shape: []const usize) ![]const u8 {
        const t = try ck.get(name);
        if (!t.is(dtype, shape)) {
            std.debug.print("shape-refuse {s}: stored {any} want {any}\n", .{ name, t.shape[0..t.rank], shape });
            return error.BadHy4CheckpointTensor;
        }
        return t.bytes;
    }


fn denseWeights(a: std.mem.Allocator, ck: *core.Checkpoint, name: []const u8, dtype: anytype, shape: []const usize, temp: std.mem.Allocator) !struct { bytes: []const u8, converted: ?[]u8 } {
    _ = a;
    const t = try ck.get(name);
    if (t.is(dtype, shape)) return .{ .bytes = t.bytes, .converted = null };
    if (t.rank != shape.len or !std.mem.eql(usize, t.shape[0..t.rank], shape))
        return error.BadHy4CheckpointTensor;
    if (dtype != .bf16 or t.dtype != .f8_e4m3) {
        std.debug.print("fp8-refuse {s}: req-dtype-tag {any} stored {any}\n", .{ name, dtype, t.dtype });
        return error.BadHy4CheckpointTensor;
    }
    var sbuf: [256]u8 = undefined;
    const scale_base = if (std.mem.endsWith(u8, name, ".weight")) name[0 .. name.len - ".weight".len] else name;
    const sname = try std.fmt.bufPrint(&sbuf, "{s}.weight_scale_inv", .{scale_base});
    std.debug.print("tf-scale {s}\n", .{sname});
    const scales = try ck.get(sname);
    const out = try temp.alloc(u8, t.bytes.len * 2);
    errdefer temp.free(out);
    const rows: usize = shape[0];
    const cols: usize = shape[1];
    const br: usize = (rows + 127) / 128;
    const bc: usize = (cols + 127) / 128;
    var inv: []align(1) const f32 = undefined;
    var wide: ?[]f32 = null;
    defer if (wide) |w| temp.free(w);
    switch (scales.dtype) {
        .f32 => {
            if (scales.bytes.len != br * bc * 4 or scales.rank != 2 or scales.shape[0] != br or scales.shape[1] != bc) {
                std.debug.print("unexpected scale {s}: {any} {any}\n", .{ sname, scales.dtype, scales.shape[0..scales.rank] });
                return error.BadHy4CheckpointTensor;
            }
            inv = @alignCast(std.mem.bytesAsSlice(f32, scales.bytes));
        },
        .bf16 => {
            if (scales.bytes.len != br * bc * 2 or scales.rank != 2 or scales.shape[0] != br or scales.shape[1] != bc) {
                std.debug.print("unexpected scale {s}: {any} {any}\n", .{ sname, scales.dtype, scales.shape[0..scales.rank] });
                return error.BadHy4CheckpointTensor;
            }
            const raw: []align(1) const u16 = @alignCast(std.mem.bytesAsSlice(u16, scales.bytes));
            const widened = try temp.alloc(f32, br * bc);
            for (raw, 0..) |h, i| {
                const e: u32 = @as(u32, h) << 16;
                widened[i] = @bitCast(e);
            }
            wide = widened;
            inv = @alignCast(widened);
        },
        else => {
            std.debug.print("unexpected scale {s}: {any} {any}\n", .{ sname, scales.dtype, scales.shape[0..scales.rank] });
            return error.BadHy4CheckpointTensor;
        },
    }
    const vals: []align(1) const u8 = t.bytes;
    const dst: []align(1) u16 = @alignCast(std.mem.bytesAsSlice(u16, out));
    var i: usize = 0;
    while (i < rows) : (i += 1) {

        var j: usize = 0;
        while (j < cols) : (j += 1) {
            dst[i * cols + j] = toBf16(f8ToF32(vals[i * cols + j]) * inv[(i / 128) * bc + j / 128]);
        }
    }
    return .{ .bytes = out, .converted = out };
}

fn f8ToF32(u: u8) f32 {
    const sign: f32 = if (u & 0x80 != 0) -1 else 1;
    const exp: u4 = @intCast((u >> 3) & 0xF);
    const man: u3 = @intCast(u & 0x7);
    // this pack's f8 has no NaN/Inf codes: exp 15 continues the curve (256..480)
    if (exp == 0) return sign * @as(f32, @floatFromInt(man)) * (1.0 / 512.0);
    const mag: f32 = 1.0 + @as(f32, @floatFromInt(man)) / 8.0;
    return sign * mag * std.math.pow(f32, 2, @as(f32, @floatFromInt(exp)) - 7.0);
}

fn toBf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    return @intCast((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}

    fn up(self: *Model, ck: *core.Checkpoint, comptime fmt: []const u8, args: anytype, dtype: anytype, shape: []const usize) !u64 {
        var buf: [192]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, fmt, args);
        const host = try denseWeights(self.a, ck, name, dtype, shape, self.a);
        defer if (host.converted) |b| self.a.free(b);
        return self.own.dev(host.bytes);
    }

    fn zeros(self: *Model, bytes: usize) !u64 {
        const z = try std.heap.page_allocator.alloc(u8, bytes);
        defer std.heap.page_allocator.free(z);
        @memset(z, 0);
        return self.own.dev(z);
    }

    pub fn open(a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, s: cuda.Stream, k: *Kernels, dir: []const u8, capacity: usize, max_rows: usize) !*Model {
        const config_text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "config.json" }), a, .limited(1 << 20));
        const cfg = try config.parse(a, config_text);
        // The checkpoint's own word: routed experts come as EXL3; anything else leaves before a load.
        if (!(try exl3Pack(a, config_text))) return error.Hy4NotExl3;
        const t = cfg.value.text_config;
        if (t.hidden_size != H or t.hc_mult != S or t.routed_expert_hidden_size != L or t.moe_intermediate_size != L or
            t.intermediate_size != D or t.n_routed_experts != E or t.num_experts_per_tok != K or t.q_lora_rank != Q)
            return error.BadHy4ModelShape;
        if (capacity > sparse_stride or max_rows == 0 or max_rows > 1024) return error.BadHy4RunnerCapacity;
        const ck = try a.create(core.Checkpoint);
        ck.* = try core.Checkpoint.openModel(a, io, dir);
        const self = try a.create(Model);
        self.* = .{
            .a = a,
            .d = d,
            .s = s,
            .k = k,
            .text = t,
            // Upload tables and staging go through a freeing allocator: the arena would keep them all.
            .own = .{ .d = d, .gpa = std.heap.page_allocator, .max_buffers = 1 << 20, .slab = 256 << 20 },
            .layers = try a.alloc(Layer, t.num_hidden_layers),
            .embed = undefined,
            .head_hc = undefined,
            .final_norm = undefined,
            .lm_head = undefined,
            .scratch = undefined,
            .max_rows = max_rows,
            .capacity = capacity,
        };
        const V: usize = t.vocab_size;
        self.embed = try self.up(ck, "model.embed_tokens.weight", .{}, .bf16, &.{ V, H });
        self.head_hc = .{
            .project = try self.up(ck, "model.hc_head.hc_head_fn", .{}, .f32, &.{ S, S * H }),
            .bias = try self.up(ck, "model.hc_head.hc_head_base", .{}, .f32, &.{S}),
            .scale = try self.up(ck, "model.hc_head.hc_head_scale", .{}, .f32, &.{1}),
        };
        self.final_norm = try self.up(ck, "model.norm.weight", .{}, .bf16, &.{H});
        self.lm_head = try self.up(ck, "lm_head.weight", .{}, .bf16, &.{ V, H });
        ck.release();
        // UMA guard: device memory is not charged to the cgroup, so stop before the system runs dry.
        const floor_gib = try @import("sys").environment.read(usize, "TF_HY4_MEM_FLOOR_GIB", 10);
        for (self.layers, 0..) |*layer, i| {
            const avail = memAvailable();
            std.debug.print("layer {d:>2}: device {d:.2} GiB, MemAvailable {d:.2} GiB\n", .{ i, gib(self.own.bytes), gib(avail) });
            if (avail < floor_gib << 30) {
                std.debug.print("stopping: MemAvailable below the {d} GiB floor\n", .{floor_gib});
                return error.Hy4MemoryFloor;
            }
            layer.* = try self.loadLayer(a, io, ck, dir, i);
            // The layer is on the device: its mapped shard pages (and page cache) would otherwise stay resident.
            ck.release();
        }
        std.debug.print("all layers: device {d:.2} GiB, MemAvailable {d:.2} GiB\n", .{ gib(self.own.bytes), gib(memAvailable()) });
        self.weight_bytes = self.own.bytes;
        self.scratch = try self.allocScratch();
        return self;
    }


    fn probe(self: *Model, ptr: u64, n_elems: usize, comptime kind: enum { bf16, f32 }, label: []const u8) !void {
        const bytes = n_elems * if (kind == .bf16) @as(usize, 2) else 4;
        const host = try std.heap.page_allocator.alloc(u8, bytes);
        defer std.heap.page_allocator.free(host);
        try self.s.synchronize();
        try cuda.memory.download(self.d, ptr, host);
        var lo: f32 = std.math.floatMax(f32);
        var hi: f32 = -std.math.floatMax(f32);
        var bad: usize = 0;
        for (0..n_elems) |i| {
            const v: f32 = switch (kind) {
                .bf16 => @bitCast(@as(u32, @as(u16, std.mem.readInt(u16, host[2 * i ..][0..2], .little))) << 16),
                .f32 => @bitCast(std.mem.readInt(u32, host[4 * i ..][0..4], .little)),
            };
            if (!std.math.isFinite(v)) bad += 1 else {
                lo = @min(lo, v);
                hi = @max(hi, v);
            }
        }
        std.debug.print("tf-probe {s} lo={d:.3} hi={d:.3} nonfinite={d}/{d}\n", .{ label, lo, hi, bad, n_elems });
    }
    fn gib(bytes: usize) f64 {
        return @as(f64, @floatFromInt(bytes)) / (1 << 30);
    }

    /// MemAvailable from /proc/meminfo in bytes (0 when unreadable, which trips the floor).
    fn memAvailable() usize {
        var buf: [4096]u8 = undefined;
        const fd = std.c.open("/proc/meminfo", .{});
        if (fd < 0) return 0;
        defer _ = std.c.close(fd);
        const got = std.c.read(fd, &buf, buf.len);
        if (got <= 0) return 0;
        const n: usize = @intCast(got);
        var it = std.mem.tokenizeScalar(u8, buf[0..n], '\n');
        while (it.next()) |line| if (std.mem.startsWith(u8, line, "MemAvailable:")) {
            var f = std.mem.tokenizeScalar(u8, line["MemAvailable:".len..], ' ');
            const kb = std.fmt.parseInt(usize, f.next() orelse return 0, 10) catch return 0;
            return kb * 1024;
        };
        return 0;
    }

    fn loadLayer(self: *Model, a: std.mem.Allocator, io: std.Io, ck: *core.Checkpoint, dir: []const u8, i: usize) !Layer {
        const t = self.text;
        const sliding = t.layer_types[i] == .sliding_attention;
        const heads: u32 = if (sliding) t.swa_num_attention_heads else t.num_attention_heads;
        const hh: usize = heads;
        const p = "model.layers.{d}.";
        var kv_b_name: [96]u8 = undefined;
        const kv_b_host = try denseWeights(a, ck, try std.fmt.bufPrint(&kv_b_name, p ++ "self_attn.kv_b_proj.weight", .{i}), .bf16, &.{ hh * (m.nope + m.value_width), m.latent }, a);
        defer if (kv_b_host.converted) |b| a.free(b);
        const kv_b = kv_b_host.bytes;
        const tmp = std.heap.page_allocator;
        const split = try m.splitBf16(tmp, kv_b, heads);
        defer tmp.free(split.key);
        defer tmp.free(split.value);
        var cache = try cuda.DeviceBuffer.alloc(self.d, self.capacity * m.absorbed_stride * 2);
        try cache.fill8(0, null);
        const att: Attention = .{
            .sliding = sliding,
            .heads = heads,
            .hc = .{
                .project = try self.up(ck, p ++ "hc_attn_layer.hc_pre.hc_fn", .{i}, .f32, &.{ 2 * S, S * H }),
                .bias = try self.up(ck, p ++ "hc_attn_layer.hc_pre.hc_base", .{i}, .f32, &.{2 * S}),
                .scale = try self.up(ck, p ++ "hc_attn_layer.hc_pre.hc_scale", .{i}, .f32, &.{2}),
            },
            .input_norm = try self.up(ck, p ++ "input_layernorm.weight", .{i}, .bf16, &.{H}),
            .q_a = try self.up(ck, p ++ "self_attn.q_a_proj.weight", .{i}, .bf16, &.{ Q, H }),
            .q_norm = try self.up(ck, p ++ "self_attn.q_a_layernorm.weight", .{i}, .bf16, &.{Q}),
            .q_b = try self.up(ck, p ++ "self_attn.q_b_proj.weight", .{i}, .bf16, &.{ hh * m.query_stride, Q }),
            .kv_a = try self.up(ck, p ++ "self_attn.kv_a_proj_with_mqa.weight", .{i}, .bf16, &.{ m.absorbed_stride, H }),
            .kv_norm = try self.up(ck, p ++ "self_attn.kv_a_layernorm.weight", .{i}, .bf16, &.{m.latent}),
            .key = try self.own.dev(std.mem.sliceAsBytes(split.key)),
            .value = try self.own.dev(std.mem.sliceAsBytes(split.value)),
            .gate = try self.up(ck, p ++ "self_attn.linear_gate.weight", .{i}, .bf16, &.{ hh * m.value_width, H }),
            .output = try self.up(ck, p ++ "self_attn.o_proj.weight", .{i}, .bf16, &.{ H, hh * m.value_width }),
            .output_norm = try self.up(ck, p ++ "self_attn.post_proj_norm.weight", .{i}, .bf16, &.{H}),
            .sinks = try self.up(ck, p ++ "self_attn.learnable_sink_param", .{i}, .f32, &.{hh}),
            .cache = cache,
        };
        const dense = t.mlp_layer_types[i] == .dense;
        const feed: Feed = .{
            .hc = .{
                .project = try self.up(ck, p ++ "hc_mlp_layer.hc_pre.hc_fn", .{i}, .f32, &.{ 2 * S, S * H }),
                .bias = try self.up(ck, p ++ "hc_mlp_layer.hc_pre.hc_base", .{i}, .f32, &.{2 * S}),
                .scale = try self.up(ck, p ++ "hc_mlp_layer.hc_pre.hc_scale", .{i}, .f32, &.{2}),
            },
            .input_norm = try self.up(ck, p ++ "post_attention_layernorm.weight", .{i}, .bf16, &.{H}),
            .output_norm = if (dense)
                try self.up(ck, p ++ "mlp.post_fc2_norm.weight", .{i}, .bf16, &.{H})
            else
                try self.up(ck, p ++ "mlp.combine_output_norm.weight", .{i}, .bf16, &.{H}),
            .mlp = if (dense) .{ .dense = .{
                .gate = try self.up(ck, p ++ "mlp.gate_proj.weight", .{i}, .bf16, &.{ D, H }),
                .up = try self.up(ck, p ++ "mlp.up_proj.weight", .{i}, .bf16, &.{ D, H }),
                .down = try self.up(ck, p ++ "mlp.down_proj.weight", .{i}, .bf16, &.{ H, D }),
            } } else .{ .sparse = blk: {
                const experts = try std.heap.page_allocator.alloc(expert_weights.Expert, E);
                defer std.heap.page_allocator.free(experts);
                for (experts, 0..) |*f, j| {
                    f.* = expert_weights.readExpert(ck, i, j, L, L) catch |err| {
                        std.debug.print("expert {d} failed: {s}\n", .{ j, @errorName(err) });
                        return err;
                    };
                }
                break :blk .{
                    .latent_in = try self.up(ck, p ++ "mlp.routed_expert_down_proj.weight", .{i}, .bf16, &.{ L, H }),
                    .latent_out = try self.up(ck, p ++ "mlp.routed_expert_up_proj.weight", .{i}, .bf16, &.{ H, L }),
                    .shared_gate = try self.up(ck, p ++ "mlp.shared_experts.gate_proj.weight", .{i}, .bf16, &.{ L, H }),
                    .shared_up = try self.up(ck, p ++ "mlp.shared_experts.up_proj.weight", .{i}, .bf16, &.{ L, H }),
                    .shared_down = try self.up(ck, p ++ "mlp.shared_experts.down_proj.weight", .{i}, .bf16, &.{ H, L }),
                    .router = try self.up(ck, p ++ "mlp.gate.weight", .{i}, .bf16, &.{ E, H }),
                    .router_bias = try self.up(ck, p ++ "mlp.gate.e_score_correction_bias", .{i}, .f32, &.{E}),
                    .routed = try exl3.Routed.init(&self.own, experts, L, L, self.max_rows, K, t.swiglu_limit),
                };
            } },
        };
        var eng: ?Engram = null;
        if (std.mem.indexOfScalar(u32, t.engram_layer_ids, @intCast(i + 1)) != null) {
            const e = "model.layers.{d}.engram_layer.";
            const ein: usize = engram_cuda.input;
            var key: [S]u64 = undefined;
            for (&key, 0..) |*dst, j| dst.* = try self.up(ck, e ++ "key_projs.{d}.weight", .{ i, j }, .bf16, &.{ H, ein });
            const cat = struct {
                fn run(mm: *Model, c: *core.Checkpoint, layer: usize, comptime fmt: []const u8, count: usize, size: usize) !u64 {
                    const out = try mm.a.alloc(u8, count * size * 2);
                    defer mm.a.free(out);
                    for (0..count) |j| {
                        var nb: [128]u8 = undefined;
                        @memcpy(out[j * size * 2 ..][0 .. size * 2], try tensor(c, try std.fmt.bufPrint(&nb, fmt, .{ layer, j }), .bf16, &.{size}));
                    }
                    return mm.own.dev(out);
                }
            }.run;
            const w: engram_cuda.Weights = .{
                .lookup_gamma = try cat(self, ck, i, e ++ "lookup_per_head_norms.{d}.weight", engram_cuda.columns, engram_cuda.head_width),
                .value = try self.up(ck, e ++ "value_proj.weight", .{i}, .bf16, &.{ H, ein }),
                .value_bias = try self.up(ck, e ++ "value_proj.bias", .{i}, .bf16, &.{H}),
                .value_gain = try self.up(ck, e ++ "unified_pre_gamma", .{i}, .bf16, &.{H}),
                .key = key,
                .key_bias = try cat(self, ck, i, e ++ "key_projs.{d}.bias", S, H),
                .key_gain = try cat(self, ck, i, e ++ "norm_key.{d}.weight", S, H),
                .query_gain = try cat(self, ck, i, e ++ "norm_query.{d}.weight", S, H),
                .delta_gain = try cat(self, ck, i, e ++ "unified_per_hc_scales.{d}", S, H),
            };
            var nb: [128]u8 = undefined;
            const lookup = (try ck.get(try std.fmt.bufPrint(&nb, e ++ "lookup_table", .{i}))).bytes;
            const map = try engram.TokenMap.open(lookup, t);
            const hash = try engram.Layer.open(t, @intCast(i + 1));
            var tb: [128]u8 = undefined;
            const table_name = try std.fmt.bufPrint(&tb, e ++ "multi_head_embedding.embedding.weight", .{i});
            const shard = try shardOf(a, io, dir, table_name);
            // The config's per-layer row count, so a wrong or repacked table fails here and not mid-forward.
            const rows_expected = try t.engramRows(@intCast(i + 1));
            eng = .{ .w = w, .table = try direct.openDirect(a, shard, table_name, map, hash, rows_expected) };
        }
        return .{ .engram = eng, .attention = att, .feed = feed };
    }

    /// The safetensors file holding `name` (the Engram tables sit alone in their shards).
    fn shardOf(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) ![:0]const u8 {
        const index = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" }), a, .limited(1 << 26));
        const parsed = try std.json.parseFromSliceLeaky(std.json.Value, a, index, .{});
        const file = parsed.object.get("weight_map").?.object.get(name) orelse return error.MissingHy4EngramTable;
        return std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir, file.string }, 0);
    }

    fn allocScratch(self: *Model) !Scratch {
        const r = self.max_rows;
        const shape = try attn.Shape.init(@intCast(r), 64, sparse_stride);
        var sc: Scratch = undefined;
        for (&sc.streams) |*x| x.* = try cuda.DeviceBuffer.alloc(self.d, r * S * H * 2);
        sc.collapsed = try self.zeros(r * H * 2);
        sc.post = try self.zeros(r * S * 4);
        sc.hidden = try self.zeros(r * H * 2);
        sc.norm_tmp = try self.zeros(r * D * 2);
        sc.q_raw = try self.zeros(r * Q * 2);
        sc.q_tmp = try self.zeros(r * Q * 2);
        sc.q_resid = try self.zeros(r * Q * 2);
        sc.q = try self.zeros(r * 64 * m.query_stride * 2);
        sc.kv = try self.zeros(r * m.absorbed_stride * 2);
        sc.gate = try self.zeros(r * 64 * m.value_width * 2);
        sc.absorb = try self.zeros(r * 64 * m.absorbed_stride * 2);
        sc.latent_out = try self.zeros(r * 64 * m.latent * 2);
        sc.values = try self.zeros(r * 64 * m.value_width * 2);
        sc.branch = try self.zeros(r * H * 2);
        sc.branch_norm = try self.zeros(r * H * 2);
        sc.attn_scratch_bytes = shape.scratchBytes();
        sc.attn_scratch = try self.zeros(sc.attn_scratch_bytes);
        sc.positions = try cuda.DeviceBuffer.alloc(self.d, r * 12);
        sc.table = try self.zeros(r * 64 * 2);
        sc.slots = try cuda.DeviceBuffer.alloc(self.d, r * sparse_stride * 4);
        sc.counts = try cuda.DeviceBuffer.alloc(self.d, r * 4);
        sc.write_slots = try cuda.DeviceBuffer.alloc(self.d, r * 4);
        sc.media = try self.zeros(r * 4);
        sc.latent = try self.zeros(r * L * 2);
        sc.logits = try self.zeros(r * E * 4);
        sc.picks = try self.zeros(r * K * 4);
        sc.probabilities = try self.zeros(r * K * 4);
        sc.mix = try self.zeros(r * L * 2);
        sc.route_out = try self.zeros(r * H * 2);
        sc.ff_gate = try self.zeros(r * D * 2);
        sc.ff_up = try self.zeros(r * D * 2);
        sc.ff_inner = try self.zeros(r * D * 2);
        sc.shared = try self.zeros(r * H * 2);
        sc.e_raw = try cuda.DeviceBuffer.alloc(self.d, r * engram_cuda.input * 2);
        sc.e_normalized = try self.zeros(r * engram_cuda.input * 2);
        sc.e_keys = try self.zeros(S * r * H * 2);
        sc.e_value = try self.zeros(r * H * 2);
        sc.head_hidden = try self.zeros(r * H * 2);
        sc.head_norm = try self.zeros(r * H * 2);
        sc.lm_logits = try self.zeros(@as(usize, self.text.vocab_size) * 4);
        return sc;
    }

    fn hcShape(self: *const Model, rows: usize, gates: u32) ihc.Shape {
        return .{ .rows = @intCast(rows), .width = H, .gates = gates, .eps = self.text.rms_norm_eps, .gate_eps = self.text.hc_eps, .post_magnitude = self.text.hc_magnitude };
    }

    /// Every device buffer the model owns, then the uploads, then the engram tables' files.
    pub fn deinit(self: *Model) void {
        for (self.layers) |*layer| {
            layer.attention.cache.free();
            if (layer.engram) |*e| e.table.close();
        }
        const sc = &self.scratch;
        for (&sc.streams) |*b| b.free();
        sc.positions.free();
        sc.slots.free();
        sc.counts.free();
        sc.write_slots.free();
        sc.e_raw.free();
        self.own.deinit();
    }


    /// When TF_HY4_DUMP is set, append a stage's full row set as raw bf16 to /tmp/tf_dump.bin.
    fn dumpBuf(self: *Model, li: usize, name: []const u8, ptr: u64, bytes: usize) !void {
        if (li != 0 or dump_rows > 8) return;
        if (std.c.getenv("TF_HY4_DUMP") == null) return;
        const host = try std.heap.page_allocator.alloc(u8, bytes);
        defer std.heap.page_allocator.free(host);
        try self.s.synchronize();
        try cuda.memory.download(self.d, ptr, host);
        var head: [128]u8 = undefined;
        const text = try std.fmt.bufPrint(&head, "tf-dump {s} rows-bytes {d}\n", .{ name, bytes });
        const c = std.c;
        const file = (c.fopen("/tmp/tf_dump.bin", "ab") orelse return);
        _ = c.fwrite(text.ptr, 1, text.len, file);
        _ = c.fwrite(host.ptr, 1, host.len, file);
        _ = c.fclose(file);
    }

    /// Checksum the last row of the live stream `x` (one of the sc.streams slot buffers).
    fn traceRow(self: *Model, li: usize, stage: usize, x: usize, rows: usize) !void {
        if (std.c.getenv("TF_HY4_TRACE") == null) return;
        const row_bytes = S * H * 2;
        const hb = try self.a.alloc(u8, row_bytes);
        defer self.a.free(hb);
        try cuda.memory.download(self.d, @as(u64, x + (rows - 1) * row_bytes), hb);
        var sum: u64 = 0;
        var hash: u64 = 0;
        for (hb, 0..) |b, i| {
            sum += b;
            hash = (@as(u64, hash) *% 1000003 +% @as(u64, b) +% @as(u64, i)) & 0xfffffffffffffff;
        }
        std.debug.print("tftrace layer={d} stage={d} rows={d} sum={d} hash={d}\n", .{ li, stage, rows, sum, hash });
    }
    fn traceBuf(self: *Model, li: usize, name: []const u8, ptr: u64, row_bytes: usize, rows: usize) !void {
        if (std.c.getenv("TF_HY4_TRACE") == null) return;
        const hb = try self.a.alloc(u8, row_bytes);
        defer self.a.free(hb);
        try cuda.memory.download(self.d, ptr + @as(u64, (rows - 1) * row_bytes), hb);
        var sum: u64 = 0;
        var hash: u64 = 0;
        for (hb, 0..) |b, i| {
            sum += b;
            hash = (hash *% 1000003 +% @as(u64, b) +% @as(u64, i)) & ((1 << 56) - 1);
        }
        std.debug.print("tfbuf layer={d} stage={s} rows={d} sum={d} hash={d}\n", .{ li, name, rows, sum, hash });
    }

    pub fn pass(self: *Model, tokens: []const u32, logits_out: []f32) !void {
        dump_rows = tokens.len;
        const rows = tokens.len;
        const start = self.history.items.len;
        if (rows == 0 or rows > self.max_rows or start + rows > self.capacity) return error.BadHy4PassRows;
        try self.history.appendSlice(self.a, tokens);
        const sc = self.scratch;
        const s = self.s;
        const k = self.k;
        const eps = self.text.rms_norm_eps;
        // Embedding into four identical streams.
        var x = sc.streams[0].ptr;
        for (tokens, 0..) |tok, r| for (0..S) |st| {
            try sc.streams[0].copyFrom((r * S + st) * H * 2, self.embed + @as(u64, tok) * H * 2, H * 2, s.handle);
        };
        try self.traceRow(999, 0, x, rows);
        // Positions (text: t = h = w), write slots and key lists for both attention kinds.
        const positions = try self.a.alloc(i32, rows * 3);
        defer self.a.free(positions);
        const writes = try self.a.alloc(u32, rows);
        defer self.a.free(writes);
        for (0..rows) |r| {
            const p: i32 = @intCast(start + r);
            positions[r * 3 ..][0..3].* = .{ p, p, p };
            writes[r] = @intCast(start + r);
        }
        try sc.positions.uploadAsync(0, std.mem.sliceAsBytes(positions), s.handle);
        try sc.write_slots.uploadAsync(0, std.mem.sliceAsBytes(writes), s.handle);        try self.dumpBuf(0, "positions", sc.positions.ptr, rows * 12);
        try self.dumpBuf(0, "writes", sc.write_slots.ptr, rows * 4);

        const keys_sparse = try self.a.alloc(u32, rows * sparse_stride);
        defer self.a.free(keys_sparse);
        const keys_sliding = try self.a.alloc(u32, rows * sliding_stride);
        defer self.a.free(keys_sliding);
        const counts_sparse = try self.a.alloc(u32, rows);
        defer self.a.free(counts_sparse);
        const counts_sliding = try self.a.alloc(u32, rows);
        defer self.a.free(counts_sliding);
        @memset(keys_sparse, 0);
        @memset(keys_sliding, 0);
        for (0..rows) |r| {
            const pos = start + r;
            counts_sparse[r] = @intCast(pos + 1);
            for (0..pos + 1) |j| keys_sparse[r * sparse_stride + j] = @intCast(j);
            const first = if (pos + 1 > sliding_stride) pos + 1 - sliding_stride else 0;
            counts_sliding[r] = @intCast(pos + 1 - first);
            for (first..pos + 1, 0..) |j, n| keys_sliding[r * sliding_stride + n] = @intCast(j);
        }
        // Engram windows from the committed history.
        const windows = try self.a.alloc([engram.order]u32, rows);
        defer self.a.free(windows);
        const pad = self.text.engram_pad_id;
        for (windows, 0..) |*w, r| {
            const pos = start + r;
            for (w, 0..) |*v, back| v.* = if (pos >= back) self.history.items[pos - back] else pad;
        }
        const raw_rows = try self.a.alloc(u16, rows * engram_cuda.input);
        defer self.a.free(raw_rows);

        var slot: usize = 0; // index of the stream buffer x lives in
        for (self.layers, 0..) |*layer, li| {
            // Engram residual into the streams.
            if (layer.engram) |*e| {
                try e.table.gather(self.a, windows, raw_rows);
                try sc.e_raw.uploadAsync(0, std.mem.sliceAsBytes(raw_rows), s.handle);
                const out = sc.streams[(slot + 1) % 3].ptr;
                try k.engram.forward(s, &k.gemm, e.w, .{
                    .raw = sc.e_raw.ptr,
                    .normalized = sc.e_normalized,
                    .keys = sc.e_keys,
                    .value = sc.e_value,
                    .media = sc.media,
                    .state = x,
                    .output = out,
                }, @intCast(rows), @intCast(rows), H, self.text.engram_fp8_embedding);
                slot = (slot + 1) % 3;
                x = out;
                try self.traceBuf(li, "engram_out", out, S * H * 2, rows);
            }
            // Attention half.
            const at = &layer.attention;
            const hh: usize = at.heads;
            try k.ihc.collapse(s, .{ .hidden = x, .projection = at.hc.project, .bias = at.hc.bias, .scale = at.hc.scale, .output = sc.collapsed, .post = sc.post }, self.hcShape(rows, 2 * S));
            try self.traceBuf(li, "collapsed", sc.collapsed, H * 2, rows);
            try self.dumpBuf(0, "x_in", x, rows * S * H * 2);
            try self.dumpBuf(0, "collapsed", sc.collapsed, rows * H * 2);
            try self.dumpBuf(0, "post", sc.post, rows * S * 4);
            try self.traceBuf(li, "post", sc.post, S * 4, rows);
            try k.norms.addRmsTail(s, sc.collapsed, null, at.input_norm, sc.norm_tmp, sc.hidden, null, rows, H, eps, .{ .round = true });
            try self.traceBuf(li, "hidden", sc.hidden, H * 2, rows);
            try self.dumpBuf(0, "hidden", sc.hidden, rows * H * 2);
            try k.gemm.run(s, sc.hidden, at.q_a, sc.q_raw, rows, Q, H, .bf16);
            try self.traceBuf(li, "q_raw", sc.q_raw, Q * 2, rows);
            try self.dumpBuf(0, "q_raw", sc.q_raw, rows * Q * 2);
            try k.norms.addRmsTail(s, sc.q_raw, null, at.q_norm, sc.q_tmp, sc.q_resid, null, rows, Q, eps, .{ .round = true });
            try k.gemm.run(s, sc.q_resid, at.q_b, sc.q, rows, hh * m.query_stride, Q, .bf16);
            try self.traceBuf(li, "q", sc.q, hh * m.query_stride * 2, rows);
            try self.dumpBuf(0, "q_resid", sc.q_resid, rows * Q * 2);
            try self.dumpBuf(0, "q", sc.q, rows * hh * m.query_stride * 2);
            try k.gemm.run(s, sc.hidden, at.kv_a, sc.kv, rows, m.absorbed_stride, H, .bf16);
            try self.traceBuf(li, "kv", sc.kv, m.absorbed_stride * 2, rows);
            try self.dumpBuf(0, "kv", sc.kv, rows * m.absorbed_stride * 2);
            try k.gemm.run(s, sc.hidden, at.gate, sc.gate, rows, hh * m.value_width, H, .bf16);
            try self.traceBuf(li, "gate", sc.gate, hh * m.value_width * 2, rows);
            try self.dumpBuf(0, "gate", sc.gate, rows * hh * m.value_width * 2);
            try self.dumpBuf(0, "table", sc.table, rows * 64 * 2);
            const stride: usize = if (at.sliding) sliding_stride else sparse_stride;
            try sc.slots.uploadAsync(0, std.mem.sliceAsBytes(if (at.sliding) keys_sliding else keys_sparse), s.handle);
            try sc.counts.uploadAsync(0, std.mem.sliceAsBytes(if (at.sliding) counts_sliding else counts_sparse), s.handle);
            const shape = try attn.Shape.init(@intCast(rows), at.heads, @intCast(stride));
            const p_params = try self.text.rotary(if (at.sliding) .sliding_attention else .deepseek_sparse_attention);
            try self.dumpBuf(0, "positions2", sc.positions.ptr, rows * 12);
            try k.project.rotaryTable(s, sc.positions.ptr, sc.table, shape.rows, p_params);
            try k.project.absorb(s, sc.q, at.key, sc.absorb, shape.rows, shape.heads);
            try self.traceBuf(li, "absorb0", sc.absorb, hh * m.absorbed_stride * 2, rows);
            try k.project.rotate(s, sc.q + m.nope * 2, sc.table, sc.absorb + m.latent * 2, .{ .rows = shape.rows, .heads = shape.heads, .input_stride = m.query_stride, .output_stride = m.absorbed_stride });
            try self.traceBuf(li, "rot", sc.absorb, hh * m.absorbed_stride * 2, rows);
            try self.dumpBuf(0, "absorb", sc.absorb, rows * hh * m.absorbed_stride * 2);
            try k.project.cacheKeys(s, sc.kv, at.kv_norm, sc.table, sc.write_slots.ptr, at.cache.ptr, .{ .rows = shape.rows, .capacity = @intCast(self.capacity), .epsilon = eps });
            try k.attention.attend(s, .{ .query = sc.absorb, .cache = at.cache.ptr, .slots = sc.slots.ptr, .counts = sc.counts.ptr, .sinks = at.sinks, .scratch = sc.attn_scratch, .scratch_bytes = sc.attn_scratch_bytes, .output = sc.latent_out }, shape);
            try self.traceBuf(li, "attend", sc.latent_out, hh * m.latent * 2, rows);
            try self.dumpBuf(0, "latent_out", sc.latent_out, rows * hh * m.latent * 2);

            try k.project.expand(s, sc.latent_out, at.value, sc.values, shape.rows, shape.heads);
            try self.traceBuf(li, "expand", sc.values, hh * m.value_width * 2, rows);
            try self.dumpBuf(0, "values", sc.values, rows * hh * m.value_width * 2);

            try k.project.gate(s, sc.values, sc.gate, shape.rows, shape.heads);
            try k.gemm.run(s, sc.values, at.output, sc.branch, rows, H, hh * m.value_width, .bf16);
            try self.traceBuf(li, "branch", sc.branch, H * 2, rows);
            try self.dumpBuf(0, "branch", sc.branch, rows * H * 2);
            try k.norms.addRmsTail(s, sc.branch, null, at.output_norm, sc.norm_tmp, sc.branch_norm, null, rows, H, eps, .{ .round = true });
            try self.traceBuf(li, "branch_norm", sc.branch_norm, H * 2, rows);
            try self.dumpBuf(0, "branch_norm", sc.branch_norm, rows * H * 2);
            const after_attn = sc.streams[(slot + 1) % 3].ptr;
            try k.ihc.add(s, .{ .residual = x, .branch = sc.branch_norm, .post = sc.post, .output = after_attn }, self.hcShape(rows, 2 * S));
            slot = (slot + 1) % 3;
            x = after_attn;
            try self.traceRow(li, 1, x, rows);
            // Feed-forward half.
            const f = &layer.feed;
            try k.ihc.collapse(s, .{ .hidden = x, .projection = f.hc.project, .bias = f.hc.bias, .scale = f.hc.scale, .output = sc.collapsed, .post = sc.post }, self.hcShape(rows, 2 * S));
                        try self.dumpBuf(0, "ff_x", x, rows * S * H * 2);
            try self.dumpBuf(0, "ff_collapsed", sc.collapsed, rows * H * 2);
            try self.dumpBuf(0, "ff_post", sc.post, rows * S * 4);
            try k.norms.addRmsTail(s, sc.collapsed, null, f.input_norm, sc.norm_tmp, sc.hidden, null, rows, H, eps, .{ .round = true });
                        try self.dumpBuf(0, "ff_hidden", sc.hidden, rows * H * 2);
            switch (f.mlp) {
                .dense => |dn| {
                    try k.gemm.run(s, sc.hidden, dn.gate, sc.ff_gate, rows, D, H, .bf16);
                    try k.gemm.run(s, sc.hidden, dn.up, sc.ff_up, rows, D, H, .bf16);
                    try k.act.swiglu(s, sc.ff_gate, D, sc.ff_up, D, sc.ff_inner, null, rows, D);
                    try k.gemm.run(s, sc.ff_inner, dn.down, sc.shared, rows, H, D, .bf16);
                                try self.dumpBuf(0, "ff_gate", sc.ff_gate, rows * D * 2);
            try self.dumpBuf(0, "ff_up", sc.ff_up, rows * D * 2);
            try self.dumpBuf(0, "ff_inner", sc.ff_inner, rows * D * 2);
            try self.dumpBuf(0, "ff_shared", sc.shared, rows * H * 2);

                    try k.norms.addRmsTail(s, sc.shared, null, f.output_norm, sc.norm_tmp, sc.branch_norm, null, rows, H, eps, .{ .round = true });
                                try self.dumpBuf(0, "ff_branch_norm", sc.branch_norm, rows * H * 2);
                },
                .sparse => |sp| {
                    try k.gemm.run(s, sc.hidden, sp.latent_in, sc.latent, rows, L, H, .bf16);
                    try k.router.logits(s, sc.hidden, H, sp.router, sc.logits, rows, H, E);
                    try k.glue.topk(s, sc.logits, sp.router_bias, .{ .experts = E, .topk = K, .shared = false, .score = true, .normalize = true, .scale = self.text.routed_scaling_factor }, sc.picks, sc.probabilities, rows);
                    const y = try sp.routed.run(&k.experts, s, sc.latent, sc.picks, rows);
                    try k.glue.combine(s, y, .f32, sc.probabilities, sc.mix, .bf16, rows, K, L);
                    try k.gemm.run(s, sc.mix, sp.latent_out, sc.route_out, rows, H, L, .bf16);
                    try k.gemm.run(s, sc.hidden, sp.shared_gate, sc.ff_gate, rows, L, H, .bf16);
                    try k.gemm.run(s, sc.hidden, sp.shared_up, sc.ff_up, rows, L, H, .bf16);
                    try k.act.swiglu(s, sc.ff_gate, L, sc.ff_up, L, sc.ff_inner, null, rows, L);
                    try k.gemm.run(s, sc.ff_inner, sp.shared_down, sc.shared, rows, H, L, .bf16);
                    try k.norms.addRmsTail(s, sc.route_out, sc.shared, f.output_norm, sc.norm_tmp, sc.branch_norm, null, rows, H, eps, .{ .round = true });
                },
            }
            const after_ff = sc.streams[(slot + 1) % 3].ptr;
            try k.ihc.add(s, .{ .residual = x, .branch = sc.branch_norm, .post = sc.post, .output = after_ff }, self.hcShape(rows, 2 * S));
            slot = (slot + 1) % 3;
            x = after_ff;            try self.dumpBuf(0, "ff_out", after_ff, rows * S * H * 2);

            try self.traceRow(li, 2, x, rows);
        }
        // Hyper head, final norm and LM head on the last row.
        const last = x + (rows - 1) * S * H * 2;
        try k.ihc.collapse(s, .{ .hidden = last, .projection = self.head_hc.project, .bias = self.head_hc.bias, .scale = self.head_hc.scale, .output = sc.head_hidden, .post = 0 }, self.hcShape(1, S));
        try k.norms.addRmsTail(s, sc.head_hidden, null, self.final_norm, sc.norm_tmp, sc.head_norm, null, 1, H, eps, .{ .round = true });
        try k.gemm.run(s, sc.head_norm, self.lm_head, sc.lm_logits, 1, self.text.vocab_size, H, .f32);
        try s.synchronize();
        try cuda.memory.download(self.d, sc.lm_logits, std.mem.sliceAsBytes(logits_out));
    }
};
