# Hy4-flash CUDA port (GB10) — what this fork adds on top of TensorFold v1.3.2

Upstream serves Tencent Hy4-flash (`hy_v4_flash`) **on Metal only**. This fork adds a full CUDA
family for it, and it serves on NVIDIA GB10 (DGX Spark). **Status: serving end-to-end, quality
gate NOT cleared yet** — decode enters a single-token loop after ~30 tokens (evidence below).
For quality-cleared serving today use the vLLM overlay
[`styles01/hy4-exl3-vllm-serving`](https://github.com/styles01/hy4-exl3-vllm-serving) (16.1 tok/s
coherent); full context + runbook in
[`styles01/sparkrun-recipes` → runbooks/hy4-flash-exl3-tensorfold-cuda-port.md](https://github.com/styles01/sparkrun-recipes).

## Added vs v1.3.2

- `zig/src/families/hy4_flash/cuda_native.zig` — the `native` CUDA family: single-sequence
  lanes.Backend over the family's CUDA model (512-row prefill chunks, host-side sampling like
  the Metal serve path, 1-row verify windows, keep-rollback, UMA warm pass at open). Exports
  `model_type = "hy_v4_flash"`, `formats = {"exl3"}`.
- `zig/src/families/hy4_flash/cuda_family.zig`, `zig/build/hy4_cuda.zig`, wiring into
  `zig/build/cuda_engines.zig` / `cuda_kernels.zig` / `zig/src/cuda/kernels.zig`
  (hy4 engram fatbin), `zig/kernels/cuda/hy4_flash/engram.cu`.
- `zig/src/native/cuda_families.zig` — registry entry added.
- `zig/src/cli/info.zig` — EXL3 detection + Hy4-aware floor (engram tables priced as
  O_DIRECT disk-streamed, not resident).
- Ported + fixed: `families/hy4_flash/{cuda_model.zig (+deinit, +EXL3 config detection),
  engram_cuda.zig, exl3_weights.zig, moe_exl3_cuda.zig, bf16_rows.zig, mla_weights.zig,
  host_tests.zig}` from the `hy4-flash-cuda` branch lineage.

## Build + serve (GB10)

```bash
zig build -Dgpu=spark -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=ReleaseSafe   # zig 0.17, CUDA 13
zig-out/native/bin/tensorfold-native info <pack>    # reads format: exl3 (3.0-bit, mcg codebook), floor 63.57 GiB + 118 GiB engram streamed
zig-out/native/bin/tensorfold-native serve <pack> --port 8508   # loads 87.5 GiB in ~270 s, OpenAI-compatible /v1
```

Use a BF16-dense EXL3 pack (dq2-style) — the ported dense projection path assumes MLX
affine-4/q8 conventions and needs a real FP8-blockwise dense path added to `cuda_projection.zig`.

## Debug evidence for the remaining loop (for whoever continues)

- Early tokens correct ⇒ weights/kernels roughly right; loop at ~30 tokens at temperature 0 ⇒
  KV/position handling.
- Rotary table phases step correctly per row but carry a constant offset ⇒ positions handed to
  `mla_cuda` are stale/uninitialized for later rows — suspect the queue/read/verify/keep-rollback
  slot bookkeeping in `cuda_native.zig` (compare `serve_metal.zig`, which is correct upstream).
- `exl3_weights.zig` asserts `mul1` codebook; the tested 3.0-bit pack is MCG and works — note so
  the assert isn't "fixed" backwards.

Provenance: snapshotted from upstream v1.3.2 (Apache-2.0, LICENSE intact); all our additions
greppable by "hy4" in `zig/src`. Credits: Tencent Hy team (model, DFly2); @ashxhart (TensorFold);
the #4003 lineage (Hy4 CUDA modules + upload slabs); vcruz305/atakann (EXL3-on-GB10 recipe lineage).
