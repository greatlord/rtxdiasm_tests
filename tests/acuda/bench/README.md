# Full speed-test run

`tests/bench/speed-test.sh` runs the complete suite — ctest with
validation on and off, every `bench_math` mode plus its portable
fallback, the tiled-GEMM autotuner for four problem sizes (plus the
`ACUDA_NO_FDOT2=1` fallback and the forced tiles), and the three
`bench_dp4a` configurations — then rewrites `docs/speed-test.md` with
the current numbers, device, and commit. The document is generated
output: do not edit it by hand.

```sh
# defaults to build/release; pass another build dir as the argument
./tests/bench/speed-test.sh
./tests/bench/speed-test.sh build/debug

# or through CMake (builds the benchmarks first)
cmake --build build/release --target speed-test
```

Hardware dot paths run with `ACUDA_NO_VALIDATION=1` because the
installed validation layer predates
`SPV_VALVE_mixed_float_dot_product`; the validation-on ctest run still
covers the fallback paths. `ACUDA_BENCH_BIN` overrides where the
benchmark binaries are read from.

# dp4a native-vs-expansion benchmark

`bench_dp4a` measures the optional integer dot-product path
(`OpSDot`/`OpUDot`/`OpSUDot` from `SPV_KHR_integer_dot_product`, enabled
when the device reports `shaderIntegerDotProduct`) against the
portable lane-extend/multiply/add expansion.

It is not a CTest: the numbers are informational and machine-dependent.
The native path is selected per packed 4x8-bit form from
`VkPhysicalDeviceShaderIntegerDotProductProperties`; on the RX 6800 XT
the signed and unsigned packed forms are accelerated but the
mixed-signedness form is not, so mixed `dp4a`/`mma` forms use the exact
scalar expansion. `ACUDA_PTX_NO_DOT_ACCEL=1` ignores the properties for
A/B testing.

```sh
# default: 20 launches per round, 1024 CTAs, 64 loop iterations, 256 threads
./bin/release/bench_dp4a

# explicit: launches, grid, reps, block
./bin/release/bench_dp4a 20 2048 64 256

# independent-buffer throughput (8 buffers, round-robined)
./bin/release/bench_dp4a 20 2048 64 256 8
# compare against the old fully serialized submission path
ACUDA_FORCE_SERIAL=1 ./bin/release/bench_dp4a 20 2048 64 256 8
# override the default 32-command batch size
ACUDA_BATCH=8 ./bin/release/bench_dp4a 20 2048 64 256 1
# force 32-lane subgroups for this barrier-free kernel (native ~2-3%
# faster on RADV; the no-dot expansion regresses, hence opt-in)
ACUDA_SUBGROUP_SIZE=32 ./bin/release/bench_dp4a 20 2048 64 256 1
```

`nbuf > 1` gives each consecutive launch a distinct buffer, so the
measured effect is submission overlap rather than a data dependency.
The tool prints both total and CPU-side enqueue time per launch.

The kernel (`dp4a_bench.ptx`) performs

```
32 dp4a + 32 dependent adds per loop iteration
```

per thread, with four independent accumulator chains. `reps` is a push
constant (>= 1), so the same binary covers both regimes:

- `reps=1` is dominated by launch/scheduling overhead.
- `reps=64` (2048 dp4a/thread) is compute-bound.

The tool compiles the kernel twice (native and fallback), warms both
pipelines, then times seven alternating rounds and reports the minimum
and median. It also prints the two SPIR-V function-body instruction
counts. `ACUDA_PTX_NO_DOT_PRODUCT=1` makes the runtime itself disable
the extension for an end-to-end fallback run.

## Reference results

RX 6800 XT / RADV Mesa 26.2.3 (RDNA2, maps the opcode to
`v_dot4_u32_u8`):

| Regime | Native | Fallback | Speedup |
|---|---|---|---|
| compute-bound (`reps=64`, grid 2048, 1 buffer) | 0.223 ms, 4819 Gdp4a/s | 0.672 ms, 1597 Gdp4a/s | **3.02x** |
| launch-bound (`reps=1`, grid 2048, 1 buffer) | 0.003 ms | 0.004 ms | 1.4x |
| compute-bound, 8 independent buffers | 0.223 ms | 0.672 ms | **3.02x** |
| compute-bound, 8 buffers, `ACUDA_FORCE_SERIAL=1` | 0.222 ms | 0.679 ms | 3.05x |

The launch-bound figure is with the default 32-command batch. Before
command/descriptor-pool recycling, overlap, batching, bind-state
skipping, and descriptor-set reuse, the same measurement was ~0.33 ms
per launch. `ACUDA_BATCH=1` restores one-command submissions
(~0.02–0.03 ms); `ACUDA_PROFILE_LAUNCH=1` reports recording and
submission time separately.

Static counts (same kernel):

| | Native | Fallback |
|---|---|---|
| SPIR-V function body | 169 instructions (32 dot) | 809 instructions |
| RADV ISA (`RADV_DEBUG=shaderstats`) | 101 instructions | 568 instructions |

On a device without the extension, or with
`ACUDA_PTX_NO_DOT_PRODUCT=1`, the tool prints that the native path is
unavailable and exits without benchmarking.

# Math / AI throughput benchmark

`bench_math` covers six formulations:

```sh
./bin/release/bench_math all 20 512 1024 128
# mode: all|f32|f16|f16mma|bf16|sparse|i8
./bin/release/bench_math f16mma 20 512 1024 128
```

- `f32` — 16 independent dependent `fma.rn.f32` chains
  (`c = c*a + b`).
- `f16` — 16 independent dependent `fma.rn.f16x2` chains.
- `f16mma` — dense `mma.sync.m16n8k16.f32.f16.f16.f32`, two accumulator
  chains, four MMAs per loop on distinct A/B pairs.
- `bf16` — dense `mma.sync.m16n8k16.f32.bf16.bf16.f32`, same structure.
- `sparse` — `mma.sp::ordered_metadata.m16n8k32.f32.f16.f16.f32`, same
  structure.
- `i8` — dense `mma.sync.m16n8k32.s32.s8.s8.s32`, same structure.

The tensor modes require `block % 32 == 0`. Results on RX 6800 XT /
RADV Mesa 26.2.3, `all 20 512 1024 128`, `ACUDA_NO_VALIDATION=1` (the
validation layer predates the mixed-float-dot extension):

| Mode | GFLOP/s |
|---|---|
| f32 | ~14,000 |
| f16x2 (native, probe) | ~26,000 |
| f16x2 (`ACUDA_F16_EMULATED=1`) | ~280 |
| dense FP16 MMA (`OpFDot2MixAcc32VALVE`) | ~23,900 |
| dense FP16 MMA (`ACUDA_NO_FDOT2=1`) | ~3,750 |
| dense BF16 MMA | ~11,400 |
| sparse FP16 MMA (`OpFDot2MixAcc32VALVE`) | ~14,700 |
| sparse FP16 MMA (`ACUDA_NO_FDOT2=1`) | ~2,230 |
| dense int8 MMA (`OpSDot`) | ~48,100 |
| dense int8 MMA (`ACUDA_PTX_NO_DOT_PRODUCT=1`) | ~3,630 |

With `reps=4096` the numbers settle slightly higher: f32 ~16,300,
f16x2 ~31,100, dense FP16 MMA ~23,600, sparse ~14,700, i8 ~47,700.
These are the *executed* rates: an earlier version of this benchmark
used repeated `c += a*b` chains and four identical MMAs, which RADV
rewrote into scaled adds and CSE-ed dot products, respectively,
inflating the reported f32 rate above the 20.7 TFLOP/s hardware peak.
The current kernels are reassociation/CSE resistant; verify with
`ACUDA_DUMP_ISA` that the expected instruction count is present.

# Tiled GEMM benchmark

`bench_gemm` runs `C[M][N] = A[M][K] * Bt[N][K]^T` with f16 inputs and
f32 accumulation using the `acuda.dot2.f32.f16` intrinsic. Block tile
Three shapes are compiled and timed once at startup (after a 50-launch
clock ramp): 64x64/256 threads, 64x128/512 and 128x64/512, all with
16-accumulator warp tiles. The fastest is used for the reported number;
`ACUDA_GEMM_TILE=64x64|64x128|128x64` forces one. Typical winners:
64x128 at 1024^2, 64x64 at 2048^3 and above. A and Bt
tiles are staged in shared memory with two K-step buffers: the next
16-wide K slice is prefetched into registers while the current slice is
used, then written to the other buffer. Shared rows are padded to 12
words (instead of 8) to reduce LDS bank conflicts. Each lane loads exactly the
f16x2 words its dot products need, so there are no `mma.sync` fragment
shuffles. `K` must be a multiple of 32, `N` of 64, `M` of 128.

```sh
./bin/release/bench_gemm 20 2048 2048 2048
```

A 128x64 output tile is checked against the CPU expectation first (all
inputs 1.0 gives `C = K`). Results on RX 6800 XT / RADV,
`ACUDA_NO_VALIDATION=1`:

| M x N x K | GFLOP/s |
|---|---|
| 1024 x 1024 x 2048 | ~16,270 (64x128) |
| 2048 x 2048 x 2048 | ~21,220 (64x64) |
| 4096 x 4096 x 2048 | ~23,970 (64x64) |

Split-K was evaluated and is deliberately not shipped. Partitioning K
into 2 or 4 chunks (partial-C buffer plus a reduction kernel, correct
prologue offsets) never beat the unsplit kernel: at 128^2/256^2/512^2
and at tall-K small-output shapes (128x128 up to K=16384) the split
autotune selected `split1`. On this GPU small shapes are limited by
per-CTA memory latency and launch overhead, not by the number of CTAs,
so the extra partial-C writes and the reduction launch cost more than
the added parallelism. The test above also only exercises `C = K`, which
cannot detect a wrong K range; any future split kernel needs a
non-uniform A/B check.

`ACUDA_NO_FDOT2=1` (widen + FMA fallback) gives ~3,900 GFLOP/s at
2048^3, so the intrinsic is ~5.4x faster end to end. The benchmark
allocates its buffers with `acuda_malloc_device` (VRAM); with
host-visible `acuda_malloc` the same kernel drops below 1 TFLOP/s
because every tile load crosses PCIe.

Important context:

- f32 uses hardware fused FMA (`GLSL.std.450 Fma`) after the
  `OpFmaKHR` probe fails on RADV. `ACUDA_PTX_NO_FMA=1` forces the
  integer software path for a same-binary A/B: **~78 GFLOP/s** versus
  **~14,000 GFLOP/s** on this machine.
- f16 uses native `Float16` FMA when `shaderFloat16` is available and
  the runtime probe measures it faster (it does on RADV: ~26,000
  versus ~280 GFLOP/s emulated). F16→F32 widening also uses native conversion; f16
  subnormals are flushed to match the established emulated semantics.
  Without that path, packed halves are unpacked to f32, computed, and
  packed again; `ACUDA_F16_EMULATED=1` forces the emulated comparison.
- The MMA paths are the runtime's shuffle-gathered software lowering,
  not native tensor cores. Dense f16/bf16 `m16n8k16` processes both
  halves of each 32-bit fragment register per shuffle and blocks all
  four accumulators; BF16→F32 widening is a single shift. Sparse
  `m16n8k32` shares B fragments across the four output rows, packs each
  pair of dense K positions into one 32-bit word, and uses
  `OpFDot2MixAcc32VALVE` on the same f16 pairs. Both selector patterns
  (`{0,2}/{1,3}` and `{0,3}/{1,2}`) are covered by `test_ptx`.
- The mixed-float-dot path requires
  `VK_VALVE_shader_mixed_float_dot_product` and the f16→f32 feature.
  When validation is enabled it is disabled by default because the
  installed layer predates the extension; `ACUDA_FDOT2=1` forces it and
  `ACUDA_NO_FDOT2=1` disables it.
- Int8 K-groups are combined into packed `OpSDot`/`OpUDot`/`OpSUDot`
  operations, ~13x faster than the scalar IMul/IAdd chain
  (`ACUDA_PTX_NO_DOT_PRODUCT=1` disables it). Vulkan 1.3 has no core
  matrix instruction. Newer RADV builds (RDNA3+) expose
  `VK_KHR_cooperative_matrix`; the RDNA2 device used for these results
  exposes no cooperative-matrix extension at all (KHR, NV, and NV2 are
  all absent). Even where present, PTX `mma.sync` takes register
  fragments in a lane-distributed layout, which does not map directly
  to `OpCooperativeMatrixLoadKHR` without a data rearrangement through
  memory.

# Shared-memory f32 atomic add benchmark

`bench_atomic` compares the native `OpAtomicFAddEXT` lowering with the
u32-word CAS loop for `red.shared.add.f32` / `atom.shared.add.f32`.
The native path is selected only when the device enables
`VK_EXT_shader_atomic_float`'s `shaderSharedFloat32AtomicAdd` **and**
every shared access in the kernel is scalar f32 (the compiler declares
those shared variables as float instead of u32 words); mixed shared
types keep the CAS loop.

```sh
# launches, grid, reps per thread, block
./bin/release/bench_atomic 20 1024 64 256
```

Each CTA has one f32 accumulator and every lane hammers it, which is
the contended reduction shape. The kernel is checked against
`block * reps` before timing; the tool then times native and CAS
kernels in alternating rounds. Results on RX 6800 XT / RADV,
`ACUDA_NO_VALIDATION=1`:

| Path | min | median | Logical atomics/s at min |
|---|---|---|---|
| native `OpAtomicFAddEXT` | ~0.015 ms | ~0.023 ms | ~1,100 G/s |
| CAS fallback | ~18.5 ms | ~18.5 ms | ~0.9 G/s |

The speedup is ~790–1250x on this all-lanes-one-slot shape, but the
figure counts logical lane atomics: RADV coalesces a uniform per-wave
shared atomic into one `ds_add_f32` with the active-lane count (visible
in `ACUDA_DUMP_ISA`), while the CAS fallback cannot be coalesced that
way. The gap shrinks when lanes target distinct slots. `grep
ds_add_f32` in the dumped ISA to confirm the native instruction is
present.

The CAS path is also available end-to-end with
`ACUDA_PTX_NO_ATOMIC_FLOAT=1` (or the older
`ACUDA_PTX_NO_ATOMIC_F32_ADD=1`), which disables every native float
atomic path.

Buffer (BDA) f32/f64, shared f64, and image `sured.p.add.f32` lowerings
are also implemented, but the test GPU enables none of those features
(`shaderBufferFloat32/64AtomicAdd = 0`, `shaderSharedFloat64AtomicAdd =
0`, `shaderImageFloat32AtomicAdd = 0`). They are covered at compile
time by `test_ptx` and validate as SPIR-V; only the shared f32 path is
runtime-tested and benchmarked here.

Image `sured.p.min/max.f32` uses `OpAtomicFMinEXT`/`OpAtomicFMaxEXT`
from `SPV_EXT_shader_atomic_float_min_max`; this GPU enables
`shaderImageFloat32AtomicMinMax`, so that path is runtime-tested
(`sured_minmax_f32.ptx`, R32f surface) but not benchmarked.
