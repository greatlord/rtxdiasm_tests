#!/usr/bin/env bash
# Run the ACuda speed-test suite and regenerate docs/speed-test.md.
#
# Usage: tests/bench/speed-test.sh [build-dir]
#
# build-dir defaults to $ACUDA_BUILD_DIR or build/release. Benchmarks are
# read from bin/<config>/ (the repo's standard output layout), so the
# document always reflects the binaries from the given build.
#
# Hardware dot-product paths run with ACUDA_NO_VALIDATION=1 because the
# installed validation layer predates SPV_VALVE_mixed_float_dot_product.
# The validation-on ctest run below covers the portable fallback paths.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
# Prefer the NVIDIA Vulkan ICD on native Ubuntu; WSL has no Linux NVIDIA
# Vulkan ICD and therefore falls back to the default loader (llvmpipe).
source "$ROOT/tools/select_nvidia_vulkan.sh" 2>/dev/null || true
BUILD_DIR=${1:-${ACUDA_BUILD_DIR:-$ROOT/build/release}}
OUT=$ROOT/docs/speed-test.md

if [ -f "$BUILD_DIR/CMakeCache.txt" ]; then
  CONFIG=$(sed -n 's/^CMAKE_BUILD_TYPE:STRING=//p' "$BUILD_DIR/CMakeCache.txt")
else
  CONFIG=$(basename "$BUILD_DIR")
fi
case "$CONFIG" in
  Debug) BIN=${ACUDA_BENCH_BIN:-$ROOT/bin/debug} ;;
  *)     BIN=${ACUDA_BENCH_BIN:-$ROOT/bin/release} ;;
esac

for tool in bench_math bench_gemm bench_dp4a bench_atomic; do
  if [ ! -x "$BIN/$tool" ]; then
    echo "speed-test: missing $BIN/$tool (build first)" >&2
    exit 1
  fi
done

# Some environments ship a broken `ctest` wrapper early on PATH; use the
# first invocation that actually works. ACUDA_CTEST overrides the choice.
CTEST=${ACUDA_CTEST:-}
if [ -z "$CTEST" ]; then
  if ctest --version >/dev/null 2>&1; then
    CTEST=ctest
  elif [ -x /usr/bin/ctest ] && /usr/bin/ctest --version >/dev/null 2>&1; then
    CTEST=/usr/bin/ctest
  else
    echo "speed-test: no working ctest found (set ACUDA_CTEST)" >&2
    exit 1
  fi
fi

WORK=$(mktemp -d "${TMPDIR:-/tmp}/acuda-speed-test.XXXXXX")
trap 'rm -rf "$WORK"' EXIT

log() { printf '\n== %s ==\n' "$*"; }

# ------------------------------------------------------------------ host
DEVICE="unknown device"
DRIVER="unknown driver"
API="unknown"
if command -v vulkaninfo >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
  INFO=$(vulkaninfo --summary 2>/dev/null | python3 -c '
import re, sys
text = sys.stdin.read()
for block in re.split(r"\n(?=GPU\d+:\s*\n)", text):
    if "PHYSICAL_DEVICE_TYPE_DISCRETE_GPU" not in block:
        continue
    def field(key):
        m = re.search(r"^\s*" + key + r"\s*=\s*(.*)$", block, re.M)
        return m.group(1).strip() if m else "?"
    print("|".join([field("deviceName"), field("driverInfo"),
                    field("apiVersion")]))
    break
' || true)
  if [ -n "${INFO:-}" ]; then
    IFS='|' read -r DEVICE DRIVER API <<<"$INFO"
  fi
fi

COMMIT=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)
# The document itself is generated output, so it does not make the tree
# "dirty"; any other tracked change does.
if [ -n "$(git -C "$ROOT" status --porcelain --untracked-files=no -- . \
  ':(exclude)docs/speed-test.md' 2>/dev/null)" ]; then
  COMMIT="$COMMIT-dirty"
fi
DATE=$(date -u '+%Y-%m-%d %H:%M UTC')

# ------------------------------------------------------------- test suite
log "ctest (validation on)"
if ! "$CTEST" --test-dir "$BUILD_DIR" >"$WORK/ctest_on.log" 2>&1; then
  cat "$WORK/ctest_on.log" >&2
  exit 1
fi
CT_ON=$(sed -n 's/.*tests failed out of \([0-9][0-9]*\).*/\1/p' \
  "$WORK/ctest_on.log" | tail -1)

log "ctest (validation off)"
if ! ACUDA_NO_VALIDATION=1 "$CTEST" --test-dir "$BUILD_DIR" \
  >"$WORK/ctest_off.log" 2>&1; then
  cat "$WORK/ctest_off.log" >&2
  exit 1
fi
CT_OFF=$(sed -n 's/.*tests failed out of \([0-9][0-9]*\).*/\1/p' \
  "$WORK/ctest_off.log" | tail -1)

# ---------------------------------------------------------------- helpers
# Last GFLOP/s value in a log ("<mode> min ... GFLOP/s").
last_gflops() {
  awk '{ for (i = 1; i <= NF; i++) if ($i == "GFLOP/s") v = $(i - 1) }
       END { print v }' "$1"
}
# GFLOP/s for one bench_math mode in a multi-mode log.
math_gflops() {
  awk -v m="$1" '$1 == m { for (i = 1; i <= NF; i++)
                              if ($i == "GFLOP/s") v = $(i - 1) }
                  END { print v }' "$2"
}
gemm_tile() {
  sed -n 's/^gemm .*\[\([^]]*\)\].*/\1/p' "$1" | tail -1
}
dp4a_rate() {
  awk -v want="$1" '$1 == want {
        for (i = 1; i <= NF; i++)
          if ($i == "Gdp4a/s") { v = $(i - 1); gsub(/[^0-9.]/, "", v) }
      }
      END { print v }' "$2"
}
dp4a_speedup() {
  awk '/^min speedup:/ { gsub(/x/, "", $3); print $3; exit }' "$1"
}
# Round to a comma-grouped integer for the tables.
fmt_int() {
  awk -v v="$1" 'BEGIN {
    if (v == "" || v + 0 == 0 && v != "0") { print "n/a"; exit }
    s = sprintf("%.0f", v)
    out = ""
    while (length(s) > 3) {
      out = "," substr(s, length(s) - 2, 3) out
      s = substr(s, 1, length(s) - 3)
    }
    print s out
  }'
}
# One-decimal rate for the shared-atomic table (sub-1 G/s fallback).
fmt_rate() {
  awk -v v="$1" 'BEGIN {
    if (v == "") { print "n/a"; exit }
    printf "%.1f", v
  }'
}
# Hardware speedup over the portable/fallback path.
fmt_ratio() {
  awk -v a="$1" -v b="$2" 'BEGIN {
    if (a == "" || b == "" || b + 0 == 0) { print "n/a"; exit }
    printf "%.1f", a / b
  }'
}

# ------------------------------------------------------------ math bench
log "bench_math (all modes)"
ACUDA_NO_VALIDATION=1 "$BIN/bench_math" all 20 512 1024 128 \
  >"$WORK/math.log"

log "bench_math fallbacks"
ACUDA_NO_VALIDATION=1 ACUDA_PTX_NO_FMA=1 "$BIN/bench_math" f32 \
  20 512 1024 128 >"$WORK/math_f32_sw.log"
ACUDA_NO_VALIDATION=1 ACUDA_F16_EMULATED=1 "$BIN/bench_math" f16 \
  20 512 1024 128 >"$WORK/math_f16_sw.log"
ACUDA_NO_VALIDATION=1 ACUDA_NO_FDOT2=1 "$BIN/bench_math" f16mma \
  20 512 1024 128 >"$WORK/math_f16mma_sw.log"
ACUDA_NO_VALIDATION=1 ACUDA_NO_FDOT2=1 "$BIN/bench_math" sparse \
  20 512 1024 128 >"$WORK/math_sparse_sw.log"
ACUDA_NO_VALIDATION=1 ACUDA_PTX_NO_DOT_PRODUCT=1 "$BIN/bench_math" i8 \
  20 512 1024 128 >"$WORK/math_i8_sw.log"

M_F32=$(math_gflops f32 "$WORK/math.log")
M_F16=$(math_gflops f16 "$WORK/math.log")
M_F16MMA=$(math_gflops f16mma "$WORK/math.log")
M_BF16=$(math_gflops bf16 "$WORK/math.log")
M_SPARSE=$(math_gflops sparse "$WORK/math.log")
M_I8=$(math_gflops i8 "$WORK/math.log")

F_F32=$(last_gflops "$WORK/math_f32_sw.log")
F_F16=$(last_gflops "$WORK/math_f16_sw.log")
F_F16MMA=$(last_gflops "$WORK/math_f16mma_sw.log")
F_SPARSE=$(last_gflops "$WORK/math_sparse_sw.log")
F_I8=$(last_gflops "$WORK/math_i8_sw.log")

# ------------------------------------------------------------ gemm bench
log "bench_gemm (autotuned)"
for tag in 512 1024 2048 4096; do
  case "$tag" in
    512)  args="512 512 2048" ;;
    1024) args="1024 1024 2048" ;;
    2048) args="2048 2048 2048" ;;
    4096) args="4096 4096 2048" ;;
  esac
  ACUDA_NO_VALIDATION=1 "$BIN/bench_gemm" 20 $args \
    >"$WORK/gemm_$tag.log"
done

log "bench_gemm fallback and forced tiles"
ACUDA_NO_VALIDATION=1 ACUDA_NO_FDOT2=1 "$BIN/bench_gemm" 20 2048 2048 2048 \
  >"$WORK/gemm_fb.log"
for t in 64x64 64x128 128x64; do
  ACUDA_NO_VALIDATION=1 ACUDA_GEMM_TILE=$t "$BIN/bench_gemm" \
    20 2048 2048 2048 >"$WORK/gemm_tile_$t.log"
done

G_512=$(last_gflops "$WORK/gemm_512.log")
G_1024=$(last_gflops "$WORK/gemm_1024.log")
G_2048=$(last_gflops "$WORK/gemm_2048.log")
G_4096=$(last_gflops "$WORK/gemm_4096.log")
T_512=$(gemm_tile "$WORK/gemm_512.log")
T_1024=$(gemm_tile "$WORK/gemm_1024.log")
T_2048=$(gemm_tile "$WORK/gemm_2048.log")
T_4096=$(gemm_tile "$WORK/gemm_4096.log")

G_FB=$(last_gflops "$WORK/gemm_fb.log")
T_6464=$(last_gflops "$WORK/gemm_tile_64x64.log")
T_64128=$(last_gflops "$WORK/gemm_tile_64x128.log")
T_12864=$(last_gflops "$WORK/gemm_tile_128x64.log")

# ------------------------------------------------------------- dp4a bench
log "bench_dp4a"
ACUDA_NO_VALIDATION=1 "$BIN/bench_dp4a" 20 1024 64 256 1 \
  >"$WORK/dp4a_1.log"
ACUDA_NO_VALIDATION=1 "$BIN/bench_dp4a" 20 1024 64 256 4 \
  >"$WORK/dp4a_4.log"
ACUDA_NO_VALIDATION=1 ACUDA_FORCE_SERIAL=1 "$BIN/bench_dp4a" \
  20 1024 64 256 4 >"$WORK/dp4a_4s.log"

D_N1=$(dp4a_rate native "$WORK/dp4a_1.log")
D_F1=$(dp4a_rate fallback "$WORK/dp4a_1.log")
S_N1=$(dp4a_speedup "$WORK/dp4a_1.log")
D_N4=$(dp4a_rate native "$WORK/dp4a_4.log")
D_F4=$(dp4a_rate fallback "$WORK/dp4a_4.log")
S_N4=$(dp4a_speedup "$WORK/dp4a_4.log")
D_NS=$(dp4a_rate native "$WORK/dp4a_4s.log")
D_FS=$(dp4a_rate fallback "$WORK/dp4a_4s.log")
S_NS=$(dp4a_speedup "$WORK/dp4a_4s.log")

# --------------------------------------------------------- atomic bench
log "bench_atomic (shared f32 add)"
ACUDA_NO_VALIDATION=1 "$BIN/bench_atomic" 20 1024 64 256 \
  >"$WORK/atomic.log"
AT_NMIN=$(awk '$1 == "native" { print $3 }' "$WORK/atomic.log")
AT_FMIN=$(awk '$1 == "fallback" { print $3 }' "$WORK/atomic.log")
AT_NRATE=$(awk '$1 == "native" {
    for (i = 1; i <= NF; i++)
      if ($i == "Gatom/s") { v = $(i - 1); gsub(/[^0-9.]/, "", v) }
  } END { print v }' "$WORK/atomic.log")
AT_FRATE=$(awk '$1 == "fallback" {
    for (i = 1; i <= NF; i++)
      if ($i == "Gatom/s") { v = $(i - 1); gsub(/[^0-9.]/, "", v) }
  } END { print v }' "$WORK/atomic.log")
AT_SPEED=$(awk '/^min speedup/ { gsub(/x/, "", $3); print $3 }' \
  "$WORK/atomic.log")

# ------------------------------------------------------------- write doc
log "writing $OUT"
cat >"$OUT" <<EOF
# Speed test results

Generated by \`tests/bench/speed-test.sh\` on ${DATE}.
Do not edit by hand: rerun the script to refresh every number below.

Measured on ${DEVICE} (${DRIVER}), Vulkan ${API} / SPIR-V 1.6,
with ACuda at commit \`${COMMIT}\`. All numbers are the minimum of
several rounds and include only GPU work on the timed path; the host
only records and submits launches.

Run the whole suite (this rewrites the document):

\`\`\`sh
cmake --build build/release -j8
tests/bench/speed-test.sh
\`\`\`

or drive the individual steps:

\`\`\`sh
ctest --test-dir build/release
ACUDA_NO_VALIDATION=1 ./bin/release/bench_math all 20 512 1024 128
ACUDA_NO_VALIDATION=1 ./bin/release/bench_gemm 20 2048 2048 2048
ACUDA_NO_VALIDATION=1 ./bin/release/bench_dp4a 20 1024 64 256 1
\`\`\`

Validation-off runs are used for the hardware dot paths because the
installed validation layer predates \`SPV_VALVE_mixed_float_dot_product\`.
The fallback paths are exercised by the validation-on test suite.

## Correctness / regression tests

| Suite | Result |
|---|---|
| \`ctest\`, validation on | **${CT_ON}/${CT_ON} passed** |
| \`ctest\`, validation off | **${CT_OFF}/${CT_OFF} passed** |

## \`bench_math\` — 20 launches, 512 CTAs, 1024 reps, 128 threads

| Mode | GFLOP/s | Notes |
|---|---|---|
| f32 FMA | **$(fmt_int "$M_F32")** | hardware fused FMA (\`GLSL.std.450 Fma\`) |
| f16x2 FMA | **$(fmt_int "$M_F16")** | native \`Float16\` |
| dense FP16 MMA | **$(fmt_int "$M_F16MMA")** | \`OpFDot2MixAcc32VALVE\` |
| dense BF16 MMA | **$(fmt_int "$M_BF16")** | shift-widen + f32 FMA |
| sparse FP16 MMA | **$(fmt_int "$M_SPARSE")** | fdot2, 2:4 structured A |
| dense int8 MMA | **$(fmt_int "$M_I8")** | packed \`OpSDot\` |

The benchmark kernels are written to resist driver reassociation and
CSE (dependent scalar chains, four MMAs on distinct A/B pairs). See
\`tests/bench/README.md\`.

## \`bench_gemm\` — tiled \`C = A * Bt^T\`, f16 in, f32 accumulate

The benchmark compiles three tile shapes (64x64/256, 64x128/512,
128x64/512), ramps clocks, times each, and uses the fastest.
\`ACUDA_GEMM_TILE\` forces a shape.

| M x N x K | Winner | GFLOP/s |
|---|---|---|
| 512 x 512 x 2048 | ${T_512} | $(fmt_int "$G_512") |
| 1024 x 1024 x 2048 | ${T_1024} | **$(fmt_int "$G_1024")** |
| 2048 x 2048 x 2048 | ${T_2048} | **$(fmt_int "$G_2048")** |
| 4096 x 4096 x 2048 | ${T_4096} | **$(fmt_int "$G_4096")** |

Forced tile at 2048^3 (to show the autotuner matters):

| Tile | GFLOP/s |
|---|---|
| 64x64 | $(fmt_int "$T_6464") |
| 64x128 | $(fmt_int "$T_64128") |
| 128x64 | $(fmt_int "$T_12864") |

The selected output tile is checked against the CPU expectation before
timing (all inputs 1.0 gives \`C = K\`). The GEMM
uses two 32-wide shared-memory K-step buffers with prefetch-into-
registers double buffering, 20-word padded shared rows, and the
\`acuda.dot2.f32.f16\` intrinsic. Buffers are device-local
(\`acuda_malloc_device\`); host transfers go through a staging copy.

## \`bench_dp4a\` — 1024 CTAs, 64 reps, 256 threads

| Configuration | Native | Fallback | Speedup |
|---|---|---|---|
| nbuf=1 | $(fmt_int "$D_N1") Gdp4a/s | $(fmt_int "$D_F1") | ${S_N1}x |
| nbuf=4 (submission overlap) | $(fmt_int "$D_N4") | $(fmt_int "$D_F4") | ${S_N4}x |
| nbuf=4, \`ACUDA_FORCE_SERIAL=1\` | $(fmt_int "$D_NS") | $(fmt_int "$D_FS") | ${S_NS}x |

## \`bench_atomic\` — shared-memory f32 atomic add

1024 CTAs, 256 threads, 64 atomics per thread, with every lane
contending on one f32 accumulator per CTA (the reduction idiom).
\`OpAtomicFAddEXT\` is used only when the device enables
\`shaderSharedFloat32AtomicAdd\` and the module's shared memory is
entirely scalar f32; mixed shared types keep the CAS loop.

| Path | min ms | Logical Gatom/s |
|---|---|---|
| native \`OpAtomicFAddEXT\` | ${AT_NMIN} | $(fmt_rate "$AT_NRATE") |
| CAS fallback (forced lowering) | ${AT_FMIN} | $(fmt_rate "$AT_FRATE") |

Min speedup: **${AT_SPEED}x**. RADV coalesces a uniform per-wave shared
atomic (confirmed in the ISA: one \`ds_add_f32\` per active wave, not
per lane), so the native figure includes that effect; the CAS fallback
cannot be coalesced in the same way.

## Hardware path vs portable fallback

| Path | Portable path | Hardware path | Speedup |
|---|---|---|---|
| f32 FMA (\`ACUDA_PTX_NO_FMA=1\`) | $(fmt_int "$F_F32") | $(fmt_int "$M_F32") | ~$(fmt_ratio "$M_F32" "$F_F32")x |
| f16x2 FMA (\`ACUDA_F16_EMULATED=1\`) | $(fmt_int "$F_F16") | $(fmt_int "$M_F16") | ~$(fmt_ratio "$M_F16" "$F_F16")x |
| dense FP16 MMA (\`ACUDA_NO_FDOT2=1\`) | $(fmt_int "$F_F16MMA") | $(fmt_int "$M_F16MMA") | ~$(fmt_ratio "$M_F16MMA" "$F_F16MMA")x |
| sparse FP16 MMA (\`ACUDA_NO_FDOT2=1\`) | $(fmt_int "$F_SPARSE") | $(fmt_int "$M_SPARSE") | ~$(fmt_ratio "$M_SPARSE" "$F_SPARSE")x |
| dense int8 MMA (\`ACUDA_PTX_NO_DOT_PRODUCT=1\`) | $(fmt_int "$F_I8") | $(fmt_int "$M_I8") | ~$(fmt_ratio "$M_I8" "$F_I8")x |
| tiled GEMM 2048^3 (\`ACUDA_NO_FDOT2=1\`) | $(fmt_int "$G_FB") | $(fmt_int "$G_2048") | ~$(fmt_ratio "$G_2048" "$G_FB")x |
| dp4a (nbuf=1) | $(fmt_int "$D_F1") | $(fmt_int "$D_N1") | ~${S_N1}x |
| shared f32 atomics | $(fmt_rate "$AT_FRATE") | $(fmt_rate "$AT_NRATE") | ~${AT_SPEED}x |

## Notes and limitations

- The experimental persistent-grid GEMM kernel is not included: it
  reaches the structured nested-loop lowering but hits an unresolved
  barrier-related fault (removing barriers makes it run, and a minimal
  nested kernel with barriers works). The shipped per-tile kernel is
  used instead.
- Absolute FLOP/s figures count the operations the kernels actually
  execute. Earlier versions of the benchmark were inflated by driver
  reassociation/CSE; those kernels were rewritten (see
  \`tests/bench/README.md\`).
- Native integer dot products are selected per packed 4x8-bit form
  from \`VkPhysicalDeviceShaderIntegerDotProductProperties\`; mixed
  signedness is not accelerated on this GPU and uses the scalar
  expansion.
- Native float atomics need the matching
  \`VK_EXT_shader_atomic_float\` feature. This device enables only
  \`shaderSharedFloat32AtomicAdd\` (and it additionally needs a module
  whose shared memory is entirely scalar f32, since the word-array
  representation cannot be retyped per access). Buffer f32/f64, shared
  f64, and image f32 add paths are implemented but feature-disabled
  here; they are compile-time validated, not benchmarked. Image
  \`sured.p.min/max.f32\` is enabled here
  (\`shaderImageFloat32AtomicMinMax\`) and runtime-tested by
  \`test_ptx\`, but is not part of this throughput table.
- BF16 has no hardware dot product on this device
  (\`shaderBFloat16DotProduct = 0\` and
  \`shaderMixedFloatDotProductBFloat16Acc = 0\`), so it stays on the
  shift-widen + f32 FMA path.
- Replacing host-visible GEMM buffers with device-local ones was worth
  about 18x; with \`acuda_malloc\` (host RAM) the same kernel drops below
  1 TFLOP/s because every tile load crosses PCIe.
EOF

printf '\nspeed-test: wrote %s\n' "$OUT"
printf '  device:  %s (%s), Vulkan %s\n' "$DEVICE" "$DRIVER" "$API"
printf '  commit:  %s\n' "$COMMIT"
printf '  math:    f32 %s, f16 %s, f16mma %s, bf16 %s, sparse %s, i8 %s GFLOP/s\n' \
  "$(fmt_int "$M_F32")" "$(fmt_int "$M_F16")" "$(fmt_int "$M_F16MMA")" \
  "$(fmt_int "$M_BF16")" "$(fmt_int "$M_SPARSE")" "$(fmt_int "$M_I8")"
printf '  gemm:    %s / %s / %s / %s GFLOP/s\n' \
  "$(fmt_int "$G_512")" "$(fmt_int "$G_1024")" \
  "$(fmt_int "$G_2048")" "$(fmt_int "$G_4096")"
printf '  dp4a:    %s vs %s Gdp4a/s (%sx)\n' \
  "$(fmt_int "$D_N1")" "$(fmt_int "$D_F1")" "$S_N1"
