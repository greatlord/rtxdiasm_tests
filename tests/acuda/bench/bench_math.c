// Math/AI throughput benchmark: f32 FMA, f16x2 FMA, dense f16/BF16 MMA,
// sparse FP16 MMA, dense int8 MMA.
//
// The kernels are written to resist compiler reassociation and CSE:
//   - scalar modes use 16 dependent chains c = c*a + b (a plain
//     c += a*b sum is rewritten as a scaled add by RADV);
//   - tensor modes issue four MMAs on four distinct A/B pairs (repeated
//     identical MMAs are CSE-ed by RADV).
// Sanity-check the emitted ISA with ACUDA_DUMP_ISA: the dot/FMA count
// must match the modelled work, otherwise the GFLOP/s are fictitious.
//
// Usage: bench_math [mode] [launches] [grid] [reps] [block]
//   mode      all | f32 | f16 | f16mma | bf16 | sparse | i8 (default all)
//   launches  timed launches per round          (default 20)
//   grid      CTAs                              (default 512)
//   reps      loop iterations inside the kernel (default 1024)
//   block     threads per CTA                   (default 128; tensor
//             modes require a multiple of 32)
//
// Reports min/median GFLOP/s. The MMA paths are the runtime's software
// shuffle-gathered lowering, not native tensor-core instructions; the
// comparison is between kernel formulations on the same backend.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "acuda/compiler.h"
#include "acuda/context.h"
#include "acuda/memory.h"
#include "acuda/stream.h"

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

// 16 independent dependent f32 FMA chains: c = c*a + b. Using the
// accumulator as a multiplicand keeps every step dependent, so the driver
// cannot rewrite the chain as a scaled add (which inflated earlier
// versions of this benchmark). 16 FMA per iteration.
static const char F32_PTX[] =
    ".version 8.8\n"
    ".target sm_120\n"
    ".address_size 64\n"
    ".visible .entry f32_bench(.param .u64 p, .param .u32 reps)\n"
    "{\n"
    "    .reg .u64 %r; .reg .u32 %n; .reg .pred %q;\n"
    "    .reg .f32 %a<16>, %b<16>, %c<16>;\n"
    "    ld.param.u64 %r, [p];\n"
    "    ld.param.u32 %n, [reps];\n"
    "    ld.global.f32 %a0,[%r+0]; ld.global.f32 %a1,[%r+4]; ld.global.f32 %a2,[%r+8]; ld.global.f32 %a3,[%r+12];\n"
    "    ld.global.f32 %a4,[%r+16]; ld.global.f32 %a5,[%r+20]; ld.global.f32 %a6,[%r+24]; ld.global.f32 %a7,[%r+28];\n"
    "    ld.global.f32 %a8,[%r+32]; ld.global.f32 %a9,[%r+36]; ld.global.f32 %a10,[%r+40]; ld.global.f32 %a11,[%r+44];\n"
    "    ld.global.f32 %a12,[%r+48]; ld.global.f32 %a13,[%r+52]; ld.global.f32 %a14,[%r+56]; ld.global.f32 %a15,[%r+60];\n"
    "    ld.global.f32 %b0,[%r+64]; ld.global.f32 %b1,[%r+68]; ld.global.f32 %b2,[%r+72]; ld.global.f32 %b3,[%r+76];\n"
    "    ld.global.f32 %b4,[%r+80]; ld.global.f32 %b5,[%r+84]; ld.global.f32 %b6,[%r+88]; ld.global.f32 %b7,[%r+92];\n"
    "    ld.global.f32 %b8,[%r+96]; ld.global.f32 %b9,[%r+100]; ld.global.f32 %b10,[%r+104]; ld.global.f32 %b11,[%r+108];\n"
    "    ld.global.f32 %b12,[%r+112]; ld.global.f32 %b13,[%r+116]; ld.global.f32 %b14,[%r+120]; ld.global.f32 %b15,[%r+124];\n"
    "    mov.f32 %c0,0f00000000; mov.f32 %c1,0f00000000; mov.f32 %c2,0f00000000; mov.f32 %c3,0f00000000;\n"
    "    mov.f32 %c4,0f00000000; mov.f32 %c5,0f00000000; mov.f32 %c6,0f00000000; mov.f32 %c7,0f00000000;\n"
    "    mov.f32 %c8,0f00000000; mov.f32 %c9,0f00000000; mov.f32 %c10,0f00000000; mov.f32 %c11,0f00000000;\n"
    "    mov.f32 %c12,0f00000000; mov.f32 %c13,0f00000000; mov.f32 %c14,0f00000000; mov.f32 %c15,0f00000000;\n"
    "L:\n"
    "    fma.rn.f32 %c0,%c0,%a0,%b0; fma.rn.f32 %c1,%c1,%a1,%b1; fma.rn.f32 %c2,%c2,%a2,%b2; fma.rn.f32 %c3,%c3,%a3,%b3;\n"
    "    fma.rn.f32 %c4,%c4,%a4,%b4; fma.rn.f32 %c5,%c5,%a5,%b5; fma.rn.f32 %c6,%c6,%a6,%b6; fma.rn.f32 %c7,%c7,%a7,%b7;\n"
    "    fma.rn.f32 %c8,%c8,%a8,%b8; fma.rn.f32 %c9,%c9,%a9,%b9; fma.rn.f32 %c10,%c10,%a10,%b10; fma.rn.f32 %c11,%c11,%a11,%b11;\n"
    "    fma.rn.f32 %c12,%c12,%a12,%b12; fma.rn.f32 %c13,%c13,%a13,%b13; fma.rn.f32 %c14,%c14,%a14,%b14; fma.rn.f32 %c15,%c15,%a15,%b15;\n"
    "    sub.u32 %n,%n,1; setp.ne.u32 %q,%n,0; @%q bra L;\n"
    "    add.f32 %c0,%c0,%c1; add.f32 %c0,%c0,%c2; add.f32 %c0,%c0,%c3; add.f32 %c0,%c0,%c4;\n"
    "    add.f32 %c0,%c0,%c5; add.f32 %c0,%c0,%c6; add.f32 %c0,%c0,%c7; add.f32 %c0,%c0,%c8;\n"
    "    add.f32 %c0,%c0,%c9; add.f32 %c0,%c0,%c10; add.f32 %c0,%c0,%c11; add.f32 %c0,%c0,%c12;\n"
    "    add.f32 %c0,%c0,%c13; add.f32 %c0,%c0,%c14; add.f32 %c0,%c0,%c15;\n"
    "    st.global.f32 [%r+128],%c0;\n"
    "    ret;\n"
    "}\n";

// 16 independent dependent f16x2 FMA chains: c = c*a + b. 16 packed
// FMA per iteration (32 f16 multiplies).
static const char F16_PTX[] =
    ".version 8.8\n"
    ".target sm_120\n"
    ".address_size 64\n"
    ".visible .entry f16_bench(.param .u64 p, .param .u32 reps)\n"
    "{\n"
    "    .reg .u64 %r; .reg .u32 %n; .reg .pred %q;\n"
    "    .reg .b32 %a<16>, %b<16>, %c<16>;\n"
    "    ld.param.u64 %r, [p];\n"
    "    ld.param.u32 %n, [reps];\n"
    "    ld.global.b32 %a0,[%r+0]; ld.global.b32 %a1,[%r+4]; ld.global.b32 %a2,[%r+8]; ld.global.b32 %a3,[%r+12];\n"
    "    ld.global.b32 %a4,[%r+16]; ld.global.b32 %a5,[%r+20]; ld.global.b32 %a6,[%r+24]; ld.global.b32 %a7,[%r+28];\n"
    "    ld.global.b32 %a8,[%r+32]; ld.global.b32 %a9,[%r+36]; ld.global.b32 %a10,[%r+40]; ld.global.b32 %a11,[%r+44];\n"
    "    ld.global.b32 %a12,[%r+48]; ld.global.b32 %a13,[%r+52]; ld.global.b32 %a14,[%r+56]; ld.global.b32 %a15,[%r+60];\n"
    "    ld.global.b32 %b0,[%r+64]; ld.global.b32 %b1,[%r+68]; ld.global.b32 %b2,[%r+72]; ld.global.b32 %b3,[%r+76];\n"
    "    ld.global.b32 %b4,[%r+80]; ld.global.b32 %b5,[%r+84]; ld.global.b32 %b6,[%r+88]; ld.global.b32 %b7,[%r+92];\n"
    "    ld.global.b32 %b8,[%r+96]; ld.global.b32 %b9,[%r+100]; ld.global.b32 %b10,[%r+104]; ld.global.b32 %b11,[%r+108];\n"
    "    ld.global.b32 %b12,[%r+112]; ld.global.b32 %b13,[%r+116]; ld.global.b32 %b14,[%r+120]; ld.global.b32 %b15,[%r+124];\n"
    "    mov.b32 %c0,0; mov.b32 %c1,0; mov.b32 %c2,0; mov.b32 %c3,0;\n"
    "    mov.b32 %c4,0; mov.b32 %c5,0; mov.b32 %c6,0; mov.b32 %c7,0;\n"
    "    mov.b32 %c8,0; mov.b32 %c9,0; mov.b32 %c10,0; mov.b32 %c11,0;\n"
    "    mov.b32 %c12,0; mov.b32 %c13,0; mov.b32 %c14,0; mov.b32 %c15,0;\n"
    "L:\n"
    "    fma.rn.f16x2 %c0,%c0,%a0,%b0; fma.rn.f16x2 %c1,%c1,%a1,%b1; fma.rn.f16x2 %c2,%c2,%a2,%b2; fma.rn.f16x2 %c3,%c3,%a3,%b3;\n"
    "    fma.rn.f16x2 %c4,%c4,%a4,%b4; fma.rn.f16x2 %c5,%c5,%a5,%b5; fma.rn.f16x2 %c6,%c6,%a6,%b6; fma.rn.f16x2 %c7,%c7,%a7,%b7;\n"
    "    fma.rn.f16x2 %c8,%c8,%a8,%b8; fma.rn.f16x2 %c9,%c9,%a9,%b9; fma.rn.f16x2 %c10,%c10,%a10,%b10; fma.rn.f16x2 %c11,%c11,%a11,%b11;\n"
    "    fma.rn.f16x2 %c12,%c12,%a12,%b12; fma.rn.f16x2 %c13,%c13,%a13,%b13; fma.rn.f16x2 %c14,%c14,%a14,%b14; fma.rn.f16x2 %c15,%c15,%a15,%b15;\n"
    "    sub.u32 %n,%n,1; setp.ne.u32 %q,%n,0; @%q bra L;\n"
    "    add.f16x2 %c0,%c0,%c1; add.f16x2 %c0,%c0,%c2; add.f16x2 %c0,%c0,%c3; add.f16x2 %c0,%c0,%c4;\n"
    "    add.f16x2 %c0,%c0,%c5; add.f16x2 %c0,%c0,%c6; add.f16x2 %c0,%c0,%c7; add.f16x2 %c0,%c0,%c8;\n"
    "    add.f16x2 %c0,%c0,%c9; add.f16x2 %c0,%c0,%c10; add.f16x2 %c0,%c0,%c11; add.f16x2 %c0,%c0,%c12;\n"
    "    add.f16x2 %c0,%c0,%c13; add.f16x2 %c0,%c0,%c14; add.f16x2 %c0,%c0,%c15;\n"
    "    st.global.b32 [%r+128],%c0;\n"
    "    ret;\n"
    "}\n";

// Four MMAs per iteration on two A sets and two B sets, with all four
// A/B pairs distinct. This prevents the driver from CSE-ing the dot
// products of repeated identical MMAs (which halved the counted work in
// earlier versions of this benchmark).
static const char BF16_PTX[] =
    ".version 8.8\n"
    ".target sm_120\n"
    ".address_size 64\n"
    ".visible .entry bf16_bench(.param .u64 p, .param .u32 reps)\n"
    "{\n"
    "    .reg .u64 %r, %pa, %off, %zero;\n"
    "    .reg .u32 %lane, %n, %e;\n"
    "    .reg .pred %q;\n"
    "    .reg .b32 %a<8>, %b<8>;\n"
    "    .reg .f32 %c<4>, %d<4>;\n"
    "    ld.param.u64 %r, [p];\n"
    "    ld.param.u32 %n, [reps];\n"
    "    mov.u32 %lane, %laneid;\n"
    "    mov.u64 %zero, 0;\n"
    "    mad.wide.u32 %off, %lane, 16, %zero;\n"
    "    add.u64 %pa, %r, %off;\n"
    "    ld.global.b32 %a0,[%pa]; ld.global.b32 %a1,[%pa+4];\n"
    "    ld.global.b32 %a2,[%pa+8]; ld.global.b32 %a3,[%pa+12];\n"
    "    ld.global.b32 %a4,[%pa+1024]; ld.global.b32 %a5,[%pa+1028];\n"
    "    ld.global.b32 %a6,[%pa+1032]; ld.global.b32 %a7,[%pa+1036];\n"
    "    ld.global.b32 %b0,[%pa+512]; ld.global.b32 %b1,[%pa+516];\n"
    "    ld.global.b32 %b2,[%pa+1536]; ld.global.b32 %b3,[%pa+1540];\n"
    "    mov.f32 %c0,0f00000000; mov.f32 %c1,0f00000000; mov.f32 %d0,0f00000000; mov.f32 %d1,0f00000000;\n"
    "    mov.f32 %c2,0f00000000; mov.f32 %c3,0f00000000; mov.f32 %d2,0f00000000; mov.f32 %d3,0f00000000;\n"
    "L:\n"
    "    mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32\n"
    "        {%c0,%c1,%c2,%c3}, {%a0,%a1,%a2,%a3}, {%b0,%b1}, {%c0,%c1,%c2,%c3};\n"
    "    mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32\n"
    "        {%d0,%d1,%d2,%d3}, {%a4,%a5,%a6,%a7}, {%b2,%b3}, {%d0,%d1,%d2,%d3};\n"
    "    mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32\n"
    "        {%c0,%c1,%c2,%c3}, {%a0,%a1,%a2,%a3}, {%b2,%b3}, {%c0,%c1,%c2,%c3};\n"
    "    mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32\n"
    "        {%d0,%d1,%d2,%d3}, {%a4,%a5,%a6,%a7}, {%b0,%b1}, {%d0,%d1,%d2,%d3};\n"
    "    sub.u32 %n,%n,1; setp.ne.u32 %q,%n,0; @%q bra L;\n"
    "    add.f32 %c0,%c0,%d0; add.f32 %c1,%c1,%d1;\n"
    "    add.f32 %c2,%c2,%d2; add.f32 %c3,%c3,%d3;\n"
    "    mad.wide.u32 %off, %lane, 16, %zero;\n"
    "    add.u64 %pa, %r, %off;\n"
    "    st.global.f32 [%pa+2048],%c0; st.global.f32 [%pa+2052],%c1;\n"
    "    st.global.f32 [%pa+2056],%c2; st.global.f32 [%pa+2060],%c3;\n"
    "    ret;\n"
    "}\n";

// Sparse FP16: four m16n8k32 MMAs per iteration on distinct A/B sets.
static const char SPARSE_PTX[] =
    ".version 8.8\n"
    ".target sm_120\n"
    ".address_size 64\n"
    ".visible .entry sparse_bench(.param .u64 p, .param .u32 reps)\n"
    "{\n"
    "    .reg .u64 %r, %pa, %off, %zero;\n"
    "    .reg .u32 %lane, %n, %e;\n"
    "    .reg .pred %q;\n"
    "    .reg .b32 %a<8>, %b<8>;\n"
    "    .reg .f32 %c<4>, %d<4>;\n"
    "    ld.param.u64 %r, [p];\n"
    "    ld.param.u32 %n, [reps];\n"
    "    mov.u32 %lane, %laneid;\n"
    "    mov.u64 %zero, 0;\n"
    "    mad.wide.u32 %off, %lane, 16, %zero;\n"
    "    add.u64 %pa, %r, %off;\n"
    "    ld.global.b32 %a0,[%pa]; ld.global.b32 %a1,[%pa+4];\n"
    "    ld.global.b32 %a2,[%pa+8]; ld.global.b32 %a3,[%pa+12];\n"
    "    ld.global.b32 %a4,[%pa+1024]; ld.global.b32 %a5,[%pa+1028];\n"
    "    ld.global.b32 %a6,[%pa+1032]; ld.global.b32 %a7,[%pa+1036];\n"
    "    ld.global.b32 %b0,[%pa+512]; ld.global.b32 %b1,[%pa+516];\n"
    "    ld.global.b32 %b2,[%pa+520]; ld.global.b32 %b3,[%pa+524];\n"
    "    ld.global.b32 %b4,[%pa+1536]; ld.global.b32 %b5,[%pa+1540];\n"
    "    ld.global.b32 %b6,[%pa+1544]; ld.global.b32 %b7,[%pa+1548];\n"
    "    mad.wide.u32 %off, %lane, 4, %zero;\n"
    "    add.u64 %pa, %r, %off;\n"
    "    ld.global.b32 %e,[%pa+2048];\n"
    "    mov.f32 %c0,0f00000000; mov.f32 %c1,0f00000000; mov.f32 %d0,0f00000000; mov.f32 %d1,0f00000000;\n"
    "    mov.f32 %c2,0f00000000; mov.f32 %c3,0f00000000; mov.f32 %d2,0f00000000; mov.f32 %d3,0f00000000;\n"
    "L:\n"
    "    mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32\n"
    "        {%c0,%c1,%c2,%c3}, {%a0,%a1,%a2,%a3}, {%b0,%b1,%b2,%b3}, {%c0,%c1,%c2,%c3}, %e, 0;\n"
    "    mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32\n"
    "        {%d0,%d1,%d2,%d3}, {%a4,%a5,%a6,%a7}, {%b4,%b5,%b6,%b7}, {%d0,%d1,%d2,%d3}, %e, 0;\n"
    "    mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32\n"
    "        {%c0,%c1,%c2,%c3}, {%a0,%a1,%a2,%a3}, {%b4,%b5,%b6,%b7}, {%c0,%c1,%c2,%c3}, %e, 0;\n"
    "    mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32\n"
    "        {%d0,%d1,%d2,%d3}, {%a4,%a5,%a6,%a7}, {%b0,%b1,%b2,%b3}, {%d0,%d1,%d2,%d3}, %e, 0;\n"
    "    sub.u32 %n,%n,1; setp.ne.u32 %q,%n,0; @%q bra L;\n"
    "    add.f32 %c0,%c0,%d0; add.f32 %c1,%c1,%d1;\n"
    "    add.f32 %c2,%c2,%d2; add.f32 %c3,%c3,%d3;\n"
    "    mad.wide.u32 %off, %lane, 16, %zero;\n"
    "    add.u64 %pa, %r, %off;\n"
    "    st.global.f32 [%pa+2176],%c0; st.global.f32 [%pa+2180],%c1;\n"
    "    st.global.f32 [%pa+2184],%c2; st.global.f32 [%pa+2188],%c3;\n"
    "    ret;\n"
    "}\n";

// Four MMAs per iteration on two A sets and two B sets, with all four
// A/B pairs distinct. This prevents the driver from CSE-ing the dot
// products of repeated identical MMAs (which halved the counted work in
// earlier versions of this benchmark).
static const char F16MMA_PTX[] =
    ".version 8.8\n"
    ".target sm_120\n"
    ".address_size 64\n"
    ".visible .entry f16mma_bench(.param .u64 p, .param .u32 reps)\n"
    "{\n"
    "    .reg .u64 %r, %pa, %off, %zero;\n"
    "    .reg .u32 %lane, %n, %e;\n"
    "    .reg .pred %q;\n"
    "    .reg .b32 %a<8>, %b<8>;\n"
    "    .reg .f32 %c<4>, %d<4>;\n"
    "    ld.param.u64 %r, [p];\n"
    "    ld.param.u32 %n, [reps];\n"
    "    mov.u32 %lane, %laneid;\n"
    "    mov.u64 %zero, 0;\n"
    "    mad.wide.u32 %off, %lane, 16, %zero;\n"
    "    add.u64 %pa, %r, %off;\n"
    "    ld.global.b32 %a0,[%pa]; ld.global.b32 %a1,[%pa+4];\n"
    "    ld.global.b32 %a2,[%pa+8]; ld.global.b32 %a3,[%pa+12];\n"
    "    ld.global.b32 %a4,[%pa+1024]; ld.global.b32 %a5,[%pa+1028];\n"
    "    ld.global.b32 %a6,[%pa+1032]; ld.global.b32 %a7,[%pa+1036];\n"
    "    ld.global.b32 %b0,[%pa+512]; ld.global.b32 %b1,[%pa+516];\n"
    "    ld.global.b32 %b2,[%pa+1536]; ld.global.b32 %b3,[%pa+1540];\n"
    "    mov.f32 %c0,0f00000000; mov.f32 %c1,0f00000000; mov.f32 %d0,0f00000000; mov.f32 %d1,0f00000000;\n"
    "    mov.f32 %c2,0f00000000; mov.f32 %c3,0f00000000; mov.f32 %d2,0f00000000; mov.f32 %d3,0f00000000;\n"
    "L:\n"
    "    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\n"
    "        {%c0,%c1,%c2,%c3}, {%a0,%a1,%a2,%a3}, {%b0,%b1}, {%c0,%c1,%c2,%c3};\n"
    "    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\n"
    "        {%d0,%d1,%d2,%d3}, {%a4,%a5,%a6,%a7}, {%b2,%b3}, {%d0,%d1,%d2,%d3};\n"
    "    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\n"
    "        {%c0,%c1,%c2,%c3}, {%a0,%a1,%a2,%a3}, {%b2,%b3}, {%c0,%c1,%c2,%c3};\n"
    "    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\n"
    "        {%d0,%d1,%d2,%d3}, {%a4,%a5,%a6,%a7}, {%b0,%b1}, {%d0,%d1,%d2,%d3};\n"
    "    sub.u32 %n,%n,1; setp.ne.u32 %q,%n,0; @%q bra L;\n"
    "    add.f32 %c0,%c0,%d0; add.f32 %c1,%c1,%d1;\n"
    "    add.f32 %c2,%c2,%d2; add.f32 %c3,%c3,%d3;\n"
    "    mad.wide.u32 %off, %lane, 16, %zero;\n"
    "    add.u64 %pa, %r, %off;\n"
    "    st.global.f32 [%pa+2048],%c0; st.global.f32 [%pa+2052],%c1;\n"
    "    st.global.f32 [%pa+2056],%c2; st.global.f32 [%pa+2060],%c3;\n"
    "    ret;\n"
    "}\n";

// Int8: four m16n8k32 MMAs per iteration on distinct A/B sets.
static const char I8_PTX[] =
    ".version 8.8\n"
    ".target sm_120\n"
    ".address_size 64\n"
    ".visible .entry i8_bench(.param .u64 p, .param .u32 reps)\n"
    "{\n"
    "    .reg .u64 %r, %pa, %off, %zero;\n"
    "    .reg .u32 %lane, %n, %e;\n"
    "    .reg .pred %q;\n"
    "    .reg .b32 %a<8>, %b<8>;\n"
    "    .reg .s32 %c<4>, %d<4>;\n"
    "    ld.param.u64 %r, [p];\n"
    "    ld.param.u32 %n, [reps];\n"
    "    mov.u32 %lane, %laneid;\n"
    "    mov.u64 %zero, 0;\n"
    "    mad.wide.u32 %off, %lane, 16, %zero;\n"
    "    add.u64 %pa, %r, %off;\n"
    "    ld.global.b32 %a0,[%pa]; ld.global.b32 %a1,[%pa+4];\n"
    "    ld.global.b32 %a2,[%pa+8]; ld.global.b32 %a3,[%pa+12];\n"
    "    ld.global.b32 %a4,[%pa+1024]; ld.global.b32 %a5,[%pa+1028];\n"
    "    ld.global.b32 %a6,[%pa+1032]; ld.global.b32 %a7,[%pa+1036];\n"
    "    ld.global.b32 %b0,[%pa+512]; ld.global.b32 %b1,[%pa+516];\n"
    "    ld.global.b32 %b2,[%pa+1536]; ld.global.b32 %b3,[%pa+1540];\n"
    "    mov.s32 %c0,0; mov.s32 %c1,0; mov.s32 %d0,0; mov.s32 %d1,0;\n"
    "    mov.s32 %c2,0; mov.s32 %c3,0; mov.s32 %d2,0; mov.s32 %d3,0;\n"
    "L:\n"
    "    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32\n"
    "        {%c0,%c1,%c2,%c3}, {%a0,%a1,%a2,%a3}, {%b0,%b1}, {%c0,%c1,%c2,%c3};\n"
    "    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32\n"
    "        {%d0,%d1,%d2,%d3}, {%a4,%a5,%a6,%a7}, {%b2,%b3}, {%d0,%d1,%d2,%d3};\n"
    "    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32\n"
    "        {%c0,%c1,%c2,%c3}, {%a0,%a1,%a2,%a3}, {%b2,%b3}, {%c0,%c1,%c2,%c3};\n"
    "    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32\n"
    "        {%d0,%d1,%d2,%d3}, {%a4,%a5,%a6,%a7}, {%b0,%b1}, {%d0,%d1,%d2,%d3};\n"
    "    sub.u32 %n,%n,1; setp.ne.u32 %q,%n,0; @%q bra L;\n"
    "    add.s32 %c0,%c0,%d0; add.s32 %c1,%c1,%d1;\n"
    "    add.s32 %c2,%c2,%d2; add.s32 %c3,%c3,%d3;\n"
    "    mad.wide.u32 %off, %lane, 16, %zero;\n"
    "    add.u64 %pa, %r, %off;\n"
    "    st.global.s32 [%pa+2048],%c0; st.global.s32 [%pa+2052],%c1;\n"
    "    st.global.s32 [%pa+2056],%c2; st.global.s32 [%pa+2060],%c3;\n"
    "    ret;\n"
    "}\n";

typedef enum { MODE_F32, MODE_F16, MODE_BF16, MODE_SPARSE, MODE_I8,
               MODE_F16MMA } bench_mode_t;

typedef struct {
    const char *name;
    const char *ptx;
    bench_mode_t mode;
    double flops_per_iter; // per thread-iteration (scalar modes) / per
                           // warp-iteration (tensor modes)
} bench_case_t;

static const bench_case_t cases[] = {
    { "f32", F32_PTX, MODE_F32, 16.0 * 2.0 },
    { "f16", F16_PTX, MODE_F16, 16.0 * 2.0 * 2.0 },
    { "f16mma", F16MMA_PTX, MODE_F16MMA,
      4.0 * 2.0 * 16.0 * 8.0 * 16.0 },
    { "bf16", BF16_PTX, MODE_BF16, 4.0 * 2.0 * 16.0 * 8.0 * 16.0 },
    { "sparse", SPARSE_PTX, MODE_SPARSE,
      4.0 * 2.0 * 16.0 * 8.0 * 32.0 },
    { "i8", I8_PTX, MODE_I8, 4.0 * 2.0 * 16.0 * 8.0 * 32.0 },
};

static void fill_pattern(bench_mode_t mode, uint32_t *w, size_t nwords) {
    for (size_t i = 0; i < nwords; i++)
        w[i] = (mode == MODE_F16) ? 0x3c003c00u
               : (mode == MODE_BF16) ? 0x3f803f80u
               : (mode == MODE_SPARSE) ? 0u
               : (mode == MODE_I8) ? 0x01010101u
               : (mode == MODE_F16MMA) ? 0x3c003c00u
                                       : 0x3f800000u;
}

static int run_case(acuda_context_t ctx, const bench_case_t *bc, int launches,
                    uint32_t grid, uint32_t reps, uint32_t block) {
    char err[1024];
    acuda_kernel_t *k = NULL;
    if (acuda_compile_ptx(bc->ptx, NULL, &k, err, sizeof(err), NULL) !=
        ACUDA_SUCCESS) {
        printf("%-6s compile failed: %s\n", bc->name, err);
        return 1;
    }
    acuda_kernel_info_t info;
    acuda_kernel_info(k, &info);
    acuda_buffer_t buf = NULL;
    uint32_t words[1024];
    fill_pattern(bc->mode, words, 1024);
    if (acuda_malloc(ctx, &buf, sizeof(words)) != ACUDA_SUCCESS ||
        acuda_memcpy_h2d(ctx, buf, words, sizeof(words), NULL) !=
            ACUDA_SUCCESS) {
        printf("%-6s buffer setup failed\n", bc->name);
        acuda_kernel_free(k);
        return 1;
    }
    uint32_t push = reps;
    for (int i = 0; i < 50; i++)
        acuda_launch_kernel(ctx, k, &buf, info.num_ptr_params, &push,
                            sizeof(push), grid, block, NULL);
    acuda_stream_synchronize(ctx, NULL);

    enum { ROUNDS = 7 };
    double t[ROUNDS];
    double threads = (double)grid * (double)block;
    double units = (bc->mode == MODE_F32 || bc->mode == MODE_F16)
                       ? threads
                       : threads / 32.0; // tensor modes are per warp
    double flops = units * (double)reps * bc->flops_per_iter;
    for (int r = 0; r < ROUNDS; r++) {
        double t0 = now_ms();
        for (int i = 0; i < launches; i++)
            if (acuda_launch_kernel(ctx, k, &buf, info.num_ptr_params,
                                    &push, sizeof(push), grid, block,
                                    NULL) != ACUDA_SUCCESS) {
                printf("%-6s launch failed\n", bc->name);
                acuda_free(ctx, buf);
                acuda_kernel_free(k);
                return 1;
            }
        acuda_stream_synchronize(ctx, NULL);
        t[r] = (now_ms() - t0) / (double)launches;
    }
    qsort(t, ROUNDS, sizeof(double), cmp_double);
    printf("%-6s min %8.3f ms  median %8.3f ms  %8.1f GFLOP/s\n",
           bc->name, t[0], t[ROUNDS / 2], flops / t[0] / 1e6);
    acuda_free(ctx, buf);
    acuda_kernel_free(k);
    return 0;
}

int main(int argc, char **argv) {
    const char *want = argc > 1 ? argv[1] : "all";
    int launches = argc > 2 ? atoi(argv[2]) : 20;
    uint32_t grid = argc > 3 ? (uint32_t)atoi(argv[3]) : 512;
    uint32_t reps = argc > 4 ? (uint32_t)atoi(argv[4]) : 1024;
    uint32_t block = argc > 5 ? (uint32_t)atoi(argv[5]) : 128;
    if (launches < 1 || grid < 1 || reps < 1 || block < 1 || block > 1024) {
        printf("bad arguments\n");
        return 2;
    }
    acuda_context_t ctx = NULL;
    if (acuda_init(&ctx) != ACUDA_SUCCESS) {
        printf("acuda_init failed\n");
        return 1;
    }
    int rc = 0;
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        if (strcmp(want, "all") != 0 && strcmp(want, cases[i].name) != 0)
            continue;
        if (cases[i].mode != MODE_F32 && cases[i].mode != MODE_F16 &&
            block % 32 != 0) {
            printf("%-6s skipped: block must be a multiple of 32\n",
                   cases[i].name);
            continue;
        }
        rc |= run_case(ctx, &cases[i], launches, grid, reps, block);
    }
    acuda_shutdown(ctx);
    return rc;
}
