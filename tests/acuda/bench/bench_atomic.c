// Shared-memory f32 atomic add: native OpAtomicFAddEXT
// (VK_EXT_shader_atomic_float shaderSharedFloat32AtomicAdd) versus the
// u32-word CAS loop the compiler uses everywhere else.
//
// Usage: bench_atomic [launches] [grid] [reps] [block]
//   launches  timed launches per round (default 20)
//   grid      CTAs (default 1024)
//   reps      atomics per thread (default 64)
//   block     threads per CTA (default 256)
//
// Every CTA has its own f32 accumulator in shared memory; all lanes
// hammer it with `red.shared.add.f32`, so this is the contended
// reduction shape. The kernel writes the final counter per lane and
// the host checks it before timing.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "acuda/compiler.h"
#include "acuda/context.h"
#include "acuda/memory.h"
#include "acuda/stream.h"

static const char PTX[] =
    ".version 8.5\n"
    ".target sm_80\n"
    ".address_size 64\n"
    ".visible .shared .align 4 .b8 sh[4];\n"
    ".visible .entry atom_bench(.param .u64 io, .param .u32 reps)\n"
    "{\n"
    "    .reg .u64 %rd<4>;\n"
    "    .reg .u32 %r<5>;\n"
    "    .reg .f32 %f<4>;\n"
    "    .reg .pred %p;\n"
    "    ld.param.u64 %rd1, [io];\n"
    "    ld.param.u32 %r4, [reps];\n"
    "    mov.u32 %r1, %tid.x;\n"
    "    cvta.to.global.u64 %rd2, %rd1;\n"
    "    mov.f32 %f1, 0f00000000;\n"
    "    st.shared.f32 [sh], %f1;\n"
    "    bar.sync 0;\n"
    "    mov.f32 %f2, 0f3F800000;\n"
    "    mov.u32 %r2, 0;\n"
    "LOOP:\n"
    "    setp.ge.u32 %p, %r2, %r4;\n"
    "    @%p bra DONE;\n"
    "    red.shared.add.f32 [sh], %f2;\n"
    "    add.u32 %r2, %r2, 1;\n"
    "    bra.uni LOOP;\n"
    "DONE:\n"
    "    bar.sync 0;\n"
    "    ld.shared.f32 %f3, [sh];\n"
    "    mul.wide.u32 %rd3, %r1, 4;\n"
    "    add.u64 %rd3, %rd2, %rd3;\n"
    "    st.global.f32 [%rd3], %f3;\n"
    "    ret;\n"
    "}\n";

#define ROUNDS 7

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static acuda_kernel_t *compile(const char *label, const acuda_compiler_caps_t *caps) {
    acuda_kernel_t *k = NULL;
    char err[1024];
    if (acuda_compile_ptx(PTX, NULL, &k, err, sizeof(err), caps) != ACUDA_SUCCESS) {
        printf("%s compile failed: %s\n", label, err);
        return NULL;
    }
    return k;
}

static int check(acuda_context_t ctx, acuda_kernel_t *k, uint32_t block,
                 uint32_t reps) {
    acuda_kernel_info_t info;
    if (acuda_kernel_info(k, &info) != ACUDA_SUCCESS) {
        printf("kernel info failed\n");
        return 1;
    }
    uint32_t push[1] = { reps };
    acuda_buffer_t out = NULL;
    size_t bytes = (size_t)block * 4;
    if (acuda_malloc_device(ctx, &out, bytes) != ACUDA_SUCCESS) {
        printf("device buffer failed\n");
        return 1;
    }
    float *h = malloc(bytes);
    memset(h, 0, bytes);
    int bad = 1;
    if (acuda_memcpy_h2d(ctx, out, h, bytes, NULL) == ACUDA_SUCCESS &&
        acuda_launch_kernel(ctx, k, &out, 1, push, sizeof(push), 1, block,
                            NULL) == ACUDA_SUCCESS &&
        acuda_stream_synchronize(ctx, NULL) == ACUDA_SUCCESS &&
        acuda_memcpy_d2h(ctx, h, out, bytes, NULL) == ACUDA_SUCCESS) {
        float want = (float)block * (float)reps;
        bad = 0;
        for (uint32_t i = 0; i < block; i++)
            if (h[i] != want)
                bad++;
    }
    if (bad) {
        printf("correctness check failed (want %.0f)\n",
               (double)block * reps);
    }
    free(h);
    acuda_free(ctx, out);
    return bad ? 1 : 0;
}

int main(int argc, char **argv) {
    int launches = argc > 1 ? atoi(argv[1]) : 20;
    uint32_t grid = argc > 2 ? (uint32_t)atoi(argv[2]) : 1024;
    uint32_t reps = argc > 3 ? (uint32_t)atoi(argv[3]) : 64;
    uint32_t block = argc > 4 ? (uint32_t)atoi(argv[4]) : 256;
    if (launches < 1 || grid < 1 || reps < 1 || block < 1 || block > 1024) {
        printf("bad arguments\n");
        return 2;
    }
    acuda_context_t ctx = NULL;
    if (acuda_init(&ctx) != ACUDA_SUCCESS) {
        printf("acuda_init failed\n");
        return 1;
    }
    const acuda_compiler_caps_t *ctx_caps = acuda_context_get_compiler_caps(ctx);
    acuda_compiler_caps_t caps_n, caps_f;
    if (ctx_caps) caps_n = *ctx_caps; else acuda_compiler_caps_init(&caps_n);
    caps_f = caps_n;
    unsigned atomic_mask = caps_n.atomic_float;
    if (!(atomic_mask & ACUDA_ATOMIC_F32_SHARED)) {
        printf("native shared f32 atomic add unavailable on this device\n");
        acuda_shutdown(ctx);
        return 0;
    }
    acuda_kernel_t *kn = compile("native", &caps_n);
    acuda_compiler_set_atomic_float(&caps_f, 0);
    acuda_kernel_t *kf = compile("fallback", &caps_f);
    if (!kn || !kf) {
        acuda_kernel_free(kn);
        acuda_kernel_free(kf);
        acuda_shutdown(ctx);
        return 1;
    }

    if (check(ctx, kn, block, reps) || check(ctx, kf, block, reps)) {
        acuda_kernel_free(kn);
        acuda_kernel_free(kf);
        acuda_shutdown(ctx);
        return 1;
    }

    size_t bytes = (size_t)block * 4;
    acuda_buffer_t out = NULL;
    if (acuda_malloc_device(ctx, &out, bytes) != ACUDA_SUCCESS) {
        printf("device buffer failed\n");
        return 1;
    }
    float *zero = calloc(block, 4);
    acuda_memcpy_h2d(ctx, out, zero, bytes, NULL);
    uint32_t push[1] = { reps };

    // Warm both pipelines to settle clocks.
    for (int i = 0; i < 10; i++) {
        acuda_launch_kernel(ctx, kn, &out, 1, push, sizeof(push), grid,
                            block, NULL);
        acuda_launch_kernel(ctx, kf, &out, 1, push, sizeof(push), grid,
                            block, NULL);
        // Sync per warmup round: a fallback batch at a large grid/reps
        // can otherwise exceed the driver's per-submission timeout.
        acuda_stream_synchronize(ctx, NULL);
    }

    double tn[ROUNDS], tf[ROUNDS];
    for (int r = 0; r < ROUNDS; r++)
        tn[r] = tf[r] = 1e30;
    for (int r = 0; r < ROUNDS; r++) {
        for (int pass = 0; pass < 2; pass++) {
            int native = (r % 2 == 0) ? pass == 0 : pass == 1;
            double t0 = now_ms();
            for (int i = 0; i < launches; i++) {
                acuda_kernel_t *k = native ? kn : kf;
                if (acuda_launch_kernel(ctx, k, &out, 1, push,
                                        sizeof(push), grid, block,
                                        NULL) != ACUDA_SUCCESS) {
                    printf("launch failed\n");
                    return 1;
                }
            }
            acuda_status_t st = acuda_stream_synchronize(ctx, NULL);
            double t1 = now_ms();
            if (st != ACUDA_SUCCESS)
                printf("sync failed: %s\n", acuda_strerror(st));
            (native ? tn : tf)[r] = (t1 - t0) / launches;
        }
    }
    qsort(tn, ROUNDS, sizeof(double), cmp_double);
    qsort(tf, ROUNDS, sizeof(double), cmp_double);
    double atomics = (double)grid * block * reps;
    printf("grid=%u block=%u reps=%u (%g atomics/CTA) launches=%d\n",
           grid, block, reps, (double)reps * block, launches);
    printf("native   min %8.3f  median %8.3f ms  (%.1f Gatom/s at min)\n",
           tn[0], tn[ROUNDS / 2], atomics / tn[0] / 1e6);
    printf("fallback min %8.3f  median %8.3f ms  (%.1f Gatom/s at min)\n",
           tf[0], tf[ROUNDS / 2], atomics / tf[0] / 1e6);
    printf("min speedup %.2fx   median speedup %.2fx\n",
           tf[0] / tn[0], tf[ROUNDS / 2] / tn[ROUNDS / 2]);
    printf("note: rates count logical lane-atomics; RADV may coalesce a "
           "uniform per-wave shared atomic (the CAS fallback cannot)\n");

    acuda_free(ctx, out);
    free(zero);
    acuda_kernel_free(kn);
    acuda_kernel_free(kf);
    acuda_shutdown(ctx);
    return 0;
}
