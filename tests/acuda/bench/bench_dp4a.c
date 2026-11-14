// dp4a A/B benchmark: native OpSDot/OpUDot/OpSUDot
// (SPV_KHR_integer_dot_product) versus the exact multiply-add expansion.
//
// Usage: bench_dp4a [launches] [grid] [reps] [block]
//   launches  timed launches per round (default 20)
//   grid      CTAs (default 1024)
//   reps      loop iterations inside the kernel, >= 1 (default 64)
//   block     threads per CTA (default 256)
//   nbuf      independent buffers, round-robined per launch (default 1).
//             nbuf>1 measures submission overlap; run with
//             ACUDA_FORCE_SERIAL=1 to compare against full serialization.
//
// Reports the minimum and median of several alternating rounds.  The
// feature is optional: when the device has no
// shaderIntegerDotProduct the tool says so and exits 0.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "acuda/compiler.h"
#include "acuda/context.h"
#include "acuda/memory.h"
#include "acuda/stream.h"

#ifndef ACUDA_BENCH_PTX_DIR
#define ACUDA_BENCH_PTX_DIR "tests/bench"
#endif

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = malloc((size_t)n + 1);
    if (!b || fread(b, 1, (size_t)n, f) != (size_t)n) {
        fclose(f);
        free(b);
        return NULL;
    }
    b[n] = '\0';
    fclose(f);
    return b;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

// Count function-body instructions and dot-product instructions.
static void spv_stats(const uint32_t *w, size_t n, unsigned *insns,
                      unsigned *dots) {
    int in_fn = 0;
    *insns = 0;
    *dots = 0;
    for (size_t i = 5; i < n;) {
        uint32_t wc = w[i] >> 16;
        if (wc == 0)
            break;
        uint32_t op = w[i] & 0xffffu;
        if (op == 54) { // OpFunction
            in_fn = 1;
        } else if (op == 56) { // OpFunctionEnd
            in_fn = 0;
        } else if (in_fn) {
            (*insns)++;
            if (op == 4450u || op == 4451u || op == 4452u)
                (*dots)++;
        }
        i += wc;
    }
}

static acuda_kernel_t *build(const char *ptx, char *err, size_t errlen, const acuda_compiler_caps_t *caps) {
    acuda_kernel_t *k = NULL;
    if (acuda_compile_ptx(ptx, NULL, &k, err, errlen, caps) != ACUDA_SUCCESS)
        return NULL;
    acuda_kernel_info_t info;
    if (acuda_kernel_info(k, &info) != ACUDA_SUCCESS ||
        info.num_ptr_params != 1 || info.push_bytes != 4) {
        snprintf(err, errlen,
                 "unexpected kernel ABI (ptr=%u push=%u)",
                 info.num_ptr_params, info.push_bytes);
        acuda_kernel_free(k);
        return NULL;
    }
    return k;
}

int main(int argc, char **argv) {
    int launches = argc > 1 ? atoi(argv[1]) : 20;
    uint32_t grid = argc > 2 ? (uint32_t)atoi(argv[2]) : 1024;
    uint32_t reps = argc > 3 ? (uint32_t)atoi(argv[3]) : 64;
    uint32_t block = argc > 4 ? (uint32_t)atoi(argv[4]) : 256;
    int nbuf = argc > 5 ? atoi(argv[5]) : 1;
    if (launches < 1 || grid < 1 || reps < 1 || block < 1 || block > 1024 ||
        nbuf < 1 || nbuf > 64) {
        printf("bad arguments\n");
        return 2;
    }
    char path[1024];
    snprintf(path, sizeof(path), "%s/dp4a_bench.ptx", ACUDA_BENCH_PTX_DIR);
    char *ptx = read_file(path);
    if (!ptx) {
        printf("cannot read %s\n", path);
        return 1;
    }
    acuda_context_t ctx = NULL;
    if (acuda_init(&ctx) != ACUDA_SUCCESS) {
        printf("acuda_init failed\n");
        free(ptx);
        return 1;
    }
    char err[1024] = {0};
    // Compile with the runtime's decision first: if the device lacks the
    // feature, the module is the fallback and there is nothing to A/B.
    acuda_kernel_t *probe = build(ptx, err, sizeof(err), acuda_context_get_compiler_caps(ctx));
    if (!probe) {
        printf("compile failed: %s\n", err);
        free(ptx);
        acuda_shutdown(ctx);
        return 1;
    }
    const uint32_t *words = NULL;
    size_t nwords = 0;
    unsigned insns_native = 0, dots_native = 0;
    acuda_kernel_spirv(probe, &words, &nwords);
    spv_stats(words, nwords, &insns_native, &dots_native);
    if (dots_native == 0) {
        printf("device has no shaderIntegerDotProduct; "
               "dp4a runs on the exact expansion only\n");
        acuda_kernel_free(probe);
        free(ptx);
        acuda_shutdown(ctx);
        return 0;
    }
    acuda_kernel_free(probe);

    acuda_compiler_caps_t caps_n, caps_f;
    if (acuda_context_get_compiler_caps(ctx))
        caps_n = *acuda_context_get_compiler_caps(ctx);
    else
        acuda_compiler_caps_init(&caps_n);
    caps_f = caps_n;
    acuda_compiler_set_dot_product(&caps_n, 1);
    acuda_kernel_t *kn = build(ptx, err, sizeof(err), &caps_n);
    if (!kn) {
        printf("native compile failed: %s\n", err);
        free(ptx);
        acuda_shutdown(ctx);
        return 1;
    }
    acuda_compiler_set_dot_product(&caps_f, 0);
    acuda_kernel_t *kf = build(ptx, err, sizeof(err), &caps_f);
    if (!kf) {
        printf("fallback compile failed: %s\n", err);
        acuda_kernel_free(kn);
        free(ptx);
        acuda_shutdown(ctx);
        return 1;
    }
    const uint32_t *w = NULL;
    size_t nw = 0;
    unsigned insns_fallback = 0, dots_fallback = 0;
    acuda_kernel_spirv(kn, &w, &nw);
    spv_stats(w, nw, &insns_native, &dots_native);
    acuda_kernel_spirv(kf, &w, &nw);
    spv_stats(w, nw, &insns_fallback, &dots_fallback);

    acuda_buffer_t bufs[64] = { 0 };
    uint32_t input[12] = {
        0x01020304u, 0x05060708u, 0x11121314u, 0x21222324u,
        0x31323334u, 0x41424344u, 0x51525354u, 0x61626364u,
        0, 0, 0, 0
    };
    int setup_ok = 1;
    for (int b = 0; b < nbuf; b++) {
        if (acuda_malloc(ctx, &bufs[b], sizeof(input)) != ACUDA_SUCCESS ||
            acuda_memcpy_h2d(ctx, bufs[b], input, sizeof(input), NULL) !=
                ACUDA_SUCCESS) {
            setup_ok = 0;
            break;
        }
    }
    if (!setup_ok) {
        printf("buffer setup failed\n");
        for (int b = 0; b < nbuf; b++)
            if (bufs[b])
                acuda_free(ctx, bufs[b]);
        acuda_kernel_free(kn);
        acuda_kernel_free(kf);
        free(ptx);
        acuda_shutdown(ctx);
        return 1;
    }

    acuda_kernel_info_t info;
    acuda_kernel_info(kn, &info);
    printf("kernel: %s, block_x default %u\n", info.entry, info.block_x);
    printf("SPIR-V function body: native %u instructions (%u dot), "
           "fallback %u instructions (%u dot)\n",
           insns_native, dots_native, insns_fallback, dots_fallback);
    printf("grid=%u block=%u reps=%u (%u dp4a/thread) nbuf=%d\n", grid,
           block, reps, reps * 32u, nbuf);

    uint32_t push = reps;
    // Warm up both pipelines to settle clocks.
    for (int i = 0; i < 100; i++) {
        acuda_launch_kernel(ctx, kn, &bufs[i % nbuf], 1, &push,
                            sizeof(push), grid, block, NULL);
        acuda_launch_kernel(ctx, kf, &bufs[i % nbuf], 1, &push,
                            sizeof(push), grid, block, NULL);
    }
    acuda_stream_synchronize(ctx, NULL);

    enum { ROUNDS = 7 };
    double tn[ROUNDS], tf[ROUNDS], en[ROUNDS], ef[ROUNDS];
    for (int r = 0; r < ROUNDS; r++) {
        for (int pass = 0; pass < 2; pass++) {
            int native = (r % 2 == 0) ? pass == 0 : pass == 1;
            acuda_kernel_t *k = native ? kn : kf;
            double t0 = now_ms();
            for (int i = 0; i < launches; i++)
                if (acuda_launch_kernel(ctx, k, &bufs[i % nbuf], 1, &push,
                                        sizeof(push), grid, block,
                                        NULL) != ACUDA_SUCCESS) {
                    printf("launch failed\n");
                    return 1;
                }
            double tm = now_ms();
            acuda_stream_synchronize(ctx, NULL);
            double t1 = now_ms();
            (native ? tn : tf)[r] = (t1 - t0) / (double)launches;
            (native ? en : ef)[r] = (tm - t0) / (double)launches;
        }
    }
    qsort(tn, ROUNDS, sizeof(double), cmp_double);
    qsort(tf, ROUNDS, sizeof(double), cmp_double);
    qsort(en, ROUNDS, sizeof(double), cmp_double);
    qsort(ef, ROUNDS, sizeof(double), cmp_double);
    double threads = (double)grid * (double)block;
    double per_thread = (double)reps * 32.0;
    double dots_per_launch = threads * per_thread;
    printf("native   min %.3f  median %.3f ms  (%.1f Gdp4a/s at min)\n",
           tn[0], tn[ROUNDS / 2], dots_per_launch / tn[0] / 1e6);
    printf("fallback min %.3f  median %.3f ms  (%.1f Gdp4a/s at min)\n",
           tf[0], tf[ROUNDS / 2], dots_per_launch / tf[0] / 1e6);
    printf("enqueue  min %.3f  median %.3f ms  (native), %.3f / %.3f ms "
           "(fallback)\n", en[0], en[ROUNDS / 2], ef[0], ef[ROUNDS / 2]);
    printf("min speedup: %.2fx   median speedup: %.2fx\n",
           tf[0] / tn[0], tf[ROUNDS / 2] / tn[ROUNDS / 2]);

    for (int b = 0; b < nbuf; b++)
        acuda_free(ctx, bufs[b]);
    acuda_kernel_free(kn);
    acuda_kernel_free(kf);
    free(ptx);
    acuda_shutdown(ctx);
    return 0;
}
