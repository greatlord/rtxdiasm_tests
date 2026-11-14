// Phase 4: compile hand-written PTX, launch on GPU, verify vs CPU.
// Also negative tests: numba-shaped PTX and garbage must fail cleanly.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "acuda/compiler.h"
#include "acuda/context.h"
#include "acuda/memory.h"
#include "acuda/stream.h"
// Internal parameter analyzer (regression for the shared/global lineage
// false positive); the test target adds src/compiler to its include path.
#include "ptx.h"
#include "spirv_internal.h"
static acuda_compiler_caps_t g_test_caps;

#ifndef ACUDA_TEST_PTX_DIR
#define ACUDA_TEST_PTX_DIR "tests/ptx"
#endif

#ifdef _WIN32
// MSVC has no POSIX setenv/unsetenv. The test uses them only to exercise
// the compiler's PTX/SM ceiling environment variables.
static int test_setenv(const char *name, const char *value, int overwrite) {
    (void)overwrite;
    return _putenv_s(name, value) == 0 ? 0 : -1;
}
static int test_unsetenv(const char *name) {
    return _putenv_s(name, "") == 0 ? 0 : -1;
}
#define setenv test_setenv
#define unsetenv test_unsetenv
#endif

static int failures = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        failures++; \
    } \
} while (0)
#define CHECK_ST(call) do { \
    acuda_status_t _s = (call); \
    if (_s != ACUDA_SUCCESS) { \
        printf("FAIL %s:%d: %s -> %s\n", __FILE__, __LINE__, #call, acuda_strerror(_s)); \
        failures++; \
    } \
} while (0)

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        printf("FAIL: cannot open %s\n", path);
        failures++;
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = malloc((size_t)n + 1);
    if (!b) {
        fclose(f);
        return NULL;
    }
    if (fread(b, 1, (size_t)n, f) != (size_t)n) {
        fclose(f);
        free(b);
        return NULL;
    }
    b[n] = '\0';
    fclose(f);
    return b;
}

// Detect the typed workgroup view: an array of 512 f16 elements would be
// the 1024-byte shared C tile viewed as halves instead of u32 words.
static int spv_has_typed_f16_shared_array(const uint32_t *words,
                                          size_t nwords) {
    static uint32_t const_id[4096], const_val[4096], f16_id[4096];
    int nconst = 0, nf16 = 0;
    size_t pos = 5;
    while (pos < nwords) {
        uint32_t wc = words[pos] >> 16, op = words[pos] & 0xffffu;
        if (wc == 0 || pos + wc > nwords)
            break;
        if (op == 43 /* OpConstant */ && wc >= 4 && nconst < 4096) {
            const_id[nconst] = words[pos + 2];
            const_val[nconst] = words[pos + 3];
            nconst++;
        } else if (op == 22 /* OpTypeFloat */ && wc >= 3 &&
                   words[pos + 2] == 16 && nf16 < 4096) {
            f16_id[nf16++] = words[pos + 1];
        }
        pos += wc;
    }
    pos = 5;
    while (pos < nwords) {
        uint32_t wc = words[pos] >> 16, op = words[pos] & 0xffffu;
        if (wc == 0 || pos + wc > nwords)
            break;
        if (op == 28 /* OpTypeArray */ && wc >= 4) {
            uint32_t elem = words[pos + 2], len_id = words[pos + 3];
            uint32_t len = 0;
            int is_f16 = 0;
            for (int i = 0; i < nf16; i++)
                if (f16_id[i] == elem)
                    is_f16 = 1;
            for (int i = 0; i < nconst; i++)
                if (const_id[i] == len_id)
                    len = const_val[i];
            if (is_f16 && len == 512)
                return 1;
        }
        pos += wc;
    }
    return 0;
}

static void run_typed_shared_guard(const char *file, const char *entry) {
    char path[1024], err[1024] = {0};
    snprintf(path, sizeof(path), "%s/%s", ACUDA_TEST_PTX_DIR, file);
    char *ptx = read_file(path);
    if (!ptx) {
        failures++;
        return;
    }
    acuda_compiler_set_cooperative_matrix(&g_test_caps, 1, 1);
    acuda_kernel_t *k = NULL;
    acuda_status_t st = acuda_compile_ptx(ptx, entry, &k, err,
                                          sizeof(err), &g_test_caps);
    CHECK(st == ACUDA_SUCCESS);
    if (st == ACUDA_SUCCESS) {
        const uint32_t *words = NULL;
        size_t nwords = 0;
        CHECK_ST(acuda_kernel_spirv(k, &words, &nwords));
        CHECK(words && nwords > 5);
        if (words && nwords > 5)
            CHECK(!spv_has_typed_f16_shared_array(words, nwords));
        acuda_kernel_free(k);
    }
    acuda_compiler_set_cooperative_matrix(&g_test_caps, 0, 0);
    free(ptx);
}

#define N (1u << 20)
#define GRID (N / 256)

static void run_vecadd(acuda_context_t ctx, const char *ptx,
                       const char *entry, int is_float) {
    char err[1024];
    acuda_kernel_t *k = NULL;
    acuda_status_t st = acuda_compile_ptx(ptx, entry, &k, err, sizeof(err), &g_test_caps);
    if (st != ACUDA_SUCCESS) {
        printf("FAIL: compile %s -> %s (%s)\n", entry, acuda_strerror(st), err);
        failures++;
        return;
    }
    acuda_kernel_info_t info;
    CHECK_ST(acuda_kernel_info(k, &info));
    CHECK(info.num_ptr_params == 3);
    CHECK(info.push_bytes == 4);
    CHECK(info.block_x == 256);
    const uint32_t *words = NULL;
    size_t nwords = 0;
    CHECK_ST(acuda_kernel_spirv(k, &words, &nwords));
    CHECK(words && nwords > 5 && words[0] == 0x07230203u);

    acuda_buffer_t a = NULL, b = NULL, c = NULL;
    CHECK_ST(acuda_malloc(ctx, &a, (size_t)N * 4));
    CHECK_ST(acuda_malloc(ctx, &b, (size_t)N * 4));
    CHECK_ST(acuda_malloc(ctx, &c, (size_t)N * 4));

    uint32_t *ha = malloc((size_t)N * 4);
    uint32_t *hb = malloc((size_t)N * 4);
    uint32_t *hc = malloc((size_t)N * 4);
    CHECK(ha && hb && hc);
    if (is_float) {
        float *fa = (float *)ha, *fb = (float *)hb;
        for (uint32_t i = 0; i < N; i++) {
            fa[i] = (float)i * 0.5f;
            fb[i] = (float)i * 2.0f + 1.0f;
        }
    } else {
        for (uint32_t i = 0; i < N; i++) {
            ha[i] = i;
            hb[i] = 2 * i + 1;
        }
    }
    CHECK_ST(acuda_memcpy_h2d(ctx, a, ha, (size_t)N * 4, NULL));
    CHECK_ST(acuda_memcpy_h2d(ctx, b, hb, (size_t)N * 4, NULL));

    acuda_buffer_t bufs[3] = { a, b, c };
    uint32_t n = N;
    CHECK_ST(acuda_launch_kernel(ctx, k, bufs, 3, &n, sizeof(n), GRID, 256, NULL));
    CHECK_ST(acuda_stream_synchronize(ctx, NULL));
    CHECK_ST(acuda_memcpy_d2h(ctx, hc, c, (size_t)N * 4, NULL));

    uint32_t bad = 0;
    if (is_float) {
        float *fa = (float *)ha, *fb = (float *)hb, *fc = (float *)hc;
        for (uint32_t i = 0; i < N; i++) {
            float want = fa[i] * 2.0f + fb[i];
            if (fc[i] != want) {
                if (bad < 5)
                    printf("  %s mismatch[%u]: got %f want %f\n", entry, i, fc[i], want);
                bad++;
            }
        }
    } else {
        for (uint32_t i = 0; i < N; i++) {
            uint32_t want = ha[i] + hb[i];
            if (hc[i] != want) {
                if (bad < 5)
                    printf("  %s mismatch[%u]: got %u want %u\n", entry, i, hc[i], want);
                bad++;
            }
        }
    }
    CHECK(bad == 0);
    printf("  %s: %u elements ok\n", entry, N);

    free(ha);
    free(hb);
    free(hc);
    CHECK_ST(acuda_free(ctx, a));
    CHECK_ST(acuda_free(ctx, b));
    CHECK_ST(acuda_free(ctx, c));
    acuda_kernel_free(k);
}

// One-pointer kernel: the buffer holds two input words and receives the
// result in word 0 (used by the break/continue and multi-block-callee
// regression kernels).
static void run_single_buf(acuda_context_t ctx, const char *file,
                           uint32_t in0, uint32_t in1, uint32_t want,
                           const char *label) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", ACUDA_TEST_PTX_DIR, file);
    char *ptx = read_file(path);
    if (!ptx) {
        failures++;
        return;
    }
    char err[1024];
    acuda_kernel_t *k = NULL;
    if (acuda_compile_ptx(ptx, NULL, &k, err, sizeof(err), &g_test_caps) != ACUDA_SUCCESS) {
        printf("FAIL: %s compile: %s\n", label, err);
        failures++;
        free(ptx);
        return;
    }
    acuda_kernel_info_t info;
    CHECK_ST(acuda_kernel_info(k, &info));
    CHECK(info.num_ptr_params == 1);
    CHECK(info.push_bytes == 0);
    CHECK(info.num_params == 1);
    acuda_buffer_t buf = NULL;
    CHECK_ST(acuda_malloc(ctx, &buf, 8));
    uint32_t host[2] = { in0, in1 };
    CHECK_ST(acuda_memcpy_h2d(ctx, buf, host, sizeof(host), NULL));
    CHECK_ST(acuda_launch_kernel(ctx, k, &buf, 1, NULL, 0, 1, 1, NULL));
    CHECK_ST(acuda_stream_synchronize(ctx, NULL));
    uint32_t got = 0;
    CHECK_ST(acuda_memcpy_d2h(ctx, &got, buf, sizeof(got), NULL));
    if (got != want) {
        printf("FAIL: %s: got %u want %u\n", label, got, want);
        failures++;
    } else {
        printf("  %s: %u ok\n", label, got);
    }
    CHECK_ST(acuda_free(ctx, buf));
    acuda_kernel_free(k);
    free(ptx);
}

// One-pointer kernel with a word array in/out (atom, .b128 and warp
// tests). block_x selects the launch block size.
static void run_words_block(acuda_context_t ctx, const char *file,
                            const uint32_t *in, unsigned nwords,
                            const uint32_t *want, const char *label,
                            unsigned block_x) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", ACUDA_TEST_PTX_DIR, file);
    char *ptx = read_file(path);
    if (!ptx) {
        failures++;
        return;
    }
    char err[1024];
    acuda_kernel_t *k = NULL;
    if (acuda_compile_ptx(ptx, NULL, &k, err, sizeof(err), &g_test_caps) != ACUDA_SUCCESS) {
        printf("FAIL: %s compile: %s\n", label, err);
        failures++;
        free(ptx);
        return;
    }
    acuda_kernel_info_t info;
    CHECK_ST(acuda_kernel_info(k, &info));
    CHECK(info.num_ptr_params == 1);
    CHECK(info.push_bytes == 0);
    acuda_buffer_t buf = NULL;
    size_t bytes = (size_t)nwords * 4;
    CHECK_ST(acuda_malloc(ctx, &buf, bytes));
    CHECK_ST(acuda_memcpy_h2d(ctx, buf, in, bytes, NULL));
    CHECK_ST(acuda_launch_kernel(ctx, k, &buf, 1, NULL, 0, 1, block_x, NULL));
    CHECK_ST(acuda_stream_synchronize(ctx, NULL));
    uint32_t *got = malloc(bytes);
    CHECK(got != NULL);
    if (got) {
        CHECK_ST(acuda_memcpy_d2h(ctx, got, buf, bytes, NULL));
        unsigned bad = 0;
        for (unsigned i = 0; i < nwords; i++)
            if (got[i] != want[i]) {
                printf("FAIL: %s word %u: got 0x%x want 0x%x\n", label, i,
                       got[i], want[i]);
                bad++;
            }
        if (!bad)
            printf("  %s: %u words ok\n", label, nwords);
        failures += bad ? 1 : 0;
        free(got);
    }
    CHECK_ST(acuda_free(ctx, buf));
    acuda_kernel_free(k);
    free(ptx);
}

static void run_words(acuda_context_t ctx, const char *file,
                      const uint32_t *in, unsigned nwords,
                      const uint32_t *want, const char *label) {
    run_words_block(ctx, file, in, nwords, want, label, 1);
}

// True when the module contains an integer dot-product instruction
// (OpSDot/OpUDot/OpSUDot, SPV_KHR_integer_dot_product).
// Compile a file and return its uses_bda flag.
static int kernel_uses_bda(const char *file) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", ACUDA_TEST_PTX_DIR, file);
    char *ptx = read_file(path);
    if (!ptx)
        return -1;
    acuda_kernel_t *k = NULL;
    char err[1024];
    int uses = -1;
    if (acuda_compile_ptx(ptx, NULL, &k, err, sizeof(err), &g_test_caps) == ACUDA_SUCCESS) {
        acuda_kernel_info_t info;
        if (acuda_kernel_info(k, &info) == ACUDA_SUCCESS)
            uses = (int)info.bda;
        acuda_kernel_free(k);
    }
    free(ptx);
    return uses;
}

// Compile a file and return its uses_subgroup flag (warp-32 semantics).
static int kernel_uses_subgroup(const char *file) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", ACUDA_TEST_PTX_DIR, file);
    char *ptx = read_file(path);
    if (!ptx)
        return -1;
    acuda_kernel_t *k = NULL;
    char err[1024];
    int uses = -1;
    if (acuda_compile_ptx(ptx, NULL, &k, err, sizeof(err), &g_test_caps) == ACUDA_SUCCESS) {
        acuda_kernel_info_t info;
        if (acuda_kernel_info(k, &info) == ACUDA_SUCCESS)
            uses = (int)info.uses_subgroup;
        acuda_kernel_free(k);
    }
    free(ptx);
    return uses;
}

static int spv_has_dot_product(const uint32_t *w, size_t n) {
    for (size_t i = 5; i < n;) {
        uint32_t wc = w[i] >> 16;
        if (wc == 0)
            break;
        uint32_t op = w[i] & 0xffffu;
        if (op == 4450u || op == 4451u || op == 4452u)
            return 1;
        i += wc;
    }
    return 0;
}

static uint32_t tex_f32_bits(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    return x;
}

// Compile and run one image-texture kernel with procedurally generated
// texels (element i channel c = i*nchans+c+1; mip level L adds 1000*L).
// Compile and run one surface kernel; compares the kernel's output
// words and the surface contents after the dispatch.
static void run_surface(acuda_context_t ctx, const char *file,
                        const char *entry,
                        const acuda_image_desc *idesc,
                        const uint32_t *data, unsigned surf_words,
                        const uint32_t *want, unsigned nwords,
                        const uint32_t *want_surf, const char *label) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", ACUDA_TEST_PTX_DIR, file);
    char *ptx = read_file(path);
    if (!ptx) {
        failures++;
        return;
    }
    char err[2048] = {0};
    acuda_kernel_t *k = NULL;
    if (acuda_compile_ptx(ptx, entry, &k, err, sizeof(err), &g_test_caps) !=
        ACUDA_SUCCESS) {
        printf("FAIL: %s compile: %s\n", label, err);
        failures++;
        free(ptx);
        return;
    }
    acuda_buffer_t surf = NULL, out = NULL;
    int bad = 0;
    acuda_status_t st = acuda_surface_create(ctx, &surf, idesc, data);
    if (st != ACUDA_SUCCESS) {
        printf("FAIL: %s surface create (%d)\n", label, (int)st);
        failures++;
        goto done_surf;
    }
    CHECK_ST(acuda_malloc(ctx, &out, nwords * 4));
    acuda_buffer_t bufs[2] = { surf, out };
    st = acuda_launch_kernel(ctx, k, bufs, 2, NULL, 0, 1, 1, NULL);
    if (st != ACUDA_SUCCESS) {
        printf("FAIL: %s launch (%d)\n", label, (int)st);
        failures++;
        goto done_surf;
    }
    CHECK_ST(acuda_stream_synchronize(ctx, NULL));
    uint32_t *got = malloc(nwords * 4);
    CHECK_ST(acuda_memcpy_d2h(ctx, got, out, nwords * 4, NULL));
    for (unsigned i = 0; i < nwords; i++)
        if (got[i] != want[i]) {
            printf("FAIL: %s word %u: got 0x%x want 0x%x\n", label, i,
                   got[i], want[i]);
            bad++;
        }
    free(got);
    if (want_surf) {
        uint32_t *back = malloc(surf_words * 4);
        memset(back, 0, surf_words * 4);
        CHECK_ST(acuda_surface_download(ctx, surf, back));
        for (unsigned i = 0; i < surf_words; i++)
            if (back[i] != want_surf[i]) {
                printf("FAIL: %s surface[%u]: got 0x%x want 0x%x\n",
                       label, i, back[i], want_surf[i]);
                bad++;
            }
        free(back);
    }
    if (!bad)
        printf("  %s: %u+%u words ok\n", label, nwords, surf_words);
    failures += bad ? 1 : 0;
done_surf:
    if (out)
        acuda_free(ctx, out);
    if (surf)
        acuda_free(ctx, surf);
    acuda_kernel_free(k);
    free(ptx);
}

// Surface kernel with an explicit grid: many invocations contend on the
// same texels.  Only the final surface contents are checked.
static void run_surface_many(acuda_context_t ctx, const char *file,
                             const char *entry,
                             const acuda_image_desc *idesc,
                             const uint32_t *data, unsigned surf_words,
                             unsigned grid, unsigned block,
                             const uint32_t *want_surf, const char *label) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", ACUDA_TEST_PTX_DIR, file);
    char *ptx = read_file(path);
    if (!ptx) {
        failures++;
        return;
    }
    char err[2048] = {0};
    acuda_kernel_t *k = NULL;
    if (acuda_compile_ptx(ptx, entry, &k, err, sizeof(err), &g_test_caps) !=
        ACUDA_SUCCESS) {
        printf("FAIL: %s compile: %s\n", label, err);
        failures++;
        free(ptx);
        return;
    }
    acuda_buffer_t surf = NULL;
    acuda_status_t st = acuda_surface_create(ctx, &surf, idesc, data);
    if (st != ACUDA_SUCCESS) {
        printf("FAIL: %s surface create (%d)\n", label, (int)st);
        failures++;
        goto done;
    }
    acuda_buffer_t bufs[1] = { surf };
    st = acuda_launch_kernel(ctx, k, bufs, 1, NULL, 0, grid, block, NULL);
    if (st != ACUDA_SUCCESS) {
        printf("FAIL: %s launch (%d)\n", label, (int)st);
        failures++;
        goto done;
    }
    CHECK_ST(acuda_stream_synchronize(ctx, NULL));
    uint32_t *back = malloc(surf_words * 4);
    memset(back, 0, surf_words * 4);
    CHECK_ST(acuda_surface_download(ctx, surf, back));
    int bad = 0;
    for (unsigned i = 0; i < surf_words; i++)
        if (back[i] != want_surf[i]) {
            printf("FAIL: %s surface[%u]: got 0x%x want 0x%x\n", label,
                   i, back[i], want_surf[i]);
            bad++;
        }
    free(back);
    if (!bad)
        printf("  %s: %u surface words ok\n", label, surf_words);
    failures += bad ? 1 : 0;
done:
    if (surf)
        acuda_free(ctx, surf);
    acuda_kernel_free(k);
    free(ptx);
}

static void run_tex_image(acuda_context_t ctx, const char *file,
                          const char *entry,
                          const acuda_image_desc *idesc, int linear,
                          int compare, uint32_t cmp_op,
                          const uint32_t *want, unsigned nwords,
                          const char *label) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", ACUDA_TEST_PTX_DIR, file);
    char *ptx = read_file(path);
    if (!ptx) {
        failures++;
        return;
    }
    char err[2048] = {0};
    acuda_kernel_t *k = NULL;
    if (acuda_compile_ptx(ptx, entry, &k, err, sizeof(err), &g_test_caps) !=
        ACUDA_SUCCESS) {
        printf("FAIL: %s compile: %s\n", label, err);
        failures++;
        free(ptx);
        return;
    }
    int nchans = idesc->format == 3 ? 1 : 4;
    size_t elems;
    if (idesc->dim == 1)
        elems = (size_t)idesc->width * idesc->layers;
    else if (idesc->dim == 3)
        elems = (size_t)idesc->width * idesc->height * idesc->depth;
    else
        elems = (size_t)idesc->width * idesc->height * idesc->layers;
    uint32_t *texels = malloc(elems * nchans * 4);
    if (!texels) {
        failures++;
        acuda_kernel_free(k);
        free(ptx);
        return;
    }
    for (size_t i = 0; i < elems; i++)
        for (int c = 0; c < nchans; c++) {
            uint32_t v = (uint32_t)(i * nchans + c + 1);
            texels[i * nchans + c] =
                idesc->format == 3 ? v : tex_f32_bits((float)v);
        }
    acuda_buffer_t img = NULL, smp = NULL, tex = NULL, out = NULL;
    int bad = 0;
    acuda_status_t st = acuda_image_create(ctx, &img, idesc, texels);
    if (st != ACUDA_SUCCESS) {
        printf("FAIL: %s image create (%d)\n", label, (int)st);
        failures++;
        goto done_texels;
    }
    for (uint32_t lv = 1; lv < idesc->levels; lv++) {
        uint32_t w = idesc->width >> lv;
        uint32_t h = idesc->dim == 1 ? 1u : (idesc->height >> lv);
        uint32_t d = idesc->dim == 3 ? (idesc->depth >> lv) : 1u;
        size_t n = (size_t)(w ? w : 1) * (h ? h : 1) * (d ? d : 1);
        uint32_t *m = malloc(n * nchans * 4);
        for (size_t i = 0; i < n; i++)
            for (int c = 0; c < nchans; c++) {
                uint32_t v = (uint32_t)(1000 * (int)lv + i * nchans + c + 1);
                m[i * nchans + c] =
                    idesc->format == 3 ? v : tex_f32_bits((float)v);
            }
        acuda_image_upload(ctx, img, lv, 0, m);
        free(m);
    }
    acuda_sampler_desc sd = { 0 };
    sd.filter = linear ? 1 : 0;
    sd.addr[0] = sd.addr[1] = sd.addr[2] = 2;
    sd.normalized = 1;
    sd.compare = compare ? 1 : 0;
    sd.compare_op = cmp_op;
    CHECK_ST(acuda_sampler_create(ctx, &smp, &sd));
    CHECK_ST(acuda_texture_create_image(ctx, &tex, img, smp));
    CHECK_ST(acuda_malloc(ctx, &out, nwords * 4));
    acuda_kernel_info_t info;
    CHECK_ST(acuda_kernel_info(k, &info));
    acuda_buffer_t bufs[8];
    if (info.num_ptr_params > 8) {
        failures++;
        goto done;
    }
    for (uint32_t i = 0; i < info.num_ptr_params; i++) {
        if (info.binding_kind[i] == 3)
            bufs[i] = smp;
        else if (info.binding_kind[i] == 1)
            bufs[i] = tex;
        else
            bufs[i] = out;
    }
    st = acuda_launch_kernel(ctx, k, bufs, info.num_ptr_params, NULL, 0,
                             1, 1, NULL);
    if (st != ACUDA_SUCCESS) {
        printf("FAIL: %s launch (%d)\n", label, (int)st);
        failures++;
        goto done;
    }
    CHECK_ST(acuda_stream_synchronize(ctx, NULL));
    uint32_t *got = malloc(nwords * 4);
    CHECK_ST(acuda_memcpy_d2h(ctx, got, out, nwords * 4, NULL));
    for (unsigned i = 0; i < nwords; i++)
        if (got[i] != want[i]) {
            printf("FAIL: %s word %u: got 0x%x want 0x%x\n", label, i,
                   got[i], want[i]);
            bad++;
        }
    if (!bad)
        printf("  %s: %u words ok\n", label, nwords);
    failures += bad ? 1 : 0;
    free(got);
done:
    acuda_free(ctx, out);
    acuda_free(ctx, tex);
    acuda_free(ctx, smp);
    acuda_free(ctx, img);
    acuda_kernel_free(k);
done_texels:
    free(texels);
    free(ptx);
}

// Small non-negative floats -> binary16 bits (normal range only; the
// WMMA test uses exact small integers and 1.0).
static uint32_t f32_to_f16_bits(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int exp = (int)((x >> 23) & 0xffu) - 127;
    uint32_t man = x & 0x7fffffu;
    if ((x & 0x7fffffffu) == 0)
        return sign;
    return sign | ((uint32_t)(exp + 15) << 10) | (man >> 13);
}


static uint32_t f32_bits(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    return x;
}

static uint32_t f32_to_bf16_bits(float f) { return f32_bits(f) >> 16; }

// Build the 416-word buffer for tests/ptx/mma_sp{,bf16}.ptx. A keeps
// indices (0,2) in even chunks and (1,3) in odd chunks, so every
// metadata word is 0xD8D8D8D8; D = A*B + C.
static void build_sparse_mma(uint32_t *in, uint32_t *want, int bf16,
                             int alt) {
    float sa[16][16], sb[16][8], sc[16][8];
    static const int si0[4] = {0, 1, 0, 1};
    static const int si1[4] = {2, 3, 2, 3};
    static const int si1a[4] = {3, 2, 3, 2};
    const int *s1 = alt ? si1a : si1;
    memset(sa, 0, sizeof(sa));
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 4; c++)
            for (int q = 0; q < 2; q++) {
                int k = 4 * c + (q ? s1[c] : si0[c]);
                sa[r][k] = (float)(1 + ((r * 16 + k) % 7));
            }
    for (int k = 0; k < 16; k++)
        for (int c = 0; c < 8; c++)
            sb[k][c] = (float)(1 + ((k * 8 + c) % 5));
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 8; c++)
            sc[r][c] = (float)(1 + ((r * 8 + c) % 3));
    memset(in, 0, 416 * 4);
    for (int L = 0; L < 32; L++) {
        int g = L >> 2, t = L & 3;
        for (int hr = 0; hr < 2; hr++) {
            int r = g + 8 * hr;
            uint32_t v0 = bf16 ? f32_to_bf16_bits(sa[r][4 * t + si0[t]])
                               : f32_to_f16_bits(sa[r][4 * t + si0[t]]);
            uint32_t v1 = bf16 ? f32_to_bf16_bits(sa[r][4 * t + s1[t]])
                               : f32_to_f16_bits(sa[r][4 * t + s1[t]]);
            in[L * 2 + hr] = v0 | (v1 << 16);
        }
        in[64 + L] = alt ? 0x9c9c9c9cu : 0xd8d8d8d8u;
        uint32_t b00 = bf16 ? f32_to_bf16_bits(sb[2 * t][g])
                            : f32_to_f16_bits(sb[2 * t][g]);
        uint32_t b01 = bf16 ? f32_to_bf16_bits(sb[2 * t + 1][g])
                            : f32_to_f16_bits(sb[2 * t + 1][g]);
        uint32_t b10 = bf16 ? f32_to_bf16_bits(sb[2 * t + 8][g])
                            : f32_to_f16_bits(sb[2 * t + 8][g]);
        uint32_t b11 = bf16 ? f32_to_bf16_bits(sb[2 * t + 9][g])
                            : f32_to_f16_bits(sb[2 * t + 9][g]);
        in[96 + L * 2 + 0] = b00 | (b01 << 16);
        in[96 + L * 2 + 1] = b10 | (b11 << 16);
        for (int m = 0; m < 4; m++)
            in[160 + L * 4 + m] =
                f32_bits(sc[g + 8 * (m >= 2)][2 * t + (m & 1)]);
    }
    memcpy(want, in, 288 * 4);
    for (int L = 0; L < 32; L++) {
        int g = L >> 2, t = L & 3;
        for (int m = 0; m < 4; m++) {
            int r = g + 8 * (m >= 2);
            int c = 2 * t + (m & 1);
            float d = sc[r][c];
            for (int k = 0; k < 16; k++)
                d += sa[r][k] * sb[k][c];
            want[288 + L * 4 + m] = f32_bits(d);
        }
    }
}

// 192-word buffer for tests/ptx/dot2.ptx: A 2 words/lane, B 2, C 1,
// D 1; D = C + dot2(A, B) with f32 accumulation.
static void build_dot2(uint32_t *in, uint32_t *want) {
    memset(in, 0, 192 * 4);
    for (int L = 0; L < 32; L++) {
        float a0 = 1.0f + 0.25f * (float)(L % 3);
        float a1 = 0.5f * (float)(1 + L % 5);
        float b0 = -1.0f + 0.25f * (float)(L % 7);
        float b1 = 2.0f - 0.125f * (float)(L % 4);
        float c = 0.25f * (float)L;
        in[L * 2 + 0] = f32_to_f16_bits(a0) |
                        (f32_to_f16_bits(a1) << 16);
        in[64 + L * 2 + 0] = f32_to_f16_bits(b0) |
                             (f32_to_f16_bits(b1) << 16);
        in[128 + L] = f32_bits(c);
        in[160 + L] = f32_bits(c + a0 * b0 + a1 * b1);
    }
    memcpy(want, in, 192 * 4);
    memset(in + 160, 0, 32 * 4);
}

// Build the 352-word buffer for tests/ptx/mma_k8{,_bf16}.ptx:
// A 2 words/lane, B 1, C 4, D 4; D = A*B + C.
static void build_mma_k8(uint32_t *in, uint32_t *want, int bf16) {
    float sa[16][8], sb[8][8], sc[16][8];
    for (int r = 0; r < 16; r++)
        for (int k = 0; k < 8; k++)
            sa[r][k] = (float)(1 + ((r * 8 + k) % 7));
    for (int k = 0; k < 8; k++)
        for (int c = 0; c < 8; c++)
            sb[k][c] = (float)(1 + ((k * 8 + c) % 5));
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 8; c++)
            sc[r][c] = (float)(1 + ((r * 8 + c) % 3));
    memset(in, 0, 352 * 4);
    for (int L = 0; L < 32; L++) {
        int g = L >> 2, t = L & 3;
        uint32_t a00 = bf16 ? f32_to_bf16_bits(sa[g][2 * t])
                            : f32_to_f16_bits(sa[g][2 * t]);
        uint32_t a01 = bf16 ? f32_to_bf16_bits(sa[g][2 * t + 1])
                            : f32_to_f16_bits(sa[g][2 * t + 1]);
        uint32_t a10 = bf16 ? f32_to_bf16_bits(sa[g + 8][2 * t])
                            : f32_to_f16_bits(sa[g + 8][2 * t]);
        uint32_t a11 = bf16 ? f32_to_bf16_bits(sa[g + 8][2 * t + 1])
                            : f32_to_f16_bits(sa[g + 8][2 * t + 1]);
        in[L * 2 + 0] = a00 | (a01 << 16);
        in[L * 2 + 1] = a10 | (a11 << 16);
        uint32_t b0 = bf16 ? f32_to_bf16_bits(sb[2 * t][g])
                           : f32_to_f16_bits(sb[2 * t][g]);
        uint32_t b1 = bf16 ? f32_to_bf16_bits(sb[2 * t + 1][g])
                           : f32_to_f16_bits(sb[2 * t + 1][g]);
        in[64 + L] = b0 | (b1 << 16);
        for (int m = 0; m < 4; m++)
            in[96 + L * 4 + m] =
                f32_bits(sc[g + 8 * (m >= 2)][2 * t + (m & 1)]);
    }
    memcpy(want, in, 224 * 4);
    for (int L = 0; L < 32; L++) {
        int g = L >> 2, t = L & 3;
        for (int m = 0; m < 4; m++) {
            int r = g + 8 * (m >= 2);
            int c = 2 * t + (m & 1);
            float d = sc[r][c];
            for (int k = 0; k < 8; k++)
                d += sa[r][k] * sb[k][c];
            want[224 + L * 4 + m] = f32_bits(d);
        }
    }
}

// 448-word buffer for tests/ptx/mma_k16_bf16.ptx.
static void build_mma_k16_bf16(uint32_t *in, uint32_t *want) {
    float sa[16][16], sb[16][8], sc[16][8];
    for (int r = 0; r < 16; r++)
        for (int k = 0; k < 16; k++)
            sa[r][k] = (float)(1 + ((r * 16 + k) % 7));
    for (int k = 0; k < 16; k++)
        for (int c = 0; c < 8; c++)
            sb[k][c] = (float)(1 + ((k * 8 + c) % 5));
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 8; c++)
            sc[r][c] = (float)(1 + ((r * 8 + c) % 3));
    memset(in, 0, 448 * 4);
    for (int L = 0; L < 32; L++) {
        int g = L >> 2, t = L & 3;
        uint32_t a[4];
        a[0] = f32_to_bf16_bits(sa[g][2 * t]) |
               (f32_to_bf16_bits(sa[g][2 * t + 1]) << 16);
        a[1] = f32_to_bf16_bits(sa[g + 8][2 * t]) |
               (f32_to_bf16_bits(sa[g + 8][2 * t + 1]) << 16);
        a[2] = f32_to_bf16_bits(sa[g][2 * t + 8]) |
               (f32_to_bf16_bits(sa[g][2 * t + 9]) << 16);
        a[3] = f32_to_bf16_bits(sa[g + 8][2 * t + 8]) |
               (f32_to_bf16_bits(sa[g + 8][2 * t + 9]) << 16);
        for (int j = 0; j < 4; j++)
            in[L * 4 + j] = a[j];
        in[128 + L * 2 + 0] =
            f32_to_bf16_bits(sb[2 * t][g]) |
            (f32_to_bf16_bits(sb[2 * t + 1][g]) << 16);
        in[128 + L * 2 + 1] =
            f32_to_bf16_bits(sb[2 * t + 8][g]) |
            (f32_to_bf16_bits(sb[2 * t + 9][g]) << 16);
        for (int m = 0; m < 4; m++) {
            int r = g + 8 * (m >= 2);
            int c = 2 * t + (m & 1);
            in[192 + L * 4 + m] = f32_bits(sc[r][c]);
        }
    }
    memcpy(want, in, 320 * 4);
    for (int L = 0; L < 32; L++) {
        int g = L >> 2, t = L & 3;
        for (int m = 0; m < 4; m++) {
            int r = g + 8 * (m >= 2);
            int c = 2 * t + (m & 1);
            float d = sc[r][c];
            for (int k = 0; k < 16; k++)
                d += sa[r][k] * sb[k][c];
            want[320 + L * 4 + m] = f32_bits(d);
        }
    }
    memset(in + 320, 0, 128 * 4);
}

// Same buffer layout for tests/ptx/mma_k4.ptx (tf32 A/B).
static void build_mma_k4(uint32_t *in, uint32_t *want) {
    float sa[16][4], sb[4][8], sc[16][8];
    for (int r = 0; r < 16; r++)
        for (int k = 0; k < 4; k++)
            sa[r][k] = (float)(1 + ((r * 4 + k) % 7));
    for (int k = 0; k < 4; k++)
        for (int c = 0; c < 8; c++)
            sb[k][c] = (float)(1 + ((k * 8 + c) % 5));
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 8; c++)
            sc[r][c] = (float)(1 + ((r * 8 + c) % 3));
    memset(in, 0, 352 * 4);
    for (int L = 0; L < 32; L++) {
        int g = L >> 2, t = L & 3;
        in[L * 2 + 0] = f32_bits(sa[g][t]);
        in[L * 2 + 1] = f32_bits(sa[g + 8][t]);
        in[64 + L] = f32_bits(sb[t][g]);
        for (int m = 0; m < 4; m++)
            in[96 + L * 4 + m] =
                f32_bits(sc[g + 8 * (m >= 2)][2 * t + (m & 1)]);
    }
    memcpy(want, in, 224 * 4);
    for (int L = 0; L < 32; L++) {
        int g = L >> 2, t = L & 3;
        for (int m = 0; m < 4; m++) {
            int r = g + 8 * (m >= 2);
            int c = 2 * t + (m & 1);
            float d = sc[r][c];
            for (int k = 0; k < 4; k++)
                d += sa[r][k] * sb[k][c];
            want[224 + L * 4 + m] = f32_bits(d);
        }
    }
}

// 448-word buffer for tests/ptx/mma_k8_tf32.ptx.  A 4 words/lane,
// B 2, C 4, D 4.
static void build_mma_k8_tf32(uint32_t *in, uint32_t *want) {
    float sa[16][8], sb[8][8], sc[16][8];
    for (int r = 0; r < 16; r++)
        for (int k = 0; k < 8; k++)
            sa[r][k] = (float)(1 + ((r * 8 + k) % 7));
    for (int k = 0; k < 8; k++)
        for (int c = 0; c < 8; c++)
            sb[k][c] = (float)(1 + ((k * 8 + c) % 5));
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 8; c++)
            sc[r][c] = (float)(1 + ((r * 8 + c) % 3));
    memset(in, 0, 448 * 4);
    for (int L = 0; L < 32; L++) {
        int g = L >> 2, t = L & 3;
        in[L * 4 + 0] = f32_bits(sa[g][t]);
        in[L * 4 + 1] = f32_bits(sa[g + 8][t]);
        in[L * 4 + 2] = f32_bits(sa[g][t + 4]);
        in[L * 4 + 3] = f32_bits(sa[g + 8][t + 4]);
        in[128 + L * 2 + 0] = f32_bits(sb[t][g]);
        in[128 + L * 2 + 1] = f32_bits(sb[t + 4][g]);
        for (int m = 0; m < 4; m++)
            in[192 + L * 4 + m] =
                f32_bits(sc[g + 8 * (m >= 2)][2 * t + (m & 1)]);
    }
    memcpy(want, in, 320 * 4);
    for (int L = 0; L < 32; L++) {
        int g = L >> 2, t = L & 3;
        for (int m = 0; m < 4; m++) {
            int r = g + 8 * (m >= 2);
            int c = 2 * t + (m & 1);
            float d = sc[r][c];
            for (int k = 0; k < 8; k++)
                d += sa[r][k] * sb[k][c];
            want[320 + L * 4 + m] = f32_bits(d);
        }
    }
}

// Build a buffer for the integer WMMA kernels.  A and B are bytes
// (A row-major, B column-major); C and D are s32 words.
static void build_wmma_int8_shape(uint32_t *in, uint32_t *want, int rows,
                                  int cols, int kdim, int is_signed) {
    int abytes = rows * kdim;
    int bbytes = cols * kdim;
    int cwords = rows * cols;
    int aoff = 0;
    int boff = abytes;
    int coff = abytes + bbytes;
    int doff = coff + cwords * 4;
    int total_words = (doff + cwords * 4) / 4;
    uint8_t *bytes = (uint8_t *)in;
    int *C = malloc((size_t)cwords * sizeof(int));
    int *A = malloc((size_t)abytes * sizeof(int));
    int *B = malloc((size_t)bbytes * sizeof(int));
    if (!A || !B || !C) {
        free(A); free(B); free(C);
        memset(in, 0, (size_t)total_words * 4);
        memset(want, 0, (size_t)total_words * 4);
        return;
    }
    for (int r = 0; r < rows; r++)
        for (int k = 0; k < kdim; k++)
            A[r * kdim + k] = is_signed
                                  ? ((r * kdim + k) % 5) - 2
                                  : 1 + ((r * kdim + k) % 5);
    for (int k = 0; k < kdim; k++)
        for (int c = 0; c < cols; c++)
            B[c * kdim + k] = is_signed
                                  ? ((k * 16 + c) % 5) - 2
                                  : 1 + ((k * 16 + c) % 5);
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            C[r * cols + c] = (r * 16 + c) % 7;
    memset(in, 0, (size_t)total_words * 4);
    for (int i = 0; i < abytes; i++)
        bytes[aoff + i] = (uint8_t)(int8_t)A[i];
    for (int i = 0; i < bbytes; i++)
        bytes[boff + i] = (uint8_t)(int8_t)B[i];
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            in[coff / 4 + r * cols + c] = (uint32_t)C[r * cols + c];
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++) {
            int d = C[r * cols + c];
            for (int k = 0; k < kdim; k++)
                d += A[r * kdim + k] * B[c * kdim + k];
            in[doff / 4 + r * cols + c] = (uint32_t)d;
        }
    memcpy(want, in, (size_t)total_words * 4);
    memset(in + doff / 4, 0, (size_t)cwords * 4);
    free(A); free(B); free(C);
}

// 192-word buffer for tests/ptx/wmma_int4_*_m8n8k32.ptx.
static void build_wmma_int4_m8(uint32_t *in, uint32_t *want, int is_signed) {
    uint8_t *bytes = (uint8_t *)in;
    int A[8][32], B[8][32], C[8][8];
    for (int r = 0; r < 8; r++)
        for (int k = 0; k < 32; k++)
            A[r][k] = is_signed ? ((r * 32 + k) % 16) - 8
                                : (r * 32 + k) % 16;
    for (int c = 0; c < 8; c++)
        for (int k = 0; k < 32; k++)
            B[c][k] = is_signed ? ((k * 8 + c) % 16) - 8
                                : (k * 8 + c) % 16;
    for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++)
            C[r][c] = (r * 8 + c) % 7;
    memset(in, 0, 192 * 4);
    for (int r = 0; r < 8; r++)
        for (int k = 0; k < 32; k++) {
            int idx = r * 32 + k;
            bytes[idx >> 1] |= (uint8_t)(A[r][k] & 0xf)
                               << ((idx & 1) * 4);
        }
    for (int c = 0; c < 8; c++)
        for (int k = 0; k < 32; k++) {
            int idx = c * 32 + k;
            bytes[128 + (idx >> 1)] |= (uint8_t)(B[c][k] & 0xf)
                                      << ((idx & 1) * 4);
        }
    for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++)
            in[64 + r * 8 + c] = (uint32_t)C[r][c];
    for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++) {
            int d = C[r][c];
            for (int k = 0; k < 32; k++)
                d += A[r][k] * B[c][k];
            in[128 + r * 8 + c] = (uint32_t)d;
        }
    memcpy(want, in, 192 * 4);
    memset(in + 128, 0, 64 * 4);
}

static uint32_t pack_nibbles8(const int v[8]) {
    uint32_t w = 0;
    for (int i = 0; i < 8; i++)
        w |= ((uint32_t)v[i] & 0xfu) << (4 * i);
    return w;
}

// Dense integer m8n8k32 u4.
static void build_mma_int4_m8(uint32_t *in, uint32_t *want, int is_signed) {
    int A[8][32], B[32][8], C[8][8];
    for (int r = 0; r < 8; r++)
        for (int k = 0; k < 32; k++)
            A[r][k] = is_signed ? ((r * 32 + k) % 16) - 8
                                : (r * 32 + k) % 16;
    for (int k = 0; k < 32; k++)
        for (int c = 0; c < 8; c++)
            B[k][c] = is_signed ? ((k * 8 + c) % 16) - 8
                                : (k * 8 + c) % 16;
    for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++)
            C[r][c] = (r * 8 + c) % 7;
    memset(in, 0, 192 * 4);
    for (int L = 0; L < 32; L++) {
        int g = L >> 2, t = L & 3, vals[8];
        for (int i = 0; i < 8; i++)
            vals[i] = A[g][8 * t + i];
        in[L] = pack_nibbles8(vals);
        for (int i = 0; i < 8; i++)
            vals[i] = B[8 * t + i][g];
        in[32 + L] = pack_nibbles8(vals);
        for (int m = 0; m < 2; m++)
            in[64 + L * 2 + m] = (uint32_t)C[g][2 * t + m];
    }
    memcpy(want, in, 128 * 4);
    for (int L = 0; L < 32; L++) {
        int g = L >> 2, t = L & 3;
        for (int m = 0; m < 2; m++) {
            int r = g, c = 2 * t + m, d = C[r][c];
            for (int k = 0; k < 32; k++)
                d += A[r][k] * B[k][c];
            want[128 + L * 2 + m] = (uint32_t)d;
        }
    }
    memset(in + 128, 0, 64 * 4);
}

// Dense integer m16n8k32 u4.
static void build_mma_int4_m16k32(uint32_t *in, uint32_t *want,
                                  int is_signed) {
    int A[16][32], B[32][8], C[16][8];
    for (int r = 0; r < 16; r++)
        for (int k = 0; k < 32; k++)
            A[r][k] = is_signed ? ((r * 32 + k) % 16) - 8
                                : (r * 32 + k) % 16;
    for (int k = 0; k < 32; k++)
        for (int c = 0; c < 8; c++)
            B[k][c] = is_signed ? ((k * 8 + c) % 16) - 8
                                : (k * 8 + c) % 16;
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 8; c++)
            C[r][c] = (r * 8 + c) % 7;
    memset(in, 0, 352 * 4);
    for (int L = 0; L < 32; L++) {
        int g = L >> 2, t = L & 3, vals[8];
        for (int i = 0; i < 8; i++)
            vals[i] = A[g][8 * t + i];
        in[L * 2 + 0] = pack_nibbles8(vals);
        for (int i = 0; i < 8; i++)
            vals[i] = A[g + 8][8 * t + i];
        in[L * 2 + 1] = pack_nibbles8(vals);
        for (int i = 0; i < 8; i++)
            vals[i] = B[8 * t + i][g];
        in[64 + L] = pack_nibbles8(vals);
        for (int m = 0; m < 4; m++)
            in[96 + L * 4 + m] =
                (uint32_t)C[g + 8 * (m >= 2)][2 * t + (m & 1)];
    }
    memcpy(want, in, 224 * 4);
    for (int L = 0; L < 32; L++) {
        int g = L >> 2, t = L & 3;
        for (int m = 0; m < 4; m++) {
            int r = g + 8 * (m >= 2), c = 2 * t + (m & 1);
            int d = C[r][c];
            for (int k = 0; k < 32; k++)
                d += A[r][k] * B[k][c];
            want[224 + L * 4 + m] = (uint32_t)d;
        }
    }
    memset(in + 224, 0, 128 * 4);
}

// Dense integer m16n8k64 u4.
static void build_mma_int4_m16k64(uint32_t *in, uint32_t *want,
                                  int is_signed) {
    int A[16][64], B[64][8], C[16][8];
    for (int r = 0; r < 16; r++)
        for (int k = 0; k < 64; k++)
            A[r][k] = is_signed ? ((r * 64 + k) % 16) - 8
                                : (r * 64 + k) % 16;
    for (int k = 0; k < 64; k++)
        for (int c = 0; c < 8; c++)
            B[k][c] = is_signed ? ((k * 8 + c) % 16) - 8
                                : (k * 8 + c) % 16;
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 8; c++)
            C[r][c] = (r * 8 + c) % 7;
    memset(in, 0, 448 * 4);
    for (int L = 0; L < 32; L++) {
        int g = L >> 2, t = L & 3, vals[8];
        for (int i = 0; i < 8; i++)
            vals[i] = A[g][8 * t + i];
        in[L * 4 + 0] = pack_nibbles8(vals);
        for (int i = 0; i < 8; i++)
            vals[i] = A[g + 8][8 * t + i];
        in[L * 4 + 1] = pack_nibbles8(vals);
        for (int i = 0; i < 8; i++)
            vals[i] = A[g][32 + 8 * t + i];
        in[L * 4 + 2] = pack_nibbles8(vals);
        for (int i = 0; i < 8; i++)
            vals[i] = A[g + 8][32 + 8 * t + i];
        in[L * 4 + 3] = pack_nibbles8(vals);
        for (int i = 0; i < 8; i++)
            vals[i] = B[8 * t + i][g];
        in[128 + L * 2 + 0] = pack_nibbles8(vals);
        for (int i = 0; i < 8; i++)
            vals[i] = B[32 + 8 * t + i][g];
        in[128 + L * 2 + 1] = pack_nibbles8(vals);
        for (int m = 0; m < 4; m++)
            in[192 + L * 4 + m] =
                (uint32_t)C[g + 8 * (m >= 2)][2 * t + (m & 1)];
    }
    memcpy(want, in, 320 * 4);
    for (int L = 0; L < 32; L++) {
        int g = L >> 2, t = L & 3;
        for (int m = 0; m < 4; m++) {
            int r = g + 8 * (m >= 2), c = 2 * t + (m & 1);
            int d = C[r][c];
            for (int k = 0; k < 64; k++)
                d += A[r][k] * B[k][c];
            want[320 + L * 4 + m] = (uint32_t)d;
        }
    }
    memset(in + 320, 0, 128 * 4);
}

// Dense single-bit MMA test buffers.
static void build_mma_b1_m8(uint32_t *in, uint32_t *want) {
    int A[8][128], B[128][8], C[8][8];
    for (int r = 0; r < 8; r++)
        for (int k = 0; k < 128; k++)
            A[r][k] = ((r * 7 + k * 3) % 5) < 2;
    for (int k = 0; k < 128; k++)
        for (int c = 0; c < 8; c++)
            B[k][c] = ((k * 7 + c * 5) % 5) < 2;
    for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++)
            C[r][c] = (r * 8 + c) % 7;
    memset(in, 0, 192 * 4);
    for (int L = 0; L < 32; L++) {
        int g = L >> 2, t = L & 3;
        uint32_t aw = 0, bw = 0;
        for (int i = 0; i < 32; i++) {
            aw |= (uint32_t)A[g][32 * t + i] << i;
            bw |= (uint32_t)B[32 * t + i][g] << i;
        }
        in[L] = aw;
        in[32 + L] = bw;
        for (int m = 0; m < 2; m++)
            in[64 + L * 2 + m] = (uint32_t)C[g][2 * t + m];
    }
    memcpy(want, in, 128 * 4);
    for (int L = 0; L < 32; L++) {
        int g = L >> 2, t = L & 3;
        for (int m = 0; m < 2; m++) {
            int r = g, c = 2 * t + m, pc = C[r][c];
            for (int k = 0; k < 128; k++)
                pc += A[r][k] & B[k][c];
            want[128 + L * 2 + m] = (uint32_t)pc;
        }
    }
    memset(in + 128, 0, 64 * 4);
}

static void build_mma_b1_m16k128(uint32_t *in, uint32_t *want,
                                 int is_xor) {
    int A[16][128], B[128][8], C[16][8];
    for (int r = 0; r < 16; r++)
        for (int k = 0; k < 128; k++)
            A[r][k] = ((r * 7 + k * 3) % 5) < 2;
    for (int k = 0; k < 128; k++)
        for (int c = 0; c < 8; c++)
            B[k][c] = ((k * 7 + c * 5) % 5) < 2;
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 8; c++)
            C[r][c] = (r * 8 + c) % 7;
    memset(in, 0, 352 * 4);
    for (int L = 0; L < 32; L++) {
        int g = L >> 2, t = L & 3;
        uint32_t aw0 = 0, aw1 = 0, bw = 0;
        for (int i = 0; i < 32; i++) {
            aw0 |= (uint32_t)A[g][32 * t + i] << i;
            aw1 |= (uint32_t)A[g + 8][32 * t + i] << i;
            bw |= (uint32_t)B[32 * t + i][g] << i;
        }
        in[L * 2 + 0] = aw0;
        in[L * 2 + 1] = aw1;
        in[64 + L] = bw;
        for (int m = 0; m < 4; m++)
            in[96 + L * 4 + m] =
                (uint32_t)C[g + 8 * (m >= 2)][2 * t + (m & 1)];
    }
    memcpy(want, in, 224 * 4);
    for (int L = 0; L < 32; L++) {
        int g = L >> 2, t = L & 3;
        for (int m = 0; m < 4; m++) {
            int r = g + 8 * (m >= 2), c = 2 * t + (m & 1);
            int pc = C[r][c];
            for (int k = 0; k < 128; k++)
                pc += is_xor ? (A[r][k] ^ B[k][c])
                             : (A[r][k] & B[k][c]);
            want[224 + L * 4 + m] = (uint32_t)pc;
        }
    }
    memset(in + 224, 0, 128 * 4);
}

static void build_mma_b1_m16k256(uint32_t *in, uint32_t *want) {
    int A[16][256], B[256][8], C[16][8];
    for (int r = 0; r < 16; r++)
        for (int k = 0; k < 256; k++)
            A[r][k] = ((r * 7 + k * 3) % 5) < 2;
    for (int k = 0; k < 256; k++)
        for (int c = 0; c < 8; c++)
            B[k][c] = ((k * 7 + c * 5) % 5) < 2;
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 8; c++)
            C[r][c] = (r * 8 + c) % 7;
    memset(in, 0, 448 * 4);
    for (int L = 0; L < 32; L++) {
        int g = L >> 2, t = L & 3;
        uint32_t aw[4] = {0, 0, 0, 0}, bw[2] = {0, 0};
        for (int i = 0; i < 32; i++) {
            aw[0] |= (uint32_t)A[g][32 * t + i] << i;
            aw[1] |= (uint32_t)A[g + 8][32 * t + i] << i;
            aw[2] |= (uint32_t)A[g][128 + 32 * t + i] << i;
            aw[3] |= (uint32_t)A[g + 8][128 + 32 * t + i] << i;
            bw[0] |= (uint32_t)B[32 * t + i][g] << i;
            bw[1] |= (uint32_t)B[128 + 32 * t + i][g] << i;
        }
        for (int j = 0; j < 4; j++)
            in[L * 4 + j] = aw[j];
        in[128 + L * 2 + 0] = bw[0];
        in[128 + L * 2 + 1] = bw[1];
        for (int m = 0; m < 4; m++)
            in[192 + L * 4 + m] =
                (uint32_t)C[g + 8 * (m >= 2)][2 * t + (m & 1)];
    }
    memcpy(want, in, 320 * 4);
    for (int L = 0; L < 32; L++) {
        int g = L >> 2, t = L & 3;
        for (int m = 0; m < 4; m++) {
            int r = g + 8 * (m >= 2), c = 2 * t + (m & 1);
            int pc = C[r][c];
            for (int k = 0; k < 256; k++)
                pc += A[r][k] & B[k][c];
            want[320 + L * 4 + m] = (uint32_t)pc;
        }
    }
    memset(in + 320, 0, 128 * 4);
}

// 192-word buffer for tests/ptx/wmma_b1_m8n8k128_{and,xor}.ptx.
static void build_wmma_b1_m8(uint32_t *in, uint32_t *want, int is_xor) {
    uint8_t *bytes = (uint8_t *)in;
    int A[8][128], B[8][128], C[8][8];
    for (int r = 0; r < 8; r++)
        for (int k = 0; k < 128; k++)
            A[r][k] = ((r * 7 + k * 3) % 5) < 2;
    for (int c = 0; c < 8; c++)
        for (int k = 0; k < 128; k++)
            B[c][k] = ((c * 5 + k * 7) % 5) < 2;
    for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++)
            C[r][c] = (r * 8 + c) % 7;
    memset(in, 0, 192 * 4);
    for (int r = 0; r < 8; r++)
        for (int k = 0; k < 128; k++)
            if (A[r][k])
                bytes[r * 16 + k / 8] |= (uint8_t)(1u << (k & 7));
    for (int c = 0; c < 8; c++)
        for (int k = 0; k < 128; k++)
            if (B[c][k])
                bytes[128 + c * 16 + k / 8] |= (uint8_t)(1u << (k & 7));
    for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++)
            in[64 + r * 8 + c] = (uint32_t)C[r][c];
    for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++) {
            int d = C[r][c];
            for (int k = 0; k < 128; k++)
                d += (is_xor ? (A[r][k] ^ B[c][k])
                             : (A[r][k] & B[c][k]));
            in[128 + r * 8 + c] = (uint32_t)d;
        }
    memcpy(want, in, 192 * 4);
    memset(in + 128, 0, 64 * 4);
}

// Pack four byte values into a little-endian word (s8/u8 fragments).
static uint32_t pack_bytes4(const int *v) {
    uint32_t w = 0;
    for (int i = 0; i < 4; i++)
        w |= ((uint32_t)v[i] & 0xffu) << (8 * i);
    return w;
}

// 448-word buffer for tests/ptx/mma_{u8,s8u8,u8s8}.ptx: A 4 words/lane,
// B 2, C 4, D 4; D = A*B + C with per-side signedness.
static void build_mma_int8(uint32_t *in, uint32_t *want, int asign,
                           int bsign) {
    int sa[16][32], sb[32][8], sc[16][8];
    for (int r = 0; r < 16; r++)
        for (int k = 0; k < 32; k++)
            sa[r][k] = asign ? ((r * 32 + k) % 255) - 128
                             : (r * 32 + k) % 256;
    for (int k = 0; k < 32; k++)
        for (int c = 0; c < 8; c++)
            sb[k][c] = bsign ? ((k * 8 + c) % 255) - 128
                             : (k * 8 + c) % 256;
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 8; c++)
            sc[r][c] = (r * 8 + c) % 7;
    memset(in, 0, 448 * 4);
    for (int L = 0; L < 32; L++) {
        int g = L >> 2, t = L & 3, c = L >> 2, q = L & 3;
        int v0[4], v1[4], v2[4], v3[4], w0[4], w1[4];
        for (int i = 0; i < 4; i++) {
            v0[i] = sa[g][4 * t + i];
            v1[i] = sa[g + 8][4 * t + i];
            v2[i] = sa[g][16 + 4 * t + i];
            v3[i] = sa[g + 8][16 + 4 * t + i];
            w0[i] = sb[4 * q + i][c];
            w1[i] = sb[16 + 4 * q + i][c];
        }
        in[L * 4 + 0] = pack_bytes4(v0);
        in[L * 4 + 1] = pack_bytes4(v1);
        in[L * 4 + 2] = pack_bytes4(v2);
        in[L * 4 + 3] = pack_bytes4(v3);
        in[128 + L * 2 + 0] = pack_bytes4(w0);
        in[128 + L * 2 + 1] = pack_bytes4(w1);
        for (int m = 0; m < 4; m++)
            in[192 + L * 4 + m] =
                (uint32_t)sc[g + 8 * (m >= 2)][2 * t + (m & 1)];
    }
    memcpy(want, in, 320 * 4);
    for (int L = 0; L < 32; L++) {
        int g = L >> 2, t = L & 3;
        for (int m = 0; m < 4; m++) {
            int r = g + 8 * (m >= 2);
            int c = 2 * t + (m & 1);
            int d = sc[r][c];
            for (int k = 0; k < 32; k++)
                d += sa[r][k] * sb[k][c];
            want[320 + L * 4 + m] = (uint32_t)d;
        }
    }
}

// FP8 e4m3/e5m2 encoder for small non-negative integers (exact in the
// target format, so the CPU oracle can use plain f32 arithmetic).
static int fp8_e4m3_bits(float v) {
    int sign = v < 0.0f;
    if (sign) v = -v;
    if (v == 0.0f) return 0;
    int e = 0;
    while ((float)(1 << (e + 1)) <= v && e < 8) e++;
    int m = (int)((v / (float)(1 << e) - 1.0f) * 8.0f + 0.5f);
    return (sign << 7) | ((e + 7) << 3) | (m & 7);
}

static int fp8_e5m2_bits(float v) {
    int sign = v < 0.0f;
    if (sign) v = -v;
    if (v == 0.0f) return 0;
    int e = 0;
    while ((float)(1 << (e + 1)) <= v && e < 15) e++;
    int m = (int)((v / (float)(1 << e) - 1.0f) * 4.0f + 0.5f);
    return (sign << 7) | ((e + 15) << 2) | (m & 3);
}

// 448-word buffer for tests/ptx/mma_fp8_*.ptx.  Same fragment layout as
// the m16n8k32 int8 form: A 4 words/lane, B 2, C 4, D 4.
static void build_mma_fp8(uint32_t *in, uint32_t *want, int e5) {
    int lim = e5 ? 4 : 16;
    float sa[16][32], sb[32][8], sc[16][8];
    for (int r = 0; r < 16; r++)
        for (int k = 0; k < 32; k++)
            sa[r][k] = (float)((r * 32 + k) % lim);
    for (int k = 0; k < 32; k++)
        for (int c = 0; c < 8; c++)
            sb[k][c] = (float)((k * 8 + c) % lim);
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 8; c++)
            sc[r][c] = (float)((r * 8 + c) % 7);
    memset(in, 0, 448 * 4);
    for (int L = 0; L < 32; L++) {
        int g = L >> 2, t = L & 3, c = L >> 2, q = L & 3;
        int v0[4], v1[4], v2[4], v3[4], w0[4], w1[4];
        for (int i = 0; i < 4; i++) {
            v0[i] = e5 ? fp8_e5m2_bits(sa[g][4 * t + i])
                       : fp8_e4m3_bits(sa[g][4 * t + i]);
            v1[i] = e5 ? fp8_e5m2_bits(sa[g + 8][4 * t + i])
                       : fp8_e4m3_bits(sa[g + 8][4 * t + i]);
            v2[i] = e5 ? fp8_e5m2_bits(sa[g][16 + 4 * t + i])
                       : fp8_e4m3_bits(sa[g][16 + 4 * t + i]);
            v3[i] = e5 ? fp8_e5m2_bits(sa[g + 8][16 + 4 * t + i])
                       : fp8_e4m3_bits(sa[g + 8][16 + 4 * t + i]);
            w0[i] = e5 ? fp8_e5m2_bits(sb[4 * q + i][c])
                       : fp8_e4m3_bits(sb[4 * q + i][c]);
            w1[i] = e5 ? fp8_e5m2_bits(sb[16 + 4 * q + i][c])
                       : fp8_e4m3_bits(sb[16 + 4 * q + i][c]);
        }
        in[L * 4 + 0] = pack_bytes4(v0);
        in[L * 4 + 1] = pack_bytes4(v1);
        in[L * 4 + 2] = pack_bytes4(v2);
        in[L * 4 + 3] = pack_bytes4(v3);
        in[128 + L * 2 + 0] = pack_bytes4(w0);
        in[128 + L * 2 + 1] = pack_bytes4(w1);
        for (int m = 0; m < 4; m++)
            in[192 + L * 4 + m] =
                f32_bits(sc[g + 8 * (m >= 2)][2 * t + (m & 1)]);
    }
    memcpy(want, in, 320 * 4);
    for (int L = 0; L < 32; L++) {
        int g = L >> 2, t = L & 3;
        for (int m = 0; m < 4; m++) {
            int r = g + 8 * (m >= 2);
            int c = 2 * t + (m & 1);
            float d = sc[r][c];
            for (int k = 0; k < 32; k++)
                d += sa[r][k] * sb[k][c];
            want[320 + L * 4 + m] = f32_bits(d);
        }
    }
}

// Integer m16n8k16 s8/u8: fragment maps from the PTX ISA (A 2 regs,
// B 1 reg, C/D 4 regs).  The kernel masks each source byte to 8 bits.
static void build_mma_k16_s8(uint32_t *in, uint32_t *want, int is_signed) {
    int a[16][16], b[16][8], c[16][8];
    memset(a, 0, sizeof(a));
    memset(b, 0, sizeof(b));
    memset(c, 0, sizeof(c));
    for (int L = 0; L < 32; L++) {
        int g = L >> 2, t = L & 3;
        for (int i = 0; i < 8; i++) {
            int v = (16 * L + i) & 0xff;
            a[g + 8 * (i >> 2)][4 * t + (i & 3)] =
                (is_signed && v >= 128) ? v - 256 : v;
        }
        for (int i = 0; i < 4; i++) {
            int v = (16 * L + i) & 0xff;
            b[4 * t + i][g] =
                (is_signed && v >= 128) ? v - 256 : v;
        }
        for (int m = 0; m < 4; m++)
            c[g + 8 * (m >= 2)][2 * t + (m & 1)] = (16 * L + m) & 0xff;
    }
    memset(in, 0, 128 * 4);
    for (int L = 0; L < 32; L++) {
        int g = L >> 2, t = L & 3;
        for (int m = 0; m < 4; m++) {
            int r = g + 8 * (m >= 2), cc = 2 * t + (m & 1);
            int d = c[r][cc];
            for (int k = 0; k < 16; k++)
                d += a[r][k] * b[k][cc];
            want[L * 4 + m] = (uint32_t)d;
        }
    }
}

// Integer m8n8k16 s8/u8: A 1 reg, B 1 reg, C/D 2 regs.
static void build_mma_m8s8_k16(uint32_t *in, uint32_t *want, int is_signed) {
    int a[8][16], b[16][8], c[8][8];
    memset(a, 0, sizeof(a));
    memset(b, 0, sizeof(b));
    memset(c, 0, sizeof(c));
    for (int L = 0; L < 32; L++) {
        int g = L >> 2, t = L & 3;
        for (int i = 0; i < 4; i++) {
            int v = (16 * L + i) & 0xff;
            a[g][4 * t + i] =
                (is_signed && v >= 128) ? v - 256 : v;
            b[4 * t + i][g] =
                (is_signed && v >= 128) ? v - 256 : v;
        }
        for (int i = 0; i < 2; i++)
            c[g][2 * t + i] = (16 * L + i) & 0xff;
    }
    memset(in, 0, 64 * 4);
    for (int L = 0; L < 32; L++) {
        int g = L >> 2, t = L & 3;
        for (int i = 0; i < 2; i++) {
            int cc = 2 * t + i;
            int d = c[g][cc];
            for (int k = 0; k < 16; k++)
                d += a[g][k] * b[k][cc];
            want[L * 2 + i] = (uint32_t)d;
        }
    }
}

static uint32_t pack_f16_pair(float a, float b) {
    return f32_to_f16_bits(a) | (f32_to_f16_bits(b) << 16);
}

// Buffer for tests/ptx/mma_f16acc.ptx (k8=0, 320 words) and
// tests/ptx/mma_k8_f16acc.ptx (k8=1, 224 words). D = A*B + C with
// integers small enough that every f16 partial sum is exact.
static void build_mma_f16acc(uint32_t *in, uint32_t *want, int k8) {
    int K = k8 ? 8 : 16;
    int aob = 0; // A
    int bob = k8 ? 64 : 128;
    int cob = k8 ? 96 : 192;
    int dob = k8 ? 160 : 256;
    int nwords = k8 ? 224 : 320;
    int A[16][16], B[16][8], C[16][8];
    for (int r = 0; r < 16; r++)
        for (int k = 0; k < K; k++)
            A[r][k] = 1 + ((r * K + k) % 4);
    for (int k = 0; k < K; k++)
        for (int c = 0; c < 8; c++)
            B[k][c] = 1 + ((k * 8 + c) % 3);
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 8; c++)
            C[r][c] = (r * 8 + c) % 5;
    memset(in, 0, (size_t)nwords * 4);
    for (int L = 0; L < 32; L++) {
        int g = L >> 2, t = L & 3;
        if (k8) {
            in[aob + L * 2 + 0] = pack_f16_pair(A[g][2 * t],
                                                A[g][2 * t + 1]);
            in[aob + L * 2 + 1] = pack_f16_pair(A[g + 8][2 * t],
                                                A[g + 8][2 * t + 1]);
            in[bob + L] = pack_f16_pair(B[2 * t][g], B[2 * t + 1][g]);
        } else {
            in[aob + L * 4 + 0] = pack_f16_pair(A[g][2 * t],
                                                A[g][2 * t + 1]);
            in[aob + L * 4 + 1] = pack_f16_pair(A[g + 8][2 * t],
                                                A[g + 8][2 * t + 1]);
            in[aob + L * 4 + 2] = pack_f16_pair(A[g][8 + 2 * t],
                                                A[g][8 + 2 * t + 1]);
            in[aob + L * 4 + 3] = pack_f16_pair(A[g + 8][8 + 2 * t],
                                                A[g + 8][8 + 2 * t + 1]);
            in[bob + L * 2 + 0] = pack_f16_pair(B[2 * t][g],
                                                B[2 * t + 1][g]);
            in[bob + L * 2 + 1] = pack_f16_pair(B[2 * t + 8][g],
                                                B[2 * t + 9][g]);
        }
        in[cob + L * 2 + 0] = pack_f16_pair(C[g][2 * t],
                                           C[g][2 * t + 1]);
        in[cob + L * 2 + 1] = pack_f16_pair(C[g + 8][2 * t],
                                           C[g + 8][2 * t + 1]);
    }
    memcpy(want, in, (size_t)dob * 4);
    for (int L = 0; L < 32; L++) {
        int g = L >> 2, t = L & 3;
        int d0 = C[g][2 * t], d1 = C[g][2 * t + 1];
        int d2 = C[g + 8][2 * t], d3 = C[g + 8][2 * t + 1];
        for (int k = 0; k < K; k++) {
            d0 += A[g][k] * B[k][2 * t];
            d1 += A[g][k] * B[k][2 * t + 1];
            d2 += A[g + 8][k] * B[k][2 * t];
            d3 += A[g + 8][k] * B[k][2 * t + 1];
        }
        want[dob + L * 2 + 0] =
            pack_f16_pair((float)d0, (float)d1);
        want[dob + L * 2 + 1] =
            pack_f16_pair((float)d2, (float)d3);
    }
}

// 384-word buffer for tests/ptx/mma_f64.ptx (m8n8k4 f64).
static void build_mma_f64(uint32_t *in, uint32_t *want) {
    double A[8][4], B[4][8], C[8][8];
    for (int r = 0; r < 8; r++)
        for (int k = 0; k < 4; k++)
            A[r][k] = (double)(1 + ((r * 4 + k) % 5));
    for (int k = 0; k < 4; k++)
        for (int c = 0; c < 8; c++)
            B[k][c] = (double)(1 + ((k * 8 + c) % 3));
    for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++)
            C[r][c] = (double)((r * 8 + c) % 7);
    memset(in, 0, 384 * 4);
    for (int L = 0; L < 32; L++) {
        int g = L >> 2, t = L & 3;
        uint64_t a, b;
        memcpy(&a, &A[g][t], 8);
        memcpy(&b, &B[t][g], 8);
        in[L * 2 + 0] = (uint32_t)a;
        in[L * 2 + 1] = (uint32_t)(a >> 32);
        in[64 + L * 2 + 0] = (uint32_t)b;
        in[64 + L * 2 + 1] = (uint32_t)(b >> 32);
        for (int j = 0; j < 2; j++) {
            uint64_t cv, dv;
            memcpy(&cv, &C[g][2 * t + j], 8);
            double d = C[g][2 * t + j];
            for (int k = 0; k < 4; k++)
                d += A[g][k] * B[k][2 * t + j];
            memcpy(&dv, &d, 8);
            in[128 + L * 4 + j * 2 + 0] = (uint32_t)cv;
            in[128 + L * 4 + j * 2 + 1] = (uint32_t)(cv >> 32);
            in[256 + L * 4 + j * 2 + 0] = (uint32_t)dv;
            in[256 + L * 4 + j * 2 + 1] = (uint32_t)(dv >> 32);
        }
    }
    memcpy(want, in, 256 * 4);
    for (int L = 0; L < 32; L++) {
        int g = L >> 2, t = L & 3;
        for (int j = 0; j < 2; j++) {
            double d = C[g][2 * t + j];
            for (int k = 0; k < 4; k++)
                d += A[g][k] * B[k][2 * t + j];
            uint64_t dv;
            memcpy(&dv, &d, 8);
            want[256 + L * 4 + j * 2 + 0] = (uint32_t)dv;
            want[256 + L * 4 + j * 2 + 1] = (uint32_t)(dv >> 32);
        }
    }
}

// 768-word buffer for tests/ptx/wmma_bf16.ptx (A=I, B[k][c]=16k+c,
// C[r][c]=16r+c, so D[r][c]=2*(16r+c)).
static void build_wmma_bf16(uint32_t *in, uint32_t *want) {
    memset(in, 0, 768 * 4);
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 16; c++) {
            int ai = r * 16 + c;
            uint32_t ah = (r == c) ? f32_to_bf16_bits(1.0f) : 0;
            in[ai >> 1] |= ah << ((ai & 1) * 16);
        }
    for (int c = 0; c < 16; c++)
        for (int k = 0; k < 16; k++) {
            int bi = c * 16 + k;
            uint32_t bh = f32_to_bf16_bits((float)(16 * k + c));
            in[128 + (bi >> 1)] |= bh << ((bi & 1) * 16);
        }
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 16; c++) {
            in[256 + r * 16 + c] = f32_bits((float)(16 * r + c));
            in[512 + r * 16 + c] =
                f32_bits((float)(2 * (16 * r + c)));
        }
    memcpy(want, in, 512 * 4);
    memcpy(want + 512, in + 512, 256 * 4);
}

// 512-word buffer for tests/ptx/wmma_f16acc.ptx (f16 accumulate).
static void build_wmma_f16acc(uint32_t *in, uint32_t *want) {
    memset(in, 0, 512 * 4);
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 16; c++) {
            int ai = r * 16 + c;
            uint32_t ah = (r == c) ? f32_to_f16_bits(1.0f) : 0;
            in[ai >> 1] |= ah << ((ai & 1) * 16);
        }
    for (int c = 0; c < 16; c++)
        for (int k = 0; k < 16; k++) {
            int bi = c * 16 + k;
            uint32_t bh = f32_to_f16_bits((float)(16 * k + c));
            in[128 + (bi >> 1)] |= bh << ((bi & 1) * 16);
        }
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 16; c++) {
            int i = r * 16 + c;
            uint32_t cv = f32_to_f16_bits((float)(16 * r + c));
            uint32_t dv = f32_to_f16_bits((float)(2 * (16 * r + c)));
            in[256 + (i >> 1)] |= cv << ((i & 1) * 16);
            in[384 + (i >> 1)] |= dv << ((i & 1) * 16);
        }
    memcpy(want, in, 512 * 4);
    memset(in + 384, 0, 128 * 4);
}

// 832-word buffer for the wmma m8n32k16/m32n8k16 tests.
static void build_wmma_shape(uint32_t *in, uint32_t *want, int rows, int cols,
                            int kdim, int bf16) {
    int am = rows * kdim, bn = cols * kdim;
    int abase = 0, bbase = am / 2, cbase = (am + bn) / 2;
    memset(in, 0, 832 * 4);
    for (int r = 0; r < rows; r++)
        for (int k = 0; k < kdim; k++) {
            int i = r * kdim + k;
            uint32_t h = bf16
                             ? f32_to_bf16_bits(
                                   (float)(1 + ((r * kdim + k) % 5)))
                             : f32_to_f16_bits(
                                   (float)(1 + ((r * kdim + k) % 5)));
            in[abase + (i >> 1)] |= h << ((i & 1) * 16);
        }
    for (int c = 0; c < cols; c++)
        for (int k = 0; k < kdim; k++) {
            int i = c * kdim + k;
            uint32_t h = bf16
                             ? f32_to_bf16_bits(
                                   (float)(1 + ((k * 8 + c) % 3)))
                             : f32_to_f16_bits(
                                   (float)(1 + ((k * 8 + c) % 3)));
            in[bbase + (i >> 1)] |= h << ((i & 1) * 16);
        }
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++) {
            int i = r * cols + c;
            float d = (float)((r * 8 + c) % 7);
            in[cbase + i] = f32_bits(d);
            for (int k = 0; k < kdim; k++)
                d += (float)((1 + ((r * kdim + k) % 5)) *
                             (1 + ((k * 8 + c) % 3)));
            in[cbase + rows * cols + i] = f32_bits(d);
        }
    memcpy(want, in, 832 * 4);
    memset(in + cbase + rows * cols, 0, (size_t)(rows * cols) * 4);
}

// 640-word buffer for tests/ptx/mma_m8n8k4_f16.ptx (four independent
// 8x8x4 MMAs, one per lane group).
static void build_mma_m8f16(uint32_t *in, uint32_t *want) {
    int A[4][8][4], B[4][4][8], C[4][8][8];
    for (int comp = 0; comp < 4; comp++) {
        for (int r = 0; r < 8; r++)
            for (int k = 0; k < 4; k++)
                A[comp][r][k] = 1 + ((comp * 7 + r * 4 + k) % 5);
        for (int k = 0; k < 4; k++)
            for (int c = 0; c < 8; c++)
                B[comp][k][c] = 1 + ((comp * 3 + k * 8 + c) % 3);
        for (int r = 0; r < 8; r++)
            for (int c = 0; c < 8; c++)
                C[comp][r][c] = (comp + r * 8 + c) % 7;
    }
    memset(in, 0, 640 * 4);
    for (int l = 0; l < 32; l++) {
        int comp = (l >> 2) & 3, t = l & 3, hi = l >= 16;
        int rowA = t + (hi ? 4 : 0), colB = rowA;
        in[l * 2 + 0] = f32_to_f16_bits((float)A[comp][rowA][0]) |
                        (f32_to_f16_bits((float)A[comp][rowA][1]) << 16);
        in[l * 2 + 1] = f32_to_f16_bits((float)A[comp][rowA][2]) |
                        (f32_to_f16_bits((float)A[comp][rowA][3]) << 16);
        in[64 + l * 2 + 0] = f32_to_f16_bits((float)B[comp][0][colB]) |
                             (f32_to_f16_bits((float)B[comp][1][colB]) << 16);
        in[64 + l * 2 + 1] = f32_to_f16_bits((float)B[comp][2][colB]) |
                             (f32_to_f16_bits((float)B[comp][3][colB]) << 16);
        for (int i = 0; i < 8; i++) {
            int r = (l & 1) + (i & 2) + (hi ? 4 : 0);
            int c = (i & 4) + (l & 2) + (i & 1);
            int d = C[comp][r][c];
            for (int k = 0; k < 4; k++)
                d += A[comp][r][k] * B[comp][k][c];
            in[128 + l * 8 + i] = f32_bits((float)C[comp][r][c]);
            in[384 + l * 8 + i] = f32_bits((float)d);
        }
    }
    memcpy(want, in, 640 * 4);
    memset(in + 384, 0, 256 * 4);
}

// Sparse 4:8 u4 MMA m16n8k64.  Every 8-wide chunk keeps subchunks
// {0,3}; the metadata nibble is 0b1100.
static void build_mma_sp_int4_k64(uint32_t *in, uint32_t *want,
                                  int is_signed) {
    int A[16][64], B[64][8], C[16][8];
    memset(A, 0, sizeof(A));
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 8; c++)
            for (int s = 0; s < 2; s++) {
                int sub = s ? 3 : 0;
                int v0 = is_signed ? ((r + c + s) % 8) - 4
                                   : 1 + ((r + c + s) % 7);
                int v1 = is_signed ? ((r + c + s + 1) % 8) - 4
                                   : 1 + ((r + c + s + 1) % 7);
                A[r][8 * c + 2 * sub + 0] = v0;
                A[r][8 * c + 2 * sub + 1] = v1;
            }
    for (int k = 0; k < 64; k++)
        for (int c = 0; c < 8; c++)
            B[k][c] = is_signed ? ((k + c) % 8) - 4
                                : 1 + ((k * 8 + c) % 7);
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 8; c++)
            C[r][c] = (r * 8 + c) % 7;
    memset(in, 0, 416 * 4);
    for (int L = 0; L < 32; L++) {
        int g = L >> 2, t = L & 3;
        int av[8];
        for (int q = 0; q < 2; q++) {
            int c = 2 * t + q;
            int base = q * 4;
            av[base + 0] = A[g][8 * c + 0];
            av[base + 1] = A[g][8 * c + 1];
            av[base + 2] = A[g][8 * c + 6];
            av[base + 3] = A[g][8 * c + 7];
        }
        in[L * 2 + 0] = pack_nibbles8(av);
        for (int q = 0; q < 2; q++) {
            int c = 2 * t + q;
            int base = q * 4;
            av[base + 0] = A[g + 8][8 * c + 0];
            av[base + 1] = A[g + 8][8 * c + 1];
            av[base + 2] = A[g + 8][8 * c + 6];
            av[base + 3] = A[g + 8][8 * c + 7];
        }
        in[L * 2 + 1] = pack_nibbles8(av);
        for (int j = 0; j < 2; j++) {
            int bv[8];
            for (int i = 0; i < 8; i++)
                bv[i] = B[8 * t + i + 32 * j][g];
            in[96 + L * 2 + j] = pack_nibbles8(bv);
        }
        for (int m = 0; m < 4; m++)
            in[160 + L * 4 + m] =
                (uint32_t)C[g + 8 * (m >= 2)][2 * t + (m & 1)];
    }
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 8; c++) {
            int owner = 4 * (r & 7) + (r >= 8 ? 1 : 0);
            in[64 + owner] |= 0xcu << (4 * c);
        }
    memcpy(want, in, 288 * 4);
    for (int L = 0; L < 32; L++) {
        int g = L >> 2, t = L & 3;
        for (int m = 0; m < 4; m++) {
            int r = g + 8 * (m >= 2), c = 2 * t + (m & 1);
            int d = C[r][c];
            for (int k = 0; k < 64; k++)
                d += A[r][k] * B[k][c];
            want[288 + L * 4 + m] = (uint32_t)d;
        }
    }
    memset(in + 288, 0, 128 * 4);
}

// Sparse 4:8 u4 MMA m16n8k128.  Every 8-wide chunk keeps subchunks
// {0,3}; the metadata nibble is 0b1100.
static void build_mma_sp_int4_k128(uint32_t *in, uint32_t *want,
                                   int is_signed) {
    int A[16][128], B[128][8], C[16][8];
    memset(A, 0, sizeof(A));
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 16; c++)
            for (int s = 0; s < 2; s++) {
                int sub = s ? 3 : 0;
                A[r][8 * c + 2 * sub + 0] =
                    is_signed ? ((r + c + s) % 8) - 4
                              : 1 + ((r + c + s) % 7);
                A[r][8 * c + 2 * sub + 1] =
                    is_signed ? ((r + c + s + 1) % 8) - 4
                              : 1 + ((r + c + s + 1) % 7);
            }
    for (int k = 0; k < 128; k++)
        for (int c = 0; c < 8; c++)
            B[k][c] = is_signed ? ((k + c) % 8) - 4
                                : 1 + ((k * 8 + c) % 7);
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 8; c++)
            C[r][c] = (r * 8 + c) % 7;
    memset(in, 0, 544 * 4);
    for (int L = 0; L < 32; L++) {
        int g = L >> 2, t = L & 3;
        for (int half = 0; half < 2; half++) {
            int av[8], bv[8];
            for (int i = 0; i < 8; i++)
                av[i] = bv[i] = 0;
            for (int q = 0; q < 2; q++) {
                int c = 8 * half + 2 * t + q;
                int base = q * 4;
                av[base + 0] = A[g][8 * c + 0];
                av[base + 1] = A[g][8 * c + 1];
                av[base + 2] = A[g][8 * c + 6];
                av[base + 3] = A[g][8 * c + 7];
                bv[base + 0] = A[g + 8][8 * c + 0];
                bv[base + 1] = A[g + 8][8 * c + 1];
                bv[base + 2] = A[g + 8][8 * c + 6];
                bv[base + 3] = A[g + 8][8 * c + 7];
            }
            in[L * 4 + half * 2 + 0] = pack_nibbles8(av);
            in[L * 4 + half * 2 + 1] = pack_nibbles8(bv);
        }
        for (int j = 0; j < 4; j++) {
            int bv[8];
            for (int i = 0; i < 8; i++)
                bv[i] = B[32 * j + 8 * t + i][g];
            in[160 + L * 4 + j] = pack_nibbles8(bv);
        }
        for (int m = 0; m < 4; m++)
            in[288 + L * 4 + m] =
                (uint32_t)C[g + 8 * (m >= 2)][2 * t + (m & 1)];
    }
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 16; c++) {
            int half = c >= 8;
            int owner = 4 * (r & 7) + (half ? 2 : 0) + (r >= 8 ? 1 : 0);
            int idx = half ? c - 8 : c;
            in[128 + owner] |= 0xcu << (4 * idx);
        }
    memcpy(want, in, 416 * 4);
    for (int L = 0; L < 32; L++) {
        int g = L >> 2, t = L & 3;
        for (int m = 0; m < 4; m++) {
            int r = g + 8 * (m >= 2), c = 2 * t + (m & 1);
            int d = C[r][c];
            for (int k = 0; k < 128; k++)
                d += A[r][k] * B[k][c];
            want[416 + L * 4 + m] = (uint32_t)d;
        }
    }
    memset(in + 416, 0, 128 * 4);
}

// Sparse TF32 m16n8k8/k16, 1:2 structured A.  Metadata selector is 0.
static void build_mma_sp_tf32(uint32_t *in, uint32_t *want, int K) {
    int A[16][16], B[16][8], C[16][8];
    int pos[16][8];
    int chunks = K / 2;
    memset(A, 0, sizeof(A));
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < chunks; c++) {
            pos[r][c] = (r + c) & 1;
            A[r][2 * c + pos[r][c]] = 1 + ((r * 16 + c) % 7);
        }
    for (int k = 0; k < K; k++)
        for (int c = 0; c < 8; c++)
            B[k][c] = 1 + ((k * 8 + c) % 5);
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 8; c++)
            C[r][c] = (r * 8 + c) % 7;
    int nwords = K == 8 ? 416 : 544;
    int awords = K == 8 ? 64 : 128;
    int mwords = awords;
    int bwords = mwords + 32;
    int cwords = bwords + (K == 8 ? 64 : 128);
    int dwords = cwords + 128;
    memset(in, 0, (size_t)nwords * 4);
    for (int L = 0; L < 32; L++) {
        int g = L >> 2, t = L & 3;
        if (K == 8) {
            in[L * 2 + 0] =
                f32_bits((float)A[g][2 * t + pos[g][t]]);
            in[L * 2 + 1] =
                f32_bits((float)A[g + 8][2 * t + pos[g][t]]);
        } else {
            in[L * 4 + 0] =
                f32_bits((float)A[g][2 * t + pos[g][t]]);
            in[L * 4 + 1] =
                f32_bits((float)A[g + 8][2 * t + pos[g][t]]);
            in[L * 4 + 2] =
                f32_bits((float)A[g][8 + 2 * t + pos[g][t + 4]]);
            in[L * 4 + 3] =
                f32_bits((float)A[g + 8][8 + 2 * t + pos[g][t + 4]]);
        }
        int bstride = K == 8 ? 2 : 4;
        for (int j = 0; j < K / 4; j++)
            in[bwords + L * bstride + j] =
                f32_bits((float)B[4 * j + (L & 3)][g]);
        for (int m = 0; m < 4; m++)
            in[cwords + L * 4 + m] =
                f32_bits((float)C[g + 8 * (m >= 2)][2 * t + (m & 1)]);
    }
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < chunks; c++) {
            int idx = (r >= 8 ? 4 : 0) + (c & 3);
            int owner = 4 * (r & 7) + (K == 16 && c >= 4 ? 1 : 0);
            in[mwords + owner] |=
                (uint32_t)(pos[r][c] ? 0xeu : 0x4u) << (4 * idx);
        }
    memcpy(want, in, (size_t)(cwords + 128) * 4);
    for (int L = 0; L < 32; L++) {
        int g = L >> 2, t = L & 3;
        for (int m = 0; m < 4; m++) {
            int r = g + 8 * (m >= 2), c = 2 * t + (m & 1);
            int d = C[r][c];
            for (int k = 0; k < K; k++)
                d += A[r][k] * B[k][c];
            want[dwords + L * 4 + m] = f32_bits((float)d);
        }
    }
    memset(in + dwords, 0, 128 * 4);
}

// 544-word buffer for tests/ptx/mma_sp_k32.ptx (reuse = 0) and
// tests/ptx/mma_sp_reuse.ptx (reuse = 1, two MMAs, zero accumulator).
static void build_mma_sp_k32(uint32_t *in, uint32_t *want, int alt,
                             int reuse, int bf16) {
    int A[16][32], B[32][8], C[16][8], D[16][8];
    memset(A, 0, sizeof(A));
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 8; c++) {
            int i0 = (c % 2 == 0) ? 0 : 1;
            int i1 = (c % 2 == 0) ? (alt ? 3 : 2) : (alt ? 2 : 3);
            A[r][4 * c + i0] = 1 + ((r * 32 + 4 * c + i0) % 5);
            A[r][4 * c + i1] = 1 + ((r * 32 + 4 * c + i1) % 5);
        }
    for (int k = 0; k < 32; k++)
        for (int c = 0; c < 8; c++)
            B[k][c] = 1 + ((k * 8 + c) % 3);
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 8; c++) {
            C[r][c] = (r * 8 + c) % 7;
            int d = C[r][c];
            for (int k = 0; k < 32; k++)
                d += A[r][k] * B[k][c];
            D[r][c] = d;
        }
    memset(in, 0, 544 * 4);
    for (int l = 0; l < 32; l++) {
        int g = l >> 2, t = l & 3;
        int i0 = (t % 2 == 0) ? 0 : 1;
        int i1 = (t % 2 == 0) ? (alt ? 3 : 2) : (alt ? 2 : 3);
        in[l * 4 + 0] =
            (bf16 ? f32_to_bf16_bits((float)A[g][4 * t + i0])
                  : f32_to_f16_bits((float)A[g][4 * t + i0])) |
            ((bf16 ? f32_to_bf16_bits((float)A[g][4 * t + i1])
                   : f32_to_f16_bits((float)A[g][4 * t + i1])) << 16);
        in[l * 4 + 1] =
            (bf16 ? f32_to_bf16_bits((float)A[g + 8][4 * t + i0])
                  : f32_to_f16_bits((float)A[g + 8][4 * t + i0])) |
            ((bf16 ? f32_to_bf16_bits((float)A[g + 8][4 * t + i1])
                   : f32_to_f16_bits((float)A[g + 8][4 * t + i1])) << 16);
        in[l * 4 + 2] =
            (bf16 ? f32_to_bf16_bits((float)A[g][16 + 4 * t + i0])
                  : f32_to_f16_bits((float)A[g][16 + 4 * t + i0])) |
            ((bf16 ? f32_to_bf16_bits((float)A[g][16 + 4 * t + i1])
                   : f32_to_f16_bits((float)A[g][16 + 4 * t + i1])) << 16);
        in[l * 4 + 3] =
            (bf16 ? f32_to_bf16_bits((float)A[g + 8][16 + 4 * t + i0])
                  : f32_to_f16_bits((float)A[g + 8][16 + 4 * t + i0])) |
            ((bf16 ? f32_to_bf16_bits((float)A[g + 8][16 + 4 * t + i1])
                   : f32_to_f16_bits((float)A[g + 8][16 + 4 * t + i1])) << 16);
        in[128 + l] = alt ? 0x9c9c9c9cu : 0xd8d8d8d8u;
        for (int r = 0; r < 4; r++) {
            int k0 = 8 * r + 2 * t, k1 = k0 + 1;
            in[160 + l * 4 + r] =
                (bf16 ? f32_to_bf16_bits((float)B[k0][g])
                      : f32_to_f16_bits((float)B[k0][g])) |
                ((bf16 ? f32_to_bf16_bits((float)B[k1][g])
                       : f32_to_f16_bits((float)B[k1][g])) << 16);
        }
        for (int j = 0; j < 4; j++) {
            int r = g + 8 * (j >= 2);
            int c = 2 * t + (j & 1);
            if (reuse) {
                in[288 + l * 4 + j] = 0;
                in[416 + l * 4 + j] = 0;
            } else {
                in[288 + l * 4 + j] = f32_bits((float)C[r][c]);
                in[416 + l * 4 + j] = f32_bits((float)D[r][c]);
            }
        }
    }
    memcpy(want, in, 544 * 4);
    if (reuse) {
        for (int l = 0; l < 32; l++) {
            int g = l >> 2, t = l & 3;
            for (int j = 0; j < 4; j++) {
                int r = g + 8 * (j >= 2);
                int c = 2 * t + (j & 1);
                int p = 0;
                for (int k = 0; k < 32; k++)
                    p += A[r][k] * B[k][c];
                want[288 + l * 4 + j] = f32_bits((float)p);
                want[416 + l * 4 + j] = f32_bits((float)p);
            }
        }
        memset(in + 288, 0, 256 * 4);
    } else {
        memset(in + 416, 0, 128 * 4);
    }
}

// 768-word buffer for tests/ptx/wmma_tf32.ptx.
static void build_wmma_tf32(uint32_t *in, uint32_t *want) {
    int A[16][8], B[8][16], C[16][16], D[16][16];
    for (int r = 0; r < 16; r++)
        for (int k = 0; k < 8; k++)
            A[r][k] = 1 + ((r * 8 + k) % 5);
    for (int k = 0; k < 8; k++)
        for (int c = 0; c < 16; c++)
            B[k][c] = 1 + ((k * 8 + c) % 3);
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 16; c++) {
            C[r][c] = (r * 8 + c) % 7;
            int d = C[r][c];
            for (int k = 0; k < 8; k++)
                d += A[r][k] * B[k][c];
            D[r][c] = d;
        }
    memset(in, 0, 768 * 4);
    for (int r = 0; r < 16; r++)
        for (int k = 0; k < 8; k++)
            in[r * 8 + k] = f32_bits((float)A[r][k]);
    for (int c = 0; c < 16; c++)
        for (int k = 0; k < 8; k++)
            in[128 + c * 8 + k] = f32_bits((float)B[k][c]);
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 16; c++) {
            in[256 + r * 16 + c] = f32_bits((float)C[r][c]);
            in[512 + r * 16 + c] = f32_bits((float)D[r][c]);
        }
    memcpy(want, in, 768 * 4);
    memset(in + 512, 0, 256 * 4);
}

static void put_f64(uint32_t *w, int i, double v) {
    uint64_t u;
    memcpy(&u, &v, 8);
    w[i] = (uint32_t)u;
    w[i + 1] = (uint32_t)(u >> 32);
}

// Buffer for tests/ptx/mma_f64_k4/k8/k16.ptx (sm_90 target-gated).
static void build_mma_f64_k(uint32_t *in, uint32_t *want, int K) {
    int na = K / 2, nb = K / 4;
    int nwords = (na + nb) * 64 + 512;
    int A[16][16], B[16][8], C[16][8], D[16][8];
    for (int r = 0; r < 16; r++)
        for (int k = 0; k < K; k++)
            A[r][k] = 1 + ((r * K + k) % 5);
    for (int k = 0; k < K; k++)
        for (int c = 0; c < 8; c++)
            B[k][c] = 1 + ((k * 8 + c) % 3);
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 8; c++) {
            C[r][c] = (r * 8 + c) % 7;
            int d = C[r][c];
            for (int k = 0; k < K; k++)
                d += A[r][k] * B[k][c];
            D[r][c] = d;
        }
    memset(in, 0, (size_t)nwords * 4);
    for (int l = 0; l < 32; l++) {
        int g = l >> 2, t = l & 3;
        for (int q = 0; q < K / 4; q++) {
            put_f64(in, l * (2 * na) + 4 * q, (double)A[g][4 * q + t]);
            put_f64(in, l * (2 * na) + 4 * q + 2,
                    (double)A[g + 8][4 * q + t]);
            put_f64(in, na * 64 + l * (2 * nb) + 2 * q,
                    (double)B[t + 4 * q][g]);
        }
        int base = (na + nb) * 64 + l * 8;
        for (int j = 0; j < 4; j++) {
            int r = g + 8 * (j >= 2);
            int c = 2 * t + (j & 1);
            put_f64(in, base + 2 * j, (double)C[r][c]);
            put_f64(in, base + 256 + 2 * j, (double)D[r][c]);
        }
    }
    memcpy(want, in, (size_t)nwords * 4);
    memset(in + (na + nb) * 64 + 256, 0, 256 * 4);
}

// Regression for 1b13d98: the linear lineage tracker can infer a shared
// lineage for a register that a global access uses (register reuse across
// scopes).  analyze_params must treat that as a plain value instead of
// aborting with "shared address used with global access".  The emitter
// still has the final say about whether the kernel is coherent, so this
// unit test only pins the parameter-analysis behavior.
static void check_shared_global_lineage(void) {
    static const char *ptx =
        ".version 7.0\n"
        ".target sm_80\n"
        ".address_size 64\n"
        ".shared .align 4 .b8 smem[16];\n"
        ".visible .entry shared_global_lineage(.param .u64 p0)\n"
        "{\n"
        "    .reg .b32 %r<2>;\n"
        "    mov.u32 %r0, smem;\n"
        "    add.u32 %r0, %r0, 4;\n"
        "    st.global.u32 [%r0], 0;\n"
        "    ret;\n"
        "}\n";
    ptx_unit_t unit;
    memset(&unit, 0, sizeof(unit));
    char err[512];
    err[0] = '\0';
    if (ptx_parse(ptx, &unit, err, sizeof(err)) != 0) {
        printf("FAIL: shared/global lineage PTX parse: %s\n", err);
        failures++;
        return;
    }
    const ptx_func_t *fn = NULL;
    for (size_t i = 0; i < unit.nfuncs; i++)
        if (strcmp(unit.funcs[i].name, "shared_global_lineage") == 0)
            fn = &unit.funcs[i];
    CHECK(fn != NULL);
    if (fn) {
        int kinds[64];
        memset(kinds, 0, sizeof(kinds));
        char aerr[512];
        aerr[0] = '\0';
        int rc = analyze_params(&unit, fn, kinds, aerr, sizeof(aerr));
        if (rc != 0) {
            printf("FAIL: shared/global lineage rejected: %s\n", aerr);
            failures++;
        } else {
            printf("  shared/global lineage: accepted\n");
        }
    }
    ptx_unit_free(&unit);
}

// Per-param write flags feed the runtime's barrier elision. A load-only
// input must stay clean while the written output is marked.
static void check_param_write_flags(void) {
    static const char *ptx =
        ".version 8.0\n"
        ".target sm_80\n"
        ".address_size 64\n"
        ".visible .entry wr(.param .u64 out, .param .u64 in)\n"
        "{\n"
        "    .reg .u64 %ro, %ri;\n"
        "    .reg .u32 %v;\n"
        "    ld.param.u64 %ro, [out];\n"
        "    ld.param.u64 %ri, [in];\n"
        "    ld.global.u32 %v, [%ri];\n"
        "    st.global.u32 [%ro], %v;\n"
        "    ret;\n"
        "}\n";
    acuda_kernel_t *k = NULL;
    char err[1024];
    CHECK_ST(acuda_compile_ptx(ptx, NULL, &k, err, sizeof(err), &g_test_caps));
    acuda_kernel_info_t info;
    CHECK_ST(acuda_kernel_info(k, &info));
    CHECK(info.num_ptr_params == 2);
    CHECK(info.param_written[0] == 1);
    CHECK(info.param_written[1] == 0);
    acuda_kernel_free(k);
}

// A .f64 kernel parameter is a 64-bit scalar and must reserve two push
// words. Classifying it as an unused or 32-bit parameter silently drops
// half of the value (or the whole argument).
static void check_f64_param_push(void) {
    static const char *ptx =
        ".version 8.0\n"
        ".target sm_80\n"
        ".address_size 64\n"
        ".visible .entry f64p(.param .u64 out, .param .f64 scale)\n"
        "{\n"
        "    .reg .u64 %ro;\n"
        "    .reg .f64 %fd;\n"
        "    ld.param.u64 %ro, [out];\n"
        "    ld.param.f64 %fd, [scale];\n"
        "    st.global.f64 [%ro], %fd;\n"
        "    ret;\n"
        "}\n";
    acuda_kernel_t *k = NULL;
    char err[1024];
    CHECK_ST(acuda_compile_ptx(ptx, NULL, &k, err, sizeof(err), &g_test_caps));
    acuda_kernel_info_t info;
    CHECK_ST(acuda_kernel_info(k, &info));
    CHECK(info.num_ptr_params == 1);
    CHECK(info.push_bytes == 8);
    acuda_kernel_free(k);
}

// Shared f32 atomic add: 256 lanes contend on two slots. The atom
// returns a distinct old value per lane (0..255), and both final
// counters must be 256. Runs once through the device-selected path
// (native OpAtomicFAddEXT when available) and once with the native
// lowering disabled to cover the CAS fallback.
static void run_atom_shared_f32(acuda_context_t ctx, const char *label) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/atom_shared_f32.ptx",
             ACUDA_TEST_PTX_DIR);
    char *ptx = read_file(path);
    if (!ptx) {
        failures++;
        return;
    }
    acuda_kernel_t *k = NULL;
    char err[1024];
    if (acuda_compile_ptx(ptx, NULL, &k, err, sizeof(err), &g_test_caps) != ACUDA_SUCCESS) {
        printf("FAIL: %s compile: %s\n", label, err);
        failures++;
        free(ptx);
        return;
    }
    free(ptx);
    const unsigned LANES = 256;
    const size_t bytes = 3 * (size_t)LANES * 4;
    uint32_t *got = calloc(3 * (size_t)LANES, 4);
    acuda_buffer_t buf = NULL;
    if (!got || acuda_malloc(ctx, &buf, bytes) != ACUDA_SUCCESS) {
        printf("FAIL: %s buffer\n", label);
        failures++;
        free(got);
        acuda_kernel_free(k);
        return;
    }
    if (acuda_memcpy_h2d(ctx, buf, got, bytes, NULL) != ACUDA_SUCCESS ||
        acuda_launch_kernel(ctx, k, &buf, 1, NULL, 0, 1, LANES, NULL) !=
            ACUDA_SUCCESS ||
        acuda_stream_synchronize(ctx, NULL) != ACUDA_SUCCESS ||
        acuda_memcpy_d2h(ctx, got, buf, bytes, NULL) != ACUDA_SUCCESS) {
        printf("FAIL: %s launch\n", label);
        failures++;
    } else {
        unsigned dups = 0, badred = 0, badatom = 0, sum = 0;
        unsigned char seen[256];
        memset(seen, 0, sizeof(seen));
        for (unsigned i = 0; i < LANES; i++) {
            float f;
            memcpy(&f, &got[i], 4);
            unsigned v = (unsigned)f;
            if (v >= LANES || seen[v])
                dups++;
            else
                seen[v] = 1;
            sum += v;
        }
        for (unsigned i = 0; i < LANES; i++) {
            if (got[LANES + i] != 0x43800000u) // 256.0f
                badred++;
            if (got[2 * LANES + i] != 0x43800000u)
                badatom++;
        }
        if (dups || badred || badatom || sum != LANES * (LANES - 1) / 2) {
            printf("FAIL: %s dups=%u sum=%u badred=%u badatom=%u\n",
                   label, dups, sum, badred, badatom);
            failures++;
        } else {
            printf("  %s: 256 contended adds ok\n", label);
        }
    }
    CHECK_ST(acuda_free(ctx, buf));
    acuda_kernel_free(k);
    free(got);
}

// Compile-only check for a forced native float-atomic path. The device
// may not enable the feature, but when the compiler is told it is
// available the module must carry the instruction and capability.
static void check_atomic_spirv(const char *file, unsigned mask,
                               const char *label, uint32_t opcode,
                               uint32_t cap) {
    unsigned saved = acuda_compiler_get_atomic_float(&g_test_caps);
    acuda_compiler_set_atomic_float(&g_test_caps, mask);
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", ACUDA_TEST_PTX_DIR, file);
    char *ptx = read_file(path);
    acuda_kernel_t *k = NULL;
    char err[1024];
    if (!ptx) {
        failures++;
        acuda_compiler_set_atomic_float(&g_test_caps, saved);
        return;
    }
    if (acuda_compile_ptx(ptx, NULL, &k, err, sizeof(err), &g_test_caps) != ACUDA_SUCCESS) {
        printf("FAIL: %s compile: %s\n", label, err);
        failures++;
    } else {
        const uint32_t *w = NULL;
        size_t n = 0;
        unsigned ops = 0, caps = 0;
        if (acuda_kernel_spirv(k, &w, &n) == ACUDA_SUCCESS) {
            for (size_t i = 5; i < n;) {
                uint32_t wc = w[i] >> 16, op = w[i] & 0xffffu;
                if (!wc)
                    break;
                if (op == opcode)
                    ops++;
                if (op == 17u && i + 1 < n && w[i + 1] == cap)
                    caps++;
                i += wc;
            }
        }
        if (ops && caps) {
            printf("  %s: %u atomic op(s), capability ok\n", label, ops);
        } else {
            printf("FAIL: %s: missing opcode %u or capability %u\n",
                   label, opcode, cap);
            failures++;
        }
    }
    acuda_kernel_free(k);
    free(ptx);
    acuda_compiler_set_atomic_float(&g_test_caps, saved);
}

// Compile-only check that the mixed-signedness AccSat form is emitted
// when the form is forced available (this device does not accelerate it).
static void check_dp4a_sat_native(void) {
    int s = 0, u = 0, su = 0;
    acuda_compiler_get_dot_product_forms(&g_test_caps, &s, &u, &su);
    acuda_compiler_set_dot_product(&g_test_caps, 1);
    acuda_compiler_set_dot_product_forms(&g_test_caps, 1, 1, 1);
    char path[1024];
    snprintf(path, sizeof(path), "%s/dp4a_sat.ptx", ACUDA_TEST_PTX_DIR);
    char *ptx = read_file(path);
    acuda_kernel_t *k = NULL;
    char err[1024];
    int ops = 0;
    if (ptx &&
        acuda_compile_ptx(ptx, NULL, &k, err, sizeof(err), &g_test_caps) == ACUDA_SUCCESS) {
        const uint32_t *w = NULL;
        size_t n = 0;
        if (acuda_kernel_spirv(k, &w, &n) == ACUDA_SUCCESS) {
            for (size_t i = 5; i < n;) {
                uint32_t wc = w[i] >> 16, op = w[i] & 0xffffu;
                if (!wc)
                    break;
                if (op == 4455u) // OpSUDotAccSat
                    ops++;
                i += wc;
            }
        }
    }
    if (ops != 3) { // three mixed cases
        printf("FAIL: dp4a sat mixed native: %d OpSUDotAccSat (want 3)\n",
               ops);
        failures++;
    } else {
        printf("  dp4a sat mixed native: %d OpSUDotAccSat\n", ops);
    }
    acuda_kernel_free(k);
    free(ptx);
    acuda_compiler_set_dot_product_forms(&g_test_caps, s, u, su);
}

int main(void) {
    acuda_compiler_caps_init(&g_test_caps);
    char path[1024];
    snprintf(path, sizeof(path), "%s/vecadd.ptx", ACUDA_TEST_PTX_DIR);
    char *ptx = read_file(path);
    if (!ptx)
        return 1;

    // Negative: unknown entry, garbage (numba PTX now compiles: f64).
    {
        char err[1024];
        acuda_kernel_t *k = NULL;
        CHECK(acuda_compile_ptx(ptx, "no_such_entry", &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        CHECK(acuda_compile_ptx("this is not ptx", NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        snprintf(path, sizeof(path), "%s/vecadd_numba.ptx", ACUDA_TEST_PTX_DIR);
        char *nptx = read_file(path);
        if (nptx) {
            acuda_status_t st = acuda_compile_ptx(nptx, NULL, &k, err, sizeof(err), &g_test_caps);
            CHECK(st == ACUDA_SUCCESS);
            if (k) {
                acuda_kernel_info_t info;
                CHECK(acuda_kernel_info(k, &info) == ACUDA_SUCCESS);
                CHECK(info.num_ptr_params == 4);
                printf("  numba f64: compiles, %u ptr params, %u push bytes\n",
                       info.num_ptr_params, info.push_bytes);
                acuda_kernel_free(k);
            }
            free(nptx);
        }
    }

    // By-value struct fields are plain scalars, never pointer lineage.
    // A u32 field used in mad.lo must not be misdiagnosed as ambiguous
    // pointer arithmetic (regression: every parameter word carried the
    // enclosing parameter's pointer lineage).
    {
        const char *byval =
            ".version 7.0\n"
            ".target sm_80\n"
            ".address_size 64\n"
            ".visible .entry byval_u32_field(\n"
            "    .param .u64x2 view,\n"
            "    .param .u32 scale,\n"
            "    .param .u64 output)\n"
            "{\n"
            "    .reg .b32 %r<4>;\n"
            "    .reg .b64 %rd<2>;\n"
            "    ld.param.u64 %rd0, [output];\n"
            "    ld.param.u32 %r0, [view+0];\n"
            "    ld.param.u32 %r1, [scale];\n"
            "    mad.lo.u32 %r2, %r0, %r1, 0;\n"
            "    st.global.u32 [%rd0], %r2;\n"
            "    ret;\n"
            "}\n";
        char err[1024];
        acuda_kernel_t *k = NULL;
        CHECK(acuda_compile_ptx(byval, "byval_u32_field", &k, err,
                                sizeof(err), &g_test_caps) == ACUDA_SUCCESS);
        if (k) {
            acuda_kernel_info_t info;
            CHECK(acuda_kernel_info(k, &info) == ACUDA_SUCCESS);
            CHECK(info.num_params == 3);
            CHECK(info.is_ptr[0] == 4); // by-value scalar struct
            CHECK(info.wide_words[0] == 2);
            CHECK(info.is_ptr[1] == 0); // u32 scalar
            CHECK(info.is_ptr[2] == 1); // output pointer
            acuda_kernel_free(k);
        }
    }

    check_shared_global_lineage();

    // ACUDA_PTX / ACUDA_SM ceilings (docs/ptx-baseline.md): above-cap
    // .version / .target must fail cleanly; unset/garbage = accept.
    {
        snprintf(path, sizeof(path), "%s/nb_ejit.ptx", ACUDA_TEST_PTX_DIR);
        char *gb = read_file(path);
        if (gb) {
            char err[1024];
            acuda_kernel_t *k = NULL;
            setenv("ACUDA_PTX", "8.5", 1);
            CHECK(acuda_compile_ptx(gb, NULL, &k, err, sizeof(err), &g_test_caps) !=
                  ACUDA_SUCCESS);
            unsetenv("ACUDA_PTX");
            setenv("ACUDA_PTX", "ISA 8.8", 1);
            CHECK(acuda_compile_ptx(gb, NULL, &k, err, sizeof(err), &g_test_caps) ==
                  ACUDA_SUCCESS);
            if (k) {
                acuda_kernel_free(k);
                k = NULL;
            }
            unsetenv("ACUDA_PTX");
            setenv("ACUDA_SM", "sm_70", 1);
            CHECK(acuda_compile_ptx(gb, NULL, &k, err, sizeof(err), &g_test_caps) !=
                  ACUDA_SUCCESS);
            unsetenv("ACUDA_SM");
            setenv("ACUDA_SM", "90", 1);
            CHECK(acuda_compile_ptx(gb, NULL, &k, err, sizeof(err), &g_test_caps) ==
                  ACUDA_SUCCESS);
            if (k) {
                acuda_kernel_free(k);
                k = NULL;
            }
            unsetenv("ACUDA_SM");
            setenv("ACUDA_PTX", "banana", 1);
            CHECK(acuda_compile_ptx(gb, NULL, &k, err, sizeof(err), &g_test_caps) ==
                  ACUDA_SUCCESS);
            if (k) {
                acuda_kernel_free(k);
                k = NULL;
            }
            unsetenv("ACUDA_PTX");
            // Default .version ceiling is 9.4.
            const char *vl = strstr(gb, ".version");
            char *g94 = NULL, *g95 = NULL;
            if (vl) {
                const char *nl = strchr(vl, '\n');
                size_t pre = (size_t)(vl - gb);
                size_t post = nl ? strlen(nl) : 0;
                const char *v94 = ".version 9.4";
                const char *v95 = ".version 9.5";
                g94 = malloc(pre + strlen(v94) + post + 1);
                g95 = malloc(pre + strlen(v95) + post + 1);
                if (g94 && g95) {
                    memcpy(g94, gb, pre);
                    strcpy(g94 + pre, v94);
                    if (nl)
                        strcpy(g94 + pre + strlen(v94), nl);
                    memcpy(g95, gb, pre);
                    strcpy(g95 + pre, v95);
                    if (nl)
                        strcpy(g95 + pre + strlen(v95), nl);
                    CHECK(acuda_compile_ptx(g94, NULL, &k, err,
                                            sizeof(err), &g_test_caps) == ACUDA_SUCCESS);
                    if (k) { acuda_kernel_free(k); k = NULL; }
                    CHECK(acuda_compile_ptx(g95, NULL, &k, err,
                                            sizeof(err), &g_test_caps) != ACUDA_SUCCESS);
                    setenv("ACUDA_PTX", "banana", 1);
                    CHECK(acuda_compile_ptx(g94, NULL, &k, err,
                                            sizeof(err), &g_test_caps) == ACUDA_SUCCESS);
                    if (k) { acuda_kernel_free(k); k = NULL; }
                    CHECK(acuda_compile_ptx(g95, NULL, &k, err,
                                            sizeof(err), &g_test_caps) != ACUDA_SUCCESS);
                    setenv("ACUDA_PTX", "9.4", 1);
                    CHECK(acuda_compile_ptx(g95, NULL, &k, err,
                                            sizeof(err), &g_test_caps) != ACUDA_SUCCESS);
                    unsetenv("ACUDA_PTX");
                }
                free(g94);
                free(g95);
            }
            free(gb);
        }
    }

    // .version/.target validity matrix: only released PTX ISA versions
    // and the SM names each release introduced are accepted.
    {
        static const char *fmt =
            ".version %s\n.target %s\n.address_size 64\n"
            ".visible .entry vt(.param .u64 p)\n{\n    ret;\n}\n";
        struct {
            const char *ver;
            const char *tgt;
            int ok;
        } cases[] = {
            { "1.0", "sm_10", 1 },
            { "1.2", "sm_13", 1 },
            { "1.0", "sm_20", 0 },
            { "8.5", "sm_100", 0 },
            { "8.6", "sm_100", 1 },
            { "8.7", "sm_120", 1 },
            { "9.4", "sm_107a", 1 },
            { "9.5", "sm_80", 0 },
            { "8.9", "sm_80", 0 },
            { "8.6", "sm_99", 0 },
            { "8.6", "sm_80, texmode_independent", 1 },
        };
        for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
            char src[512];
            snprintf(src, sizeof(src), fmt, cases[i].ver, cases[i].tgt);
            acuda_kernel_t *k = NULL;
            char err[1024] = {0};
            acuda_status_t cst =
                acuda_compile_ptx(src, NULL, &k, err, sizeof(err), &g_test_caps);
            if ((cst == ACUDA_SUCCESS) != cases[i].ok) {
                printf("FAIL: .version %s .target %s -> %s (%s)\n",
                       cases[i].ver, cases[i].tgt,
                       cst == ACUDA_SUCCESS ? "accepted" : "rejected", err);
                failures++;
            }
            if (k)
                acuda_kernel_free(k);
        }
        printf("  version/target matrix: %zu cases\n",
               sizeof(cases) / sizeof(cases[0]));
    }

    acuda_context_t ctx = NULL;
    acuda_status_t st = acuda_init(&ctx);
    if (st == ACUDA_ERROR_NO_DEVICE || st == ACUDA_ERROR_NOT_SUPPORTED) {
        printf("SKIPPED: no suitable Vulkan device (%s)\n", acuda_strerror(st));
        free(ptx);
        return 0;
    }
    CHECK(st == ACUDA_SUCCESS);
    if (!ctx) {
        free(ptx);
        return 1;
    }
    // Compile the rest of this suite with the device's real capability
    // set (BDA, atomics, int64 images, ...), exactly as the runtime does
    // for kernels it compiles.  Default caps leave those features off.
    if (acuda_context_get_compiler_caps(ctx))
        g_test_caps = *acuda_context_get_compiler_caps(ctx);

    run_vecadd(ctx, ptx, "vecadd_f32", 1);
    run_vecadd(ctx, ptx, "vecadd_u32", 0);
    free(ptx);

    // Real NVRTC output (tests/ptx/vecadd_nvrtc.ptx, entry "vecadd"):
    // exercises .b32/.b64 regs, mad.lo.s32, cvta, fma.rn.f32, $ labels.
    snprintf(path, sizeof(path), "%s/vecadd_nvrtc.ptx", ACUDA_TEST_PTX_DIR);
    char *nrtx = read_file(path);
    if (!nrtx)
        return 1;
    run_vecadd(ctx, nrtx, "vecadd", 1);
    free(nrtx);

    // Real numba EJIT output (tests/ptx/nb_ejit.ptx): .func, 22 .b64
    // params (4 pointer + 7 u64-value + 11 unused), diamond CFG,
    // generic ld/st, signed index math. Entry name is long; pass NULL
    // (first entry) via a dedicated runner below.
    snprintf(path, sizeof(path), "%s/nb_ejit.ptx", ACUDA_TEST_PTX_DIR);
    char *nbptx = read_file(path);
    if (!nbptx)
        return 1;
    {
        // Pinned layout: data pointers at params 4/11/18, N at 16.
        // Compile + metadata here; real launch in test_driver + live.
        char err[1024];
        acuda_kernel_t *k = NULL;
        if (acuda_compile_ptx(nbptx, NULL, &k, err, sizeof(err), &g_test_caps) !=
            ACUDA_SUCCESS) {
            printf("FAIL: numba ptx compile: %s\n", err);
            failures++;
        } else {
            acuda_kernel_info_t info;
            CHECK_ST(acuda_kernel_info(k, &info));
            CHECK(info.num_ptr_params == 3);
            const uint32_t nn = 4096;
            acuda_buffer_t bufs[4];
            acuda_buffer_t *ba = &bufs[0], *bb = &bufs[1], *bc = &bufs[2],
                           *be = &bufs[3];
            CHECK_ST(acuda_malloc(ctx, ba, (size_t)nn * 4));
            CHECK_ST(acuda_malloc(ctx, bb, (size_t)nn * 4));
            CHECK_ST(acuda_malloc(ctx, bc, (size_t)nn * 4));
            CHECK_ST(acuda_malloc(ctx, be, 8));
            float *ha = malloc((size_t)nn * 4);
            float *hb = malloc((size_t)nn * 4);
            float *hc = malloc((size_t)nn * 4);
            if (ha && hb && hc) {
                for (uint32_t i = 0; i < nn; i++) {
                    ha[i] = (float)i * 0.5f;
                    hb[i] = (float)i * 2.0f + 1.0f;
                }
                CHECK_ST(acuda_memcpy_h2d(ctx, *ba, ha, (size_t)nn * 4,
                                         NULL));
                CHECK_ST(acuda_memcpy_h2d(ctx, *bb, hb, (size_t)nn * 4,
                                         NULL));
                // NOTE: this path needs driver-level positional args;
                // tested via test_driver-style launch below using the
                // compiler API directly is insufficient (needs 22 args).
                // Full launch covered by nb_vecadd.py; here we at least
                // verify compile + metadata.
                printf("  numba ejit: compiles, %u ptr params, %u push bytes\n",
                       info.num_ptr_params, info.push_bytes);
            }
            free(ha);
            free(hb);
            free(hc);
            CHECK_ST(acuda_free(ctx, *ba));
            CHECK_ST(acuda_free(ctx, *bb));
            CHECK_ST(acuda_free(ctx, *bc));
            CHECK_ST(acuda_free(ctx, *be));
            acuda_kernel_free(k);
        }
        free(nbptx);
    }

    // Mini flow kernel (tests/ptx/mini_flow.ptx): new small ops +
    // multi-region flow. Pinned layout: 3 pointer params + 1 u32.
    // Full launch in test_driver; compile + metadata here.
    snprintf(path, sizeof(path), "%s/mini_flow.ptx", ACUDA_TEST_PTX_DIR);
    char *mfptx = read_file(path);
    if (!mfptx)
        return 1;
    {
        char err[1024];
        acuda_kernel_t *k = NULL;
        if (acuda_compile_ptx(mfptx, NULL, &k, err, sizeof(err), &g_test_caps) !=
            ACUDA_SUCCESS) {
            printf("FAIL: miniflow ptx compile: %s\n", err);
            failures++;
        } else {
            acuda_kernel_info_t info;
            CHECK_ST(acuda_kernel_info(k, &info));
            CHECK(info.num_ptr_params == 3);
            CHECK(info.push_bytes == 4);
            printf("  miniflow: compiles, %u ptr params, %u push bytes\n",
                   info.num_ptr_params, info.push_bytes);
            acuda_kernel_free(k);
        }
    }
    free(mfptx);

    // Direct .reg-form call kernel (tests/ptx/call_reg.ptx): inlined
    // multi-return callee. Pinned layout: 2 pointer params, 0 push.
    // Full launch in test_driver; compile + metadata here.
    snprintf(path, sizeof(path), "%s/call_reg.ptx", ACUDA_TEST_PTX_DIR);
    char *crptx = read_file(path);
    if (!crptx)
        return 1;
    {
        char err[1024];
        acuda_kernel_t *k = NULL;
        if (acuda_compile_ptx(crptx, "callreg", &k, err, sizeof(err), &g_test_caps) !=
            ACUDA_SUCCESS) {
            printf("FAIL: callreg ptx compile: %s\n", err);
            failures++;
        } else {
            acuda_kernel_info_t info;
            CHECK_ST(acuda_kernel_info(k, &info));
            CHECK(info.num_ptr_params == 2);
            CHECK(info.push_bytes == 0);
            printf("  callreg: compiles, %u ptr params, %u push bytes\n",
                   info.num_ptr_params, info.push_bytes);
            acuda_kernel_free(k);
        }
    }
    free(crptx);

    // Predicated-execution coverage (tests/ptx/predcover.ptx): 2
    // pointer params, 0 push. Full launch in test_driver; compile +
    // metadata here.
    snprintf(path, sizeof(path), "%s/predcover.ptx", ACUDA_TEST_PTX_DIR);
    char *pcptx = read_file(path);
    if (!pcptx)
        return 1;
    {
        char err[1024];
        acuda_kernel_t *k = NULL;
        if (acuda_compile_ptx(pcptx, NULL, &k, err, sizeof(err), &g_test_caps) !=
            ACUDA_SUCCESS) {
            printf("FAIL: predcover ptx compile: %s\n", err);
            failures++;
        } else {
            acuda_kernel_info_t info;
            CHECK_ST(acuda_kernel_info(k, &info));
            CHECK(info.num_ptr_params == 2);
            CHECK(info.push_bytes == 0);
            printf("  predcover: compiles, %u ptr params, %u push bytes\n",
                   info.num_ptr_params, info.push_bytes);
            acuda_kernel_free(k);
        }
    }
    free(pcptx);

    // Scalar brain-float kernel (tests/ptx/bf16_cvt.ptx): setp.eq.b32,
    // cvt.rn.f32.u32, cvt.rn.bf16.f32, sub-word st.u16. 2 pointer
    // params, 0 push. Full launch in test_driver; compile + metadata
    // here.
    snprintf(path, sizeof(path), "%s/bf16_cvt.ptx", ACUDA_TEST_PTX_DIR);
    char *bfptx = read_file(path);
    if (!bfptx)
        return 1;
    {
        char err[1024];
        acuda_kernel_t *k = NULL;
        if (acuda_compile_ptx(bfptx, NULL, &k, err, sizeof(err), &g_test_caps) !=
            ACUDA_SUCCESS) {
            printf("FAIL: bf16 ptx compile: %s\n", err);
            failures++;
        } else {
            acuda_kernel_info_t info;
            CHECK_ST(acuda_kernel_info(k, &info));
            CHECK(info.num_ptr_params == 2);
            CHECK(info.push_bytes == 0);
            printf("  bf16: compiles, %u ptr params, %u push bytes\n",
                   info.num_ptr_params, info.push_bytes);
            acuda_kernel_free(k);
        }
    }
    free(bfptx);

    // Real numba loop kernel (tests/ptx/loop_numba.ptx): 23 params
    // (4 pointer + 4 u64-value + 15 unused), multi-region flow with
    // two self-loops. Full launch in test_driver; metadata here.
    snprintf(path, sizeof(path), "%s/loop_numba.ptx", ACUDA_TEST_PTX_DIR);
    char *lnptx = read_file(path);
    if (!lnptx)
        return 1;
    {
        char err[1024];
        acuda_kernel_t *k = NULL;
        if (acuda_compile_ptx(lnptx, NULL, &k, err, sizeof(err), &g_test_caps) !=
            ACUDA_SUCCESS) {
            printf("FAIL: looplong ptx compile: %s\n", err);
            failures++;
        } else {
            acuda_kernel_info_t info;
            CHECK_ST(acuda_kernel_info(k, &info));
            CHECK(info.num_ptr_params == 4);
            CHECK(info.push_bytes == 32);
            CHECK(info.num_params == 23);
            printf("  looplong: compiles, %u ptr params, %u push bytes, %u params\n",
                   info.num_ptr_params, info.push_bytes, info.num_params);
            acuda_kernel_free(k);
        }
        free(lnptx);
    }

    // Phase 4 close-out kernel (tests/ptx/arith_closeout.ptx): 7 pointer
    // params, div/rem/or/xor/shifts/setp.s32. Full launch in
    // test_driver; compile + metadata here.
    snprintf(path, sizeof(path), "%s/arith_closeout.ptx",
             ACUDA_TEST_PTX_DIR);
    char *acptx = read_file(path);
    if (!acptx)
        return 1;
    {
        char err[1024];
        acuda_kernel_t *k = NULL;
        if (acuda_compile_ptx(acptx, NULL, &k, err, sizeof(err), &g_test_caps) !=
            ACUDA_SUCCESS) {
            printf("FAIL: arithclose ptx compile: %s\n", err);
            failures++;
        } else {
            acuda_kernel_info_t info;
            CHECK_ST(acuda_kernel_info(k, &info));
            CHECK(info.num_ptr_params == 7);
            CHECK(info.push_bytes == 0);
            printf("  arithclose: compiles, %u ptr params, %u push bytes\n",
                   info.num_ptr_params, info.push_bytes);
            acuda_kernel_free(k);
        }
        free(acptx);
    }

    // Shared-memory kernels: hand-written shmem (3 ptr, 1280 shared
    // bytes) and the harvested shmem_numba (4 ptr, 1024 shared bytes).
    // Full launches in test_driver; compile + metadata here.
    snprintf(path, sizeof(path), "%s/shmem.ptx", ACUDA_TEST_PTX_DIR);
    char *shmptx = read_file(path);
    if (!shmptx)
        return 1;
    {
        char err[1024];
        acuda_kernel_t *k = NULL;
        if (acuda_compile_ptx(shmptx, NULL, &k, err, sizeof(err), &g_test_caps) !=
            ACUDA_SUCCESS) {
            printf("FAIL: shmem ptx compile: %s\n", err);
            failures++;
        } else {
            acuda_kernel_info_t info;
            CHECK_ST(acuda_kernel_info(k, &info));
            CHECK(info.num_ptr_params == 3);
            CHECK(info.shared_bytes == 1280);
            printf("  shmem: compiles, %u ptr params, %u shared bytes\n",
                   info.num_ptr_params, info.shared_bytes);
            acuda_kernel_free(k);
        }
        free(shmptx);
    }
    snprintf(path, sizeof(path), "%s/shmem_numba.ptx", ACUDA_TEST_PTX_DIR);
    char *shnptx = read_file(path);
    if (!shnptx)
        return 1;
    {
        char err[1024];
        acuda_kernel_t *k = NULL;
        if (acuda_compile_ptx(shnptx, NULL, &k, err, sizeof(err), &g_test_caps) !=
            ACUDA_SUCCESS) {
            printf("FAIL: shmemnb ptx compile: %s\n", err);
            failures++;
        } else {
            acuda_kernel_info_t info;
            CHECK_ST(acuda_kernel_info(k, &info));
            CHECK(info.num_ptr_params == 4);
            CHECK(info.shared_bytes == 1024);
            printf("  shmemnb: compiles, %u ptr params, %u shared bytes\n",
                   info.num_ptr_params, info.shared_bytes);
            acuda_kernel_free(k);
        }
        free(shnptx);
    }

    // Narrow-types kernel (tests/ptx/narrow.ptx): 12 pointer params,
    // 3 shared variables (16384 bytes). Full launch in test_driver;
    // compile + metadata here.
    snprintf(path, sizeof(path), "%s/narrow.ptx", ACUDA_TEST_PTX_DIR);
    char *narptx = read_file(path);
    if (!narptx)
        return 1;
    {
        char err[1024];
        acuda_kernel_t *k = NULL;
        if (acuda_compile_ptx(narptx, NULL, &k, err, sizeof(err), &g_test_caps) !=
            ACUDA_SUCCESS) {
            printf("FAIL: narrow ptx compile: %s\n", err);
            failures++;
        } else {
            acuda_kernel_info_t info;
            CHECK_ST(acuda_kernel_info(k, &info));
            CHECK(info.num_ptr_params == 12);
            CHECK(info.shared_bytes == 16384);
            printf("  narrow: compiles, %u ptr params, %u shared bytes\n",
                   info.num_ptr_params, info.shared_bytes);
            acuda_kernel_free(k);
        }
        free(narptx);
    }

    // Real numba f16 kernel (tests/ptx/f16_numba.ptx): 22 params
    // (4 pointer + 3 u64-value + 15 unused), halves as bits. Full
    // launch in test_driver; metadata here.
    snprintf(path, sizeof(path), "%s/f16_numba.ptx", ACUDA_TEST_PTX_DIR);
    char *f16nptx = read_file(path);
    if (!f16nptx)
        return 1;
    {
        char err[1024];
        acuda_kernel_t *k = NULL;
        if (acuda_compile_ptx(f16nptx, NULL, &k, err, sizeof(err), &g_test_caps) !=
            ACUDA_SUCCESS) {
            printf("FAIL: f16nb ptx compile: %s\n", err);
            failures++;
        } else {
            acuda_kernel_info_t info;
            CHECK_ST(acuda_kernel_info(k, &info));
            CHECK(info.num_ptr_params == 4);
            CHECK(info.shared_bytes == 0);
            printf("  f16nb: compiles, %u ptr params, %u shared bytes\n",
                   info.num_ptr_params, info.shared_bytes);
            acuda_kernel_free(k);
        }
        free(f16nptx);
    }

    // Negative: spellings ptxas rejects are clean errors here too
    // (verified against ptxas 12.9): no mov.u8/.f16, no ld/st.f16,
    // no f16 division, neg takes signed types only.
    {
        char err[1024];
        acuda_kernel_t *k = NULL;
        const char *badmovu8 =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".visible .entry badmovu8(.param .u64 a)\n"
            "{\n"
            "    .reg .u8 %b;\n"
            "    .reg .u64 %rd<2>;\n"
            "    ld.param.u64 %rd1, [a];\n"
            "    mov.u8 %b, %b;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badmovu8, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative mov.u8: %s\n", err);
        const char *badldf16 =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".visible .entry badldf16(.param .u64 a)\n"
            "{\n"
            "    .reg .f16 %x;\n"
            "    .reg .u64 %rd<2>;\n"
            "    ld.param.u64 %rd1, [a];\n"
            "    ld.global.f16 %x, [%rd1];\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badldf16, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative ld.f16: %s\n", err);
        const char *baddivf16 =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".visible .entry baddivf16(.param .u64 a)\n"
            "{\n"
            "    .reg .bf16 %x<2>;\n"
            "    .reg .u64 %rd<2>;\n"
            "    ld.param.u64 %rd1, [a];\n"
            "    div.rn.bf16 %x1, %x1, %x1;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(baddivf16, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative div.bf16: %s\n", err);
        const char *badnegu32 =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".visible .entry badnegu32(.param .u64 a)\n"
            "{\n"
            "    .reg .u32 %r<2>;\n"
            "    .reg .u64 %rd<2>;\n"
            "    ld.param.u64 %rd1, [a];\n"
            "    neg.u32 %r1, %r1;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badnegu32, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative neg.u32: %s\n", err);
    }

    // Negative: dynamic .shared, counted barriers, and cross-space
    // address misuse are clean errors.
    {
        char err[1024];
        acuda_kernel_t *k = NULL;
        const char *baddyn =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".extern .shared .b8 dyn[];\n"
            ".visible .entry baddyn(.param .u64 a)\n"
            "{\n"
            "    .reg .u64 %rd<2>;\n"
            "    ld.param.u64 %rd1, [a];\n"
            "    st.global.u64 [%rd1], %rd1;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(baddyn, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative extern-shared: %s\n", err);
        const char *badbar =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".visible .entry badbar(.param .u64 a)\n"
            "{\n"
            "    .reg .u64 %rd<2>;\n"
            "    ld.param.u64 %rd1, [a];\n"
            "    bar.sync 0, 2;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badbar, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative counted-barrier: %s\n", err);
        const char *badspace =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".visible .shared .b8 sh[64];\n"
            ".visible .entry badspace(.param .u64 a)\n"
            "{\n"
            "    .reg .u64 %rd<4>;\n"
            "    .reg .f32 %f<2>;\n"
            "    ld.param.u64 %rd1, [a];\n"
            "    mov.u64 %rd2, sh;\n"
            "    ld.shared.f32 %f1, [%rd1];\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badspace, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative shared-global-mix: %s\n", err);
    }
    // Negative: float rem is a clean error (rem.f32 is not PTX).
    {
        char err[1024];
        acuda_kernel_t *k = NULL;
        const char *badremf =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".visible .entry badremf(.param .u64 a)\n"
            "{\n"
            "    .reg .f32 %f<4>;\n"
            "    .reg .u64 %rd<2>;\n"
            "    ld.param.u64 %rd1, [a];\n"
            "    mov.f32 %f1, 1.5;\n"
            "    rem.f32 %f2, %f1, %f1;\n"
            "    st.global.f32 [%rd1], %f2;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badremf, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative rem.f32: %s\n", err);
    }
    // Negative: atomics beyond the subset must fail cleanly (wrong op,
    // wide type, cluster scope, cross-space address).
    {
        char err[1024];
        acuda_kernel_t *k = NULL;
        const char *badatomop =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".visible .entry badatomop(.param .u64 a)\n"
            "{\n"
            "    .reg .u32 %r<3>;\n"
            "    .reg .f32 %f<2>;\n"
            "    .reg .u64 %rd<2>;\n"
            "    ld.param.u64 %rd1, [a];\n"
            "    atom.and.f32 %f1, [%rd1], %f2;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badatomop, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative atom.and.f32: %s\n", err);
        const char *badatom64 =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".visible .entry badatom64(.param .u64 a)\n"
            "{\n"
            "    .reg .u32 %r;\n"
            "    .reg .u64 %rd<3>;\n"
            "    ld.param.u64 %rd1, [a];\n"
            "    atom.inc.u64 %rd2, [%rd1], %rd2;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badatom64, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative atom.inc.u64: %s\n", err);
        const char *badatomspace =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".visible .shared .b8 sh[64];\n"
            ".visible .entry badatomspace(.param .u64 a)\n"
            "{\n"
            "    .reg .u32 %r<2>;\n"
            "    .reg .u64 %rd<3>;\n"
            "    ld.param.u64 %rd1, [a];\n"
            "    mov.u64 %rd2, sh;\n"
            "    atom.global.add.u32 %r1, [%rd2], %r2;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badatomspace, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative atom-global-of-shared: %s\n", err);
    }
    // Negative: calls beyond the subset must fail cleanly (unknown
    // callee, arg-count mismatch, recursion).
    {
        char err[1024];
        acuda_kernel_t *k = NULL;
        const char *badcallee =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".visible .entry badcallee(.param .u64 a)\n"
            "{\n"
            "    .reg .u64 %rd<2>;\n"
            "    .param.u64 s_in;\n"
            "    .param.u64 s_out;\n"
            "    ld.param.u64 %rd1, [a];\n"
            "    st.param.b64 [s_in], %rd1;\n"
            "    call (s_out), nosuch, (s_in);\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badcallee, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative call-undefined: %s\n", err);
        const char *badcallargs =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".visible .func (.param.u64 o) one(.param.u64 x)\n"
            "{\n"
            "    .reg .u64 %t;\n"
            "    ld.param.u64 %t, [x];\n"
            "    st.param.u64 [o], %t;\n"
            "    ret;\n"
            "}\n"
            ".visible .entry badcallargs(.param .u64 a)\n"
            "{\n"
            "    .reg .u64 %rd<2>;\n"
            "    .param.u64 s_in;\n"
            "    .param.u64 s_out;\n"
            "    ld.param.u64 %rd1, [a];\n"
            "    st.param.b64 [s_in], %rd1;\n"
            "    call (s_out), one, (s_in, s_in);\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badcallargs, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative call-argcount: %s\n", err);
        const char *badcallrec =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".visible .entry self(.param .u64 a)\n"
            "{\n"
            "    .reg .u64 %rd<2>;\n"
            "    .param.u64 s;\n"
            "    ld.param.u64 %rd1, [a];\n"
            "    st.param.b64 [s], %rd1;\n"
            "    call (s), self, (s);\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badcallrec, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative call-recursive: %s\n", err);
    }
    // Negative: predication beyond selects/guards (loads, u64 dests),
    // and pointer-pointer sub stay clean errors.
    {
        char err[1024];
        acuda_kernel_t *k = NULL;
        const char *badpredld =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".visible .entry badpredld(.param .u64 a)\n"
            "{\n"
            "    .reg .pred %p;\n"
            "    .reg .u32 %r;\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    setp.eq.u32 %p, 0, 0;\n"
            "    @%p ld.u32 %r, [%rd];\n"
            "    ret;\n"
            "}\n";
        // Predicated loads on a block edge now lower to a guard
        // diamond, so this compiles (the load is skipped when false).
        CHECK(acuda_compile_ptx(badpredld, NULL, &k, err, sizeof(err), &g_test_caps) ==
              ACUDA_SUCCESS);
        if (k)
            acuda_kernel_free(k);
        printf("  predicated-ld guard: %s\n", err);
        const char *badpredu64 =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".visible .entry badpredu64(.param .u64 a)\n"
            "{\n"
            "    .reg .pred %p;\n"
            "    .reg .u64 %r<2>;\n"
            "    setp.eq.u32 %p, 0, 0;\n"
            "    @%p add.u64 %r1, %r1, 1;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badpredu64, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative predicated-u64: %s\n", err);
        // Scalar-minus-pointer sub has no base to attribute, so it stays
        // a clean error (ptxas accepts the raw-address form; we cannot
        // represent it). ptr-ptr sub itself is now legal.
        const char *badptrsub =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".visible .entry badptrsub(.param .u64 a)\n"
            "{\n"
            "    .reg .u64 %r<3>;\n"
            "    ld.param.u64 %r0, [a];\n"
            "    ld.u64 %r2, [%r0];\n" // mark the param as a pointer
            "    sub.u64 %r1, 8, %r0;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badptrsub, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative scalar-ptr-sub: %s\n", err);
    }
    // Negative: vote forms beyond the subset (ballot on a non-.b32
    // destination; .uni without corpus coverage). The deprecated
    // unsuffixed vote.ballot.<ty> spelling is now accepted (see
    // vote_ballot_nosync.ptx), so the negative case uses a type the
    // subset still rejects.
    {
        char err[1024];
        acuda_kernel_t *k = NULL;
        const char *badvotedepr =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".visible .entry badvotedepr(.param .u64 a)\n"
            "{\n"
            "    .reg .pred %p;\n"
            "    .reg .u32 %r;\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    setp.eq.u32 %p, 0, 0;\n"
            "    vote.ballot.f32 %r, %p;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badvotedepr, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative vote-ballot-f32: %s\n", err);
        const char *badvoteuni =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".visible .entry badvoteuni(.param .u64 a)\n"
            "{\n"
            "    .reg .pred %p;\n"
            "    .reg .pred %q;\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    setp.eq.u32 %p, 0, 0;\n"
            "    vote.sync.uni.b32 %q, %p, 0xFFFFFFFF;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badvoteuni, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative vote-uni-type: %s\n", err);
    }
    // Negative: unsupported redux op stays clean (add/min/max/and/or/xor
    // are supported).
    {
        char err[1024];
        acuda_kernel_t *k = NULL;
        const char *badreduxop =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".visible .entry badreduxop(.param .u64 a)\n"
            "{\n"
            "    .reg .u32 %r;\n"
            "    .reg .u32 %v;\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    mov.u32 %v, 1;\n"
            "    redux.sync.mul.u32 %r, %v, 0xFFFFFFFF;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badreduxop, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative redux-op: %s\n", err);
    }
    // Negative: match beyond .any stays clean (.all has no corpus
    // coverage; non-32-bit value types are out).
    {
        char err[1024];
        acuda_kernel_t *k = NULL;
        const char *badmatchall =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".visible .entry badmatchall(.param .u64 a)\n"
            "{\n"
            "    .reg .u32 %r;\n"
            "    .reg .u32 %v;\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    mov.u32 %v, 1;\n"
            "    match.all.sync.b32 %r, %v, 0xFFFFFFFF;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badmatchall, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative match-all: %s\n", err);
        const char *badmatchty =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".visible .entry badmatchty(.param .u64 a)\n"
            "{\n"
            "    .reg .u64 %r;\n"
            "    .reg .u64 %v;\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    mov.u64 %v, 1;\n"
            "    match.any.sync.u64 %r, %v, 0xFFFFFFFF;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badmatchty, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative match-type: %s\n", err);
    }
    // Negative: FP8 forms ptxas itself rejects stay clean errors —
    // narrowing without .satfinite, a non-f32 narrowing source, and a
    // widening destination that is not a .b32 pattern register. Also
    // non-gt packed compares out.
    {
        char err[1024];
        acuda_kernel_t *k = NULL;
        const char *badfp8sat =
            ".version 8.6\n.target sm_90\n.address_size 64\n"
            ".visible .entry badfp8sat(.param .u64 a)\n"
            "{\n"
            "    .reg .f32 %f1, %f2;\n"
            "    .reg .b16 %v;\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    cvt.rn.e4m3x2.f32 %v, %f1, %f2;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badfp8sat, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative fp8-missing-satfinite: %s\n", err);
        const char *badfp8src =
            ".version 8.6\n.target sm_90\n.address_size 64\n"
            ".visible .entry badfp8src(.param .u64 a)\n"
            "{\n"
            "    .reg .b16 %v, %h;\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    cvt.rn.satfinite.e4m3x2.f16x2 %v, %h, %h;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badfp8src, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative fp8-narrow-f16-src: %s\n", err);
        const char *badfp8dst =
            ".version 8.6\n.target sm_90\n.address_size 64\n"
            ".visible .entry badfp8dst(.param .u64 a)\n"
            "{\n"
            "    .reg .f32 %f1;\n"
            "    .reg .b16 %v;\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    cvt.rn.f16x2.e4m3x2 %f1, %v;\n"
            "    st.f32 [%rd], %f1;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badfp8dst, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative fp8-widen-f32-dst: %s\n", err);
        const char *badmf16dst =
            ".version 8.8\n.target sm_120\n.address_size 64\n"
            ".visible .entry badmf16dst(.param .u64 a)\n"
            "{\n"
            "    .reg .f32 %f1;\n"
            "    .reg .b16 %v;\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    cvt.rn.satfinite.bf16x2.e2m3x2 %f1, %v;\n"
            "    st.f32 [%rd], %f1;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badmf16dst, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative micro-unpack-f32-dst: %s\n", err);
        const char *badmfrs =
            ".version 8.8\n.target sm_120\n.address_size 64\n"
            ".visible .entry badmfrs(.param .u64 a)\n"
            "{\n"
            "    .reg .f32 %a, %b, %c, %d;\n"
            "    .reg .b16 %o;\n"
            "    .reg .b32 %r;\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    cvt.rs.satfinite.e2m1x4.f32 %o, {%a, %b, %c, %d}, %r;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badmfrs, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative micro-x4-rs: %s\n", err);
        const char *badsetpack =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".visible .entry badsetpack(.param .u64 a)\n"
            "{\n"
            "    .reg .b32 %r;\n"
            "    .reg .b32 %x;\n"
            "    .reg .b32 %y;\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    set.eq.f16x2.f64 %r, %x, %y;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badsetpack, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative set-packcmp: %s\n", err);
    }
    // Negative: predicate/fence/sleep forms ptxas rejects stay clean —
    // not.pred with an integer source, an unknown membar scope, and a
    // non-u32 nanosleep operand.
    {
        char err[1024];
        acuda_kernel_t *k = NULL;
        const char *badnotpred =
            ".version 8.6\n.target sm_70\n.address_size 64\n"
            ".visible .entry badnotpred(.param .u64 a)\n"
            "{\n"
            "    .reg .u32 %r;\n"
            "    .reg .pred %q;\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    ld.u32 %r, [%rd];\n"
            "    not.pred %q, %r;\n"
            "    st.u32 [%rd], %r;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badnotpred, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative not-pred-int-src: %s\n", err);
        const char *badmembar =
            ".version 8.6\n.target sm_70\n.address_size 64\n"
            ".visible .entry badmembar(.param .u64 a)\n"
            "{\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    membar.global;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badmembar, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative membar-scope: %s\n", err);
        const char *badnsleep =
            ".version 8.6\n.target sm_70\n.address_size 64\n"
            ".visible .entry badnsleep(.param .u64 a)\n"
            "{\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    nanosleep.f32 1;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badnsleep, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative nanosleep-f32: %s\n", err);
    }
    // Negatives: the directed-rounding cvt matrix rejects what ptxas
    // rejects — an extra operand and a non-f32 non-int destination
    // ("Illegal rounding modifier", "Arguments mismatch").
    {
        char err[1024];
        acuda_kernel_t *k = NULL;
        const char *badrndop =
            ".version 8.6\n.target sm_70\n.address_size 64\n"
            ".visible .entry badrndop(.param .u64 a)\n"
            "{\n"
            "    .reg .f32 %f;\n"
            "    .reg .b32 %r;\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    ld.f32 %f, [%rd];\n"
            "    cvt.rpi.s32.f32 %r, %f, %f;\n"
            "    st.b32 [%rd], %r;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badrndop, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative cvt-rnd-extra-operand: %s\n", err);
        const char *badrnddst =
            ".version 8.6\n.target sm_70\n.address_size 64\n"
            ".visible .entry badrnddst(.param .u64 a)\n"
            "{\n"
            "    .reg .f32 %f;\n"
            "    .reg .b16 %h;\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    ld.f32 %f, [%rd];\n"
            "    cvt.rpi.f16.f32 %h, %f;\n"
            "    st.b16 [%rd], %h;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badrnddst, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative cvt-rnd-f16-dst: %s\n", err);
    }
    // Negative: mma beyond .m16n8k16.f32.f16/bf16 stays clean (other
    // shapes need their own fragment maps; predicated mma would break
    // the convergence contract).
    {
        char err[1024];
        acuda_kernel_t *k = NULL;
        const char *badmmashape =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".visible .entry badmmashape(.param .u64 a)\n"
            "{\n"
            "    .reg .b32 %d<4>;\n"
            "    .reg .b32 %x<4>;\n"
            "    .reg .b32 %y;\n"
            "    .reg .f32 %c<4>;\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    mma.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
            "{%d0, %d1, %d2, %d3}, {%x0, %x1, %x2, %x3}, "
            "{%y, %y}, {%c0, %c1, %c2, %c3};\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badmmashape, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative mma-shape: %s\n", err);
        const char *badmmapred =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".visible .entry badmmapred(.param .u64 a)\n"
            "{\n"
            "    .reg .pred %p;\n"
            "    .reg .f32 %d<4>;\n"
            "    .reg .b32 %x<4>;\n"
            "    .reg .b32 %y<2>;\n"
            "    .reg .f32 %c<4>;\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    setp.eq.u32 %p, 0, 0;\n"
            "    @%p mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%d0, %d1, %d2, %d3}, {%x0, %x1, %x2, %x3}, "
            "{%y0, %y1}, {%c0, %c1, %c2, %c3};\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badmmapred, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative mma-pred: %s\n", err);
        // Unsupported integer mma shapes still fail cleanly: sub-byte
        // multiplicands and s64 accumulation have no fragment map.
        const char *badminteger =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".visible .entry badminteger(.param .u64 a)\n"
            "{\n"
            "    .reg .b32 %d<4>;\n"
            "    .reg .b32 %x<4>;\n"
            "    .reg .b32 %y<2>;\n"
            "    .reg .b32 %c<4>;\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    mma.sync.aligned.m16n8k32.row.col.s32.u4.u4.s32 "
            "{%d0, %d1, %d2, %d3}, {%x0, %x1, %x2, %x3}, "
            "{%y0, %y1}, {%c0, %c1, %c2, %c3};\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badminteger, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative mma-integer-shape: %s\n", err);
        const char *badminteger64 =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".visible .entry badminteger64(.param .u64 a)\n"
            "{\n"
            "    .reg .b32 %d<4>;\n"
            "    .reg .b32 %x<4>;\n"
            "    .reg .b32 %y<2>;\n"
            "    .reg .b32 %c<4>;\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    mma.sync.aligned.m16n8k32.row.col.s64.s8.s8.s64 "
            "{%d0, %d1, %d2, %d3}, {%x0, %x1, %x2, %x3}, "
            "{%y0, %y1}, {%c0, %c1, %c2, %c3};\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badminteger64, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative mma-integer-64: %s\n", err);
    }
    // Negative: WMMA beyond the m16n16k16 f16->f32 subset stays
    // clean (other shapes need their own fragment maps, and f16
    // accumulate would need a different fragment element type).
    {
        char err[1024];
        acuda_kernel_t *k = NULL;
        const char *badwmmashape =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".visible .entry badwmmashape(.param .u64 a)\n"
            "{\n"
            "    .reg .b32 %d<8>, %x<8>, %y<8>, %c<8>;\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    wmma.mma.sync.aligned.row.col.m8n32k16.f32.f32 "
            "{%d0,%d1,%d2,%d3,%d4,%d5,%d6,%d7}, "
            "{%x0,%x1,%x2,%x3,%x4,%x5,%x6,%x7}, "
            "{%y0,%y1,%y2,%y3,%y4,%y5,%y6,%y7}, "
            "{%c0,%c1,%c2,%c3,%c4,%c5,%c6,%c7};\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badwmmashape, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative wmma-shape: %s\n", err);
        const char *badwmmaacc =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".visible .entry badwmmaacc(.param .u64 a)\n"
            "{\n"
            "    .reg .b32 %d<8>, %x<8>, %y<8>, %c<8>;\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    wmma.mma.sync.aligned.row.col.m16n16k16.f16.f16 "
            "{%d0,%d1,%d2,%d3,%d4,%d5,%d6,%d7}, "
            "{%x0,%x1,%x2,%x3,%x4,%x5,%x6,%x7}, "
            "{%y0,%y1,%y2,%y3,%y4,%y5,%y6,%y7}, "
            "{%c0,%c1,%c2,%c3,%c4,%c5,%c6,%c7};\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badwmmaacc, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative wmma-accumulate: %s\n", err);
        const char *badwmmapred =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".visible .entry badwmmapred(.param .u64 a)\n"
            "{\n"
            "    .reg .pred %p;\n"
            "    .reg .b32 %c<8>;\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    setp.eq.u32 %p, 0, 0;\n"
            "    @%p wmma.load.c.sync.aligned.m16n16k16.global.row.f32 "
            "{%c0,%c1,%c2,%c3,%c4,%c5,%c6,%c7}, [%rd];\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badwmmapred, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative wmma-pred: %s\n", err);
    }
    // Negative: mma.sp beyond m16n8k16 f16/bf16 stays clean.
    {
        char err[1024];
        acuda_kernel_t *k = NULL;
        const char *badsp_shape =
            ".version 8.5\n.target sm_80\n.address_size 64\n"
            ".visible .entry badsp_shape(.param .u64 a)\n"
            "{\n"
            "    .reg .b32 %d<4>, %x<2>, %y<2>, %c<4>, %e;\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
            "{%d0,%d1,%d2,%d3}, {%x0,%x1}, {%y0,%y1}, "
            "{%c0,%c1,%c2,%c3}, %e, 0;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badsp_shape, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative mma-sp-shape: %s\n", err);
        const char *badsp_type =
            ".version 8.5\n.target sm_80\n.address_size 64\n"
            ".visible .entry badsp_type(.param .u64 a)\n"
            "{\n"
            "    .reg .b32 %d<4>, %x<2>, %y<2>, %c<4>, %e;\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    mma.sp.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 "
            "{%d0,%d1,%d2,%d3}, {%x0,%x1}, {%y0,%y1}, "
            "{%c0,%c1,%c2,%c3}, %e, 0;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badsp_type, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative mma-sp-type: %s\n", err);
        const char *badsp_sel =
            ".version 8.5\n.target sm_80\n.address_size 64\n"
            ".visible .entry badsp_sel(.param .u64 a)\n"
            "{\n"
            "    .reg .b32 %d<4>, %x<2>, %y<2>, %c<4>, %e;\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%d0,%d1,%d2,%d3}, {%x0,%x1}, {%y0,%y1}, "
            "{%c0,%c1,%c2,%c3}, %e, 4;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badsp_sel, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative mma-sp-selector: %s\n", err);
        const char *badsp_pred =
            ".version 8.5\n.target sm_80\n.address_size 64\n"
            ".visible .entry badsp_pred(.param .u64 a)\n"
            "{\n"
            "    .reg .pred %p;\n"
            "    .reg .b32 %d<4>, %x<2>, %y<2>, %c<4>, %e;\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    setp.eq.u32 %p, 0, 0;\n"
            "    @%p mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%d0,%d1,%d2,%d3}, {%x0,%x1}, {%y0,%y1}, "
            "{%c0,%c1,%c2,%c3}, %e, 0;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badsp_pred, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative mma-sp-pred: %s\n", err);
    }
    // Negative: mismatched dense mma shapes/types stay clean.
    {
        char err[1024];
        acuda_kernel_t *k = NULL;
        const char *badk8tf32 =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".visible .entry badk8tf32(.param .u64 a)\n"
            "{\n"
            "    .reg .b32 %d<4>, %x<2>, %y, %c<4>;\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
            "{%d0,%d1,%d2,%d3}, {%x0,%x1}, {%y}, {%c0,%c1,%c2,%c3};\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badk8tf32, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative mma-k8-tf32: %s\n", err);
        const char *badk4f16 =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".visible .entry badk4f16(.param .u64 a)\n"
            "{\n"
            "    .reg .b32 %d<4>, %x<2>, %y, %c<4>;\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    mma.sync.aligned.m16n8k4.row.col.f32.f16.f16.f32 "
            "{%d0,%d1,%d2,%d3}, {%x0,%x1}, {%y}, {%c0,%c1,%c2,%c3};\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badk4f16, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative mma-k4-f16: %s\n", err);
    }
    // Negative: f16 accumulation only applies to f16 multiplicands and
    // requires matching dtype/ctype.
    {
        char err[1024];
        acuda_kernel_t *k = NULL;
        const char *badf16bf16 =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".visible .entry badf16bf16(.param .u64 a)\n"
            "{\n"
            "    .reg .b32 %d<2>, %x<4>, %y<2>, %c<2>;\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    mma.sync.aligned.m16n8k16.row.col.f16.bf16.bf16.f16 "
            "{%d0,%d1}, {%x0,%x1,%x2,%x3}, {%y0,%y1}, {%c0,%c1};\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badf16bf16, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative mma-f16acc-bf16: %s\n", err);
        const char *badf16mix =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".visible .entry badf16mix(.param .u64 a)\n"
            "{\n"
            "    .reg .b32 %d<2>, %x<4>, %y<2>, %c<2>;\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f16 "
            "{%d0,%d1}, {%x0,%x1,%x2,%x3}, {%y0,%y1}, {%c0,%c1};\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badf16mix, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative mma-f16acc-mismatch: %s\n", err);
    }
    // Negative: other m8n8k4 variants stay clean.
    {
        char err[1024];
        acuda_kernel_t *k = NULL;
        const char *badm8f16 =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".visible .entry badm8f16(.param .u64 a)\n"
            "{\n"
            "    .reg .b32 %d<8>, %x<2>, %y<2>, %c<8>;\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16 "
            "{%d0,%d1,%d2,%d3,%d4,%d5,%d6,%d7}, {%x0,%x1}, "
            "{%y0,%y1}, {%c0,%c1,%c2,%c3,%c4,%c5,%c6,%c7};\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badm8f16, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative mma-m8n8k4-f16: %s\n", err);
        const char *badf64rz =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".visible .entry badf64rz(.param .u64 a)\n"
            "{\n"
            "    .reg .f64 %d<2>, %x, %y, %c<2>;\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64.ru "
            "{%d0,%d1}, {%x}, {%y}, {%c0,%c1};\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badf64rz, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative mma-f64-invalid-rnd: %s\n", err);
    }
    // Negative: target-gated matrix families report their requirement.
    {
        char err[1024];
        acuda_kernel_t *k = NULL;
        const char *wgsrc =
            ".version 8.6\n.target sm_90\n.address_size 64\n"
            ".visible .entry wg(.param .u64 a)\n"
            "{\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(wgsrc, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative target wgmma: %s\n", err);
        const char *tcsrc =
            ".version 8.6\n.target sm_100\n.address_size 64\n"
            ".visible .entry tc(.param .u64 a)\n"
            "{\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    tcgen05.mma.cta_group::1.kind::f16;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(tcsrc, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative target tcgen05: %s\n", err);
        const char *bssrc =
            ".version 8.6\n.target sm_90\n.address_size 64\n"
            ".visible .entry bs(.param .u64 a)\n"
            "{\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    mma.sync.aligned.m16n8k64.row.col.kind::mxf8f6f4.block_scale;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(bssrc, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative target block-scale: %s\n", err);
    }
    // Negative: sm_90/sm_100 data-movement and mixed-precision forms.
    {
        char err[1024];
        acuda_kernel_t *k = NULL;
        const char *mpsrc =
            ".version 8.6\n.target sm_86\n.address_size 64\n"
            ".visible .entry mp(.param .u64 a)\n"
            "{\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    add.f32.f16;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(mpsrc, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative target mixed-precision: %s\n", err);
        const char *sasrc =
            ".version 8.6\n.target sm_86\n.address_size 64\n"
            ".visible .entry sa(.param .u64 a)\n"
            "{\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    st.async.shared::cluster.u32 [%rd], 0;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(sasrc, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative target st.async: %s\n", err);
        const char *mapsrc =
            ".version 8.6\n.target sm_86\n.address_size 64\n"
            ".visible .entry mp2(.param .u64 a)\n"
            "{\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    mapa.shared::cluster.u32 %rd, %rd, 0;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(mapsrc, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative target mapa: %s\n", err);
        const char *rdsrc =
            ".version 8.6\n.target sm_86\n.address_size 64\n"
            ".visible .entry rda(.param .u64 a)\n"
            "{\n"
            "    .reg .u64 %rd;\n"
            "    .reg .u32 %r;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    red.async.release.gpu.global.add.u32 [%rd], %r;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(rdsrc, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative target red.async: %s\n", err);
        const char *mmrsrc =
            ".version 9.3\n.target sm_90\n.address_size 64\n"
            ".visible .entry mmr(.param .u64 a)\n"
            "{\n"
            "    .reg .u64 %rd;\n"
            "    .reg .u32 %r;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    multimem.red.async.release.gpu.global.add.u32 [%rd], %r;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(mmrsrc, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative target multimem.red.async: %s\n", err);
    }
    // Negative: mixed A/B element types on the s8 shape (a matrix-multiply
    // needs matching .btype/.ctype on both sides).
    {
        char err[1024];
        acuda_kernel_t *k = NULL;
        const char *bads8mixed =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".visible .entry bads8mixed(.param .u64 a)\n"
            "{\n"
            "    .reg .b32 %d<4>;\n"
            "    .reg .b32 %x<4>;\n"
            "    .reg .b32 %y<2>;\n"
            "    .reg .b32 %c<4>;\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.f32 "
            "{%d0, %d1, %d2, %d3}, {%x0, %x1, %x2, %x3}, "
            "{%y0, %y1}, {%c0, %c1, %c2, %c3};\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(bads8mixed, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative mma-s8-mixed-ctype: %s\n", err);
    }
    // Negative: ldmatrix beyond .m8n8.x{1,2,4}[.trans].b16 — the 8x8
    // shape is 16-bit only, the other shapes need their own layout, and
    // the compressed .b8x16 formats are not modeled.
    {
        char err[1024];
        acuda_kernel_t *k = NULL;
        const char *badldm =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".shared .align 16 .u32 s[64];\n"
            ".visible .entry badldm(.param .u64 a)\n"
            "{\n"
            "    .reg .b64 %p;\n"
            "    .reg .b32 %x<2>;\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    mov.b64 %p, s;\n"
            "    cvta.shared.u64 %p, %p;\n"
            "    ldmatrix.sync.aligned.m16n16.x2.trans.b8 "
            "{%x0, %x1}, [%p];\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badldm, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative ldmatrix-shape: %s\n", err);
        // .x4/.trans are supported now; the b8 compressed formats are
        // not (m8n8 is 16-bit only, and .b8x16 belongs to m16n16).
        const char *badldmb8x16 =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".shared .align 16 .u32 s[256];\n"
            ".visible .entry badldmb8x16(.param .u64 a)\n"
            "{\n"
            "    .reg .b64 %p;\n"
            "    .reg .b32 %x<4>;\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    mov.b64 %p, s;\n"
            "    cvta.shared.u64 %p, %p;\n"
            "    ldmatrix.sync.aligned.m8n8.x4.b8x16 "
            "{%x0, %x1, %x2, %x3}, [%p];\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badldmb8x16, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative ldmatrix-b8x16: %s\n", err);
        const char *badldmtransb8 =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".shared .align 16 .u16 s[256];\n"
            ".visible .entry badldmtransb8(.param .u64 a)\n"
            "{\n"
            "    .reg .b64 %p;\n"
            "    .reg .b32 %x<2>;\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    mov.b64 %p, s;\n"
            "    cvta.shared.u64 %p, %p;\n"
            "    ldmatrix.sync.aligned.x2.trans.m8n8.shared.b8 "
            "{%x0, %x1}, [%p];\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badldmtransb8, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative ldmatrix-trans-b8: %s\n", err);
        const char *badldmpred =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".shared .align 16 .u32 s[64];\n"
            ".visible .entry badldmpred(.param .u64 a)\n"
            "{\n"
            "    .reg .pred %q;\n"
            "    .reg .b64 %p;\n"
            "    .reg .b32 %x<2>;\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    mov.b64 %p, s;\n"
            "    cvta.shared.u64 %p, %p;\n"
            "    setp.eq.u32 %q, 0, 0;\n"
            "    @%q ldmatrix.sync.aligned.m8n8.x2.b16 "
            "{%x0, %x1}, [%p];\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badldmpred, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative ldmatrix-pred: %s\n", err);
    }
    // Negative: vector forms beyond the subset (.z on a .v2 has no
    // such component; mov.v8 is supported for brace lists).
    {
        char err[1024];
        acuda_kernel_t *k = NULL;
        const char *badcomp =
            ".version 6.5\n.target sm_60\n.address_size 64\n"
            ".visible .entry badcomp(.param .u64 a)\n"
            "{\n"
            "    .reg .v2.u32 %v;\n"
            "    .reg .u32 %r;\n"
            "    .reg .u64 %rd;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    mov.u32 %r, %v.z;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badcomp, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative vec-comp: %s\n", err);
    }
    // Negative: dynamic shared beyond the subset (sized externals,
    // launch over the cap).
    {
        char err[1024];
        acuda_kernel_t *k = NULL;
        const char *badsizedex =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".extern .shared .b32 x[4];\n"
            ".visible .entry badsizedex(.param .u64 a)\n"
            "{\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badsizedex, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative sized-extern: %s\n", err);
    }
    // Negative: .global images are read-only (no device backing for
    // stores) and have no floating-point element types.
    {
        char err[1024];
        acuda_kernel_t *k = NULL;
        const char *badglobst =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".global .u32 g[4] = {1};\n"
            ".visible .entry badglobst(.param .u64 a)\n"
            "{\n"
            "    .reg .u64 %rd<3>;\n"
            "    .reg .u32 %r;\n"
            "    ld.param.u64 %rd1, [a];\n"
            "    mov.u64 %rd2, g;\n"
            "    mov.u32 %r, 7;\n"
            "    st.global.u32 [%rd2], %r;\n"
            "    st.global.u32 [%rd1], %r;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badglobst, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative global-store: %s\n", err);
        const char *badglobty =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".global .f64 g[2];\n"
            ".visible .entry badglobty(.param .u64 a)\n"
            "{\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badglobty, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative global-f64-type: %s\n", err);
    }

    // Arbitrary CFG via the generic state-machine fallback: loops with
    // multiple entry edges, guarded loops, and irreducible backedges.
    {
        static const uint32_t lmin1[2] = { 3, 1 };
        static const uint32_t lmwant1[2] = { 3, 1 };
        run_words(ctx, "cfg_loop_multi.ptx", lmin1, 2, lmwant1,
                  "cfg multi-entry loop (taken)");
        static const uint32_t lmin0[2] = { 3, 0 };
        static const uint32_t lmwant0[2] = { 101, 0 };
        run_words(ctx, "cfg_loop_multi.ptx", lmin0, 2, lmwant0,
                  "cfg multi-entry loop (fallthrough)");
        static const uint32_t mgin[2] = { 5, 0 };
        static const uint32_t mgwant[2] = { 5, 0 };
        run_words(ctx, "cfg_mixed_guard.ptx", mgin, 2, mgwant,
                  "cfg guarded loop");
        static const uint32_t irin[2] = { 10, 0 };
        static const uint32_t irwant[2] = { 11, 0 };
        run_words(ctx, "cfg_irreducible.ptx", irin, 2, irwant,
                  "cfg irreducible loop");
        static const uint32_t plin1[3] = { 1, 0, 0 };
        static const uint32_t plwant1[3] = { 1, 1, 0 };
        run_words(ctx, "cfg_ptrloop.ptx", plin1, 3, plwant1,
                  "cfg pointer across state machine (taken)");
        static const uint32_t plin0[3] = { 0, 0, 0 };
        static const uint32_t plwant0[3] = { 0, 0, 1 };
        run_words(ctx, "cfg_ptrloop.ptx", plin0, 3, plwant0,
                  "cfg pointer across state machine (fallthrough)");
        static const uint32_t rzin[2] = { 0, 0 };
        static const uint32_t rzwant[2] = { 9, 0 };
        run_words(ctx, "cfg_returns.ptx", rzin, 2, rzwant,
                  "cfg multiple returns (zero path)");
        static const uint32_t rnin[2] = { 1, 0 };
        static const uint32_t rnwant[2] = { 7, 0 };
        run_words(ctx, "cfg_returns.ptx", rnin, 2, rnwant,
                  "cfg multiple returns (nonzero path)");
        static const uint32_t nlin[2] = { 10, 0 };
        static const uint32_t nlwant[2] = { 12, 0 };
        run_words(ctx, "cfg_nested.ptx", nlin, 2, nlwant,
                  "cfg nested loops");
    }

    // Atom lowering and .b128 register/access regressions.
    {
        static const uint32_t a32in[8] = { 0x0f, 0, 0, 0, 0, 0, 0, 0 };
        static const uint32_t a32want[8] = {
            0x0f, 0x0f, 0x03, 0x0f, 0xf0, 0x55, 0x10, 0x0f
        };
        static const uint32_t a64in[7] = { 0x0f, 0, 0, 0, 0, 0, 0 };
        static const uint32_t a64want[7] = {
            0x77, 0, 0x0f, 0, 0x10f, 0, 0x77
        };
        static const uint32_t b128in[4] = {
            0x2c2b2a29u, 0x302f2e2du, 0x34333231u, 0x38373635u
        };
        static const uint32_t b128want[4] = {
            0x2c2b2a29u, 0x302f2e2du, 0x34333231u, 0x38373635u
        };
        run_words(ctx, "atom32.ptx", a32in, 8, a32want, "atom32");
        run_words(ctx, "atom64.ptx", a64in, 7, a64want, "atom64");
        static const uint32_t afmin[16] = {
            0, 0x3ff80000u, 0, 0, 5, 0, 0, 0,
            0, 0, 11, 0, 13, 0, 0, 0
        };
        static const uint32_t afwant[16] = {
            0, 0x400e0000u, 0, 0x3ff80000u, 6, 5, 7, 8,
            9, 10, 12, 11, 1, 13, 1, 0
        };
        run_words(ctx, "atom_forms.ptx", afmin, 16, afwant,
                  "atom f64/cluster/cache forms");
        static const uint32_t ahin[8] = {
            0x11111234u, 0x22223C00u, 0x33333F80u, 0x3C003C00u,
            0x3F803F80u, 0, 0, 0
        };
        static const uint32_t ahwant[8] = {
            0x11115678u, 0x22224000u, 0x33334000u, 0x40004000u,
            0x40004000u, 0x56781234u, 0, 0
        };
        run_words(ctx, "atom_half.ptx", ahin, 8, ahwant,
                  "atom b16/f16/bf16 forms");
        uint32_t as64in[64] = { 0 };
        uint32_t as64out[64];
        for (unsigned t = 0; t < 32; t++) {
            as64out[t * 2] = 32;
            as64out[t * 2 + 1] = 0;
        }
        run_words_block(ctx, "atom_shared64.ptx", as64in, 64, as64out,
                        "atom shared u64", 32);
        run_atom_shared_f32(ctx, "atom shared f32");
        unsigned saved_atomic = acuda_compiler_get_atomic_float(&g_test_caps);
        if (saved_atomic & ACUDA_ATOMIC_F32_SHARED) {
            acuda_compiler_set_atomic_float(&g_test_caps, 0);
            run_atom_shared_f32(ctx, "atom shared f32 (CAS)");
            acuda_compiler_set_atomic_float(&g_test_caps, saved_atomic);
        }
        check_atomic_spirv("atom_bda_f32.ptx", ACUDA_ATOMIC_F32_BUFFER,
                           "atom bda f32", 6035, 6033);
        check_atomic_spirv("atom_bda_f64.ptx", ACUDA_ATOMIC_F64_BUFFER,
                           "atom bda f64", 6035, 6034);
        check_atomic_spirv("atom_shared_f64.ptx", ACUDA_ATOMIC_F64_SHARED,
                           "atom shared f64", 6035, 6034);
        check_atomic_spirv("sured_f32.ptx", ACUDA_ATOMIC_IMAGE_F32_ADD,
                           "sured f32", 6035, 6033);
        check_atomic_spirv("sured_minmax_f32.ptx",
                           ACUDA_ATOMIC_IMAGE_F32_MINMAX,
                           "sured f32 min", 5614, 5612);
        check_atomic_spirv("sured_minmax_f32.ptx",
                           ACUDA_ATOMIC_IMAGE_F32_MINMAX,
                           "sured f32 max", 5615, 5612);
        run_words(ctx, "b128.ptx", b128in, 4, b128want, "b128");
        uint32_t lv_in[16] = { 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0,
                               0, 0, 0, 0 };
        uint32_t lv_want[16] = { 1, 2, 3, 4, 5, 6, 7, 8,
                                 1, 2, 3, 4, 5, 6, 7, 8 };
        run_words(ctx, "ldu_vec.ptx", lv_in, 16, lv_want,
                  "ldu vector/b128");
        uint32_t lb_in[8] = { 0x11u, 0x22u, 0x33u, 0x44u, 0, 0, 0, 0 };
        uint32_t lb_want[8] = { 0x11u, 0x22u, 0x33u, 0x44u,
                                0x11u, 0x22u, 0x33u, 0x44u };
        run_words(ctx, "ldu_b128.ptx", lb_in, 8, lb_want, "ldu.b128");
        uint32_t lc_in[10] = { 0x11u, 0x22u, 0x33u, 0x44u, 0, 0, 0, 0,
                               0, 0 };
        uint32_t lc_want[10] = { 0x11u, 0x22u, 0x33u, 0x44u,
                                 0x11u, 0x22u, 0x33u, 0x44u,
                                 0x11u, 0x22u };
        run_words(ctx, "ldst_cache.ptx", lc_in, 10, lc_want,
                  "ld/st cache operators");
        uint32_t lh_in[10] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        uint32_t lh_want[10] = { 1, 1, 3, 4, 5, 6, 7, 7, 9, 9 };
        run_words(ctx, "ldst_hints.ptx", lh_in, 10, lh_want,
                  "ld/st ordering + cache hints");
        uint32_t sc_in[2] = { 0, 0 };
        uint32_t sc_want[2] = { 0, 0x1234u };
        run_words(ctx, "ldst_shared_cta.ptx", sc_in, 2, sc_want,
                  "ld/st .shared::cta");
        uint32_t lf_in[56] = { 0 };
        uint32_t lf_want[56] = { 0 };
        for (unsigned i = 0; i < 24; i++) {
            lf_in[i] = 0x100u + i;
            lf_want[i] = lf_in[i];
        }
        for (unsigned i = 0; i < 5; i++)
            lf_want[24 + i] = 0x100u + i;
        for (unsigned i = 0; i < 4; i++)
            lf_want[29 + i] = 0x104u + i;
        for (unsigned i = 0; i < 8; i++)
            lf_want[33 + i] = 0x108u + i;
        for (unsigned i = 0; i < 8; i++)
            lf_want[41 + i] = 0x110u + i;
        run_words(ctx, "ld_forms.ptx", lf_in, 56, lf_want,
                  "ld qualifier/vector forms");
    }

    // Buffer-backed 2D texture: tex.2d (nearest), txq, istypep.
    {
        float txel[16] = { 1, 2, 3, 4, 5, 6, 7, 8,
                           9, 10, 11, 12, 13, 14, 15, 16 };
        acuda_image_desc idesc = { 2, 2, 1, 1, 1, 1, 0, 0, 0 };
        acuda_sampler_desc sdesc = { 0, 0, { 2, 2, 2 }, 1,
                                     { 0, 0, 0, 0 }, 0, 0 };
        snprintf(path, sizeof(path), "%s/tex2d.ptx", ACUDA_TEST_PTX_DIR);
        char *tp = read_file(path);
        CHECK(tp != NULL);
        if (tp) {
            acuda_kernel_t *tk = NULL;
            char terr[1024] = {0};
            CHECK_ST(acuda_compile_ptx(tp, "tex2d", &tk, terr,
                                       sizeof(terr), &g_test_caps));
            if (tk) {
                acuda_buffer_t img = NULL, smp = NULL;
                acuda_buffer_t tex = NULL, out = NULL;
                CHECK_ST(acuda_image_create(ctx, &img, &idesc, txel));
                CHECK_ST(acuda_sampler_create(ctx, &smp, &sdesc));
                CHECK_ST(acuda_texture_create_image(ctx, &tex, img, smp));
                CHECK_ST(acuda_malloc(ctx, &out, 32));
                acuda_kernel_info_t ti;
                CHECK_ST(acuda_kernel_info(tk, &ti));
                CHECK(ti.num_ptr_params == 2);
                CHECK(ti.is_tex[0] == 1);
                acuda_buffer_t tb[2] = { tex, out };
                CHECK_ST(acuda_launch_kernel(ctx, tk, tb, 2, NULL, 0,
                                             1, 1, NULL));
                CHECK_ST(acuda_stream_synchronize(ctx, NULL));
                uint32_t got[7] = {0};
                CHECK_ST(acuda_memcpy_d2h(ctx, got, out, 28, NULL));
                static const uint32_t tw[7] = {
                    0x3f800000u, 0x40000000u, 0x40400000u, 0x40800000u,
                    2, 2, 1
                };
                int tbad = 0;
                for (int i = 0; i < 7; i++)
                    if (got[i] != tw[i])
                        tbad++;
                if (tbad)
                    printf("FAIL: tex2d (%d/%d words)\n", tbad, 7);
                else
                    printf("  tex2d: 7 words ok\n");
                failures += tbad ? 1 : 0;
                // tld4.r gather (counter-clockwise from lower left).
                snprintf(path, sizeof(path), "%s/tld4.ptx",
                         ACUDA_TEST_PTX_DIR);
                char *t4p = read_file(path);
                if (t4p) {
                    acuda_kernel_t *t4k = NULL;
                    char e4[1024] = {0};
                    CHECK_ST(acuda_compile_ptx(t4p, "tld4test", &t4k, e4,
                                               sizeof(e4), &g_test_caps));
                    if (t4k) {
                        acuda_buffer_t o4 = NULL;
                        CHECK_ST(acuda_malloc(ctx, &o4, 16));
                        acuda_buffer_t b4[2] = { tex, o4 };
                        CHECK_ST(acuda_launch_kernel(ctx, t4k, b4, 2, NULL,
                                                     0, 1, 1, NULL));
                        CHECK_ST(acuda_stream_synchronize(ctx, NULL));
                        uint32_t g4[4] = {0};
                        CHECK_ST(acuda_memcpy_d2h(ctx, g4, o4, 16, NULL));
                        static const uint32_t w4[4] = {
                            0x41100000u, 0x41500000u, 0x40a00000u,
                            0x3f800000u
                        };
                        int b4bad = 0;
                        for (int i = 0; i < 4; i++)
                            if (g4[i] != w4[i])
                                b4bad++;
                        if (b4bad)
                            printf("FAIL: tld4 (%d/4 words)\n", b4bad);
                        else
                            printf("  tld4: 4 words ok\n");
                        failures += b4bad ? 1 : 0;
                        acuda_free(ctx, o4);
                        acuda_kernel_free(t4k);
                    }
                    free(t4p);
                }
                // Linear filtering: average of the four texels.
                {
                    acuda_sampler_desc ls = sdesc;
                    ls.filter = 1;
                    acuda_buffer_t lsam = NULL, ltex = NULL;
                    CHECK_ST(acuda_sampler_create(ctx, &lsam, &ls));
                    CHECK_ST(acuda_texture_create_image(ctx, &ltex, img,
                                                        lsam));
                    snprintf(path, sizeof(path), "%s/tex2d_linear.ptx",
                             ACUDA_TEST_PTX_DIR);
                    char *lpp = read_file(path);
                    if (lpp) {
                        acuda_kernel_t *lk = NULL;
                        char le[1024] = {0};
                        CHECK_ST(acuda_compile_ptx(lpp, "tex2dlin", &lk, le,
                                                   sizeof(le), &g_test_caps));
                        if (lk) {
                            acuda_buffer_t lo = NULL;
                            CHECK_ST(acuda_malloc(ctx, &lo, 32));
                            acuda_buffer_t lb[2] = { ltex, lo };
                            CHECK_ST(acuda_launch_kernel(ctx, lk, lb, 2,
                                                         NULL, 0, 1, 1,
                                                         NULL));
                            CHECK_ST(acuda_stream_synchronize(ctx, NULL));
                            uint32_t lg[7] = {0};
                            CHECK_ST(acuda_memcpy_d2h(ctx, lg, lo, 28,
                                                      NULL));
                            static const uint32_t lw[7] = {
                                0x40e00000u, 0x41000000u, 0x41100000u,
                                0x41200000u, 2, 2, 1
                            };
                            int lbad = 0;
                            for (int i = 0; i < 7; i++)
                                if (lg[i] != lw[i])
                                    lbad++;
                            if (lbad)
                                printf("FAIL: tex2d linear (%d/7 words)\n",
                                       lbad);
                            else
                                printf("  tex2d linear: 7 words ok\n");
                            failures += lbad ? 1 : 0;
                            acuda_free(ctx, lo);
                            acuda_kernel_free(lk);
                        }
                        free(lpp);
                    }
                    acuda_free(ctx, ltex);
                    acuda_free(ctx, lsam);
                }
                acuda_free(ctx, tex);
                acuda_free(ctx, out);
                acuda_free(ctx, img);
                acuda_free(ctx, smp);
                acuda_kernel_free(tk);
            }
            free(tp);
        }
    }

    // Image-backed texture geometries, sampler modes, mipmaps, offsets,
    // depth compare and multisample fetches.
    {
        static const uint32_t w1d[4] = {
            0x41100000u, 0x41200000u, 0x41300000u, 0x41400000u
        };
        acuda_image_desc d1 = { 4, 1, 1, 1, 1, 1, 0, 1, 0 };
        run_tex_image(ctx, "tex1d.ptx", "tex1d", &d1, 0, 0, 0, w1d, 4,
                      "tex.1d");
        static const uint32_t w3d[4] = {
            0x41e80000u, 0x41f00000u, 0x41f80000u, 0x42000000u
        };
        acuda_image_desc d3 = { 2, 2, 2, 1, 1, 1, 0, 3, 0 };
        run_tex_image(ctx, "tex3d.ptx", "tex3d", &d3, 0, 0, 0, w3d, 4,
                      "tex.3d");
        static const uint32_t wa2d[4] = {
            0x41e80000u, 0x41f00000u, 0x41f80000u, 0x42000000u
        };
        acuda_image_desc da2 = { 2, 2, 1, 2, 1, 1, 0, 2, 0 };
        run_tex_image(ctx, "texa2d.ptx", "texa2d", &da2, 0, 0, 0, wa2d, 4,
                      "tex.a2d");
        static const uint32_t wcube[4] = {
            0x429a0000u, 0x429c0000u, 0x429e0000u, 0x42a00000u
        };
        acuda_image_desc dc = { 2, 2, 1, 6, 1, 1, 0, 2, 1 };
        run_tex_image(ctx, "texcube.ptx", "texcube", &dc, 0, 0, 0, wcube,
                      4, "tex.cube");
        static const uint32_t wsmp[4] = {
            0x3f800000u, 0x40000000u, 0x40400000u, 0x40800000u
        };
        acuda_image_desc dsmp = { 2, 2, 1, 1, 1, 1, 0, 2, 0 };
        run_tex_image(ctx, "texsampler.ptx", "texsampler", &dsmp, 0, 0, 0,
                      wsmp, 4, "tex explicit sampler");
        static const uint32_t wlvl[4] = {
            0x447a4000u, 0x447a8000u, 0x447ac000u, 0x447b0000u
        };
        acuda_image_desc dlv = { 2, 2, 1, 1, 2, 1, 0, 2, 0 };
        run_tex_image(ctx, "texlevel.ptx", "texlevel", &dlv, 0, 0, 0,
                      wlvl, 4, "tex.level");
        static const uint32_t wgrad[4] = {
            0x3f800000u, 0x40000000u, 0x40400000u, 0x40800000u
        };
        acuda_image_desc dgrad = { 2, 2, 1, 1, 2, 1, 0, 2, 0 };
        run_tex_image(ctx, "texgrad.ptx", "texgrad", &dgrad, 0, 0, 0,
                      wgrad, 4, "tex.grad");
        static const uint32_t woff[4] = {
            0x41500000u, 0x41600000u, 0x41700000u, 0x41800000u
        };
        acuda_image_desc doff = { 2, 2, 1, 1, 1, 1, 0, 2, 0 };
        run_tex_image(ctx, "texoff.ptx", "texoff", &doff, 0, 0, 0, woff, 4,
                      "tex offset");
        static const uint32_t wcmp[4] = {
            0x3f800000u, 0x3f800000u, 0x3f800000u, 0x3f800000u
        };
        acuda_image_desc dcmp = { 2, 2, 1, 1, 1, 1, 4, 2, 0 };
        run_tex_image(ctx, "texcmp.ptx", "texcmp", &dcmp, 0, 1, 1, wcmp,
                      4, "tex depth compare");
        static const uint32_t wms[4] = {
            0x41500000u, 0x41600000u, 0x41700000u, 0x41800000u
        };
        acuda_image_desc dms = { 2, 2, 1, 1, 1, 4, 0, 2, 0 };
        run_tex_image(ctx, "texms4.ptx", "texms4", &dms, 0, 0, 0, wms, 4,
                      "tex.2dms 4x");
        static const uint32_t wmss[6] = {
            0x41500000u, 0x41600000u, 0x41700000u, 0x41800000u, 4, 2
        };
        acuda_image_desc dmss = { 2, 2, 1, 1, 1, 4, 0, 2, 0 };
        run_tex_image(ctx, "texmss.ptx", "texmss", &dmss, 0, 0, 0, wmss,
                      6, "tex.2dms num_samples");
        static const uint32_t wams4[4] = {
            0x41e80000u, 0x41f00000u, 0x41f80000u, 0x42000000u
        };
        acuda_image_desc dams4 = { 2, 2, 1, 2, 1, 4, 0, 2, 0 };
        run_tex_image(ctx, "texams4.ptx", "texams4", &dams4, 0, 0, 0,
                      wams4, 4, "tex.a2dms 4x");
        static const uint32_t wt4[4] = {
            0x41c80000u, 0x41e80000u, 0x41a80000u, 0x41880000u
        };
        acuda_image_desc dt4 = { 2, 2, 1, 2, 1, 1, 0, 2, 0 };
        run_tex_image(ctx, "tld4a2d.ptx", "tld4a2d", &dt4, 0, 0, 0, wt4,
                      4, "tld4.a2d");
        static const uint32_t wt4o[4] = {
            0x41100000u, 0x41500000u, 0x40a00000u, 0x3f800000u
        };
        acuda_image_desc dt4o = { 2, 2, 1, 1, 1, 1, 0, 2, 0 };
        run_tex_image(ctx, "tld4off.ptx", "tld4off", &dt4o, 0, 0, 0,
                      wt4o, 4, "tld4 offset");
        static const uint32_t wqa[6] = { 2, 2, 1, 2, 2, 0 };
        acuda_image_desc dqa = { 2, 2, 1, 2, 2, 1, 0, 2, 0 };
        run_tex_image(ctx, "txqa2d.ptx", "txqa2d", &dqa, 0, 0, 0, wqa, 6,
                      "txq.a2d");
        static const uint32_t wq3[5] = { 2, 2, 2, 0, 0 };
        acuda_image_desc dq3 = { 2, 2, 2, 1, 1, 1, 0, 3, 0 };
        run_tex_image(ctx, "txq3d.ptx", "txq3d", &dq3, 0, 0, 0, wq3, 5,
                      "txq.3d");
        static const uint32_t wlv[2] = { 1, 1 };
        acuda_image_desc dlvl = { 2, 2, 1, 1, 2, 1, 0, 2, 0 };
        run_tex_image(ctx, "txqlvl.ptx", "txqlvl", &dlvl, 0, 0, 0, wlv, 2,
                      "txq.level");
        static const uint32_t wa1d[4] = {
            0x42240000u, 0x42280000u, 0x422c0000u, 0x42300000u
        };
        acuda_image_desc da1 = { 4, 1, 1, 3, 1, 1, 0, 1, 0 };
        run_tex_image(ctx, "texa1d.ptx", "texa1d", &da1, 0, 0, 0, wa1d, 4,
                      "tex.a1d");
        static const uint32_t wac[4] = {
            0x432d0000u, 0x432e0000u, 0x432f0000u, 0x43300000u
        };
        acuda_image_desc dac = { 2, 2, 1, 12, 1, 1, 0, 2, 1 };
        run_tex_image(ctx, "texacube.ptx", "texacube", &dac, 0, 0, 0, wac,
                      4, "tex.acube");
        static const uint32_t wt4c[4] = {
            0x3f800000u, 0x3f800000u, 0x3f800000u, 0x3f800000u
        };
        acuda_image_desc dt4c = { 2, 2, 1, 1, 1, 1, 4, 2, 0 };
        run_tex_image(ctx, "tld4cmp.ptx", "tld4cmp", &dt4c, 0, 1, 1,
                      wt4c, 4, "tld4 depth compare");
        static const uint32_t wpred[5] = {
            0x3f800000u, 0x40000000u, 0x40400000u, 0x40800000u, 1
        };
        acuda_image_desc dpred = { 2, 2, 1, 1, 1, 1, 0, 2, 0 };
        run_tex_image(ctx, "texpred.ptx", "texpred", &dpred, 0, 0, 0,
                      wpred, 5, "tex destination predicate");
        static const uint32_t wqonly[11] = {
            2, 2, 2, 0, 0, 0, 1, 1, 3, 4, 1
        };
        acuda_image_desc dqonly = { 2, 2, 2, 1, 1, 1, 0, 3, 0 };
        {
            // txq-only kernel on a 3D image with a mirror/clamp/border
            // linear sampler: queries come from runtime metadata.
            char path2[1024];
            snprintf(path2, sizeof(path2), "%s/txqonly.ptx",
                     ACUDA_TEST_PTX_DIR);
            char *qp = read_file(path2);
            CHECK(qp != NULL);
            if (qp) {
                acuda_kernel_t *qk = NULL;
                char qe[1024] = {0};
                CHECK_ST(acuda_compile_ptx(qp, "txqonly", &qk, qe,
                                           sizeof(qe), &g_test_caps));
                if (qk) {
                    acuda_buffer_t qimg = NULL, qsmp = NULL, qtex = NULL;
                    acuda_buffer_t qout = NULL;
                    float qdata[32];
                    for (int i = 0; i < 32; i++)
                        qdata[i] = (float)(i + 1);
                    CHECK_ST(acuda_image_create(ctx, &qimg, &dqonly, qdata));
                    acuda_sampler_desc qsd = { 0 };
                    qsd.filter = 1;
                    qsd.addr[0] = 1;
                    qsd.addr[1] = 2;
                    qsd.addr[2] = 3;
                    qsd.normalized = 1;
                    CHECK_ST(acuda_sampler_create(ctx, &qsmp, &qsd));
                    CHECK_ST(acuda_texture_create_image(ctx, &qtex, qimg,
                                                        qsmp));
                    CHECK_ST(acuda_malloc(ctx, &qout, 64));
                    acuda_buffer_t qb[2] = { qtex, qout };
                    CHECK_ST(acuda_launch_kernel(ctx, qk, qb, 2, NULL, 0,
                                                 1, 1, NULL));
                    CHECK_ST(acuda_stream_synchronize(ctx, NULL));
                    uint32_t qgot[11] = {0};
                    CHECK_ST(acuda_memcpy_d2h(ctx, qgot, qout, 44, NULL));
                    int qbad = 0;
                    for (int i = 0; i < 11; i++)
                        if (qgot[i] != wqonly[i])
                            qbad++;
                    if (qbad)
                        printf("FAIL: txq-only 3D (%d/11 words)\n", qbad);
                    else
                        printf("  txq-only 3D: 11 words ok\n");
                    failures += qbad ? 1 : 0;
                    acuda_free(ctx, qout);
                    acuda_free(ctx, qtex);
                    acuda_free(ctx, qsmp);
                    acuda_free(ctx, qimg);
                    acuda_kernel_free(qk);
                }
                free(qp);
            }
        }
        static const uint32_t wqsmp[3] = { 1, 3, 0 };
        acuda_image_desc dsmpq = { 2, 2, 1, 1, 1, 1, 0, 2, 0 };
        run_tex_image(ctx, "txqsmp.ptx", "txqsmp", &dsmpq, 1, 0, 0,
                      wqsmp, 3, "txq samplerref");
        static const uint32_t wf16[2] = { 0x40003c00u, 0x44004200u };
        acuda_image_desc df16 = { 2, 2, 1, 1, 1, 1, 0, 2, 0 };
        run_tex_image(ctx, "texf16.ptx", "texf16", &df16, 0, 0, 0, wf16,
                      2, "tex.v4.f16");
        static const uint32_t wf16x2[2] = { 0x40003c00u, 0x44004200u };
        acuda_image_desc df16x2 = { 2, 2, 1, 1, 1, 1, 0, 2, 0 };
        run_tex_image(ctx, "texf16x2.ptx", "texf16x2", &df16x2, 0, 0, 0,
                      wf16x2, 2, "tex.v2.f16x2");
    }

    // Convenience acuda_texture_create path (image + sampler owner).
    {
        float txel[16] = { 1, 2, 3, 4, 5, 6, 7, 8,
                           9, 10, 11, 12, 13, 14, 15, 16 };
        acuda_texture_desc td = { 2, 2, 1, 4, 0, 0, 1, { 2, 2, 2 } };
        snprintf(path, sizeof(path), "%s/tex2d.ptx", ACUDA_TEST_PTX_DIR);
        char *tp = read_file(path);
        CHECK(tp != NULL);
        if (tp) {
            acuda_kernel_t *tk = NULL;
            char terr[1024] = {0};
            CHECK_ST(acuda_compile_ptx(tp, "tex2d", &tk, terr,
                                       sizeof(terr), &g_test_caps));
            if (tk) {
                acuda_buffer_t tex = NULL, out = NULL;
                CHECK_ST(acuda_texture_create(ctx, &tex, &td, txel));
                CHECK_ST(acuda_malloc(ctx, &out, 32));
                acuda_buffer_t tb[2] = { tex, out };
                CHECK_ST(acuda_launch_kernel(ctx, tk, tb, 2, NULL, 0, 1, 1,
                                             NULL));
                CHECK_ST(acuda_stream_synchronize(ctx, NULL));
                uint32_t got[7] = {0};
                CHECK_ST(acuda_memcpy_d2h(ctx, got, out, 28, NULL));
                static const uint32_t tw[7] = {
                    0x3f800000u, 0x40000000u, 0x40400000u, 0x40800000u,
                    2, 2, 1
                };
                int tbad = 0;
                for (int i = 0; i < 7; i++)
                    if (got[i] != tw[i])
                        tbad++;
                if (tbad)
                    printf("FAIL: acuda_texture_create (%d/7 words)\n",
                           tbad);
                else
                    printf("  acuda_texture_create: 7 words ok\n");
                failures += tbad ? 1 : 0;
                acuda_free(ctx, out);
                acuda_free(ctx, tex);
                acuda_kernel_free(tk);
            }
            free(tp);
        }
    }

    // Async copy: bulk global->shared->global round trip.
    {
        uint32_t din[32] = {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        };
        uint32_t dwant[32] = {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
        };
        run_words_block(ctx, "cp_async_bulk.ptx", din, 32, dwant,
                        "cp.async.bulk", 1);
    }

    // Async copy: runtime source size and ignore-src predicate.
    {
        uint32_t din[24] = {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
            0, 0, 0, 0, 0, 0, 0, 0
        };
        uint32_t dwant[24] = {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
            1, 2, 0, 0, 0, 0, 0, 0
        };
        run_words_block(ctx, "cp_async_forms.ptx", din, 24, dwant,
                        "cp.async forms", 1);
    }

    // elect.sync and mbarrier.
    {
        uint32_t din[8] = { 0 };
        uint32_t dwant[8] = { 0, 1, 0, 0, 0, 0, 0, 0 };
        run_words_block(ctx, "elect_sync.ptx", din, 8, dwant,
                        "elect.sync", 4);
        CHECK(kernel_uses_subgroup("elect_sync.ptx") == 1);
        CHECK(kernel_uses_subgroup("vecadd.ptx") == 0);
        // Mixed BDA: tex2d has a .texref plus an output pointer, so the
        // plain pointer uses PhysicalStorageBuffer while the texture
        // keeps its descriptor binding.
        {
            int bda = kernel_uses_bda("vecadd.ptx");
            CHECK(bda == 0 || bda == 1);
            if (bda == 1)
                CHECK(kernel_uses_bda("tex2d.ptx") == 1);
            acuda_compiler_set_bda(&g_test_caps, 0);
            CHECK(kernel_uses_bda("vecadd.ptx") == 0);
            CHECK(kernel_uses_bda("tex2d.ptx") == 0);
            acuda_compiler_set_bda(&g_test_caps, bda);
        }
        uint32_t min[1] = { 0 };
        uint32_t mwant[1] = { 1 };
        run_words_block(ctx, "mbarrier_basic.ptx", min, 1, mwant,
                        "mbarrier arrive/wait", 4);
        uint32_t tdin[6] = { 0 };
        uint32_t twant[6] = { 4, 0, 0, 0, 1, 1 };
        run_words_block(ctx, "mbarrier_tx.ptx", tdin, 6, twant,
                        "mbarrier tx/pending", 1);
        uint32_t ldin[4] = { 0 };
        uint32_t lwant[4] = { 1, 0, 1, 1 };
        run_words_block(ctx, "mbarrier_layout.ptx", ldin, 4, lwant,
                        "mbarrier layout/arrive_drop", 4);
        uint32_t pd[2] = { 0 };
        uint32_t pw[2] = { 1, 1 };
        run_words_block(ctx, "mbarrier_pred.ptx", pd, 2, pw,
                        "mbarrier predication", 1);
        uint32_t cdin[6] = { 0x11111111u, 0, 0x22222222u, 0, 0, 0 };
        uint32_t cwant[6] = { 0x11111111u, 1, 0x22222222u, 1,
                              0x11111111u, 0x22222222u };
        run_words_block(ctx, "cp_async_mbarrier.ptx", cdin, 6, cwant,
                        "cp.async.mbarrier.arrive", 1);
        // red.async: release global form (u32/s32/u64/s64 + mmio) and
        // the relaxed shared-cluster form with mbarrier completion.
        uint32_t rdin[12] = { 0 };
        uint32_t rwant[12] = { 4, 0, 0, 0, 12, 0, 20, 0,
                               0xfffffffcu, 0, 4, 0 };
        run_words_block(ctx, "red_async_global.ptx", rdin, 12, rwant,
                        "red.async global", 4);
        uint32_t mdin[4] = { 0 };
        uint32_t mmwant[4] = { 4, 8, 12, 0 };
        run_words_block(ctx, "multimem_red_async.ptx", mdin, 4, mmwant,
                        "multimem.red.async", 4);
        uint32_t midin[8] = { 1, 5, 9, 13, 0, 0x40200000u, 0, 0 };
        uint32_t midwant[8] = { 1, 5, 1, 20, 3, 0x40200000u,
                                0x40200000u, 0 };
        run_words_block(ctx, "multimem_ldst.ptx", midin, 8, midwant,
                        "multimem ld_reduce/st/red", 1);
        uint32_t clin[4] = { 0 };
        uint32_t clwant[4] = { 0, 123, 0, 0 };
        run_words_block(ctx, "cluster_rank.ptx", clin, 4, clwant,
                        "cluster mapa/getctarank", 1);
        uint32_t sdin[10] = { 0 };
        uint32_t swant[10] = { 1, 5, 3, 9, 0x0f, 0x30, 0x3a, 6, 4,
                               0xfffffff9u };
        run_words_block(ctx, "red_async_shared.ptx", sdin, 10, swant,
                        "red.async shared", 1);
    }

    // dp4a native integer dot product (SPV_KHR_integer_dot_product):
    // correctness on both the native and the exact-expansion path.
    {
        uint32_t din[7] = { 0 };
        uint32_t dwant[7] = { 170, 0xfffffffdu, 4, 6, 20, 15, 4 };
        run_words(ctx, "dp4a.ptx", din, 7, dwant, "dp4a/dp2a");
        char path[1024];
        snprintf(path, sizeof(path), "%s/dp4a.ptx", ACUDA_TEST_PTX_DIR);
        char *ptx = read_file(path);
        acuda_kernel_t *k = NULL;
        char err[1024];
        int native = 0;
        if (ptx &&
            acuda_compile_ptx(ptx, NULL, &k, err, sizeof(err), &g_test_caps) ==
                ACUDA_SUCCESS) {
            const uint32_t *w = NULL;
            size_t n = 0;
            CHECK_ST(acuda_kernel_spirv(k, &w, &n));
            native = spv_has_dot_product(w, n);
            printf("  dp4a: %s path\n",
                   native ? "native OpSDot/UDot/SUDot"
                          : "exact multiply-add");
            acuda_kernel_free(k);
        }
        free(ptx);
        acuda_compiler_set_dot_product(&g_test_caps, 0);
        run_words(ctx, "dp4a.ptx", din, 7, dwant, "dp4a/dp2a fallback");
        static const uint32_t dsat_in[8] = { 0 };
        static const uint32_t dsat_want[8] = {
            0x0000000eu, 0x7fffffffu, 0x80000000u, 0xffffffffu,
            0x00000019u, 1u, 1u, 0x80000000u
        };
        run_words(ctx, "dp4a_sat.ptx", dsat_in, 8, dsat_want,
                  "acuda.dp4a.sat");
        run_words(ctx, "dp4a_sat.ptx", dsat_in, 8, dsat_want,
                  "acuda.dp4a.sat fallback");
        acuda_compiler_set_dot_product(&g_test_caps, native);
        check_dp4a_sat_native();
    }

    // Surface instructions: suld/sust/sured/suq on storage images.
    {
        static const uint32_t sdata2[8] = {
            10, 20, 30, 40, 50, 60, 70, 80
        };
        static const uint32_t swant2[3] = { 20, 4, 2 };
        static const uint32_t swant2s[8] = {
            10, 20, 999, 45, 50, 60, 70, 80
        };
        acuda_image_desc sd2 = { 4, 2, 1, 1, 1, 1, 3, 2, 0 };
        run_surface(ctx, "surf2d.ptx", "surf2d", &sd2, sdata2, 8, swant2,
                    3, swant2s, "surface.2d");
        run_surface(ctx, "surf2d_u64.ptx", "surf2d_u64", &sd2, sdata2, 8,
                    swant2, 3, swant2s, "surface.2d u64");
        static const uint32_t swanta[2] = { 60, 2 };
        static const uint32_t swantas[8] = {
            10, 20, 30, 40, 50, 60, 777, 80
        };
        acuda_image_desc sda = { 4, 1, 1, 2, 1, 1, 3, 2, 0 };
        run_surface(ctx, "surfa2d.ptx", "surfa2d", &sda, sdata2, 8,
                    swanta, 2, swantas, "surface.a2d");
        static const uint32_t sdata3[4] = { 11, 22, 33, 44 };
        static const uint32_t swant3[1] = { 33 };
        acuda_image_desc sd3 = { 2, 1, 2, 1, 1, 1, 3, 3, 0 };
        run_surface(ctx, "surf3d.ptx", "surf3d", &sd3, sdata3, 4, swant3,
                    1, NULL, "surface.3d");
        static const uint32_t sdataf[16] = {
            0x3f800000u, 0x40000000u, 0x40400000u, 0x40800000u,
            0x40a00000u, 0x40c00000u, 0x40e00000u, 0x41000000u,
            0x41100000u, 0x41200000u, 0x41300000u, 0x41400000u,
            0x41500000u, 0x41600000u, 0x41700000u, 0x41800000u
        };
        static const uint32_t swantp[1] = { 2 };
        static const uint32_t swantps[16] = {
            0x3f800000u, 0x40000000u, 0x40400000u, 0x40800000u,
            0x40a00000u, 0x40c00000u, 0x40e00000u, 0x41000000u,
            0x41100000u, 0x41200000u, 0x41300000u, 0x41400000u,
            0x41100000u, 0x41000000u, 0x40e00000u, 0x40c00000u
        };
        acuda_image_desc sdp = { 2, 2, 1, 1, 1, 1, 0, 2, 0 };
        run_surface(ctx, "surfp.ptx", "surfp", &sdp, sdataf, 16, swantp,
                    1, swantps, "surface.sust.p");
        static const uint32_t smmin[4] = {
            0x41200000u, 0x41A00000u, 0x41F00000u, 0x42200000u
        }; // 10, 20, 30, 40
        static const uint32_t smmin_want[4] = {
            0x40A00000u, 0x41C80000u, 0x41F00000u, 0x42200000u
        }; // 5, 25, 30, 40
        static const uint32_t smmin_out[1] = { 1 };
        acuda_image_desc smm = { 4, 1, 1, 1, 1, 1, 2, 2, 0 };
        run_surface(ctx, "sured_minmax_f32.ptx", "sured_minmax_f32",
                    &smm, smmin, 4, smmin_out, 1, smmin_want,
                    "surface.sured f32 min/max");
        static const uint32_t sadd[4] = {
            0x3F800000u, 0, 0, 0 // 1, 0, 0, 0
        };
        static const uint32_t sadd_want[4] = {
            0x40000000u, 0, 0, 0 // 2, 0, 0, 0
        };
        static const uint32_t sadd_out[1] = { 0 };
        run_surface(ctx, "sured_f32.ptx", "sured_f32", &smm, sadd, 4,
                    sadd_out, 1, sadd_want, "surface.sured f32 add");
        {
            static const char *files[] = {
                "sured_dyn_minmax.ptx", "sured_mixed_f32.ptx"
            };
            for (unsigned fi = 0; fi < 2; fi++) {
                char cp[1024];
                snprintf(cp, sizeof(cp), "%s/%s", ACUDA_TEST_PTX_DIR,
                         files[fi]);
                char *cptx = read_file(cp);
                CHECK(cptx != NULL);
                if (cptx) {
                    acuda_kernel_t *ck = NULL;
                    char cerr[1024] = {0};
                    acuda_status_t cst = acuda_compile_ptx(
                        cptx, NULL, &ck, cerr, sizeof(cerr), &g_test_caps);
                    if (cst != ACUDA_SUCCESS)
                        printf("  fallback compile %s: %s\n", files[fi],
                               cerr);
                    CHECK(cst == ACUDA_SUCCESS);
                    if (ck)
                        acuda_kernel_free(ck);
                    free(cptx);
                }
            }
        }
        static const uint32_t smc[2] = {
            0x47C35000u, // 100000.0
            0xC7C35000u  // -100000.0
        };
        static const uint32_t smc_want[2] = {
            0xC6000000u, // -8192.0
            0x45FFF800u  // 8191.0
        };
        acuda_image_desc smcd = { 2, 1, 1, 1, 1, 1, 2, 2, 0 };
        run_surface_many(ctx, "sured_minmax_contended.ptx",
                         "sured_minmax_contended", &smcd, smc, 2, 128,
                         128, smc_want, "surface.sured f32 contended");
        static const uint32_t s64data[12] = {
            5, 0, 10, 0, 10, 0, 0, 0xFFFFFFFFu, 0x0Fu, 0, 0, 0
        };
        static const uint32_t s64want[12] = {
            8, 0, 7, 0, 20, 0, 0, 0, 0xFFu, 0, 8, 0
        };
        static const uint32_t s64out[1] = { 1 };
        acuda_image_desc s64d = { 6, 1, 1, 1, 1, 1, 5, 2, 0 };
        run_surface(ctx, "sured_u64.ptx", "sured_u64", &s64d, s64data,
                    12, s64out, 1, s64want, "surface.sured/ld/st u64");
        static const uint32_t sdata1[4] = { 10, 20, 30, 40 };
        static const uint32_t swant1[1] = { 30 };
        static const uint32_t swant1s[4] = { 10, 20, 30, 1234 };
        acuda_image_desc sd1 = { 4, 1, 1, 1, 1, 1, 3, 1, 0 };
        run_surface(ctx, "surf1d.ptx", "surf1d", &sd1, sdata1, 4, swant1,
                    1, swant1s, "surface.1d");
        static const uint32_t sdatab[4] = {
            0x11223344u, 0xaabbccddu, 100, 40
        };
        static const uint32_t swantb[4] = {
            0x33u, 0x1122u, 0x11223344u, 0xaabbccddu
        };
        static const uint32_t swantbs[4] = {
            0x11223344u, 0xaabbccddu, 0x1111u, 0x20u
        };
        acuda_image_desc sdb = { 4, 1, 1, 1, 1, 1, 3, 2, 0 };
        run_surface(ctx, "surfbits.ptx", "surfbits", &sdb, sdatab, 4,
                    swantb, 4, swantbs, "surface.bits");
    }

    // Micro-float / block-scaled cvt forms (f4/f6/fp8/ue8m0/s2f6).
    {
        uint32_t din[31] = { 0 };
        uint32_t dwant[31] = {
            0x24, 0x0810, 0x0C10, 0x3C004000u, 0x7F80, 0x407F,
            0x0C02, 0x3FC03E80u, 0x3FC03F00u, 0x3F804000u,
            0x7F7D, 0x807E, 0x31, 0x0C04, 0x0E08, 0x3FC03F00u,
            0x3FC03F80u, 0x3F803F80u, 0x7F20, 0xF0, 0x77, 0,
            0xFF7F, 0xFE00, 0x7F7F, 0x8181, 0x66, 0x12,
            0x38003780u, 0x3C004000u, 0x100
        };
        run_words_block(ctx, "cvt_micro.ptx", din, 31, dwant,
                        "cvt micro-float", 1);
    }

    // .common module globals: zero-initialized, writable, addressable.
    {
        uint32_t din[6] = { 0 };
        uint32_t dwant[6] = { 0, 0, 42, 0, 7, 7 };
        run_words_block(ctx, "common.ptx", din, 6, dwant,
                        "common globals", 1);
    }

    // Module/function-scope .shared, .const and .local declarations.
    {
        uint32_t din[5] = { 0 };
        uint32_t dwant[5] = { 11, 22, 33, 9, 0xc0200000u };
        run_words_block(ctx, "declspaces.ptx", din, 5, dwant,
                        "declaration spaces", 1);
    }

    // Legacy module-scope .texref declaration: accepted by the parser and
    // ignored by kernels that do not sample it.
    {
        uint32_t tin[32] = { 0 };
        uint32_t twant[32];
        for (int i = 0; i < 32; i++)
            twant[i] = 7;
        run_words_block(ctx, "texref_decl.ptx", tin, 32, twant,
                        "module .texref decl", 32);
    }

    // Legacy shfl/vote spellings and vote.sync.uni.
    {
        uint32_t legin[256] = { 0 };
        uint32_t legwant[256];
        for (unsigned l = 0; l < 32; l++) {
            unsigned b = l * 8;
            legwant[b + 0] = l == 0 ? 0u : l - 1;
            legwant[b + 1] = l == 31 ? 31u : l + 1;
            legwant[b + 2] = l ^ 1u;
            legwant[b + 3] = 0;
            legwant[b + 4] = 0;
            legwant[b + 5] = 1;
            legwant[b + 6] = 0;
            legwant[b + 7] = 0xffffu;
        }
        run_words_block(ctx, "warp_legacy.ptx", legin, 256, legwant,
                        "warp_legacy", 32);

        uint32_t uniin[160] = { 0 };
        uint32_t uniwant[160];
        for (unsigned l = 0; l < 32; l++) {
            unsigned b = l * 5;
            uniwant[b + 0] = 1; // all true
            uniwant[b + 1] = 1; // all false
            uniwant[b + 2] = 0; // mixed
            uniwant[b + 3] = 0; // mixed .all
            uniwant[b + 4] = 1; // mixed .any
        }
        run_words_block(ctx, "warp_uni.ptx", uniin, 160, uniwant,
                        "warp_uni", 32);
    }

    // Counted barriers and bar.arrive (shared-memory counter lowering).
    {
        uint32_t barin[64] = { 0 };
        uint32_t barwant[64];
        for (unsigned i = 0; i < 64; i++)
            barwant[i] = i;
        run_words_block(ctx, "counted_bar.ptx", barin, 64, barwant,
                        "counted_bar", 64);
        uint32_t bfin[192] = { 0 };
        uint32_t bfwant[192];
        for (unsigned t = 0; t < 32; t++) {
            unsigned b = t * 6;
            bfwant[b + 0] = (t + 1) & 31;
            bfwant[b + 1] = 10; // bar.red.popc
            bfwant[b + 2] = 1;  // bar.red.or
            bfwant[b + 3] = 0;  // bar.red.and
            bfwant[b + 4] = 22; // complemented popc
            bfwant[b + 5] = 10; // counted red popc
        }
        run_words_block(ctx, "bar_forms.ptx", bfin, 192, bfwant,
                        "barrier forms", 32);
        uint32_t cbin[32] = { 0 };
        uint32_t cbwant[32];
        for (unsigned t = 0; t < 32; t++)
            cbwant[t] = (t + 1) & 31;
        run_words_block(ctx, "cluster_bar.ptx", cbin, 32, cbwant,
                        "cluster barrier", 32);
        uint32_t ffin[32] = { 0 };
        uint32_t ffwant[32];
        for (unsigned t = 0; t < 32; t++)
            ffwant[t] = (t + 1) & 31;
        run_words_block(ctx, "fence_forms.ptx", ffin, 32, ffwant,
                        "fence forms", 32);
    }

    // 8-bit arithmetic widening and f16/f16x2 division.
    {
        static const uint32_t n8in[6] = { 0x64c8u, 0, 0, 0, 0, 0 };
        static const uint32_t n8want[6] = {
            0x64c8u, 0, 0x64209c2cu, 0x646420c8u, 0x2ce464c8u, 0x20u
        };
        static const uint32_t f16in[6] = {
            0, 0, 0x40004600u, 0x46004000u, 0x40004400u, 0
        };
        static const uint32_t f16want[6] = {
            0x4200u, 0x42003800u, 0x40004600u, 0x46004000u, 0x40004400u, 0
        };
        run_words(ctx, "narrow8.ptx", n8in, 6, n8want, "narrow8");
        run_words(ctx, "f16div.ptx", f16in, 6, f16want, "f16div");
    }

    // Scalar add/sub forms and packed half-word adds; family-specific
    // saturating forms and packed byte sub (ISA 9.2+, extension file).
    {
        static const uint32_t asin[32] = {
            0x0020fff0u, 0x00000020u, 0xfffffff0u, 0xffffffffu,
            0x00000020u, 0x00000000u, 0x00000020u, 0x00000000u,
            0x00000030u, 0x00000000u, 0x7fffffffu, 0x00000001u,
            0x80000000u, 0x00000001u, 0x0001fff0u, 0x00020020u,
            0x0001ffffu, 0x00010001u, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0
        };
        static const uint32_t aswant[32] = {
            0x0020fff0u, 0x00000020u, 0xfffffff0u, 0xffffffffu,
            0x00000020u, 0x00000000u, 0x00000020u, 0x00000000u,
            0x00000030u, 0x00000000u, 0x7fffffffu, 0x00000001u,
            0x80000000u, 0x00000001u, 0x0001fff0u, 0x00020020u,
            0x0001ffffu, 0x00010001u,
            0x00000010u, 0x0000ffd0u, 0x00000010u, 0x00000000u,
            0xfffffff0u, 0xffffffffu, 0x00000010u, 0x00000000u,
            0xfffffff0u, 0xffffffffu, 0x7fffffffu, 0x80000000u,
            0x00030010u, 0x00020000u
        };
        static const uint32_t afin[17] = {
            0xfffffff0u, 0x00000020u, 0xfff00001u, 0x00200002u,
            0x80fffff0u, 0x01020320u, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        };
        static const uint32_t afwant[17] = {
            0xfffffff0u, 0x00000020u, 0xfff00001u, 0x00200002u,
            0x80fffff0u, 0x01020320u,
            0xffffffffu, 0xffff0003u, 0x00100003u, 0x81ffffffu,
            0x81010210u, 0x7ffdfcd0u, 0x80fdfcd0u, 0x7ffdfcd0u,
            0x7ffdfcd0u, 0xffd0ffffu, 0xffd0ffffu
        };
        run_words(ctx, "add_sub_forms.ptx", asin, 32, aswant,
                  "add_sub_forms");
        run_words(ctx, "add_sub_family.ptx", afin, 17, afwant,
                  "add_sub_family");
    }

    // cnot/lop3/slct, uniform load, prefetch/cache hints, warp
    // barrier, and reduction atomics.
    {
        static const uint32_t miscin[22] = {
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            10, 0
        };
        static const uint32_t miscwant[22] = {
            1, 0, 0xeaeaeaeau, 0x11111111u, 0x22222222u,
            0x11111111u, 0x22222222u, 10, 15, 0x1234u,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0x1234u
        };
        run_words(ctx, "misc.ptx", miscin, 22, miscwant, "misc");
        uint32_t lin[5] = { 0, 0, 0, 0, 0 };
        uint32_t lwant[5] = { 0x7fu, 1, 0, 0, 1 };
        run_words(ctx, "lop3_boolop.ptx", lin, 5, lwant, "lop3 BoolOp");
    }

    // Warp-level WMMA m16n16k16 f16 x f16 -> f32 (tests/ptx/wmma.ptx).
    // A is 16x16 identity, B[k][c] = 16*k+c stored column-major, and
    // C[r][c] = 16*r+c, so D[r][c] = 2*(16*r+c).
    {
        static uint32_t win[768], wwant[768];
        memset(win, 0, sizeof(win));
        for (int r = 0; r < 16; r++)
            for (int c = 0; c < 16; c++) {
                int ai = r * 16 + c;
                uint32_t ah = (r == c) ? f32_to_f16_bits(1.0f) : 0;
                win[ai >> 1] |= ah << ((ai & 1) * 16);
            }
        for (int c = 0; c < 16; c++)
            for (int k = 0; k < 16; k++) {
                int bi = c * 16 + k;
                uint32_t bh = f32_to_f16_bits((float)(k * 16 + c));
                win[128 + (bi >> 1)] |= bh << ((bi & 1) * 16);
            }
        for (int r = 0; r < 16; r++)
            for (int c = 0; c < 16; c++) {
                win[256 + r * 16 + c] = f32_bits((float)(r * 16 + c));
                win[512 + r * 16 + c] =
                    f32_bits((float)(2 * (r * 16 + c)));
            }
        memcpy(wwant, win, 512 * 4);
        memcpy(wwant + 512, win + 512, 256 * 4);
        run_words_block(ctx, "wmma.ptx", win, 768, wwant, "wmma", 32);
    }

    // These kernels all contain a shared WMMA load that would normally
    // enable the typed 16-bit view, plus an access that must disable it.
    run_typed_shared_guard("wmma_shared_c.ptx", "wmma_shared_c");
    run_typed_shared_guard("wmma_guard_ldmatrix.ptx",
                           "wmma_guard_ldmatrix");
    run_typed_shared_guard("wmma_guard_partial_cpasync.ptx",
                           "wmma_guard_partial_cpasync");
    run_typed_shared_guard("wmma_guard_scalar_shared.ptx",
                           "wmma_guard_scalar_shared");
    run_typed_shared_guard("wmma_guard_barred.ptx", "wmma_guard_barred");
    run_typed_shared_guard("wmma_guard_generic.ptx", "wmma_guard_generic");
    run_typed_shared_guard("wmma_guard_store_shared.ptx",
                           "wmma_guard_store_shared");
    run_typed_shared_guard("wmma_guard_shape.ptx", "wmma_guard_shape");

    // Sparse mma m16n8k16 2:4, f16 and bf16 multiplicands.
    {
        static uint32_t spin[416], spwant[416];
        build_sparse_mma(spin, spwant, 0, 0);
        run_words_block(ctx, "mma_sp.ptx", spin, 416, spwant, "mma_sp", 32);
        build_sparse_mma(spin, spwant, 1, 0);
        run_words_block(ctx, "mma_sp_bf16.ptx", spin, 416, spwant,
                        "mma_sp_bf16", 32);
        // Alternate selector patterns {0,3}/{1,2} exercise both packed
        // halves of each dense f16 pair.
        build_sparse_mma(spin, spwant, 0, 1);
        run_words_block(ctx, "mma_sp.ptx", spin, 416, spwant,
                        "mma_sp_alt", 32);
        build_sparse_mma(spin, spwant, 1, 1);
        run_words_block(ctx, "mma_sp_bf16.ptx", spin, 416, spwant,
                        "mma_sp_bf16_alt", 32);
    }

    // Nested structured loop (outer + inner loop, kind 9).
    {
        static uint32_t nin[2], nwant[2];
        nin[0] = 0; nin[1] = 0; nwant[0] = 6; nwant[1] = 2;
        run_words_block(ctx, "nested_loop.ptx", nin, 2, nwant,
                        "nested_loop", 32);
    }

    // ACuda dot2.f32.f16 extension (f16x2 dot with f32 accumulator).
    {
        static uint32_t din[192], dwant[192];
        build_dot2(din, dwant);
        run_words_block(ctx, "dot2.ptx", din, 192, dwant, "dot2", 32);
    }

    // Dense mma.sync m16n8k8 f16/bf16 and m16n8k4/k8 tf32.
    {
        static uint32_t kin[448], kwant[448];
        build_mma_k8(kin, kwant, 0);
        run_words_block(ctx, "mma_k8.ptx", kin, 352, kwant, "mma_k8", 32);
        build_mma_k8(kin, kwant, 1);
        run_words_block(ctx, "mma_k8_bf16.ptx", kin, 352, kwant,
                        "mma_k8_bf16", 32);
        static uint32_t k16bf[448], k16bfw[448];
        build_mma_k16_bf16(k16bf, k16bfw);
        run_words_block(ctx, "mma_k16_bf16.ptx", k16bf, 448, k16bfw,
                        "mma_k16_bf16", 32);
        build_mma_k4(kin, kwant);
        run_words_block(ctx, "mma_k4.ptx", kin, 352, kwant, "mma_k4", 32);
        build_mma_k8_tf32(kin, kwant);
        run_words_block(ctx, "mma_k8_tf32.ptx", kin, 448, kwant,
                        "mma_k8_tf32", 32);
    }

    // Dense integer mma m16n8k32 s32 with s8/u8 multiplicands.
    {
        static uint32_t iin[448], iwant[448];
        build_mma_int8(iin, iwant, 0, 0);
        run_words_block(ctx, "mma_u8.ptx", iin, 448, iwant, "mma_u8", 32);
        build_mma_int8(iin, iwant, 1, 0);
        run_words_block(ctx, "mma_s8u8.ptx", iin, 448, iwant,
                        "mma_s8u8", 32);
        build_mma_int8(iin, iwant, 0, 1);
        run_words_block(ctx, "mma_u8s8.ptx", iin, 448, iwant,
                        "mma_u8s8", 32);
        static uint32_t k16in[128], k16want[128];
        build_mma_k16_s8(k16in, k16want, 1);
        run_words_block(ctx, "mma_k16_s8.ptx", k16in, 128, k16want,
                        "mma_k16_s8", 32);
        build_mma_k16_s8(k16in, k16want, 0);
        run_words_block(ctx, "mma_k16_u8.ptx", k16in, 128, k16want,
                        "mma_k16_u8", 32);
        static uint32_t m8in[64], m8want[64];
        build_mma_m8s8_k16(m8in, m8want, 1);
        run_words_block(ctx, "mma_m8s8_k16.ptx", m8in, 64, m8want,
                        "mma_m8s8_k16", 32);
        build_mma_m8s8_k16(m8in, m8want, 0);
        run_words_block(ctx, "mma_m8s8_k16_u8.ptx", m8in, 64, m8want,
                        "mma_m8s8_k16_u8", 32);
        static uint32_t q4in[448], q4want[448];
        build_mma_int4_m8(q4in, q4want, 0);
        run_words_block(ctx, "mma_int4_m8n8k32.ptx", q4in, 192, q4want,
                        "mma_int4_m8n8k32", 32);
        build_mma_int4_m8(q4in, q4want, 1);
        run_words_block(ctx, "mma_int4_s4_m8n8k32.ptx", q4in, 192, q4want,
                        "mma_int4_s4_m8n8k32", 32);
        build_mma_int4_m16k32(q4in, q4want, 0);
        run_words_block(ctx, "mma_int4_m16n8k32.ptx", q4in, 352, q4want,
                        "mma_int4_m16n8k32", 32);
        build_mma_int4_m16k32(q4in, q4want, 1);
        run_words_block(ctx, "mma_int4_s4_m16n8k32.ptx", q4in, 352, q4want,
                        "mma_int4_s4_m16n8k32", 32);
        build_mma_int4_m16k64(q4in, q4want, 0);
        run_words_block(ctx, "mma_int4_m16n8k64.ptx", q4in, 448, q4want,
                        "mma_int4_m16n8k64", 32);
        build_mma_int4_m16k64(q4in, q4want, 1);
        run_words_block(ctx, "mma_int4_s4_m16n8k64.ptx", q4in, 448, q4want,
                        "mma_int4_s4_m16n8k64", 32);
        static uint32_t b1in[448], b1want[448];
        build_mma_b1_m8(b1in, b1want);
        run_words_block(ctx, "mma_b1_m8n8k128_and.ptx", b1in, 192, b1want,
                        "mma_b1_m8n8k128_and", 32);
        build_mma_b1_m16k128(b1in, b1want, 1);
        run_words_block(ctx, "mma_b1_m16n8k128_xor.ptx", b1in, 352, b1want,
                        "mma_b1_m16n8k128_xor", 32);
        build_mma_b1_m16k256(b1in, b1want);
        run_words_block(ctx, "mma_b1_m16n8k256_and.ptx", b1in, 448, b1want,
                        "mma_b1_m16n8k256_and", 32);
    }

    // Dense FP8 mma m16n8k32 e4m3/e5m2 (plain and kind::f8f6f4).
    {
        static uint32_t pin[448], pwant[448];
        build_mma_fp8(pin, pwant, 0);
        run_words_block(ctx, "mma_fp8_e4m3.ptx", pin, 448, pwant,
                        "mma_fp8_e4m3", 32);
        build_mma_fp8(pin, pwant, 0);
        run_words_block(ctx, "mma_fp8_e4m3_kind.ptx", pin, 448, pwant,
                        "mma_fp8_e4m3_kind", 32);
        build_mma_fp8(pin, pwant, 1);
        run_words_block(ctx, "mma_fp8_e5m2.ptx", pin, 448, pwant,
                        "mma_fp8_e5m2", 32);
    }

    // Dense mma with f16 accumulators (m16n8k16 and m16n8k8).
    {
        static uint32_t fin[320], fwant[320];
        build_mma_f16acc(fin, fwant, 0);
        run_words_block(ctx, "mma_f16acc.ptx", fin, 320, fwant,
                        "mma_f16acc", 32);
        build_mma_f16acc(fin, fwant, 1);
        run_words_block(ctx, "mma_k8_f16acc.ptx", fin, 224, fwant,
                        "mma_k8_f16acc", 32);
    }

    // Dense mma m8n8k4 f64.
    {
        static uint32_t din[384], dwant[384];
        build_mma_f64(din, dwant);
        run_words_block(ctx, "mma_f64.ptx", din, 384, dwant, "mma_f64", 32);
    }

    // Warp-level WMMA m16n16k16 bf16 x bf16 -> f32.
    {
        static uint32_t bin[768], bwant[768];
        build_wmma_bf16(bin, bwant);
        run_words_block(ctx, "wmma_bf16.ptx", bin, 768, bwant,
                        "wmma_bf16", 32);
    }

    // Warp-level WMMA m16n16k16 f16 x f16 -> f16 (f16 accumulate).
    {
        static uint32_t fin2[512], fwant2[512];
        build_wmma_f16acc(fin2, fwant2);
        run_words_block(ctx, "wmma_f16acc.ptx", fin2, 512, fwant2,
                        "wmma_f16acc", 32);
    }

    // Warp-level WMMA m8n32k16 and m32n8k16 (f16 -> f32).
    {
        static uint32_t sin[832], swant[832];
        build_wmma_shape(sin, swant, 8, 32, 16, 0);
        run_words_block(ctx, "wmma_m8n32k16.ptx", sin, 832, swant,
                        "wmma_m8n32k16", 32);
        build_wmma_shape(sin, swant, 32, 8, 16, 0);
        run_words_block(ctx, "wmma_m32n8k16.ptx", sin, 832, swant,
                        "wmma_m32n8k16", 32);
        build_wmma_shape(sin, swant, 32, 8, 16, 1);
        run_words_block(ctx, "wmma_m32n8k16_bf16.ptx", sin, 832, swant,
                        "wmma_m32n8k16_bf16", 32);
        build_wmma_shape(sin, swant, 8, 32, 16, 1);
        run_words_block(ctx, "wmma_m8n32k16_bf16.ptx", sin, 832, swant,
                        "wmma_m8n32k16_bf16", 32);
    }

    // Warp-level integer WMMA m16n16k16, m8n32k16 and m32n8k16.
    {
        static uint32_t iin[672], iwant[672];
        build_wmma_int8_shape(iin, iwant, 16, 16, 16, 0);
        run_words_block(ctx, "wmma_int8_m16n16k16.ptx", iin, 640, iwant,
                        "wmma_int8_m16n16k16", 32);
        build_wmma_int8_shape(iin, iwant, 16, 16, 16, 1);
        run_words_block(ctx, "wmma_int8_s8_m16n16k16.ptx", iin, 640, iwant,
                        "wmma_int8_s8_m16n16k16", 32);
        build_wmma_int8_shape(iin, iwant, 8, 32, 16, 0);
        run_words_block(ctx, "wmma_int8_m8n32k16.ptx", iin, 672, iwant,
                        "wmma_int8_m8n32k16", 32);
        build_wmma_int8_shape(iin, iwant, 32, 8, 16, 0);
        run_words_block(ctx, "wmma_int8_m32n8k16.ptx", iin, 672, iwant,
                        "wmma_int8_m32n8k16", 32);
    }

    // Warp-level 4-bit WMMA m8n8k32 u4/s4 x u4/s4 -> s32.
    {
        static uint32_t qin[192], qwant[192];
        build_wmma_int4_m8(qin, qwant, 0);
        run_words_block(ctx, "wmma_int4_m8n8k32.ptx", qin, 192, qwant,
                        "wmma_int4_m8n8k32", 32);
        build_wmma_int4_m8(qin, qwant, 1);
        run_words_block(ctx, "wmma_int4_s4_m8n8k32.ptx", qin, 192, qwant,
                        "wmma_int4_s4_m8n8k32", 32);
    }

    // Warp-level single-bit WMMA m8n8k128 AND/XOR popcount.
    {
        static uint32_t bin[192], bwant[192];
        build_wmma_b1_m8(bin, bwant, 0);
        run_words_block(ctx, "wmma_b1_m8n8k128_and.ptx", bin, 192, bwant,
                        "wmma_b1_m8n8k128_and", 32);
        build_wmma_b1_m8(bin, bwant, 1);
        run_words_block(ctx, "wmma_b1_m8n8k128_xor.ptx", bin, 192, bwant,
                        "wmma_b1_m8n8k128_xor", 32);
    }

    // Dense mma m8n8k4 f16 (four lane-group MMAs).
    {
        static uint32_t min[640], mwant[640];
        build_mma_m8f16(min, mwant);
        run_words_block(ctx, "mma_m8n8k4_f16.ptx", min, 640, mwant,
                        "mma_m8n8k4_f16", 32);
    }

    // Sparse mma.sp m16n8k32 f16.
    {
        static uint32_t skin[544], swant[544];
        build_mma_sp_k32(skin, swant, 0, 0, 0);
        run_words_block(ctx, "mma_sp_k32.ptx", skin, 544, swant,
                        "mma_sp_k32", 32);
        build_mma_sp_k32(skin, swant, 0, 0, 1);
        run_words_block(ctx, "mma_sp_k32_bf16.ptx", skin, 544, swant,
                        "mma_sp_k32_bf16", 32);
        build_mma_sp_k32(skin, swant, 1, 0, 0);
        run_words_block(ctx, "mma_sp_k32.ptx", skin, 544, swant,
                        "mma_sp_k32_alt", 32);
        // Two MMAs on identical fragments exercise the packing-cache hit.
        build_mma_sp_k32(skin, swant, 0, 1, 0);
        run_words_block(ctx, "mma_sp_reuse.ptx", skin, 544, swant,
                        "mma_sp_reuse", 32);
        build_mma_sp_k32(skin, swant, 1, 1, 0);
        run_words_block(ctx, "mma_sp_reuse.ptx", skin, 544, swant,
                        "mma_sp_reuse_alt", 32);
        build_mma_sp_tf32(skin, swant, 8);
        run_words_block(ctx, "mma_sp_tf32_k8.ptx", skin, 416, swant,
                        "mma_sp_tf32_k8", 32);
        run_words_block(ctx, "mma_sp_om_tf32_k8.ptx", skin, 416, swant,
                        "mma_sp_om_tf32_k8", 32);
        build_mma_sp_tf32(skin, swant, 16);
        run_words_block(ctx, "mma_sp_tf32_k16.ptx", skin, 544, swant,
                        "mma_sp_tf32_k16", 32);
        run_words_block(ctx, "mma_sp_om_tf32_k16.ptx", skin, 544, swant,
                        "mma_sp_om_tf32_k16", 32);
        build_mma_sp_int4_k64(skin, swant, 0);
        run_words_block(ctx, "mma_sp_int4_m16n8k64.ptx", skin, 416, swant,
                        "mma_sp_int4_m16n8k64", 32);
        build_mma_sp_int4_k64(skin, swant, 1);
        run_words_block(ctx, "mma_sp_s4_m16n8k64.ptx", skin, 416, swant,
                        "mma_sp_s4_m16n8k64", 32);
        build_mma_sp_int4_k128(skin, swant, 0);
        run_words_block(ctx, "mma_sp_int4_m16n8k128.ptx", skin, 544, swant,
                        "mma_sp_int4_m16n8k128", 32);
        build_mma_sp_int4_k128(skin, swant, 1);
        run_words_block(ctx, "mma_sp_s4_m16n8k128.ptx", skin, 544, swant,
                        "mma_sp_s4_m16n8k128", 32);
    }

    // Warp-level WMMA m16n16k8 tf32 -> f32.
    {
        static uint32_t tin[768], twant[768];
        build_wmma_tf32(tin, twant);
        run_words_block(ctx, "wmma_tf32.ptx", tin, 768, twant,
                        "wmma_tf32", 32);
    }

    // Dense mma m16n8k4/k8/k16 f64 (target-gated to sm_90).
    {
        static uint32_t fin3[1280], fwant3[1280];
        build_mma_f64_k(fin3, fwant3, 4);
        run_words_block(ctx, "mma_f64_k4.ptx", fin3, 704, fwant3,
                        "mma_f64_k4", 32);
        run_words_block(ctx, "mma_f64_k4_rz.ptx", fin3, 704, fwant3,
                        "mma_f64_k4_rz", 32);
        run_words_block(ctx, "mma_f64_k4_rm.ptx", fin3, 704, fwant3,
                        "mma_f64_k4_rm", 32);
        run_words_block(ctx, "mma_f64_k4_rp.ptx", fin3, 704, fwant3,
                        "mma_f64_k4_rp", 32);
        build_mma_f64_k(fin3, fwant3, 8);
        run_words_block(ctx, "mma_f64_k8.ptx", fin3, 896, fwant3,
                        "mma_f64_k8", 32);
        build_mma_f64_k(fin3, fwant3, 16);
        run_words_block(ctx, "mma_f64_k16.ptx", fin3, 1280, fwant3,
                        "mma_f64_k16", 32);
    }

    // isspacep queries for shared/global/local/param addresses.
    {
        static const uint32_t isin[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
        static const uint32_t iswant[8] = { 1, 1, 1, 0, 0, 1, 0, 0 };
        run_words(ctx, "isspacep.ptx", isin, 8, iswant, "isspacep");
        static const uint32_t iscin1[2] = { 1, 0 };
        static const uint32_t iscwant1[2] = { 5, 0 };
        run_words(ctx, "isspacep_cfg.ptx", iscin1, 2, iscwant1,
                  "isspacep.const in state machine (taken)");
        static const uint32_t iscin0[2] = { 0, 0 };
        static const uint32_t iscwant0[2] = { 11, 0 };
        run_words(ctx, "isspacep_cfg.ptx", iscin0, 2, iscwant0,
                  "isspacep.const in state machine (fallthrough)");
    }

    // Half-precision set/setp matrix (f16/f16x2 comparisons).
    {
        static const uint32_t hcin[11] = { 0 };
        static const uint32_t hcwant[11] = {
            0xffffffffu, 0, 0xffffffffu, 0x0000ffffu, 0x3c003c00u,
            1, 1, 0, 0, 0, 1
        };
        run_words(ctx, "half_cmp.ptx", hcin, 11, hcwant, "half_cmp");
    }

    // cvt int<->float corners (signed/unsigned 32/64, f16 dest).
    {
        static const uint32_t cfin[74] = { 0 };
        static const uint32_t cfwant[74] = {
            0xc0400000u, 0, 0, 0xc01c0000u, 0x0000c200u,
            0xfffffffdu, 4, 0, 4, 0, 5, 0,
            4, 0, 0xfffffffbu, 0xffffffffu, 5, 0, 0, 0x40000000u
        };
        run_words(ctx, "cvt_int_float.ptx", cfin, 74, cfwant,
                  "cvt_int_float");

        static const uint32_t u8in[12] = {
            0x406ccccdu, 0xc06ccccdu, 0x43960000u, 0xc3480000u,
            0, 0, 0, 0, 0, 0, 0, 0
        };
        static const uint32_t u8want[12] = {
            0x406ccccdu, 0xc06ccccdu, 0x43960000u, 0xc3480000u,
            3, 0, 0xffu, 0, 4, 0xfffffffcU, 0x7fu, 0xffffff80U
        };
        run_words(ctx, "cvt_u8.ptx", u8in, 12, u8want,
                  "cvt float->u8/s8");

        static const uint32_t satin[20] = {
            0xfffffffbu, 0x12cu, 0x11170u, 0xfffeee90u,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        };
        static const uint32_t satwant[20] = {
            0xfffffffbu, 0x12cu, 0x11170u, 0xfffeee90u,
            0, 0xfffffffbu, 0, 0xfffffffbu,
            0xffu, 0x7fu, 0x12cu, 0x12cu,
            0xffu, 0x7fu, 0xffffu, 0x7fffu,
            0, 0xffffff80u, 0, 0xffff8000u
        };
        run_words(ctx, "cvt_sat.ptx", satin, 20, satwant, "cvt.sat narrow");

        static const uint32_t smallin[8] = {
            0x0080ffffu, 0x00008000u, 0, 0, 0, 0, 0, 0
        };
        static const uint32_t smallwant[8] = {
            0x0080ffffu, 0x00008000u, 0x477fff00u, 0xc3000000u,
            0x0000f800u, 0, 0, 0
        };
        run_words(ctx, "cvt_small.ptx", smallin, 8, smallwant,
                  "cvt small int->float");

        static const uint32_t dirin[18] = { 0 };
        static const uint32_t dirwant[18] = {
            0x4b800001u, 0x4b800000u, 0x4b800000u, 0x4b800000u,
            0xcb800000u, 0xcb800001u, 0xcb800000u, 0xcb800000u,
            2, 0x43400000u, 1, 0x43400000u, 1, 0x43400000u,
            2, 0x43400000u, 0x68006801u, 0x68006800u
        };
        run_words(ctx, "cvt_int_dir.ptx", dirin, 18, dirwant,
                  "cvt directed int->float");

        static const uint32_t tf32in[14] = { 0 };
        static const uint32_t tf32want[14] = {
            0x3f800000u, 0x3f802000u, 0x3f804000u, 0x3f804000u,
            0x3f802000u, 0x3f802000u, 0x3f800000u, 0x3f800000u,
            0xbf800000u, 0xbf802000u, 0x7fc00000u, 0xffc00000u,
            0x7f800000u, 0x80000000u
        };
        run_words(ctx, "cvt_tf32.ptx", tf32in, 14, tf32want,
                  "cvt tf32 rounding");

        static const uint32_t satfin[36] = { 0 };
        static const uint32_t satfwant[36] = {
            0x7bff7bffu, 0xfbff7bffu, 0xfe007e00u, 0xc0003e00u,
            0x00007bffu, 0x7bfffbffu, 0x7e003e00u, 0x00007bffu,
            0x3c004000u,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0
        };
        run_words(ctx, "cvt_satfinite.ptx", satfin, 36, satfwant,
                  "cvt satfinite f16");

        static const uint32_t packin[8] = { 0 };
        static const uint32_t packwant[8] = {
            0xffff0000u, 0x7fff8000u, 0x5678ff00u, 0x56787fffu,
            0x345678f0u, 0x34567878u, 0x2345678cu, 0x23456786u
        };
        run_words(ctx, "cvt_pack_forms.ptx", packin, 8, packwant,
                  "cvt.pack forms");
    }

    // Half-precision f16/f16x2 add/sub/mul/fma/neg/abs/min/max,
    // plus tanh and ex2, including ftz/sat/NaN forms.
    {
        static const uint32_t halfin[40] = {
            0x42004000u, 0xbe004400u, 0x7e000001u, 0x42004000u, 0x44004000u, 0x3c003c00u, 0xc200be00u, 0x00010001u,
            0x40004000u, 0x00000000u, 0x7e004000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
            0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
            0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
            0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
        };
        static const uint32_t halfwant[40] = {
            0x42004000u, 0xbe004400u, 0x7e000001u, 0x42004000u, 0x44004000u, 0x3c003c00u, 0xc200be00u, 0x00010001u,
            0x40004000u, 0x00000000u, 0x7e004000u, 0x00000000u, 0x00004500u, 0x00003c00u, 0x00004600u, 0x00004900u,
            0x00003c00u, 0x00004000u, 0x00003e00u, 0x00003e00u, 0x00000000u, 0x00004000u, 0x00004200u, 0x00007e00u,
            0x00000000u, 0x00000000u, 0x00004800u, 0x47004400u, 0xbc000000u, 0x4a004400u, 0x4a804500u, 0x3c003c00u,
            0x40004000u, 0x42003e00u, 0x42003e00u, 0x42004000u, 0x44004000u, 0x7e004000u, 0x00000000u, 0x48004400u,
        };
        run_words(ctx, "half_fp.ptx", halfin, 40, halfwant, "half_fp");
    }

    // Floating-point properties, signed copies, mad/fma, sat/ftz
    // arithmetic, abs/neg, min/max NaN/ftz, rcp/sqrt/rsqrt, the
    // transcendentals, and div.full/approx/ftz.
    {
        static const uint32_t fpin[88] = {
            0x3f800000u, 0x00000000u, 0x7f800000u, 0x7fc00000u, 0x00000001u, 0xbfc00000u, 0x40000000u, 0xc0000000u,
            0x40400000u, 0x40800000u, 0x41000000u, 0x00000000u, 0x00000000u, 0x3ff00000u, 0x00000000u, 0x7ff00000u,
            0x00000000u, 0x7ff80000u, 0x00000000u, 0x00000000u, 0x00000001u, 0x00000000u, 0x00000000u, 0xbff80000u,
            0x00000000u, 0x40100000u, 0x00000000u, 0x40200000u, 0x00000000u, 0x40000000u, 0x00000000u, 0x00000000u,
            0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
            0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
            0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
            0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
            0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
            0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
            0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
        };
        static const uint32_t fpwant[88] = {
            0x3f800000u, 0x00000000u, 0x7f800000u, 0x7fc00000u, 0x00000001u, 0xbfc00000u, 0x40000000u, 0xc0000000u,
            0x40400000u, 0x40800000u, 0x41000000u, 0x00000000u, 0x00000000u, 0x3ff00000u, 0x00000000u, 0x7ff00000u,
            0x00000000u, 0x7ff80000u, 0x00000000u, 0x00000000u, 0x00000001u, 0x00000000u, 0x00000000u, 0xbff80000u,
            0x00000000u, 0x40100000u, 0x00000000u, 0x40200000u, 0x00000000u, 0x40000000u, 0x00000000u, 0x00000000u,
            0x00000001u, 0x00000000u, 0x00000000u, 0x00000001u, 0x00000000u, 0x00000001u, 0x00000001u, 0x00000000u,
            0x00000001u, 0x00000001u, 0x00000001u, 0x00000001u, 0x00000001u, 0x00000001u, 0xc0000000u, 0x00000000u,
            0xc0000000u, 0x40800000u, 0x00000000u, 0x40100000u, 0x40a00000u, 0x3f800000u, 0x3f800000u, 0x00000000u,
            0x3f800000u, 0x3fc00000u, 0x00000000u, 0x3ff80000u, 0x00000000u, 0x80000000u, 0x7fc00000u, 0x7fc00000u,
            0x00000000u, 0x3e800000u, 0x00000000u, 0x3fd00000u, 0x00000000u, 0x3fd00000u, 0x40000000u, 0x40000000u,
            0x00000000u, 0x40000000u, 0x3f000000u, 0x00000000u, 0x3fe00000u, 0x3f000000u, 0x00000000u, 0x3f800000u,
            0x3f800000u, 0x40800000u, 0x00000000u, 0x40800000u, 0x40800000u, 0x40800000u, 0x00000000u, 0x40100000u,
        };
        run_words(ctx, "fp_ext.ptx", fpin, 88, fpwant, "fp_ext");
    }

    // Extended-precision add.cc/addc/sub.cc/subc/mad.cc/madc.
    {
        static const uint32_t carryin[43] = {
            0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
            0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
            0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
            0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
            0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
            0x00000000u, 0x00000000u, 0x00000000u,
        };
        static const uint32_t carrywant[43] = {
            0x00000000u, 0x00000001u, 0x00000003u, 0x00000000u, 0x0000000du, 0x00000000u, 0x0000000du, 0x00000001u,
            0x00000000u, 0x00000000u, 0x00000001u, 0x0000000du, 0x00000000u, 0x00000000u, 0xffffffffu, 0x00000001u,
            0x00000001u, 0x00000000u, 0xfffffffdu, 0x00000001u, 0xffffffffu, 0xffffffffu, 0x00000001u, 0xfffffffdu,
            0xffffffffu, 0x00000001u, 0x00000000u, 0x00000001u, 0x00000000u, 0x00000001u, 0x0000000bu, 0x00000000u,
            0xfffffffeu, 0x00000001u, 0x00000000u, 0x00000000u, 0x00000001u, 0x00000000u, 0x00000000u, 0x00000001u,
            0x0000000bu, 0x00000000u, 0x00000000u,
        };
        run_words(ctx, "ext_arith.ptx", carryin, 43, carrywant,
                  "ext_arith");
    }

    // Extended integer arithmetic: mul/mad modes, sad/div/rem 16-bit,
    // abs/neg, 64-bit clz/brev/bfind, fns, szext, packed min/max/relu,
    // dp4a/dp2a, clmad, mad24, bfe/bfi/bmsk.
    {
        static const uint32_t extin[90] = {
            0x0020fff0u, 0x00000005u, 0x007fffffu, 0x007fffffu, 0x7fffffffu, 0x00000000u, 0x00000000u, 0x80000000u,
            0x00000003u, 0x00000000u, 0x00000000u, 0x00000000u, 0xffffffffu, 0xffffffffu, 0xfff00001u, 0x00200002u,
            0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
            0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
            0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
            0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
            0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
            0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
            0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
            0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
            0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
            0x00000000u, 0x00000000u,
        };
        static const uint32_t extwant[90] = {
            0x0020fff0u, 0x00000005u, 0x007fffffu, 0x007fffffu, 0x7fffffffu, 0x00000000u, 0x00000000u, 0x80000000u,
            0x00000003u, 0x00000000u, 0x00000000u, 0x00000000u, 0xffffffffu, 0xffffffffu, 0xfff00001u, 0x00200002u,
            0x001ffe00u, 0xfffffe00u, 0x0000001fu, 0x0000ffffu, 0xffffffffu, 0xffffffffu, 0x0000fe05u, 0x00000024u,
            0x80003ffeu, 0x7fffffffu, 0x009ffdffu, 0xff000001u, 0x00003fffu, 0x0000ffd5u, 0x00000035u, 0x7fffffffu,
            0x7fffffffu, 0x000007ffu, 0x00000000u, 0x00000010u, 0x0000fff0u, 0x00000010u, 0x00000001u, 0x00000000u,
            0x00000010u, 0x011000ffu, 0x00000000u, 0x00000001u, 0x00000000u, 0x0000003fu, 0xffffffffu, 0xffffffffu,
            0x00000003u, 0x00000001u, 0xffffffffu, 0x000000ffu, 0x00200001u, 0x00200002u, 0x00000001u, 0xfff00002u,
            0x00000001u, 0x007fffffu, 0x80023b02u, 0x80003d02u, 0x80003d02u, 0x80ff7d81u, 0x7fffff80u, 0x00000000u,
            0x80000000u, 0x00000001u, 0x00000000u, 0x7f000000u, 0xbffffeffu, 0x7fffffffu, 0x00000008u, 0x00000000u,
            0x00000030u, 0x80000000u, 0xffffffffu, 0xfffffff8u, 0xffffffffu, 0x007fffffu, 0xfffffffeu, 0x00000002u,
            0x00000001u, 0xffffffffu, 0x007fffffu, 0x00000000u, 0x00000001u, 0x00000000u, 0x00000001u, 0x00000000u,
            0xffffffffu, 0xffffffffu,
        };
        run_words(ctx, "int_arith_ext.ptx", extin, 90, extwant,
                  "int_arith_ext");
    }

    // PTX ISA 9.3 has no sub.u16x2/.s16x2 and no sub.sat.u32.
    {
        char err[1024];
        acuda_kernel_t *k = NULL;
        const char *badsubsat16 =
            ".version 9.3\n.target sm_120f\n.address_size 64\n"
            ".visible .entry badsubsat16(.param .u64 a)\n"
            "{\n"
            "    .reg .b32 %r;\n"
            "    sub.sat.u16x2 %r, %r, %r;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badsubsat16, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative sub.sat.u16x2: %s\n", err);
        const char *badsubsatu32 =
            ".version 9.3\n.target sm_120f\n.address_size 64\n"
            ".visible .entry badsubsatu32(.param .u64 a)\n"
            "{\n"
            "    .reg .b32 %r;\n"
            "    sub.sat.u32 %r, %r, %r;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(badsubsatu32, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative sub.sat.u32: %s\n", err);
    }

    // Multi-block callee and loop break/continue regressions.
    run_single_buf(ctx, "loop_break.ptx", 0, 0, 10, "loop_break");
    run_single_buf(ctx, "loop_continue.ptx", 0, 0, 42, "loop_continue");
    run_single_buf(ctx, "call_multi.ptx", 41, 7, 41, "call_multi");

    // Indirect dispatch through a function pointer table (dynamic
    // index), a selp between two function addresses (with only a
    // prototype), and a .const table at a constant index.
    {
        uint32_t in0[4] = {0, 0, 0, 0}, want0[4] = {0, 0, 0x10, 0};
        uint32_t in1[4] = {1, 0, 0, 0}, want1[4] = {1, 0, 0x21, 0};
        run_words(ctx, "indirect_table.ptx", in0, 4, want0,
                  "indirect_table[0]");
        run_words(ctx, "indirect_table.ptx", in1, 4, want1,
                  "indirect_table[1]");
        run_words(ctx, "indirect_selp.ptx", in0, 4, want0,
                  "indirect_selp[0]");
        run_words(ctx, "indirect_selp.ptx", in1, 4, want1,
                  "indirect_selp[1]");
        uint32_t in5[4] = {5, 0, 0, 0}, want5[4] = {5, 0, 0x25, 0};
        run_words(ctx, "indirect_const.ptx", in5, 4, want5,
                  "indirect_const[1]");
    }

    // Packed f32x2 add/mul/fma (PTX 8.6, sm_100 shape; ACuda emulates
    // the two lanes with scalar f32 ops).
    {
        uint32_t in[12] = {
            0x3f800000u, 0x40000000u, // a = 1.0, 2.0
            0x40400000u, 0x40800000u, // b = 3.0, 4.0
            0x40a00000u, 0x40c00000u, // c = 5.0, 6.0
            0, 0, 0, 0, 0, 0
        };
        uint32_t want[12] = {
            0x3f800000u, 0x40000000u, 0x40400000u, 0x40800000u,
            0x40a00000u, 0x40c00000u,
            0x40800000u, 0x40c00000u, // add: 4, 6
            0x40400000u, 0x41000000u, // mul: 3, 8
            0x41000000u, 0x41600000u  // fma: 8, 14
        };
        run_words(ctx, "f32x2.ptx", in, 12, want, "f32x2");
    }

    // Half fma .relu (f16 scalar, f16x2, bf16x2): negative results
    // clamp to +0, positive lanes stay, NaNs canonicalize.
    {
        uint32_t in[12] = {
            0x3c00bc00u, 0x00003800u, 0, 0x4000bc00u, 0x42003c00u,
            0x3c003800u, 0, 0x4000bf80u, 0x40403f80u, 0x3f803f00u,
            0, 0
        };
        uint32_t want[12] = {
            0x3c00bc00u, 0x00003800u, 0, 0x4000bc00u, 0x42003c00u,
            0x3c003800u, 0x47000000u, 0x4000bf80u, 0x40403f80u,
            0x3f803f00u, 0x40e00000u, 0
        };
        run_words(ctx, "fma_relu.ptx", in, 12, want, "fma_relu");
    }

    // Directed-rounding f32 fma/mad (software integer path on devices
    // whose fused op fails the startup probe) and exact f32 division.
    {
        uint32_t in[9] = { 0x3f800000u, 0x3f800000u, 0x33c00000u, 0,
                           0, 0, 0, 0, 0 };
        uint32_t want[9] = { 0x3f800000u, 0x3f800000u, 0x33c00000u, 0,
                             0x3f800001u, 0x3f800000u, 0x3f800000u,
                             0x3f800001u, 0x3f800001u };
        run_words(ctx, "fma_dir.ptx", in, 9, want, "fma_dir rp/rm/rz/rn");

        uint32_t in2[9] = { 0xbf800000u, 0x3f800000u, 0xb3c00000u, 0,
                            0, 0, 0, 0, 0 };
        uint32_t want2[9] = { 0xbf800000u, 0x3f800000u, 0xb3c00000u, 0,
                              0xbf800000u, 0xbf800001u, 0xbf800000u,
                              0xbf800001u, 0xbf800001u };
        run_words(ctx, "fma_dir.ptx", in2, 9, want2, "fma_dir negative");

        uint32_t din[6] = { 0x3f800000u, 0x40400000u, 0, 0, 0, 0 };
        uint32_t dwant[6] = { 0x3f800000u, 0x40400000u,
                              0x3eaaaaabu, 0x3eaaaaaau, 0x3eaaaaaau,
                              0x3eaaaaabu };
        run_words(ctx, "div_dir.ptx", din, 6, dwant, "div_dir rp/rm/rz/rn");

        uint32_t rin[5] = { 0x40400000u, 0, 0, 0, 0 };
        uint32_t rwant[5] = { 0x40400000u, 0x3eaaaaabu, 0x3eaaaaaau,
                              0x3eaaaaaau, 0x3eaaaaabu };
        run_words(ctx, "rcp_dir.ptx", rin, 5, rwant,
                  "rcp_dir rp/rm/rz/rn");

        uint32_t min[6] = { 0x3f800001u, 0x3f800001u, 0, 0, 0, 0 };
        uint32_t mwant[6] = { 0x3f800001u, 0x3f800001u,
                              0x3f800003u, 0x3f800002u, 0x3f800002u,
                              0x3f800002u };
        run_words(ctx, "mul_dir.ptx", min, 6, mwant,
                  "mul_dir rp/rm/rz/rn");

        uint32_t ain[5] = { 0x80762fd8u, 0x8060f390u, 0, 0, 0 };
        uint32_t awant[5] = { 0x80762fd8u, 0x8060f390u, 0x80d72368u,
                              0x80d72368u, 0x80d72368u };
        run_words(ctx, "add_dir.ptx", ain, 5, awant,
                  "add_dir rp/rm/rz subnormal");
    }

    // Same for f64 (128-bit significand path).
    {
        uint32_t in[16] = {
            0x00000000u, 0x3ff00000u, // a = 1.0
            0x00000000u, 0x3ff00000u, // b = 1.0
            0x00000000u, 0x3ca80000u, // c = 0.75 ulp
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        };
        uint32_t want[16] = {
            0x00000000u, 0x3ff00000u,
            0x00000000u, 0x3ff00000u,
            0x00000000u, 0x3ca80000u,
            0x00000001u, 0x3ff00000u, // rp
            0x00000000u, 0x3ff00000u, // rm
            0x00000000u, 0x3ff00000u, // rz
            0x00000001u, 0x3ff00000u, // rn
            0x00000001u, 0x3ff00000u  // mad
        };
        run_words(ctx, "fma64_dir.ptx", in, 16, want,
                  "fma64_dir rp/rm/rz/rn");

        uint32_t din[12] = {
            0x00000000u, 0x3ff00000u, // 1.0
            0x00000000u, 0x40080000u, // 3.0
            0, 0, 0, 0, 0, 0, 0, 0
        };
        uint32_t dwant[12] = {
            0x00000000u, 0x3ff00000u,
            0x00000000u, 0x40080000u,
            0x55555556u, 0x3fd55555u, // rp
            0x55555555u, 0x3fd55555u, // rm
            0x55555555u, 0x3fd55555u, // rz
            0x55555555u, 0x3fd55555u  // rn
        };
        run_words(ctx, "div64_dir.ptx", din, 12, dwant,
                  "div64_dir rp/rm/rz/rn");

        uint32_t rin[10] = {
            0x00000000u, 0x40080000u, // 3.0
            0, 0, 0, 0, 0, 0, 0, 0
        };
        uint32_t rwant[10] = {
            0x00000000u, 0x40080000u,
            0x55555556u, 0x3fd55555u, // rp
            0x55555555u, 0x3fd55555u, // rm
            0x55555555u, 0x3fd55555u, // rz
            0x55555555u, 0x3fd55555u  // rn
        };
        run_words(ctx, "rcp64_dir.ptx", rin, 10, rwant,
                  "rcp64_dir rp/rm/rz/rn");

        uint32_t min[12] = {
            0x00000001u, 0x3ff00000u,
            0x00000001u, 0x3ff00000u,
            0, 0, 0, 0, 0, 0, 0, 0
        };
        uint32_t mwant[12] = {
            0x00000001u, 0x3ff00000u,
            0x00000001u, 0x3ff00000u,
            0x00000003u, 0x3ff00000u, // rp
            0x00000002u, 0x3ff00000u, // rm
            0x00000002u, 0x3ff00000u, // rz
            0x00000002u, 0x3ff00000u  // rn
        };
        run_words(ctx, "mul64_dir.ptx", min, 12, mwant,
                  "mul64_dir rp/rm/rz/rn");

        uint32_t ain[12] = {
            0x00000000u, 0x3ff00000u, // 1.0
            0x00000000u, 0x3ca00000u, // 2^-53
            0, 0, 0, 0, 0, 0, 0, 0
        };
        uint32_t awant[12] = {
            0x00000000u, 0x3ff00000u,
            0x00000000u, 0x3ca00000u,
            0x00000001u, 0x3ff00000u, // rp
            0x00000000u, 0x3ff00000u, // rm
            0x00000000u, 0x3ff00000u, // rz
            0x00000000u, 0x3ff00000u  // rn
        };
        run_words(ctx, "add64_dir.ptx", ain, 12, awant,
                  "add64_dir rp/rm/rz/rn");
    }

    // brx.idx over a .branchtargets list (four entries), plus a
    // predicated brx whose false path continues after the instruction.
    {
        uint32_t in0[4] = {0, 0, 0, 0}, want0[4] = {0, 0, 0x10, 0};
        uint32_t in1[4] = {1, 0, 0, 0}, want1[4] = {1, 0, 0x20, 0};
        uint32_t in2[4] = {2, 0, 0, 0}, want2[4] = {2, 0, 0x30, 0};
        uint32_t in3[4] = {3, 0, 0, 0}, want3[4] = {3, 0, 0x40, 0};
        run_words(ctx, "brx.ptx", in0, 4, want0, "brx[0]");
        run_words(ctx, "brx.ptx", in1, 4, want1, "brx[1]");
        run_words(ctx, "brx.ptx", in2, 4, want2, "brx[2]");
        run_words(ctx, "brx.ptx", in3, 4, want3, "brx[3]");
        run_words(ctx, "brx_pred.ptx", in0, 4, want0, "brx_pred[0]");
        run_words(ctx, "brx_pred.ptx", in1, 4, want1, "brx_pred[1]");
        uint32_t in5[4] = {5, 0, 0, 0}, want5[4] = {5, 0, 0x99, 0};
        run_words(ctx, "brx_pred.ptx", in5, 4, want5, "brx_pred[5]");
        uint32_t ppin[4] = {0, 0, 0, 0};
        uint32_t ppwant[4] = {0, 0x3fd00000u, 0, 0};
        run_words(ctx, "pred_pair_mov.ptx", ppin, 4, ppwant,
                  "predicated pair move");
        uint32_t pvin[5] = {0, 0, 0, 0, 0};
        uint32_t pvwant[5] = {
            0xaaaaaaaau, 0xbbbbbbbbu, 0x11111111u, 0x22222222u,
            0x12345678u
        };
        run_words(ctx, "pred_vec_mem.ptx", pvin, 5, pvwant,
                  "predicated vector ld/st and RZ destination");
    }

    // A .u64 parameter that reaches an address only through a value
    // operation (lineage broken by xor) is classified as a 64-bit
    // scalar.  Physical-address mode must still be kept so the derived
    // device address is usable; otherwise the load fails with
    // "dynamic address has no pointer homes".
    {
        char err[1024];
        acuda_kernel_t *k = NULL;
        const char *scalar_bda =
            ".version 7.1\n.target sm_86\n.address_size 64\n"
            ".shared .align 8 .b8 sb[8];\n"
            ".visible .entry bdak(.param .u64 p0)\n"
            "{\n"
            "    .reg .u64 %rd, %rd2;\n"
            "    .reg .b32 %r;\n"
            "    ld.param.u64 %rd, [p0];\n"
            "    st.shared.u64 [sb], %rd;\n"
            "    ld.shared.u64 %rd2, [sb];\n"
            "    xor.b64 %rd2, %rd2, 0;\n"
            "    ld.global.u32 %r, [%rd2];\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(scalar_bda, NULL, &k, err, sizeof(err), &g_test_caps) ==
              ACUDA_SUCCESS);
        if (k) {
            acuda_kernel_info_t info;
            memset(&info, 0, sizeof(info));
            CHECK_ST(acuda_kernel_info(k, &info));
            CHECK(info.is_ptr[0] == 2); // scalar, not a binding
            acuda_kernel_free(k);
        }
    }

    // A copied pointer word is a u32 value, not a 32-bit pointer copy:
    // the value must remain usable in scalar comparisons.
    {
        char err[1024];
        acuda_kernel_t *k = NULL;
        const char *ptr_word =
            ".version 7.1\n.target sm_86\n.address_size 64\n"
            ".visible .entry ptrword(.param .u64 p0, .param .u64 out)\n"
            "{\n"
            "    .reg .b32 %a, %b, %r;\n"
            "    .reg .pred %p;\n"
            "    .reg .b64 %rd;\n"
            "    ld.param.u32 %a, [p0];\n"
            "    mov.b32 %b, %a;\n"
            "    setp.ne.u32 %p, %b, 0;\n"
            "    selp.u32 %r, 1, 0, %p;\n"
            "    ld.param.u64 %rd, [out];\n"
            "    st.global.u32 [%rd], %r;\n"
            "    ret;\n"
            "}\n";
        CHECK(acuda_compile_ptx(ptr_word, NULL, &k, err, sizeof(err), &g_test_caps) ==
              ACUDA_SUCCESS);
        if (k) acuda_kernel_free(k);
    }

    // A .u64 parameter that is read both full-width and as a 32-bit
    // word is used, even though the full-width value only ever flows
    // through register halves (word extraction + repack) and never
    // becomes an address itself.  If it is classified as unused the
    // parameter load is lowered to an undefined value, the repacked
    // address is garbage, and the result store never happens.
    {
        static const char *ptx =
            ".version 8.0\n.target sm_80\n.address_size 64\n"
            ".visible .entry ptrwordval(.param .u64 out, .param .u64 src)\n"
            "{\n"
            "    .reg .b32 %lo, %hi, %c1, %c2, %v;\n"
            "    .reg .b64 %sp, %rd, %o;\n"
            "    ld.param.u64 %sp, [src];\n"
            "    ld.param.u32 %lo, [src+0];\n"
            "    mov.b64 {%c1, %c2}, %sp;\n"
            "    mad.lo.s32 %hi, 0, 0, %c2;\n"
            "    mov.b32 %c1, %lo;\n"
            "    mov.b64 %rd, {%c1, %hi};\n"
            "    ld.global.u32 %v, [%rd+4];\n"
            "    ld.param.u64 %o, [out];\n"
            "    st.global.u32 [%o], %v;\n"
            "    ret;\n"
            "}\n";
        char err[1024];
        acuda_kernel_t *k = NULL;
        if (acuda_compile_ptx(ptx, NULL, &k, err, sizeof(err), &g_test_caps) !=
            ACUDA_SUCCESS) {
            printf("FAIL: ptrwordval compile: %s\n", err);
            failures++;
        } else {
            acuda_kernel_info_t info;
            CHECK_ST(acuda_kernel_info(k, &info));
            CHECK(info.num_ptr_params == 1);
            CHECK(info.push_bytes == 8);
            acuda_buffer_t buf = NULL, src = NULL;
            CHECK_ST(acuda_malloc(ctx, &buf, 8));
            CHECK_ST(acuda_malloc(ctx, &src, 8));
            uint32_t in[2] = {0x11111111u, 0x22222222u};
            CHECK_ST(acuda_memcpy_h2d(ctx, src, in, sizeof(in), NULL));
            uint32_t init[2] = {0xa5a5a5a5u, 0xa5a5a5a5u};
            CHECK_ST(acuda_memcpy_h2d(ctx, buf, init, sizeof(init), NULL));
            uint64_t src_addr = 0;
            CHECK_ST(acuda_buffer_device_address(src, &src_addr));
            CHECK_ST(acuda_launch_kernel(ctx, k, &buf, 1, &src_addr,
                                         sizeof(src_addr), 1, 1, NULL));
            CHECK_ST(acuda_stream_synchronize(ctx, NULL));
            uint32_t got[2] = {0, 0};
            CHECK_ST(acuda_memcpy_d2h(ctx, got, buf, sizeof(got), NULL));
            if (got[0] == 0x22222222u && got[1] == 0xa5a5a5a5u)
                printf("  ptrwordval: repacked address load ok\n");
            else {
                printf("FAIL: ptrwordval: got 0x%x 0x%x want 0x%x 0x%x\n",
                       got[0], got[1], 0x22222222u, 0xa5a5a5a5u);
                failures++;
            }
            CHECK_ST(acuda_free(ctx, src));
            CHECK_ST(acuda_free(ctx, buf));
            acuda_kernel_free(k);
        }
    }

    // Generic state-machine (kind 8) kernel that splits a .u64 pointer
    // parameter into 32-bit words, carries the pair around a loop, and
    // repacks it.  The low word register is a *dynamic home* register
    // (HM_SCALAR|HM_BDA): the lifted cuDNN convolve kernel uses it both
    // as an address offset and as one half of a 64-bit pointer.  Its
    // state slot holds the home-relative offset (zero here), so the
    // machine must also round-trip the materialized numeric word -
    // without that the
    // repacked pointer becomes {0, high} and every access lands at a
    // bogus low address.  The reconstructed words are compared against
    // the buffer's device address instead of dereferenced, so a
    // regression fails cleanly rather than faulting the device.
    //
    // The scalar word first also keeps the pointer's BDA push word at a
    // nonzero index (the address region starts after the scalar words).
    {
        static const char *ptx =
            ".version 7.0\n.target sm_80\n.address_size 64\n"
            ".visible .entry smword(.param .u32 n, .param .u64 addr)\n"
            "{\n"
            "    .reg .b32 %i, %lo, %hi, %r52, %r53,"
            " %r56, %r43, %t, %u, %nz;\n"
            "    .reg .pred %p, %q;\n"
            "    .reg .b64 %o, %rd, %a, %b, %cc;\n"
            "    ld.param.u64 %o, [addr];\n"
            "    ld.global.u32 %t, [%o];\n"
            "    ld.param.u32 %nz, [n];\n"
            "    ld.param.u32 %lo, [addr+0];\n"
            "    ld.param.u32 %hi, [addr+4];\n"
            "    mad.lo.s32 %r56, 0, 0, %lo;\n"
            "    add.s32 %r56, %r56, 0;\n"
            "    mad.lo.s32 %r43, 0, 0, %hi;\n"
            "    mov.u32 %i, 0;\n"
            "    setp.lt.u32 %q, %i, 10;\n"
            "    @%q bra B;\n"
            "    bra C;\n"
            "B:\n"
            "    mov.u32 %t, 0;\n"
            "    bra C;\n"
            "C:\n"
            "    mov.u32 %t, 0;\n"
            "L:\n"
            "    mov.b32 %r52, %r56;\n"
            "    mad.lo.s32 %r53, 0, 0, %r43;\n"
            "    mov.b64 %rd, {%r52, %r53};\n"
            "    st.global.u32 [%o+8], %r52;\n"
            "    st.global.u32 [%o+12], %r53;\n"
            "    and.b32 %nz, %nz, 0;\n"
            "    add.u32 %r52, %r52, %nz;\n"
            "    mov.u32 %t, 0;\n"
            "    mov.b64 %a, {%r52, %t};\n"
            "    mov.u32 %u, 0;\n"
            "    mov.b64 %b, {%u, %r53};\n"
            "    add.u64 %cc, %a, %b;\n"
            "    mov.b64 {%r56, %r43}, %cc;\n"
            "    add.u32 %i, %i, 1;\n"
            "    setp.lt.u32 %p, %i, 2;\n"
            "    @%p bra L;\n"
            "    ret;\n"
            "}\n";
        char err[1024];
        acuda_kernel_t *k = NULL;
        if (acuda_compile_ptx(ptx, NULL, &k, err, sizeof(err),
                              &g_test_caps) != ACUDA_SUCCESS) {
            printf("FAIL: smword compile: %s\n", err);
            failures++;
        } else {
            acuda_kernel_info_t info;
            CHECK_ST(acuda_kernel_info(k, &info));
            CHECK(info.num_ptr_params == 1);
            CHECK(info.push_bytes == 4);
            // The generic CFG machine dispatches with a subgroup
            // reduction; wave64 codegen for large machines has been
            // observed to drop subsequent cases, so the runtime must
            // force 32-lane subgroups.
            if (info.uses_subgroup != 1) {
                printf("FAIL: smword: state machine must request a "
                       "32-lane subgroup (uses_subgroup=%u)\n",
                       info.uses_subgroup);
                failures++;
            }
            acuda_buffer_t buf = NULL;
            CHECK_ST(acuda_malloc(ctx, &buf, 24));
            uint32_t in[6] = {0x22222222u, 0xa5a5a5a5u, 0, 0, 0, 0};
            CHECK_ST(acuda_memcpy_h2d(ctx, buf, in, sizeof(in), NULL));
            uint64_t addr = 0;
            CHECK_ST(acuda_buffer_device_address(buf, &addr));
            uint32_t n = 1;
            // A full 64-thread wave: the launch size that exposed the
            // wave64 dispatch miscompile.
            CHECK_ST(acuda_launch_kernel(ctx, k, &buf, 1, &n, sizeof(n),
                                         1, 64, NULL));
            CHECK_ST(acuda_stream_synchronize(ctx, NULL));
            uint32_t got[6] = {0, 0, 0, 0, 0, 0};
            CHECK_ST(acuda_memcpy_d2h(ctx, got, buf, sizeof(got), NULL));
            if (got[0] == 0x22222222u && got[1] == 0xa5a5a5a5u &&
                got[2] == (uint32_t)addr &&
                got[3] == (uint32_t)(addr >> 32))
                printf("  smword: pointer word survives state machine\n");
            else {
                printf("FAIL: smword: got words %08x %08x, "
                       "addr %08x %08x want %08x %08x\n",
                       got[2], got[3], (uint32_t)addr,
                       (uint32_t)(addr >> 32), (uint32_t)addr,
                       (uint32_t)(addr >> 32));
                failures++;
            }
            CHECK_ST(acuda_free(ctx, buf));
            acuda_kernel_free(k);
        }
    }

    // 32-bit PTX with BDA: cvta/add/ld/st over .u32 device pointers must
    // use the full 64-bit base, not the truncated low word (regression
    // for the FluidMark radixSortBlocks VK_ERROR_DEVICE_LOST).
    {
        static const char *ptx =
            ".version 4.1\n.target sm_52\n.address_size 32\n"
            ".visible .entry bda32copy(.param .u32 in_ptr, .param .u32 out_ptr, .param .u32 n)\n"
            "{\n"
            "    .reg .u32 %r<8>;\n"
            "    .reg .pred %p;\n"
            "    ld.param.u32 %r1, [in_ptr];\n"
            "    ld.param.u32 %r2, [out_ptr];\n"
            "    ld.param.u32 %r3, [n];\n"
            "    cvta.to.global.u32 %r4, %r1;\n"
            "    cvta.to.global.u32 %r5, %r2;\n"
            "    mov.u32 %r0, %tid.x;\n"
            "    setp.ge.u32 %p, %r0, %r3;\n"
            "    @%p bra DONE;\n"
            "    mul.lo.s32 %r6, %r0, 4;\n"
            "    add.s32 %r4, %r4, %r6;\n"
            "    add.s32 %r5, %r5, %r6;\n"
            "    ld.global.u32 %r7, [%r4];\n"
            "    st.global.u32 [%r5], %r7;\n"
            "DONE:\n"
            "    ret;\n"
            "}\n";
        char err[1024];
        acuda_kernel_t *k = NULL;
        if (acuda_compile_ptx(ptx, NULL, &k, err, sizeof(err),
                              &g_test_caps) != ACUDA_SUCCESS) {
            printf("FAIL: bda32copy compile: %s\n", err);
            failures++;
        } else {
            const uint32_t n = 64;
            acuda_buffer_t bufin = NULL, bufout = NULL;
            CHECK_ST(acuda_malloc(ctx, &bufin, (size_t)n * 4));
            CHECK_ST(acuda_malloc(ctx, &bufout, (size_t)n * 4));
            uint32_t host[64];
            for (uint32_t i = 0; i < n; i++)
                host[i] = 0x10000000u + i * 0x1111111u;
            CHECK_ST(acuda_memcpy_h2d(ctx, bufin, host, sizeof(host),
                                      NULL));
            CHECK_ST(acuda_memset_d8(ctx, bufout, 0, 0, (size_t)n * 4,
                                     NULL));
            acuda_buffer_t bufs[2] = { bufin, bufout };
            CHECK_ST(acuda_launch_kernel(ctx, k, bufs, 2, &n, sizeof(n),
                                         1, 64, NULL));
            CHECK_ST(acuda_stream_synchronize(ctx, NULL));
            uint32_t got[64] = {0};
            CHECK_ST(acuda_memcpy_d2h(ctx, got, bufout, sizeof(got),
                                      NULL));
            if (memcmp(got, host, sizeof(host)) == 0)
                printf("  bda32copy: 32-bit BDA copy ok\n");
            else {
                printf("FAIL: bda32copy: mismatch at");
                for (uint32_t i = 0; i < n; i++)
                    if (got[i] != host[i]) {
                        printf(" [%u got %08x want %08x]", i, got[i],
                               host[i]);
                        if (i > 4)
                            break;
                    }
                printf("\n");
                failures++;
            }
            CHECK_ST(acuda_free(ctx, bufin));
            CHECK_ST(acuda_free(ctx, bufout));
            acuda_kernel_free(k);
        }
    }

    // 32-bit PTX taking a .local address into a .u32 register and doing
    // address arithmetic on it must emit valid SPIR-V (regression: the
    // address-taken mov stored an AccessChain pointer id in the
    // register's numeric word, so add.u32 emitted an OpIAdd with a
    // pointer operand - invalid SPIR-V that crashed nvgpucomp32 on the
    // FluidMark Force pipeline).
    {
        static const char *ptx =
            ".version 4.1\n.target sm_52\n.address_size 32\n"
            ".visible .entry localaddr32(.param .u32 out_ptr, .param .u32 n)\n"
            "{\n"
            "    .local .align 4 .b8 tmp[256];\n"
            "    .reg .u32 %r<10>;\n"
            "    .reg .pred %p;\n"
            "    ld.param.u32 %r1, [out_ptr];\n"
            "    ld.param.u32 %r3, [n];\n"
            "    mov.u32 %r0, %tid.x;\n"
            "    setp.ge.u32 %p, %r0, %r3;\n"
            "    @%p bra DONE;\n"
            "    cvta.to.global.u32 %r6, %r1;\n"
            "    mov.u32 %r4, tmp;\n"
            "    mul.lo.s32 %r5, %r0, 4;\n"
            "    add.u32 %r4, %r4, %r5;\n"
            "    st.local.u32 [%r4], %r0;\n"
            "    ld.local.u32 %r7, [%r4];\n"
            "    mul.lo.s32 %r8, %r0, 4;\n"
            "    add.u32 %r6, %r6, %r8;\n"
            "    st.global.u32 [%r6], %r7;\n"
            "DONE:\n"
            "    ret;\n"
            "}\n";
        char err[1024];
        acuda_kernel_t *k = NULL;
        if (acuda_compile_ptx(ptx, NULL, &k, err, sizeof(err),
                              &g_test_caps) != ACUDA_SUCCESS) {
            printf("FAIL: localaddr32 compile: %s\n", err);
            failures++;
        } else {
            const uint32_t n = 64;
            acuda_buffer_t bufout = NULL;
            CHECK_ST(acuda_malloc(ctx, &bufout, (size_t)n * 4));
            CHECK_ST(acuda_memset_d8(ctx, bufout, 0, 0, (size_t)n * 4,
                                     NULL));
            CHECK_ST(acuda_launch_kernel(ctx, k, &bufout, 1, &n, sizeof(n),
                                         1, 64, NULL));
            CHECK_ST(acuda_stream_synchronize(ctx, NULL));
            uint32_t got[64] = {0};
            CHECK_ST(acuda_memcpy_d2h(ctx, got, bufout, sizeof(got),
                                      NULL));
            int ok = 1;
            for (uint32_t i = 0; i < n; i++)
                if (got[i] != i)
                    ok = 0;
            if (ok)
                printf("  localaddr32: .local address arithmetic ok\n");
            else {
                printf("FAIL: localaddr32: mismatch at");
                for (uint32_t i = 0; i < n; i++)
                    if (got[i] != i) {
                        printf(" [%u got %08x want %08x]", i, got[i],
                               i);
                        if (i > 4)
                            break;
                    }
                printf("\n");
                failures++;
            }
            CHECK_ST(acuda_free(ctx, bufout));
            acuda_kernel_free(k);
        }
    }

    // 1D unsigned-int texture fetch over multiple tile rows, with
    // 0xFFFFFFFF sentinel texels (regression: FluidMark's cell
    // tables are large 1D u32 textures full of -1 empty markers;
    // only float 1D sampling was covered before).
    {
        static const char *ptx =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".visible .entry u321dfetch(.param .texref tex, .param .u64 out, .param .u32 n)\n"
            "{\n"
            "    .reg .pred %p;\n"
            "    .reg .s32 %tx, %bx, %bd, %t, %n, %b;\n"
            "    .reg .u64 %o, %ob, %addr;\n"
            "    .reg .u32 %r0, %r1, %r2, %r3;\n"
            "    ld.param.u64 %o, [out];\n"
            "    ld.param.u32 %n, [n];\n"
            "    mov.u32 %tx, %tid.x;\n"
            "    mov.u32 %bx, %ctaid.x;\n"
            "    mov.u32 %bd, %ntid.x;\n"
            "    mad.lo.s32 %t, %bx, %bd, %tx;\n"
            "    setp.ge.u32 %p, %t, %n;\n"
            "    @%p bra DONE;\n"
            "    mul.lo.s32 %b, %t, 4;\n"
            "    cvt.u64.u32 %ob, %b;\n"
            "    add.u64 %addr, %o, %ob;\n"
            "    tex.1d.v4.u32.s32 {%r0, %r1, %r2, %r3}, [tex, {%t}];\n"
            "    st.global.u32 [%addr], %r0;\n"
            "DONE:\n"
            "    ret;\n"
            "}\n";
        const uint32_t n = 20000;
        char err[1024];
        acuda_kernel_t *k = NULL;
        if (acuda_compile_ptx(ptx, "u321dfetch", &k, err, sizeof(err),
                              &g_test_caps) != ACUDA_SUCCESS) {
            printf("FAIL: u321dfetch compile: %s\n", err);
            failures++;
        } else {
            uint32_t *texels = malloc((size_t)n * 4);
            if (!texels) {
                printf("FAIL: u321dfetch oom\n");
                failures++;
            } else {
                for (uint32_t i = 0; i < n; i++)
                    texels[i] = (i ^ 0x9e3779b9u) + 0x1234567u;
                texels[7] = 0xffffffffu;
                texels[16383] = 0xffffffffu;
                texels[16384] = 0x12345678u;
                texels[n - 1] = 0xffffffffu;
                acuda_image_desc idesc = { n, 1, 1, 1, 1, 1, 3, 1, 0 };
                acuda_sampler_desc sdesc = { 0, 0, { 2, 2, 2 }, 1,
                                             { 0, 0, 0, 0 }, 0, 0 };
                acuda_buffer_t img = NULL, smp = NULL;
                acuda_buffer_t tex = NULL, out = NULL;
                CHECK_ST(acuda_image_create(ctx, &img, &idesc, texels));
                CHECK_ST(acuda_sampler_create(ctx, &smp, &sdesc));
                CHECK_ST(acuda_texture_create_image(ctx, &tex, img, smp));
                CHECK_ST(acuda_malloc(ctx, &out, (size_t)n * 4));
                acuda_buffer_t tb[2] = { tex, out };
                uint32_t nn = n;
                CHECK_ST(acuda_launch_kernel(ctx, k, tb, 2, &nn,
                                             sizeof(nn),
                                             (n + 255) / 256, 256, NULL));
                CHECK_ST(acuda_stream_synchronize(ctx, NULL));
                uint32_t *got = malloc((size_t)n * 4);
                if (!got) {
                    printf("FAIL: u321dfetch oom\n");
                    failures++;
                } else {
                    CHECK_ST(acuda_memcpy_d2h(ctx, got, out,
                                             (size_t)n * 4, NULL));
                    int bad = 0;
                    for (uint32_t i = 0; i < n; i++)
                        if (got[i] != texels[i]) {
                            if (bad < 6)
                                printf("FAIL: u321dfetch [%u] got %08x want %08x\n",
                                       i, got[i], texels[i]);
                            bad++;
                        }
                    if (bad)
                        printf("FAIL: u321dfetch: %d mismatches\n", bad);
                    else
                        printf("  u321dfetch: 1D u32 fetch incl. tiling ok\n");
                    failures += bad ? 1 : 0;
                    free(got);
                }
                CHECK_ST(acuda_free(ctx, out));
                CHECK_ST(acuda_free(ctx, tex));
                CHECK_ST(acuda_free(ctx, smp));
                CHECK_ST(acuda_free(ctx, img));
                free(texels);
            }
            acuda_kernel_free(k);
        }
    }

    // Divergent CFG state-machine kernel with a block-wide barrier:
    // must not hang (regression for subgroup-min PC lockstep deadlock)
    // and must produce correct per-thread results.
    {
        static const char *ptx =
            ".version 6.0\n.target sm_50\n.address_size 64\n"
            ".visible .entry cfg_sm_bar_diverge(.param .u64 out_ptr)\n"
            "{\n"
            "    .reg .pred %p<4>;\n"
            "    .reg .u32 %r<12>;\n"
            "    .reg .u64 %rd<4>;\n"
            "    ld.param.u64 %rd1, [out_ptr];\n"
            "    mov.u32 %r0, %tid.x;\n"
            "    mov.u32 %r1, 0;\n"
            "    mov.u32 %r2, %r0;\n"
            "LOOP:\n"
            "    setp.lt.u32 %p1, %r1, 4;\n"
            "    @!%p1 bra DONE;\n"
            "    and.b32 %r3, %r2, 1;\n"
            "    setp.eq.u32 %p2, %r3, 0;\n"
            "    @%p2 bra EVEN;\n"
            "ODD:\n"
            "    add.u32 %r4, %r2, %r1;\n"
            "    mad.lo.u32 %r2, %r4, 3, 1;\n"
            "    setp.lt.u32 %p3, %r2, 100;\n"
            "    @%p3 bra MERGE;\n"
            "    sub.u32 %r2, %r2, 7;\n"
            "    bra MERGE;\n"
            "EVEN:\n"
            "    add.u32 %r5, %r2, %r0;\n"
            "    mad.lo.u32 %r2, %r5, 5, 2;\n"
            "    setp.ge.u32 %p3, %r2, 50;\n"
            "    @%p3 bra MERGE;\n"
            "    add.u32 %r2, %r2, 9;\n"
            "MERGE:\n"
            "    bar.sync 0;\n"
            "    add.u32 %r1, %r1, 1;\n"
            "    setp.lt.u32 %p1, %r1, 4;\n"
            "    @%p1 bra LOOP;\n"
            "DONE:\n"
            "    mul.wide.u32 %rd2, %r0, 4;\n"
            "    add.s64 %rd3, %rd1, %rd2;\n"
            "    st.global.u32 [%rd3], %r2;\n"
            "    ret;\n"
            "}\n";
        char err[1024];
        acuda_kernel_t *k = NULL;
        if (acuda_compile_ptx(ptx, NULL, &k, err, sizeof(err),
                              &g_test_caps) != ACUDA_SUCCESS) {
            printf("FAIL: cfg_sm_bar_diverge compile: %s\n", err);
            failures++;
        } else {
            acuda_buffer_t buf = NULL;
            CHECK_ST(acuda_malloc(ctx, &buf, 16));
            uint32_t zero[4] = {0, 0, 0, 0};
            CHECK_ST(acuda_memcpy_h2d(ctx, buf, zero, sizeof(zero), NULL));
            CHECK_ST(acuda_launch_kernel(ctx, k, &buf, 1, NULL, 0, 1, 4, NULL));
            CHECK_ST(acuda_stream_synchronize(ctx, NULL));
            uint32_t got[4] = {0, 0, 0, 0};
            CHECK_ST(acuda_memcpy_d2h(ctx, got, buf, sizeof(got), NULL));
            const uint32_t want[4] = {336u, 564u, 876u, 606u};
            if (memcmp(got, want, sizeof(want)) == 0)
                printf("  cfg_sm_bar_diverge: ok\n");
            else {
                printf("FAIL: cfg_sm_bar_diverge: got %u %u %u %u want %u %u %u %u\n",
                       got[0], got[1], got[2], got[3],
                       want[0], want[1], want[2], want[3]);
                failures++;
            }
            CHECK_ST(acuda_free(ctx, buf));
            acuda_kernel_free(k);
        }
    }

    // Negative brx.idx forms: missing .branchtargets list and a
    // backward target (the structured CFG only supports forward edges).
    {
        char err[1024];
        acuda_kernel_t *k = NULL;
        const char *badlist =
            ".version 6.5\n.target sm_30\n.address_size 64\n"
            ".visible .entry brxmissing(.param .u64 a)\n"
            "{\n    .reg .u64 %rd;\n    .reg .u32 %r;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    ld.global.u32 %r, [%rd];\n"
            "    brx.idx %r, nope;\n    ret;\n}\n";
        CHECK(acuda_compile_ptx(badlist, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative brx missing list: %s\n", err);
        const char *badback =
            ".version 6.5\n.target sm_30\n.address_size 64\n"
            ".visible .entry brxback(.param .u64 a)\n"
            "{\n    .reg .u64 %rd;\n    .reg .u32 %r;\n"
            "    ld.param.u64 %rd, [a];\n"
            "    ld.global.u32 %r, [%rd];\n"
            "L0: mov.u32 %r, 1;\n"
            "ts: .branchtargets L0;\n"
            "    brx.idx %r, ts;\n    ret;\n}\n";
        CHECK(acuda_compile_ptx(badback, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative brx backward target: %s\n", err);
    }

    // Negative: a function pointer loaded from memory without a
    // .calltargets list cannot be dispatched, and a list naming an
    // unknown function is rejected cleanly.
    {
        char err[1024];
        acuda_kernel_t *k = NULL;
        const char *badind =
            ".version 6.5\n.target sm_30\n.address_size 64\n"
            ".func (.reg .u64 out) f1(.reg .u64 x)\n"
            "{\n    add.u64 out, x, 1;\n    ret;\n}\n"
            ".global .u64 tbl[1] = { f1 };\n"
            ".visible .entry badind(.param .u64 data)\n"
            "{\n    .reg .u64 base, val, r, f, p, off;\n"
            "    .reg .u32 idx;\n"
            "    ld.param.u64 base, [data];\n"
            "    ld.global.u64 val, [base];\n"
            "    mov.u32 idx, 0;\n"
            "    mov.u64 p, tbl;\n"
            "    mul.wide.u32 off, idx, 8;\n"
            "    add.u64 p, p, off;\n"
            "    ld.global.u64 f, [p];\n"
            "proto: .callprototype (.param .b64 r) _ (.param .b64 x);\n"
            "    call (r), f, (val), proto;\n"
            "    st.u64 [base+8], r;\n    ret;\n}\n";
        CHECK(acuda_compile_ptx(badind, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative indirect without calltargets: %s\n", err);
        const char *badlist =
            ".version 6.5\n.target sm_30\n.address_size 64\n"
            ".visible .entry badlist(.param .u64 data)\n"
            "{\n    .reg .u64 base, val, r, f;\n"
            "    ld.param.u64 base, [data];\n"
            "    ld.global.u64 val, [base];\n"
            "    mov.u64 f, nope;\n"
            "tgt: .calltargets nope;\n"
            "    call (r), f, (val), tgt;\n"
            "    st.u64 [base+8], r;\n    ret;\n}\n";
        CHECK(acuda_compile_ptx(badlist, NULL, &k, err, sizeof(err), &g_test_caps) !=
              ACUDA_SUCCESS);
        printf("  negative indirect unknown target: %s\n", err);
    }

    // Dynamic ld.const with an out-of-range index must not fault:
    // real CUDA tolerates OOB constant reads (undefined value), while
    // our Private-bank lowering used to emit an unclamped private access
    // that killed the device (FluidMark density reads const params with
    // a data-dependent index that can exceed the declared bank).
    // OOB reads saturate to the last bank word (zero-filled here).
    {
        static const char *ptx =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".const .align 4 .b32 cvals[4] = {11, 22, 33, 44};\n"
            ".visible .entry constbank_oob(.param .u64 out)\n"
            "{\n"
            "    .reg .pred %p;\n"
            "    .reg .u32 %t, %i, %b, %v, %base, %off;\n"
            "    .reg .u64 %o, %addr;\n"
            "    ld.param.u64 %o, [out];\n"
            "    mov.u32 %t, %tid.x;\n"
            "    setp.eq.u32 %p, %t, 0;\n"
            "    mov.u32 %i, 1000000;\n"
            "    @%p mov.u32 %i, 1;\n"
            "    mov.u32 %base, cvals;\n"
            "    mad.lo.s32 %off, %i, 4, %base;\n"
            "    ld.const.u32 %v, [%off];\n"
            "    mul.wide.u32 %addr, %t, 4;\n"
            "    add.u64 %addr, %o, %addr;\n"
            "    st.global.u32 [%addr], %v;\n"
            "    ret;\n"
            "}\n";
        char err[1024];
        acuda_kernel_t *k = NULL;
        if (acuda_compile_ptx(ptx, "constbank_oob", &k, err, sizeof(err),
                              &g_test_caps) != ACUDA_SUCCESS) {
            printf("FAIL: constbank_oob compile: %s\n", err);
            failures++;
        } else {
            acuda_buffer_t buf = NULL;
            CHECK_ST(acuda_malloc(ctx, &buf, 8));
            uint32_t zero[2] = {0, 0};
            CHECK_ST(acuda_memcpy_h2d(ctx, buf, zero, sizeof(zero), NULL));
            CHECK_ST(acuda_launch_kernel(ctx, k, &buf, 1, NULL, 0, 1, 2, NULL));
            CHECK_ST(acuda_stream_synchronize(ctx, NULL));
            uint32_t got[2] = {0, 0};
            CHECK_ST(acuda_memcpy_d2h(ctx, got, buf, sizeof(got), NULL));
            if (got[0] == 22u && got[1] == 0u)
                printf("  constbank_oob: in-range ok, OOB saturates to zero\n");
            else {
                printf("FAIL: constbank_oob: got %u %u want 22 0\n",
                       got[0], got[1]);
                failures++;
            }
            CHECK_ST(acuda_free(ctx, buf));
            acuda_kernel_free(k);
        }
    }

    // Do-while latch guard: a backward `cbra` on a `setp.ne` latch over a
    // +1-stepped counter must terminate on a degenerate range. tid0 runs
    // a valid range (0..3 -> sum 0+1+2=3); tid1 runs a degenerate range
    // (start=1 > end=0 -> body once, sum=1). Without the NE->LTU latch
    // rewrite tid1 spins ~2^32 iterations and the sync below never
    // returns (FluidMark density hang).
    {
        static const char *ptx =
            ".version 8.6\n.target sm_80\n.address_size 64\n"
            ".visible .entry latch_degenerate(.param .u64 out)\n"
            "{\n"
            "    .reg .pred %q, %r;\n"
            "    .reg .u32 %p, %e, %s, %sum, %t;\n"
            "    .reg .u64 %o, %addr;\n"
            "    ld.param.u64 %o, [out];\n"
            "    mov.u32 %t, %tid.x;\n"
            "    mov.u32 %s, %t;\n"
            "    setp.eq.u32 %r, %t, 0;\n"
            "    selp.u32 %e, 3, 0, %r;\n"
            "    mov.u32 %p, %s;\n"
            "    mov.u32 %sum, 0;\n"
            "LOOP:\n"
            "    add.u32 %sum, %sum, %p;\n"
            "    add.u32 %p, %p, 1;\n"
            "    setp.ne.u32 %q, %p, %e;\n"
            "    @%q bra LOOP;\n"
            "    mul.wide.u32 %addr, %t, 4;\n"
            "    add.u64 %addr, %o, %addr;\n"
            "    st.global.u32 [%addr], %sum;\n"
            "    ret;\n"
            "}\n";
        char err[1024];
        acuda_kernel_t *k = NULL;
        if (acuda_compile_ptx(ptx, "latch_degenerate", &k, err, sizeof(err),
                              &g_test_caps) != ACUDA_SUCCESS) {
            printf("FAIL: latch_degenerate compile: %s\n", err);
            failures++;
        } else {
            acuda_buffer_t buf = NULL;
            CHECK_ST(acuda_malloc(ctx, &buf, 8));
            uint32_t zero[2] = {0, 0};
            CHECK_ST(acuda_memcpy_h2d(ctx, buf, zero, sizeof(zero), NULL));
            CHECK_ST(acuda_launch_kernel(ctx, k, &buf, 1, NULL, 0, 1, 2, NULL));
            CHECK_ST(acuda_stream_synchronize(ctx, NULL));
            uint32_t got[2] = {0, 0};
            CHECK_ST(acuda_memcpy_d2h(ctx, got, buf, sizeof(got), NULL));
            if (got[0] == 3u && got[1] == 1u)
                printf("  latch_degenerate: valid ok, degenerate exits\n");
            else {
                printf("FAIL: latch_degenerate: got %u %u want 3 1\n",
                       got[0], got[1]);
                failures++;
            }
            CHECK_ST(acuda_free(ctx, buf));
            acuda_kernel_free(k);
        }
    }

    uint32_t verr = 0;
    CHECK_ST(acuda_validation_errors(ctx, &verr));
    CHECK(verr == 0);
    CHECK_ST(acuda_shutdown(ctx));

    check_param_write_flags();
    check_f64_param_push();

    if (failures == 0)
        printf("ptx compiler integration test passed\n");
    return failures != 0;
}
