// Integration: device query + memory round-trip on a real GPU.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "acuda/context.h"
#include "acuda/memory.h"
#include "acuda/stream.h"

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

#define N (1u << 20)

int main(void) {
    acuda_context_t ctx = NULL;
    acuda_status_t st = acuda_init(&ctx);
    if (st == ACUDA_ERROR_NO_DEVICE || st == ACUDA_ERROR_NOT_SUPPORTED) {
        printf("SKIPPED: no suitable Vulkan device (%s)\n", acuda_strerror(st));
        return 0;
    }
    CHECK(st == ACUDA_SUCCESS);
    if (!ctx)
        return 1;

    uint32_t ndev = 0, sel = 0;
    CHECK_ST(acuda_device_count(ctx, &ndev));
    CHECK(ndev >= 1);
    CHECK_ST(acuda_device_selected(ctx, &sel));
    CHECK(sel < ndev);
    for (uint32_t i = 0; i < ndev; i++) {
        acuda_device_info_t info;
        CHECK_ST(acuda_device_get(ctx, i, &info));
        printf("  device[%u] %s discrete=%d api=%u.%u.%u%s\n", i, info.name,
               info.is_discrete_gpu, info.api_major, info.api_minor, info.api_patch,
               i == sel ? " (selected)" : "");
    }
    acuda_device_info_t bad;
    CHECK(acuda_device_get(ctx, ndev, &bad) == ACUDA_ERROR_INVALID_VALUE);

    acuda_buffer_t a = NULL, b = NULL;
    CHECK(acuda_malloc(ctx, &a, 0) == ACUDA_ERROR_INVALID_VALUE);
    CHECK_ST(acuda_malloc(ctx, &a, (size_t)N * sizeof(float)));
    CHECK_ST(acuda_malloc(ctx, &b, (size_t)N * sizeof(float)));
    size_t sz = 0;
    CHECK_ST(acuda_buffer_size(a, &sz));
    CHECK(sz == (size_t)N * sizeof(float));
    {
        uint64_t daddr = 0;
        acuda_status_t ast = acuda_buffer_device_address(a, &daddr);
        CHECK(ast == ACUDA_SUCCESS ||
              ast == ACUDA_ERROR_NOT_SUPPORTED);
        if (ast == ACUDA_SUCCESS) {
            CHECK(daddr != 0);
            printf("  buffer device address: 0x%llx\n",
                   (unsigned long long)daddr);
        }
    }

    float *host = malloc((size_t)N * sizeof(float));
    float *back = malloc((size_t)N * sizeof(float));
    CHECK(host && back);
    for (uint32_t i = 0; i < N; i++)
        host[i] = (float)i * 1.25f;

    CHECK_ST(acuda_memcpy_h2d(ctx, a, host, (size_t)N * sizeof(float), NULL));
    CHECK_ST(acuda_memcpy_d2d(ctx, b, a, (size_t)N * sizeof(float), NULL));
    // Overwrite first 256 floats of b with 32-bit pattern, rest with byte pattern.
    CHECK_ST(acuda_memset_d32(ctx, b, 0x3f800000u, 0, 256 * sizeof(float), NULL));
    CHECK_ST(acuda_memset_d8(ctx, b, 0x00, 256 * sizeof(float),
                             ((size_t)N - 256) * sizeof(float), NULL));
    CHECK_ST(acuda_stream_synchronize(ctx, NULL));
    CHECK_ST(acuda_memcpy_d2h(ctx, back, b, (size_t)N * sizeof(float), NULL));

    uint32_t bad_count = 0;
    for (uint32_t i = 0; i < N; i++) {
        float want = (i < 256) ? 1.0f : 0.0f;
        if (back[i] != want && bad_count < 5)
            printf("  mismatch[%u]: got %f want %f\n", i, back[i], want);
        if (back[i] != want)
            bad_count++;
    }
    CHECK(bad_count == 0);

    // D2H of untouched buffer a must equal host pattern.
    CHECK_ST(acuda_memcpy_d2h(ctx, back, a, (size_t)N * sizeof(float), NULL));
    CHECK(memcmp(back, host, (size_t)N * sizeof(float)) == 0);

    // Error paths with live context.
    CHECK(acuda_memcpy_h2d(ctx, a, host, sz + 1, NULL) == ACUDA_ERROR_INVALID_VALUE);
    CHECK(acuda_memset_d32(ctx, a, 0, 1, 4, NULL) == ACUDA_ERROR_INVALID_VALUE);
    CHECK(acuda_memset_d8(ctx, a, 0, 0, 3, NULL) == ACUDA_ERROR_INVALID_VALUE);

    free(host);
    free(back);
    CHECK_ST(acuda_free(ctx, a));
    CHECK_ST(acuda_free(ctx, b));
    uint32_t verr = 0;
    CHECK_ST(acuda_validation_errors(ctx, &verr));
    CHECK(verr == 0);
    CHECK_ST(acuda_shutdown(ctx));

    if (failures == 0)
        printf("memory integration test passed\n");
    return failures != 0;
}
