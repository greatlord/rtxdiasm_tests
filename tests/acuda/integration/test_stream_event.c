// Integration: streams + events ordering on a real GPU.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
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

#define NBYTES (1u << 20)

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

    acuda_stream_t s1 = NULL, s2 = NULL;
    acuda_event_t e = NULL, fresh = NULL;
    CHECK_ST(acuda_stream_create(ctx, &s1));
    CHECK_ST(acuda_stream_create(ctx, &s2));
    CHECK_ST(acuda_event_create(ctx, &e));
    CHECK_ST(acuda_event_create(ctx, &fresh));

    // Fresh event: query/record-wait semantics before first record.
    CHECK(acuda_event_query(ctx, fresh) == ACUDA_ERROR_UNINITIALIZED);
    CHECK(acuda_event_synchronize(ctx, fresh) == ACUDA_ERROR_UNINITIALIZED);
    CHECK(acuda_stream_wait_event(ctx, s1, fresh) == ACUDA_ERROR_UNINITIALIZED);

    acuda_buffer_t a = NULL, b = NULL;
    CHECK_ST(acuda_malloc(ctx, &a, NBYTES));
    CHECK_ST(acuda_malloc(ctx, &b, NBYTES));

    // s1: fill a with 0xAB bytes, then record e.
    CHECK_ST(acuda_memset_d8(ctx, a, 0xAB, 0, NBYTES, s1));
    CHECK_ST(acuda_event_record(ctx, e, s1));
    // s2 must wait for e before copying a -> b.
    CHECK_ST(acuda_stream_wait_event(ctx, s2, e));
    CHECK_ST(acuda_memcpy_d2d(ctx, b, a, NBYTES, s2));

    CHECK_ST(acuda_stream_synchronize(ctx, s1));
    CHECK_ST(acuda_stream_synchronize(ctx, s2));
    CHECK_ST(acuda_event_synchronize(ctx, e));
    CHECK(acuda_event_query(ctx, e) == ACUDA_SUCCESS);

    unsigned char *back = malloc(NBYTES);
    CHECK(back != NULL);
    CHECK_ST(acuda_memcpy_d2h(ctx, back, b, NBYTES, NULL));
    uint32_t bad = 0;
    for (size_t i = 0; i < NBYTES; i += 4096) {
        if (back[i] != 0xAB) {
            if (bad < 5)
                printf("  mismatch[%zu]: got 0x%02x want 0xab\n", i, back[i]);
            bad++;
        }
    }
    CHECK(bad == 0);
    free(back);

    // Re-record works (generation counter).
    CHECK_ST(acuda_memset_d8(ctx, a, 0xCD, 0, NBYTES, s1));
    CHECK_ST(acuda_event_record(ctx, e, s1));
    CHECK_ST(acuda_event_synchronize(ctx, e));
    CHECK(acuda_event_query(ctx, e) == ACUDA_SUCCESS);

    // More than one batch threshold worth of small fills: ordering must
    // survive the open-batch flush boundary.
    for (uint32_t i = 0; i < 100; i++)
        CHECK_ST(acuda_memset_d32(ctx, a, i, i * 4u, 4, s1));
    CHECK_ST(acuda_stream_synchronize(ctx, s1));
    uint32_t *words = calloc(100, sizeof(uint32_t));
    CHECK(words != NULL);
    CHECK_ST(acuda_memcpy_d2h(ctx, words, a, 100 * sizeof(uint32_t), NULL));
    for (uint32_t i = 0; i < 100; i++)
        CHECK(words[i] == i);
    free(words);

    // Optional GPU timestamps: elapsed time between two recorded events.
    acuda_event_t t0 = NULL, t1 = NULL;
    CHECK_ST(acuda_event_create(ctx, &t0));
    CHECK_ST(acuda_event_create(ctx, &t1));
    CHECK_ST(acuda_event_record(ctx, t0, s1));
    CHECK_ST(acuda_memset_d8(ctx, a, 0x11, 0, NBYTES, s1));
    CHECK_ST(acuda_event_record(ctx, t1, s1));
    uint64_t elapsed_ns = 0;
    st = acuda_event_elapsed(ctx, t0, t1, &elapsed_ns);
    CHECK(st == ACUDA_SUCCESS || st == ACUDA_ERROR_NOT_SUPPORTED);
    if (st == ACUDA_SUCCESS) {
        // A 1 MiB memset should complete well under a second of GPU time.
        CHECK(elapsed_ns < 1000000000ull);
    }
    CHECK_ST(acuda_event_destroy(ctx, t0));
    CHECK_ST(acuda_event_destroy(ctx, t1));

    CHECK_ST(acuda_event_destroy(ctx, e));
    CHECK_ST(acuda_event_destroy(ctx, fresh));
    CHECK_ST(acuda_stream_destroy(ctx, s1));
    CHECK_ST(acuda_stream_destroy(ctx, s2));
    CHECK_ST(acuda_free(ctx, a));
    CHECK_ST(acuda_free(ctx, b));
    uint32_t verr = 0;
    CHECK_ST(acuda_validation_errors(ctx, &verr));
    CHECK(verr == 0);
    CHECK_ST(acuda_shutdown(ctx));

    if (failures == 0)
        printf("stream/event integration test passed\n");
    return failures != 0;
}
