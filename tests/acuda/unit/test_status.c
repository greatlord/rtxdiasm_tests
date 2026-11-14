// Unit tests for status codes and GPU-free error paths.
#include <stdio.h>
#include <string.h>
#include "acuda/runtime.h"
#include "acuda/memory.h"
#include "acuda/stream.h"

static int failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        failures++; \
    } \
} while (0)

int main(void) {
    CHECK(strcmp(acuda_strerror(ACUDA_SUCCESS), "success") == 0);
    CHECK(strcmp(acuda_strerror(ACUDA_ERROR_NOT_READY), "not ready") == 0);
    CHECK(strcmp(acuda_strerror(ACUDA_ERROR_TIMEOUT), "timeout") == 0);
    CHECK(acuda_strerror((acuda_status_t)999) != NULL);

    // All of these must fail cleanly without a GPU.
    CHECK(acuda_init(NULL) == ACUDA_ERROR_INVALID_VALUE);
    CHECK(acuda_shutdown(NULL) == ACUDA_ERROR_INVALID_HANDLE);
    CHECK(acuda_device_count(NULL, NULL) == ACUDA_ERROR_INVALID_HANDLE);
    CHECK(acuda_malloc(NULL, NULL, 0) == ACUDA_ERROR_INVALID_HANDLE);
    CHECK(acuda_free(NULL, NULL) == ACUDA_ERROR_INVALID_HANDLE);
    CHECK(acuda_buffer_size(NULL, NULL) == ACUDA_ERROR_INVALID_VALUE);
    CHECK(acuda_memcpy_d2d(NULL, NULL, NULL, 0, NULL) == ACUDA_ERROR_INVALID_HANDLE);
    CHECK(acuda_memset_d32(NULL, NULL, 0, 0, 0, NULL) == ACUDA_ERROR_INVALID_HANDLE);
    CHECK(acuda_stream_create(NULL, NULL) == ACUDA_ERROR_INVALID_HANDLE);
    CHECK(acuda_stream_synchronize(NULL, NULL) == ACUDA_ERROR_INVALID_HANDLE);
    CHECK(acuda_event_create(NULL, NULL) == ACUDA_ERROR_INVALID_HANDLE);
    CHECK(acuda_event_query(NULL, NULL) == ACUDA_ERROR_INVALID_HANDLE);
    CHECK(acuda_event_record(NULL, NULL, NULL) == ACUDA_ERROR_INVALID_HANDLE);
    CHECK(acuda_validation_errors(NULL, NULL) == ACUDA_ERROR_INVALID_HANDLE);

    if (failures == 0)
        printf("unit status/error-path tests passed\n");
    return failures != 0;
}
