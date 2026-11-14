// Windows GPU smoke test for ACuda's steady launch-cache path.
//
// Exercises the driver API exactly like a CUDA client: repeated identical
// launches (descriptor + recorded command cache), changing push constants,
// changing output buffers, enqueued launches, and stream/event completion.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "acuda/driver.h"

#ifndef ACUDA_TEST_PTX_DIR
#define ACUDA_TEST_PTX_DIR "tests/ptx"
#endif

#define N 4096u
#define THREADS 128u

static int failures = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        failures++; \
    } \
} while (0)
#define CHECK_CU(call) do { \
    CUresult _r = (call); \
    if (_r != CUDA_SUCCESS) { \
        printf("FAIL %s:%d: %s -> %d\n", __FILE__, __LINE__, #call, \
               (int)_r); \
        failures++; \
    } \
} while (0)

static char *read_text(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n <= 0) {
        fclose(f);
        return NULL;
    }
    char *text = malloc((size_t)n + 1);
    if (!text || fread(text, 1, (size_t)n, f) != (size_t)n) {
        free(text);
        fclose(f);
        return NULL;
    }
    text[n] = '\0';
    fclose(f);
    return text;
}

static int verify(const char *label, const uint32_t *input,
                  const uint32_t *output, uint32_t inc) {
    uint32_t bad = 0;
    for (uint32_t i = 0; i < N; i++) {
        uint32_t want = input[i] + inc;
        if (output[i] != want) {
            if (bad < 4)
                printf("  %s mismatch[%u]: got 0x%08x want 0x%08x\n",
                       label, i, output[i], want);
            bad++;
        }
    }
    if (bad) {
        printf("FAIL %s: %u/%u words differ\n", label, bad, N);
        failures++;
        return 0;
    }
    return 1;
}

static CUresult launch(CUfunction function, CUdeviceptr out,
                       CUdeviceptr in, uint32_t n, uint32_t inc,
                       CUstream stream, void *args[4]) {
    args[0] = &out;
    args[1] = &in;
    args[2] = &n;
    args[3] = &inc;
    return cuLaunchKernel(function, (N + THREADS - 1) / THREADS, 1, 1,
                          THREADS, 1, 1, 0, stream, args, NULL);
}

int main(void) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/launch_smoke.ptx",
             ACUDA_TEST_PTX_DIR);
    char *ptx = read_text(path);
    if (!ptx) {
        printf("FAIL: cannot read %s\n", path);
        return 1;
    }

    CHECK_CU(cuInit(0));
    CUdevice device = 0;
    CHECK_CU(cuDeviceGet(&device, 0));
    CUcontext context = NULL;
    CHECK_CU(cuCtxCreate(&context, 0, device));
    CUmodule module = NULL;
    CHECK_CU(cuModuleLoadDataEx(&module, ptx, 0, NULL, NULL));
    CUfunction function = NULL;
    CUfunction sat_function = NULL;
    CHECK_CU(cuModuleGetFunction(&function, module, "smoke_add"));
    CHECK_CU(cuModuleGetFunction(&sat_function, module, "smoke_sat"));
    if (failures != 0) {
        if (module)
            cuModuleUnload(module);
        if (context)
            cuCtxDestroy(context);
        free(ptx);
        return failures != 0;
    }

    const size_t bytes = (size_t)N * sizeof(uint32_t);
    CUdeviceptr in = 0, out0 = 0, out1 = 0;
    CHECK_CU(cuMemAlloc(&in, bytes));
    CHECK_CU(cuMemAlloc(&out0, bytes));
    CHECK_CU(cuMemAlloc(&out1, bytes));
    uint32_t *input = malloc(bytes);
    uint32_t *output = malloc(bytes);
    CHECK(input && output);
    if (!input || !output) {
        free(input);
        free(output);
        free(ptx);
        return failures != 0;
    }
    for (uint32_t i = 0; i < N; i++)
        input[i] = 0x10000000u + i * 3u;
    CHECK_CU(cuMemcpyHtoD(in, input, bytes));

    void *args[4];
    uint32_t n = N;
    uint32_t inc = 0;

    // Warm up the pipeline and make the steady descriptor entry.
    CHECK_CU(cuMemsetD32(out0, 0, N));
    CHECK_CU(launch(function, out0, in, n, 1, NULL, args));
    CHECK_CU(cuCtxSynchronize());

    // Repeated identical launches: the command buffer should be reused
    // after the signature has been seen twice.
    for (uint32_t i = 0; i < 32; i++) {
        CHECK_CU(launch(function, out0, in, n, 1, NULL, args));
        CHECK_CU(cuCtxSynchronize());
        CHECK_CU(cuMemcpyDtoH(output, out0, bytes));
        verify("steady", input, output, 1);
    }

    // Changing push constants: the cache must not serve a stale command.
    for (uint32_t i = 1; i <= 32; i++) {
        CHECK_CU(launch(function, out0, in, n, i, NULL, args));
        CHECK_CU(cuCtxSynchronize());
        CHECK_CU(cuMemcpyDtoH(output, out0, bytes));
        verify("changing-push", input, output, i);
    }

    // Changing output buffers: descriptor and command cache invalidation.
    for (uint32_t i = 0; i < 32; i++) {
        CUdeviceptr out = (i & 1u) ? out1 : out0;
        uint32_t value = 100u + i;
        CHECK_CU(cuMemsetD32(out, 0, N));
        CHECK_CU(launch(function, out, in, n, value, NULL, args));
        CHECK_CU(cuCtxSynchronize());
        CHECK_CU(cuMemcpyDtoH(output, out, bytes));
        verify("changing-buffer", input, output, value);
    }

    // Enqueued launches: stream order must hold even when the command
    // cache is not reusable because the stream is busy.
    CHECK_CU(cuMemsetD32(out0, 0, N));
    for (uint32_t i = 0; i < 64; i++)
        CHECK_CU(launch(function, out0, in, n, 9, NULL, args));
    CHECK_CU(cuCtxSynchronize());
    CHECK_CU(cuMemcpyDtoH(output, out0, bytes));
    verify("enqueued", input, output, 9);

    // Stream/event completion on a non-default stream.
    CUstream stream = NULL;
    CUevent event = NULL;
    CHECK_CU(cuStreamCreate(&stream, 0));
    CHECK_CU(cuEventCreate(&event, 0));
    CHECK_CU(cuMemsetD32(out1, 0, N));
    CHECK_CU(launch(function, out1, in, n, 77, stream, args));
    CHECK_CU(cuEventRecord(event, stream));
    CHECK_CU(cuEventSynchronize(event));
    CHECK_CU(cuMemcpyDtoH(output, out1, bytes));
    verify("stream-event", input, output, 77);
    CHECK_CU(cuStreamSynchronize(stream));
    CHECK_CU(cuEventDestroy(event));
    CHECK_CU(cuStreamDestroy(stream));

    // Signed packed saturating byte subtraction regression. Inputs:
    //   [4] = 0x80fffff0, [5] = 0x01020320
    // Expected: 0x80fdfcd0 (lane 3 must saturate -128 - 1 to -128).
    CUdeviceptr sat = 0;
    CHECK_CU(cuMemAlloc(&sat, 64));
    uint32_t satin[16] = {0};
    satin[4] = 0x80fffff0u;
    satin[5] = 0x01020320u;
    CHECK_CU(cuMemcpyHtoD(sat, satin, sizeof(satin)));
    void *sat_args[1] = { &sat };
    CHECK_CU(cuLaunchKernel(sat_function, 1, 1, 1, 1, 1, 1, 0, NULL,
                            sat_args, NULL));
    CHECK_CU(cuCtxSynchronize());
    uint32_t satout[16] = {0};
    CHECK_CU(cuMemcpyDtoH(satout, sat, sizeof(satout)));
    CHECK(satout[0] == 0x80fdfcd0u);

    // A 536-byte by-value struct exceeds the usual 256-byte push-constant
    // limit; ACuda must pass it through a descriptor-bound storage buffer.
    {
        char bigpath[1024];
        snprintf(bigpath, sizeof(bigpath), "%s/large_byval_param.ptx",
                 ACUDA_TEST_PTX_DIR);
        char *bigptx = read_text(bigpath);
        CHECK(bigptx != NULL);
        if (bigptx) {
            CUmodule bigmod = NULL;
            CUfunction bigfn = NULL;
            CUdeviceptr bigout = 0;
            CHECK_CU(cuModuleLoadDataEx(&bigmod, bigptx, 0, NULL, NULL));
            CHECK_CU(cuModuleGetFunction(&bigfn, bigmod,
                                         "large_byval_param"));
            CHECK_CU(cuMemAlloc(&bigout, 4));
            uint32_t zero = 0;
            CHECK_CU(cuMemcpyHtoD(bigout, &zero, 4));
            struct { uint64_t w[67]; } big;
            void *bigargs[2] = { &big, &bigout };
            for (uint32_t iter = 0; iter < 2; iter++) {
                memset(&big, 0, sizeof(big));
                big.w[0] = 7 + iter;
                big.w[66] = 35 - iter;
                CHECK_CU(cuLaunchKernel(bigfn, 1, 1, 1, 1, 1, 1, 0, NULL,
                                        bigargs, NULL));
                CHECK_CU(cuCtxSynchronize());
                uint32_t got = 0;
                CHECK_CU(cuMemcpyDtoH(&got, bigout, 4));
                CHECK(got == 42);
            }
            CHECK_CU(cuMemFree(bigout));
            if (bigmod)
                CHECK_CU(cuModuleUnload(bigmod));
            free(bigptx);
        }
    }

    free(input);
    free(output);
    CHECK_CU(cuMemFree(sat));
    CHECK_CU(cuMemFree(out1));
    CHECK_CU(cuMemFree(out0));
    CHECK_CU(cuMemFree(in));
    CHECK_CU(cuModuleUnload(module));
    CHECK_CU(cuCtxDestroy(context));
    free(ptx);

    if (failures == 0)
        printf("Windows launch-cache smoke test passed\n");
    return failures != 0;
}
