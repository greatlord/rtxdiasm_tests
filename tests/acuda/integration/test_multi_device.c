// Multi-device driver integration test.
//
// Runs a real PTX kernel on every CUDA device ACuda exposes.  With
// ACUDA_MULTI_DEVICE=1 this exercises NVIDIA + AMD iGPU contexts and their
// separate arenas.  On a single-GPU machine it still validates the normal
// device-0 path.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "acuda/driver.h"

#if defined(_WIN32)
#define TEST_SETENV(name, value) _putenv_s((name), (value))
#else
#define TEST_SETENV(name, value) setenv((name), (value), 1)
#endif

static const char *test_ptx =
    ".version 6.5\n"
    ".target sm_30\n"
    ".address_size 64\n"
    ".visible .entry write_one(\n"
    "    .param .u64 out,\n"
    "    .param .u32 value)\n"
    "{\n"
    "    .reg .b64 %rd<2>;\n"
    "    .reg .b32 %r<2>;\n"
    "    ld.param.u64 %rd1, [out];\n"
    "    ld.param.u32 %r1, [value];\n"
    "    st.global.u32 [%rd1], %r1;\n"
    "    ret;\n"
    "}\n";

static int run_device(int dev) {
    char name[256] = {0};
    CUresult r = cuDeviceGetName(name, (int)sizeof(name), dev);
    printf("[%d] %s\n", dev, name);
    if (r != CUDA_SUCCESS) {
        printf("FAIL: cuDeviceGetName -> %d\n", (int)r);
        return 1;
    }

    CUcontext ctx = NULL;
    r = cuCtxCreate_v2(&ctx, 0, dev);
    if (r != CUDA_SUCCESS) {
        printf("FAIL: cuCtxCreate_v2 -> %d\n", (int)r);
        return 1;
    }

    int failures = 0;
    CUdeviceptr dptr = 0;
    CUmodule mod = NULL;
    CUfunction fn = NULL;
    uint32_t out = 0xDEADBEEFu;

    r = cuMemAlloc_v2(&dptr, 16);
    if (r != CUDA_SUCCESS) {
        printf("FAIL: cuMemAlloc_v2 -> %d\n", (int)r);
        failures++;
        goto cleanup;
    }

    uint32_t seed = 0x11110000u + (uint32_t)dev;
    r = cuMemcpyHtoD_v2(dptr, &seed, sizeof(seed));
    if (r != CUDA_SUCCESS) {
        printf("FAIL: cuMemcpyHtoD_v2 -> %d\n", (int)r);
        failures++;
        goto cleanup;
    }

    r = cuModuleLoadData(&mod, test_ptx);
    if (r != CUDA_SUCCESS) {
        printf("FAIL: cuModuleLoadData -> %d\n", (int)r);
        failures++;
        goto cleanup;
    }

    r = cuModuleGetFunction(&fn, mod, "write_one");
    if (r != CUDA_SUCCESS) {
        printf("FAIL: cuModuleGetFunction -> %d\n", (int)r);
        failures++;
        goto cleanup;
    }

    uint32_t value = 0xA5A50000u + (uint32_t)dev;
    void *args[2] = {&dptr, &value};
    r = cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, NULL, args, NULL);
    if (r != CUDA_SUCCESS) {
        printf("FAIL: cuLaunchKernel -> %d\n", (int)r);
        failures++;
        goto cleanup;
    }

    r = cuCtxSynchronize();
    if (r != CUDA_SUCCESS) {
        printf("FAIL: cuCtxSynchronize -> %d\n", (int)r);
        failures++;
        goto cleanup;
    }

    r = cuMemcpyDtoH_v2(&out, dptr, sizeof(out));
    if (r != CUDA_SUCCESS) {
        printf("FAIL: cuMemcpyDtoH_v2 -> %d\n", (int)r);
        failures++;
        goto cleanup;
    }

    if (out != value) {
        printf("FAIL: got 0x%08x expected 0x%08x\n", out, value);
        failures++;
        goto cleanup;
    }
    printf("  PASS out=0x%08x\n", out);

cleanup:
    if (mod)
        cuModuleUnload(mod);
    if (dptr)
        cuMemFree_v2(dptr);
    cuCtxDestroy_v2(ctx);
    return failures ? 1 : 0;
}

int main(void) {
    // Do not assume a device count: the driver auto-detects every
    // supported Vulkan device, and this test runs whatever it finds.
    TEST_SETENV("ACUDA_NO_VALIDATION", "1");

    CUresult r = cuInit(0);
    if (r != CUDA_SUCCESS) {
        printf("FAIL: cuInit -> %d\n", (int)r);
        return 1;
    }

    int n = 0;
    r = cuDeviceGetCount(&n);
    if (r != CUDA_SUCCESS || n < 1) {
        printf("FAIL: cuDeviceGetCount -> %d, count=%d\n", (int)r, n);
        return 1;
    }
    printf("CUDA devices: %d\n", n);
    if (n == 1)
        printf("NOTE: only one CUDA device available; single-device path\n");
    else
        printf("NOTE: running on every exposed CUDA device\n");

    int failures = 0;
    for (int dev = 0; dev < n; dev++)
        failures += run_device(dev);

    if (failures) {
        printf("FAIL: %d device(s) failed\n", failures);
        return 1;
    }
    printf("ALL DEVICES PASS\n");
    return 0;
}
