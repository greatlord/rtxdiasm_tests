// Standalone reproducer: launch PhysX's lifted KernelFluidCalcHashPacked
// (the batch that never signals in FluidMark) with fabricated buffers.
// If cuCtxSynchronize returns, the kernel completes and the hang is
// environment-specific; if it times out/hangs, the shader (or its bind)
// hangs on its own.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "acuda/driver.h"

static int failures = 0;
#define CHECK_CU(call) do { \
    CUresult _r = (call); \
    if (_r != CUDA_SUCCESS) { \
        printf("FAIL %s:%d: %s -> %d\n", __FILE__, __LINE__, #call, \
               (int)_r); \
        failures++; \
    } \
} while (0)

static char *read_file(const char *path, size_t *out_n) {
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
    if (!b || fread(b, 1, (size_t)n, f) != (size_t)n) {
        fclose(f);
        free(b);
        return NULL;
    }
    b[n] = '\0';
    fclose(f);
    *out_n = (size_t)n;
    return b;
}

int main(int argc, char **argv) {
    const char *cubin_path =
        (argc > 1) ? argv[1]
                   : "C:/Users/Magnus/AppData/Local/Temp/opencode/calchash_fatbin.bin";
    const char *ptx_path = (argc > 2) ? argv[2] : NULL;
    const char *entry_name =
        (argc > 3) ? argv[3]
                   : "_Z26KernelFluidCalcHashPackedD17PxgFixedSizeArrayIjES0_S_IK12"
                     "PxcVectorPadE11PxgGpuArrayIKjES6_S_IK18PxgFluidParametersEj";
    CHECK_CU(cuInit(0));
    CUdevice dev;
    CHECK_CU(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK_CU(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK_CU(cuCtxSetCurrent(ctx));
    if (failures)
        return 1;

    CUmodule mod = NULL;
    if (ptx_path) {
        size_t n = 0;
        char *ptx = read_file(ptx_path, &n);
        if (!ptx)
            return 1;
        CHECK_CU(cuModuleLoadData(&mod, ptx));
        free(ptx);
    } else {
        size_t n = 0;
        char *blob = read_file(cubin_path, &n);
        if (!blob)
            return 1;
        CHECK_CU(cuModuleLoadData(&mod, blob));
        free(blob);
    }
    if (failures)
        return 1;

    CUfunction func = NULL;
    CHECK_CU(cuModuleGetFunction(&func, mod, entry_name));
    if (failures)
        return 1;
    printf("function ready\n");

    // 6 pointer params + 1 u32 count; 256 threads touch at most a few KB.
    enum { NBUF = 6, ELEMS = 1024 };
    CUdeviceptr d[NBUF];
    for (int i = 0; i < NBUF; i++) {
        CHECK_CU(cuMemAlloc(&d[i], (size_t)ELEMS * 16));
    }
    if (failures)
        return 1;
    // positions-ish buffer (p3): finite float4 grid values.
    float *hp = malloc((size_t)ELEMS * 16);
    for (int i = 0; i < ELEMS; i++) {
        hp[i * 4 + 0] = (float)(i % 16) * 0.5f;
        hp[i * 4 + 1] = (float)((i / 16) % 16) * 0.5f;
        hp[i * 4 + 2] = (float)(i / 256) * 0.5f;
        hp[i * 4 + 3] = 1.0f;
    }
    CHECK_CU(cuMemcpyHtoD(d[3], hp, (size_t)ELEMS * 16));
    free(hp);
    if (failures)
        return 1;

    uint32_t count = 256;
    void *args[] = { &d[0], &d[1], &d[2], &d[3],
                     &d[4], &d[5], &count };
    printf("launching grid=4x64...\n");
    fflush(stdout);
    CHECK_CU(cuLaunchKernel(func, 4, 1, 1, 64, 1, 1, 0, NULL, args, NULL));
    if (failures)
        return 1;
    printf("launched, synchronizing...\n");
    fflush(stdout);
    CHECK_CU(cuCtxSynchronize());
    printf("synchronized: kernel COMPLETED\n");
    // Sanity: the hash output (p1 buffer) should hold small cell
    // indices, not garbage.
    {
        uint32_t *ho = malloc((size_t)ELEMS * 4);
        if (ho) {
            CHECK_CU(cuMemcpyDtoH(ho, d[1], (size_t)ELEMS * 4));
            unsigned bad = 0;
            for (int i = 0; i < 256; i++)
                if (ho[i] > 1000000u)
                    bad++;
            printf("hash sample: %u %u %u %u (bad=%u/256)\n",
                   ho[0], ho[1], ho[2], ho[3], bad);
            if (bad > 200)
                failures++;
            free(ho);
        }
    }
    for (int i = 0; i < NBUF; i++)
        cuMemFree(d[i]);
    if (failures)
        return 1;
    printf("PASS\n");
    return 0;
}
