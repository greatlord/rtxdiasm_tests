// Phase 5c: exercise libcudart directly (linked, no dlopen): props,
// malloc/memcpy/launch via fatbin registration, streams/events, pools.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "acuda/runtime_api.h"

#ifndef ACUDA_TEST_PTX_DIR
#define ACUDA_TEST_PTX_DIR "tests/ptx"
#endif

static int failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        failures++; \
    } \
} while (0)
#define CHECK_CU(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        printf("FAIL %s:%d: %s -> %d (%s)\n", __FILE__, __LINE__, #call, \
               (int)_e, cudaGetErrorString(_e)); \
        failures++; \
    } \
} while (0)

#define N (1u << 20)

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
    if (!b || fread(b, 1, (size_t)n, f) != (size_t)n) {
        fclose(f);
        free(b);
        return NULL;
    }
    b[n] = '\0';
    fclose(f);
    return b;
}

// Fake host-side function keys (registration maps these to entries).
static int k_f32, k_u32, k_tex, k_surf, k_surfdyn;

int main(void) {
    int ndev = 0;
    CHECK_CU(cudaGetDeviceCount(&ndev));
    CHECK(ndev >= 1);
    struct cudaDeviceProp prop;
    CHECK_CU(cudaGetDeviceProperties(&prop, 0));
    printf("  device: %s cc=%d.%d mem=%zu\n", prop.name, prop.major,
           prop.minor, prop.totalGlobalMem);
    CHECK(prop.major == 8 && (prop.minor == 0 || prop.minor == 6));
    CHECK(prop.maxThreadsPerBlock == 256);
    CHECK(prop.warpSize == 32);
    int rt = 0, drv = 0;
    CHECK_CU(cudaRuntimeGetVersion(&rt));
    CHECK_CU(cudaDriverGetVersion(&drv));
    printf("  runtime %d driver %d\n", rt, drv);
    // Pinned CUDA 13.4 pair (see docs/ptx-baseline.md): the runtime
    // reports its own build version, the driver delegates to libcuda.
    CHECK(rt == 13040);
    CHECK(drv == 13040);
    int dev = -1;
    CHECK_CU(cudaSetDevice(0));
    CHECK_CU(cudaGetDevice(&dev));
    CHECK(dev == 0);
    CHECK(strcmp(cudaGetErrorName(cudaSuccess), "cudaSuccess") == 0);

    char path[1024];
    snprintf(path, sizeof(path), "%s/vecadd.ptx", ACUDA_TEST_PTX_DIR);
    char *ptx = read_file(path);
    if (!ptx)
        return 1;

    // Fatbin registration with raw PTX (documented test path).
    void **fat = __cudaRegisterFatBinary(ptx);
    CHECK(fat != NULL);
    __cudaRegisterFunction(fat, (const char *)&k_f32, NULL, "vecadd_f32", -1,
                           NULL, NULL, NULL, NULL, NULL);
    __cudaRegisterFunction(fat, (const char *)&k_u32, NULL, "vecadd_u32", -1,
                           NULL, NULL, NULL, NULL, NULL);

    float *ha = malloc((size_t)N * 4);
    float *hb = malloc((size_t)N * 4);
    float *hc = malloc((size_t)N * 4);
    uint32_t *ia = (uint32_t *)ha, *ib = (uint32_t *)hb, *ic = (uint32_t *)hc;
    CHECK(ha && hb && hc);
    for (uint32_t i = 0; i < N; i++) {
        ha[i] = (float)i * 0.5f;
        hb[i] = (float)i * 2.0f + 1.0f;
    }

    void *da = NULL, *db = NULL, *dc = NULL;
    CHECK_CU(cudaMalloc(&da, (size_t)N * 4));
    CHECK_CU(cudaMalloc(&db, (size_t)N * 4));
    CHECK_CU(cudaMalloc(&dc, (size_t)N * 4));
    size_t free0 = 0, total0 = 0;
    CHECK_CU(cudaMemGetInfo(&free0, &total0));
    CHECK(total0 > 0 && free0 < total0);
    CHECK_CU(cudaMemcpy(da, ha, (size_t)N * 4, cudaMemcpyHostToDevice));
    CHECK_CU(cudaMemcpy(db, hb, (size_t)N * 4, cudaMemcpyHostToDevice));

    cudaStream_t stream = NULL;
    CHECK_CU(cudaStreamCreate(&stream));
    CHECK(stream != NULL);
    uint32_t n = N;
    void *args[4] = { &da, &db, &dc, &n };
    dim3 grid = { N / 256, 1, 1 }, block = { 256, 1, 1 };
    CHECK_CU(cudaLaunchKernel(&k_f32, grid, block, args, 0, stream));
    CHECK_CU(cudaStreamSynchronize(stream));
    CHECK_CU(cudaMemcpy(hc, dc, (size_t)N * 4, cudaMemcpyDeviceToHost));
    uint32_t bad = 0;
    for (uint32_t i = 0; i < N; i++) {
        float want = ha[i] * 2.0f + hb[i];
        if (hc[i] != want) {
            if (bad < 5)
                printf("  f32 mismatch[%u]: got %f want %f\n", i, hc[i], want);
            bad++;
        }
    }
    CHECK(bad == 0);
    printf("  cudart vecadd_f32: %u elements ok\n", N);

    // Async pool path + events.
    void *dd = NULL;
    CHECK_CU(cudaMallocAsync(&dd, 64, stream));
    CHECK_CU(cudaMemsetAsync(dd, 0xAB, 64, stream));
    cudaEvent_t ev = NULL;
    CHECK_CU(cudaEventCreate(&ev));
    CHECK_CU(cudaEventRecord(ev, stream));
    CHECK_CU(cudaEventSynchronize(ev));
    CHECK(cudaEventQuery(ev) == cudaSuccess);
    CHECK_CU(cudaEventDestroy(ev));
    CHECK_CU(cudaFreeAsync(dd, stream));

    // Func attributes + occupancy estimate.
    struct cudaFuncAttributes attr;
    CHECK_CU(cudaFuncGetAttributes(&attr, &k_f32));
    CHECK(attr.maxThreadsPerBlock == 256);
    int blocks = 0;
    CHECK_CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, &k_f32,
                                                          256, 0));
    CHECK(blocks > 0);
    struct cudaPointerAttributes pa;
    CHECK_CU(cudaPointerGetAttributes(&pa, dc));
    CHECK(pa.type == cudaMemoryTypeDevice);

    // u32 kernel through the default stream.
    for (uint32_t i = 0; i < N; i++) {
        ia[i] = i;
        ib[i] = 2 * i + 1;
    }
    CHECK_CU(cudaMemcpyAsync(da, ha, (size_t)N * 4, cudaMemcpyHostToDevice,
                             NULL));
    CHECK_CU(cudaMemcpyAsync(db, hb, (size_t)N * 4, cudaMemcpyHostToDevice,
                             NULL));
    CHECK_CU(cudaLaunchKernel(&k_u32, grid, block, args, 0, NULL));
    CHECK_CU(cudaDeviceSynchronize());
    CHECK_CU(cudaMemcpyAsync(hc, dc, (size_t)N * 4, cudaMemcpyDeviceToHost,
                             NULL));
    bad = 0;
    for (uint32_t i = 0; i < N; i++) {
        if (ic[i] != ia[i] + ib[i]) {
            if (bad < 5)
                printf("  u32 mismatch[%u]\n", i);
            bad++;
        }
    }
    CHECK(bad == 0);
    printf("  cudart vecadd_u32: %u elements ok\n", N);

    // Texture array + texture object through the runtime API.
    {
        char tpath[1024];
        snprintf(tpath, sizeof(tpath), "%s/tex2d.ptx", ACUDA_TEST_PTX_DIR);
        char *tptx = read_file(tpath);
        CHECK(tptx != NULL);
        if (tptx) {
            void **tfat = __cudaRegisterFatBinary(tptx);
            CHECK(tfat != NULL);
            __cudaRegisterFunction(tfat, (const char *)&k_tex, NULL, "tex2d",
                                   -1, NULL, NULL, NULL, NULL, NULL);
            cudaChannelFormatDesc cd = { 32, 32, 32, 32,
                                         cudaChannelFormatKindFloat };
            cudaArray_t arr = NULL;
            CHECK_CU(cudaMallocArray(&arr, &cd, 2, 2, 0));
            float texels[16] = { 1, 2, 3, 4, 5, 6, 7, 8,
                                 9, 10, 11, 12, 13, 14, 15, 16 };
            CHECK_CU(cudaMemcpy2DToArray(arr, 0, 0, texels, 32, 32, 2,
                                         cudaMemcpyHostToDevice));
            float back[16];
            memset(back, 0, sizeof(back));
            CHECK_CU(cudaMemcpy2DFromArray(back, 32, arr, 0, 0, 32, 2,
                                           cudaMemcpyDeviceToHost));
            int rbad = 0;
            for (int i = 0; i < 16; i++)
                if (back[i] != texels[i])
                    rbad++;
            CHECK(rbad == 0);
            cudaResourceDesc rd;
            memset(&rd, 0, sizeof(rd));
            rd.resType = cudaResourceTypeArray;
            rd.res.array.array = arr;
            cudaTextureDesc td;
            memset(&td, 0, sizeof(td));
            td.addressMode[0] = cudaAddressModeClamp;
            td.addressMode[1] = cudaAddressModeClamp;
            td.addressMode[2] = cudaAddressModeClamp;
            td.filterMode = cudaFilterModePoint;
            td.mipmapFilterMode = cudaFilterModePoint;
            td.normalizedCoords = 1;
            cudaTextureObject_t tobj = 0;
            CHECK_CU(cudaCreateTextureObject(&tobj, &rd, &td, NULL));
            cudaTextureDesc tget;
            CHECK_CU(cudaGetTextureObjectTextureDesc(&tget, tobj));
            CHECK(tget.filterMode == cudaFilterModePoint);
            void *dout = NULL;
            CHECK_CU(cudaMalloc(&dout, 64));
            void *targs[2] = { &tobj, &dout };
            dim3 tg = { 1, 1, 1 }, tb = { 1, 1, 1 };
            CHECK_CU(cudaLaunchKernel(&k_tex, tg, tb, targs, 0, NULL));
            CHECK_CU(cudaDeviceSynchronize());
            uint32_t got[7] = {0};
            CHECK_CU(cudaMemcpy(got, dout, 28, cudaMemcpyDeviceToHost));
            static const uint32_t want[7] = {
                0x3f800000u, 0x40000000u, 0x40400000u, 0x40800000u,
                2, 2, 1
            };
            int tbad = 0;
            for (int i = 0; i < 7; i++)
                if (got[i] != want[i])
                    tbad++;
            CHECK(tbad == 0);
            if (!tbad)
                printf("  cudart texture object: 7 words ok\n");
            CHECK_CU(cudaDestroyTextureObject(tobj));

            cudaSurfaceObject_t sobj = 0;
            CHECK_CU(cudaCreateSurfaceObject(&sobj, &rd));
            cudaResourceDesc srd;
            memset(&srd, 0, sizeof(srd));
            CHECK_CU(cudaGetSurfaceObjectResourceDesc(&srd, sobj));
            CHECK(srd.resType == cudaResourceTypeArray);
            CHECK(srd.res.array.array == arr);
            CHECK_CU(cudaDestroySurfaceObject(sobj));
            printf("  cudart surface object: array round-trip ok\n");
            CHECK_CU(cudaFree(dout));
            CHECK_CU(cudaFreeArray(arr));
            __cudaUnregisterFatBinary(tfat);
            free(tptx);
        }
    }

    // Modern cudaSurfaceObject_t kernel: .u64 handle used by surf ops.
    {
        char spath[1024];
        snprintf(spath, sizeof(spath), "%s/surf2d_u64.ptx",
                 ACUDA_TEST_PTX_DIR);
        char *sptx = read_file(spath);
        CHECK(sptx != NULL);
        if (sptx) {
            void **sfat = __cudaRegisterFatBinary(sptx);
            CHECK(sfat != NULL);
            __cudaRegisterFunction(sfat, (const char *)&k_surf, NULL,
                                   "surf2d_u64", -1, NULL, NULL, NULL,
                                   NULL, NULL);
            cudaChannelFormatDesc cd = { 32, 0, 0, 0,
                                         cudaChannelFormatKindUnsigned };
            cudaArray_t arr = NULL;
            CHECK_CU(cudaMallocArray(&arr, &cd, 4, 2, 0));
            uint32_t sdata[8] = { 10, 20, 30, 40, 50, 60, 70, 80 };
            CHECK_CU(cudaMemcpy2DToArray(arr, 0, 0, sdata, 16, 16, 2,
                                         cudaMemcpyHostToDevice));
            cudaResourceDesc rd;
            memset(&rd, 0, sizeof(rd));
            rd.resType = cudaResourceTypeArray;
            rd.res.array.array = arr;
            cudaSurfaceObject_t sobj = 0;
            CHECK_CU(cudaCreateSurfaceObject(&sobj, &rd));
            void *sout = NULL;
            CHECK_CU(cudaMalloc(&sout, 16));
            void *sargs[2] = { &sobj, &sout };
            dim3 sg = { 1, 1, 1 }, sb = { 1, 1, 1 };
            CHECK_CU(cudaLaunchKernel(&k_surf, sg, sb, sargs, 0, NULL));
            CHECK_CU(cudaDeviceSynchronize());
            uint32_t sgot[4] = {0};
            CHECK_CU(cudaMemcpy(sgot, sout, 12, cudaMemcpyDeviceToHost));
            CHECK(sgot[0] == 20 && sgot[1] == 4 && sgot[2] == 2);
            uint32_t sback[8] = {0};
            CHECK_CU(cudaMemcpy2DFromArray(sback, 16, arr, 0, 0, 16, 2,
                                           cudaMemcpyDeviceToHost));
            CHECK(sback[0] == 10 && sback[1] == 20 && sback[2] == 999 &&
                  sback[3] == 45);
            printf("  cudart surface-object launch: 3 words + writes ok\n");
            CHECK_CU(cudaDestroySurfaceObject(sobj));
            CHECK_CU(cudaFree(sout));
            CHECK_CU(cudaFreeArray(arr));
            __cudaUnregisterFatBinary(sfat);
            free(sptx);
        }
    }

    // CUDART runtime surface selection through the dynamic table.
    {
        char dpath[1024];
        snprintf(dpath, sizeof(dpath), "%s/surf2d_dyn.ptx",
                 ACUDA_TEST_PTX_DIR);
        char *dptx = read_file(dpath);
        CHECK(dptx != NULL);
        if (dptx) {
            void **dfat = __cudaRegisterFatBinary(dptx);
            CHECK(dfat != NULL);
            __cudaRegisterFunction(dfat, (const char *)&k_surfdyn, NULL,
                                   "surf2d_dyn", -1, NULL, NULL, NULL,
                                   NULL, NULL);
            cudaChannelFormatDesc cd = { 32, 0, 0, 0,
                                         cudaChannelFormatKindUnsigned };
            cudaArray_t a = NULL, b = NULL;
            CHECK_CU(cudaMallocArray(&a, &cd, 4, 2, 0));
            CHECK_CU(cudaMallocArray(&b, &cd, 4, 2, 0));
            uint32_t adata[8] = { 10, 20, 30, 40, 50, 60, 70, 80 };
            uint32_t bdata[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
            CHECK_CU(cudaMemcpy2DToArray(a, 0, 0, adata, 16, 16, 2,
                                         cudaMemcpyHostToDevice));
            CHECK_CU(cudaMemcpy2DToArray(b, 0, 0, bdata, 16, 16, 2,
                                         cudaMemcpyHostToDevice));
            cudaResourceDesc rd;
            memset(&rd, 0, sizeof(rd));
            rd.resType = cudaResourceTypeArray;
            rd.res.array.array = a;
            cudaSurfaceObject_t sa = 0, sb = 0;
            CHECK_CU(cudaCreateSurfaceObject(&sa, &rd));
            rd.res.array.array = b;
            CHECK_CU(cudaCreateSurfaceObject(&sb, &rd));
            void *hbuf = NULL, *dout = NULL;
            CHECK_CU(cudaMalloc(&hbuf, 8));
            CHECK_CU(cudaMalloc(&dout, 8));
            void *dargs[2] = { &hbuf, &dout };
            dim3 dg = { 1, 1, 1 }, db = { 1, 1, 1 };
            CHECK_CU(cudaMemcpy(hbuf, &sa, 8, cudaMemcpyHostToDevice));
            CHECK_CU(cudaLaunchKernel(&k_surfdyn, dg, db, dargs, 0, NULL));
            CHECK_CU(cudaDeviceSynchronize());
            uint32_t dgot[2] = {0};
            CHECK_CU(cudaMemcpy(dgot, dout, 8, cudaMemcpyDeviceToHost));
            CHECK(dgot[0] == 20 && dgot[1] == 4);
            uint32_t aback[8] = {0};
            CHECK_CU(cudaMemcpy2DFromArray(aback, 16, a, 0, 0, 16, 2,
                                           cudaMemcpyDeviceToHost));
            CHECK(aback[1] == 20 && aback[2] == 777);
            CHECK_CU(cudaMemcpy(hbuf, &sb, 8, cudaMemcpyHostToDevice));
            CHECK_CU(cudaLaunchKernel(&k_surfdyn, dg, db, dargs, 0, NULL));
            CHECK_CU(cudaDeviceSynchronize());
            CHECK_CU(cudaMemcpy(dgot, dout, 8, cudaMemcpyDeviceToHost));
            CHECK(dgot[0] == 2 && dgot[1] == 4);
            uint32_t bback[8] = {0};
            CHECK_CU(cudaMemcpy2DFromArray(bback, 16, b, 0, 0, 16, 2,
                                           cudaMemcpyDeviceToHost));
            CHECK(bback[1] == 2 && bback[2] == 777);
            printf("  cudart dynamic surface selection: 2 objects ok\n");
            CHECK_CU(cudaDestroySurfaceObject(sa));
            CHECK_CU(cudaDestroySurfaceObject(sb));
            CHECK_CU(cudaFree(hbuf));
            CHECK_CU(cudaFree(dout));
            CHECK_CU(cudaFreeArray(a));
            CHECK_CU(cudaFreeArray(b));
            __cudaUnregisterFatBinary(dfat);
            free(dptx);
        }
    }

    // Error paths.
    CHECK(cudaMalloc(NULL, 0) == cudaErrorInvalidValue);
    CHECK(cudaLaunchKernel(NULL, grid, block, args, 0, NULL) ==
          cudaErrorInvalidValue);
    CHECK(cudaLaunchKernel(&k_f32, grid, block, NULL, 0, NULL) ==
          cudaErrorInvalidValue);
    free(ha);
    free(hb);
    free(hc);
    CHECK_CU(cudaStreamDestroy(stream));
    CHECK_CU(cudaFree(da));
    CHECK_CU(cudaFree(db));
    CHECK_CU(cudaFree(dc));
    __cudaUnregisterFatBinary(fat);
    free(ptx);

    if (failures == 0)
        printf("cudart integration test passed\n");
    return failures != 0;
}
