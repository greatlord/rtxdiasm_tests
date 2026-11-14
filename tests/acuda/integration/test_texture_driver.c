// CUDA driver texture surface: arrays, 2D/3D copies, mipmapped arrays,
// texture objects and launching a kernel with a CUtexObject argument.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "acuda/driver.h"

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
    CUresult _r = (call); \
    if (_r != CUDA_SUCCESS) { \
        printf("FAIL %s:%d: %s -> %d\n", __FILE__, __LINE__, #call, \
               (int)_r); \
        failures++; \
    } \
} while (0)

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
    b[n] = 0;
    fclose(f);
    return b;
}

static void fill_2d_texels(float *t, unsigned n) {
    for (unsigned i = 0; i < n; i++)
        t[i] = (float)(i + 1);
}

int main(void) {
    CHECK_CU(cuInit(0));
    CUdevice dev;
    CHECK_CU(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK_CU(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK_CU(cuCtxSetCurrent(ctx));

    char path[1024];
    snprintf(path, sizeof(path), "%s/tex2d.ptx", ACUDA_TEST_PTX_DIR);
    char *ptx = read_file(path);
    CHECK(ptx != NULL);
    if (!ptx)
        return 1;
    CUmodule mod;
    CHECK_CU(cuModuleLoadData(&mod, ptx));
    CUfunction fn;
    CHECK_CU(cuModuleGetFunction(&fn, mod, "tex2d"));

    // 2D RGBA float array filled with 1..16.
    CUarray arr;
    CUDA_ARRAY_DESCRIPTOR ad = { 2, 2, CU_AD_FORMAT_FLOAT, 4 };
    CHECK_CU(cuArrayCreate(&arr, &ad));
    float texels[16];
    fill_2d_texels(texels, 16);
    CUDA_MEMCPY2D cp;
    memset(&cp, 0, sizeof(cp));
    cp.srcMemoryType = CU_MEMORYTYPE_HOST;
    cp.srcHost = texels;
    cp.srcPitch = 32;
    cp.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    cp.dstArray = arr;
    cp.WidthInBytes = 32;
    cp.Height = 2;
    CHECK_CU(cuMemcpy2D(&cp));

    // Read back through cuMemcpy2D (array -> host).
    float back[16];
    memset(back, 0, sizeof(back));
    CUDA_MEMCPY2D rb = cp;
    rb.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    rb.srcArray = arr;
    rb.dstMemoryType = CU_MEMORYTYPE_HOST;
    rb.dstHost = back;
    rb.dstPitch = 32;
    CHECK_CU(cuMemcpy2D(&rb));
    int rb_bad = 0;
    for (int i = 0; i < 16; i++)
        if (back[i] != texels[i])
            rb_bad++;
    CHECK(rb_bad == 0);
    if (!rb_bad)
        printf("  cuMemcpy2D array round-trip: 16 texels ok\n");

    CUDA_TEXTURE_DESC td;
    memset(&td, 0, sizeof(td));
    td.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
    td.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
    td.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
    td.filterMode = CU_TR_FILTER_MODE_POINT;
    td.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
    CUDA_RESOURCE_DESC rd;
    memset(&rd, 0, sizeof(rd));
    rd.resType = CU_RESOURCE_TYPE_ARRAY;
    rd.res.array.hArray = arr;
    CUtexObject tobj;
    CHECK_CU(cuTexObjectCreate(&tobj, &rd, &td, NULL));

    CUDA_TEXTURE_DESC tget;
    CHECK_CU(cuTexObjectGetTextureDesc(&tget, tobj));
    CHECK(tget.filterMode == CU_TR_FILTER_MODE_POINT);
    CUDA_RESOURCE_DESC rget;
    CHECK_CU(cuTexObjectGetResourceDesc(&rget, tobj));
    CHECK(rget.resType == CU_RESOURCE_TYPE_ARRAY);

    CUdeviceptr out;
    CHECK_CU(cuMemAlloc(&out, 64));
    void *args[2] = { &tobj, &out };
    CHECK_CU(cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, NULL, args, NULL));
    CHECK_CU(cuCtxSynchronize());
    uint32_t got[7] = {0};
    CHECK_CU(cuMemcpyDtoH(got, out, 28));
    static const uint32_t want[7] = {
        0x3f800000u, 0x40000000u, 0x40400000u, 0x40800000u, 2, 2, 1
    };
    int bad = 0;
    for (int i = 0; i < 7; i++)
        if (got[i] != want[i])
            bad++;
    CHECK(bad == 0);
    if (!bad)
        printf("  texture object launch: 7 words ok\n");

    // Surface object aliases the storage-capable array image.
    CUsurfObject sobj;
    CHECK_CU(cuSurfObjectCreate(&sobj, &rd));
    CUDA_RESOURCE_DESC sget;
    CHECK_CU(cuSurfObjectGetResourceDesc(&sget, sobj));
    CHECK(sget.resType == CU_RESOURCE_TYPE_ARRAY);
    CHECK(sget.res.array.hArray == arr);
    CHECK_CU(cuSurfObjectDestroy(sobj));
    printf("  cuSurfObjectCreate/Get/Destroy: array round-trip ok\n");

    // Modern cudaSurfaceObject_t-style PTX: the surface handle is a .u64
    // parameter loaded into a register.
    {
        char spath[1024];
        snprintf(spath, sizeof(spath), "%s/surf2d_u64.ptx",
                 ACUDA_TEST_PTX_DIR);
        char *sptx = read_file(spath);
        CHECK(sptx != NULL);
        if (sptx) {
            CUmodule smod;
            CHECK_CU(cuModuleLoadData(&smod, sptx));
            CUfunction sfn;
            CHECK_CU(cuModuleGetFunction(&sfn, smod, "surf2d_u64"));
            CUarray sarr;
            CUDA_ARRAY_DESCRIPTOR sad = { 4, 2, CU_AD_FORMAT_UNSIGNED_INT32,
                                          1 };
            CHECK_CU(cuArrayCreate(&sarr, &sad));
            uint32_t sdata[8] = { 10, 20, 30, 40, 50, 60, 70, 80 };
            CUDA_MEMCPY2D s2;
            memset(&s2, 0, sizeof(s2));
            s2.srcMemoryType = CU_MEMORYTYPE_HOST;
            s2.srcHost = sdata;
            s2.srcPitch = 16;
            s2.dstMemoryType = CU_MEMORYTYPE_ARRAY;
            s2.dstArray = sarr;
            s2.WidthInBytes = 16;
            s2.Height = 2;
            CHECK_CU(cuMemcpy2D(&s2));
            CUDA_RESOURCE_DESC srd;
            memset(&srd, 0, sizeof(srd));
            srd.resType = CU_RESOURCE_TYPE_ARRAY;
            srd.res.array.hArray = sarr;
            CUsurfObject su;
            CHECK_CU(cuSurfObjectCreate(&su, &srd));
            CUdeviceptr sout;
            CHECK_CU(cuMemAlloc(&sout, 16));
            void *sargs[2] = { &su, &sout };
            CHECK_CU(cuLaunchKernel(sfn, 1, 1, 1, 1, 1, 1, 0, NULL,
                                    sargs, NULL));
            CHECK_CU(cuCtxSynchronize());
            uint32_t sgot[4] = {0};
            CHECK_CU(cuMemcpyDtoH(sgot, sout, 12));
            CHECK(sgot[0] == 20 && sgot[1] == 4 && sgot[2] == 2);
            uint32_t sback[8] = {0};
            CUDA_MEMCPY2D srb = s2;
            srb.srcMemoryType = CU_MEMORYTYPE_ARRAY;
            srb.srcArray = sarr;
            srb.dstMemoryType = CU_MEMORYTYPE_HOST;
            srb.dstHost = sback;
            srb.dstPitch = 16;
            CHECK_CU(cuMemcpy2D(&srb));
            CHECK(sback[0] == 10 && sback[1] == 20 && sback[2] == 999 &&
                  sback[3] == 45);
            printf("  surf2d .u64 object launch: 3 words + writes ok\n");
            CHECK_CU(cuSurfObjectDestroy(su));
            CHECK_CU(cuMemFree(sout));
            CHECK_CU(cuArrayDestroy(sarr));
            CHECK_CU(cuModuleUnload(smod));
            free(sptx);
        }
    }

    // Runtime selection between surface objects: the handle is loaded
    // from device memory, so the compiler binds the dynamic table.
    {
        char dpath[1024];
        snprintf(dpath, sizeof(dpath), "%s/surf2d_dyn.ptx",
                 ACUDA_TEST_PTX_DIR);
        char *dptx = read_file(dpath);
        CHECK(dptx != NULL);
        if (dptx) {
            CUmodule dmod;
            CHECK_CU(cuModuleLoadData(&dmod, dptx));
            CUfunction dfn;
            CHECK_CU(cuModuleGetFunction(&dfn, dmod, "surf2d_dyn"));
            CUDA_ARRAY_DESCRIPTOR dad = { 4, 2, CU_AD_FORMAT_UNSIGNED_INT32,
                                          1 };
            CUarray da, db;
            CHECK_CU(cuArrayCreate(&da, &dad));
            CHECK_CU(cuArrayCreate(&db, &dad));
            uint32_t adata[8] = { 10, 20, 30, 40, 50, 60, 70, 80 };
            uint32_t bdata[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
            CUDA_MEMCPY2D cp2;
            memset(&cp2, 0, sizeof(cp2));
            cp2.srcMemoryType = CU_MEMORYTYPE_HOST;
            cp2.srcHost = adata;
            cp2.srcPitch = 16;
            cp2.dstMemoryType = CU_MEMORYTYPE_ARRAY;
            cp2.dstArray = da;
            cp2.WidthInBytes = 16;
            cp2.Height = 2;
            CHECK_CU(cuMemcpy2D(&cp2));
            cp2.srcHost = bdata;
            cp2.dstArray = db;
            CHECK_CU(cuMemcpy2D(&cp2));
            CUDA_RESOURCE_DESC dres;
            memset(&dres, 0, sizeof(dres));
            dres.resType = CU_RESOURCE_TYPE_ARRAY;
            dres.res.array.hArray = da;
            CUsurfObject sa, sb;
            CHECK_CU(cuSurfObjectCreate(&sa, &dres));
            dres.res.array.hArray = db;
            CHECK_CU(cuSurfObjectCreate(&sb, &dres));
            CUdeviceptr hbuf, dout;
            CHECK_CU(cuMemAlloc(&hbuf, 8));
            CHECK_CU(cuMemAlloc(&dout, 8));
            void *dargs[2] = { &hbuf, &dout };
            CHECK_CU(cuMemcpyHtoD(hbuf, &sa, 8));
            CHECK_CU(cuLaunchKernel(dfn, 1, 1, 1, 1, 1, 1, 0, NULL,
                                    dargs, NULL));
            CHECK_CU(cuCtxSynchronize());
            uint32_t dgot[2] = {0};
            CHECK_CU(cuMemcpyDtoH(dgot, dout, 8));
            CHECK(dgot[0] == 20 && dgot[1] == 4);
            uint32_t aback[8] = {0}, bback[8] = {0};
            CUDA_MEMCPY2D brb = cp2;
            brb.srcMemoryType = CU_MEMORYTYPE_ARRAY;
            brb.dstMemoryType = CU_MEMORYTYPE_HOST;
            brb.dstHost = aback;
            brb.dstPitch = 16;
            brb.srcArray = da;
            CHECK_CU(cuMemcpy2D(&brb));
            brb.dstHost = bback;
            brb.srcArray = db;
            CHECK_CU(cuMemcpy2D(&brb));
            CHECK(aback[1] == 20 && aback[2] == 777);
            CHECK(bback[1] == 2 && bback[2] == 3);
            CHECK_CU(cuMemcpyHtoD(hbuf, &sb, 8));
            CHECK_CU(cuLaunchKernel(dfn, 1, 1, 1, 1, 1, 1, 0, NULL,
                                    dargs, NULL));
            CHECK_CU(cuCtxSynchronize());
            CHECK_CU(cuMemcpyDtoH(dgot, dout, 8));
            CHECK(dgot[0] == 2 && dgot[1] == 4);
            brb.dstHost = bback;
            brb.srcArray = db;
            CHECK_CU(cuMemcpy2D(&brb));
            CHECK(bback[1] == 2 && bback[2] == 777);
            printf("  surf2d dynamic selection: 2 objects ok\n");

            // More than 1024 live handles exercise the runtime-sized
            // descriptor arrays (no fixed slot cap).
            enum { MANY = 1030 };
            CUsurfObject many[MANY];
            dres.res.array.hArray = da;
            for (int i = 0; i < MANY; i++)
                CHECK_CU(cuSurfObjectCreate(&many[i], &dres));
            CHECK_CU(cuMemcpyHtoD(hbuf, &many[MANY - 1], 8));
            CHECK_CU(cuLaunchKernel(dfn, 1, 1, 1, 1, 1, 1, 0, NULL,
                                    dargs, NULL));
            CHECK_CU(cuCtxSynchronize());
            CHECK_CU(cuMemcpyDtoH(dgot, dout, 8));
            CHECK(dgot[0] == 20 && dgot[1] == 4);
            for (int i = 0; i < MANY; i++)
                CHECK_CU(cuSurfObjectDestroy(many[i]));
            printf("  surf2d >1024 live handles: ok\n");

            CHECK_CU(cuSurfObjectDestroy(sa));
            CHECK_CU(cuSurfObjectDestroy(sb));
            CHECK_CU(cuMemFree(hbuf));
            CHECK_CU(cuMemFree(dout));
            CHECK_CU(cuArrayDestroy(da));
            CHECK_CU(cuArrayDestroy(db));
            CHECK_CU(cuModuleUnload(dmod));
            free(dptx);
        }
    }

    // Mixed dynamic surface types: 2D and 3D tables in one kernel.
    {
        char mpath[1024];
        snprintf(mpath, sizeof(mpath), "%s/surf_mixed_types.ptx",
                 ACUDA_TEST_PTX_DIR);
        char *mptx = read_file(mpath);
        CHECK(mptx != NULL);
        if (mptx) {
            CUmodule mmod;
            CHECK_CU(cuModuleLoadData(&mmod, mptx));
            CUfunction mfn;
            CHECK_CU(cuModuleGetFunction(&mfn, mmod, "surf_mixed_types"));
            CUarray m2;
            CUDA_ARRAY_DESCRIPTOR m2d = { 4, 2, CU_AD_FORMAT_UNSIGNED_INT32,
                                          1 };
            CHECK_CU(cuArrayCreate(&m2, &m2d));
            CUarray m3;
            CUDA_ARRAY3D_DESCRIPTOR m3d = { 2, 1, 2,
                                            CU_AD_FORMAT_UNSIGNED_INT32, 1,
                                            0 };
            CHECK_CU(cuArray3DCreate(&m3, &m3d));
            uint32_t m2data[8] = { 10, 20, 30, 40, 50, 60, 70, 80 };
            uint32_t m3data[4] = { 11, 22, 33, 44 };
            CUDA_MEMCPY2D mcp;
            memset(&mcp, 0, sizeof(mcp));
            mcp.srcMemoryType = CU_MEMORYTYPE_HOST;
            mcp.srcHost = m2data;
            mcp.srcPitch = 16;
            mcp.dstMemoryType = CU_MEMORYTYPE_ARRAY;
            mcp.dstArray = m2;
            mcp.WidthInBytes = 16;
            mcp.Height = 2;
            CHECK_CU(cuMemcpy2D(&mcp));
            CUDA_MEMCPY3D m3cp;
            memset(&m3cp, 0, sizeof(m3cp));
            m3cp.srcMemoryType = CU_MEMORYTYPE_HOST;
            m3cp.srcHost = m3data;
            m3cp.srcPitch = 8;
            m3cp.srcHeight = 1;
            m3cp.dstMemoryType = CU_MEMORYTYPE_ARRAY;
            m3cp.dstArray = m3;
            m3cp.WidthInBytes = 8;
            m3cp.Height = 1;
            m3cp.Depth = 2;
            CHECK_CU(cuMemcpy3D(&m3cp));
            CUDA_RESOURCE_DESC mres;
            memset(&mres, 0, sizeof(mres));
            mres.resType = CU_RESOURCE_TYPE_ARRAY;
            mres.res.array.hArray = m2;
            CUsurfObject ms2, ms3;
            CHECK_CU(cuSurfObjectCreate(&ms2, &mres));
            mres.res.array.hArray = m3;
            CHECK_CU(cuSurfObjectCreate(&ms3, &mres));
            CUdeviceptr mh, mo;
            CHECK_CU(cuMemAlloc(&mh, 16));
            CHECK_CU(cuMemAlloc(&mo, 8));
            CUsurfObject mhs[2] = { ms2, ms3 };
            CHECK_CU(cuMemcpyHtoD(mh, mhs, 16));
            void *margs[2] = { &mh, &mo };
            CHECK_CU(cuLaunchKernel(mfn, 1, 1, 1, 1, 1, 1, 0, NULL,
                                    margs, NULL));
            CHECK_CU(cuCtxSynchronize());
            uint32_t mgot[2] = {0};
            CHECK_CU(cuMemcpyDtoH(mgot, mo, 8));
            CHECK(mgot[0] == 20 && mgot[1] == 33);
            printf("  surf2d/3d mixed dynamic tables: 2 words ok\n");
            CHECK_CU(cuSurfObjectDestroy(ms2));
            CHECK_CU(cuSurfObjectDestroy(ms3));
            CHECK_CU(cuMemFree(mh));
            CHECK_CU(cuMemFree(mo));
            CHECK_CU(cuArrayDestroy(m2));
            CHECK_CU(cuArrayDestroy(m3));
            CHECK_CU(cuModuleUnload(mmod));
            free(mptx);
        }
    }

    CUarray arr3;
    CUDA_ARRAY3D_DESCRIPTOR d3 = { 2, 2, 2, CU_AD_FORMAT_FLOAT, 4, 0 };
    CHECK_CU(cuArray3DCreate(&arr3, &d3));
    float vol[32];
    fill_2d_texels(vol, 32);
    CUDA_MEMCPY3D c3;
    memset(&c3, 0, sizeof(c3));
    c3.srcMemoryType = CU_MEMORYTYPE_HOST;
    c3.srcHost = vol;
    c3.srcPitch = 32;
    c3.srcHeight = 2;
    c3.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    c3.dstArray = arr3;
    c3.WidthInBytes = 32;
    c3.Height = 2;
    c3.Depth = 2;
    CHECK_CU(cuMemcpy3D(&c3));
    float vback[32];
    memset(vback, 0, sizeof(vback));
    CUDA_MEMCPY3D r3 = c3;
    r3.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    r3.srcArray = arr3;
    r3.dstMemoryType = CU_MEMORYTYPE_HOST;
    r3.dstHost = vback;
    r3.dstPitch = 32;
    r3.dstHeight = 2;
    CHECK_CU(cuMemcpy3D(&r3));
    int vbad = 0;
    for (int i = 0; i < 32; i++)
        if (vback[i] != vol[i])
            vbad++;
    CHECK(vbad == 0);
    if (!vbad)
        printf("  cuMemcpy3D array round-trip: 32 texels ok\n");

    // Linear texture object: a device buffer snapshotted into a 1D
    // image and fetched with an unnormalized (integer) coordinate.
    {
        CUdeviceptr lin;
        CHECK_CU(cuMemAlloc(&lin, 64));
        CHECK_CU(cuMemcpyHtoD(lin, texels, 64));
        CUDA_RESOURCE_DESC lrd;
        memset(&lrd, 0, sizeof(lrd));
        lrd.resType = CU_RESOURCE_TYPE_LINEAR;
        lrd.res.linear.devPtr = lin;
        lrd.res.linear.format = CU_AD_FORMAT_FLOAT;
        lrd.res.linear.numChannels = 4;
        lrd.res.linear.sizeInBytes = 64;
        CUDA_TEXTURE_DESC ltd = td; // clamp, point, normalized flag clear
        ltd.flags = 0;
        CUtexObject lto;
        CHECK_CU(cuTexObjectCreate(&lto, &lrd, &ltd, NULL));
        CUfunction lfn;
        char lpath[1024];
        snprintf(lpath, sizeof(lpath), "%s/tex1dfetch.ptx",
                 ACUDA_TEST_PTX_DIR);
        char *lptx = read_file(lpath);
        CHECK(lptx != NULL);
        if (lptx) {
            CUmodule lmod;
            CHECK_CU(cuModuleLoadData(&lmod, lptx));
            CHECK_CU(cuModuleGetFunction(&lfn, lmod, "tex1dfetch"));
            CUdeviceptr lo;
            CHECK_CU(cuMemAlloc(&lo, 64));
            void *la[2] = { &lto, &lo };
            CHECK_CU(cuLaunchKernel(lfn, 1, 1, 1, 1, 1, 1, 0, NULL, la,
                                    NULL));
            CHECK_CU(cuCtxSynchronize());
            uint32_t lg[4] = {0};
            CHECK_CU(cuMemcpyDtoH(lg, lo, 16));
            static const uint32_t lw[4] = {
                0x41100000u, 0x41200000u, 0x41300000u, 0x41400000u
            };
            int lbad = 0;
            for (int i = 0; i < 4; i++)
                if (lg[i] != lw[i])
                    lbad++;
            CHECK(lbad == 0);
            if (!lbad)
                printf("  linear texture fetch: 4 words ok\n");
            CHECK_CU(cuMemFree(lo));
            CHECK_CU(cuModuleUnload(lmod));
            free(lptx);
        }
        CHECK_CU(cuTexObjectDestroy(lto));
        CHECK_CU(cuMemFree(lin));
    }

    // Pitch2D texture object: 2x2 rows with a 32-byte pitch.
    {
        CUdeviceptr pit;
        CHECK_CU(cuMemAlloc(&pit, 64));
        CHECK_CU(cuMemcpyHtoD(pit, texels, 64));
        CUDA_RESOURCE_DESC prd;
        memset(&prd, 0, sizeof(prd));
        prd.resType = CU_RESOURCE_TYPE_PITCH2D;
        prd.res.pitch2D.devPtr = pit;
        prd.res.pitch2D.format = CU_AD_FORMAT_FLOAT;
        prd.res.pitch2D.numChannels = 4;
        prd.res.pitch2D.width = 2;
        prd.res.pitch2D.height = 2;
        prd.res.pitch2D.pitchInBytes = 32;
        CUDA_TEXTURE_DESC ptd = td;
        ptd.flags = 0;
        CUtexObject pto;
        CHECK_CU(cuTexObjectCreate(&pto, &prd, &ptd, NULL));
        CUfunction pfn;
        char ppath[1024];
        snprintf(ppath, sizeof(ppath), "%s/tex2dfetch.ptx",
                 ACUDA_TEST_PTX_DIR);
        char *pptx = read_file(ppath);
        CHECK(pptx != NULL);
        if (pptx) {
            CUmodule pmod;
            CHECK_CU(cuModuleLoadData(&pmod, pptx));
            CHECK_CU(cuModuleGetFunction(&pfn, pmod, "tex2dfetch"));
            CUdeviceptr po;
            CHECK_CU(cuMemAlloc(&po, 64));
            void *pa[2] = { &pto, &po };
            CHECK_CU(cuLaunchKernel(pfn, 1, 1, 1, 1, 1, 1, 0, NULL, pa,
                                    NULL));
            CHECK_CU(cuCtxSynchronize());
            uint32_t pg[4] = {0};
            CHECK_CU(cuMemcpyDtoH(pg, po, 16));
            static const uint32_t pw[4] = {
                0x41500000u, 0x41600000u, 0x41700000u, 0x41800000u
            };
            int pbad = 0;
            for (int i = 0; i < 4; i++)
                if (pg[i] != pw[i])
                    pbad++;
            CHECK(pbad == 0);
            if (!pbad)
                printf("  pitch2D texture fetch: 4 words ok\n");
            CHECK_CU(cuMemFree(po));
            CHECK_CU(cuModuleUnload(pmod));
            free(pptx);
        }
        CHECK_CU(cuTexObjectDestroy(pto));
        CHECK_CU(cuMemFree(pit));
    }

    // Mipmapped array: level 1 has a 1x1 extent.
    CUmipmappedArray mip;
    CHECK_CU(cuMipmappedArrayCreate(&mip, &d3, 1));
    CUarray lvl;
    CHECK_CU(cuMipmappedArrayGetLevel(&lvl, mip, 0));
    CHECK_CU(cuArrayDestroy(lvl));
    CHECK_CU(cuMipmappedArrayDestroy(mip));

    // Legacy module-scope .texref: cuModuleGetTexRef + cuTexRefSetAddress
    // must bind the hidden texture parameter at launch.
    {
        char rpath[1024];
        snprintf(rpath, sizeof(rpath), "%s/texfetch.ptx",
                 ACUDA_TEST_PTX_DIR);
        char *rptx = read_file(rpath);
        CHECK(rptx != NULL);
        if (rptx) {
            CUmodule rmod;
            CHECK_CU(cuModuleLoadData(&rmod, rptx));
            CUtexref tr;
            CHECK_CU(cuModuleGetTexRef(&tr, rmod, "tex1"));
            CHECK(tr != NULL);
            CUdeviceptr rin, rout;
            CHECK_CU(cuMemAlloc(&rin, 32 * 4));
            CHECK_CU(cuMemAlloc(&rout, 32 * 4));
            float rhost[32];
            for (int i = 0; i < 32; i++)
                rhost[i] = (float)(100 + i);
            CHECK_CU(cuMemcpyHtoD(rin, rhost, sizeof(rhost)));
            CHECK_CU(cuMemcpyHtoD(rout, rhost, sizeof(rhost)));
            CHECK_CU(cuTexRefSetFormat(tr, CU_AD_FORMAT_FLOAT, 1));
            size_t roff = 123;
            CHECK_CU(cuTexRefSetAddress_v2(&roff, tr, rin,
                                           sizeof(rhost)));
            CHECK(roff == 0);
            CHECK_CU(cuTexRefSetFilterMode(tr, CU_TR_FILTER_MODE_POINT));
            CHECK_CU(cuTexRefSetAddressMode(tr, 0,
                                            CU_TR_ADDRESS_MODE_CLAMP));
            CUfunction rfn;
            CHECK_CU(cuModuleGetFunction(&rfn, rmod, "texfetch"));
            void *rargs[1] = { &rout };
            CHECK_CU(cuLaunchKernel(rfn, 1, 1, 1, 32, 1, 1, 0, NULL,
                                    rargs, NULL));
            CHECK_CU(cuCtxSynchronize());
            float rgot[32];
            CHECK_CU(cuMemcpyDtoH(rgot, rout, sizeof(rgot)));
            int rbad = 0;
            for (int i = 0; i < 32; i++)
                if (rgot[i] != 100.0f)
                    rbad++;
            CHECK(rbad == 0);
            if (!rbad)
                printf("  module-scope texref fetch: 32 words ok\n");
            CHECK_CU(cuMemFree(rout));
            CHECK_CU(cuMemFree(rin));
            CHECK_CU(cuModuleUnload(rmod));
            free(rptx);
        }
    }

    CHECK_CU(cuTexObjectDestroy(tobj));
    CHECK_CU(cuArrayDestroy(arr3));
    CHECK_CU(cuArrayDestroy(arr));
    CHECK_CU(cuMemFree(out));
    CHECK_CU(cuModuleUnload(mod));
    free(ptx);

    if (failures == 0)
        printf("driver texture test passed\n");
    return failures != 0;
}
