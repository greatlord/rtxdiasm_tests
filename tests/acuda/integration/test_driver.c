// Phase 5a: drive libcuda.so like a JIT client (numba-style sequence):
// init -> device -> context -> malloc -> HtoD -> link PTX -> module ->
// function -> launch -> DtoH/verify -> cleanup. Plus error paths.
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "acuda/driver_attrs.h"

#ifndef ACUDA_DRIVER_PATH
#define ACUDA_DRIVER_PATH "libcuda.so.1"
#endif
#ifndef ACUDA_TEST_PTX_DIR
#define ACUDA_TEST_PTX_DIR "tests/ptx"
#endif

typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;
typedef struct CUlink_st *CUlinkState;
typedef unsigned long long CUdeviceptr;

static int failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        failures++; \
    } \
} while (0)
#define CHECK_CU(call) do { \
    CUresult _r = (call); \
    if (_r != 0) { \
        printf("FAIL %s:%d: %s -> %d\n", __FILE__, __LINE__, #call, _r); \
        failures++; \
    } \
} while (0)

#define N (1u << 20)

// Host-side f16 bit helpers (oracles for narrow/f16 tests). Same
// specified algorithms as the compiler's lowering (RNE everywhere;
// f16 denormals produced on narrow, flushed on widen; f32/f64
// denormal inputs flush to signed zero), cross-validated against
// numpy float16 (0 mismatches over ~9000 values incl. ties,
// denormals, overflow, NaN).
static uint32_t host_f32_to_f16(uint32_t w) {
    uint32_t s = w >> 31, e = (w >> 23) & 0xffu, m = w & 0x7fffffu;
    if (e == 0xffu) {
        uint32_t mo = m ? ((m >> 13) | 0x200u) : 0;
        return (s << 15) | (0x1fu << 10) | mo;
    }
    if (e == 0)
        return s << 15;
    int32_t exp = (int32_t)e - 127;
    if (exp >= 16)
        return (s << 15) | (0x1fu << 10);
    if (exp >= -14) {
        uint32_t mant = m >> 13, drop = m & 0x1fffu;
        if (drop > 0x1000u || (drop == 0x1000u && (mant & 1u))) {
            mant++;
            if (mant == 0x400u) {
                mant = 0;
                exp++;
                if (exp >= 16)
                    return (s << 15) | (0x1fu << 10);
            }
        }
        return (s << 15) | (((uint32_t)(exp + 15)) << 10) | mant;
    }
    uint32_t full = m | 0x800000u;
    int32_t rsh = -exp - 1;
    if (rsh >= 25)
        return s << 15;
    uint32_t mant = full >> (unsigned)rsh;
    uint32_t drop = full & ~(0xffffffffu << (unsigned)rsh);
    uint32_t half = 1u << (unsigned)(rsh - 1);
    if (drop > half || (drop == half && (mant & 1u)))
        mant++;
    if (mant >= 1024u)
        return (s << 15) | (1u << 10);
    return (s << 15) | mant;
}

static uint32_t host_f64_to_f16(uint64_t w) {
    uint32_t s = (uint32_t)(w >> 63);
    uint32_t e = (uint32_t)((w >> 52) & 0x7ffu);
    uint64_t m = w & 0xfffffffffffffull;
    if (e == 0x7ffu) {
        uint32_t mo = m ? (uint32_t)((m >> 42) | 0x200u) : 0;
        return (s << 15) | (0x1fu << 10) | mo;
    }
    if (e == 0)
        return s << 15;
    int64_t exp = (int64_t)e - 1023;
    if (exp >= 16)
        return (s << 15) | (0x1fu << 10);
    if (exp >= -14) {
        uint64_t mant = m >> 42, drop = m & 0x3ffffffffffull;
        uint64_t half = 0x20000000000ull;
        if (drop > half || (drop == half && (mant & 1u))) {
            mant++;
            if (mant == 0x400u) {
                mant = 0;
                exp++;
                if (exp >= 16)
                    return (s << 15) | (0x1fu << 10);
            }
        }
        return (s << 15) | (((uint32_t)(exp + 15)) << 10) |
               (uint32_t)mant;
    }
    uint64_t full = m | 0x10000000000000ull;
    int64_t rsh = 28 - exp;
    if (rsh >= 54)
        return s << 15;
    uint64_t mant = full >> (unsigned)rsh;
    uint64_t drop = full & ~(~(uint64_t)0 << (unsigned)rsh);
    uint64_t half = 1ull << (unsigned)(rsh - 1);
    if (drop > half || (drop == half && (mant & 1u)))
        mant++;
    if (mant >= 1024u)
        return (s << 15) | (1u << 10);
    return (s << 15) | (uint32_t)mant;
}

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

// Non-negative normal floats -> binary16 bits (WMMA test inputs are
// exact small integers and 1.0).
static uint32_t drv_f32_to_f16_bits(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    if ((x & 0x7fffffffu) == 0)
        return (x >> 16) & 0x8000u;
    int exp = (int)((x >> 23) & 0xffu) - 127;
    uint32_t man = x & 0x7fffffu;
    return ((x >> 16) & 0x8000u) | ((uint32_t)(exp + 15) << 10) |
           (man >> 13);
}

static uint32_t drv_f32_bits(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    return x;
}

// Build the 416-word f16 sparse-mma buffer (same layout as
// tests/ptx/mma_sp.ptx): A is 2:4 sparse with indices (0,2)/(1,3),
// metadata 0xD8D8D8D8, and D = A*B + C.
static void drv_build_sparse_mma(uint32_t *in, uint32_t *want) {
    float sa[16][16], sb[16][8], sc[16][8];
    static const int si0[4] = {0, 1, 0, 1};
    static const int si1[4] = {2, 3, 2, 3};
    memset(sa, 0, sizeof(sa));
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 4; c++)
            for (int q = 0; q < 2; q++) {
                int k = 4 * c + (q ? si1[c] : si0[c]);
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
            uint32_t v0 = drv_f32_to_f16_bits(sa[r][4 * t + si0[t]]);
            uint32_t v1 = drv_f32_to_f16_bits(sa[r][4 * t + si1[t]]);
            in[L * 2 + hr] = v0 | (v1 << 16);
        }
        in[64 + L] = 0xd8d8d8d8u;
        in[96 + L * 2 + 0] =
            drv_f32_to_f16_bits(sb[2 * t][g]) |
            (drv_f32_to_f16_bits(sb[2 * t + 1][g]) << 16);
        in[96 + L * 2 + 1] =
            drv_f32_to_f16_bits(sb[2 * t + 8][g]) |
            (drv_f32_to_f16_bits(sb[2 * t + 9][g]) << 16);
        for (int m = 0; m < 4; m++)
            in[160 + L * 4 + m] =
                drv_f32_bits(sc[g + 8 * (m >= 2)][2 * t + (m & 1)]);
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
            want[288 + L * 4 + m] = drv_f32_bits(d);
        }
    }
}

static uint32_t drv_f32_to_bf16(float f) { return drv_f32_bits(f) >> 16; }

// 352-word buffer for tests/ptx/mma_k8{,_bf16}.ptx (D = A*B + C).
static void drv_build_k8(uint32_t *in, uint32_t *want, int bf16) {
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
        uint32_t a00 = bf16 ? drv_f32_to_bf16(sa[g][2 * t])
                            : drv_f32_to_f16_bits(sa[g][2 * t]);
        uint32_t a01 = bf16 ? drv_f32_to_bf16(sa[g][2 * t + 1])
                            : drv_f32_to_f16_bits(sa[g][2 * t + 1]);
        uint32_t a10 = bf16 ? drv_f32_to_bf16(sa[g + 8][2 * t])
                            : drv_f32_to_f16_bits(sa[g + 8][2 * t]);
        uint32_t a11 = bf16 ? drv_f32_to_bf16(sa[g + 8][2 * t + 1])
                            : drv_f32_to_f16_bits(sa[g + 8][2 * t + 1]);
        in[L * 2 + 0] = a00 | (a01 << 16);
        in[L * 2 + 1] = a10 | (a11 << 16);
        uint32_t b0 = bf16 ? drv_f32_to_bf16(sb[2 * t][g])
                           : drv_f32_to_f16_bits(sb[2 * t][g]);
        uint32_t b1 = bf16 ? drv_f32_to_bf16(sb[2 * t + 1][g])
                           : drv_f32_to_f16_bits(sb[2 * t + 1][g]);
        in[64 + L] = b0 | (b1 << 16);
        for (int m = 0; m < 4; m++)
            in[96 + L * 4 + m] =
                drv_f32_bits(sc[g + 8 * (m >= 2)][2 * t + (m & 1)]);
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
            want[224 + L * 4 + m] = drv_f32_bits(d);
        }
    }
}

// 352-word buffer for tests/ptx/mma_k4.ptx (tf32 A/B).
static void drv_build_k4(uint32_t *in, uint32_t *want) {
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
        in[L * 2 + 0] = drv_f32_bits(sa[g][t]);
        in[L * 2 + 1] = drv_f32_bits(sa[g + 8][t]);
        in[64 + L] = drv_f32_bits(sb[t][g]);
        for (int m = 0; m < 4; m++)
            in[96 + L * 4 + m] =
                drv_f32_bits(sc[g + 8 * (m >= 2)][2 * t + (m & 1)]);
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
            want[224 + L * 4 + m] = drv_f32_bits(d);
        }
    }
}

static uint32_t drv_pack_bytes4(const int *v) {
    uint32_t w = 0;
    for (int i = 0; i < 4; i++)
        w |= ((uint32_t)v[i] & 0xffu) << (8 * i);
    return w;
}

// 448-word buffer for tests/ptx/mma_{u8,s8u8,u8s8}.ptx.
static void drv_build_int8(uint32_t *in, uint32_t *want, int asign,
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
        in[L * 4 + 0] = drv_pack_bytes4(v0);
        in[L * 4 + 1] = drv_pack_bytes4(v1);
        in[L * 4 + 2] = drv_pack_bytes4(v2);
        in[L * 4 + 3] = drv_pack_bytes4(v3);
        in[128 + L * 2 + 0] = drv_pack_bytes4(w0);
        in[128 + L * 2 + 1] = drv_pack_bytes4(w1);
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

static uint32_t drv_pack_f16_pair(float a, float b) {
    return drv_f32_to_f16_bits(a) | (drv_f32_to_f16_bits(b) << 16);
}

// Buffer for tests/ptx/mma_f16acc.ptx (k8=0) or mma_k8_f16acc.ptx.
static void drv_build_f16acc(uint32_t *in, uint32_t *want, int k8) {
    int K = k8 ? 8 : 16;
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
            in[L * 2 + 0] = drv_pack_f16_pair(A[g][2 * t],
                                              A[g][2 * t + 1]);
            in[L * 2 + 1] = drv_pack_f16_pair(A[g + 8][2 * t],
                                              A[g + 8][2 * t + 1]);
            in[bob + L] = drv_pack_f16_pair(B[2 * t][g],
                                            B[2 * t + 1][g]);
        } else {
            in[L * 4 + 0] = drv_pack_f16_pair(A[g][2 * t],
                                              A[g][2 * t + 1]);
            in[L * 4 + 1] = drv_pack_f16_pair(A[g + 8][2 * t],
                                              A[g + 8][2 * t + 1]);
            in[L * 4 + 2] = drv_pack_f16_pair(A[g][8 + 2 * t],
                                              A[g][8 + 2 * t + 1]);
            in[L * 4 + 3] = drv_pack_f16_pair(A[g + 8][8 + 2 * t],
                                              A[g + 8][8 + 2 * t + 1]);
            in[bob + L * 2 + 0] = drv_pack_f16_pair(B[2 * t][g],
                                                    B[2 * t + 1][g]);
            in[bob + L * 2 + 1] = drv_pack_f16_pair(B[2 * t + 8][g],
                                                    B[2 * t + 9][g]);
        }
        in[cob + L * 2 + 0] = drv_pack_f16_pair(C[g][2 * t],
                                               C[g][2 * t + 1]);
        in[cob + L * 2 + 1] = drv_pack_f16_pair(C[g + 8][2 * t],
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
        want[dob + L * 2 + 0] = drv_pack_f16_pair((float)d0, (float)d1);
        want[dob + L * 2 + 1] = drv_pack_f16_pair((float)d2, (float)d3);
    }
}

// 384-word buffer for tests/ptx/mma_f64.ptx (m8n8k4 f64).
static void drv_build_f64(uint32_t *in, uint32_t *want) {
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

// 768-word buffer for tests/ptx/wmma_bf16.ptx.
static void drv_build_wmma_bf16(uint32_t *in, uint32_t *want) {
    memset(in, 0, 768 * 4);
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 16; c++) {
            int ai = r * 16 + c;
            uint32_t ah = (r == c) ? drv_f32_to_bf16(1.0f) : 0;
            in[ai >> 1] |= ah << ((ai & 1) * 16);
        }
    for (int c = 0; c < 16; c++)
        for (int k = 0; k < 16; k++) {
            int bi = c * 16 + k;
            uint32_t bh = drv_f32_to_bf16((float)(16 * k + c));
            in[128 + (bi >> 1)] |= bh << ((bi & 1) * 16);
        }
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 16; c++) {
            in[256 + r * 16 + c] = drv_f32_bits((float)(16 * r + c));
            in[512 + r * 16 + c] =
                drv_f32_bits((float)(2 * (16 * r + c)));
        }
    memcpy(want, in, 512 * 4);
    memcpy(want + 512, in + 512, 256 * 4);
}

// 512-word buffer for tests/ptx/wmma_f16acc.ptx.
static void drv_build_wmma_f16acc(uint32_t *in, uint32_t *want) {
    memset(in, 0, 512 * 4);
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 16; c++) {
            int ai = r * 16 + c;
            uint32_t ah = (r == c) ? drv_f32_to_f16_bits(1.0f) : 0;
            in[ai >> 1] |= ah << ((ai & 1) * 16);
        }
    for (int c = 0; c < 16; c++)
        for (int k = 0; k < 16; k++) {
            int bi = c * 16 + k;
            uint32_t bh = drv_f32_to_f16_bits((float)(16 * k + c));
            in[128 + (bi >> 1)] |= bh << ((bi & 1) * 16);
        }
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 16; c++) {
            int i = r * 16 + c;
            uint32_t cv = drv_f32_to_f16_bits((float)(16 * r + c));
            uint32_t dv = drv_f32_to_f16_bits((float)(2 * (16 * r + c)));
            in[256 + (i >> 1)] |= cv << ((i & 1) * 16);
            in[384 + (i >> 1)] |= dv << ((i & 1) * 16);
        }
    memcpy(want, in, 512 * 4);
    memset(in + 384, 0, 128 * 4);
}

// 832-word buffer for the wmma m8n32k16/m32n8k16 tests.
static void drv_build_wmma_shape(uint32_t *in, uint32_t *want, int rows,
                                 int cols, int kdim) {
    int am = rows * kdim, bn = cols * kdim;
    int bbase = am / 2, cbase = (am + bn) / 2;
    memset(in, 0, 832 * 4);
    for (int r = 0; r < rows; r++)
        for (int k = 0; k < kdim; k++) {
            int i = r * kdim + k;
            uint32_t h = drv_f32_to_f16_bits(
                (float)(1 + ((r * kdim + k) % 5)));
            in[(i >> 1)] |= h << ((i & 1) * 16);
        }
    for (int c = 0; c < cols; c++)
        for (int k = 0; k < kdim; k++) {
            int i = c * kdim + k;
            uint32_t h = drv_f32_to_f16_bits(
                (float)(1 + ((k * 8 + c) % 3)));
            in[bbase + (i >> 1)] |= h << ((i & 1) * 16);
        }
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++) {
            int i = r * cols + c;
            float d = (float)((r * 8 + c) % 7);
            in[cbase + i] = drv_f32_bits(d);
            for (int k = 0; k < kdim; k++)
                d += (float)((1 + ((r * kdim + k) % 5)) *
                             (1 + ((k * 8 + c) % 3)));
            in[cbase + rows * cols + i] = drv_f32_bits(d);
        }
    memcpy(want, in, 832 * 4);
    memset(in + cbase + rows * cols, 0,
           (size_t)(rows * cols) * 4);
}

// 640-word buffer for tests/ptx/mma_m8n8k4_f16.ptx.
static void drv_build_m8f16(uint32_t *in, uint32_t *want) {
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
        in[l * 2 + 0] = drv_f32_to_f16_bits((float)A[comp][rowA][0]) |
                        (drv_f32_to_f16_bits((float)A[comp][rowA][1]) << 16);
        in[l * 2 + 1] = drv_f32_to_f16_bits((float)A[comp][rowA][2]) |
                        (drv_f32_to_f16_bits((float)A[comp][rowA][3]) << 16);
        in[64 + l * 2 + 0] =
            drv_f32_to_f16_bits((float)B[comp][0][colB]) |
            (drv_f32_to_f16_bits((float)B[comp][1][colB]) << 16);
        in[64 + l * 2 + 1] =
            drv_f32_to_f16_bits((float)B[comp][2][colB]) |
            (drv_f32_to_f16_bits((float)B[comp][3][colB]) << 16);
        for (int i = 0; i < 8; i++) {
            int r = (l & 1) + (i & 2) + (hi ? 4 : 0);
            int c = (i & 4) + (l & 2) + (i & 1);
            int d = C[comp][r][c];
            for (int k = 0; k < 4; k++)
                d += A[comp][r][k] * B[comp][k][c];
            in[128 + l * 8 + i] = drv_f32_bits((float)C[comp][r][c]);
            in[384 + l * 8 + i] = drv_f32_bits((float)d);
        }
    }
    memcpy(want, in, 640 * 4);
    memset(in + 384, 0, 256 * 4);
}

// 544-word buffer for tests/ptx/mma_sp_k32.ptx.
static void drv_build_sp_k32(uint32_t *in, uint32_t *want) {
    int A[16][32], B[32][8], C[16][8], D[16][8];
    memset(A, 0, sizeof(A));
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 8; c++) {
            int i0 = (c % 2 == 0) ? 0 : 1;
            int i1 = (c % 2 == 0) ? 2 : 3;
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
        int i1 = (t % 2 == 0) ? 2 : 3;
        in[l * 4 + 0] = drv_f32_to_f16_bits((float)A[g][4 * t + i0]) |
                        (drv_f32_to_f16_bits((float)A[g][4 * t + i1]) << 16);
        in[l * 4 + 1] =
            drv_f32_to_f16_bits((float)A[g + 8][4 * t + i0]) |
            (drv_f32_to_f16_bits((float)A[g + 8][4 * t + i1]) << 16);
        in[l * 4 + 2] =
            drv_f32_to_f16_bits((float)A[g][16 + 4 * t + i0]) |
            (drv_f32_to_f16_bits((float)A[g][16 + 4 * t + i1]) << 16);
        in[l * 4 + 3] =
            drv_f32_to_f16_bits((float)A[g + 8][16 + 4 * t + i0]) |
            (drv_f32_to_f16_bits((float)A[g + 8][16 + 4 * t + i1]) << 16);
        in[128 + l] = 0xd8d8d8d8u;
        for (int r = 0; r < 4; r++) {
            int k0 = 8 * r + 2 * t, k1 = k0 + 1;
            in[160 + l * 4 + r] =
                drv_f32_to_f16_bits((float)B[k0][g]) |
                (drv_f32_to_f16_bits((float)B[k1][g]) << 16);
        }
        for (int j = 0; j < 4; j++) {
            int r = g + 8 * (j >= 2);
            int c = 2 * t + (j & 1);
            in[288 + l * 4 + j] = drv_f32_bits((float)C[r][c]);
            in[416 + l * 4 + j] = drv_f32_bits((float)D[r][c]);
        }
    }
    memcpy(want, in, 544 * 4);
    memset(in + 416, 0, 128 * 4);
}

// 768-word buffer for tests/ptx/wmma_tf32.ptx.
static void drv_build_wmma_tf32(uint32_t *in, uint32_t *want) {
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
            in[r * 8 + k] = drv_f32_bits((float)A[r][k]);
    for (int c = 0; c < 16; c++)
        for (int k = 0; k < 8; k++)
            in[128 + c * 8 + k] = drv_f32_bits((float)B[k][c]);
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 16; c++) {
            in[256 + r * 16 + c] = drv_f32_bits((float)C[r][c]);
            in[512 + r * 16 + c] = drv_f32_bits((float)D[r][c]);
        }
    memcpy(want, in, 768 * 4);
    memset(in + 512, 0, 256 * 4);
}

static void drv_put_f64(uint32_t *w, int i, double v) {
    uint64_t u;
    memcpy(&u, &v, 8);
    w[i] = (uint32_t)u;
    w[i + 1] = (uint32_t)(u >> 32);
}

// Buffer for tests/ptx/mma_f64_k4/k8/k16.ptx (sm_90 target-gated).
static void drv_build_f64_k(uint32_t *in, uint32_t *want, int K) {
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
            drv_put_f64(in, l * (2 * na) + 4 * q,
                        (double)A[g][4 * q + t]);
            drv_put_f64(in, l * (2 * na) + 4 * q + 2,
                        (double)A[g + 8][4 * q + t]);
            drv_put_f64(in, na * 64 + l * (2 * nb) + 2 * q,
                        (double)B[t + 4 * q][g]);
        }
        int base = (na + nb) * 64 + l * 8;
        for (int j = 0; j < 4; j++) {
            int r = g + 8 * (j >= 2);
            int c = 2 * t + (j & 1);
            drv_put_f64(in, base + 2 * j, (double)C[r][c]);
            drv_put_f64(in, base + 256 + 2 * j, (double)D[r][c]);
        }
    }
    memcpy(want, in, (size_t)nwords * 4);
    memset(in + (na + nb) * 64 + 256, 0, 256 * 4);
}

int main(void) {
    void *h = dlopen(ACUDA_DRIVER_PATH, RTLD_NOW);
    if (!h) {
        printf("FAIL: dlopen %s: %s\n", ACUDA_DRIVER_PATH, dlerror());
        return 1;
    }
#define LOAD(sym) \
    CUresult (*sym)
    LOAD(p_cuInit)(unsigned int);
    LOAD(p_cuDriverGetVersion)(int *);
    LOAD(p_cuDeviceGetCount)(int *);
    LOAD(p_cuDeviceGetName)(char *, int, CUdevice);
    LOAD(p_cuDeviceComputeCapability)(int *, int *, CUdevice);
    LOAD(p_cuCtxCreate_v2)(CUcontext *, unsigned int, CUdevice);
    LOAD(p_cuCtxDestroy_v2)(CUcontext);
    LOAD(p_cuCtxSynchronize)(void);
    LOAD(p_cuMemAlloc_v2)(CUdeviceptr *, size_t);
    LOAD(p_cuMemFree_v2)(CUdeviceptr);
    LOAD(p_cuMemcpyHtoD_v2)(CUdeviceptr, const void *, size_t);
    LOAD(p_cuMemcpyDtoH_v2)(void *, CUdeviceptr, size_t);
    LOAD(p_cuMemcpyDtoD_v2)(CUdeviceptr, CUdeviceptr, size_t);
    LOAD(p_cuMemsetD32_v2)(CUdeviceptr, unsigned int, size_t);
    LOAD(p_cuModuleLoadData)(CUmodule *, const void *);
    LOAD(p_cuModuleLoadFatBinary)(CUmodule *, const void *);
    LOAD(p_cuModuleUnload)(CUmodule);
    LOAD(p_cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
    LOAD(p_cuFuncGetAttribute)(int *, int, CUfunction);
    LOAD(p_cuFuncSetAttribute)(CUfunction, int, int);
    LOAD(p_cuLaunchKernel)
    (CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned,
     unsigned, unsigned, CUstream, void **, void **);
    LOAD(p_cuLaunchKernelEx)(const void *, CUfunction, void **, void **);
    LOAD(p_cuLinkCreate)(unsigned, void *, void **, CUlinkState *);
    LOAD(p_cuLinkAddData)
    (CUlinkState, int, void *, size_t, const char *, unsigned, void *,
     void **);
    LOAD(p_cuLinkAddFile)
    (CUlinkState, int, const char *, unsigned, void *, void **);
    LOAD(p_cuLinkComplete)(CUlinkState, void **, size_t *);
    LOAD(p_cuLinkDestroy)(CUlinkState);
    LOAD(p_cuGetErrorString)(CUresult, const char **);
#undef LOAD
#define SYM(var, name) \
    var = dlsym(h, name); \
    if (!var) { \
        printf("FAIL: missing symbol %s\n", name); \
        failures++; \
    }
    SYM(p_cuInit, "cuInit");
    SYM(p_cuDriverGetVersion, "cuDriverGetVersion");
    SYM(p_cuDeviceGetCount, "cuDeviceGetCount");
    SYM(p_cuDeviceGetName, "cuDeviceGetName");
    SYM(p_cuDeviceComputeCapability, "cuDeviceComputeCapability");
    SYM(p_cuCtxCreate_v2, "cuCtxCreate_v2");
    SYM(p_cuCtxDestroy_v2, "cuCtxDestroy_v2");
    SYM(p_cuCtxSynchronize, "cuCtxSynchronize");
    SYM(p_cuMemAlloc_v2, "cuMemAlloc_v2");
    SYM(p_cuMemFree_v2, "cuMemFree_v2");
    SYM(p_cuMemcpyHtoD_v2, "cuMemcpyHtoD_v2");
    SYM(p_cuMemcpyDtoH_v2, "cuMemcpyDtoH_v2");
    SYM(p_cuMemcpyDtoD_v2, "cuMemcpyDtoD_v2");
    SYM(p_cuMemsetD32_v2, "cuMemsetD32_v2");
    SYM(p_cuModuleLoadData, "cuModuleLoadData");
    SYM(p_cuModuleLoadFatBinary, "cuModuleLoadFatBinary");
    SYM(p_cuModuleUnload, "cuModuleUnload");
    SYM(p_cuModuleGetFunction, "cuModuleGetFunction");
    SYM(p_cuFuncGetAttribute, "cuFuncGetAttribute");
    SYM(p_cuFuncSetAttribute, "cuFuncSetAttribute");
    SYM(p_cuLaunchKernel, "cuLaunchKernel");
    SYM(p_cuLaunchKernelEx, "cuLaunchKernelEx");
    SYM(p_cuLinkCreate, "cuLinkCreate");
    SYM(p_cuLinkAddData, "cuLinkAddData");
    SYM(p_cuLinkAddFile, "cuLinkAddFile");
    SYM(p_cuLinkComplete, "cuLinkComplete");
    SYM(p_cuLinkDestroy, "cuLinkDestroy");
    SYM(p_cuGetErrorString, "cuGetErrorString");
#undef SYM
    if (failures)
        return 1;

    // API before init must fail.
    {
        int n = -1;
        CHECK(p_cuDeviceGetCount(&n) == 3); // NOT_INITIALIZED
    }

    CHECK_CU(p_cuInit(0));
    CHECK(p_cuInit(1) == 1); // INVALID_VALUE
    int ver = 0;
    CHECK_CU(p_cuDriverGetVersion(&ver));
    printf("  driver version %d\n", ver);
    // Pinned: the driver reports CUDA 13.4 by design (see
    // docs/ptx-baseline.md). Version checks are minimums, and our PTX
    // acceptance (.version up to 9.4) matches a 13.4-era driver.
    CHECK(ver == 13040);
    int ndev = 0;
    CHECK_CU(p_cuDeviceGetCount(&ndev));
    CHECK(ndev >= 1);
    char name[256];
    CHECK_CU(p_cuDeviceGetName(name, sizeof(name), 0));
    int major = 0, minor = 0;
    CHECK_CU(p_cuDeviceComputeCapability(&major, &minor, 0));
    printf("  device[0] %s cc=%d.%d\n", name, major, minor);

    CUcontext ctx = NULL;
    CHECK_CU(p_cuCtxCreate_v2(&ctx, 0, 0));
    CHECK(ctx != NULL);

    char path[1024];
    snprintf(path, sizeof(path), "%s/vecadd.ptx", ACUDA_TEST_PTX_DIR);
    char *ptx = read_file(path);
    if (!ptx)
        return 1;

    // Link flow (numba bind path): PTX in, "cubin" out.
    CUlinkState ls = NULL;
    CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &ls));
    CHECK(ls != NULL);
    CHECK_CU(p_cuLinkAddData(ls, 1 /*PTX*/, ptx, strlen(ptx), "vecadd", 0,
                             NULL, NULL));
    char cubin_mark[] = { 0x7f, 'E', 'L', 'F' };
    CHECK(p_cuLinkAddData(ls, 0 /*CUBIN*/, cubin_mark, sizeof(cubin_mark),
                          "fake", 0, NULL, NULL) == 200); // INVALID_IMAGE
    void *cubin = NULL;
    size_t cubin_size = 0;
    CHECK_CU(p_cuLinkComplete(ls, &cubin, &cubin_size));
    CHECK(cubin != NULL && cubin_size > 0);

    CUmodule mod = NULL;
    CHECK_CU(p_cuModuleLoadData(&mod, cubin));
    CHECK(mod != NULL);
    {
        // Negative load must not clobber the live module handle.
        CUmodule junk = NULL;
        CHECK(p_cuModuleLoadData(&junk, "garbage") == 200); // INVALID_IMAGE
        CHECK(junk == NULL);
    }

    CUfunction fn = NULL;
    CHECK_CU(p_cuModuleGetFunction(&fn, mod, "vecadd_f32"));
    CHECK(fn != NULL);
    CUfunction bad = NULL;
    CHECK(p_cuModuleGetFunction(&bad, mod, "nope") == 500); // NOT_FOUND

    CUdeviceptr da = 0, db = 0, dc = 0;
    CHECK_CU(p_cuMemAlloc_v2(&da, (size_t)N * 4));
    CHECK_CU(p_cuMemAlloc_v2(&db, (size_t)N * 4));
    CHECK_CU(p_cuMemAlloc_v2(&dc, (size_t)N * 4));
    float *ha = malloc((size_t)N * 4);
    float *hb = malloc((size_t)N * 4);
    float *hc = malloc((size_t)N * 4);
    CHECK(ha && hb && hc);
    for (uint32_t i = 0; i < N; i++) {
        ha[i] = (float)i * 0.5f;
        hb[i] = (float)i * 2.0f + 1.0f;
    }
    CHECK_CU(p_cuMemcpyHtoD_v2(da, ha, (size_t)N * 4));
    CHECK_CU(p_cuMemcpyHtoD_v2(db, hb, (size_t)N * 4));

    uint32_t n = N;
    void *params[4] = { &da, &db, &dc, &n };
    CHECK_CU(p_cuLaunchKernel(fn, N / 256, 1, 1, 256, 1, 1, 0, NULL, params,
                             NULL));
    CHECK_CU(p_cuCtxSynchronize());
    CHECK_CU(p_cuMemcpyDtoH_v2(hc, dc, (size_t)N * 4));
    uint32_t bad_count = 0;
    for (uint32_t i = 0; i < N; i++) {
        float want = ha[i] * 2.0f + hb[i];
        if (hc[i] != want) {
            if (bad_count < 5)
                printf("  mismatch[%u]: got %f want %f\n", i, hc[i], want);
            bad_count++;
        }
    }
    CHECK(bad_count == 0);
    printf("  driver vecadd_f32: %u elements ok\n", N);

    const char *str = NULL;
    CHECK_CU(p_cuGetErrorString(500, &str));
    printf("  500 = %s\n", str);

    free(ha);
    free(hb);
    free(hc);
    CHECK_CU(p_cuMemFree_v2(da));
    CHECK_CU(p_cuMemFree_v2(db));
    CHECK_CU(p_cuMemFree_v2(dc));

    // Offset pointers: sub-views with plain arithmetic (caching allocators
    // depend on this; opaque handles could never do it).
    {
        CUdeviceptr big = 0;
        CHECK_CU(p_cuMemAlloc_v2(&big, 1u << 20));
        float pat[4] = { 11.0f, 22.0f, 33.0f, 44.0f };
        CHECK_CU(p_cuMemcpyHtoD_v2(big + 4096, pat, sizeof(pat)));
        CHECK_CU(p_cuMemcpyDtoD_v2(big + 8192, big + 4096, sizeof(pat)));
        float got[4] = { 0 };
        CHECK_CU(p_cuMemcpyDtoH_v2(got, big + 8192, sizeof(pat)));
        CHECK(memcmp(got, pat, sizeof(pat)) == 0);
        CHECK_CU(p_cuMemsetD32_v2(big + 12288, 0x3f800000u, 4));
        float ones[4] = { 0 };
        CHECK_CU(p_cuMemcpyDtoH_v2(ones, big + 12288, sizeof(ones)));
        CHECK(ones[0] == 1.0f && ones[3] == 1.0f);
        // Launch through offset base pointers (sub-allocations).
        CUfunction fn2 = NULL;
        CHECK_CU(p_cuModuleGetFunction(&fn2, mod, "vecadd_f32"));
        const uint32_t n2 = 4096;
        CUdeviceptr c2 = 0;
        CHECK_CU(p_cuMemAlloc_v2(&c2, (size_t)n2 * 4));
        float *pa = malloc((size_t)n2 * 4);
        float *pb = malloc((size_t)n2 * 4);
        float *pc = malloc((size_t)n2 * 4);
        CHECK(pa && pb && pc);
        for (uint32_t i = 0; i < n2; i++) {
            pa[i] = (float)i;
            pb[i] = 1000.0f - (float)i;
        }
        CUdeviceptr ao = big + 65536, bo = big + 131072;
        CHECK_CU(p_cuMemcpyHtoD_v2(ao, pa, (size_t)n2 * 4));
        CHECK_CU(p_cuMemcpyHtoD_v2(bo, pb, (size_t)n2 * 4));
        uint32_t nn2 = n2;
        void *p2[4] = { (void *)&ao, (void *)&bo, (void *)&c2,
                        (void *)&nn2 };
        CHECK_CU(p_cuLaunchKernel(fn2, n2 / 256, 1, 1, 256, 1, 1, 0, NULL,
                                 p2, NULL));
        CHECK_CU(p_cuCtxSynchronize());
        CHECK_CU(p_cuMemcpyDtoH_v2(pc, c2, (size_t)n2 * 4));
        uint32_t bad2 = 0;
        for (uint32_t i = 0; i < n2; i++) {
            float want = pa[i] * 2.0f + pb[i];
            if (pc[i] != want) {
                if (bad2 < 5)
                    printf("  offset mismatch[%u]: got %f want %f\n", i,
                           pc[i], want);
                bad2++;
            }
        }
        CHECK(bad2 == 0);
        printf("  driver offset views: %u elements ok\n", n2);
        free(pa);
        free(pb);
        free(pc);
        CHECK_CU(p_cuMemFree_v2(c2));
        CHECK_CU(p_cuMemFree_v2(big));
    }

    // Numba EJIT kernel (tests/ptx/nb_ejit.ptx): full launch with
    // positional args. Pinned layout (verified via ACUDA_DEBUG_CLASSIFY):
    // data pointers at params 4/11/18, N at 16; params 5/12/19 are
    // don't-care masks (idx>=0 always); rest unused.
    {
        snprintf(path, sizeof(path), "%s/nb_ejit.ptx", ACUDA_TEST_PTX_DIR);
        char *nb = read_file(path);
        if (!nb)
            return 1;
        CUlinkState nls = NULL;
        CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &nls));
        CHECK_CU(p_cuLinkAddData(nls, 1, nb, strlen(nb), "nb", 0, NULL,
                                 NULL));
        void *ncubin = NULL;
        size_t ncubin_size = 0;
        CHECK_CU(p_cuLinkComplete(nls, &ncubin, &ncubin_size));
        CUmodule nmod = NULL;
        CHECK_CU(p_cuModuleLoadData(&nmod, ncubin));
        CUfunction nfn = NULL;
        CHECK_CU(p_cuModuleGetFunction(&nfn, nmod,
            "_ZN6cudapy8__main__10vector_addB2v1B96cw51cXTLSUwv1sCUt9Ww0FEw0NRRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAk0Bynm21OizQ1K0UoIGvDpQE8oxrNQE_3dE5ArrayIfLi1E1C7mutable7alignedE5ArrayIfLi1E1C7mutable7alignedE5ArrayIfLi1E1C7mutable7alignedE"));
        const uint32_t nn = 4096;
        CUdeviceptr na = 0, nbp = 0, nc = 0;
        CHECK_CU(p_cuMemAlloc_v2(&na, (size_t)nn * 4));
        CHECK_CU(p_cuMemAlloc_v2(&nbp, (size_t)nn * 4));
        CHECK_CU(p_cuMemAlloc_v2(&nc, (size_t)nn * 4));
        float *nha = malloc((size_t)nn * 4);
        float *nhb = malloc((size_t)nn * 4);
        float *nhc = malloc((size_t)nn * 4);
        if (nha && nhb && nhc) {
            for (uint32_t i = 0; i < nn; i++) {
                nha[i] = (float)i * 0.5f;
                nhb[i] = (float)i * 2.0f + 1.0f;
            }
            CHECK_CU(p_cuMemcpyHtoD_v2(na, nha, (size_t)nn * 4));
            CHECK_CU(p_cuMemcpyHtoD_v2(nbp, nhb, (size_t)nn * 4));
            CUdeviceptr args[21];
            memset(args, 0, sizeof(args));
            uint64_t nnu = nn;
            args[4] = na;
            args[11] = nbp;
            args[16] = nnu;
            args[18] = nc;
            void *par[21];
            for (int i = 0; i < 21; i++)
                par[i] = &args[i];
            CHECK_CU(p_cuLaunchKernel(nfn, nn / 256, 1, 1, 256, 1, 1, 0,
                                     NULL, par, NULL));
            CHECK_CU(p_cuCtxSynchronize());
            CHECK_CU(p_cuMemcpyDtoH_v2(nhc, nc, (size_t)nn * 4));
            uint32_t bad3 = 0;
            for (uint32_t i = 0; i < nn; i++) {
                float want = nha[i] + nhb[i];
                if (nhc[i] != want) {
                    if (bad3 < 5)
                        printf("  nb mismatch[%u]: got %f want %f\n", i,
                               nhc[i], want);
                    bad3++;
                }
            }
            CHECK(bad3 == 0);
            {
                float mn = nhc[0], mx = nhc[0];
                uint32_t zeros = 0;
                for (uint32_t i = 0; i < nn; i++) {
                    if (nhc[i] < mn)
                        mn = nhc[i];
                    if (nhc[i] > mx)
                        mx = nhc[i];
                    if (nhc[i] == 0.0f)
                        zeros++;
                }
                printf("  nb stats: min=%f max=%f zeros=%u/4096\n", mn, mx,
                       zeros);
                for (uint32_t i = 1020; i < 1032; i++)
                    printf("  nb c[%u]=%f (want %f)\n", i, nhc[i],
                           nha[i] + nhb[i]);
            }
            printf("  driver numba-ejit: %u elements ok\n", nn);
        }
        free(nha);
        free(nhb);
        free(nhc);
        CHECK_CU(p_cuMemFree_v2(na));
        CHECK_CU(p_cuMemFree_v2(nbp));
        CHECK_CU(p_cuMemFree_v2(nc));
        CHECK_CU(p_cuModuleUnload(nmod));
        CHECK_CU(p_cuLinkDestroy(nls));
        free(nb);
    }

    // Numba .func kernel (tests/ptx/nb_f32.ptx): same vecadd, different
    // ABI — 22 .b64 params + retval, explicit element-size params.
    // Pinned layout: status ptr at 0, data at 5/12/19, N at 17,
    // masks 6/13/20 = 0, elsizes 7/14/21 = 4; rest unused.
    {
        snprintf(path, sizeof(path), "%s/nb_f32.ptx", ACUDA_TEST_PTX_DIR);
        char *nb = read_file(path);
        if (!nb)
            return 1;
        CUlinkState nls = NULL;
        CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &nls));
        CHECK_CU(p_cuLinkAddData(nls, 1, nb, strlen(nb), "nb32", 0, NULL,
                                 NULL));
        void *ncubin = NULL;
        size_t ncubin_size = 0;
        CHECK_CU(p_cuLinkComplete(nls, &ncubin, &ncubin_size));
        CUmodule nmod = NULL;
        CHECK_CU(p_cuModuleLoadData(&nmod, ncubin));
        CUfunction nfn = NULL;
        CHECK_CU(p_cuModuleGetFunction(&nfn, nmod,
            "_ZN8__main__10vector_addB2v1B74cw51cXTLSUwv1sCUt9Ww0FEw0NRRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAltAEAA_3d_3dE5ArrayIfLi1E1A7mutable7alignedE5ArrayIfLi1E1A7mutable7alignedE5ArrayIfLi1E1A7mutable7alignedE"));
        const uint32_t nn = 4096;
        CUdeviceptr na = 0, nbp = 0, nc = 0, nst = 0;
        CHECK_CU(p_cuMemAlloc_v2(&na, (size_t)nn * 4));
        CHECK_CU(p_cuMemAlloc_v2(&nbp, (size_t)nn * 4));
        CHECK_CU(p_cuMemAlloc_v2(&nc, (size_t)nn * 4));
        CHECK_CU(p_cuMemAlloc_v2(&nst, 8));
        float *nha = malloc((size_t)nn * 4);
        float *nhb = malloc((size_t)nn * 4);
        float *nhc = malloc((size_t)nn * 4);
        if (nha && nhb && nhc) {
            for (uint32_t i = 0; i < nn; i++) {
                nha[i] = (float)i * 0.5f;
                nhb[i] = (float)i * 2.0f + 1.0f;
            }
            CHECK_CU(p_cuMemcpyHtoD_v2(na, nha, (size_t)nn * 4));
            CHECK_CU(p_cuMemcpyHtoD_v2(nbp, nhb, (size_t)nn * 4));
            CUdeviceptr args[22];
            memset(args, 0, sizeof(args));
            uint64_t nnu = nn, four = 4;
            args[0] = nst;
            args[5] = na;
            args[7] = four;
            args[12] = nbp;
            args[14] = four;
            args[17] = nnu;
            args[19] = nc;
            args[21] = four;
            void *par[22];
            for (int i = 0; i < 22; i++)
                par[i] = &args[i];
            CHECK_CU(p_cuLaunchKernel(nfn, nn / 256, 1, 1, 256, 1, 1, 0,
                                     NULL, par, NULL));
            CHECK_CU(p_cuCtxSynchronize());
            CHECK_CU(p_cuMemcpyDtoH_v2(nhc, nc, (size_t)nn * 4));
            uint32_t bad4 = 0;
            for (uint32_t i = 0; i < nn; i++) {
                float want = nha[i] + nhb[i];
                if (nhc[i] != want) {
                    if (bad4 < 5)
                        printf("  nb32 mismatch[%u]: got %f want %f\n", i,
                               nhc[i], want);
                    bad4++;
                }
            }
            CHECK(bad4 == 0);
            printf("  driver numba-func: %u elements ok\n", nn);
        }
        free(nha);
        free(nhb);
        free(nhc);
        CHECK_CU(p_cuMemFree_v2(na));
        CHECK_CU(p_cuMemFree_v2(nbp));
        CHECK_CU(p_cuMemFree_v2(nc));
        CHECK_CU(p_cuMemFree_v2(nst));
        CHECK_CU(p_cuModuleUnload(nmod));
        CHECK_CU(p_cuLinkDestroy(nls));
        free(nb);
    }

    // Original numba harvest (tests/ptx/vecadd_numba.ptx, f64): same ABI
    // as nb_f32 (status 0, data 5/12/19, N 17, masks 0, elsizes 8).
    {
        snprintf(path, sizeof(path), "%s/vecadd_numba.ptx", ACUDA_TEST_PTX_DIR);
        char *nb = read_file(path);
        if (!nb)
            return 1;
        CUlinkState nls = NULL;
        CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &nls));
        CHECK_CU(p_cuLinkAddData(nls, 1, nb, strlen(nb), "nb64", 0, NULL,
                                 NULL));
        void *ncubin = NULL;
        size_t ncubin_size = 0;
        CHECK_CU(p_cuLinkComplete(nls, &ncubin, &ncubin_size));
        CUmodule nmod = NULL;
        CHECK_CU(p_cuModuleLoadData(&nmod, ncubin));
        CUfunction nfn = NULL;
        CHECK_CU(p_cuModuleGetFunction(&nfn, nmod,
            "_ZN8__main__10vector_addB2v1B74cw51cXTLSUwv1sCUt9Ww0FEw1dRRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAltAEAA_3d_3dE5ArrayIdLi1E1A7mutable7alignedE5ArrayIdLi1E1A7mutable7alignedE5ArrayIdLi1E1A7mutable7alignedE"));
        const uint32_t nn = 4096;
        CUdeviceptr na = 0, nbp = 0, nc = 0, nst = 0;
        CHECK_CU(p_cuMemAlloc_v2(&na, (size_t)nn * 8));
        CHECK_CU(p_cuMemAlloc_v2(&nbp, (size_t)nn * 8));
        CHECK_CU(p_cuMemAlloc_v2(&nc, (size_t)nn * 8));
        CHECK_CU(p_cuMemAlloc_v2(&nst, 8));
        double *nha = malloc((size_t)nn * 8);
        double *nhb = malloc((size_t)nn * 8);
        double *nhc = malloc((size_t)nn * 8);
        if (nha && nhb && nhc) {
            for (uint32_t i = 0; i < nn; i++) {
                nha[i] = (double)i * 0.5;
                nhb[i] = (double)i * 2.0 + 1.0;
            }
            CHECK_CU(p_cuMemcpyHtoD_v2(na, nha, (size_t)nn * 8));
            CHECK_CU(p_cuMemcpyHtoD_v2(nbp, nhb, (size_t)nn * 8));
            CUdeviceptr args[22];
            memset(args, 0, sizeof(args));
            uint64_t nnu = nn, eight = 8;
            args[0] = nst;
            args[5] = na;
            args[7] = eight;
            args[12] = nbp;
            args[14] = eight;
            args[17] = nnu;
            args[19] = nc;
            args[21] = eight;
            void *par[22];
            for (int i = 0; i < 22; i++)
                par[i] = &args[i];
            CHECK_CU(p_cuLaunchKernel(nfn, nn / 256, 1, 1, 256, 1, 1, 0,
                                     NULL, par, NULL));
            CHECK_CU(p_cuCtxSynchronize());
            CHECK_CU(p_cuMemcpyDtoH_v2(nhc, nc, (size_t)nn * 8));
            uint32_t bad5 = 0;
            for (uint32_t i = 0; i < nn; i++) {
                double want = nha[i] + nhb[i];
                if (nhc[i] != want) {
                    if (bad5 < 5)
                        printf("  nb64 mismatch[%u]: got %f want %f\n", i,
                               nhc[i], want);
                    bad5++;
                }
            }
            CHECK(bad5 == 0);
            printf("  driver numba-f64: %u elements ok\n", nn);
        }
        free(nha);
        free(nhb);
        free(nhc);
        CHECK_CU(p_cuMemFree_v2(na));
        CHECK_CU(p_cuMemFree_v2(nbp));
        CHECK_CU(p_cuMemFree_v2(nc));
        CHECK_CU(p_cuMemFree_v2(nst));
        CHECK_CU(p_cuModuleUnload(nmod));
        CHECK_CU(p_cuLinkDestroy(nls));
        free(nb);
    }
    // New driver surface (torch needs every one of these to exist):
    // func attributes, extended launch, link-from-file, fatbin loading.
    // Uses fn (vecadd_f32, function scope) throughout.
    {
        int v = -1;
        CHECK_CU(p_cuFuncGetAttribute(&v, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, fn));
        CHECK(v == 1024); // any 1..1024 launches (workgroup specialization)
        CHECK_CU(p_cuFuncGetAttribute(&v, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fn));
        CHECK(v == 0);
        CHECK_CU(p_cuFuncGetAttribute(&v, CU_FUNC_ATTRIBUTE_NUM_REGS, fn));
        CHECK(v > 0);
        CHECK_CU(p_cuFuncGetAttribute(&v, CU_FUNC_ATTRIBUTE_PTX_VERSION, fn));
        CHECK(v == 70); // tests/ptx/vecadd.ptx is .version 7.0
        CHECK_CU(p_cuFuncGetAttribute(&v, CU_FUNC_ATTRIBUTE_BINARY_VERSION, fn));
        CHECK(v == 80); // .target sm_80
        // Carveout and newer advisory attributes answer with a benign
        // default instead of an error (the CUDA 13 runtime caches a
        // failed probe and then rejects later attribute/occupancy calls).
        CHECK_CU(p_cuFuncGetAttribute(&v, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, fn));
        CHECK(v == 0);
        CHECK(p_cuFuncGetAttribute(NULL, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, fn) != 0);
        CHECK_CU(p_cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 16384));
        CHECK_CU(p_cuFuncGetAttribute(&v, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, fn));
        CHECK(v == 16384);
        CHECK_CU(p_cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 0));
        CHECK(p_cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, -1) != 0);
        printf("  driver func attributes ok\n");

        // Extended launch (CUDA 12.4 path torch prefers when present).
        const uint32_t nnx = 256;
        CUdeviceptr ax = 0, bx = 0, cx = 0;
        CHECK_CU(p_cuMemAlloc_v2(&ax, (size_t)nnx * 4));
        CHECK_CU(p_cuMemAlloc_v2(&bx, (size_t)nnx * 4));
        CHECK_CU(p_cuMemAlloc_v2(&cx, (size_t)nnx * 4));
        float *pxa = malloc((size_t)nnx * 4);
        float *pxb = malloc((size_t)nnx * 4);
        float *pxc = malloc((size_t)nnx * 4);
        if (pxa && pxb && pxc) {
            for (uint32_t i = 0; i < nnx; i++) {
                pxa[i] = (float)i;
                pxb[i] = 1.0f;
            }
            CHECK_CU(p_cuMemcpyHtoD_v2(ax, pxa, (size_t)nnx * 4));
            CHECK_CU(p_cuMemcpyHtoD_v2(bx, pxb, (size_t)nnx * 4));
            // CUlaunchConfig mirror (offsets verified vs NVIDIA cuda.h):
            // 7x u32 grid/block/shared, ptr stream, ptr attrs, u32 numAttrs.
            unsigned char cfg[56];
            memset(cfg, 0, sizeof(cfg));
            uint32_t g = 1, b = 256, one1 = 1;
            memcpy(cfg + 0, &g, 4);
            memcpy(cfg + 4, &g, 4);
            memcpy(cfg + 8, &g, 4);
            memcpy(cfg + 12, &b, 4);
            memcpy(cfg + 16, &one1, 4);
            memcpy(cfg + 20, &one1, 4);
            void *parx[4] = { (void *)&ax, (void *)&bx, (void *)&cx,
                              NULL };
            uint32_t nnx4 = nnx;
            parx[3] = &nnx4;
            CHECK_CU(p_cuLaunchKernelEx(cfg, fn, parx, NULL));
            CHECK_CU(p_cuCtxSynchronize());
            CHECK_CU(p_cuMemcpyDtoH_v2(pxc, cx, (size_t)nnx * 4));
            uint32_t badx = 0;
            for (uint32_t i = 0; i < nnx; i++) {
                float want = pxa[i] * 2.0f + pxb[i];
                if (pxc[i] != want)
                    badx++;
            }
            CHECK(badx == 0);
            // Same kernel, different block size (specialized pipeline).
            CHECK_CU(p_cuLaunchKernel(fn, nnx / 32, 1, 1, 32, 1, 1, 0, NULL,
                                     parx, NULL));
            CHECK_CU(p_cuCtxSynchronize());
            CHECK_CU(p_cuMemcpyDtoH_v2(pxc, cx, (size_t)nnx * 4));
            badx = 0;
            for (uint32_t i = 0; i < nnx; i++) {
                float want = pxa[i] * 2.0f + pxb[i];
                if (pxc[i] != want)
                    badx++;
            }
            CHECK(badx == 0);
            printf("  driver block-32 ok\n");
            // Extended attrs are cleanly rejected (none supported).
            uint32_t one = 1;
            memcpy(cfg + 48, &one, 4);
            CHECK(p_cuLaunchKernelEx(cfg, fn, parx, NULL) != 0);
            printf("  driver launch-ex ok\n");
        }
        free(pxa);
        free(pxb);
        free(pxc);
        CHECK_CU(p_cuMemFree_v2(ax));
        CHECK_CU(p_cuMemFree_v2(bx));
        CHECK_CU(p_cuMemFree_v2(cx));

        // Link-from-file delegates to the PTX path.
        {
            snprintf(path, sizeof(path), "%s/vecadd.ptx", ACUDA_TEST_PTX_DIR);
            CUlinkState fls = NULL;
            CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &fls));
            CHECK_CU(p_cuLinkAddFile(fls, 1, path, 0, NULL, NULL));
            CHECK(p_cuLinkAddFile(fls, 1, "/nonexistent.ptx", 0, NULL,
                                  NULL) != 0);
            void *fcubin = NULL;
            size_t fsize = 0;
            CHECK_CU(p_cuLinkComplete(fls, &fcubin, &fsize));
            CUmodule fmod = NULL;
            CHECK_CU(p_cuModuleLoadData(&fmod, fcubin));
            CUfunction ffn = NULL;
            CHECK_CU(p_cuModuleGetFunction(&ffn, fmod, "vecadd_f32"));
            CHECK_CU(p_cuModuleUnload(fmod));
            CHECK_CU(p_cuLinkDestroy(fls));
            printf("  driver link-add-file ok\n");
        }

        // Fatbin container: synthetic PTX member loads, garbage does not.
        {
            CUmodule junk = NULL;
            CHECK(p_cuModuleLoadFatBinary(&junk, "garbage") != 0);
            CHECK(junk == NULL);
            // Minimal wrapper + blob around the real vecadd PTX.
            snprintf(path, sizeof(path), "%s/vecadd.ptx", ACUDA_TEST_PTX_DIR);
            char *fptx = read_file(path);
            if (!fptx)
                return 1;
            size_t plen = strlen(fptx);
            size_t cap = 16 + 64 + plen;
            unsigned char *blob = calloc(1, cap);
            CHECK(blob != NULL);
            if (blob) {
                // container header: magic, ver, hdrsize, files_size
                blob[0] = 0x50; blob[1] = 0xED; blob[2] = 0x55; blob[3] = 0xBA;
                blob[4] = 1; blob[6] = 16;
                uint64_t fs = 64 + plen;
                memcpy(blob + 8, &fs, 8);
                // member: kind=PTX, ver=0x101, header_size=64
                blob[16] = 1; blob[18] = 1; blob[19] = 1;
                blob[20] = 64;
                memcpy(blob + 24, &plen, 4);
                memcpy(blob + 80, fptx, plen);
                unsigned char wrap[24];
                memset(wrap, 0, sizeof(wrap));
                wrap[0] = 0xB1; wrap[1] = 0x43; wrap[2] = 0x62; wrap[3] = 0x46;
                wrap[4] = 1;
                memcpy(wrap + 8, &blob, sizeof(blob));
                CUmodule fbmod = NULL;
                CHECK_CU(p_cuModuleLoadFatBinary(&fbmod, wrap));
                CUfunction fbfn = NULL;
                CHECK_CU(p_cuModuleGetFunction(&fbfn, fbmod, "vecadd_f32"));
                CHECK_CU(p_cuModuleUnload(fbmod));
                free(blob);
                printf("  driver fatbin ok\n");
            }
            free(fptx);
        }
    }

    // Subset torture kernel (tests/ptx/torture_f32.ptx): every
    // supported op class in one launch. c/d/e zeroed up front (the f64
    // gate skips i == 0); a/b/f uploaded. d/e verify on-device
    // roundtrips, c verifies the f32 data path.
    {
        snprintf(path, sizeof(path), "%s/torture_f32.ptx", ACUDA_TEST_PTX_DIR);
        char *tb = read_file(path);
        if (!tb)
            return 1;
        CUlinkState tls = NULL;
        CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &tls));
        CHECK_CU(p_cuLinkAddData(tls, 1, tb, strlen(tb), "torture", 0,
                                 NULL, NULL));
        void *tcubin = NULL;
        size_t tcubin_size = 0;
        CHECK_CU(p_cuLinkComplete(tls, &tcubin, &tcubin_size));
        CUmodule tmod = NULL;
        CHECK_CU(p_cuModuleLoadData(&tmod, tcubin));
        CUfunction tfn = NULL;
        CHECK_CU(p_cuModuleGetFunction(&tfn, tmod, "torture_f32"));
        const uint32_t nn = 4096;
        CUdeviceptr ta = 0, tbp = 0, tc = 0, td = 0, te = 0, tf = 0;
        CHECK_CU(p_cuMemAlloc_v2(&ta, (size_t)nn * 4));
        CHECK_CU(p_cuMemAlloc_v2(&tbp, (size_t)nn * 4));
        CHECK_CU(p_cuMemAlloc_v2(&tc, (size_t)nn * 4));
        CHECK_CU(p_cuMemAlloc_v2(&td, (size_t)nn * 8));
        CHECK_CU(p_cuMemAlloc_v2(&te, (size_t)nn * 4));
        CHECK_CU(p_cuMemAlloc_v2(&tf, (size_t)nn * 8));
        float *tha = malloc((size_t)nn * 4);
        float *thb = malloc((size_t)nn * 4);
        float *thc = malloc((size_t)nn * 4);
        double *thf = malloc((size_t)nn * 8);
        uint64_t *thd = malloc((size_t)nn * 8);
        uint32_t *the = malloc((size_t)nn * 4);
        if (tha && thb && thc && thf && thd && the) {
            for (uint32_t i = 0; i < nn; i++) {
                tha[i] = (float)i * 0.5f;
                thb[i] = (float)i * 2.0f + 1.0f;
                thf[i] = 1.25 * (double)i;
            }
            CHECK_CU(p_cuMemcpyHtoD_v2(ta, tha, (size_t)nn * 4));
            CHECK_CU(p_cuMemcpyHtoD_v2(tbp, thb, (size_t)nn * 4));
            CHECK_CU(p_cuMemcpyHtoD_v2(tf, thf, (size_t)nn * 8));
            CHECK_CU(p_cuMemsetD32_v2(tc, 0, (size_t)nn));
            CHECK_CU(p_cuMemsetD32_v2(td, 0, (size_t)nn * 2));
            CHECK_CU(p_cuMemsetD32_v2(te, 0, (size_t)nn));
            CUdeviceptr args[6];
            args[0] = ta;
            args[1] = tbp;
            args[2] = tc;
            args[3] = td;
            args[4] = te;
            args[5] = tf;
            void *par[6];
            for (int i = 0; i < 6; i++)
                par[i] = &args[i];
            CHECK_CU(p_cuLaunchKernel(tfn, nn / 256, 1, 1, 256, 1, 1, 0,
                                     NULL, par, NULL));
            CHECK_CU(p_cuCtxSynchronize());
            CHECK_CU(p_cuMemcpyDtoH_v2(thc, tc, (size_t)nn * 4));
            CHECK_CU(p_cuMemcpyDtoH_v2(thd, td, (size_t)nn * 8));
            CHECK_CU(p_cuMemcpyDtoH_v2(the, te, (size_t)nn * 4));
            uint32_t badt = 0;
            for (uint32_t i = 0; i < nn; i++) {
                double wantc = 1.25 * (double)i;
                if ((double)thc[i] != wantc) {
                    if (badt < 5)
                        printf("  torture c[%u]: got %f want %f\n", i,
                               thc[i], wantc);
                    badt++;
                }
                if (thd[i] != (uint64_t)i) {
                    if (badt < 5)
                        printf("  torture d[%u]: got %llu want %u\n", i,
                               (unsigned long long)thd[i], i);
                    badt++;
                }
                if (the[i] != i) {
                    if (badt < 5)
                        printf("  torture e[%u]: got %u want %u\n", i,
                               the[i], i);
                    badt++;
                }
            }
            CHECK(badt == 0);
            printf("  driver torture_f32: %u elements ok\n", nn);
        }
        free(tha);
        free(thb);
        free(thc);
        free(thf);
        free(thd);
        free(the);
        CHECK_CU(p_cuMemFree_v2(ta));
        CHECK_CU(p_cuMemFree_v2(tbp));
        CHECK_CU(p_cuMemFree_v2(tc));
        CHECK_CU(p_cuMemFree_v2(td));
        CHECK_CU(p_cuMemFree_v2(te));
        CHECK_CU(p_cuMemFree_v2(tf));
        CHECK_CU(p_cuModuleUnload(tmod));
        CHECK_CU(p_cuLinkDestroy(tls));
        free(tb);
    }
    // Numba dot-product (tests/ptx/dot_numba.ptx): unrolled f32 loop
    // with f64 accumulation (cvt.f64.f32 + cvt.rn.f32.f64). Same ABI
    // family (status 0, data 5/12/19, masks 0, elsizes 4); values are
    // exactly representable so == comparison is sound.
    {
        snprintf(path, sizeof(path), "%s/dot_numba.ptx", ACUDA_TEST_PTX_DIR);
        char *nb = read_file(path);
        if (!nb)
            return 1;
        CUlinkState nls = NULL;
        CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &nls));
        CHECK_CU(p_cuLinkAddData(nls, 1, nb, strlen(nb), "dot", 0, NULL,
                                 NULL));
        void *ncubin = NULL;
        size_t ncubin_size = 0;
        CHECK_CU(p_cuLinkComplete(nls, &ncubin, &ncubin_size));
        CUmodule nmod = NULL;
        CHECK_CU(p_cuModuleLoadData(&nmod, ncubin));
        CUfunction nfn = NULL;
        CHECK_CU(p_cuModuleGetFunction(&nfn, nmod,
            "_ZN13_3cdynamic_3e10dot_kernelB2v1B74cw51cXTLSUwv1sCUt9Ww0FEw0NRRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAltAEAA_3d_3dE5ArrayIfLi1E1A7mutable7alignedE5ArrayIfLi1E1A7mutable7alignedE5ArrayIfLi1E1A7mutable7alignedE"));
        const uint32_t nn = 4096, kk = 8;
        CUdeviceptr na = 0, nbp = 0, nc = 0, nst = 0;
        CHECK_CU(p_cuMemAlloc_v2(&na, (size_t)nn * kk * 4));
        CHECK_CU(p_cuMemAlloc_v2(&nbp, (size_t)nn * kk * 4));
        CHECK_CU(p_cuMemAlloc_v2(&nc, (size_t)nn * 4));
        CHECK_CU(p_cuMemAlloc_v2(&nst, 8));
        float *nha = malloc((size_t)nn * kk * 4);
        float *nhb = malloc((size_t)nn * kk * 4);
        float *nhc = malloc((size_t)nn * 4);
        if (nha && nhb && nhc) {
            for (uint32_t i = 0; i < nn * kk; i++) {
                nha[i] = (float)(i % 5 + 1);
                nhb[i] = (float)(i % 3 + 1);
            }
            CHECK_CU(p_cuMemcpyHtoD_v2(na, nha, (size_t)nn * kk * 4));
            CHECK_CU(p_cuMemcpyHtoD_v2(nbp, nhb, (size_t)nn * kk * 4));
            CUdeviceptr args[22];
            memset(args, 0, sizeof(args));
            uint64_t four = 4;
            args[0] = nst;
            args[5] = na;
            args[7] = four;
            args[12] = nbp;
            args[14] = four;
            args[19] = nc;
            args[21] = four;
            void *par[22];
            for (int i = 0; i < 22; i++)
                par[i] = &args[i];
            CHECK_CU(p_cuLaunchKernel(nfn, nn / 256, 1, 1, 256, 1, 1, 0,
                                     NULL, par, NULL));
            CHECK_CU(p_cuCtxSynchronize());
            CHECK_CU(p_cuMemcpyDtoH_v2(nhc, nc, (size_t)nn * 4));
            uint32_t bad6 = 0;
            for (uint32_t i = 0; i < nn; i++) {
                double want = 0;
                for (uint32_t j = 0; j < kk; j++)
                    want += (double)nha[i * kk + j] * (double)nhb[i * kk + j];
                if ((double)nhc[i] != want) {
                    if (bad6 < 5)
                        printf("  dot mismatch[%u]: got %f want %f\n", i,
                               nhc[i], want);
                    bad6++;
                }
            }
            CHECK(bad6 == 0);
            printf("  driver numba-dot: %u elements ok\n", nn);
        }
        free(nha);
        free(nhb);
        free(nhc);
        CHECK_CU(p_cuMemFree_v2(na));
        CHECK_CU(p_cuMemFree_v2(nbp));
        CHECK_CU(p_cuMemFree_v2(nc));
        CHECK_CU(p_cuMemFree_v2(nst));
        CHECK_CU(p_cuModuleUnload(nmod));
        CHECK_CU(p_cuLinkDestroy(nls));
        free(nb);
    }
    // Mini flow kernel (tests/ptx/mini_flow.ptx): selp/not/neg/min/max,
    // sub.s64, setp.u64, negative literals, .pragma, guard early-exit,
    // middle skip with a join phi, and a 2-iteration self-loop.
    // nn=4096 threads, n=4000: threads i>=n exit early (c[i] stays 0).
    {
        snprintf(path, sizeof(path), "%s/mini_flow.ptx", ACUDA_TEST_PTX_DIR);
        char *nb = read_file(path);
        if (!nb)
            return 1;
        CUlinkState nls = NULL;
        CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &nls));
        CHECK_CU(p_cuLinkAddData(nls, 1, nb, strlen(nb), "miniflow", 0,
                                 NULL, NULL));
        void *ncubin = NULL;
        size_t ncubin_size = 0;
        CHECK_CU(p_cuLinkComplete(nls, &ncubin, &ncubin_size));
        CUmodule nmod = NULL;
        CHECK_CU(p_cuModuleLoadData(&nmod, ncubin));
        CUfunction nfn = NULL;
        CHECK_CU(p_cuModuleGetFunction(&nfn, nmod, "flowmini"));
        const uint32_t nn = 4096, ncut = 4000;
        CUdeviceptr na = 0, nbp = 0, nc = 0;
        CHECK_CU(p_cuMemAlloc_v2(&na, (size_t)nn * 4));
        CHECK_CU(p_cuMemAlloc_v2(&nbp, (size_t)nn * 4));
        CHECK_CU(p_cuMemAlloc_v2(&nc, (size_t)nn * 4));
        float *nha = malloc((size_t)nn * 4);
        float *nhb = malloc((size_t)nn * 4);
        float *nhc = malloc((size_t)nn * 4);
        if (nha && nhb && nhc) {
            for (uint32_t i = 0; i < nn; i++) {
                nha[i] = (float)(i % 7 + 1);
                nhb[i] = (float)(i % 5 + 1);
                nhc[i] = 0.0f;
            }
            CHECK_CU(p_cuMemcpyHtoD_v2(na, nha, (size_t)nn * 4));
            CHECK_CU(p_cuMemcpyHtoD_v2(nbp, nhb, (size_t)nn * 4));
            CHECK_CU(p_cuMemcpyHtoD_v2(nc, nhc, (size_t)nn * 4));
            CUdeviceptr args[4];
            args[0] = na;
            args[1] = nbp;
            args[2] = nc;
            args[3] = ncut;
            void *par[4];
            for (int i = 0; i < 4; i++)
                par[i] = &args[i];
            CHECK_CU(p_cuLaunchKernel(nfn, nn / 256, 1, 1, 256, 1, 1, 0,
                                     NULL, par, NULL));
            CHECK_CU(p_cuCtxSynchronize());
            CHECK_CU(p_cuMemcpyDtoH_v2(nhc, nc, (size_t)nn * 4));
            uint32_t bad8 = 0;
            for (uint32_t i = 0; i < nn; i++) {
                float want;
                if (i >= ncut) {
                    want = 0.0f; // early exit: no store, c stays zeroed
                } else {
                    float ai = nha[i], bi = nhb[i];
                    float maxf = ai > bi ? ai : bi;
                    int64_t r10 = (int64_t)i < 7 ? (int64_t)i : 7;
                    uint64_t r11 = (uint64_t)(7 - r10);
                    uint64_t r12 = (uint64_t)(-(int64_t)r11);
                    uint64_t r13 = ~r12;
                    uint64_t r14 = (uint64_t)i < 5 ? r11 : r13;
                    uint64_t r15 = r14 & 255;
                    uint64_t idx2 = ((uint64_t)i + r15) & 4095;
                    float z = nhb[idx2];
                    float acc = -1.5f;
                    if (r15 != 0)
                        acc = (acc + nhb[(i + 0) & 4095]) +
                              nhb[(i + 1) & 4095];
                    want = (maxf + acc) + z;
                }
                if (nhc[i] != want) {
                    if (bad8 < 5)
                        printf("  flow mismatch[%u]: got %f want %f\n", i,
                               nhc[i], want);
                    bad8++;
                }
            }
            CHECK(bad8 == 0);
            printf("  driver miniflow: %u elements ok\n", nn);
        }
        free(nha);
        free(nhb);
        free(nhc);
        CHECK_CU(p_cuMemFree_v2(na));
        CHECK_CU(p_cuMemFree_v2(nbp));
        CHECK_CU(p_cuMemFree_v2(nc));
        CHECK_CU(p_cuModuleUnload(nmod));
        CHECK_CU(p_cuLinkDestroy(nls));
        free(nb);
    }
    // Direct .reg-form call kernel (tests/ptx/call_reg.ptx): a
    // multi-return callee (.reg .u32 + .reg .pred) inlined at the call
    // site, pred result materialized with predicated mov (predicated
    // stores stay out of subset). Single-thread launches.
    {
        snprintf(path, sizeof(path), "%s/call_reg.ptx", ACUDA_TEST_PTX_DIR);
        char *nb = read_file(path);
        if (!nb)
            return 1;
        CUlinkState nls = NULL;
        CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &nls));
        CHECK_CU(p_cuLinkAddData(nls, 1, nb, strlen(nb), "callreg", 0,
                                 NULL, NULL));
        void *ncubin = NULL;
        size_t ncubin_size = 0;
        CHECK_CU(p_cuLinkComplete(nls, &ncubin, &ncubin_size));
        CUmodule nmod = NULL;
        CHECK_CU(p_cuModuleLoadData(&nmod, ncubin));
        CUfunction nfn = NULL;
        CHECK_CU(p_cuModuleGetFunction(&nfn, nmod, "callreg"));
        const uint32_t in0[2] = { 5, 20 };
        const uint32_t want0[2] = { 6, 1 };
        const uint32_t want1[2] = { 21, 0 };
        for (int t = 0; t < 2; t++) {
            CUdeviceptr na = 0, ncp = 0;
            CHECK_CU(p_cuMemAlloc_v2(&na, 16));
            CHECK_CU(p_cuMemAlloc_v2(&ncp, 16));
            CHECK_CU(p_cuMemcpyHtoD_v2(na, &in0[t], 4));
            uint64_t zc[2] = { 0, 0 };
            CHECK_CU(p_cuMemcpyHtoD_v2(ncp, zc, 16));
            void *par[2] = { &na, &ncp };
            CHECK_CU(p_cuLaunchKernel(nfn, 1, 1, 1, 1, 1, 1, 0, NULL,
                                      par, NULL));
            CHECK_CU(p_cuCtxSynchronize());
            uint32_t got[2] = { 0, 0 };
            CHECK_CU(p_cuMemcpyDtoH_v2(got, ncp, 8));
            const uint32_t *want = t == 0 ? want0 : want1;
            if (got[0] != want[0] || got[1] != want[1]) {
                printf("  callreg mismatch[t=%d]: got [%u,%u] want [%u,%u]\n",
                       t, got[0], got[1], want[0], want[1]);
                failures++;
            }
            CHECK_CU(p_cuMemFree_v2(na));
            CHECK_CU(p_cuMemFree_v2(ncp));
        }
        printf("  driver callreg: .reg-form call ok\n");
        CHECK_CU(p_cuModuleUnload(nmod));
        CHECK_CU(p_cuLinkDestroy(nls));
        free(nb);
    }
    // Dynamic-shared launch cap: static + dynamic over 64KB refuses
    // before touching the GPU (CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES).
    {
        snprintf(path, sizeof(path), "%s/call_reg.ptx", ACUDA_TEST_PTX_DIR);
        char *nb = read_file(path);
        if (!nb)
            return 1;
        CUlinkState nls = NULL;
        CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &nls));
        CHECK_CU(p_cuLinkAddData(nls, 1, nb, strlen(nb), "callreg", 0,
                                 NULL, NULL));
        void *ncubin = NULL;
        size_t ncubin_size = 0;
        CHECK_CU(p_cuLinkComplete(nls, &ncubin, &ncubin_size));
        CUmodule nmod = NULL;
        CHECK_CU(p_cuModuleLoadData(&nmod, ncubin));
        CUfunction nfn = NULL;
        CHECK_CU(p_cuModuleGetFunction(&nfn, nmod, "callreg"));
        CUdeviceptr na = 0, ncp = 0;
        CHECK_CU(p_cuMemAlloc_v2(&na, 16));
        CHECK_CU(p_cuMemAlloc_v2(&ncp, 16));
        void *par[2] = { &na, &ncp };
        CHECK(p_cuLaunchKernel(nfn, 1, 1, 1, 1, 1, 1, 70000, NULL, par,
                               NULL) == 701); // LAUNCH_OUT_OF_RESOURCES
        printf("  driver shm-cap: oversized dynamic shared refused\n");
        CHECK_CU(p_cuMemFree_v2(na));
        CHECK_CU(p_cuMemFree_v2(ncp));
        CHECK_CU(p_cuModuleUnload(nmod));
        CHECK_CU(p_cuLinkDestroy(nls));
        free(nb);
    }
    // Predicated-execution coverage (tests/ptx/predcover.ptx):
    // data-dependent pred drives select desugars (taken + skipped,
    // plain + negated + chained) and one trailing predicated store
    // (guard taken + skipped across launches).
    {
        snprintf(path, sizeof(path), "%s/predcover.ptx", ACUDA_TEST_PTX_DIR);
        char *nb = read_file(path);
        if (!nb)
            return 1;
        CUlinkState nls = NULL;
        CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &nls));
        CHECK_CU(p_cuLinkAddData(nls, 1, nb, strlen(nb), "predcover", 0,
                                 NULL, NULL));
        void *ncubin = NULL;
        size_t ncubin_size = 0;
        CHECK_CU(p_cuLinkComplete(nls, &ncubin, &ncubin_size));
        CUmodule nmod = NULL;
        CHECK_CU(p_cuModuleLoadData(&nmod, ncubin));
        CUfunction nfn = NULL;
        CHECK_CU(p_cuModuleGetFunction(&nfn, nmod, "predcover"));
        const uint32_t in0[2] = { 0, 5 };
        const uint32_t want0[2] = { 8, 0 };
        const uint32_t want1[2] = { 105, 105 };
        for (int t = 0; t < 2; t++) {
            CUdeviceptr na = 0, ncp = 0;
            CHECK_CU(p_cuMemAlloc_v2(&na, 16));
            CHECK_CU(p_cuMemAlloc_v2(&ncp, 16));
            CHECK_CU(p_cuMemcpyHtoD_v2(na, &in0[t], 4));
            uint64_t zc[2] = { 0, 0 };
            CHECK_CU(p_cuMemcpyHtoD_v2(ncp, zc, 16));
            void *par[2] = { &na, &ncp };
            CHECK_CU(p_cuLaunchKernel(nfn, 1, 1, 1, 1, 1, 1, 0, NULL,
                                      par, NULL));
            CHECK_CU(p_cuCtxSynchronize());
            uint32_t got[2] = { 0, 0 };
            CHECK_CU(p_cuMemcpyDtoH_v2(got, ncp, 8));
            const uint32_t *want = t == 0 ? want0 : want1;
            if (got[0] != want[0] || got[1] != want[1]) {
                printf("  predcover mismatch[t=%d]: got [%u,%u] want [%u,%u]\n",
                       t, got[0], got[1], want[0], want[1]);
                failures++;
            }
            CHECK_CU(p_cuMemFree_v2(na));
            CHECK_CU(p_cuMemFree_v2(ncp));
        }
        printf("  driver predcover: predicated ALU + store ok\n");
        CHECK_CU(p_cuModuleUnload(nmod));
        CHECK_CU(p_cuLinkDestroy(nls));
        free(nb);
    }
    // Scalar brain-float kernel (tests/ptx/bf16_cvt.ptx): lane tid
    // stores bf16((float)((tid == 0) ? 7 : 9)) as u16 plus a u32
    // witness. 64 lanes: lane0 0x40E0/7, rest 0x4110/9.
    {
        snprintf(path, sizeof(path), "%s/bf16_cvt.ptx", ACUDA_TEST_PTX_DIR);
        char *nb = read_file(path);
        if (!nb)
            return 1;
        CUlinkState nls = NULL;
        CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &nls));
        CHECK_CU(p_cuLinkAddData(nls, 1, nb, strlen(nb), "bf16_cvt", 0,
                                 NULL, NULL));
        void *ncubin = NULL;
        size_t ncubin_size = 0;
        CHECK_CU(p_cuLinkComplete(nls, &ncubin, &ncubin_size));
        CUmodule nmod = NULL;
        CHECK_CU(p_cuModuleLoadData(&nmod, ncubin));
        CUfunction nfn = NULL;
        CHECK_CU(p_cuModuleGetFunction(&nfn, nmod, "bf16_cvt"));
        CUdeviceptr n16 = 0, n32 = 0;
        CHECK_CU(p_cuMemAlloc_v2(&n16, 128));
        CHECK_CU(p_cuMemAlloc_v2(&n32, 256));
        uint64_t zc[32] = { 0 };
        CHECK_CU(p_cuMemcpyHtoD_v2(n16, zc, 128));
        CHECK_CU(p_cuMemcpyHtoD_v2(n32, zc, 256));
        void *par[2] = { &n16, &n32 };
        CHECK_CU(p_cuLaunchKernel(nfn, 1, 1, 1, 64, 1, 1, 0, NULL,
                                  par, NULL));
        CHECK_CU(p_cuCtxSynchronize());
        uint16_t got16[64] = { 0 };
        uint32_t got32[64] = { 0 };
        CHECK_CU(p_cuMemcpyDtoH_v2(got16, n16, 128));
        CHECK_CU(p_cuMemcpyDtoH_v2(got32, n32, 256));
        for (int i = 0; i < 64; i++) {
            uint16_t w16 = i == 0 ? 0x40E0u : 0x4110u;
            uint32_t w32 = i == 0 ? 7u : 9u;
            if (got16[i] != w16 || got32[i] != w32) {
                printf("  bf16 mismatch[i=%d]: got [%04x,%u] want [%04x,%u]\n",
                       i, got16[i], got32[i], w16, w32);
                failures++;
            }
        }
        printf("  driver bf16: scalar bf16 cvt + st.u16 ok\n");
        CHECK_CU(p_cuMemFree_v2(n16));
        CHECK_CU(p_cuMemFree_v2(n32));
        CHECK_CU(p_cuModuleUnload(nmod));
        CHECK_CU(p_cuLinkDestroy(nls));
        free(nb);
    }
    // Integer mma.m16n8k32.s32.s8.s8.s32 + narrow 16-bit ALU wrap,
    // verified against the same NVIDIA-generated vectors ZLUDA pins.
    // 64 lanes x 4 s32 accumulators = 256 words, contiguous.
    {
        static const uint32_t exp_s8[256] = {
#include "mma_s8_expected.inc"
        };
        snprintf(path, sizeof(path), "%s/mma_s8.ptx", ACUDA_TEST_PTX_DIR);
        char *mb = read_file(path);
        if (!mb) {
            printf("  SKIP mma_s8: corpus absent\n");
        } else {
            CUlinkState mls = NULL;
            CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &mls));
            CHECK_CU(p_cuLinkAddData(mls, 1, mb, strlen(mb), "mma_s8",
                                     0, NULL, NULL));
            void *mcubin = NULL;
            size_t mcubin_size = 0;
            CHECK_CU(p_cuLinkComplete(mls, &mcubin, &mcubin_size));
            CUmodule mmod = NULL;
            CHECK_CU(p_cuModuleLoadData(&mmod, mcubin));
            CUfunction mfn = NULL;
            CHECK_CU(p_cuModuleGetFunction(&mfn, mmod, "mma_s8"));
            CUdeviceptr mo = 0;
            CHECK_CU(p_cuMemAlloc_v2(&mo, 1024));
            uint32_t mz[256] = { 0 };
            CHECK_CU(p_cuMemcpyHtoD_v2(mo, mz, 1024));
            void *mpar[1] = { &mo };
            CHECK_CU(p_cuLaunchKernel(mfn, 1, 1, 1, 64, 1, 1, 0, NULL,
                                      mpar, NULL));
            CHECK_CU(p_cuCtxSynchronize());
            uint32_t mgot[256] = { 0 };
            CHECK_CU(p_cuMemcpyDtoH_v2(mgot, mo, 1024));
            int mbad = 0;
            for (int i = 0; i < 256; i++) {
                if (mgot[i] != exp_s8[i]) {
                    if (mbad < 4)
                        printf("  mma_s8 mismatch[i=%d]: got %08x want %08x\n",
                               i, mgot[i], exp_s8[i]);
                    mbad++;
                }
            }
            if (mbad)
                failures += mbad;
            else
                printf("  driver mma_s8: mma.m16n8k32.s32.s8.s8.s32 (64 lanes) matches NVIDIA vectors\n");
            CHECK_CU(p_cuMemFree_v2(mo));
            CHECK_CU(p_cuModuleUnload(mmod));
            CHECK_CU(p_cuLinkDestroy(mls));
            free(mb);
        }
    }
    // Warp-level wmma.m16n16k16.f16->f32 (tests/ptx/wmma.ptx): with
    // A = I and B[k][c] = 16*k+c, C[r][c] = 16*r+c, the result is
    // D[r][c] = 2*(16*r+c) regardless of the target-dependent fragment
    // layout. Verified on the real driver (NVIDIA ptxas + hardware).
    {
        snprintf(path, sizeof(path), "%s/wmma.ptx", ACUDA_TEST_PTX_DIR);
        char *wb = read_file(path);
        if (!wb) {
            printf("  SKIP wmma: corpus absent\n");
        } else {
            static uint32_t win[768], wwant[768], wgot[768];
            memset(win, 0, sizeof(win));
            for (int r = 0; r < 16; r++)
                for (int c = 0; c < 16; c++) {
                    int ai = r * 16 + c;
                    uint32_t ah = (r == c) ? drv_f32_to_f16_bits(1.0f) : 0;
                    win[ai >> 1] |= ah << ((ai & 1) * 16);
                }
            for (int c = 0; c < 16; c++)
                for (int k = 0; k < 16; k++) {
                    int bi = c * 16 + k;
                    uint32_t bh = drv_f32_to_f16_bits((float)(k * 16 + c));
                    win[128 + (bi >> 1)] |= bh << ((bi & 1) * 16);
                }
            for (int r = 0; r < 16; r++)
                for (int c = 0; c < 16; c++) {
                    win[256 + r * 16 + c] =
                        drv_f32_bits((float)(r * 16 + c));
                    win[512 + r * 16 + c] =
                        drv_f32_bits((float)(2 * (r * 16 + c)));
                }
            memcpy(wwant, win, 512 * 4);
            memcpy(wwant + 512, win + 512, 256 * 4);
            CUlinkState wls = NULL;
            CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &wls));
            CHECK_CU(p_cuLinkAddData(wls, 1, wb, strlen(wb), "wmma",
                                     0, NULL, NULL));
            void *wcubin = NULL;
            size_t wcubin_size = 0;
            CHECK_CU(p_cuLinkComplete(wls, &wcubin, &wcubin_size));
            CUmodule wmod = NULL;
            CHECK_CU(p_cuModuleLoadData(&wmod, wcubin));
            CUfunction wfn = NULL;
            CHECK_CU(p_cuModuleGetFunction(&wfn, wmod, "wmma_mma"));
            CUdeviceptr wo = 0;
            CHECK_CU(p_cuMemAlloc_v2(&wo, sizeof(win)));
            CHECK_CU(p_cuMemcpyHtoD_v2(wo, win, sizeof(win)));
            void *wpar[1] = { &wo };
            CHECK_CU(p_cuLaunchKernel(wfn, 1, 1, 1, 32, 1, 1, 0, NULL,
                                      wpar, NULL));
            CHECK_CU(p_cuCtxSynchronize());
            CHECK_CU(p_cuMemcpyDtoH_v2(wgot, wo, sizeof(wgot)));
            int wbad = 0;
            for (int i = 0; i < 768; i++) {
                if (wgot[i] != wwant[i]) {
                    if (wbad < 4)
                        printf("  wmma mismatch[i=%d]: got %08x want %08x\n",
                               i, wgot[i], wwant[i]);
                    wbad++;
                }
            }
            if (wbad)
                failures += wbad;
            else
                printf("  driver wmma: m16n16k16 f16xf16->f32 (32 lanes) ok\n");
            CHECK_CU(p_cuMemFree_v2(wo));
            CHECK_CU(p_cuModuleUnload(wmod));
            CHECK_CU(p_cuLinkDestroy(wls));
            free(wb);
        }
    }
    // Sparse mma.sp m16n8k16 f16 (tests/ptx/mma_sp.ptx): 2:4 A with
    // metadata 0xD8D8D8D8; D = A*B + C. Layout mirrors the Vulkan test.
    {
        snprintf(path, sizeof(path), "%s/mma_sp.ptx", ACUDA_TEST_PTX_DIR);
        char *spb = read_file(path);
        if (!spb) {
            printf("  SKIP mma_sp: corpus absent\n");
        } else {
            static uint32_t spin[416], spwant[416], spgot[416];
            drv_build_sparse_mma(spin, spwant);
            CUlinkState sls = NULL;
            CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &sls));
            CHECK_CU(p_cuLinkAddData(sls, 1, spb, strlen(spb), "mma_sp",
                                     0, NULL, NULL));
            void *scubin = NULL;
            size_t scubin_size = 0;
            CHECK_CU(p_cuLinkComplete(sls, &scubin, &scubin_size));
            CUmodule smod = NULL;
            CHECK_CU(p_cuModuleLoadData(&smod, scubin));
            CUfunction sfn = NULL;
            CHECK_CU(p_cuModuleGetFunction(&sfn, smod, "mma_sp"));
            CUdeviceptr so = 0;
            CHECK_CU(p_cuMemAlloc_v2(&so, sizeof(spin)));
            CHECK_CU(p_cuMemcpyHtoD_v2(so, spin, sizeof(spin)));
            void *spar[1] = { &so };
            CHECK_CU(p_cuLaunchKernel(sfn, 1, 1, 1, 32, 1, 1, 0, NULL,
                                      spar, NULL));
            CHECK_CU(p_cuCtxSynchronize());
            CHECK_CU(p_cuMemcpyDtoH_v2(spgot, so, sizeof(spgot)));
            int sbad = 0;
            for (int i = 0; i < 416; i++) {
                if (spgot[i] != spwant[i]) {
                    if (sbad < 4)
                        printf("  mma_sp mismatch[i=%d]: got %08x want %08x\n",
                               i, spgot[i], spwant[i]);
                    sbad++;
                }
            }
            if (sbad)
                failures += sbad;
            else
                printf("  driver mma_sp: m16n8k16 f16 2:4 (32 lanes) ok\n");
            CHECK_CU(p_cuMemFree_v2(so));
            CHECK_CU(p_cuModuleUnload(smod));
            CHECK_CU(p_cuLinkDestroy(sls));
            free(spb);
        }
    }
    // Dense mma m16n8k8 f16/bf16 and m16n8k4 tf32 through the CUDA
    // driver path (same 352-word layout as the Vulkan tests).
    {
        static const struct {
            const char *file, *entry;
            int kind; // 0 = k8 f16, 1 = k8 bf16, 2 = k4 tf32
        } kc[3] = {
            { "mma_k8.ptx", "mma_k8", 0 },
            { "mma_k8_bf16.ptx", "mma_k8_bf16", 1 },
            { "mma_k4.ptx", "mma_k4", 2 },
        };
        for (int ci = 0; ci < 3; ci++) {
            snprintf(path, sizeof(path), "%s/%s", ACUDA_TEST_PTX_DIR,
                     kc[ci].file);
            char *kb = read_file(path);
            if (!kb) {
                printf("  SKIP %s: corpus absent\n", kc[ci].entry);
                continue;
            }
            static uint32_t kin[352], kwant[352], kgot[352];
            if (kc[ci].kind == 2)
                drv_build_k4(kin, kwant);
            else
                drv_build_k8(kin, kwant, kc[ci].kind);
            CUlinkState kls = NULL;
            CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &kls));
            CHECK_CU(p_cuLinkAddData(kls, 1, kb, strlen(kb), kc[ci].entry,
                                     0, NULL, NULL));
            void *kcubin = NULL;
            size_t kcubin_size = 0;
            CHECK_CU(p_cuLinkComplete(kls, &kcubin, &kcubin_size));
            CUmodule kmod = NULL;
            CHECK_CU(p_cuModuleLoadData(&kmod, kcubin));
            CUfunction kfn = NULL;
            CHECK_CU(p_cuModuleGetFunction(&kfn, kmod, kc[ci].entry));
            CUdeviceptr ko = 0;
            CHECK_CU(p_cuMemAlloc_v2(&ko, sizeof(kin)));
            CHECK_CU(p_cuMemcpyHtoD_v2(ko, kin, sizeof(kin)));
            void *kpar[1] = { &ko };
            CHECK_CU(p_cuLaunchKernel(kfn, 1, 1, 1, 32, 1, 1, 0, NULL,
                                      kpar, NULL));
            CHECK_CU(p_cuCtxSynchronize());
            CHECK_CU(p_cuMemcpyDtoH_v2(kgot, ko, sizeof(kgot)));
            int kbad = 0;
            for (int i = 0; i < 352; i++) {
                if (kgot[i] != kwant[i]) {
                    if (kbad < 4)
                        printf("  %s mismatch[i=%d]: got %08x want %08x\n",
                               kc[ci].entry, i, kgot[i], kwant[i]);
                    kbad++;
                }
            }
            if (kbad)
                failures += kbad;
            else
                printf("  driver %s: 352 words ok\n", kc[ci].entry);
            CHECK_CU(p_cuMemFree_v2(ko));
            CHECK_CU(p_cuModuleUnload(kmod));
            CHECK_CU(p_cuLinkDestroy(kls));
            free(kb);
        }
    }
    // Dense integer mma m16n8k32 s32 with s8/u8 multiplicands.
    {
        static const struct {
            const char *file, *entry;
            int asign, bsign;
        } ic[3] = {
            { "mma_u8.ptx", "mma_u8", 0, 0 },
            { "mma_s8u8.ptx", "mma_s8u8", 1, 0 },
            { "mma_u8s8.ptx", "mma_u8s8", 0, 1 },
        };
        for (int ci = 0; ci < 3; ci++) {
            snprintf(path, sizeof(path), "%s/%s", ACUDA_TEST_PTX_DIR,
                     ic[ci].file);
            char *ib = read_file(path);
            if (!ib) {
                printf("  SKIP %s: corpus absent\n", ic[ci].entry);
                continue;
            }
            static uint32_t iin[448], iwant[448], igot[448];
            drv_build_int8(iin, iwant, ic[ci].asign, ic[ci].bsign);
            CUlinkState ils = NULL;
            CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &ils));
            CHECK_CU(p_cuLinkAddData(ils, 1, ib, strlen(ib), ic[ci].entry,
                                     0, NULL, NULL));
            void *icubin = NULL;
            size_t icubin_size = 0;
            CHECK_CU(p_cuLinkComplete(ils, &icubin, &icubin_size));
            CUmodule imod = NULL;
            CHECK_CU(p_cuModuleLoadData(&imod, icubin));
            CUfunction ifn = NULL;
            CHECK_CU(p_cuModuleGetFunction(&ifn, imod, ic[ci].entry));
            CUdeviceptr io = 0;
            CHECK_CU(p_cuMemAlloc_v2(&io, sizeof(iin)));
            CHECK_CU(p_cuMemcpyHtoD_v2(io, iin, sizeof(iin)));
            void *ipar[1] = { &io };
            CHECK_CU(p_cuLaunchKernel(ifn, 1, 1, 1, 32, 1, 1, 0, NULL,
                                      ipar, NULL));
            CHECK_CU(p_cuCtxSynchronize());
            CHECK_CU(p_cuMemcpyDtoH_v2(igot, io, sizeof(igot)));
            int ibad = 0;
            for (int i = 0; i < 448; i++) {
                if (igot[i] != iwant[i]) {
                    if (ibad < 4)
                        printf("  %s mismatch[i=%d]: got %08x want %08x\n",
                               ic[ci].entry, i, igot[i], iwant[i]);
                    ibad++;
                }
            }
            if (ibad)
                failures += ibad;
            else
                printf("  driver %s: 448 words ok\n", ic[ci].entry);
            CHECK_CU(p_cuMemFree_v2(io));
            CHECK_CU(p_cuModuleUnload(imod));
            CHECK_CU(p_cuLinkDestroy(ils));
            free(ib);
        }
    }
    // Dense mma with f16 accumulators (m16n8k16 and m16n8k8).
    {
        static const struct {
            const char *file, *entry;
            int k8, nwords;
        } fc[2] = {
            { "mma_f16acc.ptx", "mma_f16acc", 0, 320 },
            { "mma_k8_f16acc.ptx", "mma_k8_f16acc", 1, 224 },
        };
        for (int ci = 0; ci < 2; ci++) {
            snprintf(path, sizeof(path), "%s/%s", ACUDA_TEST_PTX_DIR,
                     fc[ci].file);
            char *fb = read_file(path);
            if (!fb) {
                printf("  SKIP %s: corpus absent\n", fc[ci].entry);
                continue;
            }
            static uint32_t fin[320], fwant[320], fgot[320];
            drv_build_f16acc(fin, fwant, fc[ci].k8);
            CUlinkState fls = NULL;
            CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &fls));
            CHECK_CU(p_cuLinkAddData(fls, 1, fb, strlen(fb), fc[ci].entry,
                                     0, NULL, NULL));
            void *fcubin = NULL;
            size_t fcubin_size = 0;
            CHECK_CU(p_cuLinkComplete(fls, &fcubin, &fcubin_size));
            CUmodule fmod = NULL;
            CHECK_CU(p_cuModuleLoadData(&fmod, fcubin));
            CUfunction ffn = NULL;
            CHECK_CU(p_cuModuleGetFunction(&ffn, fmod, fc[ci].entry));
            CUdeviceptr fo = 0;
            CHECK_CU(p_cuMemAlloc_v2(&fo,
                                     (size_t)fc[ci].nwords * 4));
            CHECK_CU(p_cuMemcpyHtoD_v2(fo, fin,
                                       (size_t)fc[ci].nwords * 4));
            void *fpar[1] = { &fo };
            CHECK_CU(p_cuLaunchKernel(ffn, 1, 1, 1, 32, 1, 1, 0, NULL,
                                      fpar, NULL));
            CHECK_CU(p_cuCtxSynchronize());
            CHECK_CU(p_cuMemcpyDtoH_v2(fgot, fo,
                                       (size_t)fc[ci].nwords * 4));
            int fbad = 0;
            for (int i = 0; i < fc[ci].nwords; i++) {
                if (fgot[i] != fwant[i]) {
                    if (fbad < 4)
                        printf("  %s mismatch[i=%d]: got %08x want %08x\n",
                               fc[ci].entry, i, fgot[i], fwant[i]);
                    fbad++;
                }
            }
            if (fbad)
                failures += fbad;
            else
                printf("  driver %s: %d words ok\n", fc[ci].entry,
                       fc[ci].nwords);
            CHECK_CU(p_cuMemFree_v2(fo));
            CHECK_CU(p_cuModuleUnload(fmod));
            CHECK_CU(p_cuLinkDestroy(fls));
            free(fb);
        }
    }
    // Dense mma m8n8k4 f64.
    {
        snprintf(path, sizeof(path), "%s/mma_f64.ptx", ACUDA_TEST_PTX_DIR);
        char *db = read_file(path);
        if (!db) {
            printf("  SKIP mma_f64: corpus absent\n");
        } else {
            static uint32_t din[384], dwant[384], dgot[384];
            drv_build_f64(din, dwant);
            CUlinkState dls = NULL;
            CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &dls));
            CHECK_CU(p_cuLinkAddData(dls, 1, db, strlen(db), "mma_f64",
                                     0, NULL, NULL));
            void *dcubin = NULL;
            size_t dcubin_size = 0;
            CHECK_CU(p_cuLinkComplete(dls, &dcubin, &dcubin_size));
            CUmodule dmod = NULL;
            CHECK_CU(p_cuModuleLoadData(&dmod, dcubin));
            CUfunction dfn = NULL;
            CHECK_CU(p_cuModuleGetFunction(&dfn, dmod, "mma_f64"));
            CUdeviceptr dov = 0;
            CHECK_CU(p_cuMemAlloc_v2(&dov, sizeof(din)));
            CHECK_CU(p_cuMemcpyHtoD_v2(dov, din, sizeof(din)));
            void *dpar[1] = { &dov };
            CHECK_CU(p_cuLaunchKernel(dfn, 1, 1, 1, 32, 1, 1, 0, NULL,
                                      dpar, NULL));
            CHECK_CU(p_cuCtxSynchronize());
            CHECK_CU(p_cuMemcpyDtoH_v2(dgot, dov, sizeof(dgot)));
            int dbad = 0;
            for (int i = 0; i < 384; i++) {
                if (dgot[i] != dwant[i]) {
                    if (dbad < 4)
                        printf("  mma_f64 mismatch[i=%d]: got %08x want %08x\n",
                               i, dgot[i], dwant[i]);
                    dbad++;
                }
            }
            if (dbad)
                failures += dbad;
            else
                printf("  driver mma_f64: m8n8k4 f64 (32 lanes) ok\n");
            CHECK_CU(p_cuMemFree_v2(dov));
            CHECK_CU(p_cuModuleUnload(dmod));
            CHECK_CU(p_cuLinkDestroy(dls));
            free(db);
        }
    }
    // Warp-level WMMA m16n16k16 bf16 -> f32.
    {
        snprintf(path, sizeof(path), "%s/wmma_bf16.ptx",
                 ACUDA_TEST_PTX_DIR);
        char *bb = read_file(path);
        if (!bb) {
            printf("  SKIP wmma_bf16: corpus absent\n");
        } else {
            static uint32_t bin[768], bwant[768], bgot[768];
            drv_build_wmma_bf16(bin, bwant);
            CUlinkState bls = NULL;
            CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &bls));
            CHECK_CU(p_cuLinkAddData(bls, 1, bb, strlen(bb), "wmma_bf16",
                                     0, NULL, NULL));
            void *bcubin = NULL;
            size_t bcubin_size = 0;
            CHECK_CU(p_cuLinkComplete(bls, &bcubin, &bcubin_size));
            CUmodule bmod = NULL;
            CHECK_CU(p_cuModuleLoadData(&bmod, bcubin));
            CUfunction bfn = NULL;
            CHECK_CU(p_cuModuleGetFunction(&bfn, bmod, "wmma_bf16"));
            CUdeviceptr bo = 0;
            CHECK_CU(p_cuMemAlloc_v2(&bo, sizeof(bin)));
            CHECK_CU(p_cuMemcpyHtoD_v2(bo, bin, sizeof(bin)));
            void *bpar[1] = { &bo };
            CHECK_CU(p_cuLaunchKernel(bfn, 1, 1, 1, 32, 1, 1, 0, NULL,
                                      bpar, NULL));
            CHECK_CU(p_cuCtxSynchronize());
            CHECK_CU(p_cuMemcpyDtoH_v2(bgot, bo, sizeof(bgot)));
            int bbad = 0;
            for (int i = 0; i < 768; i++) {
                if (bgot[i] != bwant[i]) {
                    if (bbad < 4)
                        printf("  wmma_bf16 mismatch[i=%d]: got %08x want %08x\n",
                               i, bgot[i], bwant[i]);
                    bbad++;
                }
            }
            if (bbad)
                failures += bbad;
            else
                printf("  driver wmma_bf16: 768 words ok\n");
            CHECK_CU(p_cuMemFree_v2(bo));
            CHECK_CU(p_cuModuleUnload(bmod));
            CHECK_CU(p_cuLinkDestroy(bls));
            free(bb);
        }
    }
    // Warp-level WMMA m16n16k16 f16 -> f16 (f16 accumulate).
    {
        snprintf(path, sizeof(path), "%s/wmma_f16acc.ptx",
                 ACUDA_TEST_PTX_DIR);
        char *fb2 = read_file(path);
        if (!fb2) {
            printf("  SKIP wmma_f16acc: corpus absent\n");
        } else {
            static uint32_t fin2[512], fwant2[512], fgot2[512];
            drv_build_wmma_f16acc(fin2, fwant2);
            CUlinkState fls2 = NULL;
            CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &fls2));
            CHECK_CU(p_cuLinkAddData(fls2, 1, fb2, strlen(fb2),
                                     "wmma_f16acc", 0, NULL, NULL));
            void *fcubin2 = NULL;
            size_t fcubin2_size = 0;
            CHECK_CU(p_cuLinkComplete(fls2, &fcubin2, &fcubin2_size));
            CUmodule fmod2 = NULL;
            CHECK_CU(p_cuModuleLoadData(&fmod2, fcubin2));
            CUfunction ffn2 = NULL;
            CHECK_CU(p_cuModuleGetFunction(&ffn2, fmod2,
                                           "wmma_f16acc"));
            CUdeviceptr fo2 = 0;
            CHECK_CU(p_cuMemAlloc_v2(&fo2, sizeof(fin2)));
            CHECK_CU(p_cuMemcpyHtoD_v2(fo2, fin2, sizeof(fin2)));
            void *fpar2[1] = { &fo2 };
            CHECK_CU(p_cuLaunchKernel(ffn2, 1, 1, 1, 32, 1, 1, 0, NULL,
                                      fpar2, NULL));
            CHECK_CU(p_cuCtxSynchronize());
            CHECK_CU(p_cuMemcpyDtoH_v2(fgot2, fo2, sizeof(fgot2)));
            int fbad2 = 0;
            for (int i = 0; i < 512; i++) {
                if (fgot2[i] != fwant2[i]) {
                    if (fbad2 < 4)
                        printf("  wmma_f16acc mismatch[i=%d]: got %08x want %08x\n",
                               i, fgot2[i], fwant2[i]);
                    fbad2++;
                }
            }
            if (fbad2)
                failures += fbad2;
            else
                printf("  driver wmma_f16acc: 512 words ok\n");
            CHECK_CU(p_cuMemFree_v2(fo2));
            CHECK_CU(p_cuModuleUnload(fmod2));
            CHECK_CU(p_cuLinkDestroy(fls2));
            free(fb2);
        }
    }
    // Warp-level WMMA m8n32k16 and m32n8k16 (f16 -> f32).
    {
        static const struct {
            const char *file, *entry;
            int rows, cols, kdim;
        } wc[2] = {
            { "wmma_m8n32k16.ptx", "wmma_m8n32k16", 8, 32, 16 },
            { "wmma_m32n8k16.ptx", "wmma_m32n8k16", 32, 8, 16 },
        };
        for (int ci = 0; ci < 2; ci++) {
            snprintf(path, sizeof(path), "%s/%s", ACUDA_TEST_PTX_DIR,
                     wc[ci].file);
            char *wb2 = read_file(path);
            if (!wb2) {
                printf("  SKIP %s: corpus absent\n", wc[ci].entry);
                continue;
            }
            static uint32_t win2[832], wwant2[832], wgot2[832];
            drv_build_wmma_shape(win2, wwant2, wc[ci].rows,
                                 wc[ci].cols, wc[ci].kdim);
            CUlinkState wls2 = NULL;
            CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &wls2));
            CHECK_CU(p_cuLinkAddData(wls2, 1, wb2, strlen(wb2),
                                     wc[ci].entry, 0, NULL, NULL));
            void *wcubin2 = NULL;
            size_t wcubin2_size = 0;
            CHECK_CU(p_cuLinkComplete(wls2, &wcubin2, &wcubin2_size));
            CUmodule wmod2 = NULL;
            CHECK_CU(p_cuModuleLoadData(&wmod2, wcubin2));
            CUfunction wfn2 = NULL;
            CHECK_CU(p_cuModuleGetFunction(&wfn2, wmod2, wc[ci].entry));
            CUdeviceptr wo2 = 0;
            CHECK_CU(p_cuMemAlloc_v2(&wo2, sizeof(win2)));
            CHECK_CU(p_cuMemcpyHtoD_v2(wo2, win2, sizeof(win2)));
            void *wpar2[1] = { &wo2 };
            CHECK_CU(p_cuLaunchKernel(wfn2, 1, 1, 1, 32, 1, 1, 0, NULL,
                                      wpar2, NULL));
            CHECK_CU(p_cuCtxSynchronize());
            CHECK_CU(p_cuMemcpyDtoH_v2(wgot2, wo2, sizeof(wgot2)));
            int wbad2 = 0;
            for (int i = 0; i < 832; i++) {
                if (wgot2[i] != wwant2[i]) {
                    if (wbad2 < 4)
                        printf("  %s mismatch[i=%d]: got %08x want %08x\n",
                               wc[ci].entry, i, wgot2[i], wwant2[i]);
                    wbad2++;
                }
            }
            if (wbad2)
                failures += wbad2;
            else
                printf("  driver %s: 832 words ok\n", wc[ci].entry);
            CHECK_CU(p_cuMemFree_v2(wo2));
            CHECK_CU(p_cuModuleUnload(wmod2));
            CHECK_CU(p_cuLinkDestroy(wls2));
            free(wb2);
        }
    }
    // Dense mma m8n8k4 f16.
    {
        snprintf(path, sizeof(path), "%s/mma_m8n8k4_f16.ptx",
                 ACUDA_TEST_PTX_DIR);
        char *mb = read_file(path);
        if (!mb) {
            printf("  SKIP mma_m8n8k4_f16: corpus absent\n");
        } else {
            static uint32_t min[640], mwant[640], mgot[640];
            drv_build_m8f16(min, mwant);
            CUlinkState mls = NULL;
            CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &mls));
            CHECK_CU(p_cuLinkAddData(mls, 1, mb, strlen(mb),
                                     "mma_m8n8k4_f16", 0, NULL, NULL));
            void *mcubin = NULL;
            size_t mcubin_size = 0;
            CHECK_CU(p_cuLinkComplete(mls, &mcubin, &mcubin_size));
            CUmodule mmod = NULL;
            CHECK_CU(p_cuModuleLoadData(&mmod, mcubin));
            CUfunction mfn = NULL;
            CHECK_CU(p_cuModuleGetFunction(&mfn, mmod,
                                           "mma_m8n8k4_f16"));
            CUdeviceptr mo = 0;
            CHECK_CU(p_cuMemAlloc_v2(&mo, sizeof(min)));
            CHECK_CU(p_cuMemcpyHtoD_v2(mo, min, sizeof(min)));
            void *mpar[1] = { &mo };
            CHECK_CU(p_cuLaunchKernel(mfn, 1, 1, 1, 32, 1, 1, 0, NULL,
                                      mpar, NULL));
            CHECK_CU(p_cuCtxSynchronize());
            CHECK_CU(p_cuMemcpyDtoH_v2(mgot, mo, sizeof(mgot)));
            int mbad = 0;
            for (int i = 0; i < 640; i++) {
                if (mgot[i] != mwant[i]) {
                    if (mbad < 4)
                        printf("  mma_m8n8k4_f16 mismatch[i=%d]: got %08x want %08x\n",
                               i, mgot[i], mwant[i]);
                    mbad++;
                }
            }
            if (mbad)
                failures += mbad;
            else
                printf("  driver mma_m8n8k4_f16: 640 words ok\n");
            CHECK_CU(p_cuMemFree_v2(mo));
            CHECK_CU(p_cuModuleUnload(mmod));
            CHECK_CU(p_cuLinkDestroy(mls));
            free(mb);
        }
    }
    // Sparse mma.sp m16n8k32 f16.
    {
        snprintf(path, sizeof(path), "%s/mma_sp_k32.ptx",
                 ACUDA_TEST_PTX_DIR);
        char *sb = read_file(path);
        if (!sb) {
            printf("  SKIP mma_sp_k32: corpus absent\n");
        } else {
            static uint32_t sin[544], swant[544], sgot[544];
            drv_build_sp_k32(sin, swant);
            CUlinkState sls = NULL;
            CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &sls));
            CHECK_CU(p_cuLinkAddData(sls, 1, sb, strlen(sb),
                                     "mma_sp_k32", 0, NULL, NULL));
            void *subin = NULL;
            size_t subin_size = 0;
            CHECK_CU(p_cuLinkComplete(sls, &subin, &subin_size));
            CUmodule smod = NULL;
            CHECK_CU(p_cuModuleLoadData(&smod, subin));
            CUfunction sfn = NULL;
            CHECK_CU(p_cuModuleGetFunction(&sfn, smod, "mma_sp_k32"));
            CUdeviceptr so = 0;
            CHECK_CU(p_cuMemAlloc_v2(&so, sizeof(sin)));
            CHECK_CU(p_cuMemcpyHtoD_v2(so, sin, sizeof(sin)));
            void *spar[1] = { &so };
            CHECK_CU(p_cuLaunchKernel(sfn, 1, 1, 1, 32, 1, 1, 0, NULL,
                                      spar, NULL));
            CHECK_CU(p_cuCtxSynchronize());
            CHECK_CU(p_cuMemcpyDtoH_v2(sgot, so, sizeof(sgot)));
            int sbad = 0;
            for (int i = 0; i < 544; i++) {
                if (sgot[i] != swant[i]) {
                    if (sbad < 4)
                        printf("  mma_sp_k32 mismatch[i=%d]: got %08x want %08x\n",
                               i, sgot[i], swant[i]);
                    sbad++;
                }
            }
            if (sbad)
                failures += sbad;
            else
                printf("  driver mma_sp_k32: 544 words ok\n");
            CHECK_CU(p_cuMemFree_v2(so));
            CHECK_CU(p_cuModuleUnload(smod));
            CHECK_CU(p_cuLinkDestroy(sls));
            free(sb);
        }
    }
    // Warp-level WMMA m16n16k8 tf32.
    {
        snprintf(path, sizeof(path), "%s/wmma_tf32.ptx",
                 ACUDA_TEST_PTX_DIR);
        char *tb = read_file(path);
        if (!tb) {
            printf("  SKIP wmma_tf32: corpus absent\n");
        } else {
            static uint32_t tin[768], twant[768], tgot[768];
            drv_build_wmma_tf32(tin, twant);
            CUlinkState tls = NULL;
            CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &tls));
            CHECK_CU(p_cuLinkAddData(tls, 1, tb, strlen(tb),
                                     "wmma_tf32", 0, NULL, NULL));
            void *tcubin = NULL;
            size_t tcubin_size = 0;
            CHECK_CU(p_cuLinkComplete(tls, &tcubin, &tcubin_size));
            CUmodule tmod = NULL;
            CHECK_CU(p_cuModuleLoadData(&tmod, tcubin));
            CUfunction tfn = NULL;
            CHECK_CU(p_cuModuleGetFunction(&tfn, tmod, "wmma_tf32"));
            CUdeviceptr to = 0;
            CHECK_CU(p_cuMemAlloc_v2(&to, sizeof(tin)));
            CHECK_CU(p_cuMemcpyHtoD_v2(to, tin, sizeof(tin)));
            void *tpar[1] = { &to };
            CHECK_CU(p_cuLaunchKernel(tfn, 1, 1, 1, 32, 1, 1, 0, NULL,
                                      tpar, NULL));
            CHECK_CU(p_cuCtxSynchronize());
            CHECK_CU(p_cuMemcpyDtoH_v2(tgot, to, sizeof(tgot)));
            int tbad = 0;
            for (int i = 0; i < 768; i++) {
                if (tgot[i] != twant[i]) {
                    if (tbad < 4)
                        printf("  wmma_tf32 mismatch[i=%d]: got %08x want %08x\n",
                               i, tgot[i], twant[i]);
                    tbad++;
                }
            }
            if (tbad)
                failures += tbad;
            else
                printf("  driver wmma_tf32: 768 words ok\n");
            CHECK_CU(p_cuMemFree_v2(to));
            CHECK_CU(p_cuModuleUnload(tmod));
            CHECK_CU(p_cuLinkDestroy(tls));
            free(tb);
        }
    }
    // Dense m16n8k4/k8/k16 f64 (target-gated to sm_90).
    {
        static const struct {
            const char *file, *entry;
            int K, nwords;
        } fc2[3] = {
            { "mma_f64_k4.ptx", "mma_f64_k4", 4, 704 },
            { "mma_f64_k8.ptx", "mma_f64_k8", 8, 896 },
            { "mma_f64_k16.ptx", "mma_f64_k16", 16, 1280 },
        };
        for (int ci = 0; ci < 3; ci++) {
            snprintf(path, sizeof(path), "%s/%s", ACUDA_TEST_PTX_DIR,
                     fc2[ci].file);
            char *fb = read_file(path);
            if (!fb) {
                printf("  SKIP %s: corpus absent\n", fc2[ci].entry);
                continue;
            }
            static uint32_t fin[1280], fwant[1280], fgot[1280];
            drv_build_f64_k(fin, fwant, fc2[ci].K);
            CUlinkState fls = NULL;
            CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &fls));
            CHECK_CU(p_cuLinkAddData(fls, 1, fb, strlen(fb),
                                     fc2[ci].entry, 0, NULL, NULL));
            void *fcubin = NULL;
            size_t fcubin_size = 0;
            CHECK_CU(p_cuLinkComplete(fls, &fcubin, &fcubin_size));
            CUmodule fmod = NULL;
            CHECK_CU(p_cuModuleLoadData(&fmod, fcubin));
            CUfunction ffn = NULL;
            CHECK_CU(p_cuModuleGetFunction(&ffn, fmod, fc2[ci].entry));
            CUdeviceptr fo = 0;
            CHECK_CU(p_cuMemAlloc_v2(&fo,
                                     (size_t)fc2[ci].nwords * 4));
            CHECK_CU(p_cuMemcpyHtoD_v2(fo, fin,
                                       (size_t)fc2[ci].nwords * 4));
            void *fpar[1] = { &fo };
            CHECK_CU(p_cuLaunchKernel(ffn, 1, 1, 1, 32, 1, 1, 0, NULL,
                                      fpar, NULL));
            CHECK_CU(p_cuCtxSynchronize());
            CHECK_CU(p_cuMemcpyDtoH_v2(fgot, fo,
                                       (size_t)fc2[ci].nwords * 4));
            int fbad = 0;
            for (int i = 0; i < fc2[ci].nwords; i++) {
                if (fgot[i] != fwant[i]) {
                    if (fbad < 4)
                        printf("  %s mismatch[i=%d]: got %08x want %08x\n",
                               fc2[ci].entry, i, fgot[i], fwant[i]);
                    fbad++;
                }
            }
            if (fbad)
                failures += fbad;
            else
                printf("  driver %s: %d words ok\n", fc2[ci].entry,
                       fc2[ci].nwords);
            CHECK_CU(p_cuMemFree_v2(fo));
            CHECK_CU(p_cuModuleUnload(fmod));
            CHECK_CU(p_cuLinkDestroy(fls));
            free(fb);
        }
    }
    // isspacep: shared/global/local/param predicates.
    {
        snprintf(path, sizeof(path), "%s/isspacep.ptx", ACUDA_TEST_PTX_DIR);
        char *isb = read_file(path);
        if (!isb) {
            printf("  SKIP isspacep: corpus absent\n");
        } else {
            static const uint32_t isin[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
            static const uint32_t iswant[8] = { 1, 1, 1, 0, 0, 1, 0, 0 };
            uint32_t isgot[8] = { 0 };
            CUlinkState isls = NULL;
            CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &isls));
            CHECK_CU(p_cuLinkAddData(isls, 1, isb, strlen(isb),
                                     "isspacep_test", 0, NULL, NULL));
            void *iscubin = NULL;
            size_t iscubin_size = 0;
            CHECK_CU(p_cuLinkComplete(isls, &iscubin, &iscubin_size));
            CUmodule ismod = NULL;
            CHECK_CU(p_cuModuleLoadData(&ismod, iscubin));
            CUfunction isfn = NULL;
            CHECK_CU(p_cuModuleGetFunction(&isfn, ismod,
                                           "isspacep_test"));
            CUdeviceptr iso = 0;
            CHECK_CU(p_cuMemAlloc_v2(&iso, sizeof(isin)));
            CHECK_CU(p_cuMemcpyHtoD_v2(iso, isin, sizeof(isin)));
            void *ispar[1] = { &iso };
            CHECK_CU(p_cuLaunchKernel(isfn, 1, 1, 1, 1, 1, 1, 0, NULL,
                                      ispar, NULL));
            CHECK_CU(p_cuCtxSynchronize());
            CHECK_CU(p_cuMemcpyDtoH_v2(isgot, iso, sizeof(isgot)));
            int isbad = 0;
            for (int i = 0; i < 8; i++)
                if (isgot[i] != iswant[i]) {
                    printf("  isspacep mismatch[%d]: got %u want %u\n",
                           i, isgot[i], iswant[i]);
                    isbad++;
                }
            if (isbad)
                failures += isbad;
            else
                printf("  driver isspacep: 8 words ok\n");
            CHECK_CU(p_cuMemFree_v2(iso));
            CHECK_CU(p_cuModuleUnload(ismod));
            CHECK_CU(p_cuLinkDestroy(isls));
            free(isb);
        }
    }
    // Half-precision set/setp comparison matrix.
    {
        snprintf(path, sizeof(path), "%s/half_cmp.ptx", ACUDA_TEST_PTX_DIR);
        char *hcb = read_file(path);
        if (!hcb) {
            printf("  SKIP half_cmp: corpus absent\n");
        } else {
            static const uint32_t hcin[11] = { 0 };
            static const uint32_t hcwant[11] = {
                0xffffffffu, 0, 0xffffffffu, 0x0000ffffu, 0x3c003c00u,
                1, 1, 0, 0, 0, 1
            };
            uint32_t hcgot[11] = { 0 };
            CUlinkState hcls = NULL;
            CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &hcls));
            CHECK_CU(p_cuLinkAddData(hcls, 1, hcb, strlen(hcb),
                                     "half_cmp", 0, NULL, NULL));
            void *hccubin = NULL;
            size_t hccubin_size = 0;
            CHECK_CU(p_cuLinkComplete(hcls, &hccubin, &hccubin_size));
            CUmodule hcmod = NULL;
            CHECK_CU(p_cuModuleLoadData(&hcmod, hccubin));
            CUfunction hcfn = NULL;
            CHECK_CU(p_cuModuleGetFunction(&hcfn, hcmod, "half_cmp"));
            CUdeviceptr hco = 0;
            CHECK_CU(p_cuMemAlloc_v2(&hco, sizeof(hcin)));
            CHECK_CU(p_cuMemcpyHtoD_v2(hco, hcin, sizeof(hcin)));
            void *hcpar[1] = { &hco };
            CHECK_CU(p_cuLaunchKernel(hcfn, 1, 1, 1, 1, 1, 1, 0, NULL,
                                      hcpar, NULL));
            CHECK_CU(p_cuCtxSynchronize());
            CHECK_CU(p_cuMemcpyDtoH_v2(hcgot, hco, sizeof(hcgot)));
            int hcbad = 0;
            for (int i = 0; i < 11; i++)
                if (hcgot[i] != hcwant[i]) {
                    printf("  half_cmp mismatch[%d]: got %08x want %08x\n",
                           i, hcgot[i], hcwant[i]);
                    hcbad++;
                }
            if (hcbad)
                failures += hcbad;
            else
                printf("  driver half_cmp: 11 words ok\n");
            CHECK_CU(p_cuMemFree_v2(hco));
            CHECK_CU(p_cuModuleUnload(hcmod));
            CHECK_CU(p_cuLinkDestroy(hcls));
            free(hcb);
        }
    }
    // cvt int<->float corners.
    {
        snprintf(path, sizeof(path), "%s/cvt_int_float.ptx",
                 ACUDA_TEST_PTX_DIR);
        char *cfb = read_file(path);
        if (!cfb) {
            printf("  SKIP cvt_int_float: corpus absent\n");
        } else {
            static const uint32_t cfin[74] = { 0 };
            static const uint32_t cfwant[74] = {
                0xc0400000u, 0, 0, 0xc01c0000u, 0x0000c200u,
                0xfffffffdu, 4, 0, 4, 0, 5, 0,
                4, 0, 0xfffffffbu, 0xffffffffu, 5, 0, 0, 0x40000000u
            };
            uint32_t cfgot[74] = { 0 };
            CUlinkState cfls = NULL;
            CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &cfls));
            CHECK_CU(p_cuLinkAddData(cfls, 1, cfb, strlen(cfb),
                                     "cvt_int_float", 0, NULL, NULL));
            void *cfcubin = NULL;
            size_t cfcubin_size = 0;
            CHECK_CU(p_cuLinkComplete(cfls, &cfcubin, &cfcubin_size));
            CUmodule cfmod = NULL;
            CHECK_CU(p_cuModuleLoadData(&cfmod, cfcubin));
            CUfunction cffn = NULL;
            CHECK_CU(p_cuModuleGetFunction(&cffn, cfmod,
                                           "cvt_int_float"));
            CUdeviceptr cfo = 0;
            CHECK_CU(p_cuMemAlloc_v2(&cfo, sizeof(cfin)));
            CHECK_CU(p_cuMemcpyHtoD_v2(cfo, cfin, sizeof(cfin)));
            void *cfpar[1] = { &cfo };
            CHECK_CU(p_cuLaunchKernel(cffn, 1, 1, 1, 1, 1, 1, 0, NULL,
                                      cfpar, NULL));
            CHECK_CU(p_cuCtxSynchronize());
            CHECK_CU(p_cuMemcpyDtoH_v2(cfgot, cfo, sizeof(cfgot)));
            int cfbd = 0;
            for (int i = 0; i < 74; i++)
                if (cfgot[i] != cfwant[i]) {
                    printf("  cvt_int_float mismatch[%d]: got %08x want %08x\n",
                           i, cfgot[i], cfwant[i]);
                    cfbd++;
                }
            if (cfbd)
                failures += cfbd;
            else
                printf("  driver cvt_int_float: 74 words ok\n");
            CHECK_CU(p_cuMemFree_v2(cfo));
            CHECK_CU(p_cuModuleUnload(cfmod));
            CHECK_CU(p_cuLinkDestroy(cfls));
            free(cfb);
        }
    }
    // Narrow 16-bit ALU wrap: add.u16 and mul.lo.u16 must truncate to 16
    // bits (the s8-mma pack functions depend on it). Lane tid writes two
    // u32s at byte tid*24.
    {
        const char *narsrc =
            ".version 6.5\n.target sm_60\n.address_size 64\n"
            ".visible .entry nar(.param .u64 o)\n"
            "{\n"
            "    .reg .u64 a; .reg .u64 x; .reg .u32 t;\n"
            "    .reg .u16 u; .reg .u16 v; .reg .u16 t16;\n"
            "    .reg .u32 p, q;\n"
            "    ld.param.u64 a, [o];\n"
            "    mov.u32 t, %tid.x;\n"
            "    cvt.u16.u32 t16, t;\n"
            "    add.u16 u, t16, 65535;\n"
            "    mul.lo.u16 v, t16, 257;\n"
            "    cvt.u64.u32 x, t;\n"
            "    mul.lo.u64 x, x, 24;\n"
            "    add.u64 a, a, x;\n"
            "    cvt.u32.u16 p, u;\n"
            "    cvt.u32.u16 q, v;\n"
            "    st.u32 [a], p;\n"
            "    st.u32 [a+8], q;\n"
            "    ret;\n"
            "}\n";
        char *narsrc_copy = strdup(narsrc);
        if (!narsrc_copy)
            return 1;
        CUlinkState nls2 = NULL;
        CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &nls2));
        CHECK_CU(p_cuLinkAddData(nls2, 1, narsrc_copy, strlen(narsrc_copy),
                                 "nar", 0, NULL, NULL));
        void *ncub2 = NULL;
        size_t ncub2_size = 0;
        CHECK_CU(p_cuLinkComplete(nls2, &ncub2, &ncub2_size));
        CUmodule nmod2 = NULL;
        CHECK_CU(p_cuModuleLoadData(&nmod2, ncub2));
        CUfunction nfn2 = NULL;
        CHECK_CU(p_cuModuleGetFunction(&nfn2, nmod2, "nar"));
        CUdeviceptr no = 0;
        CHECK_CU(p_cuMemAlloc_v2(&no, 24 * 64));
        uint32_t nz[24 * 16] = { 0 };
        CHECK_CU(p_cuMemcpyHtoD_v2(no, nz, 24 * 64));
        void *npar[1] = { &no };
        CHECK_CU(p_cuLaunchKernel(nfn2, 1, 1, 1, 64, 1, 1, 0, NULL, npar,
                                  NULL));
        CHECK_CU(p_cuCtxSynchronize());
        uint32_t nres[24 * 16] = { 0 };
        CHECK_CU(p_cuMemcpyDtoH_v2(nres, no, 24 * 64));
        int nbad = 0;
        for (int i = 0; i < 64; i++) {
            uint32_t wu = (uint32_t)((i + 65535) & 0xFFFFu);
            uint32_t wv = (uint32_t)(((unsigned)i * 257u) & 0xFFFFu);
            if (nres[i * 6] != wu) {
                if (nbad < 4)
                    printf("  add.u16 mismatch[i=%d]: got %u want %u\n", i,
                           nres[i * 6], wu);
                nbad++;
            }
            if (nres[i * 6 + 2] != wv) {
                if (nbad < 4)
                    printf("  mul.lo.u16 mismatch[i=%d]: got %u want %u\n", i,
                           nres[i * 6 + 2], wv);
                nbad++;
            }
        }
        if (nbad)
            failures += nbad;
        else
            printf("  driver narrow: add.u16 / mul.lo.u16 wrap to 16 bits (64 lanes)\n");
        CHECK_CU(p_cuMemFree_v2(no));
        CHECK_CU(p_cuModuleUnload(nmod2));
        CHECK_CU(p_cuLinkDestroy(nls2));
        free(narsrc_copy);
    }
    // Vector call ABI (tests/zluda-ptx/vector.ptx): impl() adds the two
    // components and returns a .v2.u32 whose halves are both the sum.
    {
        snprintf(path, sizeof(path), "%s/vector_call.ptx", ACUDA_TEST_PTX_DIR);
        char *vb = read_file(path);
        if (!vb) {
            printf("  SKIP vector_call: corpus absent\n");
        } else {
            CUlinkState vls = NULL;
            CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &vls));
            CHECK_CU(p_cuLinkAddData(vls, 1, vb, strlen(vb), "vector_call", 0,
                                     NULL, NULL));
            void *vcub = NULL;
            size_t vcub_size = 0;
            CHECK_CU(p_cuLinkComplete(vls, &vcub, &vcub_size));
            CUmodule vmod = NULL;
            CHECK_CU(p_cuModuleLoadData(&vmod, vcub));
            CUfunction vfn = NULL;
            CHECK_CU(p_cuModuleGetFunction(&vfn, vmod, "vector_call"));
            CUdeviceptr vi = 0, vo = 0;
            CHECK_CU(p_cuMemAlloc_v2(&vi, 64));
            CHECK_CU(p_cuMemAlloc_v2(&vo, 64));
            uint32_t hin[4] = { 11u, 31u, 0u, 0u };
            uint32_t hout[4] = { 0 };
            CHECK_CU(p_cuMemcpyHtoD_v2(vi, hin, sizeof hin));
            CHECK_CU(p_cuMemcpyHtoD_v2(vo, hout, sizeof hout));
            void *vpar[2] = { &vi, &vo };
            CHECK_CU(p_cuLaunchKernel(vfn, 1, 1, 1, 1, 1, 1, 0, NULL, vpar,
                                      NULL));
            CHECK_CU(p_cuCtxSynchronize());
            CHECK_CU(p_cuMemcpyDtoH_v2(hout, vo, sizeof hout));
            uint32_t wsum = hin[0] + hin[1];
            if (hout[0] != wsum || hout[1] != wsum) {
                printf("  vector_call mismatch: got [%u,%u] want [%u,%u]\n",
                       hout[0], hout[1], wsum, wsum);
                failures++;
            } else {
                printf("  driver vector_call: .v2.u32 in/out through call() ok\n");
            }
            CHECK_CU(p_cuMemFree_v2(vi));
            CHECK_CU(p_cuMemFree_v2(vo));
            CHECK_CU(p_cuModuleUnload(vmod));
            CHECK_CU(p_cuLinkDestroy(vls));
            free(vb);
        }
    }
    // ldmatrix.sync.aligned.m8n8.x2.b16 (tests/ptx/ldmatrix_x2.ptx):
    // 32 lanes x 2 destination registers, pinned against ZLUDA's
    // NVIDIA-generated vectors.
    {
        static const uint32_t exp_ldm[64] = {
#include "ldmatrix_expected.inc"
        };
        snprintf(path, sizeof(path), "%s/ldmatrix_x2.ptx", ACUDA_TEST_PTX_DIR);
        char *lb = read_file(path);
        if (!lb) {
            printf("  SKIP ldmatrix: kernel absent\n");
        } else {
            CUlinkState lls = NULL;
            CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &lls));
            CHECK_CU(p_cuLinkAddData(lls, 1, lb, strlen(lb), "ldmatrix_x2",
                                     0, NULL, NULL));
            void *lcub = NULL;
            size_t lcub_size = 0;
            CHECK_CU(p_cuLinkComplete(lls, &lcub, &lcub_size));
            CUmodule lmod = NULL;
            CHECK_CU(p_cuModuleLoadData(&lmod, lcub));
            CUfunction lfn = NULL;
            CHECK_CU(p_cuModuleGetFunction(&lfn, lmod, "ldmatrix_x2"));
            CUdeviceptr lo = 0;
            CHECK_CU(p_cuMemAlloc_v2(&lo, 256));
            uint32_t lz[64] = { 0 };
            CHECK_CU(p_cuMemcpyHtoD_v2(lo, lz, 256));
            void *lpar[1] = { &lo };
            CHECK_CU(p_cuLaunchKernel(lfn, 1, 1, 1, 32, 1, 1, 0, NULL, lpar,
                                      NULL));
            CHECK_CU(p_cuCtxSynchronize());
            uint32_t lgot[64] = { 0 };
            CHECK_CU(p_cuMemcpyDtoH_v2(lgot, lo, 256));
            int lbad = 0;
            for (int i = 0; i < 64; i++) {
                if (lgot[i] != exp_ldm[i]) {
                    if (lbad < 4)
                        printf("  ldmatrix mismatch[i=%d]: got %u want %u\n", i,
                               lgot[i], exp_ldm[i]);
                    lbad++;
                }
            }
            if (lbad)
                failures += lbad;
            else
                printf("  driver ldmatrix: ldmatrix.sync.aligned.m8n8.x2.b16 (32 lanes) matches NVIDIA vectors\n");
            CHECK_CU(p_cuMemFree_v2(lo));
            CHECK_CU(p_cuModuleUnload(lmod));
            CHECK_CU(p_cuLinkDestroy(lls));
            free(lb);
        }
    }
    // ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 (tests/ptx/
    // ldmatrix_trans.ptx, the corpus kernel): 32 lanes x 4 destination
    // registers, pinned against ZLUDA's NVIDIA-generated ldmatrix_trans
    // vectors (256/256 exact).
    {
        static const uint32_t exp_ldt[256] = {
#include "ldmatrix_trans_expected.inc"
        };
        snprintf(path, sizeof(path), "%s/ldmatrix_trans.ptx",
                 ACUDA_TEST_PTX_DIR);
        char *tb = read_file(path);
        if (!tb) {
            printf("  SKIP ldmatrix_trans: kernel absent\n");
        } else {
            CUlinkState tls = NULL;
            CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &tls));
            CHECK_CU(p_cuLinkAddData(tls, 1, tb, strlen(tb), "ldmatrix_trans",
                                     0, NULL, NULL));
            void *tcub = NULL;
            size_t tcub_size = 0;
            CHECK_CU(p_cuLinkComplete(tls, &tcub, &tcub_size));
            CUmodule tmod = NULL;
            CHECK_CU(p_cuModuleLoadData(&tmod, tcub));
            CUfunction tfn = NULL;
            CHECK_CU(p_cuModuleGetFunction(&tfn, tmod, "ldmatrix_trans"));
            CUdeviceptr to = 0;
            CHECK_CU(p_cuMemAlloc_v2(&to, 1024));
            uint32_t tz[256] = { 0 };
            CHECK_CU(p_cuMemcpyHtoD_v2(to, tz, sizeof tz));
            void *tpar[1] = { &to };
            CHECK_CU(p_cuLaunchKernel(tfn, 1, 1, 1, 64, 1, 1, 0, NULL, tpar,
                                      NULL));
            CHECK_CU(p_cuCtxSynchronize());
            uint32_t tgot[256] = { 0 };
            CHECK_CU(p_cuMemcpyDtoH_v2(tgot, to, sizeof tgot));
            int tbad = 0;
            for (int i = 0; i < 256; i++) {
                if (tgot[i] != exp_ldt[i]) {
                    if (tbad < 4)
                        printf("  ldmatrix_trans mismatch[i=%d]: got %u want %u\n",
                               i, tgot[i], exp_ldt[i]);
                    tbad++;
                }
            }
            if (tbad)
                failures += tbad;
            else
                printf("  driver ldmatrix_trans: x4.trans (32 lanes) matches NVIDIA vectors\n");
            CHECK_CU(p_cuMemFree_v2(to));
            CHECK_CU(p_cuModuleUnload(tmod));
            CHECK_CU(p_cuLinkDestroy(tls));
            free(tb);
        }
    }
    // Plain ldmatrix.sync.aligned.m8n8.x4.b16 (tests/ptx/ldmatrix_x4.ptx):
    // same corpus tile, four matrices, non-trans formula (hand-derived
    // from values_g; 256/256 verified on-GPU).
    {
        static const uint32_t exp_lx4[256] = {
#include "ldmatrix_x4_expected.inc"
        };
        snprintf(path, sizeof(path), "%s/ldmatrix_x4.ptx", ACUDA_TEST_PTX_DIR);
        char *xb = read_file(path);
        if (!xb) {
            printf("  SKIP ldmatrix_x4: kernel absent\n");
        } else {
            CUlinkState xls = NULL;
            CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &xls));
            CHECK_CU(p_cuLinkAddData(xls, 1, xb, strlen(xb), "ldmatrix_x4",
                                     0, NULL, NULL));
            void *xcub = NULL;
            size_t xcub_size = 0;
            CHECK_CU(p_cuLinkComplete(xls, &xcub, &xcub_size));
            CUmodule xmod = NULL;
            CHECK_CU(p_cuModuleLoadData(&xmod, xcub));
            CUfunction xfn = NULL;
            CHECK_CU(p_cuModuleGetFunction(&xfn, xmod, "ldmatrix_x4"));
            CUdeviceptr xo = 0;
            CHECK_CU(p_cuMemAlloc_v2(&xo, 1024));
            uint32_t xz[256] = { 0 };
            CHECK_CU(p_cuMemcpyHtoD_v2(xo, xz, sizeof xz));
            void *xpar[1] = { &xo };
            CHECK_CU(p_cuLaunchKernel(xfn, 1, 1, 1, 64, 1, 1, 0, NULL, xpar,
                                      NULL));
            CHECK_CU(p_cuCtxSynchronize());
            uint32_t xgot[256] = { 0 };
            CHECK_CU(p_cuMemcpyDtoH_v2(xgot, xo, sizeof xgot));
            int xbad = 0;
            for (int i = 0; i < 256; i++) {
                if (xgot[i] != exp_lx4[i]) {
                    if (xbad < 4)
                        printf("  ldmatrix_x4 mismatch[i=%d]: got %u want %u\n",
                               i, xgot[i], exp_lx4[i]);
                    xbad++;
                }
            }
            if (xbad)
                failures += xbad;
            else
                printf("  driver ldmatrix_x4: m8n8.x4.b16 (32 lanes) matches derived vectors\n");
            CHECK_CU(p_cuMemFree_v2(xo));
            CHECK_CU(p_cuModuleUnload(xmod));
            CHECK_CU(p_cuLinkDestroy(xls));
            free(xb);
        }
    }
    // ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 (tests/ptx/
    // ldmatrix_x2_trans.ptx): 32 lanes x 2 destination registers, values
    // derived from the trans fragment map proven above (64/64 on-GPU).
    {
        static const uint32_t exp_l2t[64] = {
#include "ldmatrix_x2_trans_expected.inc"
        };
        snprintf(path, sizeof(path), "%s/ldmatrix_x2_trans.ptx",
                 ACUDA_TEST_PTX_DIR);
        char *yb = read_file(path);
        if (!yb) {
            printf("  SKIP ldmatrix_x2_trans: kernel absent\n");
        } else {
            CUlinkState yls = NULL;
            CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &yls));
            CHECK_CU(p_cuLinkAddData(yls, 1, yb, strlen(yb),
                                     "ldmatrix_x2_trans", 0, NULL, NULL));
            void *ycub = NULL;
            size_t ycub_size = 0;
            CHECK_CU(p_cuLinkComplete(yls, &ycub, &ycub_size));
            CUmodule ymod = NULL;
            CHECK_CU(p_cuModuleLoadData(&ymod, ycub));
            CUfunction yfn = NULL;
            CHECK_CU(p_cuModuleGetFunction(&yfn, ymod, "ldmatrix_x2_trans"));
            CUdeviceptr yo = 0;
            CHECK_CU(p_cuMemAlloc_v2(&yo, 256));
            uint32_t yz[64] = { 0 };
            CHECK_CU(p_cuMemcpyHtoD_v2(yo, yz, sizeof yz));
            void *ypar[1] = { &yo };
            CHECK_CU(p_cuLaunchKernel(yfn, 1, 1, 1, 32, 1, 1, 0, NULL, ypar,
                                      NULL));
            CHECK_CU(p_cuCtxSynchronize());
            uint32_t ygot[64] = { 0 };
            CHECK_CU(p_cuMemcpyDtoH_v2(ygot, yo, sizeof ygot));
            int ybad = 0;
            for (int i = 0; i < 64; i++) {
                if (ygot[i] != exp_l2t[i]) {
                    if (ybad < 4)
                        printf("  ldmatrix_x2_trans mismatch[i=%d]: got %u want %u\n",
                               i, ygot[i], exp_l2t[i]);
                    ybad++;
                }
            }
            if (ybad)
                failures += ybad;
            else
                printf("  driver ldmatrix_x2_trans: x2.trans (32 lanes) matches derived vectors\n");
            CHECK_CU(p_cuMemFree_v2(yo));
            CHECK_CU(p_cuModuleUnload(ymod));
            CHECK_CU(p_cuLinkDestroy(yls));
            free(yb);
        }
    }
    // fp8_cvt.ptx: cvt.rn.satfinite.{e4m3x2,e5m2x2}.f32 and
    // cvt.rn.f16x2.{e4m3x2,e5m2x2} over 64 lanes (two f32 each), pinned
    // against a literal port of NVIDIA's cuda_fp8.hpp reference (that
    // port was checked bit-exact over 2.3M patterns in both formats).
    // Inputs cover zeros, NaN/Inf, denormals, rounding ties for both
    // formats, saturation boundaries and random finite patterns.
    {
        static const uint32_t exp_fp8[256] = {
#include "fp8_cvt_expected.inc"
        };
        snprintf(path, sizeof(path), "%s/fp8_cvt.ptx", ACUDA_TEST_PTX_DIR);
        char *fb = read_file(path);
        if (!fb) {
            printf("  SKIP fp8_cvt: kernel absent\n");
        } else {
            CUlinkState fls = NULL;
            CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &fls));
            CHECK_CU(p_cuLinkAddData(fls, 1, fb, strlen(fb), "fp8_cvt", 0,
                                     NULL, NULL));
            void *fcub = NULL;
            size_t fcub_size = 0;
            CHECK_CU(p_cuLinkComplete(fls, &fcub, &fcub_size));
            CUmodule fmod = NULL;
            CHECK_CU(p_cuModuleLoadData(&fmod, fcub));
            CUfunction ffn = NULL;
            CHECK_CU(p_cuModuleGetFunction(&ffn, fmod, "fp8_cvt"));
            CUdeviceptr fo = 0;
            CHECK_CU(p_cuMemAlloc_v2(&fo, 1024));
            uint32_t fz[256] = { 0 };
            CHECK_CU(p_cuMemcpyHtoD_v2(fo, fz, sizeof fz));
            void *fpar[1] = { &fo };
            CHECK_CU(p_cuLaunchKernel(ffn, 1, 1, 1, 64, 1, 1, 0, NULL, fpar,
                                      NULL));
            CHECK_CU(p_cuCtxSynchronize());
            uint32_t fgot[256] = { 0 };
            CHECK_CU(p_cuMemcpyDtoH_v2(fgot, fo, sizeof fgot));
            int fbad = 0;
            for (int i = 0; i < 256; i++) {
                if (fgot[i] != exp_fp8[i]) {
                    if (fbad < 4)
                        printf("  fp8_cvt mismatch[i=%d]: got %08x want %08x\n",
                               i, fgot[i], exp_fp8[i]);
                    fbad++;
                }
            }
            if (fbad)
                failures += fbad;
            else
                printf("  driver fp8_cvt: e4m3/e5m2 x2 cvt (64 lanes) matches cuda_fp8.hpp reference\n");
            CHECK_CU(p_cuMemFree_v2(fo));
            CHECK_CU(p_cuModuleUnload(fmod));
            CHECK_CU(p_cuLinkDestroy(fls));
            free(fb);
        }
    }
    // vec_ldst64.ptx: .v2.b64 global ld, .v2.b64 shared st/ld round-trip,
    // mov.b32 into a .v2.b16 register, and st.b16 of a component through
    // a generic pointer. Outputs are hand-computed from the global
    // initializer (0x1111..0x8888 as eight u16s).
    {
        static const uint32_t exp_v64[6] = { 0x22221111u, 0x44443333u,
                                             0x66665555u, 0x88887777u,
                                             0x22221111u, 0u };
        snprintf(path, sizeof(path), "%s/vec_ldst64.ptx", ACUDA_TEST_PTX_DIR);
        char *vb64b = read_file(path);
        if (!vb64b) {
            printf("  SKIP vec_ldst64: kernel absent\n");
        } else {
            CUlinkState v64ls = NULL;
            CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &v64ls));
            CHECK_CU(p_cuLinkAddData(v64ls, 1, vb64b, strlen(vb64b),
                                     "vec_ldst64", 0, NULL, NULL));
            void *v64cub = NULL;
            size_t v64cub_size = 0;
            CHECK_CU(p_cuLinkComplete(v64ls, &v64cub, &v64cub_size));
            CUmodule v64mod = NULL;
            CHECK_CU(p_cuModuleLoadData(&v64mod, v64cub));
            CUfunction v64fn = NULL;
            CHECK_CU(p_cuModuleGetFunction(&v64fn, v64mod, "vec_ldst64"));
            CUdeviceptr v64o = 0;
            CHECK_CU(p_cuMemAlloc_v2(&v64o, 24));
            uint32_t v64z[6] = { 0 };
            CHECK_CU(p_cuMemcpyHtoD_v2(v64o, v64z, sizeof v64z));
            void *v64par[1] = { &v64o };
            CHECK_CU(p_cuLaunchKernel(v64fn, 1, 1, 1, 1, 1, 1, 0, NULL, v64par,
                                      NULL));
            CHECK_CU(p_cuCtxSynchronize());
            uint32_t v64got[6] = { 0 };
            CHECK_CU(p_cuMemcpyDtoH_v2(v64got, v64o, sizeof v64got));
            int v64bad = 0;
            for (int i = 0; i < 6; i++) {
                if (v64got[i] != exp_v64[i]) {
                    printf("  vec_ldst64 mismatch[i=%d]: got %08x want %08x\n",
                           i, v64got[i], exp_v64[i]);
                    v64bad++;
                }
            }
            if (v64bad)
                failures += v64bad;
            else
                printf("  driver vec_ldst64: .v2.b64 ld/st + mov.b32->.v2.b16 + st.b16 generic ok\n");
            CHECK_CU(p_cuMemFree_v2(v64o));
            CHECK_CU(p_cuModuleUnload(v64mod));
            CHECK_CU(p_cuLinkDestroy(v64ls));
            free(vb64b);
        }
    }
    // pred_membar.ptx: not.pred plus membar.cta/gl/sys between the load
    // and the store. Two launches: (10,20) -> 2 and (20,10) -> 1 (the
    // setp.lt result is inverted, then @pred writes 1 / @!pred writes 2).
    {
        static const uint64_t pm_in[2][2] = { { 10u, 20u }, { 20u, 10u } };
        static const uint64_t pm_exp[2] = { 2u, 1u };
        snprintf(path, sizeof(path), "%s/pred_membar.ptx", ACUDA_TEST_PTX_DIR);
        char *pmb = read_file(path);
        if (!pmb) {
            printf("  SKIP pred_membar: kernel absent\n");
        } else {
            CUlinkState pmls = NULL;
            CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &pmls));
            CHECK_CU(p_cuLinkAddData(pmls, 1, pmb, strlen(pmb), "pred_membar",
                                     0, NULL, NULL));
            void *pmcub = NULL;
            size_t pmcub_size = 0;
            CHECK_CU(p_cuLinkComplete(pmls, &pmcub, &pmcub_size));
            CUmodule pmmod = NULL;
            CHECK_CU(p_cuModuleLoadData(&pmmod, pmcub));
            CUfunction pmfn = NULL;
            CHECK_CU(p_cuModuleGetFunction(&pmfn, pmmod, "pred_membar"));
            CUdeviceptr pmin = 0, pmout = 0;
            CHECK_CU(p_cuMemAlloc_v2(&pmin, 16));
            CHECK_CU(p_cuMemAlloc_v2(&pmout, 8));
            int pmbad = 0;
            for (int k = 0; k < 2; k++) {
                CHECK_CU(p_cuMemcpyHtoD_v2(pmin, pm_in[k], 16));
                uint64_t pmz = 0;
                CHECK_CU(p_cuMemcpyHtoD_v2(pmout, &pmz, 8));
                void *pmpar[2] = { &pmin, &pmout };
                CHECK_CU(p_cuLaunchKernel(pmfn, 1, 1, 1, 1, 1, 1, 0, NULL,
                                          pmpar, NULL));
                CHECK_CU(p_cuCtxSynchronize());
                uint64_t pmgot = 0;
                CHECK_CU(p_cuMemcpyDtoH_v2(&pmgot, pmout, 8));
                if (pmgot != pm_exp[k]) {
                    printf("  pred_membar mismatch[k=%d]: got %llu want %llu\n",
                           k, (unsigned long long)pmgot,
                           (unsigned long long)pm_exp[k]);
                    pmbad++;
                }
            }
            if (pmbad)
                failures += pmbad;
            else
                printf("  driver pred_membar: not.pred + membar.cta/gl/sys ok\n");
            CHECK_CU(p_cuMemFree_v2(pmin));
            CHECK_CU(p_cuMemFree_v2(pmout));
            CHECK_CU(p_cuModuleUnload(pmmod));
            CHECK_CU(p_cuLinkDestroy(pmls));
            free(pmb);
        }
    }
    // cvt_rnd.ptx: cvt.rni/rzi/rmi/rpi float->int (plus a .ftz column) and
    // cvt.rmi/rpi f32.f32 over a 32-value sweep (ties, signed zeros,
    // +/-Inf, NaN, saturation bounds, huge values, tiny normals).
    // Expected words are IEEE trunc/floor/ceil/nearest-even
    // plus PTX saturation (NaN -> 0, out of range -> INT_MIN/INT_MAX),
    // hand-derived in the generator. 32 lanes: 5 int words at out+20t,
    // 2 f32-pattern words at out+640+8t (224 words).
    {
        static const uint32_t exp_cvt_rnd[224] = {
#include "cvt_rnd_expected.inc"
        };
        snprintf(path, sizeof(path), "%s/cvt_rnd.ptx", ACUDA_TEST_PTX_DIR);
        char *rb = read_file(path);
        if (!rb) {
            printf("  SKIP cvt_rnd: kernel absent\n");
        } else {
            CUlinkState rls = NULL;
            CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &rls));
            CHECK_CU(p_cuLinkAddData(rls, 1, rb, strlen(rb), "cvt_rnd", 0,
                                     NULL, NULL));
            void *rcub = NULL;
            size_t rcub_size = 0;
            CHECK_CU(p_cuLinkComplete(rls, &rcub, &rcub_size));
            CUmodule rmod = NULL;
            CHECK_CU(p_cuModuleLoadData(&rmod, rcub));
            CUfunction rfn = NULL;
            CHECK_CU(p_cuModuleGetFunction(&rfn, rmod, "cvt_rnd"));
            CUdeviceptr ro = 0;
            CHECK_CU(p_cuMemAlloc_v2(&ro, 896));
            uint32_t rz[224] = { 0 };
            CHECK_CU(p_cuMemcpyHtoD_v2(ro, rz, sizeof rz));
            void *rpar[1] = { &ro };
            CHECK_CU(p_cuLaunchKernel(rfn, 1, 1, 1, 32, 1, 1, 0, NULL, rpar,
                                      NULL));
            CHECK_CU(p_cuCtxSynchronize());
            uint32_t rgot[224] = { 0 };
            CHECK_CU(p_cuMemcpyDtoH_v2(rgot, ro, sizeof rgot));
            int rbad = 0;
            for (int i = 0; i < 224; i++) {
                if (rgot[i] != exp_cvt_rnd[i]) {
                    if (rbad < 6)
                        printf("  cvt_rnd mismatch[i=%d]: got %08x want %08x\n",
                               i, rgot[i], exp_cvt_rnd[i]);
                    rbad++;
                }
            }
            if (rbad)
                failures += rbad;
            else
                printf("  driver cvt_rnd: rni/rzi/rmi/rpi int + f32.f32 (32 lanes, 224 words) ok\n");
            CHECK_CU(p_cuMemFree_v2(ro));
            CHECK_CU(p_cuModuleUnload(rmod));
            CHECK_CU(p_cuLinkDestroy(rls));
            free(rb);
        }
    }
    // addr_arith.ptx: add.u64/sub.u64 of two addresses yield a plain
    // value, not an address (ptxas parity; stateful_neg_offset). Only
    // differences are representation-independent: q-out = 8 and
    // (q+8)-q = 8 under both descriptor tokens and physical addresses.
    // out[0] echoes the input word, out[1] = 8, out[2] = 8.
    {
        snprintf(path, sizeof(path), "%s/addr_arith.ptx", ACUDA_TEST_PTX_DIR);
        char *ab = read_file(path);
        if (!ab) {
            printf("  SKIP addr_arith: kernel absent\n");
        } else {
            CUlinkState als = NULL;
            CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &als));
            CHECK_CU(p_cuLinkAddData(als, 1, ab, strlen(ab), "addr_arith", 0,
                                     NULL, NULL));
            void *acub = NULL;
            size_t acub_size = 0;
            CHECK_CU(p_cuLinkComplete(als, &acub, &acub_size));
            CUmodule amod = NULL;
            CHECK_CU(p_cuModuleLoadData(&amod, acub));
            CUfunction afn = NULL;
            CHECK_CU(p_cuModuleGetFunction(&afn, amod, "addr_arith"));
            CUdeviceptr ain = 0, aout = 0;
            CHECK_CU(p_cuMemAlloc_v2(&ain, 8));
            CHECK_CU(p_cuMemAlloc_v2(&aout, 24));
            const uint64_t awant = 0x1122334455667788ull;
            CHECK_CU(p_cuMemcpyHtoD_v2(ain, &awant, 8));
            uint64_t az[3] = { 0, 0, 0 };
            CHECK_CU(p_cuMemcpyHtoD_v2(aout, az, sizeof az));
            void *apar[2] = { &ain, &aout };
            CHECK_CU(p_cuLaunchKernel(afn, 1, 1, 1, 1, 1, 1, 0, NULL, apar,
                                      NULL));
            CHECK_CU(p_cuCtxSynchronize());
            uint64_t agot[3] = { 0, 0, 0 };
            CHECK_CU(p_cuMemcpyDtoH_v2(agot, aout, sizeof agot));
            const uint64_t adiff = 8, adiff2 = 8;
            if (agot[0] != awant || agot[1] != adiff || agot[2] != adiff2) {
                printf("  addr_arith mismatch: echo %llx/%llx diff %llx/%llx "
                       "step %llx/%llx\n",
                       (unsigned long long)agot[0],
                       (unsigned long long)awant,
                       (unsigned long long)agot[1],
                       (unsigned long long)adiff,
                       (unsigned long long)agot[2],
                       (unsigned long long)adiff2);
                failures++;
            } else {
                printf("  driver addr_arith: address add/sub deltas (8/8), echo ok\n");
            }
            CHECK_CU(p_cuMemFree_v2(ain));
            CHECK_CU(p_cuMemFree_v2(aout));
            CHECK_CU(p_cuModuleUnload(amod));
            CHECK_CU(p_cuLinkDestroy(als));
            free(ab);
        }
    }
    // two self-loops + forward dispatch branches. 23 positional params:
    // data pointers at 5/12/19, scratch u64 store at 0, strides 1 at
    // 6/13/20, trip count n at 22; rest unused. SEG=8 elements per
    // thread; n in {0,4,6,8} covers early-exit, main-only, and
    // main+remainder paths (n in 1..3 is undefined on NVIDIA too:
    // the remainder loop would read registers the skipped prologue
    // never initialized, so those shapes are not tested).
    {
        snprintf(path, sizeof(path), "%s/loop_numba.ptx", ACUDA_TEST_PTX_DIR);
        char *nb = read_file(path);
        if (!nb)
            return 1;
        CUlinkState nls = NULL;
        CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &nls));
        CHECK_CU(p_cuLinkAddData(nls, 1, nb, strlen(nb), "looplong", 0,
                                 NULL, NULL));
        void *ncubin = NULL;
        size_t ncubin_size = 0;
        CHECK_CU(p_cuLinkComplete(nls, &ncubin, &ncubin_size));
        CUmodule nmod = NULL;
        CHECK_CU(p_cuModuleLoadData(&nmod, ncubin));
        CUfunction nfn = NULL;
        CHECK_CU(p_cuModuleGetFunction(&nfn, nmod,
            "_ZN8__main__7loopaddB2v1B74cw51cXTLSUwv1sCUt9Ww0FEw0NRRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAltAEAA_3d_3dE5ArrayIfLi1E1C7mutable7alignedE5ArrayIfLi1E1C7mutable7alignedE5ArrayIfLi1E1C7mutable7alignedEx"));
        const uint32_t nn = 1024, seg = 8;
        CUdeviceptr na = 0, nbp = 0, nc = 0, nst = 0;
        CHECK_CU(p_cuMemAlloc_v2(&na, (size_t)nn * seg * 4));
        CHECK_CU(p_cuMemAlloc_v2(&nbp, (size_t)nn * seg * 4));
        CHECK_CU(p_cuMemAlloc_v2(&nc, (size_t)nn * 4));
        CHECK_CU(p_cuMemAlloc_v2(&nst, 8));
        float *nha = malloc((size_t)nn * seg * 4);
        float *nhb = malloc((size_t)nn * seg * 4);
        float *nhc = malloc((size_t)nn * 4);
        if (nha && nhb && nhc) {
            for (uint32_t i = 0; i < nn * seg; i++) {
                nha[i] = (float)(i % 5 + 1);
                nhb[i] = (float)(i % 3 + 1);
            }
            CHECK_CU(p_cuMemcpyHtoD_v2(na, nha, (size_t)nn * seg * 4));
            CHECK_CU(p_cuMemcpyHtoD_v2(nbp, nhb, (size_t)nn * seg * 4));
            const uint64_t trips[] = { 0, 4, 6, 8 };
            for (size_t ti = 0; ti < sizeof(trips) / sizeof(trips[0]); ti++) {
                for (uint32_t i = 0; i < nn; i++)
                    nhc[i] = 0.0f;
                CHECK_CU(p_cuMemcpyHtoD_v2(nc, nhc, (size_t)nn * 4));
                CUdeviceptr args[23];
                memset(args, 0, sizeof(args));
                uint64_t one = 1, nnu = trips[ti];
                args[0] = nst;
                args[5] = na;
                args[6] = one;
                args[12] = nbp;
                args[13] = one;
                args[19] = nc;
                args[20] = one;
                args[22] = nnu;
                void *par[23];
                for (int i = 0; i < 23; i++)
                    par[i] = &args[i];
                CHECK_CU(p_cuLaunchKernel(nfn, nn / 256, 1, 1, 256, 1, 1,
                                         0, NULL, par, NULL));
                CHECK_CU(p_cuCtxSynchronize());
                CHECK_CU(p_cuMemcpyDtoH_v2(nhc, nc, (size_t)nn * 4));
                uint32_t bad9 = 0;
                for (uint32_t i = 0; i < nn; i++) {
                    float want = 0.0f;
                    for (uint64_t k = 0; k < nnu; k++)
                        want = want + nha[i * seg + k] * nhb[i * seg + k];
                    if (nhc[i] != want) {
                        if (bad9 < 5)
                            printf("  looplong n=%llu mismatch[%u]: got %f want %f\n",
                                   (unsigned long long)nnu, i, nhc[i], want);
                        bad9++;
                    }
                }
                CHECK(bad9 == 0);
                printf("  driver looplong n=%llu: %u elements ok\n",
                       (unsigned long long)nnu, nn);
            }
        }
        free(nha);
        free(nhb);
        free(nhc);
        CHECK_CU(p_cuMemFree_v2(na));
        CHECK_CU(p_cuMemFree_v2(nbp));
        CHECK_CU(p_cuMemFree_v2(nc));
        CHECK_CU(p_cuMemFree_v2(nst));
        CHECK_CU(p_cuModuleUnload(nmod));
        CHECK_CU(p_cuLinkDestroy(nls));
        free(nb);
    }
    // Phase 4 close-out kernel (tests/ptx/arith_closeout.ptx): div/rem
    // (u32/s32/u64/s64, f32/f64), or/xor.b32/b64, u32 shifts, setp.s32.
    // 7 pointer params (a/b in, c/d/e/f/g out). Thread 0 skips via the
    // setp.s32 guard; outputs zeroed up front.
    {
        snprintf(path, sizeof(path), "%s/arith_closeout.ptx",
                 ACUDA_TEST_PTX_DIR);
        char *nb = read_file(path);
        if (!nb)
            return 1;
        CUlinkState nls = NULL;
        CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &nls));
        CHECK_CU(p_cuLinkAddData(nls, 1, nb, strlen(nb), "arith", 0,
                                 NULL, NULL));
        void *ncubin = NULL;
        size_t ncubin_size = 0;
        CHECK_CU(p_cuLinkComplete(nls, &ncubin, &ncubin_size));
        CUmodule nmod = NULL;
        CHECK_CU(p_cuModuleLoadData(&nmod, ncubin));
        CUfunction nfn = NULL;
        CHECK_CU(p_cuModuleGetFunction(&nfn, nmod, "arith_close"));
        const uint32_t nn = 4096;
        CUdeviceptr na = 0, nbp = 0, nc = 0, nd = 0, ne = 0, nf = 0,
                    ng = 0;
        CHECK_CU(p_cuMemAlloc_v2(&na, (size_t)nn * 4));
        CHECK_CU(p_cuMemAlloc_v2(&nbp, (size_t)nn * 4));
        CHECK_CU(p_cuMemAlloc_v2(&nc, (size_t)nn * 4));
        CHECK_CU(p_cuMemAlloc_v2(&nd, (size_t)nn * 4));
        CHECK_CU(p_cuMemAlloc_v2(&ne, (size_t)nn * 8));
        CHECK_CU(p_cuMemAlloc_v2(&nf, (size_t)nn * 4));
        CHECK_CU(p_cuMemAlloc_v2(&ng, (size_t)nn * 8));
        float *nha = malloc((size_t)nn * 4);
        float *nhb = malloc((size_t)nn * 4);
        float *nhc = malloc((size_t)nn * 4);
        uint32_t *nhd = malloc((size_t)nn * 4);
        uint64_t *nhe = malloc((size_t)nn * 8);
        uint32_t *nhf = malloc((size_t)nn * 4);
        double *nhg = malloc((size_t)nn * 8);
        if (nha && nhb && nhc && nhd && nhe && nhf && nhg) {
            for (uint32_t i = 0; i < nn; i++) {
                nha[i] = (float)(i * 8);
                nhb[i] = (float)(i * 4);
                nhc[i] = 0.0f;
                nhd[i] = 0;
                nhe[i] = 0;
                nhf[i] = 0;
                nhg[i] = 0.0;
            }
            CHECK_CU(p_cuMemcpyHtoD_v2(na, nha, (size_t)nn * 4));
            CHECK_CU(p_cuMemcpyHtoD_v2(nbp, nhb, (size_t)nn * 4));
            CHECK_CU(p_cuMemcpyHtoD_v2(nc, nhc, (size_t)nn * 4));
            CHECK_CU(p_cuMemcpyHtoD_v2(nd, nhd, (size_t)nn * 4));
            CHECK_CU(p_cuMemcpyHtoD_v2(ne, nhe, (size_t)nn * 8));
            CHECK_CU(p_cuMemcpyHtoD_v2(nf, nhf, (size_t)nn * 4));
            CHECK_CU(p_cuMemcpyHtoD_v2(ng, nhg, (size_t)nn * 8));
            CUdeviceptr args[7];
            args[0] = na;
            args[1] = nbp;
            args[2] = nc;
            args[3] = nd;
            args[4] = ne;
            args[5] = nf;
            args[6] = ng;
            void *par[7];
            for (int i = 0; i < 7; i++)
                par[i] = &args[i];
            CHECK_CU(p_cuLaunchKernel(nfn, nn / 256, 1, 1, 256, 1, 1, 0,
                                     NULL, par, NULL));
            CHECK_CU(p_cuCtxSynchronize());
            CHECK_CU(p_cuMemcpyDtoH_v2(nhc, nc, (size_t)nn * 4));
            CHECK_CU(p_cuMemcpyDtoH_v2(nhd, nd, (size_t)nn * 4));
            CHECK_CU(p_cuMemcpyDtoH_v2(nhe, ne, (size_t)nn * 8));
            CHECK_CU(p_cuMemcpyDtoH_v2(nhf, nf, (size_t)nn * 4));
            CHECK_CU(p_cuMemcpyDtoH_v2(nhg, ng, (size_t)nn * 8));
            uint32_t bad10 = 0;
            for (uint32_t i = 0; i < nn; i++) {
                uint32_t wantd = 0;
                uint64_t wante = 0;
                uint32_t wantf = 0;
                float wantc = 0.0f;
                double wantg = 0.0;
                if (i != 0) {
                    uint32_t q32 = i / 7, r32 = i % 7;
                    int32_t si = (int32_t)i - 2000;
                    int32_t sq = si / 7, sr = si % 7;
                    wantd = q32 | (r32 << 20) |
                            ((uint32_t)(sr + 16) << 26);
                    wantf = (uint32_t)sq;
                    uint64_t q64 = (uint64_t)i / 13, r64 = (uint64_t)i % 13;
                    int64_t si64 = (int64_t)i - 3000;
                    int64_t sq64 = si64 / 13, sr64 = si64 % 13;
                    wante = (q64 ^ r64) |
                            (((uint64_t)sq64) << 32) |
                            (((uint64_t)(sr64 + 16)) << 48);
                    wantc = nha[i] / 8.0f + nhb[i] / 4.0f;
                    wantg = (double)nha[i] / 8.0;
                }
                if (nhd[i] != wantd || nhf[i] != wantf ||
                    nhe[i] != wante || nhc[i] != wantc ||
                    nhg[i] != wantg) {
                    if (bad10 < 5)
                        printf("  arith mismatch[%u]: got (%u,%llu,%u,%f,%f)"
                               " want (%u,%llu,%u,%f,%f)\n",
                               i, nhd[i],
                               (unsigned long long)nhe[i], nhf[i], nhc[i],
                               nhg[i], wantd,
                               (unsigned long long)wante, wantf, wantc,
                               wantg);
                    bad10++;
                }
            }
            CHECK(bad10 == 0);
            printf("  driver arithclose: %u elements ok\n", nn);
        }
        free(nha);
        free(nhb);
        free(nhc);
        free(nhd);
        free(nhe);
        free(nhf);
        free(nhg);
        CHECK_CU(p_cuMemFree_v2(na));
        CHECK_CU(p_cuMemFree_v2(nbp));
        CHECK_CU(p_cuMemFree_v2(nc));
        CHECK_CU(p_cuMemFree_v2(nd));
        CHECK_CU(p_cuMemFree_v2(ne));
        CHECK_CU(p_cuMemFree_v2(nf));
        CHECK_CU(p_cuMemFree_v2(ng));
        CHECK_CU(p_cuModuleUnload(nmod));
        CHECK_CU(p_cuLinkDestroy(nls));
        free(nb);
    }
    // Shared-memory kernel (tests/ptx/shmem.ptx): two static .shared
    // variables, mov-from-symbol bases, st/ld.shared with computed
    // offsets, two full-CTA bar.sync rendezvous. 4 blocks of 256:
    // per-workgroup isolation is the point (tile t = i & 255).
    {
        snprintf(path, sizeof(path), "%s/shmem.ptx", ACUDA_TEST_PTX_DIR);
        char *nb = read_file(path);
        if (!nb)
            return 1;
        CUlinkState nls = NULL;
        CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &nls));
        CHECK_CU(p_cuLinkAddData(nls, 1, nb, strlen(nb), "shmem", 0,
                                 NULL, NULL));
        void *ncubin = NULL;
        size_t ncubin_size = 0;
        CHECK_CU(p_cuLinkComplete(nls, &ncubin, &ncubin_size));
        CUmodule nmod = NULL;
        CHECK_CU(p_cuModuleLoadData(&nmod, ncubin));
        CUfunction nfn = NULL;
        CHECK_CU(p_cuModuleGetFunction(&nfn, nmod, "shmem"));
        int shv = -1;
        CHECK_CU(p_cuFuncGetAttribute(&shv,
                                      CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                                      nfn));
        CHECK(shv == 1280); // sh[1024] + sh2[256]
        const uint32_t nn = 1024;
        CUdeviceptr na = 0, nbp = 0, nc = 0;
        CHECK_CU(p_cuMemAlloc_v2(&na, (size_t)nn * 4));
        CHECK_CU(p_cuMemAlloc_v2(&nbp, (size_t)nn * 4));
        CHECK_CU(p_cuMemAlloc_v2(&nc, (size_t)nn * 4));
        float *nha = malloc((size_t)nn * 4);
        float *nhb = malloc((size_t)nn * 4);
        float *nhc = malloc((size_t)nn * 4);
        if (nha && nhb && nhc) {
            for (uint32_t i = 0; i < nn; i++) {
                nha[i] = (float)(i % 7 + 1);
                nhb[i] = (float)(i % 5 + 1);
                nhc[i] = 0.0f;
            }
            CHECK_CU(p_cuMemcpyHtoD_v2(na, nha, (size_t)nn * 4));
            CHECK_CU(p_cuMemcpyHtoD_v2(nbp, nhb, (size_t)nn * 4));
            CHECK_CU(p_cuMemcpyHtoD_v2(nc, nhc, (size_t)nn * 4));
            CUdeviceptr args[3];
            args[0] = na;
            args[1] = nbp;
            args[2] = nc;
            void *par[3];
            for (int i = 0; i < 3; i++)
                par[i] = &args[i];
            CHECK_CU(p_cuLaunchKernel(nfn, nn / 256, 1, 1, 256, 1, 1, 0,
                                     NULL, par, NULL));
            CHECK_CU(p_cuCtxSynchronize());
            CHECK_CU(p_cuMemcpyDtoH_v2(nhc, nc, (size_t)nn * 4));
            uint32_t bad11 = 0;
            for (uint32_t i = 0; i < nn; i++) {
                uint32_t base = (i / 256) * 256, t = i & 255;
                float want = nha[base + ((t + 1) & 255)] + nhb[i];
                if (nhc[i] != want) {
                    if (bad11 < 5)
                        printf("  shmem mismatch[%u]: got %f want %f\n", i,
                               nhc[i], want);
                    bad11++;
                }
            }
            CHECK(bad11 == 0);
            printf("  driver shmem: %u elements ok\n", nn);
        }
        free(nha);
        free(nhb);
        free(nhc);
        CHECK_CU(p_cuMemFree_v2(na));
        CHECK_CU(p_cuMemFree_v2(nbp));
        CHECK_CU(p_cuMemFree_v2(nc));
        CHECK_CU(p_cuModuleUnload(nmod));
        CHECK_CU(p_cuLinkDestroy(nls));
        free(nb);
    }
    // Real numba shared-memory kernel (tests/ptx/shmem_numba.ptx,
    // harvested): one 1024-byte .shared variable, computed shared
    // offsets, bare bar.sync 0. Same 22-param layout as the loop
    // kernels: data at 5/12/19, scratch at 0, strides-1 at 6/13/20.
    {
        snprintf(path, sizeof(path), "%s/shmem_numba.ptx",
                 ACUDA_TEST_PTX_DIR);
        char *nb = read_file(path);
        if (!nb)
            return 1;
        CUlinkState nls = NULL;
        CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &nls));
        CHECK_CU(p_cuLinkAddData(nls, 1, nb, strlen(nb), "shmemnb", 0,
                                 NULL, NULL));
        void *ncubin = NULL;
        size_t ncubin_size = 0;
        CHECK_CU(p_cuLinkComplete(nls, &ncubin, &ncubin_size));
        CUmodule nmod = NULL;
        CHECK_CU(p_cuModuleLoadData(&nmod, ncubin));
        CUfunction nfn = NULL;
        CHECK_CU(p_cuModuleGetFunction(&nfn, nmod,
            "_ZN8__main__8shmemaddB2v1B74cw51cXTLSUwv1sCUt9Ww0FEw0NRRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAltAEAA_3d_3dE5ArrayIfLi1E1C7mutable7alignedE5ArrayIfLi1E1C7mutable7alignedE5ArrayIfLi1E1C7mutable7alignedE"));
        int shv = -1;
        CHECK_CU(p_cuFuncGetAttribute(&shv,
                                      CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                                      nfn));
        CHECK(shv == 1024); // _cudapy_smem_1[1024]
        const uint32_t nn = 1024;
        CUdeviceptr na = 0, nbp = 0, nc = 0, nst = 0;
        CHECK_CU(p_cuMemAlloc_v2(&na, (size_t)nn * 4));
        CHECK_CU(p_cuMemAlloc_v2(&nbp, (size_t)nn * 4));
        CHECK_CU(p_cuMemAlloc_v2(&nc, (size_t)nn * 4));
        CHECK_CU(p_cuMemAlloc_v2(&nst, 8));
        float *nha = malloc((size_t)nn * 4);
        float *nhb = malloc((size_t)nn * 4);
        float *nhc = malloc((size_t)nn * 4);
        if (nha && nhb && nhc) {
            for (uint32_t i = 0; i < nn; i++) {
                nha[i] = (float)(i % 7 + 1);
                nhb[i] = (float)(i % 5 + 1);
                nhc[i] = 0.0f;
            }
            CHECK_CU(p_cuMemcpyHtoD_v2(na, nha, (size_t)nn * 4));
            CHECK_CU(p_cuMemcpyHtoD_v2(nbp, nhb, (size_t)nn * 4));
            CHECK_CU(p_cuMemcpyHtoD_v2(nc, nhc, (size_t)nn * 4));
            CUdeviceptr args[22];
            memset(args, 0, sizeof(args));
            uint64_t one = 1;
            args[0] = nst;
            args[5] = na;
            args[6] = one;
            args[12] = nbp;
            args[13] = one;
            args[19] = nc;
            args[20] = one;
            void *par[22];
            for (int i = 0; i < 22; i++)
                par[i] = &args[i];
            CHECK_CU(p_cuLaunchKernel(nfn, nn / 256, 1, 1, 256, 1, 1, 0,
                                     NULL, par, NULL));
            CHECK_CU(p_cuCtxSynchronize());
            CHECK_CU(p_cuMemcpyDtoH_v2(nhc, nc, (size_t)nn * 4));
            uint32_t bad12 = 0;
            for (uint32_t i = 0; i < nn; i++) {
                uint32_t base = (i / 256) * 256, t = i & 255;
                float want = nha[base + ((t + 1) & 255)] + nhb[i];
                if (nhc[i] != want) {
                    if (bad12 < 5)
                        printf("  shmemnb mismatch[%u]: got %f want %f\n",
                               i, nhc[i], want);
                    bad12++;
                }
            }
            CHECK(bad12 == 0);
            printf("  driver shmemnb: %u elements ok\n", nn);
        }
        free(nha);
        free(nhb);
        free(nhc);
        CHECK_CU(p_cuMemFree_v2(na));
        CHECK_CU(p_cuMemFree_v2(nbp));
        CHECK_CU(p_cuMemFree_v2(nc));
        CHECK_CU(p_cuMemFree_v2(nst));
        CHECK_CU(p_cuModuleUnload(nmod));
        CHECK_CU(p_cuLinkDestroy(nls));
        free(nb);
    }
    // Narrow-types kernel (tests/ptx/narrow.ptx): u8/s8/u16 ld/st
    // (global+shared, sub-word + RMW), full integer cvt matrix,
    // setp.u16/.s16, f16 arithmetic/select/cvt on halves-as-bits,
    // .b16 logic, RNE rounding lanes. 12 pointer params, 3 shared
    // variables (16384 bytes), N=4096 straight-line.
    {
        snprintf(path, sizeof(path), "%s/narrow.ptx", ACUDA_TEST_PTX_DIR);
        char *nb = read_file(path);
        if (!nb)
            return 1;
        CUlinkState nls = NULL;
        CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &nls));
        CHECK_CU(p_cuLinkAddData(nls, 1, nb, strlen(nb), "narrow", 0,
                                 NULL, NULL));
        void *ncubin = NULL;
        size_t ncubin_size = 0;
        CHECK_CU(p_cuLinkComplete(nls, &ncubin, &ncubin_size));
        CUmodule nmod = NULL;
        CHECK_CU(p_cuModuleLoadData(&nmod, ncubin));
        CUfunction nfn = NULL;
        CHECK_CU(p_cuModuleGetFunction(&nfn, nmod, "narrow"));
        int shv = -1;
        CHECK_CU(p_cuFuncGetAttribute(&shv,
                                      CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                                      nfn));
        CHECK(shv == 16384); // shb[4096] + shh[8192] + shs[4096]
        const uint32_t nn = 4096;
        CUdeviceptr na = 0, nbp = 0, ncc = 0, nd = 0, ne = 0, nf = 0,
                    ng = 0, ngg = 0, nh = 0, nn8 = 0, nn8b = 0, nn16 = 0;
        CHECK_CU(p_cuMemAlloc_v2(&na, (size_t)nn));
        CHECK_CU(p_cuMemAlloc_v2(&nbp, (size_t)nn));
        CHECK_CU(p_cuMemAlloc_v2(&ncc, (size_t)nn * 2));
        CHECK_CU(p_cuMemAlloc_v2(&nd, (size_t)nn * 4));
        CHECK_CU(p_cuMemAlloc_v2(&ne, (size_t)nn * 8));
        CHECK_CU(p_cuMemAlloc_v2(&nf, (size_t)nn * 4));
        CHECK_CU(p_cuMemAlloc_v2(&ng, (size_t)nn * 2));
        CHECK_CU(p_cuMemAlloc_v2(&ngg, (size_t)nn * 2));
        CHECK_CU(p_cuMemAlloc_v2(&nh, (size_t)nn * 2));
        CHECK_CU(p_cuMemAlloc_v2(&nn8, (size_t)nn));
        CHECK_CU(p_cuMemAlloc_v2(&nn8b, (size_t)nn));
        CHECK_CU(p_cuMemAlloc_v2(&nn16, (size_t)nn * 2));
        static const uint16_t gtab[12] = {
            0xBC00, 0xBA00, 0xB800, 0xB400, 0x0000, 0x3400,
            0x3800, 0x3A00, 0x3C00, 0x3D00, 0x3E00, 0x3F00
        };
        static const uint16_t ggtab[7] = {
            0xB800, 0xB400, 0x0000, 0x3400, 0x3800, 0x3A00, 0x3C00
        };
        static const uint32_t lanef[8] = {
            0x3F800000u, 0x3F802000u, 0x7F800000u, 0x80000000u,
            0xFFC00000u, 0x00000000u, 0xC0200000u, 0x00000000u
        };
        static const uint16_t laneh[8] = {
            0x3C00, 0x3C01, 0x7C00, 0x8000,
            0xFE00, 0x0080, 0xC100, 0x0000
        };
        uint8_t *nha = malloc(nn);
        uint8_t *nhb = malloc(nn);
        uint16_t *nhc = malloc((size_t)nn * 2);
        uint32_t *nhd = malloc((size_t)nn * 4);
        uint64_t *nhe = malloc((size_t)nn * 8);
        uint32_t *nhf = malloc((size_t)nn * 4);
        uint16_t *nhg = malloc((size_t)nn * 2);
        uint16_t *nhgg = malloc((size_t)nn * 2);
        uint16_t *nhh = malloc((size_t)nn * 2);
        uint8_t *nh8 = malloc(nn);
        uint8_t *nh8b = malloc(nn);
        uint16_t *nh16 = malloc((size_t)nn * 2);
        if (nha && nhb && nhc && nhd && nhe && nhf && nhg && nhgg &&
            nhh && nh8 && nh8b && nh16) {
            for (uint32_t i = 0; i < nn; i++) {
                nha[i] = (uint8_t)(i % 200);
                nhb[i] = (uint8_t)((int)(i % 41) - 20);
                nhc[i] = (uint16_t)((i * 3) % 1000);
                nhd[i] = 0;
                nhe[i] = 0;
                nhf[i] = 0;
                nhg[i] = gtab[i % 12];
                nhgg[i] = ggtab[i % 7];
                nhh[i] = 0;
                nh8[i] = 0;
                nh8b[i] = 0;
                nh16[i] = 0;
            }
            CHECK_CU(p_cuMemcpyHtoD_v2(na, nha, (size_t)nn));
            CHECK_CU(p_cuMemcpyHtoD_v2(nbp, nhb, (size_t)nn));
            CHECK_CU(p_cuMemcpyHtoD_v2(ncc, nhc, (size_t)nn * 2));
            CHECK_CU(p_cuMemcpyHtoD_v2(nd, nhd, (size_t)nn * 4));
            CHECK_CU(p_cuMemcpyHtoD_v2(ne, nhe, (size_t)nn * 8));
            CHECK_CU(p_cuMemcpyHtoD_v2(nf, nhf, (size_t)nn * 4));
            CHECK_CU(p_cuMemcpyHtoD_v2(ng, nhg, (size_t)nn * 2));
            CHECK_CU(p_cuMemcpyHtoD_v2(ngg, nhgg, (size_t)nn * 2));
            CHECK_CU(p_cuMemcpyHtoD_v2(nh, nhh, (size_t)nn * 2));
            CHECK_CU(p_cuMemcpyHtoD_v2(nn8, nh8, (size_t)nn));
            CHECK_CU(p_cuMemcpyHtoD_v2(nn8b, nh8b, (size_t)nn));
            CHECK_CU(p_cuMemcpyHtoD_v2(nn16, nh16, (size_t)nn * 2));
            CUdeviceptr args[12];
            args[0] = na;
            args[1] = nbp;
            args[2] = ncc;
            args[3] = nd;
            args[4] = ne;
            args[5] = nf;
            args[6] = ng;
            args[7] = ngg;
            args[8] = nh;
            args[9] = nn8;
            args[10] = nn8b;
            args[11] = nn16;
            void *par[12];
            for (int i = 0; i < 12; i++)
                par[i] = &args[i];
            CHECK_CU(p_cuLaunchKernel(nfn, nn / 256, 1, 1, 256, 1, 1, 0,
                                     NULL, par, NULL));
            CHECK_CU(p_cuCtxSynchronize());
            CHECK_CU(p_cuMemcpyDtoH_v2(nhd, nd, (size_t)nn * 4));
            CHECK_CU(p_cuMemcpyDtoH_v2(nhe, ne, (size_t)nn * 8));
            CHECK_CU(p_cuMemcpyDtoH_v2(nhf, nf, (size_t)nn * 4));
            CHECK_CU(p_cuMemcpyDtoH_v2(nhh, nh, (size_t)nn * 2));
            CHECK_CU(p_cuMemcpyDtoH_v2(nh8, nn8, (size_t)nn));
            CHECK_CU(p_cuMemcpyDtoH_v2(nh8b, nn8b, (size_t)nn));
            CHECK_CU(p_cuMemcpyDtoH_v2(nh16, nn16, (size_t)nn * 2));
            uint32_t bad13 = 0, bd = 0, be = 0, bf = 0, bh = 0, b8 = 0,
                       b8b = 0, b16 = 0;
            for (uint32_t i = 0; i < nn; i++) {
                uint32_t u = nha[i];
                int32_t sw = (int8_t)nhb[i];
                uint32_t q = u / 7, r = u % 7;
                int32_t sq = sw / 7, sr = sw % 7;
                uint32_t wantd = q | ((uint32_t)(sq + 4) << 6) |
                                 (r << 20) | ((uint32_t)(sr + 16) << 26);
                uint64_t wante = (uint64_t)nhc[i] + q + (u & 15);
                uint32_t wantf;
                uint32_t wanth;
                if (i < 8) {
                    wantf = lanef[i];
                    wanth = laneh[i];
                } else {
                    float gv = (float)((int)(i % 12) - 4) * 0.25f;
                    float ggv = (float)((int)(i % 7) - 2) * 0.25f;
                    float s = gv + ggv, p = gv * ggv, z = s + p;
                    memcpy(&wantf, &z, 4);
                    float hs = gv < ggv ? s : p;
                    uint32_t hb;
                    memcpy(&hb, &hs, 4);
                    wanth = host_f32_to_f16(hb);
                }
                uint32_t want8 = (u * 3 + 1) & 0xFFu;
                uint32_t want16 = (u * 1000 + 1) & 0xFFFFu;
                if (nhd[i] != wantd) {
                    if (bd < 3)
                        printf("  narrow d[%u]: got %u want %u\n", i,
                               nhd[i], wantd);
                    bd++;
                }
                if (nhe[i] != wante) {
                    if (be < 3)
                        printf("  narrow e[%u]: got %llu want %llu\n", i,
                               (unsigned long long)nhe[i],
                               (unsigned long long)wante);
                    be++;
                }
                if (nhf[i] != wantf) {
                    if (bf < 3)
                        printf("  narrow f[%u]: got %x want %x\n", i,
                               nhf[i], wantf);
                    bf++;
                }
                if (nhh[i] != wanth) {
                    if (bh < 3)
                        printf("  narrow h[%u]: got %x want %x\n", i, nhh[i],
                               wanth);
                    bh++;
                }
                if (nh8[i] != want8) {
                    if (b8 < 3)
                        printf("  narrow n8[%u]: got %u want %u\n", i,
                               nh8[i], want8);
                    b8++;
                }
                if (nh8b[i] != want8) {
                    if (b8b < 3)
                        printf("  narrow n8b[%u]: got %u want %u\n", i,
                               nh8b[i], want8);
                    b8b++;
                }
                if (nh16[i] != want16) {
                    if (b16 < 3)
                        printf("  narrow n16[%u]: got %u want %u\n", i,
                               nh16[i], want16);
                    b16++;
                }
                bad13 = bd + be + bf + bh + b8 + b8b + b16;
            }
            printf("  narrow fails: d=%u e=%u f=%u h=%u n8=%u n8b=%u n16=%u\n",
                   bd, be, bf, bh, b8, b8b, b16);
            CHECK(bad13 == 0);
            printf("  driver narrow: %u elements ok\n", nn);
        }
        free(nha);
        free(nhb);
        free(nhc);
        free(nhd);
        free(nhe);
        free(nhf);
        free(nhg);
        free(nhgg);
        free(nhh);
        free(nh8);
        free(nh8b);
        free(nh16);
        CHECK_CU(p_cuMemFree_v2(na));
        CHECK_CU(p_cuMemFree_v2(nbp));
        CHECK_CU(p_cuMemFree_v2(ncc));
        CHECK_CU(p_cuMemFree_v2(nd));
        CHECK_CU(p_cuMemFree_v2(ne));
        CHECK_CU(p_cuMemFree_v2(nf));
        CHECK_CU(p_cuMemFree_v2(ng));
        CHECK_CU(p_cuMemFree_v2(ngg));
        CHECK_CU(p_cuMemFree_v2(nh));
        CHECK_CU(p_cuMemFree_v2(nn8));
        CHECK_CU(p_cuMemFree_v2(nn8b));
        CHECK_CU(p_cuMemFree_v2(nn16));
        CHECK_CU(p_cuModuleUnload(nmod));
        CHECK_CU(p_cuLinkDestroy(nls));
        free(nb);
    }
    // Real numba f16 kernel (tests/ptx/f16_numba.ptx, harvested):
    // halves live as bits in .b16 regs (ld.u16 in, cvt.f64.f16 widen,
    // fma.rn.f64 accumulate, cvt.rn.f16.f64 narrow, st.u16 out).
    // Same 22-param layout as the loop kernels: data at 5/12/19,
    // scratch at 0, strides-1 at 6/13/20. N=1024 exact grid.
    {
        snprintf(path, sizeof(path), "%s/f16_numba.ptx",
                 ACUDA_TEST_PTX_DIR);
        char *nb = read_file(path);
        if (!nb)
            return 1;
        CUlinkState nls = NULL;
        CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &nls));
        CHECK_CU(p_cuLinkAddData(nls, 1, nb, strlen(nb), "f16nb", 0,
                                 NULL, NULL));
        void *ncubin = NULL;
        size_t ncubin_size = 0;
        CHECK_CU(p_cuLinkComplete(nls, &ncubin, &ncubin_size));
        CUmodule nmod = NULL;
        CHECK_CU(p_cuModuleLoadData(&nmod, ncubin));
        CUfunction nfn = NULL;
        CHECK_CU(p_cuModuleGetFunction(&nfn, nmod,
            "_ZN8__main__6f16addB2v1B74cw51cXTLSUwv1sCUt9Ww0FEw0NRRQPKiLTj0gIGIFp_2b2oLQFEYYkHSQB1OQAltAEAA_3d_3dE5ArrayIDhLi1E1C7mutable7alignedE5ArrayIDhLi1E1C7mutable7alignedE5ArrayIDhLi1E1C7mutable7alignedE"));
        int shv = -1;
        CHECK_CU(p_cuFuncGetAttribute(&shv,
                                      CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                                      nfn));
        CHECK(shv == 0); // no .shared in this kernel
        const uint32_t nn = 1024;
        CUdeviceptr na = 0, nbp = 0, nc = 0, nst = 0;
        CHECK_CU(p_cuMemAlloc_v2(&na, (size_t)nn * 2));
        CHECK_CU(p_cuMemAlloc_v2(&nbp, (size_t)nn * 2));
        CHECK_CU(p_cuMemAlloc_v2(&nc, (size_t)nn * 2));
        CHECK_CU(p_cuMemAlloc_v2(&nst, 8));
        uint16_t *nha = malloc((size_t)nn * 2);
        uint16_t *nhb = malloc((size_t)nn * 2);
        uint16_t *nhc = malloc((size_t)nn * 2);
        if (nha && nhb && nhc) {
            for (uint32_t i = 0; i < nn; i++) {
                float av = (float)((i % 7) + 1) * 0.5f;
                float bv = (float)((i % 5) + 1) * 0.25f;
                uint32_t ab, bb;
                memcpy(&ab, &av, 4);
                memcpy(&bb, &bv, 4);
                nha[i] = (uint16_t)host_f32_to_f16(ab);
                nhb[i] = (uint16_t)host_f32_to_f16(bb);
                nhc[i] = 0;
            }
            CHECK_CU(p_cuMemcpyHtoD_v2(na, nha, (size_t)nn * 2));
            CHECK_CU(p_cuMemcpyHtoD_v2(nbp, nhb, (size_t)nn * 2));
            CHECK_CU(p_cuMemcpyHtoD_v2(nc, nhc, (size_t)nn * 2));
            CUdeviceptr args[22];
            memset(args, 0, sizeof(args));
            uint64_t one = 1;
            args[0] = nst;
            args[5] = na;
            args[6] = one;
            args[12] = nbp;
            args[13] = one;
            args[19] = nc;
            args[20] = one;
            void *par[22];
            for (int i = 0; i < 22; i++)
                par[i] = &args[i];
            CHECK_CU(p_cuLaunchKernel(nfn, nn / 256, 1, 1, 256, 1, 1, 0,
                                     NULL, par, NULL));
            CHECK_CU(p_cuCtxSynchronize());
            CHECK_CU(p_cuMemcpyDtoH_v2(nhc, nc, (size_t)nn * 2));
            uint32_t bad14 = 0;
            for (uint32_t i = 0; i < nn; i++) {
                double av = (double)((i % 7) + 1) * 0.5;
                double bv = (double)((i % 5) + 1) * 0.25;
                double dot = av + bv * 2.0;
                uint64_t db;
                memcpy(&db, &dot, 8);
                uint32_t want = host_f64_to_f16(db);
                if (nhc[i] != want) {
                    if (bad14 < 5)
                        printf("  f16nb mismatch[%u]: got %x want %x\n", i,
                               nhc[i], want);
                    bad14++;
                }
            }
            CHECK(bad14 == 0);
            printf("  driver f16nb: %u elements ok\n", nn);
        }
        free(nha);
        free(nhb);
        free(nhc);
        CHECK_CU(p_cuMemFree_v2(na));
        CHECK_CU(p_cuMemFree_v2(nbp));
        CHECK_CU(p_cuMemFree_v2(nc));
        CHECK_CU(p_cuMemFree_v2(nst));
        CHECK_CU(p_cuModuleUnload(nmod));
        CHECK_CU(p_cuLinkDestroy(nls));
        free(nb);
    }
    // Looped dot-product (tests/ptx/loopdot.ptx): single-block while
    // loop with register-carried induction + accumulator (phis). Same
    // data layout as the unrolled dot; values exactly representable.
    {
        snprintf(path, sizeof(path), "%s/loopdot.ptx", ACUDA_TEST_PTX_DIR);
        char *nb = read_file(path);
        if (!nb)
            return 1;
        CUlinkState nls = NULL;
        CHECK_CU(p_cuLinkCreate(0, NULL, NULL, &nls));
        CHECK_CU(p_cuLinkAddData(nls, 1, nb, strlen(nb), "loopdot", 0,
                                 NULL, NULL));
        void *ncubin = NULL;
        size_t ncubin_size = 0;
        CHECK_CU(p_cuLinkComplete(nls, &ncubin, &ncubin_size));
        CUmodule nmod = NULL;
        CHECK_CU(p_cuModuleLoadData(&nmod, ncubin));
        CUfunction nfn = NULL;
        CHECK_CU(p_cuModuleGetFunction(&nfn, nmod, "loopdot"));
        const uint32_t nn = 4096, kk = 8;
        CUdeviceptr na = 0, nbp = 0, nc = 0;
        CHECK_CU(p_cuMemAlloc_v2(&na, (size_t)nn * kk * 4));
        CHECK_CU(p_cuMemAlloc_v2(&nbp, (size_t)nn * kk * 4));
        CHECK_CU(p_cuMemAlloc_v2(&nc, (size_t)nn * 4));
        float *nha = malloc((size_t)nn * kk * 4);
        float *nhb = malloc((size_t)nn * kk * 4);
        float *nhc = malloc((size_t)nn * 4);
        if (nha && nhb && nhc) {
            for (uint32_t i = 0; i < nn * kk; i++) {
                nha[i] = (float)(i % 5 + 1);
                nhb[i] = (float)(i % 3 + 1);
            }
            CHECK_CU(p_cuMemcpyHtoD_v2(na, nha, (size_t)nn * kk * 4));
            CHECK_CU(p_cuMemcpyHtoD_v2(nbp, nhb, (size_t)nn * kk * 4));
            uint32_t nnk = kk;
            CUdeviceptr args[3];
            args[0] = na;
            args[1] = nbp;
            args[2] = nc;
            void *par[4];
            for (int i = 0; i < 3; i++)
                par[i] = &args[i];
            par[3] = &nnk;
            CHECK_CU(p_cuLaunchKernel(nfn, nn / 256, 1, 1, 256, 1, 1, 0,
                                     NULL, par, NULL));
            CHECK_CU(p_cuCtxSynchronize());
            CHECK_CU(p_cuMemcpyDtoH_v2(nhc, nc, (size_t)nn * 4));
            uint32_t bad7 = 0;
            for (uint32_t i = 0; i < nn; i++) {
                double want = 0;
                for (uint32_t j = 0; j < kk; j++)
                    want += (double)nha[i * kk + j] * (double)nhb[i * kk + j];
                if ((double)nhc[i] != want) {
                    if (bad7 < 5)
                        printf("  loop mismatch[%u]: got %f want %f\n", i,
                               nhc[i], want);
                    bad7++;
                }
            }
            CHECK(bad7 == 0);
            printf("  driver loopdot: %u elements ok\n", nn);
        }
        free(nha);
        free(nhb);
        free(nhc);
        CHECK_CU(p_cuMemFree_v2(na));
        CHECK_CU(p_cuMemFree_v2(nbp));
        CHECK_CU(p_cuMemFree_v2(nc));
        CHECK_CU(p_cuModuleUnload(nmod));
        CHECK_CU(p_cuLinkDestroy(nls));
        free(nb);
    }
    CHECK_CU(p_cuModuleUnload(mod));
    CHECK_CU(p_cuLinkDestroy(ls)); // cubin buffer dies with the link state
    free(ptx);
    CHECK_CU(p_cuCtxDestroy_v2(ctx));
    dlclose(h);

    if (failures == 0)
        printf("driver integration test passed\n");
    return failures != 0;
}
