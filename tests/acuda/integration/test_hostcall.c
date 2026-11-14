// Split-dispatch host-call tests: unresolved `.extern .func` calls are
// compiled as a per-invocation PC state machine and serviced by
// registered callbacks. Covers branches, loops, divergence, predicated
// calls, pointer/narrow arguments, wide returns, and pointer spill.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "acuda/compiler.h"
#include "acuda/context.h"
#include "acuda/memory.h"
#include "acuda/stream.h"
#include "internal.h"

#ifndef ACUDA_TEST_PTX_DIR
#define ACUDA_TEST_PTX_DIR "tests/ptx"
#endif

static int failures;
#define CHECK(cond) do { \
    if (!(cond)) { \
        printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        failures++; \
    } \
} while (0)

static int callback_calls;
static uint64_t arg_u64(const uint8_t *args, size_t bytes);
static acuda_context_t nested_ctx;
static acuda_kernel_t *nested_kernel;
static acuda_buffer_t nested_buf;
static int nested_ok;

static int cb_nested(void *user, const uint8_t *args, size_t bytes,
                     uint8_t *ret, size_t ret_bytes) {
    (void)user;
    callback_calls++;
    if (!nested_ctx || !nested_kernel || !nested_buf)
        return -2;
    acuda_status_t st = acuda_launch_kernel(nested_ctx, nested_kernel,
                                            &nested_buf, 1, NULL, 0, 1, 1,
                                            NULL);
    if (st != ACUDA_SUCCESS)
        return -3;
    uint32_t v = 0;
    if (acuda_memcpy_d2h(nested_ctx, &v, nested_buf, sizeof(v), NULL) !=
        ACUDA_SUCCESS)
        return -4;
    if (v != 7)
        return -5;
    nested_ok = 1;
    uint64_t arg = arg_u64(args, bytes) + 1;
    memset(ret, 0, ret_bytes);
    if (ret_bytes >= sizeof(arg))
        memcpy(ret, &arg, sizeof(arg));
    return 0;
}


static uint64_t arg_u64(const uint8_t *args, size_t bytes) {
    uint64_t v = 0;
    memcpy(&v, args, bytes < sizeof(v) ? bytes : sizeof(v));
    return v;
}

static int cb_add1(void *user, const uint8_t *args, size_t bytes,
                   uint8_t *ret, size_t ret_bytes) {
    (void)user;
    callback_calls++;
    uint64_t v = arg_u64(args, bytes) + 1;
    memset(ret, 0, ret_bytes);
    if (ret_bytes >= sizeof(v))
        memcpy(ret, &v, sizeof(v));
    return 0;
}

static int cb_add2(void *user, const uint8_t *args, size_t bytes,
                   uint8_t *ret, size_t ret_bytes) {
    (void)user;
    callback_calls++;
    uint64_t v = arg_u64(args, bytes) + 2;
    memset(ret, 0, ret_bytes);
    if (ret_bytes >= sizeof(v))
        memcpy(ret, &v, sizeof(v));
    return 0;
}

static int cb_ptr(void *user, const uint8_t *args, size_t bytes,
                  uint8_t *ret, size_t ret_bytes) {
    (void)user;
    callback_calls++;
    CHECK(bytes >= 8);
    uint64_t p = 0, v = 0;
    memcpy(&p, args, sizeof(p));
    if (p)
        memcpy(&v, (const void *)(uintptr_t)p, sizeof(v));
    v += 1;
    memset(ret, 0, ret_bytes);
    if (ret_bytes >= sizeof(v))
        memcpy(ret, &v, sizeof(v));
    return 0;
}

static int cb_narrow(void *user, const uint8_t *args, size_t bytes,
                     uint8_t *ret, size_t ret_bytes) {
    (void)user;
    callback_calls++;
    CHECK(bytes >= 10);
    uint8_t b = 0;
    uint16_t w = 0;
    memcpy(&b, args, 1);
    memcpy(&w, args + 8, 2);
    uint64_t v = (uint64_t)b | ((uint64_t)w << 8);
    memset(ret, 0, ret_bytes);
    if (ret_bytes >= sizeof(v))
        memcpy(ret, &v, sizeof(v));
    return 0;
}

static int cb_sum10(void *user, const uint8_t *args, size_t bytes,
                    uint8_t *ret, size_t ret_bytes) {
    (void)user;
    callback_calls++;
    uint64_t v = 0;
    for (int i = 0; i < 10 && (size_t)(i + 1) * 8 <= bytes; i++) {
        uint64_t x = 0;
        memcpy(&x, args + i * 8, 8);
        v += x;
    }
    memset(ret, 0, ret_bytes);
    if (ret_bytes >= sizeof(v))
        memcpy(ret, &v, sizeof(v));
    return 0;
}

static int cb_write99(void *user, const uint8_t *args, size_t bytes,
                      uint8_t *ret, size_t ret_bytes) {
    (void)user;
    callback_calls++;
    uint64_t p = 0;
    memcpy(&p, args, sizeof(p));
    if (p && bytes >= sizeof(p)) {
        uint64_t v = 99;
        memcpy((void *)(uintptr_t)p, &v, sizeof(v));
    }
    uint64_t v = 99;
    memset(ret, 0, ret_bytes);
    if (ret_bytes >= sizeof(v))
        memcpy(ret, &v, sizeof(v));
    return 0;
}

static int cb_ret16(void *user, const uint8_t *args, size_t bytes,
                    uint8_t *ret, size_t ret_bytes) {
    (void)user;
    callback_calls++;
    uint64_t v = arg_u64(args, bytes);
    uint64_t r0 = v + 1, r1 = v + 2;
    memset(ret, 0, ret_bytes);
    memcpy(ret, &r0, sizeof(r0));
    if (ret_bytes >= 16)
        memcpy(ret + 8, &r1, sizeof(r1));
    return 0;
}

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    long n = ftell(f);
    if (n < 0 || fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return NULL;
    }
    char *buf = malloc((size_t)n + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) {
        free(buf);
        fclose(f);
        return NULL;
    }
    buf[n] = '\0';
    fclose(f);
    return buf;
}

static int run_case(acuda_context_t ctx, const char *file, const char *entry,
                    uint64_t in0, uint64_t in1, uint32_t block,
                    acuda_host_call_fn cb, const uint64_t *want, int nwant,
                    int want_calls) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", ACUDA_TEST_PTX_DIR, file);
    char *ptx = read_file(path);
    CHECK(ptx != NULL);
    if (!ptx)
        return 1;

    char err[1024] = {0};
    acuda_kernel_t *k = NULL;
    acuda_status_t st =
        acuda_compile_ptx(ptx, entry, &k, err, sizeof(err), NULL);
    if (st != ACUDA_SUCCESS)
        printf("FAIL %s: compile: %s\n", file, err);
    CHECK(st == ACUDA_SUCCESS);
    if (st != ACUDA_SUCCESS) {
        free(ptx);
        return 1;
    }
    acuda_kernel_info_t info;
    memset(&info, 0, sizeof(info));
    CHECK(acuda_kernel_info(k, &info) == ACUDA_SUCCESS);
    CHECK(info.host_state_machine == 1);
    CHECK(info.host_segment_count == 1);
    CHECK(info.num_ptr_params == 2);

    acuda_buffer_t in = NULL, out = NULL;
    CHECK(acuda_malloc(ctx, &in, 32) == ACUDA_SUCCESS);
    CHECK(acuda_malloc(ctx, &out, 32) == ACUDA_SUCCESS);
    uint8_t ib[32] = {0}, ob[32] = {0};
    memcpy(ib, &in0, 8);
    memcpy(ib + 8, &in1, 8);
    CHECK(acuda_memcpy_h2d(ctx, in, ib, sizeof(ib), NULL) == ACUDA_SUCCESS);
    CHECK(acuda_memcpy_h2d(ctx, out, ob, sizeof(ob), NULL) == ACUDA_SUCCESS);

    CHECK(acuda_host_call_register(ctx, "foobar", cb, NULL) ==
          ACUDA_SUCCESS);
    callback_calls = 0;
    acuda_buffer_t bufs[2] = {in, out};
    st = acuda_launch_kernel(ctx, k, bufs, 2, NULL, 0, 1, block, NULL);
    if (st != ACUDA_SUCCESS)
        printf("FAIL %s: launch: %s\n", file, acuda_strerror(st));
    CHECK(st == ACUDA_SUCCESS);
    CHECK(callback_calls == want_calls);
    CHECK(acuda_memcpy_d2h(ctx, ob, out, sizeof(ob), NULL) ==
          ACUDA_SUCCESS);
    for (int i = 0; i < nwant; i++) {
        uint64_t got;
        memcpy(&got, ob + 8 * i, 8);
        if (got != want[i]) {
            printf("FAIL %s: out[%d]=%llu want %llu\n", file, i,
                   (unsigned long long)got, (unsigned long long)want[i]);
            failures++;
        }
    }
    CHECK(acuda_host_call_unregister(ctx, "foobar") == ACUDA_SUCCESS);
    acuda_kernel_free(k);
    acuda_free(ctx, in);
    acuda_free(ctx, out);
    free(ptx);
    return 0;
}

int main(void) {
    acuda_context_t ctx = NULL;
    CHECK(acuda_init(&ctx) == ACUDA_SUCCESS);
    if (!ctx)
        return 1;

    {
        char path[1024];
        snprintf(path, sizeof(path), "%s/nested.ptx", ACUDA_TEST_PTX_DIR);
        char *nptx = read_file(path);
        CHECK(nptx != NULL);
        if (nptx) {
            char nerr[1024] = {0};
            CHECK(acuda_compile_ptx(nptx, "nested", &nested_kernel, nerr,
                                    sizeof(nerr), NULL) == ACUDA_SUCCESS);
            CHECK(acuda_malloc(ctx, &nested_buf, 4) == ACUDA_SUCCESS);
            nested_ctx = ctx;
            free(nptx);
        }
    }

    {
        uint64_t want[] = {0x1235};
        run_case(ctx, "hostcall.ptx", "hostcall", 0x1234, 0, 1,
                 cb_add1, want, 1, 1);
    }
    {
        uint64_t want[] = {0x1235};
        nested_ok = 0;
        run_case(ctx, "hostcall.ptx", "hostcall", 0x1234, 0, 1,
                 cb_nested, want, 1, 1);
        CHECK(nested_ok == 1);
    }
    {
        uint64_t want[] = {0x9c};
        run_case(ctx, "hostcall2.ptx", "hostcall2", 0x99, 0, 1,
                 cb_add1, want, 1, 2);
    }
    {
        // brx.idx jump table feeding a host-call argument.
        uint64_t want0[] = {0x101};
        run_case(ctx, "hostcall_brx.ptx", "brxhc", 0, 0, 1, cb_add1,
                 want0, 1, 1);
        uint64_t want1[] = {0x201};
        run_case(ctx, "hostcall_brx.ptx", "brxhc", 1, 0, 1, cb_add1,
                 want1, 1, 1);
    }
    {
        // Multi-target indirect dispatch in a host-call kernel: the
        // callback result selects the table entry before dispatch.
        uint64_t want0[] = {0x11};
        run_case(ctx, "hostcall_dispatch.ptx", "hostcall_dispatch", 0, 0,
                 1, cb_add1, want0, 1, 1);
        uint64_t want1[] = {0x22};
        run_case(ctx, "hostcall_dispatch.ptx", "hostcall_dispatch", 1, 0,
                 1, cb_add1, want1, 1, 1);
    }
    {
        uint64_t want[] = {6};
        run_case(ctx, "hostcall_branch.ptx", "hostcall_branch", 5, 0, 1,
                 cb_add1, want, 1, 1);
        uint64_t zero[] = {0};
        run_case(ctx, "hostcall_branch.ptx", "hostcall_branch", 0, 0, 1,
                 cb_add1, zero, 1, 0);
    }
    {
        uint64_t want[] = {0x13};
        run_case(ctx, "hostcall_loop.ptx", "hostcall_loop", 0x10, 0, 1,
                 cb_add1, want, 1, 3);
    }
    {
        uint64_t want[] = {0x12, 0x10};
        run_case(ctx, "hostcall_div.ptx", "hostcall_div", 0x10, 0, 2,
                 cb_add2, want, 2, 1);
    }
    {
        uint64_t want[] = {0x12, 0x10};
        run_case(ctx, "hostcall_pred.ptx", "hostcall_pred", 0x10, 0, 2,
                 cb_add2, want, 2, 1);
    }
    {
        uint64_t want[] = {0x1235};
        run_case(ctx, "hostcall_ptr.ptx", "hostcall_ptr", 0x1234, 0, 1,
                 cb_ptr, want, 1, 1);
    }
    {
        uint64_t want[] = {0x7f7f};
        run_case(ctx, "hostcall_narrow.ptx", "hostcall_narrow", 0, 0, 1,
                 cb_narrow, want, 1, 1);
    }
    {
        uint64_t want[] = {0x101, 0x102};
        run_case(ctx, "hostcall_ret16.ptx", "hostcall_ret16", 0x100, 0, 1,
                 cb_ret16, want, 2, 1);
    }
    {
        uint64_t want[] = {0, 0x1235};
        run_case(ctx, "hostcall_movptr.ptx", "hostcall_movptr", 0x1234, 0,
                 1, cb_add1, want, 2, 1);
    }
    {
        uint64_t want[] = {0x1234};
        run_case(ctx, "hostcall_local.ptx", "hostcall_local", 0x1234, 0,
                 1, cb_add1, want, 1, 1);
    }
    {
        uint64_t want[] = {0x1234};
        run_case(ctx, "hostcall_shared.ptx", "hostcall_shared", 0x1234, 0,
                 2, cb_add1, want, 1, 2);
    }
    {
        uint64_t want[] = {99};
        run_case(ctx, "hostcall_sharedptr.ptx", "hostcall_sharedptr",
                 0x1234, 0, 1, cb_write99, want, 1, 1);
    }
    {
        uint64_t want[] = {55};
        run_case(ctx, "hostcall_args.ptx", "hostcall_args", 0, 0, 1,
                 cb_sum10, want, 1, 1);
    }

    {
        uint64_t want[] = {0, 1, 0x1234, 0};
        run_case(ctx, "hostcall_atom.ptx", "hostcall_atom", 0x1234, 0, 2,
                 cb_add1, want, 4, 2);
    }
    {
        uint64_t want[] = {0x1235};
        run_case(ctx, "hostcall_selp.ptx", "hostcall_selp", 0x1234, 0, 2,
                 cb_add1, want, 1, 2);
    }
    {
        uint64_t want[] = {0x1235, 0x1235};
        run_case(ctx, "hostcall_mad.ptx", "hostcall_mad", 0x1234, 0, 2,
                 cb_add1, want, 2, 2);
    }    {
        uint64_t want[] = {0, 1, 0x1234, 0};
        run_case(ctx, "hostcall_atom.ptx", "hostcall_atom", 0x1234, 0, 2,
                 cb_add1, want, 4, 2);
    }
    {
        uint64_t want[] = {0x1235};
        run_case(ctx, "hostcall_mix.ptx", "hostcall_mix", 0x1234, 0, 2,
                 cb_add1, want, 1, 2);
    }
    {
        // Dynamic .extern .shared: launch with 8 dynamic shared bytes.
        char path[1024];
        snprintf(path, sizeof(path), "%s/hostcall_dynsh.ptx",
                 ACUDA_TEST_PTX_DIR);
        char *sptx = read_file(path);
        CHECK(sptx != NULL);
        acuda_kernel_t *sk = NULL;
        char serr[1024] = {0};
        if (sptx)
            CHECK(acuda_compile_ptx(sptx, "hostcall_dynsh", &sk, serr,
                                    sizeof(serr), NULL) == ACUDA_SUCCESS);
        CHECK(sk != NULL);
        if (sk) {
            acuda_buffer_t sin = NULL, sout = NULL;
            CHECK(acuda_malloc(ctx, &sin, 8) == ACUDA_SUCCESS);
            CHECK(acuda_malloc(ctx, &sout, 8) == ACUDA_SUCCESS);
            uint64_t sv = 0x1234, so = 0;
            CHECK(acuda_memcpy_h2d(ctx, sin, &sv, 8, NULL) ==
                  ACUDA_SUCCESS);
            CHECK(acuda_memcpy_h2d(ctx, sout, &so, 8, NULL) ==
                  ACUDA_SUCCESS);
            CHECK(acuda_host_call_register(ctx, "foobar", cb_add1, NULL) ==
                  ACUDA_SUCCESS);
            callback_calls = 0;
            acuda_buffer_t sbufs[2] = {sin, sout};
            acuda_status_t sst = acuda_launch_kernel_ex(
                ctx, sk, sbufs, 2, NULL, 0, 1, 2, 8, NULL);
            CHECK(sst == ACUDA_SUCCESS);
            CHECK(callback_calls == 2);
            CHECK(acuda_memcpy_d2h(ctx, &so, sout, 8, NULL) ==
                  ACUDA_SUCCESS);
            CHECK(so == 0x1234);
            CHECK(acuda_host_call_unregister(ctx, "foobar") ==
                  ACUDA_SUCCESS);
            acuda_kernel_free(sk);
            acuda_free(ctx, sin);
            acuda_free(ctx, sout);
        }
        free(sptx);
    }
    {
        // A pointer register holds different global homes on different
        // threads; the store after the call must dispatch on the saved
        // home tag. Thread 0 writes through input, thread 1 through
        // output.
        char path[1024];
        snprintf(path, sizeof(path), "%s/hostcall_dyn.ptx",
                 ACUDA_TEST_PTX_DIR);
        char *dptx = read_file(path);
        CHECK(dptx != NULL);
        acuda_kernel_t *dk = NULL;
        char derr[1024] = {0};
        if (dptx)
            CHECK(acuda_compile_ptx(dptx, "hostcall_dyn", &dk, derr,
                                    sizeof(derr), NULL) == ACUDA_SUCCESS);
        CHECK(dk != NULL);
        if (dk) {
            acuda_buffer_t din = NULL, dout = NULL;
            CHECK(acuda_malloc(ctx, &din, 16) == ACUDA_SUCCESS);
            CHECK(acuda_malloc(ctx, &dout, 16) == ACUDA_SUCCESS);
            uint8_t dib[16] = {0}, dob[16] = {0};
            uint64_t dv = 0x1234;
            memcpy(dib, &dv, 8);
            CHECK(acuda_memcpy_h2d(ctx, din, dib, sizeof(dib), NULL) ==
                  ACUDA_SUCCESS);
            CHECK(acuda_memcpy_h2d(ctx, dout, dob, sizeof(dob), NULL) ==
                  ACUDA_SUCCESS);
            CHECK(acuda_host_call_register(ctx, "foobar", cb_add1, NULL) ==
                  ACUDA_SUCCESS);
            callback_calls = 0;
            acuda_buffer_t dbufs[2] = {din, dout};
            acuda_status_t dst = acuda_launch_kernel(
                ctx, dk, dbufs, 2, NULL, 0, 1, 2, NULL);
            CHECK(dst == ACUDA_SUCCESS);
            CHECK(callback_calls == 2);
            CHECK(acuda_memcpy_d2h(ctx, dib, din, sizeof(dib), NULL) ==
                  ACUDA_SUCCESS);
            CHECK(acuda_memcpy_d2h(ctx, dob, dout, sizeof(dob), NULL) ==
                  ACUDA_SUCCESS);
            uint64_t iv = 0, ov = 0;
            memcpy(&iv, dib + 8, 8);
            memcpy(&ov, dob + 8, 8);
            CHECK(iv == 0x1235);
            CHECK(ov == 0x1235);
            CHECK(acuda_host_call_unregister(ctx, "foobar") ==
                  ACUDA_SUCCESS);
            acuda_kernel_free(dk);
            acuda_free(ctx, din);
            acuda_free(ctx, dout);
        }
        free(dptx);
    }

    {
        // Raw driver/cudart path must auto-detect host-call kernels.
        char path[1024];
        snprintf(path, sizeof(path), "%s/hostcall.ptx", ACUDA_TEST_PTX_DIR);
        char *rptx = read_file(path);
        CHECK(rptx != NULL);
        acuda_kernel_t *rk = NULL;
        char rerr[1024] = {0};
        if (rptx)
            CHECK(acuda_compile_ptx(rptx, "hostcall", &rk, rerr,
                                    sizeof(rerr), NULL) == ACUDA_SUCCESS);
        CHECK(rk != NULL);
        if (rk) {
            acuda_buffer_t rin = NULL, rout = NULL;
            CHECK(acuda_malloc(ctx, &rin, 8) == ACUDA_SUCCESS);
            CHECK(acuda_malloc(ctx, &rout, 8) == ACUDA_SUCCESS);
            uint64_t rv = 0x1234, ro = 0;
            CHECK(acuda_memcpy_h2d(ctx, rin, &rv, 8, NULL) ==
                  ACUDA_SUCCESS);
            CHECK(acuda_memcpy_h2d(ctx, rout, &ro, 8, NULL) ==
                  ACUDA_SUCCESS);
            CHECK(acuda_host_call_register(ctx, "foobar", cb_add1, NULL) ==
                  ACUDA_SUCCESS);
            callback_calls = 0;
            VkBuffer raw[2] = {rin->buffer, rout->buffer};
            size_t off[2] = {0, 0};
            size_t range[2] = {rin->size, rout->size};
            acuda_status_t rst = acuda_launch_raw(
                ctx, rk, raw, off, range, 2, NULL, 0, 1, 1, 0, NULL);
            CHECK(rst == ACUDA_SUCCESS);
            CHECK(callback_calls == 1);
            CHECK(acuda_memcpy_d2h(ctx, &ro, rout, 8, NULL) ==
                  ACUDA_SUCCESS);
            CHECK(ro == 0x1235);
            CHECK(acuda_host_call_unregister(ctx, "foobar") ==
                  ACUDA_SUCCESS);
            acuda_kernel_free(rk);
            acuda_free(ctx, rin);
            acuda_free(ctx, rout);
        }
        free(rptx);
    }

    {
        uint64_t want[] = {0x1234, 0x5678};
        run_case(ctx, "hostcall_gimg.ptx", "hostcall_gimg", 0x1234, 0, 2,
                 cb_add1, want, 2, 2);
    }
    {
        uint64_t want[] = {0x1235};
        run_case(ctx, "hostcall_paddr.ptx", "hostcall_paddr", 0x1234, 0, 2,
                 cb_add1, want, 1, 2);
    }
    {
        // Statically resolvable indirect call through a register.
        char path[1024];
        snprintf(path, sizeof(path), "%s/indirect.ptx", ACUDA_TEST_PTX_DIR);
        char *iptx = read_file(path);
        CHECK(iptx != NULL);
        acuda_kernel_t *ik = NULL;
        char ierr[1024] = {0};
        if (iptx)
            CHECK(acuda_compile_ptx(iptx, "indirect", &ik, ierr,
                                    sizeof(ierr), NULL) == ACUDA_SUCCESS);
        CHECK(ik != NULL);
        if (ik) {
            acuda_buffer_t iin = NULL, iout = NULL;
            CHECK(acuda_malloc(ctx, &iin, 8) == ACUDA_SUCCESS);
            CHECK(acuda_malloc(ctx, &iout, 8) == ACUDA_SUCCESS);
            uint64_t iv = 0x1234, ov = 0;
            CHECK(acuda_memcpy_h2d(ctx, iin, &iv, 8, NULL) ==
                  ACUDA_SUCCESS);
            CHECK(acuda_memcpy_h2d(ctx, iout, &ov, 8, NULL) ==
                  ACUDA_SUCCESS);
            acuda_buffer_t ibufs[2] = {iin, iout};
            CHECK(acuda_launch_kernel(ctx, ik, ibufs, 2, NULL, 0, 1, 1,
                                      NULL) == ACUDA_SUCCESS);
            CHECK(acuda_memcpy_d2h(ctx, &ov, iout, 8, NULL) ==
                  ACUDA_SUCCESS);
            CHECK(ov == 0x1235);
            acuda_kernel_free(ik);
            acuda_free(ctx, iin);
            acuda_free(ctx, iout);
        }
        free(iptx);
    }

    {
        // Plain raw path: the runtime maps buffers created by
        // acuda_malloc itself, so global pointer callback arguments
        // work without acuda_launch_raw_ex.
        char path[1024];
        snprintf(path, sizeof(path), "%s/hostcall_ptr.ptx",
                 ACUDA_TEST_PTX_DIR);
        char *pptx = read_file(path);
        CHECK(pptx != NULL);
        acuda_kernel_t *pk = NULL;
        char perr[1024] = {0};
        if (pptx)
            CHECK(acuda_compile_ptx(pptx, "hostcall_ptr", &pk, perr,
                                    sizeof(perr), NULL) == ACUDA_SUCCESS);
        CHECK(pk != NULL);
        if (pk) {
            acuda_buffer_t pin = NULL, pout = NULL;
            CHECK(acuda_malloc(ctx, &pin, 8) == ACUDA_SUCCESS);
            CHECK(acuda_malloc(ctx, &pout, 8) == ACUDA_SUCCESS);
            uint64_t pv = 0x1234, po = 0;
            CHECK(acuda_memcpy_h2d(ctx, pin, &pv, 8, NULL) ==
                  ACUDA_SUCCESS);
            CHECK(acuda_memcpy_h2d(ctx, pout, &po, 8, NULL) ==
                  ACUDA_SUCCESS);
            CHECK(acuda_host_call_register(ctx, "foobar", cb_ptr, NULL) ==
                  ACUDA_SUCCESS);
            callback_calls = 0;
            VkBuffer praw[2] = {pin->buffer, pout->buffer};
            size_t poff[2] = {0, 0};
            size_t prange[2] = {pin->size, pout->size};
            acuda_status_t pst = acuda_launch_raw(
                ctx, pk, praw, poff, prange, 2, NULL, 0, 1, 1, 0, NULL);
            CHECK(pst == ACUDA_SUCCESS);
            CHECK(callback_calls == 1);
            CHECK(acuda_memcpy_d2h(ctx, &po, pout, 8, NULL) ==
                  ACUDA_SUCCESS);
            CHECK(po == 0x1235);
            CHECK(acuda_host_call_unregister(ctx, "foobar") ==
                  ACUDA_SUCCESS);
            acuda_kernel_free(pk);
            acuda_free(ctx, pin);
            acuda_free(ctx, pout);
        }
        free(pptx);
    }

    if (nested_kernel)
        acuda_kernel_free(nested_kernel);
    if (nested_buf)
        acuda_free(ctx, nested_buf);
    acuda_shutdown(ctx);
    if (failures) {
        printf("hostcall test: %d failure(s)\n", failures);
        return 1;
    }
    printf("hostcall test passed\n");
    return 0;
}
