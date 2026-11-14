#!/usr/bin/env python3
"""Simple PyTorch end-to-end smoke tests for ACuda + the SASS lifter.

Not a full conformance suite: one small case per operator, compared
against the CPU implementation.  Run through run-torch-ops.sh (Linux) or
set the ACuda/rtxdisam environment manually and run this file.

Exit code is non-zero if a checked operator fails.  Operators that are
known-broken are listed as XFAIL and do not fail the run.
"""
import math
import sys

import torch

DEV = "cuda"
FAILED = []
CHECKED = 0


def check(name, got, want, rtol=1e-4, atol=1e-4):
    global CHECKED
    CHECKED += 1
    torch.cuda.synchronize()
    g = got.detach().cpu()
    c = want.detach().cpu()
    if g.dtype.is_floating_point:
        ok = torch.allclose(g, c, rtol=rtol, atol=atol, equal_nan=True)
    else:
        ok = torch.equal(g, c)
    print(("ok   " if ok else "FAIL"), name, flush=True)
    if not ok:
        FAILED.append(name)
        if g.dtype.is_floating_point:
            print("     max abs err", float((g.float() - c.float()).abs().max()),
                  flush=True)


def xfail(name, fn):
    """Run a known-broken operator without failing the suite."""
    try:
        fn()
        print("xpass", name, "(now works; move it out of xfail)", flush=True)
    except Exception as exc:
        print("xfail", name, ":", str(exc).splitlines()[0], flush=True)


def main():
    torch.manual_seed(0)

    # --- sort (cub radix / warp merge, 64-bit keys) -----------------
    for n in (256, 1000):
        x = torch.randint(-10**12, 10**12, (n,), device=DEV, dtype=torch.int64)
        check(f"sort int64 n={n}", torch.sort(x).values,
              torch.sort(x.cpu()).values)
    for n in (128, 1000):
        x = torch.randn(n, device=DEV, dtype=torch.float64)
        check(f"sort float64 n={n}", torch.sort(x).values,
              torch.sort(x.cpu()).values)

    # --- conv2d (cuDNN implicit GEMM, shared-memory tiling) ---------
    x = torch.randn(1, 2, 8, 8, device=DEV)
    w = torch.randn(3, 2, 3, 3, device=DEV)
    import torch.nn.functional as F
    check("conv2d (1,2,8,8) 3x3 pad=1",
          F.conv2d(x, w, padding=1), F.conv2d(x.cpu(), w.cpu(), padding=1),
          rtol=1e-4, atol=1e-4)

    # --- f64 sin/cos, including the Payne-Hanek slow path -----------
    vals = [-3e9, 1e10, 1e300, -0.75, 0.0, 1.5]
    x = torch.tensor(vals, device=DEV, dtype=torch.float64)
    ref = torch.tensor([math.sin(v) for v in vals], dtype=torch.float64)
    check("sin float64 (slow path)", torch.sin(x), ref)
    ref = torch.tensor([math.cos(v) for v in vals], dtype=torch.float64)
    check("cos float64 (slow path)", torch.cos(x), ref)

    # --- cumsum / var / std ----------------------------------------
    for dt in (torch.float32, torch.float64):
        x = torch.randn(16, 64, device=DEV, dtype=dt)
        check(f"cumsum {dt}", x.cumsum(1), x.cpu().cumsum(1),
              rtol=1e-5 if dt is torch.float32 else 1e-10,
              atol=1e-5 if dt is torch.float32 else 1e-10)
        check(f"var {dt}", x.var(1), x.cpu().var(1), rtol=1e-4, atol=1e-4)
        check(f"std {dt}", x.std(1), x.cpu().std(1), rtol=1e-4, atol=1e-4)

    # --- reductions -------------------------------------------------
    for dt in (torch.float32, torch.float64):
        x = torch.randn(16, 64, device=DEV, dtype=dt)
        xc = x.cpu()
        check(f"sum all {dt}", x.sum(), xc.sum())
        check(f"mean all {dt}", x.mean(), xc.mean())
        check(f"max all {dt}", x.max(), xc.max())
        check(f"sum dim1 {dt}", x.sum(1), xc.sum(1))
        check(f"mean dim1 {dt}", x.mean(1), xc.mean(1))
        check(f"max dim1 {dt}", x.max(1).values, xc.max(1).values)

    # --- matmul / softmax ------------------------------------------
    for dt in (torch.float32, torch.float64):
        a = torch.randn(16, 32, device=DEV, dtype=dt)
        b = torch.randn(32, 16, device=DEV, dtype=dt)
        check(f"matmul {dt}", a @ b, a.cpu() @ b.cpu(), rtol=1e-3, atol=1e-3)
        check(f"softmax {dt}", torch.softmax(a, 1),
              torch.softmax(a.cpu(), 1), rtol=1e-4, atol=1e-5)

    # --- median / kthvalue / mode (radix select, UFLO, 64-bit shared
    # --- atomics) ---------------------------------------------------
    for shape, dim in (((4, 4), 1), ((16, 16), 1), ((8, 4), 0)):
        x = torch.randn(*shape, device=DEV)
        check(f"median {shape} dim={dim}", torch.median(x, dim).values,
              torch.median(x.cpu(), dim).values)
        check(f"kthvalue {shape} dim={dim}", torch.kthvalue(x, 2, dim).values,
              torch.kthvalue(x.cpu(), 2, dim).values)
        check(f"mode {shape} dim={dim}", torch.mode(x, dim).values,
              torch.mode(x.cpu(), dim).values)
    x = torch.randn(2, 2000, device=DEV)
    check("median 2x2000 dim=1", torch.median(x, 1).values,
          torch.median(x.cpu(), 1).values)
    check("kthvalue 2x2000 dim=1", torch.kthvalue(x, 2, 1).values,
          torch.kthvalue(x.cpu(), 2, 1).values)
    # 1024-thread compute_mode instantiation: the dispatcher must stay
    # affordable for the driver's register allocator.
    check("mode 2x2000 dim=1", torch.mode(x, 1).values,
          torch.mode(x.cpu(), 1).values)

    # --- argmax / argmin indices (int64 pair high word) -------------
    for shape, dim in (((4, 8), 1), ((8, 4), 0), ((2, 2000), 1)):
        x = torch.randn(*shape, device=DEV)
        check(f"argmax {shape} dim={dim}", x.argmax(dim), x.cpu().argmax(dim))
        check(f"argmin {shape} dim={dim}", x.argmin(dim), x.cpu().argmin(dim))

    # --- topk (radix select + block-wide prefix scan) --------------
    for shape, k in (((4, 16), 4), ((16, 16), 4), ((2, 2000), 4),
                     ((4, 16), 1)):
        x = torch.randn(*shape, device=DEV)
        got_v, got_i = torch.topk(x, k, 1)
        want_v, want_i = torch.topk(x.cpu(), k, 1)
        check(f"topk {shape} k={k} values", got_v, want_v)
        check(f"topk {shape} k={k} indices", got_i, want_i)

    print(f"\n{CHECKED - len(FAILED)}/{CHECKED} checks passed")
    if FAILED:
        print("failed:", ", ".join(FAILED))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
