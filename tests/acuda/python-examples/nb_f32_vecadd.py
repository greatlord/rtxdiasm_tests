"""Fast Python repro of C test_driver numba-.func section (Stage 6).

Loads tests/ptx/nb_f32.ptx (.func + retval, 22 .b64 params),
launches with status ptr at 0, data ptrs at 5/12/19, N at 17,
masks 6/13/20 = 0, elsizes 7/14/21 = 4; verifies 4096 floats.

Usage (from repo root):
    "/home/magnusolsen/git/aicode-ptx/virtual python/ven-cu134/bin/python" rtxdiasm_modules/rtxdiasm_tests/tests/acuda/python-examples/nb_f32_vecadd.py
"""
import ctypes
import os
import sys

try:
    import numpy as np
except ImportError:
    sys.exit("numpy not found: run with '/home/magnusolsen/git/aicode-ptx/virtual python/ven-cu134/bin/python'")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ACUDA_ROOT = os.path.join(REPO_ROOT, "src", "acuda")
TESTS_ACUDA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = ACUDA_ROOT


def find_lib():
    override = os.environ.get("ACUDA_LIBCUDA")
    if override:
        return override
    for cand in ("bin/debug/libcuda.so.1.0.0",
                 "bin/release/libcuda.so.1.0.0"):
        path = os.path.join(ROOT, cand)
        if os.path.exists(path):
            return path
    sys.exit("libcuda.so not found: build first")


LIB = find_lib()
PTX = os.path.join(TESTS_ACUDA_DIR, "ptx", "nb_f32.ptx")
ENTRY = (b"_ZN8__main__10vector_addB2v1B74cw51cXTLSUwv1sCUt9Ww0FEw0NRRQPKiLTj0gIGIFp_2b2o"
         b"LQFEYYkHSQB1OQAltAEAA_3d_3dE5ArrayIfLi1E1A7mutable7alignedE5ArrayIfLi1E1A7mutable7a"
         b"lignedE5ArrayIfLi1E1A7mutable7alignedE")

NN = 4096

cuda = ctypes.CDLL(LIB)
cuda.cuInit.argtypes = [ctypes.c_uint]
cuda.cuInit.restype = ctypes.c_int
cuda.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                ctypes.c_uint, ctypes.c_int]
cuda.cuCtxCreate_v2.restype = ctypes.c_int
cuda.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
cuda.cuCtxDestroy_v2.restype = ctypes.c_int
cuda.cuCtxSynchronize.argtypes = []
cuda.cuCtxSynchronize.restype = ctypes.c_int
cuda.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_ulonglong),
                               ctypes.c_size_t]
cuda.cuMemAlloc_v2.restype = ctypes.c_int
cuda.cuMemFree_v2.argtypes = [ctypes.c_ulonglong]
cuda.cuMemFree_v2.restype = ctypes.c_int
cuda.cuMemcpyHtoD_v2.argtypes = [ctypes.c_ulonglong, ctypes.c_void_p,
                                 ctypes.c_size_t]
cuda.cuMemcpyHtoD_v2.restype = ctypes.c_int
cuda.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong,
                                 ctypes.c_size_t]
cuda.cuMemcpyDtoH_v2.restype = ctypes.c_int
cuda.cuLinkCreate.argtypes = [ctypes.c_uint, ctypes.c_void_p,
                              ctypes.c_void_p,
                              ctypes.POINTER(ctypes.c_void_p)]
cuda.cuLinkCreate.restype = ctypes.c_int
cuda.cuLinkAddData.argtypes = [ctypes.c_void_p, ctypes.c_int,
                               ctypes.c_void_p, ctypes.c_size_t,
                               ctypes.c_char_p, ctypes.c_uint,
                               ctypes.c_void_p, ctypes.c_void_p]
cuda.cuLinkAddData.restype = ctypes.c_int
cuda.cuLinkComplete.argtypes = [ctypes.c_void_p,
                                ctypes.POINTER(ctypes.c_void_p),
                                ctypes.POINTER(ctypes.c_size_t)]
cuda.cuLinkComplete.restype = ctypes.c_int
cuda.cuLinkDestroy.argtypes = [ctypes.c_void_p]
cuda.cuLinkDestroy.restype = ctypes.c_int
cuda.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                  ctypes.c_void_p]
cuda.cuModuleLoadData.restype = ctypes.c_int
cuda.cuModuleUnload.argtypes = [ctypes.c_void_p]
cuda.cuModuleUnload.restype = ctypes.c_int
cuda.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                     ctypes.c_void_p, ctypes.c_char_p]
cuda.cuModuleGetFunction.restype = ctypes.c_int
cuda.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + \
    [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
cuda.cuLaunchKernel.restype = ctypes.c_int


def check(code, what):
    if code != 0:
        raise RuntimeError(f"{what} failed: {code}")


def main():
    with open(PTX, "rb") as f:
        ptx = f.read()
    check(cuda.cuInit(0), "cuInit")
    ctx = ctypes.c_void_p()
    check(cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, 0), "cuCtxCreate")

    nbytes = NN * 4
    na, nbp, nc, nst = (ctypes.c_ulonglong() for _ in range(4))
    check(cuda.cuMemAlloc_v2(ctypes.byref(na), nbytes), "alloc a")
    check(cuda.cuMemAlloc_v2(ctypes.byref(nbp), nbytes), "alloc b")
    check(cuda.cuMemAlloc_v2(ctypes.byref(nc), nbytes), "alloc c")
    check(cuda.cuMemAlloc_v2(ctypes.byref(nst), 8), "alloc status")

    ha = (np.arange(NN, dtype=np.float32) * 0.5).astype(np.float32)
    hb = (np.arange(NN, dtype=np.float32) * 2.0 + 1.0).astype(np.float32)
    hc = np.zeros(NN, dtype=np.float32)
    check(cuda.cuMemcpyHtoD_v2(na.value, ha.ctypes.data, nbytes), "HtoD a")
    check(cuda.cuMemcpyHtoD_v2(nbp.value, hb.ctypes.data, nbytes), "HtoD b")

    ls = ctypes.c_void_p()
    check(cuda.cuLinkCreate(0, None, None, ctypes.byref(ls)), "link")
    check(cuda.cuLinkAddData(ls, 1, ptx, len(ptx), b"nb32", 0, None,
                             None), "linkadd")
    cubin = ctypes.c_void_p()
    cubin_size = ctypes.c_size_t()
    check(cuda.cuLinkComplete(ls, ctypes.byref(cubin),
                              ctypes.byref(cubin_size)), "linkcomplete")
    mod = ctypes.c_void_p()
    check(cuda.cuModuleLoadData(ctypes.byref(mod), cubin), "modload")
    fn = ctypes.c_void_p()
    check(cuda.cuModuleGetFunction(ctypes.byref(fn), mod, ENTRY), "getfunc")

    vals = [ctypes.c_ulonglong(0) for _ in range(22)]
    vals[0].value = nst.value
    vals[5].value = na.value
    vals[7].value = 4
    vals[12].value = nbp.value
    vals[14].value = 4
    vals[17].value = NN
    vals[19].value = nc.value
    vals[21].value = 4
    par = (ctypes.c_void_p * 22)()
    for i in range(22):
        par[i] = ctypes.addressof(vals[i])
    check(cuda.cuLaunchKernel(fn, NN // 256, 1, 1, 256, 1, 1, 0, None,
                              par, None), "launch")
    check(cuda.cuCtxSynchronize(), "sync")
    check(cuda.cuMemcpyDtoH_v2(hc.ctypes.data, nc.value, nbytes), "DtoH")

    want = ha + hb
    bad = np.nonzero(hc != want)[0]
    if len(bad):
        for i in bad[:5]:
            print(f"  nb32 mismatch[{i}]: got {hc[i]} want {want[i]}")
        raise SystemExit(f"FAIL: {len(bad)} mismatches")
    print(f"nb_f32 via Python: {NN} elements ok, sample {hc[:5].tolist()}")

    check(cuda.cuModuleUnload(mod), "unload")
    check(cuda.cuLinkDestroy(ls), "linkdestroy")
    check(cuda.cuMemFree_v2(na.value), "free a")
    check(cuda.cuMemFree_v2(nbp.value), "free b")
    check(cuda.cuMemFree_v2(nc.value), "free c")
    check(cuda.cuMemFree_v2(nst.value), "free status")
    check(cuda.cuCtxDestroy_v2(ctx), "destroy")
    print("python nb_f32 passed")


if __name__ == "__main__":
    sys.exit(main())
