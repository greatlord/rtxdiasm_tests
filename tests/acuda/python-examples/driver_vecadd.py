"""Python integration: drive ACuda's libcuda.so via ctypes (numba-style flow).

Proves the C ABI seam future Numba/CuPy work needs: init -> context ->
alloc -> HtoD -> link PTX -> module -> function -> launch -> DtoH.
Numba itself needs Phase 6 compiler work (.func, generic ld/st, signed
ops, diamond CFG); see docs/supported-isa.md section 9.

Usage (from ACuda/):
    ""/home/magnusolsen/git/aicode-ptx/virtual python/ven-cu134/bin/python"" python-test/driver_vecadd.py
"""
import ctypes
import os
import sys

try:
    import numpy as np
except ImportError:
    sys.exit("numpy not found: run with '/home/magnusolsen/git/aicode-ptx/virtual python/ven-cu134/bin/python'")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_lib():
    override = os.environ.get("ACUDA_LIBCUDA")
    if override:
        return override
    for cand in ("bin/debug/libcuda.so.1.0.0",
                 "bin/release/libcuda.so.1.0.0",
                 "build/libcuda.so.1.0.0"):  # legacy location
        path = os.path.join(ROOT, cand)
        if os.path.exists(path):
            return path
    sys.exit("libcuda.so not found: build first "
             "(cmake -S . -B build/release && cmake --build build/release), "
             "or set ACUDA_LIBCUDA=/path/to/libcuda.so.1.0.0")


LIB = find_lib()
PTX = os.path.join(ROOT, "tests", "ptx", "vecadd.ptx")
NUMBA_PTX = os.path.join(ROOT, "tests", "ptx", "vecadd_numba.ptx")

N = 1 << 20
GRID = N // 256
BLOCK = 256

cuda = ctypes.CDLL(LIB)
cuda.cuInit.argtypes = [ctypes.c_uint]
cuda.cuInit.restype = ctypes.c_int
cuda.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
cuda.cuDeviceGetCount.restype = ctypes.c_int
cuda.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
cuda.cuDeviceGetName.restype = ctypes.c_int
cuda.cuDeviceComputeCapability.argtypes = [ctypes.POINTER(ctypes.c_int),
                                           ctypes.POINTER(ctypes.c_int),
                                           ctypes.c_int]
cuda.cuDeviceComputeCapability.restype = ctypes.c_int
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
cuda.cuGetErrorString.argtypes = [ctypes.c_int,
                                  ctypes.POINTER(ctypes.c_char_p)]
cuda.cuGetErrorString.restype = ctypes.c_int


def check(code, what):
    if code != 0:
        msg = ctypes.c_char_p()
        cuda.cuGetErrorString(code, ctypes.byref(msg))
        raise RuntimeError(f"{what} failed: {code} "
                           f"({msg.value.decode() if msg.value else '?'})")


def main():
    with open(PTX, "rb") as f:
        ptx = f.read()

    check(cuda.cuInit(0), "cuInit")
    ndev = ctypes.c_int()
    check(cuda.cuDeviceGetCount(ctypes.byref(ndev)), "cuDeviceGetCount")
    print(f"devices: {ndev.value}")
    for i in range(ndev.value):
        buf = ctypes.create_string_buffer(256)
        check(cuda.cuDeviceGetName(buf, 256, i), "cuDeviceGetName")
        major, minor = ctypes.c_int(), ctypes.c_int()
        check(cuda.cuDeviceComputeCapability(ctypes.byref(major),
                                             ctypes.byref(minor), i),
              "cuDeviceComputeCapability")
        print(f"  device[{i}]: {buf.value.decode()} "
              f"cc={major.value}.{minor.value}")
    print("  (ACuda selects the discrete GPU when present)")
    ctx = ctypes.c_void_p()
    check(cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, 0), "cuCtxCreate")

    nbytes = N * 4
    da, db, dc = (ctypes.c_ulonglong() for _ in range(3))
    check(cuda.cuMemAlloc_v2(ctypes.byref(da), nbytes), "alloc a")
    check(cuda.cuMemAlloc_v2(ctypes.byref(db), nbytes), "alloc b")
    check(cuda.cuMemAlloc_v2(ctypes.byref(dc), nbytes), "alloc c")

    ha = (np.arange(N, dtype=np.float32) * 0.5).astype(np.float32)
    hb = (np.arange(N, dtype=np.float32) * 2.0 + 1.0).astype(np.float32)
    hc = np.zeros(N, dtype=np.float32)
    check(cuda.cuMemcpyHtoD_v2(da.value, ha.ctypes.data, nbytes), "HtoD a")
    check(cuda.cuMemcpyHtoD_v2(db.value, hb.ctypes.data, nbytes), "HtoD b")

    ls = ctypes.c_void_p()
    check(cuda.cuLinkCreate(0, None, None, ctypes.byref(ls)), "cuLinkCreate")
    check(cuda.cuLinkAddData(ls, 1, ptx, len(ptx), b"vecadd", 0, None,
                             None), "cuLinkAddData")
    # Negative: cubin blob must be rejected, state stays usable.
    rc = cuda.cuLinkAddData(ls, 0, b"\x7fELFxxxx", 8, b"fake", 0, None,
                            None)
    assert rc == 200, f"expected INVALID_IMAGE, got {rc}"
    cubin = ctypes.c_void_p()
    cubin_size = ctypes.c_size_t()
    check(cuda.cuLinkComplete(ls, ctypes.byref(cubin),
                              ctypes.byref(cubin_size)), "cuLinkComplete")

    mod = ctypes.c_void_p()
    check(cuda.cuModuleLoadData(ctypes.byref(mod), cubin), "cuModuleLoadData")
    fn = ctypes.c_void_p()
    check(cuda.cuModuleGetFunction(ctypes.byref(fn), mod,
                                   b"vecadd_f32"), "cuModuleGetFunction")
    bad = ctypes.c_void_p()
    assert cuda.cuModuleGetFunction(ctypes.byref(bad), mod, b"nope") == 500

    n = ctypes.c_uint32(N)
    # params[i] points at the i-th argument value: 8-byte deviceptr
    # handles for .u64 params, 4-byte word for the scalar.
    params = (ctypes.c_void_p * 4)(ctypes.addressof(da),
                                   ctypes.addressof(db),
                                   ctypes.addressof(dc),
                                   ctypes.addressof(n))
    check(cuda.cuLaunchKernel(fn, GRID, 1, 1, BLOCK, 1, 1, 0, None,
                              params, None), "cuLaunchKernel")
    check(cuda.cuCtxSynchronize(), "cuCtxSynchronize")
    check(cuda.cuMemcpyDtoH_v2(hc.ctypes.data, dc.value, nbytes), "DtoH")

    want = ha * 2.0 + hb
    bad_idx = np.nonzero(hc != want)[0]
    assert len(bad_idx) == 0, f"mismatches: {bad_idx[:5]}"
    print(f"vecadd_f32 via Python: {N} elements ok, sample {hc[:5].tolist()}")

    # Numba f64 PTX links cleanly since Stage 6 (f64 support).
    with open(NUMBA_PTX, "rb") as f:
        numba_ptx = f.read()
    ls2 = ctypes.c_void_p()
    check(cuda.cuLinkCreate(0, None, None, ctypes.byref(ls2)), "link2")
    rc = cuda.cuLinkAddData(ls2, 1, numba_ptx, len(numba_ptx), b"nb", 0,
                            None, None)
    assert rc == 0, f"numba f64 ptx should link, got {rc}"
    check(cuda.cuLinkDestroy(ls2), "link2 destroy")
    print("numba f64 ptx links as expected (f64 supported)")

    check(cuda.cuModuleUnload(mod), "cuModuleUnload")
    check(cuda.cuLinkDestroy(ls), "cuLinkDestroy")
    check(cuda.cuMemFree_v2(da.value), "free a")
    check(cuda.cuMemFree_v2(db.value), "free b")
    check(cuda.cuMemFree_v2(dc.value), "free c")
    check(cuda.cuCtxDestroy_v2(ctx), "cuCtxDestroy")
    print("python driver integration passed")


if __name__ == "__main__":
    sys.exit(main())
