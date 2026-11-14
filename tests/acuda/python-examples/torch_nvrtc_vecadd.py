"""torch + NVRTC + ACuda end-to-end (AMD, no NVIDIA GPU).

torch manages memory (its caching allocator works over our arena with
real addresses); NVRTC compiles the kernel to PTX; our libcuda compiles
PTX to SPIR-V and runs it on Vulkan. Proves the full Python path for
subset-compatible kernels. See docs/torch-nvrtc.md.
"""
import ctypes
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
from cuda.bindings import nvrtc

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
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


KERNEL = r"""
extern "C" __global__
void vecadd(const float *a, const float *b, float *c, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * 2.0f + b[i];
}
"""

N = 1 << 20
GRID = N // 256


def main():
    assert torch.cuda.is_available(), "torch sees no CUDA device"
    print("device:", torch.cuda.get_device_name(0))

    # 1. NVRTC: CUDA C++ -> PTX (no GPU needed for this step).
    _, prog = nvrtc.nvrtcCreateProgram(KERNEL.encode(), b"vecadd.cu", 0,
                                       [], [])
    rc = nvrtc.nvrtcCompileProgram(prog, 1, [b"--gpu-architecture=sm_80"])
    assert rc[0].value == 0, f"nvrtc failed: {rc}"
    _, size = nvrtc.nvrtcGetPTXSize(prog)
    buf = bytearray(size)
    nvrtc.nvrtcGetPTX(prog, buf)
    ptx = bytes(buf).split(b"\x00")[0]
    assert b".entry vecadd" in ptx, "unexpected NVRTC output shape"
    print(f"nvrtc: {len(ptx)} bytes of PTX")

    # 2. Our driver: PTX -> SPIR-V -> Vulkan.
    cuda = ctypes.CDLL(find_lib())
    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuInit.restype = ctypes.c_int
    cuda.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                    ctypes.c_uint, ctypes.c_int]
    cuda.cuCtxCreate_v2.restype = ctypes.c_int
    cuda.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                      ctypes.c_void_p]
    cuda.cuModuleLoadData.restype = ctypes.c_int
    cuda.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                         ctypes.c_void_p, ctypes.c_char_p]
    cuda.cuModuleGetFunction.restype = ctypes.c_int
    cuda.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + \
        [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    cuda.cuLaunchKernel.restype = ctypes.c_int
    cuda.cuCtxSynchronize.argtypes = []
    cuda.cuCtxSynchronize.restype = ctypes.c_int

    def check(code, what):
        assert code == 0, f"{what} failed: {code}"

    check(cuda.cuInit(0), "cuInit")
    ctx = ctypes.c_void_p()
    check(cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, 0), "cuCtxCreate")
    mod = ctypes.c_void_p()
    check(cuda.cuModuleLoadData(ctypes.byref(mod), ptx), "cuModuleLoadData")
    fn = ctypes.c_void_p()
    check(cuda.cuModuleGetFunction(ctypes.byref(fn), mod, b"vecadd"),
          "cuModuleGetFunction")

    # 3. torch tensors (allocator over our arena; filled via copies, no
    # torch kernels involved) + our launch.
    ha = (np.arange(N, dtype=np.float32) * 0.5).astype(np.float32)
    hb = (np.arange(N, dtype=np.float32) * 2.0 + 1.0).astype(np.float32)
    a = torch.from_numpy(ha).to(device="cuda")
    b = torch.from_numpy(hb).to(device="cuda")
    c = torch.empty(N, dtype=torch.float32, device="cuda")
    n = ctypes.c_uint32(N)
    da, db, dc = (ctypes.c_ulonglong(a.data_ptr()),
                  ctypes.c_ulonglong(b.data_ptr()),
                  ctypes.c_ulonglong(c.data_ptr()))
    params = (ctypes.c_void_p * 4)(ctypes.addressof(da),
                                   ctypes.addressof(db),
                                   ctypes.addressof(dc),
                                   ctypes.addressof(n))
    check(cuda.cuLaunchKernel(fn, GRID, 1, 1, 256, 1, 1, 0, None, params,
                              None), "cuLaunchKernel")
    check(cuda.cuCtxSynchronize(), "cuCtxSynchronize")

    got = c.cpu().numpy()
    want = (np.arange(N, dtype=np.float32) * 0.5) * 2.0 + \
        (np.arange(N, dtype=np.float32) * 2.0 + 1.0)
    bad = np.nonzero(got != want)[0]
    assert len(bad) == 0, f"mismatches at {bad[:5]}"
    print(f"torch+nvrtc+acuda: {N} elements ok, sample {got[:5].tolist()}")


if __name__ == "__main__":
    sys.exit(main())
