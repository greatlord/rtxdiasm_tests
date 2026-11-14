
import ctypes
import os
import sys
from pathlib import Path

# --------------------------------------------------
# ACuda library configuration (no hardcoded paths:
# ACUDA_LIBCUDA wins, else the repo's own build output
# relative to this file's location).
# --------------------------------------------------

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

CUresult = ctypes.c_int
CUdevice = ctypes.c_int
CUcontext = ctypes.c_void_p
CUmodule = ctypes.c_void_p
CUfunction = ctypes.c_void_p
CUdeviceptr = ctypes.c_uint64
SIZE_T = ctypes.c_size_t
VOID_P = ctypes.c_void_p


cuda = ctypes.CDLL(find_lib())




def get_api_function(*names):
    """Find an exported CUDA API symbol, supporting common aliases."""
    for name in names:
        try:
            return getattr(cuda, name)
        except AttributeError:
            pass

    raise RuntimeError(
        f"Missing API function. Tried: {', '.join(names)}"
    )


def bind(names, argtypes):
    if isinstance(names, str):
        names = (names,)

    fn = get_api_function(*names)
    fn.argtypes = argtypes
    fn.restype = CUresult
    return fn


def check(result, operation):
    if result != 0:
        raise RuntimeError(
            f"{operation} failed: CUDA error code {result}"
        )


# --------------------------------------------------
# Bind CUDA Driver API
# --------------------------------------------------

cuInit = bind("cuInit", [ctypes.c_uint])

cuDeviceGet = bind(
    "cuDeviceGet",
    [ctypes.POINTER(CUdevice), ctypes.c_int],
)

cuCtxCreate = bind(
    ("cuCtxCreate_v2", "cuCtxCreate"),
    [ctypes.POINTER(CUcontext), ctypes.c_uint, CUdevice],
)

cuCtxSynchronize = bind("cuCtxSynchronize", [])

cuCtxDestroy = bind(
    ("cuCtxDestroy_v2", "cuCtxDestroy"),
    [CUcontext],
)

cuMemAlloc = bind(
    ("cuMemAlloc_v2", "cuMemAlloc"),
    [ctypes.POINTER(CUdeviceptr), SIZE_T],
)

cuMemFree = bind(
    ("cuMemFree_v2", "cuMemFree"),
    [CUdeviceptr],
)

cuMemcpyHtoD = bind(
    ("cuMemcpyHtoD_v2", "cuMemcpyHtoD"),
    [CUdeviceptr, VOID_P, SIZE_T],
)

cuMemcpyDtoH = bind(
    ("cuMemcpyDtoH_v2", "cuMemcpyDtoH"),
    [VOID_P, CUdeviceptr, SIZE_T],
)

cuModuleLoadData = bind(
    "cuModuleLoadData",
    [ctypes.POINTER(CUmodule), VOID_P],
)

cuModuleGetFunction = bind(
    "cuModuleGetFunction",
    [ctypes.POINTER(CUfunction), CUmodule, ctypes.c_char_p],
)

cuModuleUnload = bind("cuModuleUnload", [CUmodule])

cuLaunchKernel = bind(
    "cuLaunchKernel",
    [
        CUfunction,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.POINTER(VOID_P),
        ctypes.POINTER(VOID_P),
    ],
)


# --------------------------------------------------
# PTX kernel
# Adds two float arrays: c[i] = a[i] + b[i]
# --------------------------------------------------

ptx = r"""
.version 7.0
.target sm_80
.address_size 64

.visible .entry add_kernel(
    .param .u64 param_a,
    .param .u64 param_b,
    .param .u64 param_c,
    .param .u32 param_n
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<3>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<4>;

    ld.param.u64 %rd1, [param_a];
    ld.param.u64 %rd2, [param_b];
    ld.param.u64 %rd3, [param_c];
    ld.param.u32 %r1, [param_n];

    mov.u32 %r2, %tid.x;
    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra DONE;

    mul.wide.u32 %rd4, %r2, 4;
    add.u64 %rd5, %rd1, %rd4;
    add.u64 %rd6, %rd2, %rd4;
    add.u64 %rd7, %rd3, %rd4;

    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];
    add.f32 %f3, %f1, %f2;
    st.global.f32 [%rd7], %f3;

DONE:
    ret;
}
"""

# --------------------------------------------------
# Initialize CUDA and create context
# --------------------------------------------------

context = CUcontext()
module = CUmodule()
allocated = []
success = False

try:
    check(cuInit(0), "cuInit")

    device = CUdevice()
    check(cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")

    check(
        cuCtxCreate(ctypes.byref(context), 0, device),
        "cuCtxCreate",
    )

    print("CUDA context created.")

    # --------------------------------------------------
    # Allocate and initialize host data
    # --------------------------------------------------

    a_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    b_values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]

    n = len(a_values)
    byte_count = n * ctypes.sizeof(ctypes.c_float)

    HostArray = ctypes.c_float * n

    host_a = HostArray(*a_values)
    host_b = HostArray(*b_values)
    host_c = HostArray()

    # --------------------------------------------------
    # Allocate GPU/device memory
    # --------------------------------------------------

    device_a = CUdeviceptr()
    device_b = CUdeviceptr()
    device_c = CUdeviceptr()

    for pointer in (device_a, device_b, device_c):
        # Allocate separately below because ctypes values
        # are copied into the allocation list.
        pass

    check(
        cuMemAlloc(ctypes.byref(device_a), byte_count),
        "cuMemAlloc(a)",
    )
    allocated.append(device_a.value)

    check(
        cuMemAlloc(ctypes.byref(device_b), byte_count),
        "cuMemAlloc(b)",
    )
    allocated.append(device_b.value)

    check(
        cuMemAlloc(ctypes.byref(device_c), byte_count),
        "cuMemAlloc(c)",
    )
    allocated.append(device_c.value)

    # --------------------------------------------------
    # Copy input data to device
    # --------------------------------------------------

    check(
        cuMemcpyHtoD(
            device_a,
            ctypes.cast(host_a, VOID_P),
            byte_count,
        ),
        "cuMemcpyHtoD(a)",
    )

    check(
        cuMemcpyHtoD(
            device_b,
            ctypes.cast(host_b, VOID_P),
            byte_count,
        ),
        "cuMemcpyHtoD(b)",
    )

    print("Input data copied to device.")

    # --------------------------------------------------
    # JIT: load PTX and resolve the kernel
    # --------------------------------------------------

    ptx_buffer = ctypes.create_string_buffer(ptx.encode("utf-8"))

    check(
        cuModuleLoadData(
            ctypes.byref(module),
            ctypes.cast(ptx_buffer, VOID_P),
        ),
        "cuModuleLoadData",
    )

    function = CUfunction()

    check(
        cuModuleGetFunction(
            ctypes.byref(function),
            module,
            b"add_kernel",
        ),
        "cuModuleGetFunction",
    )

    print("PTX module loaded and kernel resolved.")

    # --------------------------------------------------
    # Set up kernel arguments
    # Each argument is a pointer to its host-side value.
    # --------------------------------------------------

    kernel_n = ctypes.c_uint32(n)

    kernel_params = (VOID_P * 4)(
        ctypes.cast(ctypes.byref(device_a), VOID_P),
        ctypes.cast(ctypes.byref(device_b), VOID_P),
        ctypes.cast(ctypes.byref(device_c), VOID_P),
        ctypes.cast(ctypes.byref(kernel_n), VOID_P),
    )

    # --------------------------------------------------
    # Launch kernel: one block, n threads
    # --------------------------------------------------

    check(
        cuLaunchKernel(
            function,
            1, 1, 1,       # Grid dimensions
            n, 1, 1,       # Block dimensions
            0,              # Dynamic shared memory bytes
            None,           # Default stream
            kernel_params,
            None,           # Extra launch options
        ),
        "cuLaunchKernel",
    )

    check(cuCtxSynchronize(), "cuCtxSynchronize")

    print("Kernel executed.")

    # --------------------------------------------------
    # Copy result back to host
    # --------------------------------------------------

    check(
        cuMemcpyDtoH(
            ctypes.cast(host_c, VOID_P),
            device_c,
            byte_count,
        ),
        "cuMemcpyDtoH",
    )

    result = list(host_c)
    expected = [a + b for a, b in zip(a_values, b_values)]

    print("\nInput A:", a_values)
    print("Input B:", b_values)
    print("Result: ", result)
    print("Expected:", expected)

    assert all(
        abs(actual - wanted) < 1e-6
        for actual, wanted in zip(result, expected)
    ), "Kernel returned incorrect results!"

    print("\nPASS: CUDA JIT kernel executed successfully.")
    success = True

finally:
    # --------------------------------------------------
    # Release resources
    # --------------------------------------------------

    for pointer in reversed(allocated):
        try:
            check(cuMemFree(pointer), "cuMemFree")
        except Exception as exc:
            print("Cleanup warning:", exc)

    if module.value:
        try:
            check(cuModuleUnload(module), "cuModuleUnload")
        except Exception as exc:
            print("Cleanup warning:", exc)

    if context.value:
        try:
            check(cuCtxDestroy(context), "cuCtxDestroy")
        except Exception as exc:
            print("Cleanup warning:", exc)

if not success:
    raise RuntimeError("CUDA JIT test did not complete.")
