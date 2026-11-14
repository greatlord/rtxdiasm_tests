// NVIDIA regression for ptx_to_spv_error/func_ptr.
//
// The kernel writes input + 1 + address(foobar).  The absolute function
// address is driver/module dependent, so the test checks the invariant:
// two different inputs produce the same offset, and the offset is greater
// than one (the function address is nonzero).
#include <cuda.h>
#include <cstdio>
#include <fstream>
#include <string>

static bool check(CUresult result, const char *what) {
    if (result == CUDA_SUCCESS) return true;
    const char *name = nullptr;
    cuGetErrorName(result, &name);
    std::fprintf(stderr, "%s failed: %s\n", what, name ? name : "?");
    return false;
}

static bool run_one(CUfunction function, CUdeviceptr device_input,
                    CUdeviceptr device_output, unsigned long long input,
                    unsigned long long &output) {
    if (!check(cuMemcpyHtoD(device_input, &input, sizeof(input)),
               "cuMemcpyHtoD")) return false;
    void *parameters[2] = {&device_input, &device_output};
    if (!check(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr,
                              parameters, nullptr),
               "cuLaunchKernel")) return false;
    if (!check(cuCtxSynchronize(), "cuCtxSynchronize")) return false;
    if (!check(cuMemcpyDtoH(&output, device_output, sizeof(output)),
               "cuMemcpyDtoH")) return false;
    return true;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "func_ptr.ptx";
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "cannot read %s\n", path);
        return 2;
    }
    std::string ptx((std::istreambuf_iterator<char>(in)),
                    std::istreambuf_iterator<char>());

    if (!check(cuInit(0), "cuInit")) return 3;
    CUdevice device;
    if (!check(cuDeviceGet(&device, 0), "cuDeviceGet")) return 3;
    CUcontext context;
    if (!check(cuDevicePrimaryCtxRetain(&context, device),
               "cuDevicePrimaryCtxRetain")) return 3;
    if (!check(cuCtxSetCurrent(context), "cuCtxSetCurrent")) return 3;

    CUmodule module;
    if (!check(cuModuleLoadDataEx(&module, ptx.c_str(), 0, nullptr, nullptr),
               "cuModuleLoadDataEx")) return 4;
    CUfunction function;
    if (!check(cuModuleGetFunction(&function, module, "func_ptr"),
               "cuModuleGetFunction")) return 4;

    CUdeviceptr device_input = 0;
    CUdeviceptr device_output = 0;
    if (!check(cuMemAlloc(&device_input, sizeof(unsigned long long)),
               "cuMemAlloc input")) return 5;
    if (!check(cuMemAlloc(&device_output, sizeof(unsigned long long)),
               "cuMemAlloc output")) return 5;

    unsigned long long out1 = 0;
    unsigned long long out2 = 0;
    if (!run_one(function, device_input, device_output, 41, out1)) return 6;
    if (!run_one(function, device_input, device_output, 42, out2)) return 6;

    unsigned long long offset1 = out1 - 41;
    unsigned long long offset2 = out2 - 42;
    std::printf("func_ptr(41) = 0x%llx\n", out1);
    std::printf("func_ptr(42) = 0x%llx\n", out2);
    std::printf("offset       = 0x%llx\n", offset1);
    bool ok = offset1 == offset2 && offset1 > 1;
    std::printf("%s\n", ok ? "PASS" : "FAIL");

    cuMemFree(device_input);
    cuMemFree(device_output);
    cuModuleUnload(module);
    cuDevicePrimaryCtxRelease(device);
    return ok ? 0 : 7;
}
