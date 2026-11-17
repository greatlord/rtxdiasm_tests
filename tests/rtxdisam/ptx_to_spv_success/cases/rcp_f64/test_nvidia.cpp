// NVIDIA regression for ptx_to_spv_error/rcp_f64.
// Loads the PTX through the CUDA driver API and checks rcp.rz.f64(4.0) = 0.25.
#include <cuda.h>
#include <cmath>
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

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "rcp_f64.ptx";
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
    if (!check(cuModuleGetFunction(&function, module, "rcp_f64"),
               "cuModuleGetFunction")) return 4;

    double input = 4.0;
    double output = 0.0;
    CUdeviceptr device_input = 0;
    CUdeviceptr device_output = 0;
    if (!check(cuMemAlloc(&device_input, sizeof(input)), "cuMemAlloc input"))
        return 5;
    if (!check(cuMemAlloc(&device_output, sizeof(output)), "cuMemAlloc output"))
        return 5;
    if (!check(cuMemcpyHtoD(device_input, &input, sizeof(input)),
               "cuMemcpyHtoD")) return 5;

    void *parameters[2] = {&device_input, &device_output};
    if (!check(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr,
                              parameters, nullptr),
               "cuLaunchKernel")) return 6;
    if (!check(cuCtxSynchronize(), "cuCtxSynchronize")) return 6;
    if (!check(cuMemcpyDtoH(&output, device_output, sizeof(output)),
               "cuMemcpyDtoH")) return 6;

    std::printf("rcp_f64(4.0) = %.17g\n", output);
    bool ok = std::fabs(output - 0.25) < 1e-12;
    std::printf("%s\n", ok ? "PASS" : "FAIL");

    cuMemFree(device_input);
    cuMemFree(device_output);
    cuModuleUnload(module);
    cuDevicePrimaryCtxRelease(device);
    return ok ? 0 : 7;
}
