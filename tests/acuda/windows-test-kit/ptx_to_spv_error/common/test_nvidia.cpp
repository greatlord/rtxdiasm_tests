// Deterministic NVIDIA truth test for a two-pointer PTX kernel.
//
// usage: test_nvidia.exe <ptx> <case-file>
//
// The case file uses key=value lines:
//   entry=name
//   input=<hex bytes>
//   expected=<hex bytes>
//   block=<threads in x>          (optional, default 1)
//
// The kernel is launched as 1 block of block threads.  The two parameters are
// the device input pointer and the device output pointer.  The output buffer
// is exactly the size of the expected byte string.
#include <cuda.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static bool check(CUresult result, const char *what) {
    if (result == CUDA_SUCCESS) return true;
    const char *name = nullptr;
    cuGetErrorName(result, &name);
    std::fprintf(stderr, "%s failed: %s\n", what, name ? name : "?");
    return false;
}

static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static bool parse_hex(const std::string &text, std::vector<unsigned char> &out) {
    if (text.size() % 2 != 0) return false;
    out.assign(text.size() / 2, 0);
    for (size_t i = 0; i < out.size(); i++) {
        int hi = hexval(text[2 * i]);
        int lo = hexval(text[2 * i + 1]);
        if (hi < 0 || lo < 0) return false;
        out[i] = (unsigned char)((hi << 4) | lo);
    }
    return true;
}

static std::string hex(const std::vector<unsigned char> &bytes) {
    static const char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(bytes.size() * 2);
    for (unsigned char byte : bytes) {
        result.push_back(digits[byte >> 4]);
        result.push_back(digits[byte & 15]);
    }
    return result;
}

static bool read_file(const char *path, std::string &text) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    text.assign((std::istreambuf_iterator<char>(in)),
                std::istreambuf_iterator<char>());
    return true;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <ptx> <case-file>\n", argv[0]);
        return 2;
    }
    const char *ptx_path = argv[1];
    const char *case_path = argv[2];

    std::string case_text;
    if (!read_file(case_path, case_text)) {
        std::fprintf(stderr, "cannot read %s\n", case_path);
        return 2;
    }

    std::string entry;
    std::string input_hex;
    std::string expected_hex;
    unsigned block_x = 1;
    unsigned param_count = 2;
    bool expect_error = false;
    size_t start = 0;
    while (start <= case_text.size()) {
        size_t end = case_text.find('\n', start);
        if (end == std::string::npos) end = case_text.size();
        std::string line = case_text.substr(start, end - start);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        size_t equals = line.find('=');
        if (equals != std::string::npos && line[0] != '#') {
            std::string key = line.substr(0, equals);
            std::string value = line.substr(equals + 1);
            if (key == "entry") entry = value;
            else if (key == "input") input_hex = value;
            else if (key == "expected") expected_hex = value;
            else if (key == "block") block_x = (unsigned)std::strtoul(
                value.c_str(), nullptr, 0);
            else if (key == "params") param_count = (unsigned)std::strtoul(
                value.c_str(), nullptr, 0);
            else if (key == "expect_error") expect_error = value != "0";
        }
        start = end + 1;
    }
    if (entry.empty() || expected_hex.empty() || block_x == 0) {
        std::fprintf(stderr, "case file needs entry, expected, and block\n");
        return 2;
    }

    std::vector<unsigned char> input;
    std::vector<unsigned char> expected;
    if (!parse_hex(input_hex, input) || !parse_hex(expected_hex, expected)) {
        std::fprintf(stderr, "bad input or expected hex\n");
        return 2;
    }

    std::string ptx;
    if (!read_file(ptx_path, ptx)) {
        std::fprintf(stderr, "cannot read %s\n", ptx_path);
        return 2;
    }

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
    if (!check(cuModuleGetFunction(&function, module, entry.c_str()),
               "cuModuleGetFunction")) return 4;

    CUdeviceptr device_input = 0;
    CUdeviceptr device_output = 0;
    if (!check(cuMemAlloc(&device_input, input.empty() ? 1 : input.size()),
               "cuMemAlloc input")) return 5;
    if (!check(cuMemAlloc(&device_output, expected.size()),
               "cuMemAlloc output")) return 5;
    if (!input.empty() &&
        !check(cuMemcpyHtoD(device_input, input.data(), input.size()),
               "cuMemcpyHtoD")) return 5;

    if (param_count != 1 && param_count != 2) {
        std::fprintf(stderr, "params must be 1 or 2\n");
        return 2;
    }
    void *parameters[2] = {&device_input, &device_output};
    if (param_count == 1)
        parameters[0] = &device_output;
    CUresult launch = cuLaunchKernel(function, 1, 1, 1, block_x, 1, 1, 0,
                                     nullptr, parameters, nullptr);
    CUresult sync = launch == CUDA_SUCCESS ? cuCtxSynchronize() : launch;
    CUresult failure = launch != CUDA_SUCCESS ? launch : sync;

    if (expect_error) {
        std::printf("entry=%s block=%u expect_error=1\n", entry.c_str(),
                    block_x);
        if (failure == CUDA_SUCCESS) {
            std::printf("FAIL (expected a launch or synchronize error)\n");
            cuMemFree(device_input);
            cuMemFree(device_output);
            cuModuleUnload(module);
            cuDevicePrimaryCtxRelease(device);
            return 7;
        }
        const char *name = nullptr;
        cuGetErrorName(failure, &name);
        std::printf("error   =%s\n", name ? name : "?");
        std::printf("PASS\n");
        cuMemFree(device_input);
        cuMemFree(device_output);
        cuModuleUnload(module);
        cuDevicePrimaryCtxRelease(device);
        return 0;
    }

    if (!check(launch, "cuLaunchKernel")) return 6;
    if (!check(sync, "cuCtxSynchronize")) return 6;

    std::vector<unsigned char> actual(expected.size(), 0);
    if (!check(cuMemcpyDtoH(actual.data(), device_output, actual.size()),
               "cuMemcpyDtoH")) return 6;

    std::printf("entry=%s block=%u\n", entry.c_str(), block_x);
    std::printf("input   =%s\n", input_hex.c_str());
    std::printf("expected=%s\n", expected_hex.c_str());
    std::printf("actual  =%s\n", hex(actual).c_str());
    bool ok = actual == expected;
    std::printf("%s\n", ok ? "PASS" : "FAIL");

    cuMemFree(device_input);
    cuMemFree(device_output);
    cuModuleUnload(module);
    cuDevicePrimaryCtxRelease(device);
    return ok ? 0 : 7;
}
