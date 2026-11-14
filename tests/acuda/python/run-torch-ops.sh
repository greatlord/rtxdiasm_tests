#!/usr/bin/env bash
# Run the simple PyTorch operator smoke tests against the local ACuda
# build.  Usage:
#   ACUDA_TORCH_PYTHON=/path/to/python rtxdiasm_tests/tests/acuda/python/run-torch-ops.sh
# Optional: RTXDISAM_LIBRARY=/path/to/rtxdisam_native.so
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../../.." && pwd)"        # rtxdisam repo root
acuda="${ACUDA_BIN_DIR:-$root/src/acuda/bin/release}"
cublas="${ACUDA_CUBLAS_BIN_DIR:-$root/src/acuda_cublas/bin/release}"

if [[ ! -e "$acuda/libcuda.so.1" ]]; then
    echo "ACuda release build not found at $acuda" >&2
    exit 1
fi

# Prefer the native lifter freshly built from this checkout; the
# deployed out/linux-x64 copy can be stale relative to the source.
if [[ -z "${RTXDISAM_LIBRARY:-}" ]]; then
    for candidate in \
        "$root/src/rtxdisam.native/bin/Release/net8.0/linux-x64/native/rtxdisam_native.so" \
        "$root/src/rtxdisam.native/bin/Debug/net8.0/linux-x64/native/rtxdisam_native.so" \
        "$root/out/linux-x64/rtxdisam_native.so" \
        "$acuda/librtxdisam_native.so"
    do
        if [[ -e "$candidate" ]]; then
            export RTXDISAM_LIBRARY="$candidate"
            break
        fi
    done
fi

export ACUDA_CUDA_DRIVER="$acuda/libcuda.so.1"
export ACUDA_NO_VALIDATION=1
# Avoid a shader cache entry produced by an older lifter.
export ACUDA_NO_CACHE="${ACUDA_NO_CACHE:-1}"
export LD_LIBRARY_PATH="$cublas:$acuda${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ -e "$cublas/libcublasLt.so.13" ]]; then
    export LD_PRELOAD="$cublas/libcublasLt.so.13${LD_PRELOAD:+:$LD_PRELOAD}"
fi

exec "${ACUDA_TORCH_PYTHON:-python3}" "$here/test_torch_ops.py" "$@"
