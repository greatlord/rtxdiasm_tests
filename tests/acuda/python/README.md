# PyTorch end-to-end smoke tests

A small, non-exhaustive set of operator checks run through the real
PyTorch stack (`torch` → cuDNN/cub kernels → ACuda → Vulkan) and the SASS
lifter.  Each operator is exercised with one or two small shapes and
compared against the CPU result:

- `torch.sort` (int64 and float64, sizes that select the radix path)
- `conv2d` (cuDNN implicit-GEMM, shared-memory tiling)
- f64 `sin` / `cos` (including a large argument that takes the
  Payne–Hanek range-reduction slow path)
- f32/f64 `cumsum`, `var`, `std`
- f32/f64 `argmax` / `argmin` (fastest-dimension reduce, int64 indices)
- f32/f64 reductions (`sum`, `mean`, `max`, `dim=` variants)
- f32/f64 `matmul` and `softmax`

Known-broken operators (`f32 topk`, see `gatherTopK`) are reported as
`xfail` and do not fail the run.

This is intentionally a smoke test, not a conformance or performance
suite.

## Run

From a Linux tree with the ACuda release build:

```sh
ACUDA_TORCH_PYTHON=/path/to/venv/bin/python \
  rtxdiasm_tests/tests/acuda/python/run-torch-ops.sh
```

The script points `ACUDA_CUDA_DRIVER`, `LD_PRELOAD` (cublasLt),
`LD_LIBRARY_PATH` and `RTXDISAM_LIBRARY` at the local build. Override
`ACUDA_BIN_DIR` / `ACUDA_CUBLAS_BIN_DIR` / `RTXDISAM_LIBRARY` for a
different build layout.

To register it with ctest, configure with:

```sh
cmake ... -DACUDA_TORCH_PYTHON=/path/to/venv/bin/python
ctest --test-dir src/acuda/build/release -R test_torch_ops
```
