# Windows test-kit corpus (imported)

These cases are imported from `windows-test-kit/` (RTX 3070 + CUDA truth
runs). Each directory holds one `.ptx` and a `case.txt`:

- NVIDIA-truth cases use `entry`, `input`/`expected` hex, `block`, and
  optional `params` (1 = single output pointer, 2 = input+output) or
  `expect_error`. These are the former `acuda_ptx_to_spv_error` cases.
- Success-kit cases use `entry`, `args` (`--run-buffer SIZE:VALUE`,
  `--run-scalar VALUE`), and `expected=bufN[0]=VALUE`. These are the
  former `acuda_ptx_to_spv_success/cases` cases.
- `func_ptr` uses `invariant=func_ptr` (the same function-address
  invariant as the NVIDIA C++ test).

`tests/integration/test_ptx_kit.c` compiles every case, runs `spirv-val`
when available, launches one workgroup on the local Vulkan device, and
compares the recorded truth.

Import notes:

- `b64tof64`, `rcp_f64`, and `func_ptr` had no `case.txt` upstream (they
  use per-case C++ truth tests); their case files were derived from
  those tests.
- `sm50_addr_pair` reads word 41 of its input buffer. Upstream's 16-byte
  buffer only happened to work on the reference allocator; the imported
  case uses 256 bytes so the test is well defined.
- `trap` expects an NVIDIA launch error. ACuda lowers `trap` to a clean
  end-of-invocation instead, so the runner accepts either outcome and
  reports which one occurred.
