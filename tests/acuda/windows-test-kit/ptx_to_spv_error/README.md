# PTX -> SPIR-V error cases

Each subdirectory is a self-contained reproduction for PTX that the local
PTX -> SPIR-V backend or runtime cannot execute yet. Cases either include an
NVIDIA CUDA driver-API truth test with deterministic output, or document a
CUDA-valid PTX shape that ACuda currently rejects or cannot launch.

## Cases

- [`trap`](trap/README.md) — `unsupported instruction 'trap'`; RTX 3070
  reports `CUDA_ERROR_LAUNCH_FAILED` as expected.

`trap` remains because it is a negative test: the kernel is expected to fail
at runtime. All former ACuda compiler and host-callback repros now live under
[`ptx_to_spv_success/cases`](../ptx_to_spv_success/cases/). A handoff summary
is in [`docs/acuda-bugs.md`](../docs/acuda-bugs.md).

The former `switch_chain` repro was removed: the lifter now emits a
branchless select for simple constant-case switches, validated by
[`sass_to_ptx_success/sm50_switch`](../sass_to_ptx_success/sm50_switch/README.md).

## Moved to success

The updated ACuda backend now accepts the former repros for `rcp_f64`,
`b64tof64`, `param_is_addressable`, `vshr_clamp`, `cvt_pack`,
`bar_red_and_pred`, `atomics_128`, `call_rnd`, `const_ident`,
`uint_to_fp_bf16`, `bar_sync`, `cp_async`, `createpolicy`, `block`,
`mov_address`, `nanosleep`, `func_ptr`, `bfind_u32`, `vote_all_sub`,
`brkpt`, `param_pointer_value`, `brx_chain`, `atom_inc_emit`,
`packed_address`, `diamond_merge`, `brx4_merge`, `dynamic_ptr_ldc`,
`extern_func`, and `fma_oob`.
They now live under [`ptx_to_spv_success/cases`](../ptx_to_spv_success/cases/)
with the ACuda success runner and NVIDIA truth tests (`nvidia_case.txt`) where
the case has a compatible two-pointer shape.

## Run

Run the remaining cases:

```powershell
.\run_all.bat
```

`trap` is the intentional negative NVIDIA case. `extern_func` now lives in
the success kit and runs through the identity host-call stub.

## Check for fixed cases

After an ACuda update, run:

```powershell
.\check_fixed.bat
```

It re-runs each error case. A case that no longer reproduces is moved to
`ptx_to_spv_success/cases/` only when the case directory has a `success.txt`
with:

```text
entry=<ptx entry>
args=<runner arguments>
expected=<substring>
# or expected_regex=<regex>
```

Cases without valid success metadata are reported but not moved. `trap` is
always skipped because it is an intentional negative test.

The shared driver-API checker is in [`common`](common/test_nvidia.cpp).
