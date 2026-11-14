# PTX -> SPIR-V success cases

This is the success counterpart to [`ptx_to_spv_error`](../ptx_to_spv_error/).
Each case is PTX that currently compiles and runs through the local ACuda
PTX -> SPIR-V backend.

## Cases

| Case | Expected |
|---|---|
| `sm50_load_add` | `buf0[0]=42` |
| `sm50_xmad_store` | `buf0[0]=220` |
| `sm50_ops50` | `buf0[0]=19457` |
| `ops30` | `buf0[0]=19457` |
| `sm32_load_add` | `buf0[0]=42` |
| `sm50_addr_pair` | `buf1[0]=41` |
| `sm50_lea_pointer` | `buf1[0]=41` |
| `sm50_atom_add` | `buf1[0]=51` |
| `sm50_atom_shared` | `buf1[0]=51` |
| `atom_inc_bound` | `4`, old `3` |
| `atom_dec_bound` | `2`, old `3` |
| `atom_add_f32` | `3.75`, old `1.5` |
| `loop_sum` | `45` |
| `sm50_lds_dynamic` | `buf1[0]=41` |
| `sm50_lds_simple` | `buf1[0]=41` |
| `sm50_ldg128_simple` | `buf1[0]=41` |
| `sm50_ldg128_indexed` | `buf1[0]=41` |
| `sm50_bits` | `3`, `0x9abcde10`, `0xdef05678` |
| `sm50_dadd` | `1.5 + 3.25 = 4.75` |
| `sm50_fmnmx` | min `1.0`, max `2.0` |
| `sm50_fsetp` | `2.0 >= 1.0` → `1` |
| `sm50_mufu` | `sin(pi/2) = 1.0` |
| `sm50_popc_flo` | `buf1[0]=8` |
| `sm50_shf` | `0xabcdef01` |
| `sm70_iadd3` | `buf1[0]=6` |
| `sm50_shf_r` | `0x01234567` |
| `sm50_shfl` | `buf1[0]=41` |
| `sm50_vote` | `buf1[0]=1` |
| `sm50_ballot` | `buf1[0]=1` |

## Former error cases now passing

The updated ACuda backend accepts these former `ptx_to_spv_error` repros.
NVIDIA truth tests are preserved as `nvidia_case.txt` where the case has a
two-pointer shape compatible with the shared checker.

| Case | Expected |
|---|---|
| `atom_inc_emit` | `2`, old `0`, `1` |
| `brx4_merge` | `buf0[0]=10` for selector `0` |
| `diamond_merge` | `buf0[0]=6` for input `5` |
| `dynamic_ptr_ldc` | selectors `0..2` → `10/20/30` |
| `extern_func` | identity host callback returns input `4660` |
| `packed_address` | `buf0[0]=7` |
| `atomics_128` | `2, 3` |
| `b64tof64` | double `1.5` preserved |
| `bar_red_and_pred` | `2, 2` |
| `bar_sync` | compiles and launches |
| `bfind_u32` | `5` for input `41` |
| `block` | `42` for input `41` |
| `brkpt` | compiles and launches; `buf0[0]=7` |
| `brx_chain` | compiles and launches |
| `param_pointer_value` | `buf0[0]=42` |
| `call_rnd` | `0x3f800001`, `0x3f800000` |
| `const_ident` | four zero words |
| `cp_async` | `1, 2, 3, 0` |
| `createpolicy` | compiles and launches |
| `cvt_pack` | `0x01007f80`, `0x01007f7f` |
| `func_ptr` | nonzero function-address offset |
| `mov_address` | compiles and launches |
| `nanosleep` | compiles and launches |
| `param_is_addressable` | `0` |
| `rcp_f64` | `0.25` for input `4.0` |
| `uint_to_fp_bf16` | bfloat16 `1.0` = `0x3f80` |
| `vote_all_sub` | lane 0 `0`, lanes 1..31 `1` |
| `vshr_clamp` | `0x12345678`, `0x11111111` |
| `fma_oob` | `buf0[0]=16384` (`fma.rn.oob.f16` with ordinary operands) |
| `func_local` | `buf0[0]=6` for input `5` (callee-local frame) |

`trap` remains in `ptx_to_spv_error` because it is a negative test: the
kernel is expected to fail at runtime.

## Run

Build the app first:

```powershell
dotnet build .\src\rtxdisam.cli\rtxdisam.cli.csproj -c Debug
```

Then run every success case through ACuda:

```powershell
cd tests\ptx_to_spv_success
.\run_all.bat
```

Prerequisites are the same as `--run-ptx`: WSL Ubuntu with the local ACuda
tree, system Vulkan headers, and `spirv-val`. The runner also requires
`validation_errors=0` for every case.

## Adding a case

Create `cases/<name>/` containing:

- `<name>.ptx`
- `case.txt`:

  ```text
  entry=k
  args=--run-buffer 16:41 --run-buffer 16:0
  expected=buf1[0]=41
  ```

  Supported runner arguments include:

  - `--run-buffer size[:initial-u32]`
  - `--run-buffer-hex size:hex-bytes`
  - `--run-block x`
  - `--run-no-buffer`
  - `--run-scalar value`

  Use `expected_regex=` instead of `expected=` when the output spans
  multiple words, for example:

  ```text
  expected_regex=buf1\[0\]=0x12345678[\s\S]*buf1\[1\]=0x11111111
  ```

The runner invokes:

```text
rtxdisam --run-ptx <name>.ptx --ptx-entry <entry> <args>
```

and checks the expected value plus `validation_errors=0`.

## Corpus sweep

The local ZLUDA corpus under `ACuda/tests/zluda-ptx` is intentionally not
copied here. It is a local-only reference corpus and is never committed.
Instead, `scan_corpus.bat` compiles and validates every corpus entry in place:

```powershell
.\scan_corpus.bat
```

The updated local snapshot reports:

```text
ok=196 fail=0 skip=2
```

`corpus_success.txt` is the generated manifest of the 196 files that compile
and pass `spirv-val`. `corpus_failure.txt` is empty. Re-run the sweep after
updating ACuda to refresh both.

To check against the committed baseline without rewriting it:

```powershell
.\check_corpus.bat
```

This fails if any previously successful file now fails. New failures and
fixed files are reported separately.

The former failures now compile and run and have moved into this success kit.
`trap` remains under [`ptx_to_spv_error`](../ptx_to_spv_error/) as the
negative test. The remaining corpus failure, `extern_func`, requires
param-slot arrays and is not a standalone truth test.
