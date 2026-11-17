# SASS -> PTX lifter error cases

This kit is for pre-Volta SASS shapes that the SASS -> PTX lifter cannot
translate yet. The lifter must fail cleanly, never emit incorrect PTX.

These are distinct from [`ptx_to_spv_error`](../ptx_to_spv_error/), which
covers PTX that ACuda's PTX -> SPIR-V backend rejects.

## Cases

There are currently no reproducing error cases with `run.bat` scripts.

The former cases moved to the success kit once their shapes were supported:

- [`atom_add_uniform`](../sass_to_ptx_success/atom_add_uniform/README.md)
  (interleaved `XMAD.MRG`)
- [`dynamic_ldc`](../sass_to_ptx_success/dynamic_ldc/README.md)
  (unbounded dynamic bank-0 `LDC`)

Non-empty leaf `CAL` is supported, including branchy leaf bodies with
internal labels, two-return diamonds, single-`RET` if/else diamonds, and
guarded returns; see
[`sm35_leaf_cal`](../sass_to_ptx_success/sm35_leaf_cal/README.md),
[`sm35_leaf_cal_branchy`](../sass_to_ptx_success/sm35_leaf_cal_branchy/README.md),
[`sm35_leaf_cal_early_return`](../sass_to_ptx_success/sm35_leaf_cal_early_return/README.md),
and
[`sm35_leaf_cal_single_merge`](../sass_to_ptx_success/sm35_leaf_cal_single_merge/README.md).
ABI-heavy callees using parameter slots, stack frames, structured
reconvergence, or other multiple-return shapes still reject cleanly.

## Run

```powershell
cd tests\sass_to_ptx_error
.\run_all.bat
```
