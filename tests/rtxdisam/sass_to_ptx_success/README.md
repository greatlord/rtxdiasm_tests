# SASS -> PTX lifter success cases

This kit exercises SASS -> PTX -> ACuda -> Vulkan for cubin shapes that now
lift correctly. The `dynamic_ldc` case validates the generated PTX; the
ACuda-side dynamic-constant path is covered by
`tests/ptx_to_spv_success/cases/dynamic_const`. It is the companion to
[`sass_to_ptx_error`](../sass_to_ptx_error/README.md).

## Cases

- [`sm50_icmp`](sm50_icmp/README.md) — `ICMP.LE` compare-to-zero select,
  checked for positive and negative inputs.
- [`sm50_pred_atom`](sm50_pred_atom/README.md) — predicated global atomic
  max with `SSY`/`SYNC` reconvergence and a four-thread skip path.
- [`sm30_shared_add`](sm30_shared_add/README.md) - shared atomic add
  emulated with `LDSLK`/`STSCUL` and a CAS retry loop, checked with one and
  four threads.
- [`atom_add_uniform`](atom_add_uniform/README.md) - warp-aggregated shared
  atomic add with interleaved `XMAD.MRG` chains.
- [`dynamic_ldc`](dynamic_ldc/README.md) - unbounded runtime bank-0 `LDC.64`
  lowered to dynamic `ld.const` loads.
- [`sm50_fadd32i`](sm50_fadd32i/README.md) — `FADD32I` with a float
  immediate, assembled directly from SASS and checked with two inputs.
- [`sm50_const_ldc`](sm50_const_ldc/README.md) — constant-offset bank-0
  `LDC`, assembled from SASS and checked with two scalar inputs.
- [`sm35_empty_cal`](sm35_empty_cal/README.md) — direct `CAL` to an empty
  leaf function, lifted and launched through ACuda.
- [`sm50_add_pair_bar`](sm50_add_pair_bar/README.md) — generic
  `IADD32I.CC`/`IADD.X` pointer add, pointer lineage through `MOV`, and a
  barrier.
- [`sm50_switch`](sm50_switch/README.md) — simple constant-case switch
  lowered to a branchless select and checked for selectors 0–4.
- [`sm35_brx`](sm35_brx/README.md) — constant-bank `LDC` + `BRX` jump table,
  checked for selectors 0–2.
- [`sm50_dynamic_ptr_ldc`](sm50_dynamic_ptr_ldc/README.md) — bounded dynamic
  pointer-table `LDC.64` selecting and dereferencing one of three pointers.
- [`sm50_dynamic_ldc`](sm50_dynamic_ldc/README.md) — bounded dynamic bank-0
  `LDC.64` via a shared parameter-block copy, checked for selectors 0–3.
- [`sm35_leaf_cal`](sm35_leaf_cal/README.md) — straight-line non-empty leaf
  `CAL` inlined at the call site.
- [`sm35_leaf_cal_branchy`](sm35_leaf_cal_branchy/README.md) — branchy leaf
  `CAL` inlined with cloned, uniquely named labels.
- [`sm35_leaf_cal_early_return`](sm35_leaf_cal_early_return/README.md) — leaf
  `CAL` with two return paths, if-converted to complementary predicates.
- [`sm35_leaf_cal_single_merge`](sm35_leaf_cal_single_merge/README.md) —
  single-`RET` if/else diamond if-converted before a common merge.
- [`sm35_brx4_select`](sm35_brx4_select/README.md) — 4-entry `BRX` with a
  common merge lowered to a branchless select.
- [`sm35_leaf_cal_multi_ret`](sm35_leaf_cal_multi_ret/README.md) — general
  multi-return leaf `CAL` lowered through a shared inline-exit label.
- [`sm50_leaf_cal_nested`](sm50_leaf_cal_nested/README.md) — nested direct
  leaf calls `k -> f -> g`, inlined recursively.
- [`sm50_func_call`](sm50_func_call/README.md) — opt-in real `.func`/`call`
  for a straight-line leaf with one scalar argument and return.
- [`sm50_func_ptr`](sm50_func_ptr/README.md) — opt-in `.func`/`call` with two
  `.u64` pointer arguments plus global load/store.
- [`sm50_func_ret2`](sm50_func_ret2/README.md) - two-register return packed
  into one `.u64` function return slot.
- [`sm35_func_local`](sm35_func_local/README.md) - real `.func` with a callee
  local frame, inlined through fresh caller-local arrays.
- [`reloc_jcal`](reloc_jcal/README.md) - relocatable `JCAL` across separate
  `.text` sections, lifted from a `ptxas -c` object.
- [`sm50_func_shared`](sm50_func_shared/README.md) — shared memory and barriers
  inside a `.func`.
- [`sm50_func_pred`](sm50_func_pred/README.md) — internal predicate state
  (`ISETP`/`SEL`) inside a `.func`.
- [`sm50_func_carry`](sm50_func_carry/README.md) — internal `IADD.CC`/`IADD.X`
  carry chain inside a `.func`.
- [`sm35_leaf_cal_local_frame`](sm35_leaf_cal_local_frame/README.md) — leaf
  `CAL` with `R1`-relative `STL`/`LDL` spills mapped to a fixed local frame.
- [`sm35_leaf_cal_local64`](sm35_leaf_cal_local64/README.md) — 64-bit
  local-frame spills (`STL.64`/`LDL.64`).
- [`sm35_leaf_cal_local_array`](sm35_leaf_cal_local_array/README.md) — local
  array addressing through `R1` arithmetic and computed `LD.E`/`ST.E`.
- [`sm50_leaf_cal_param_slots`](sm50_leaf_cal_param_slots/README.md) — 17
  arguments with three spilled to caller parameter slots before `CAL`.
- [`sm50_leaf_cal_ret_slots`](sm50_leaf_cal_ret_slots/README.md) — a 20-word
  structure return written to caller return slots and reloaded after `CAL`.
- [`sm50_leaf_cal_structured`](sm50_leaf_cal_structured/README.md) — leaf
  `CAL` with `SSY`/`SYNC` structured reconvergence.

## Run

```powershell
cd tests\sass_to_ptx_success
.\run_all.bat
```

Each case builds its cubin from PTX with CUDA 10.2 `ptxas` and checks the
ACuda output.
