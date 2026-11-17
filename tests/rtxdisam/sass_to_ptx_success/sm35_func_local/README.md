# sm35_func_local SASS -> PTX success

This exercises `--ptx-functions` for a real `.func` whose callee has a local
frame:

```ptx
.func (.param .u32 __ret) fn_L__k_f(.param .u32 __a0)
{
    .local .align 16 .b8 __callee_frame[16];
    st.local.u32 [__callee_frame+0], %r0;
    ...
    call.uni (__fc0_r), fn_L__k_f, (__fc0_a0);
}
```

ACuda lowers `.local` inside an inlined callee by allocating a fresh
caller-local backing array for every call and renaming the callee's local
accesses to it.

## Run

```powershell
.\run.bat
```

`x = 5` and `x = 10` produce `11` and `21`.
