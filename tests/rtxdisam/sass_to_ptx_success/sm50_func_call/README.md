# sm50_func_call SASS -> PTX success

This exercises **real PTX `.func`/`call` emission** instead of inlining.
Pass `--ptx-functions` to enable it:

```powershell
dotnet rtxdisam.dll --lift-ptx sm50_func_call.cubin --ptx-entry k `
  --ptx-param x:u32:0x140 --ptx-param out:u64:0x148 --ptx-functions
```

The callee is a straight-line leaf with one argument and one return register:

```text
$k$f:
    ... add 7 ...
    RET
```

It is lifted to a module-level function with an explicit argument and a
scalar return slot:

```ptx
.func (.param .u32 __ret) fn_L__k_f(.param .u32 __a0)
{
    .reg .b32 %r<1>;
    ld.param.u32 %r0, [__a0];
    add.s32 %r0, %r0, 0x7;
    st.param.u32 [__ret], %r0;
    ret;
}

.visible .entry k(...)
{
    ...
    .param .u32 __fc0_a0;
    st.param.u32 [__fc0_a0], %r0;
    .param .u32 __fc0_r;
    call.uni (__fc0_r), fn_L__k_f, (__fc0_a0);
    ld.param.u32 %r0, [__fc0_r];
    ...
}
```

The current function ABI is intentionally narrow: straight-line body, one
scalar return register, no memory/local/shared access, no predicates, no carry
chains, and no nested calls. Other callees stay on the inline path. ACuda only
supports a single scalar return slot, so multi-register returns are not emitted
as `.func` yet.

## Run

```powershell
.\run.bat
```

```text
5 + 7 -> buf0[0]=12
PASS (real .func/call)
```
