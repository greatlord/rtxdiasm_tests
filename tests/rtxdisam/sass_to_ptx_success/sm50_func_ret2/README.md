# sm50_func_ret2 SASS -> PTX success

This exercises a two-register function return. ACuda accepts only one scalar
return slot, so the two SASS return registers are packed into one `.u64`
return value:

```ptx
.func (.param .u64 __ret) fn_L__k_f(.param .u32 __a0)
{
    ld.param.u32 %r0, [__a0];
    add.s32 %r2, %r0, 0x1;
    mov.b64 %rdt0, {%r0, %r2};
    st.param.u64 [__ret], %rdt0;
    ret;
}
```

The caller unpacks the pair after `call.uni`:

```ptx
.param .u64 __fc0_r;
call.uni (__fc0_r), fn_L__k_f, (__fc0_a0);
ld.param.u64 %rdt0, [__fc0_r];
mov.b64 {%r0, %r2}, %rdt0;
```

More than two return registers still fall back to the inline path.

## Run

```powershell
.\run.bat
```

```text
input 3 -> pair (3, 4) -> sum 7
PASS (two-register function return)
```
