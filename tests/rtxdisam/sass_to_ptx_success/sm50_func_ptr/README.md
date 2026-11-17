# sm50_func_ptr SASS -> PTX success

This extends the opt-in real `.func`/`call` path to global-memory pointer
arguments.

The callee takes two pointer arguments, loads one word, stores `value + 1` to
the second pointer, and returns `value + 2`:

```ptx
.func (.param .u32 __ret) fn_L__k_f(
    .param .u64 __a0, .param .u64 __a1)
{
    .reg .b64 %rdfn<2>;
    ld.param.u64 %rdfn0, [__a0];
    ld.param.u64 %rdfn1, [__a1];
    ld.global.u32 %r2, [%rdfn0+0];
    add.s32 %r0, %r2, 0x1;
    st.global.u32 [%rdfn1+0], %r0;
    add.s32 %r2, %r2, 0x2;
    mov.b32 %r0, %r2;
    st.param.u32 [__ret], %r0;
    ret;
}
```

SASS address register pairs become one `.u64` function parameter. The caller
passes its existing pointer mapping:

```ptx
.param .u64 __fc0_a0;
st.param.u64 [__fc0_a0], %rpptr0;
.param .u64 __fc0_a1;
st.param.u64 [__fc0_a1], %rpptr1;
call.uni (__fc0_r), fn_L__k_f, (__fc0_a0, __fc0_a1);
```

Register liveness around the call determines the return set. The pointer
registers are callee scratch, so only the scalar result is returned.

## Run

```powershell
.\run.bat
```

```text
input 41 -> pout=42, result=43
PASS (function pointer args and global memory)
```
