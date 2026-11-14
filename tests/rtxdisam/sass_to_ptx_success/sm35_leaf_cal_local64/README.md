# sm35_leaf_cal_local64 SASS -> PTX success

This extends the leaf-CAL local frame to 64-bit spills. The callee stores a
`u64` to a local array and reloads it:

```ptx
.func (.param .u32 output) f(.param .u32 x) {
    .local .align 8 .b8 tmp[16];
    ld.param.u32 x0, [x];
    cvt.u64.u32 q, x0;
    st.local.u64 [tmp], q;
    ld.local.u64 r, [tmp];
    cvt.u32.u64 r0, r;
    st.param.u32 [output], r0;
    ret;
}
```

Compiled with `ptxas -O0 -arch=sm_35`, the callee uses `STL.64`/`LDL.64`.

## Lowering

```ptx
.local .align 16 .b8 __callee_frame[16];
...
st.local.v2.u32 [__callee_frame+0], {%r2, %r3};
ld.local.v2.u32 {%r2, %r3}, [__callee_frame+0];
```

## Run

```powershell
.\run.bat
```

```text
x=5  -> buf0[0]=5
x=10 -> buf0[0]=10
PASS (leaf CAL 64-bit local frame 5 and 10)
```
