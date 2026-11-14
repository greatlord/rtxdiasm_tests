# sm35_leaf_cal_multi_ret SASS -> PTX success

This exercises the general multi-return leaf `CAL` lowering. ACuda now merges
the branch-to-inline-exit shape, so the lifter no longer needs to restrict
multi-return callees to if-convertible diamonds.

## PTX

`sm35_leaf_cal_multi_ret.ptx` calls a leaf function with three return paths:

```ptx
.func (.param .u32 output) f(.param .u32 x) {
    ld.param.u32 r0, [x];
    setp.lt.u32 p, r0, 10;
    @p bra small;
    setp.gt.u32 p, r0, 20;
    @p bra large;
    add.u32 r0, r0, 100;
    st.param.u32 [output], r0;
    ret;
small:
    add.u32 r0, r0, 1;
    st.param.u32 [output], r0;
    ret;
large:
    add.u32 r0, r0, 1000;
    st.param.u32 [output], r0;
    ret;
}
```

Compiled with `ptxas -O0 -arch=sm_35`, the callee has two conditional branches
and three `RET`s.

## Lowering

Each reachable `RET` becomes a branch to a unique inline-exit label. The final
`RET` falls through to the exit label, then the caller continues.

```ptx
@%p0 bra L_inline_0__L_1;
setp.gt.u32 %p0, %r0, 0x14;
@%p0 bra L_inline_0__L_2;
add.s32 %r0, %r0, 0x64;
...
bra L_inline_0_exit;
L_inline_0__L_1:
...
bra L_inline_0_exit;
L_inline_0__L_2:
...
L_inline_0_exit:
```

## Run

```powershell
.\run.bat
```

```text
x=5  -> buf0[0]=6
x=15 -> buf0[0]=115
x=25 -> buf0[0]=1025
PASS (multi-return leaf CAL 5, 15, 25)
```
