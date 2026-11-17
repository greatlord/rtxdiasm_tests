# sm35_leaf_cal_early_return SASS -> PTX success

This extends leaf `CAL` inlining to a callee with two return paths.

## PTX

`sm35_leaf_cal_early_return.ptx` calls a leaf function with an early return:

```ptx
.func (.param .u32 output) f(.param .u32 x) {
    ld.param.u32 r0, [x];
    setp.lt.u32 p, r0, 10;
    @p bra small;
    add.u32 r1, r0, 100;
    st.param.u32 [output], r1;
    ret;
small:
    add.u32 r1, r0, 1;
    st.param.u32 [output], r1;
    ret;
}
```

Compiled with `ptxas -O0 -arch=sm_35`, the callee is a two-return diamond.

## Lowering

Instead of branching to a shared inline-exit label, the lifter if-converts the
diamond: the prefix executes normally, then the two straight-line arms execute
under complementary predicates.

```ptx
setp.lt.u32 %p0, %r0, 0xa;
@!%p0 add.s32 %r0, %r0, 0x64;
@!%p0 mov.b32 %r0, %r0;
@%p0 add.s32 %r0, %r0, 0x1;
@%p0 mov.b32 %r0, %r0;
```

This avoids the `bra`-to-merge shape that ACuda currently rejects with
`diamond merges an undefined value`.

Supported shape: one forward conditional branch, two straight-line arms, each
ending in an unguarded `RET`, and only predicable scalar instructions in the
arms. Other multi-return or guarded-`RET` shapes still reject cleanly.

## Run

```powershell
.\run.bat
```

```text
x=5  -> buf0[0]=6
x=15 -> buf0[0]=115
PASS (early-return leaf CAL if-conversion 5 and 15)
```
