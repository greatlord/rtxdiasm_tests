# sm35_leaf_cal_single_merge SASS -> PTX success

This extends leaf `CAL` if-conversion to an if/else with a common merge and a
single `RET`.

## PTX

`sm35_leaf_cal_single_merge.ptx` calls a leaf function where both arms converge
before one return:

```ptx
.func (.param .u32 output) f(.param .u32 x) {
    ld.param.u32 r0, [x];
    setp.lt.u32 p, r0, 10;
    @p bra small;
    add.u32 r0, r0, 100;
    bra done;
small:
    add.u32 r0, r0, 1;
done:
    st.param.u32 [output], r0;
    ret;
}
```

Compiled with `ptxas -O0 -arch=sm_35`, the callee has a conditional branch, an
unconditional skip, and a single `RET` after the merge.

## Lowering

The lifter identifies the merge point, predicates the two straight-line arms,
then emits the merge block normally:

```ptx
setp.lt.u32 %p0, %r0, 0xa;
@!%p0 add.s32 %r0, %r0, 0x64;
@!%p0 mov.b32 %r0, %r0;
@%p0 add.s32 %r0, %r0, 0x1;
@%p0 mov.b32 %r0, %r0;
mov.b32 %r0, %r0;
```

This avoids the branch-to-merge diamond that ACuda currently rejects.

## Run

```powershell
.\run.bat
```

```text
x=5  -> buf0[0]=6
x=15 -> buf0[0]=115
PASS (single-RET diamond CAL if-conversion 5 and 15)
```
