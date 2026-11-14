# sm35_brx4_select SASS -> PTX success

This is the 4-entry `BRX` jump table that used to hit ACuda's structured-exit
validation. The lifter now detects constant-move arms with a common merge and
emits a branchless select instead of a compare/branch chain.

## SASS source

`sm35_brx4_select.ptx` contains:

```ptx
ts:
    .branchtargets L0, L1, L2, L3;
    brx.idx s, ts;
L0: mov.u32 result, 10; bra done;
L1: mov.u32 result, 11; bra done;
L2: mov.u32 result, 12; bra done;
L3: mov.u32 result, 13;
done:
    st.global.u32 [p], result;
    ret;
```

## Lowering

The table index is still the scaled byte offset, but the four arms become
predicated moves:

```ptx
mov.u32 %r4, 0xa;
setp.eq.u32 %pt0, %r0, 4;
@%pt0 mov.u32 %r4, 0xb;
setp.eq.u32 %pt1, %r0, 8;
@%pt1 mov.u32 %r4, 0xc;
setp.eq.u32 %pt2, %r0, 12;
@%pt2 mov.u32 %r4, 0xd;
L__L_1:
st.global.u32 [%rpptr1+0], %r4;
```

This avoids the structured-selection failure recorded in
[`ptx_to_spv_success/brx4_merge`](../../ptx_to_spv_success/cases/brx4_merge/).

## Run

```powershell
.\run.bat
```

```text
selector 0 -> buf0[0]=10
selector 1 -> buf0[0]=11
selector 2 -> buf0[0]=12
selector 3 -> buf0[0]=13
PASS (4-entry BRX branchless select 0..3)
```
