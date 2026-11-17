# sm35_leaf_cal_branchy SASS -> PTX success

This is CAL phase 2b: a leaf callee containing an internal loop is inlined at
the call site with cloned, uniquely named labels.

## PTX

`sm35_leaf_cal_branchy.ptx` calls a leaf function that computes `1 + ... + n`:

```ptx
.func (.param .u32 output) f(.param .u32 n) {
    ld.param.u32 r0, [n];
    mov.u32 r1, 0;
    setp.le.u32 p, r0, 0;
    @p bra done;
loop:
    add.u32 r1, r1, r0;
    sub.u32 r0, r0, 1;
    setp.gt.u32 p, r0, 0;
    @p bra loop;
done:
    st.param.u32 [output], r1;
    ret;
}
```

Compiled with `ptxas -O0 -arch=sm_35`, the callee is a separate branchy function
with local `.L_1`/`.L_2` labels.

## Lowering

The lifter clones the body, rewrites internal branch targets to unique
`L_inline_<serial>_...` labels, emits the loop, and omits the unreachable
callee definition:

```ptx
L_inline_0__k_f:
mov.b32 %r0, %r0;
mov.b32 %r2, 0;
setp.le.u32 %p0, %r0, 0;
@%p0 bra L_inline_0__L_1;
L_inline_0__L_2:
add.s32 %r2, %r2, %r0;
...
@%p0 bra L_inline_0__L_2;
L_inline_0__L_1:
mov.b32 %r0, %r2;
```

A branch whose target is outside the callee body is rejected cleanly, as are
callees using `R1`, constant-bank operands, barriers, carry-flag add/LEA
chains, nested calls, or structured reconvergence instructions.

## Run

```powershell
.\run.bat
```

```text
n=0 -> buf0[0]=0
n=4 -> buf0[0]=10
n=5 -> buf0[0]=15
PASS (branchy leaf CAL inline 0, 4, 5)
```
