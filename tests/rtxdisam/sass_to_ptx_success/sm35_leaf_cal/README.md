# sm35_leaf_cal SASS -> PTX success

This is CAL phase 2: a straight-line leaf callee with no parameter-slot
arguments or returns is inlined at the call site.

## PTX

`sm35_leaf_cal.ptx` calls a leaf function that writes `7` through its pointer
argument:

```ptx
.func f(.param .u64 out) {
    ld.param.u64 p, [out];
    st.global.u32 [p], 7;
    ret;
}
.visible .entry k(.param .u64 out) {
    call.uni f, (arg);
    ret;
}
```

Compiled with `ptxas -O0 -arch=sm_35`, the callee is straight-line:

```text
$k$f:
MOV R2, R2
MOV R3, R3
MOV R2, R2
MOV R3, R3
MOV32I R0, 0x7
ST.E   [R2], R0
RET
```

## Lowering

The lifter inlines the body and omits the unreachable callee definition:

```ptx
mov.u32 %r0, 0x7;
st.global.u32 [%rpptr0+0], %r0;
ret;
```

The callee must not use `R1` or constant-bank operands and must not contain
nested calls, barriers, or structured reconvergence instructions. Branchy leaf
callees with internal labels are supported separately; see
[`sm35_leaf_cal_branchy`](../sm35_leaf_cal_branchy/README.md). Other callee
shapes still reject cleanly.

## Run

```powershell
.\run.bat
```

```text
PASS (straight-line leaf CAL inline)
```
