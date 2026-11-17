# sm50_add_pair_bar SASS -> PTX success

This case exercises a generic `IADD32I.CC` + `IADD.X` 64-bit pointer add,
pointer lineage through `MOV`, and a barrier.

## SASS

```text
MOV     R2, c[0x0][0x140]
MOV     R3, c[0x0][0x144]
MOV     R4, R2
MOV     R5, R3
IADD32I R6.CC, R4, 0x4
IADD.X  R7, R5, RZ
BAR.SYNC 0x0
LDG.E   R0, [R6]
MOV     R8, c[0x0][0x148]
STG.E   [R8], R0
EXIT
```

## Lowering

```ptx
ld.param.u32 %r2, [p0+0];
ld.param.u32 %r3, [p0+4];
mov.b32 %r4, %r2;
mov.b32 %r5, %r3;
add.u64 %rdt0, %rpptr0, 0x4;
cvta.to.global.u64 %rdt1, %rdt0;
bar.sync 0;
ld.global.u32 %r0, [%rdt1+0];
st.global.u32 [%rpptr1+0], %r0;
ret;
```

Pointer identity from `p0` is propagated through the `MOV` copies, so the
generic add uses `%rpptr0` directly and keeps global-pointer lineage.

## Run

```powershell
.\run.bat
```

Input word at `input+4` is `42`; output is `42`.

```text
PASS (generic add pair + barrier)
```
