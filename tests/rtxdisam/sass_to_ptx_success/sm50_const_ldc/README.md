# sm50_const_ldc SASS -> PTX success

This case exercises constant-offset bank-0 `LDC` end to end.

## SASS

`sm50_const_ldc.sass` is assembled directly with the project assembler:

```text
MOV32I R0, 0x140
LDC    R0, c[0x0][R0]
MOV    R2, c[0x0][0x148]
STG.E  [R2], R0
EXIT
```

`0x140` is the input scalar parameter offset; `0x148` is the output pointer.

## Lowering

```ptx
mov.u32 %r0, %rpp0;
st.global.u32 [%rpptr1+0], %r0;
ret;
```

The `MOV32I` address setup and `LDC` are folded into a direct parameter load.

## Run

```powershell
.\run.bat
```

```text
scalar 42 -> buf0[0]=42
scalar 7  -> buf0[0]=7
PASS (constant-offset LDC 42 and 7)
```
