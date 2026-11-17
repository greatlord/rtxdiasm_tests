# sm50_fadd32i SASS -> PTX success

This case exercises `FADD32I` with a normal float immediate and bitcast
round-tripping.

## SASS

`sm50_fadd32i.sass` is assembled directly with the project assembler:

```text
MOV     R2, c[0x0][0x140]
LDG.E   R2, [R2]
FADD32I R3, R2, 1.5
MOV     R4, c[0x0][0x148]
STG.E   [R4], R3
EXIT
```

## Lowering

The lifter decodes the immediate as a 32-bit float bit pattern and emits:

```ptx
ld.global.u32 %r2, [%rpptr0+0];
mov.b32 %ft0, %r2;
mov.b32 %ft1, 0x3fc00000;
add.f32 %ft2, %ft0, %ft1;
mov.b32 %r3, %ft2;
st.global.u32 [%rpptr1+0], %r3;
```

The final `mov.b32` keeps the SASS destination as a b32 register, which
avoids a SPIR-V pointer/object type mismatch in ACuda.

`FADD32I.CC` is accepted only when the carry is overwritten before any
carry-consuming `.X` instruction; otherwise the lifter rejects the
instruction.

## Run

```powershell
.\run.bat
```

```text
2.0 + 1.5 -> buf1[0]=1080033280 (0x40600000)
0.5 + 1.5 -> buf1[0]=1073741824 (0x40000000)
PASS (FADD32I 2.0+1.5 and 0.5+1.5)
```
