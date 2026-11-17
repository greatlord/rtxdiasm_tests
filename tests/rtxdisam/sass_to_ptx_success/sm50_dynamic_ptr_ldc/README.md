# sm50_dynamic_ptr_ldc SASS -> PTX success

This is the pointer-table version of bounded dynamic bank-0 `LDC`. A bounded
`LDC.64` selects one of three pointer parameters, then dereferences it.

## SASS

```text
MOV R0, c[0x0][0x140]          ; selector
MOV32I R1, 0x148
ISETP.LT.U32.AND P0, PT, R0, 0x3, PT
@!P0 MOV32I R0, 0x0            ; clamp
SHL R0, R0, 0x3
IADD R0, R0, R1                ; address = base + selector * 8
LDC.64 R2, c[0x0][R0]          ; selected pointer
LDG.E R5, [R2]
MOV R4, c[0x0][0x160]
STG.E [R4], R5
EXIT
```

## Lowering

The parameter block is copied into shared memory. The dynamic `LDC.64` loads
the selected pointer's two words from shared memory and packs them into a
64-bit address for `ld.global`.

```ptx
ld.shared.u32 %r2, [%rdt4+0x0];
ld.shared.u32 %r3, [%rdt4+0x4];
mov.b64 %rdt5, {%r2, %r3};
...
ld.global.u32 %r5, [%rdt5+0];
```

The updated ACuda accepts this shared-loaded pointer address.

## Run

```powershell
.\run.bat
```

```text
selector 0 -> buf3[0]=10
selector 1 -> buf3[0]=20
selector 2 -> buf3[0]=30
selector 3 -> buf3[0]=10 (clamp)
PASS (dynamic pointer LDC 0..2, clamped out-of-range)
```
