# sm50_dynamic_ldc SASS -> PTX success

This is the bounded dynamic bank-0 `LDC` subset. An address register with a
proven upper bound is lowered through a shared parameter-block copy.

## SASS

`sm50_dynamic_ldc.sass` bounds the selector before forming the affine
constant-bank address, then loads one of three 64-bit table entries:

```text
MOV R0, c[0x0][0x140]          ; selector
MOV32I R1, 0x148
ISETP.LT.U32.AND P0, PT, R0, 0x3, PT
@!P0 MOV32I R0, 0x0            ; clamp out-of-range selector
SHL R0, R0, 0x3                ; selector * 8
IADD R0, R0, R1                ; address = base + selector * 8
LDC.64 R2, c[0x0][R0]
MOV R4, c[0x0][0x160]
STG.E [R4], R2
EXIT
```

## Lowering

The lifter copies the bounded parameter block (`0x140..0x160`) into shared
memory once at kernel entry, then indexes it with a clamped relative offset:

```ptx
ld.param.u64 %rdt0, [p1];
mov.b64 {%rt1, %rt2}, %rdt0;
st.shared.u32 [__shared+0x8], %rt1;
st.shared.u32 [__shared+0xc], %rt2;
...
setp.ge.u32 %pt0, %r0, 0x140;
@%pt0 sub.u32 %rt7, %r0, 0x140;
@!%pt0 mov.u32 %rt7, 0;
setp.lt.u32 %pt1, %rt7, 32;
@!%pt1 mov.u32 %rt7, 0;
ld.shared.u32 %r2, [%rdt4+0x0];
ld.shared.u32 %r3, [%rdt4+0x4];
```

The index is clamped rather than branched, so the lifted PTX keeps a single
structured exit. Out-of-range selectors are defined to read entry `0`.

Parameters that lie beyond the proven bound are not copied, so pointer-valued
output parameters do not get stored as scalar shared data. `IMAD`, `SHL` +
`IADD`, and `IADD`/`IADD32I` address forms are recognized; `LDC.128` is
rejected explicitly.

## Run

```powershell
.\run.bat
```

```text
selector 0 -> buf0[0]=11
selector 1 -> buf0[0]=22
selector 2 -> buf0[0]=33
selector 3 -> buf0[0]=11 (clamp)
PASS (bounded dynamic LDC 0..2, clamped out-of-range)
```

Unbounded dynamic `LDC` remains a clean rejection; see
[`dynamic_ldc`](../../sass_to_ptx_error/dynamic_ldc/README.md).
