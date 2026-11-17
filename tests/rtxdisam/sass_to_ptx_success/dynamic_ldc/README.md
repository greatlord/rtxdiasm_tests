# dynamic_ldc SASS -> PTX success

This case covers a dynamic bank-0 `LDC.64` at a runtime constant-bank
offset.

## SASS

```text
MOV R2, c[0x0][0x140]
LDC.64 R2, c[0x0][R2]
EXIT
```

The constant-bank address is not known until runtime.  The lifter emits a
temporary address register and two `ld.const.u32` operations:

```ptx
mov.u32 %rt0, %r2;
ld.const.u32 %r2, [%rt0];
ld.const.u32 %r3, [%rt0+4];
```

When parameter metadata is available, `--ptx-param` can describe the value at
`0x140`.

## Run

```powershell
.\run.bat
```

```text
PASS (dynamic bank-0 LDC)
```
