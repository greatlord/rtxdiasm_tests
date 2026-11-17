# sm50_switch SASS -> PTX success

This case exercises the switch-to-select lowering.

## SASS

`sm50_switch.sass` implements:

```text
selector 0 -> 1
selector 1 -> 2
selector 2 -> 3
selector 3 -> 4
default    -> 5
```

ptxas-style SASS uses nested `ISETP`/`BRA` dispatch and per-case
`MOV32I`/`STG`/`BRA` blocks.

## Lowering

The lifter recognizes the simple constant-case switch and emits a branchless
select:

```ptx
mov.b32 %rt0, %r0;
mov.u32 %rt1, 5;
setp.eq.u32 %pt0, %rt0, 0;
@%pt0 mov.u32 %rt1, 1;
...
st.global.u32 [%rpptr0+0], %rt1;
ret;
```

This is structured, so ACuda accepts it.

## Run

```powershell
.\run.bat
```

```text
PASS (switch-to-select 0..4)
```
