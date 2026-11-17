# sm35_brx SASS -> PTX success

This case exercises constant-bank jump-table lifting with `BRX`.

## PTX

`sm35_brx.ptx` uses a PTX branch-target table:

```ptx
ld.param.u32 s, [selector];
ts:
    .branchtargets L0, L1, L2;
    brx.idx s, ts;
L0: mov.u32 result, 10; ret;
L1: mov.u32 result, 11; ret;
L2: mov.u32 result, 12; ret;
```

Compiled with CUDA 10.2 `ptxas -arch=sm_35`, the table becomes
`LDC c[0x2][R0]` plus `BRX`.

## Lowering

The lifter resolves the `.nv.constant2` table to SASS target addresses and
lowers the indirect branch to a compare/branch chain with synthesized labels.
All three selectors execute a case block and exit cleanly.

## Run

```powershell
.\run.bat
```

```text
PASS (BRX jump table 0..2)
```
