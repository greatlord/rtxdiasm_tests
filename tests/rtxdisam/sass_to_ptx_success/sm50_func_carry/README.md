# sm50_func_carry SASS -> PTX success

This exercises an internal carry chain in an opt-in `.func`. The callee turns
two pairs of 32-bit arguments into 64-bit values, adds them (`IADD.CC` +
`IADD.X`), and returns the two-word result.

## Run

```powershell
.\run.bat
```

```text
1 + 2 = 3
PASS (carry chain in .func)
```
