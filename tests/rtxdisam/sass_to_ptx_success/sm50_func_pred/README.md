# sm50_func_pred SASS -> PTX success

This exercises internal predicate state in an opt-in `.func`. The callee uses
`ISETP` plus `SEL` to choose one argument, adds 100, and returns the result.
No predicate is live across the call boundary.

## Run

```powershell
.\run.bat
```

```text
min(3, 4) + 100 = 103
PASS (internal predicate in .func)
```
