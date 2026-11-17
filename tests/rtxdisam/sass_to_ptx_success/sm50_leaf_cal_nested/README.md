# sm50_leaf_cal_nested SASS -> PTX success

This exercises nested direct calls:

```text
k -> f -> g
```

At `ptxas -O0`, both edges remain `CAL` instructions and the functions are
laid out sequentially:

```text
k:  ... CAL `($k$f)
$k$f: ... CAL `($k$g) ... RET
$k$g: ... RET
```

The lifter recursively inlines `g` into `f` and then `f` into `k`, using a
fresh label prefix per call site. Recursion is rejected through the inline
call stack.

## Run

```powershell
.\run.bat
```

```text
x=5 -> g adds 1 -> f adds 1 -> buf0[0]=7
PASS (nested leaf CAL inline)
```
