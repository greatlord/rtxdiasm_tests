# sm50_func_shared SASS -> PTX success

This exercises shared memory inside an opt-in `.func`. The callee stores a
value to the module-wide `__shared` array, synchronizes, reloads it, adds one,
and returns the result. The `.func` contains `st.shared`, `bar.sync`,
`membar.gl`, and `ld.shared`.

## Run

```powershell
.\run.bat
```

```text
5 -> 6
PASS (shared memory in .func)
```
