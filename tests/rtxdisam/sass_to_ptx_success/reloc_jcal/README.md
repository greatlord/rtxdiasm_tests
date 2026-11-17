# reloc_jcal SASS -> PTX success

This case exercises a relocatable object produced by `ptxas -c`. The caller
and callee live in separate `.text` sections and the call is a `JCAL` with a
relocation to `f`:

```text
.text.k: JCAL 0x0                  (*"RELOCREL ... f ..."*)
.text.f: IADD32I R4, R4, 0x7
```

The CLI preserves relocation annotations when lifting relocatable cubins,
resolves the `JCAL` symbol, and inlines the callee from the linked `.text.f`
section.

## Run

```powershell
.\run.bat
```

`x = 3` produces `x + 7 = 10`.

```text
buf0[0]=10 (0x0000000a)
PASS (relocatable JCAL / separate .text)
```
