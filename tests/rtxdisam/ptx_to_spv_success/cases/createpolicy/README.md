> This case now passes on the updated ACuda backend and is part of the
> success regression kit.

# createpolicy PTX -> SPIR-V regression

This folder is a reproduction case for the PTX -> SPIR-V backend. It is not a
failure claim against the newer ACuda snapshot.

## PTX

`createpolicy.ptx` creates an L2 cache policy:

```ptx
createpolicy.fractional.L2::evict_last.b64 policy, 1.0;
```

The entry has two parameters but writes no output. The test requires the
kernel to compile, launch, and synchronize successfully.

The local ACuda snapshot rejects it with:

```text
compile: PTX parse error: line 11: expected directive, label or instruction
```

## NVIDIA truth

`run.bat` launches one block of one thread. The output buffer is checked for
the deterministic zero bytes:

```text
entry=createpolicy block=1
input   =00
expected=0000000000000000
actual  =0000000000000000
PASS
```

## Run on NVIDIA

```powershell
.\run.bat
```

The shared driver-API checker is built automatically on first use. To rebuild
it explicitly:

```powershell
..\common\build_test_nvidia.bat
```
