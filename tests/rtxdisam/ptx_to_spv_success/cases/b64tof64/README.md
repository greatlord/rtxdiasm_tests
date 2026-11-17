> This case now passes on the updated ACuda backend and is part of the
> success regression kit.

# b64tof64 PTX -> SPIR-V regression

This folder is a reproduction case for the PTX -> SPIR-V backend.  It is not a
failure claim against the newer ACuda snapshot.

## PTX

`b64tof64.ptx` bitcasts the input pointer through an `f64` register, loads a
64-bit value, and stores it unchanged.

The local ACuda snapshot rejects it with:

```text
PTX lower error: line 16: type mismatch on assignment
```

## NVIDIA truth

`test_nvidia.cpp` loads the PTX through the CUDA driver API and checks:

```text
b64tof64(1.5) = 1.5
```

The test is deterministic and must always pass on the RTX 3070.

## Run on NVIDIA

```powershell
.\build_and_run.bat
```

Expected:

```text
b64tof64(1.5) = 1.5
PASS
```
