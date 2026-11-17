> This case now passes on the updated ACuda backend and is part of the
> success regression kit.

# rcp_f64 PTX -> SPIR-V regression

This folder is a reproduction case for the PTX -> SPIR-V backend.  It is not
a failure claim against the newer ACuda snapshot: the updated ZLUDA test kit is
reported to pass 100%, but that code is not present in the local snapshot.

## PTX

`rcp_f64.ptx` loads an `f64`, executes `rcp.rz.f64`, and stores the result.

The local ACuda snapshot rejects it with:

```text
PTX lower error: line 18: unsupported instruction 'rcp.rz.f64'
```

## NVIDIA truth

`test_nvidia.cpp` loads the PTX through the CUDA driver API and checks:

```text
rcp_f64(4.0) = 0.25
```

The test is deterministic and must always pass on the RTX 3070.

## Run on NVIDIA

```powershell
.\build_and_run.bat
```

Expected:

```text
rcp_f64(4.0) = 0.25
PASS
```

## Expected new-backend behavior

When the newer PTX -> SPIR-V backend is available, it should compile this PTX
successfully and produce the same `0.25` result.
