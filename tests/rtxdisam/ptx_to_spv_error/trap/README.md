# trap PTX -> SPIR-V regression

This folder is a reproduction case for the PTX -> SPIR-V backend. It is not a
failure claim against the newer ACuda snapshot.

## PTX

`trap.ptx` executes the trap instruction:

```ptx
trap;
ret;
```

On NVIDIA the kernel launches but the trap makes synchronization fail with
`CUDA_ERROR_LAUNCH_FAILED`. The test expects that error rather than an output
value.

The local ACuda snapshot rejects it with:

```text
compile: PTX parse error: line 9: unsupported instruction 'trap'
```

## NVIDIA truth

`run.bat` launches one block of one thread and requires a launch or
synchronize error:

```text
entry=trap block=1 expect_error=1
error   =CUDA_ERROR_LAUNCH_FAILED
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
