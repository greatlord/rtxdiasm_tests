> This case now passes on the updated ACuda backend and is part of the
> success regression kit.

# bar_sync PTX -> SPIR-V regression

This folder is a reproduction case for the PTX -> SPIR-V backend. It is not a
failure claim against the newer ACuda snapshot.

## PTX

`bar_sync.ptx` uses a register barrier id:

```ptx
.reg .u32 temp_32;
bar.sync temp_32;
ret;
```

The entry has no parameters and no output. The test only requires the kernel
to compile, launch, and synchronize successfully.

The local ACuda snapshot rejects it with:

```text
compile: PTX parse error: line 8: bar.sync needs a barrier id 0..15
```

## NVIDIA truth

`run.bat` launches one block of one thread. The output buffer is checked for
the deterministic zero bytes:

```text
entry=bar_sync block=1
input   =00
expected=00000000
actual  =00000000
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
