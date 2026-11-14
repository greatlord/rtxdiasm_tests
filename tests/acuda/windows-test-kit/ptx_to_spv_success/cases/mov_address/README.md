> This case now passes on the updated ACuda backend and is part of the
> success regression kit.

# mov_address PTX -> SPIR-V regression

This folder is a reproduction case for the PTX -> SPIR-V backend. It is not a
failure claim against the newer ACuda snapshot.

## PTX

`mov_address.ptx` takes the address of an entry-local array:

```ptx
.local .b8 __local_depot0[8];
mov.u64 temp, __local_depot0;
ret;
```

The entry writes no output. The test requires the kernel to compile, launch,
and synchronize successfully.

The local ACuda snapshot rejects it with:

```text
compile: PTX lower error: kernel needs at least one param in this subset
```

## NVIDIA truth

`run.bat` launches one block of one thread. The output buffer is checked for
the deterministic zero bytes:

```text
entry=mov_address block=1
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
