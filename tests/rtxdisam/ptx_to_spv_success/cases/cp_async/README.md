> This case now passes on the updated ACuda backend and is part of the
> success regression kit.

# cp_async PTX -> SPIR-V regression

This folder is a reproduction case for the PTX -> SPIR-V backend. It is not a
failure claim against the newer ACuda snapshot.

## PTX

`cp_async.ptx` uses an entry-local global array and an asynchronous copy:

```ptx
.global .b32 from[4] = {1, 2, 3, 4};
.shared .b32 to[4];
cp.async.ca.shared.global [to], [from], 16, 12;
cp.async.commit_group;
cp.async.wait_group 0;
```

The copy size is 16 bytes but the source size is 12, so the last four bytes
are zero-filled. NVIDIA therefore writes `1, 2, 3, 0`.

The local ACuda snapshot rejects it with:

```text
compile: PTX parse error: line 10: unsupported directive '.global'
```

## NVIDIA truth

`run.bat` launches one block of one thread with the input bytes from
`nvidia_case.txt` and checks the exact output bytes:

```text
entry=cp_async block=1
input   =00
expected=01000000020000000300000000000000
actual  =01000000020000000300000000000000
PASS
```

The expected 32-bit words are `1, 2, 3, 0`.

## Run on NVIDIA

```powershell
.\run.bat
```

The shared driver-API checker is built automatically on first use. To rebuild
it explicitly:

```powershell
..\common\build_test_nvidia.bat
```
