> This case now passes on the updated ACuda backend and is part of the
> success regression kit.

# const_ident PTX -> SPIR-V regression

This folder is a reproduction case for the PTX -> SPIR-V backend. It is not a
failure claim against the newer ACuda snapshot.

## PTX

`const_ident.ptx` uses 64-bit `.const` initializers whose entries refer to
other constants:

```ptx
.const .u64 x = 1;
.const .u64 y[4] = {4,5,6};
.const .u64 constparams[2] = { x, y };
```

The kernel compares the addresses of `x` and `y` with the corresponding
`constparams` entries. Both comparisons cancel to zero.

The local ACuda snapshot rejects it with:

```text
compile: PTX parse error: line 5: unsupported const type '.u64'
```

## NVIDIA truth

`run.bat` launches one block of one thread with the input bytes from
`nvidia_case.txt` and checks the exact output bytes:

```text
entry=const_ident block=1
input   =01000000000000000200000000000000
expected=00000000000000000000000000000000
actual  =00000000000000000000000000000000
PASS
```

The expected 64-bit words are both zero.

## Run on NVIDIA

```powershell
.\run.bat
```

The shared driver-API checker is built automatically on first use. To rebuild
it explicitly:

```powershell
..\common\build_test_nvidia.bat
```
