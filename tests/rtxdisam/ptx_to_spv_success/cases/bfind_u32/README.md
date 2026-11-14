> This case now passes on the updated ACuda backend and is part of the
> success regression kit.

# bfind_u32 PTX -> SPIR-V regression

This folder is a reproduction case for the PTX -> SPIR-V backend. It is not a
failure claim against the newer ACuda snapshot.

## PTX

`bfind_u32.ptx` finds the highest set bit:

```ptx
ld.global.u32 value, [in_addr];
bfind.u32 result, value;
st.global.u32 [out_addr], result;
```

For input `41` (`0b101001`) the highest set bit is bit `5`, so NVIDIA writes
`5`. For input zero, `bfind.u32` returns `0xffffffff`.

The local ACuda snapshot rejects it with:

```text
compile: PTX parse error: line 19: unsupported instruction 'bfind.u32'
```

The SASS-to-PTX lifter lowers `FLO.U32` to `clz.b32` plus a subtraction so
the lifted path does not depend on `bfind`.

## NVIDIA truth

`run.bat` launches one block of one thread with the input bytes from
`nvidia_case.txt` and checks the exact output bytes:

```text
entry=bfind_u32 block=1
input   =29000000
expected=05000000
actual  =05000000
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
