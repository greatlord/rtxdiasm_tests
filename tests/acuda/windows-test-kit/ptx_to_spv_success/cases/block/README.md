> This case now passes on the updated ACuda backend and is part of the
> success regression kit.

# block PTX -> SPIR-V regression

This folder is a reproduction case for the PTX -> SPIR-V backend. It is not a
failure claim against the newer ACuda snapshot.

## PTX

`block.ptx` uses a nested register scope:

```ptx
add.u64 temp2, temp, 1;
{
    .reg .u64 temp2;
    add.u64 temp2, temp2, 1;
}
st.u64 [out_addr], temp2;
```

The inner `temp2` shadows the outer register, so the value stored is the
outer `temp2 = input + 1`. For input `41`, NVIDIA writes `42`.

The local ACuda snapshot rejects it with:

```text
compile: PTX parse error: line 20: expected directive, label or instruction
```

## NVIDIA truth

`run.bat` launches one block of one thread with the input bytes from
`nvidia_case.txt` and checks the exact output bytes:

```text
entry=block block=1
input   =2900000000000000
expected=2a00000000000000
actual  =2a00000000000000
PASS
```

The expected 64-bit word is `42`.

## Run on NVIDIA

```powershell
.\run.bat
```

The shared driver-API checker is built automatically on first use. To rebuild
it explicitly:

```powershell
..\common\build_test_nvidia.bat
```
