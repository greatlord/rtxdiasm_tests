> This case now passes on the updated ACuda backend and is part of the
> success regression kit.

# param_is_addressable PTX -> SPIR-V regression

This folder is a reproduction case for the PTX -> SPIR-V backend. It is not a
failure claim against the newer ACuda snapshot.

## PTX

`param_is_addressable.ptx` moves the `input` parameter itself into a register,
then reads the parameter through that address:

```ptx
mov.b64      temp, input;
ld.param.b64 temp, [temp];
sub.u64      temp, temp, in_addr;
st.u64       [out_addr], temp;
```

On NVIDIA the parameter address is valid, so the loaded value equals
`in_addr` and the kernel writes zero.

The local ACuda snapshot rejects it with:

```text
compile: PTX lower error: line 17: bad operand kind for scalar position
```

## NVIDIA truth

`run.bat` launches one block of one thread with the input bytes from
`nvidia_case.txt` and checks the exact output bytes:

```text
entry=param_is_addressable block=1
input   =01000000000000000200000000000000
expected=0000000000000000
actual  =0000000000000000
PASS
```

The test is deterministic and must always pass on the RTX 3070.

## Run on NVIDIA

```powershell
.\run.bat
```

The shared driver-API checker is built automatically on first use. To rebuild
it explicitly:

```powershell
..\common\build_test_nvidia.bat
```
