> This case now passes on the updated ACuda backend and is part of the
> success regression kit.

# uint_to_fp_bf16 PTX -> SPIR-V regression

This folder is a reproduction case for the PTX -> SPIR-V backend. It is not a
failure claim against the newer ACuda snapshot.

## PTX

`uint_to_fp_bf16.ptx` dispatches to a bfloat16 conversion path:

```ptx
selp.u32            check_u32, 1, 0, check;
cvt.rn.f32.u32      check_f32, check_u32;
cvt.rn.bf16.f32     result, check_f32;
st.global.u16       [out_addr], result;
```

With `value = 1` and `dispatch = 3`, NVIDIA writes the bfloat16 encoding of
`1.0`, which is `0x3f80`.

The local ACuda snapshot rejects it with:

```text
compile: PTX lower error: at most one conditional forward branch per kernel
```

## NVIDIA truth

`run.bat` launches one block of one thread with the input bytes from
`nvidia_case.txt` and checks the exact output bytes:

```text
entry=uint_to_fp_bf16 block=1
input   =0100000003000000
expected=803f
actual  =803f
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
