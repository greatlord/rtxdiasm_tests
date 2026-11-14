> This case now passes on the updated ACuda backend and is part of the
> success regression kit.

# cvt_pack PTX -> SPIR-V regression

This folder is a reproduction case for the PTX -> SPIR-V backend. It is not a
failure claim against the newer ACuda snapshot.

## PTX

`cvt_pack.ptx` exercises saturating conversion and packing:

```ptx
cvt.pack.sat.u8.s32.b32 result1, src1, src2, src3;
cvt.pack.sat.s8.s32.b32 result2, src1, src2, src3;
```

The inputs in `nvidia_case.txt` are `src1 = 127`, `src2 = 128`, and `src3 = 256`.
The unsigned pack keeps `128`, while the signed pack saturates it to `127`.

The local ACuda snapshot rejects it with:

```text
compile: PTX parse error: line 25: bad opcode 'cvt.pack.sat.u8.s32.b32'
```

## NVIDIA truth

`run.bat` launches one block of one thread with the input bytes from
`nvidia_case.txt` and checks the exact output bytes:

```text
entry=cvt_pack block=1
input   =7f0000008000000000010000
expected=807f00017f7f0001
actual  =807f00017f7f0001
PASS
```

The expected 32-bit words are:

```text
out[0] = 0x01007f80   (unsigned)
out[1] = 0x01007f7f   (signed)
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
