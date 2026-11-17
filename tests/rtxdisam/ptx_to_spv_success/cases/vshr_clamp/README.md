> This case now passes on the updated ACuda backend and is part of the
> success regression kit.

# vshr_clamp PTX -> SPIR-V regression

This folder is a reproduction case for the PTX -> SPIR-V backend. It is not a
failure claim against the newer ACuda snapshot.

## PTX

`vshr_clamp.ptx` uses the funnel-shift form with clamping and addition:

```ptx
vshr.u32.u32.u32.clamp.add temp0, temp1, temp2, temp3;
vshr.u32.u32.u32.clamp.add temp0, temp1, 32, temp3;
```

For `temp1 = 0x12345678`, `temp2 = 4`, and `temp3 = 0x11111111`, NVIDIA
computes `(temp1 >> 4) + temp3 = 0x12345678`; the second form clamps the
shift to 32 and returns `0x11111111`.

The local ACuda snapshot rejects it with:

```text
compile: PTX parse error: line 20: bad opcode 'vshr.u32.u32.u32.clamp.add'
```

## NVIDIA truth

`run.bat` launches one block of one thread with the input bytes from
`nvidia_case.txt` and checks the exact output bytes:

```text
entry=vshr_clamp block=1
input   =785634120400000011111111
expected=7856341211111111
actual  =7856341211111111
PASS
```

The expected 32-bit words are:

```text
out[0] = 0x12345678
out[1] = 0x11111111
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
