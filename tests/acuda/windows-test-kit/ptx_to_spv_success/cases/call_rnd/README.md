> This case now passes on the updated ACuda backend and is part of the
> success regression kit.

# call_rnd PTX -> SPIR-V regression

This folder is a reproduction case for the PTX -> SPIR-V backend. It is not a
failure claim against the newer ACuda snapshot.

## PTX

`call_rnd.ptx` calls two device functions that use directed floating-point
rounding:

```ptx
add.rp.f32 x_value, x_value, y_value;
add.rm.f32 x_value, x_value, y_value;
```

The inputs are `1.0` and `2^-24`. Round-up produces the next representable
value, while round-down stays at `1.0`.

The local ACuda snapshot rejects it with:

```text
compile: PTX parse error: line 56: unsupported instruction 'add.rm.f32'
```

## NVIDIA truth

`run.bat` launches one block of one thread with the input bytes from
`nvidia_case.txt` and checks the exact output bytes:

```text
entry=call_rnd block=1
input   =0000803f000080330000803f00008033
expected=0100803f0000803f
actual  =0100803f0000803f
PASS
```

The expected 32-bit words are:

```text
out[0] = 0x3f800001   (add.rp)
out[1] = 0x3f800000   (add.rm)
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
