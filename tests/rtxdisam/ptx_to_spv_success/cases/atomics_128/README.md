> This case now passes on the updated ACuda backend and is part of the
> success regression kit.

# atomics_128 PTX -> SPIR-V regression

This folder is a reproduction case for the PTX -> SPIR-V backend. It is not a
failure claim against the newer ACuda snapshot.

## PTX

`atomics_128.ptx` uses the 128-bit vector acquire/release forms:

```ptx
ld.acquire.gpu.v2.u64 {temp1, temp2}, [in_addr];
st.release.gpu.v2.u64 [out_addr], {temp1, temp2};
```

It loads two 64-bit values, adds one to each, and stores them back. For inputs
`1` and `2`, NVIDIA writes `2` and `3`.

The local ACuda snapshot rejects it with:

```text
compile: PTX parse error: line 18: unsupported instruction 'ld.acquire.gpu.v2.u64'
```

## NVIDIA truth

`run.bat` launches one block of one thread with the input bytes from
`nvidia_case.txt` and checks the exact output bytes:

```text
entry=atomics_128 block=1
input   =01000000000000000200000000000000
expected=02000000000000000300000000000000
actual  =02000000000000000300000000000000
PASS
```

The expected 64-bit words are:

```text
out[0] = 2
out[1] = 3
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
