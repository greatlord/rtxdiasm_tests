> This case now passes on the updated ACuda backend and is part of the
> success regression kit.

# bar_red_and_pred PTX -> SPIR-V regression

This folder is a reproduction case for the PTX -> SPIR-V backend. It is not a
failure claim against the newer ACuda snapshot.

## PTX

`bar_red_and_pred.ptx` uses barrier reductions over a predicate:

```ptx
bar.red.and.pred pred, 1, cond;
bar.red.or.pred  pred, 1, cond;
```

With two threads in the block, the first reduction is false for the odd
thread and true for the even thread; the `or` reduction is true for both.
After the remaining checks the kernel writes `2` for both threads.

The local ACuda snapshot rejects it with:

```text
compile: PTX parse error: line 29: unsupported instruction 'bar.red.and.pred'
```

## NVIDIA truth

`run.bat` launches one block of two threads with the input bytes from
`nvidia_case.txt` and checks the exact output bytes:

```text
entry=bar_red_and_pred block=2
input   =00
expected=0200000002000000
actual  =0200000002000000
PASS
```

The expected 32-bit words are:

```text
out[0] = 2
out[1] = 2
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
