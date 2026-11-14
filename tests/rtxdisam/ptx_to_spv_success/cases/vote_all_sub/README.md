> This case now passes on the updated ACuda backend and is part of the
> success regression kit.

# vote_all_sub PTX -> SPIR-V regression

This folder is a reproduction case for the PTX -> SPIR-V backend. It is not a
failure claim against the newer ACuda snapshot.

## PTX

`vote_all_sub.ptx` uses `vote.sync.all.pred` with a full member mask while
only lanes 1..31 execute:

```ptx
@first_lane bra EXIT;
vote.sync.all.pred result_pred, 1, 0xFFFFFFFF;
EXIT:
selp.u32 result, 1, 0, result_pred;
```

Lane 0 writes `0`; lanes 1..31 write `1`.

The local ACuda snapshot rejects the generated SPIR-V with a dominance error:

```text
error: line 94: ID '72[%72]' defined in block '29[%29]' does not dominate its use in block '30[%30]'
```

## NVIDIA truth

`run.bat` launches one block of 32 threads. The output buffer is checked for
one zero word followed by 31 one words:

```text
entry=vote_all_sub block=32
expected=0000000001000000...01000000
actual  =0000000001000000...01000000
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
