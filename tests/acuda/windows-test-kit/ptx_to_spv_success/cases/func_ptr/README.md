> This case now passes on the updated ACuda backend and is part of the
> success regression kit.

# func_ptr PTX -> SPIR-V regression

This folder is a reproduction case for the PTX -> SPIR-V backend. It is not a
failure claim against the newer ACuda snapshot.

## PTX

`func_ptr.ptx` materializes the address of a device function:

```ptx
mov.u64 f_addr, foobar;
add.u64 temp2, temp2, f_addr;
st.u64  [out_addr], temp2;
```

The kernel writes `input + 1 + address(foobar)`. The absolute address is
driver/module dependent, so the test checks the invariant instead of a fixed
address: two different inputs must produce the same offset, and the offset
must be greater than one.

The local ACuda snapshot rejects it with:

```text
compile: PTX lower error: line 27: bad operand kind for scalar position
```

## NVIDIA truth

`build_and_run.bat` launches the kernel with inputs `41` and `42`:

```text
func_ptr(41) = 0xea
func_ptr(42) = 0xeb
offset       = 0xc1
PASS
```

The offset is `1 + address(foobar)`; `0xc1` is specific to this driver and
module, so the test only requires that both runs agree and the offset is
nonzero.

## Run on NVIDIA

```powershell
.\build_and_run.bat
```
