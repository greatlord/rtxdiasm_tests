# sm35_empty_cal SASS -> PTX success

This is the first `CAL` prototype: a direct call to an empty leaf function.

## PTX

`sm35_empty_cal.ptx` calls a device function whose body is only `ret`:

```ptx
.func f(.param .u32 p) { ret; }
.visible .entry k(.param .u32 input) {
    ld.param.u32 value, [input];
    st.param.u32 [arg], value;
    call.uni f, (arg);
    ret;
}
```

Compiled with CUDA 10.2 `ptxas -O0 -arch=sm_35`, the caller and callee share
`.text.k`:

```text
MOV32I R0, 0x140
LDC    R0, c[0x0][R0]
CAL    `($k$f)
...
$k$f:
RET
```

## Lowering

The lifter recognizes that the callee body is only `RET`, so the call has no
observable effect. It omits the `CAL` and the unreachable callee body:

```ptx
mov.u32 %r0, %rpp0;
ret;
```

This is intentionally limited to empty leaf functions. A non-empty callee
still rejects cleanly until full multi-function ABI support is added.

## Run

```powershell
.\run.bat
```

```text
PASS (empty-leaf CAL)
```
