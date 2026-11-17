# sm35_leaf_cal_local_array SASS -> PTX success

This exercises local-array addressing through `R1` arithmetic. The callee
receives a stack parameter array and sums its first two `u32` values:

```ptx
.func (.param .u32 output) f(.param .align 8 .b8 data[32]) {
    mov.u64 p, data;
    ld.global.u32 r0, [p];
    ld.global.u32 r1, [p+4];
    add.u32 r0, r0, r1;
    st.param.u32 [output], r0;
    ret;
}
```

Compiled with `ptxas -O0 -arch=sm_35`, the callee spills the array, forms a
local pointer from `R1`, and uses `LD.E` from that pointer.

## Lowering

The lifter tracks `R1` offsets through `IADD`, `I2I`, `MOV`, and the
`IADD32I.CC`/`IADD.X` pair, then rewrites computed local `LD.E`/`ST.E`
accesses to `ld.local`/`st.local` against `__callee_frame`.

## Run

```powershell
.\run.bat
```

```text
input 10, 20 -> output 30
PASS (local-array leaf CAL 10+20)
```
