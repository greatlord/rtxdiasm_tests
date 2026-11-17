# sm50_leaf_cal_param_slots SASS -> PTX success

This exercises the SASS call ABI when arguments are too numerous for the
register budget. `ptxas -O0 -maxrregcount 16` spills three of the 17
arguments into caller-allocated stack/parameter slots before the `CAL`:

```text
STL [R1+0x4], R2
STL [R1], R3
...
CAL `($k$f)
```

The callee reloads those words with `LDL.LU`, sums all 17 arguments, and
returns the result in `R0`:

```text
LDL.LU R15, [R1+0x4]
LDL.LU R15, [R1]
...
IADD R0, R0, R15
RET
```

## Lowering

The entry-point stack pointer and the inlined callee share one per-thread
`.local` frame. Caller stores and callee loads keep their original offsets:

```ptx
st.local.u32 [__callee_frame+0x4], %r2;
st.local.u32 [__callee_frame+0x0], %r3;
...
ld.local.u32 %r15, [__callee_frame+0x4];
ld.local.u32 %r15, [__callee_frame+0x0];
```

## Run

```powershell
.\run.bat
```

```text
arguments 1..17 -> buf0[0]=153
PASS (leaf CAL parameter slots)
```
