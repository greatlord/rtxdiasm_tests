# sm50_leaf_cal_ret_slots SASS -> PTX success

This exercises parameter-slot **returns**. The callee returns a 20-word
structure, which exceeds the register budget under
`ptxas -O0 -maxrregcount 16`. The caller provides a return slot in its stack
frame, the callee stores the words there, and the caller reloads them after
`CAL`:

```text
CAL `($k$f)
...
STL [R1], R0
STL [R1+0x4], R0
...
LDL.LU R15, [R1]
LDL.LU R15, [R1+0x4]
...
```

The lifted PTX maps the caller frame and callee stores to the same per-thread
`.local` frame:

```ptx
st.local.u32 [__callee_frame+0x0], %r0;
st.local.u32 [__callee_frame+0x4], %r0;
...
ld.local.u32 %r15, [__callee_frame+0x0];
ld.local.u32 %r15, [__callee_frame+0x4];
```

## Run

```powershell
.\run.bat
```

With input `x=1`, the callee returns `2..21`; their sum is `230`.

```text
buf0[0]=230
PASS (leaf CAL return slots)
```
