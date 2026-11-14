# sm50_leaf_cal_structured SASS -> PTX success

This exercises structured reconvergence inside a leaf callee. The callee uses
a predicated global atomic and stores its old value to a second buffer:

```ptx
mov.u32 old, 0;
setp.eq.u32 first, tid, 0;
@first atom.global.add.u32 old, [p], v;
ld.global.u32 now, [p];
st.global.u32 [o], old;
st.param.u32 [output], now;
ret;
```

The `ptxas -O0 -arch=sm_50` callee contains:

```text
SSY .L_1
@!P0 SYNC .L_1
...
ATOM.E.ADD ...
SYNC .L_1
.L_1:
```

## Lowering

The lifter if-converts the guarded region: the atomic executes under the
complement of the sync guard, and the redundant unconditional `SYNC` branch is
dropped. This avoids both the branch-to-merge and predicated-load limitations
of the current ACuda backend.

## Run

```powershell
.\run.bat
```

```text
atomic add 5 -> out[0]=5, old[0]=0
PASS (structured-reconvergence leaf CAL)
```
