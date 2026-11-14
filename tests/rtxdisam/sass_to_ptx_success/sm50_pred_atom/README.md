# sm50_pred_atom SASS -> PTX success

This case exercises pre-Volta reconvergence control flow around a predicated
global atomic.

## PTX

`sm50_pred_atom.ptx` performs a conditional `atom.global.max.u32` and stores
the returned old value:

```ptx
mov.u32 tid, %tid.x;
setp.eq.u32 first, tid, 0;
@first atom.global.max.u32 old, [out_addr], value;
@first st.global.u32 [out_addr+4], old;
```

Compiled with CUDA 10.2 `ptxas -arch=sm_50`, the atomic region is guarded by
`SSY`/`SYNC` and a conditional exit:

```text
SSY `(.L_1)`
ISETP.NE.U32.AND P0, PT, R0, RZ, PT
@P0 SYNC (*"TARGET= .L_1 "*)
LDG.E R2, [R2]
ATOM.E.MAX R6, [R4], R2
SYNC (*"TARGET= .L_1 "*)
.L_1:
@P0 EXIT
STG.E [R4+0x4], R6
```

## Lowering

The lifter preserves the reconvergence target from the `TARGET=` annotation
and lowers the early-exit region as:

```ptx
setp.ne.u32 %p0, %r0, 0;
mov.u32 %r6, 0;
@%p0 bra L__L_1;
ld.global.u32 %r2, [...];
atom.global.max.u32 %r6, [...], %r2;
L__L_1:
@!%p0 st.global.u32 [...], %r6;
ret;
```

`SSY` is a validated no-op. The guarded `SYNC` becomes the skip branch, the
unconditional `SYNC` reconverges, and the guarded `EXIT` becomes a predicated
tail with a single final `ret`. The atomic destination is initialized so the
merge has a definition on both paths.

## Run

```powershell
.\run.bat
```

The test launches four threads, so three lanes exercise the skip branch:

```text
input 42, output 10 -> buf1[0]=42, buf1[1]=10
input 5,  output 10 -> buf1[0]=10, buf1[1]=10
PASS (predicated atomic max and old value)
```
