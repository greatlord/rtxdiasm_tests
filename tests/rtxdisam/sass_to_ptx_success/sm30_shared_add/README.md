# sm30_shared_add SASS -> PTX success

This case exercises the pre-Volta shared-memory atomic emulation built from
`LDSLK`/`STSCUL`.

## PTX

`sm30_shared_add.ptx` performs a shared `atom.shared.add.u32` and then loads
the updated value:

```ptx
.shared .align 4 .u32 s[1];
st.shared.u32 [s], 0;
atom.shared.add.u32 old, [s], 1;
ld.shared.u32 current, [s];
```

Compiled with CUDA 10.2 `ptxas -arch=sm_30`, the shared atomic is lowered by
ptxas into a warp-aggregated load-linked/store-conditional loop:

```text
SSY `(.L_1)`
VOTE.ALL R0, PT, PT
FLO.U32 R6, R0
ISETP.EQ.U32.AND P0, PT, R6, R2, PT
@!P0 NOP.S (*"TARGET= .L_1 "*)
...
.L_3:
LDSLK P1, R7, [RZ]
@P1 IADD R3, R2, R7
@P1 STSCUL P0, [RZ], R3
@!P0 BRA `(.L_3)
```

## Lowering

The lifter maps the linked pair to an atomic compare-and-swap loop:

```ptx
ld.shared.u32 %r7, [__shared];
setp.eq.u32 %p1, 0, 0;
...
atom.shared.cas.b32 %rt, [__shared], %r7, %r3;
setp.eq.u32 %p0, %rt, %r7;
@!%p0 bra L__L_3;
```

`LDSLK` loads the linked value and reports a valid link. `STSCUL` becomes a
shared CAS with the linked value as the expected value; its success predicate
drives the retry loop. Guarded `NOP.S` reconvergence skips become guarded
branches. `PSETP`, `POPC`, `ISET`, and `ICMP` use 32-bit boolean intermediates
and unguarded comparisons so ACuda accepts them.

## Run

```powershell
.\run.bat
```

The test checks one thread and four threads:

```text
1 thread  -> buf0[0]=0, buf0[1]=1
4 threads -> buf0[1]=4
PASS (LL/SC shared atomic add, one and four threads)
```

The four-thread run exercises the CAS retry path and warp-aggregated shared
atomic update.
