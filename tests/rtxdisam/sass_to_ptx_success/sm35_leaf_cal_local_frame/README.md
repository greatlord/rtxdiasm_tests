# sm35_leaf_cal_local_frame SASS -> PTX success

This starts the CAL ABI work for callees with a local stack frame. The callee
spills `x` and `x+1` to a local array and reloads them:

```ptx
.func (.param .u32 output) f(.param .u32 x) {
    .local .align 4 .b8 tmp[8];
    ld.param.u32 r0, [x];
    st.local.u32 [tmp], r0;
    add.u32 r1, r0, 1;
    st.local.u32 [tmp+4], r1;
    ld.local.u32 r1, [tmp];
    ld.local.u32 r2, [tmp+4];
    add.u32 r0, r1, r2;
    st.param.u32 [output], r0;
    ret;
}
```

Compiled with `ptxas -O0 -arch=sm_35`, the callee uses `R1`-relative
`STL`/`LDL` spills.

## Lowering

The lifter allocates a fixed local frame in the entry body and maps the
callee's `R1`-relative local accesses to it:

```ptx
.local .align 16 .b8 __callee_frame[16];
...
st.local.u32 [__callee_frame+0], %r0;
st.local.u32 [__callee_frame+0x4], %r0;
ld.local.u32 %r0, [__callee_frame+0];
ld.local.u32 %r2, [__callee_frame+0x4];
```

The ptxas stack-limit guard (`SR_LMEMHIOFF`, `ISETP`, `BPT`) is elided; the
lifted kernel uses a fixed local frame instead of the SASS stack pointer.

## Scope

Supported: leaf callees with `R1`-relative `STL`/`LDL` 32-bit spills and no
`R1` arithmetic or local-frame recursion. General parameter-slot ABI, dynamic
`R1` address formation, and 64-bit local spills remain later work.

## Run

```powershell
.\run.bat
```

```text
x=5  -> buf0[0]=11
x=10 -> buf0[0]=21
PASS (leaf CAL local frame 5 and 10)
```
