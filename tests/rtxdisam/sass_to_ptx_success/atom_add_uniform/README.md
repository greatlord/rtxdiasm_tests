# atom_add_uniform SASS -> PTX success

This case exercises interleaved `XMAD.MRG` chains in a warp-aggregated
shared-memory atomic add.

## PTX

`atom_add_uniform.ptx` performs a uniform shared-memory atomic add with a
runtime addend:

```ptx
st.shared.u32      [smem], value;
atom.shared.add.u32 old, [smem], addend;
ld.shared.u32      result, [smem];
```

Compiled with CUDA 10.2 `ptxas -arch=sm_50`, it produces two interleaved
multiply chains:

```text
XMAD          R7, R4, R5, RZ
STS           [RZ], R0
XMAD.MRG      R5, R4, R5.H1, RZ
XMAD.MRG      R11, R4, R9.H1, RZ
XMAD.PSL.CBCC R7, R4.H1, R5.H1, R7
@P0 ATOMS.ADD R7, [RZ], R7
...
XMAD          R6, R4, R9, R2
XMAD.PSL.CBCC R4, R4.H1, R11.H1, R6
```

The lifter now recognizes an `XMAD.MRG` that appears before its later
`XMAD`, allowing the second chain's merge operation to remain interleaved
with the first chain.

## Run

```powershell
.\run.bat
```

Expected result for value `41` and addend `10`:

```text
PASS (interleaved XMAD.MRG shared atomic add)
buf1[0]=51
buf1[1]=41
```
