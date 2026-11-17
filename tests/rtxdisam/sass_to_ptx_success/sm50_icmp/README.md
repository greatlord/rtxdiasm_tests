# sm50_icmp SASS -> PTX success

This is a SASS-to-PTX lifter success case, not an ACuda PTX-to-SPIR-V case.

## PTX

`sm50_icmp.ptx` computes `result = value > 0 ? 2 : 1` through a branchy
select. Compiled with CUDA 10.2 `ptxas -arch=sm_50`, it becomes:

```text
MOV32I R0, 0x1
ICMP.LE R6, R0, 0x2, R2
STG.E [R4], R6
```

`ICMP.LE` is the Maxwell “compare to zero and select source” form:

```text
Rd = (Rc <= 0) ? Ra : Rb
```

The lifter lowers it to:

```ptx
setp.le.s32 %pt0, %r2, 0;
selp.b32 %r6, %r0, 0x2, %pt0;
```

## Run

```powershell
.\run.bat
```

The test builds the cubin and checks both signs:

```text
input 1  -> buf1[0]=2
input -1 -> buf1[0]=1
PASS (ICMP positive and negative)
```
