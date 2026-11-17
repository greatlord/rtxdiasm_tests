# fma.rn.oob.f16

The tensor out-of-bounds modifier on a half-precision FMA:

```ptx
fma.rn.oob.f16 d, a, b, c;
```

`.oob` requires sm_90+ and PTX ISA 8.1. With ordinary operands it computes
`1.0 * 1.0 + 1.0 = 2.0`.

The tensor OOB-NaN payload is not documented, so ACuda accepts `.oob` and
lowers it as an ordinary fused multiply-add. That is exact for every
non-OOB input; the special OOB-NaN clamp is not modelled.

Linux kit contract: 4-byte output buffer expecting `buf0[0]=16384` (half
`2.0`).
