# Pointer loaded from shared memory used as an address

A bounded dynamic `LDC.64` selects a global pointer from a parameter
table that the SASS lifter materializes in shared memory:

```ptx
ld.shared.u32 %r2, [%rdt4+0x0];
ld.shared.u32 %r3, [%rdt4+0x4];
mov.b64 %rdt5, {%r2, %r3};
ld.global.u32 %r5, [%rdt5+0];
```

Pointer parameters used as integer words now materialize a runtime
pointer token: the high 32 bits hold the descriptor binding (or a
shared/global/parameter sentinel) and the low 32 bits the byte offset.
A plain 64-bit address in a global/generic load or store is decoded at
runtime and dispatched to the matching binding with a structured
switch. The parameter classifier follows pointer halves through their
store into and reload from shared memory, so `p1..p3` remain pointer
parameters in the launch ABI.

Linux kit contract: selector `0` with three pointers to `10`, `20`,
`30` and an output buffer produces `buf3[0]=10`; selectors `1`/`2`
produce `20`/`30`.
