# 4-entry BRX branch chain with a common merge

This is the lifted four-entry compare/branch jump table with a common
merge and store:

```ptx
setp.eq.u32 %pt0, %r0, 0;
@%pt0 bra L_jt_0;
...
bra L_jt_0;
L_jt_0: mov.u32 %r4, 0xa; bra L__L_1;
...
L__L_1: st.global.u32 [%rpptr1], %r4;
```

The final false arm branches back to the first case, which previously
made an inner selection branch directly to an outer selection's merge
block. That is invalid structured SPIR-V. The flow emitter now
duplicates the target arm on that edge instead, so the selection exits
through its own merge.

Correctness target: selector `0 -> 10`, `1 -> 11`, `2 -> 12`,
`3 -> 13`, and any other selector falls back to `10`.

Linux kit contract: `case.txt` launches one thread with scalar `p0`
and a 4-byte output buffer, expecting `buf0[0]=10`.
