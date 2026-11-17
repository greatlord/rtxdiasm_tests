# func_ret3 PTX -> SPIR-V success

This exercises ACuda's mixed multi-return ABI: a `.reg` return list followed
by a `.param` argument list.

```ptx
.func (.reg .u32 %a, .reg .u32 %b, .reg .u32 %c) f(.param .u32 x)
...
call.uni (%r0, %r1, %r2), f, (slot);
```

For `x = 5`, the three results are `5`, `6`, and `7`.
