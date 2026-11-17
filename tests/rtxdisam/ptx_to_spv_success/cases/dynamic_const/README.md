# dynamic_const PTX -> SPIR-V success

This case checks ACuda's register-indexed `ld.const` path. The PTX takes the
address of a module-scope `.const` array, adds a runtime byte offset, and loads
the selected lane:

```ptx
.const .u32 table[4] = { 11, 22, 33, 44 };
mov.u64 addr, table;
add.u64 addr, addr, 4;
ld.const.u32 value, [addr];
```

`table[1]` is `22`, which the kernel stores through `out`.
