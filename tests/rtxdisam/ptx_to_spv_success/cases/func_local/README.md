# func_local PTX -> SPIR-V success

This case calls a `.func` that uses a callee-local array. ACuda now allocates a
fresh caller-local backing array per call and renames the callee's `.local`
accesses, so the inline path returns `input + 1`.

With input `5`, the kernel stores `6` in `buf0[0]`.
