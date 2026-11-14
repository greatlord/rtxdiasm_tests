# fma_oob PTX -> SPIR-V success

This case exercises sm_90 `fma.rn.oob.f16` with ordinary (non-OOB) operands.
ACuda accepts the instruction and computes `1.0 * 1.0 + 1.0 = 2.0`, stored as
half `0x4000` (`16384`) in `buf0[0]`.

The case moved here from `tests/ptx_to_spv_error` after the tensor OOB-NaN
path stopped rejecting ordinary operands.
