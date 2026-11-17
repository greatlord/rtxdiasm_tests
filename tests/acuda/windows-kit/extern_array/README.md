# extern_array: repeated host calls over an array

ACuda extension case.  The kernel loops over five `u64` input elements,
calls the external `foobar` once per element through the host-call
dispatcher, and stores the returned value.  The Linux kit registers an
identity host callback, so the expected output is `1, 2, 3, 4, 5`.

This also exercises the loop-carried pointer pattern: addresses are
recomputed from carried byte offsets rather than carried pointers.
