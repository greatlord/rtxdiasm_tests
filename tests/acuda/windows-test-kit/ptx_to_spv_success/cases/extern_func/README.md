# External function host callback

A valid PTX 6.5 external-call shape. The kernel calls an external
function returning a 16-byte value and reads the first word back:

```ptx
.extern .func (.param .align 16 .b8 func_retval0[16]) foobar
(
    .param .b64 extern_func
);
...
call.uni (retval0), foobar, (param0);
```

The parameter slot array compiles, and the Linux kit registers the same
identity host-call stub used by the rest of the success corpus. The
input value `4660` is returned unchanged.

Linux kit contract: 8-byte input `3412000000000000` and an 8-byte
output buffer expecting `buf1[0]=4660`.
