# chassis-ffi

The C API for Chassis: a shared library, `libchassis_ffi`, and its header, `include/chassis.h`.

```bash
cargo build --release -p chassis-ffi
```

This builds `target/release/libchassis_ffi.so` on Linux, `libchassis_ffi.dylib` on macOS and
`chassis_ffi.dll` on Windows, and regenerates the header.

How to compile against it, every function and its errors: the
[C API reference](../docs/src/guide/c-api.md). A complete program: [examples/example.c](examples/example.c).
