# Building

This page shows how to build Chassis from a checkout, on Linux, macOS or Windows.

## Requirements

- Rust 1.88 or later
- A C compiler (for some dependencies)
- Git

## Clone

```bash
git clone https://github.com/tanvincible/chassis.git
cd chassis
```

## Build

Build the project in release mode:

```bash
cargo build --release
```

The compiled library will be in `target/release/`.

For development builds without optimizations:

```bash
cargo build
```

## Checks

Run the linter and the formatter as CI does, and fix any warning before you open a pull request:

```bash
cargo clippy --all-features --all-targets -- -D warnings
cargo fmt --all -- --check
```

## Platforms

### Linux

No special setup required. The build uses standard glibc.

### macOS

Xcode command line tools are required for the C compiler:

```bash
xcode-select --install
```

### Windows

Install Visual Studio Build Tools or the full Visual Studio IDE. The Rust installer will detect and use the MSVC toolchain automatically.

## Cross-compiling

Chassis runs on x86-64 and ARM64. To build for ARM on an x86 machine:

```bash
rustup target add aarch64-unknown-linux-gnu
cargo build --target aarch64-unknown-linux-gnu --release
```

Test on ARM hardware too: an emulator may not catch alignment mistakes.
