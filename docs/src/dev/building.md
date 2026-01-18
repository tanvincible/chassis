# Building from Source

## Requirements

- Rust 1.85 or later
- A C compiler (for some dependencies)
- Git

## Clone the Repository

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

## Check for Issues

Run the linter and formatter:

```bash
cargo clippy
cargo fmt --check
```

Fix any warnings before submitting a pull request.

## Platform-Specific Notes

### Linux

No special setup required. The build uses standard glibc.

### macOS

Xcode command line tools are required for the C compiler:

```bash
xcode-select --install
```

### Windows

Install Visual Studio Build Tools or the full Visual Studio IDE. The Rust installer will detect and use the MSVC toolchain automatically.

## Cross-Compilation

Chassis is designed to work on both x86 and ARM. To cross-compile for ARM on an x86 host:

```bash
rustup target add aarch64-unknown-linux-gnu
cargo build --target aarch64-unknown-linux-gnu --release
```

Test on actual ARM hardware, as emulation may not catch alignment issues.
