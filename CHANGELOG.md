# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.1/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/tanvincible/chassis/compare/v0.6.3...HEAD) - 6 May 2026

## [v0.6.3](https://github.com/tanvincible/chassis/compare/v0.6.2...v0.6.3) - 1 May 2026

### Fixed

- fix: (CI) Win storage, clippy cleanups, FFI lints ([a6d09c3](https://github.com/tanvincible/chassis/commit/a6d09c3ef20e317d5f41e7afa1045b7374424d58))

## [v0.6.2](https://github.com/tanvincible/chassis/compare/v0.6.1...v0.6.2) - 1 May 2026

### Fixed

- fix: (CI) Win storage, clippy cleanups, FFI lints ([bb2aa7f](https://github.com/tanvincible/chassis/commit/bb2aa7f8f148ce6c0dfba19f442218fb146b9b10))
- fix: Windows tests, clippy, rustdoc, and graph relocation on add ([932e8f7](https://github.com/tanvincible/chassis/commit/932e8f70225449fef632f4b30e6f87df39586954))

## [v0.6.1](https://github.com/tanvincible/chassis/compare/v0.6.0...v0.6.1) - 1 May 2026

### Fixed

- fix: Windows tests, clippy, rustdoc, and graph relocation on add ([a90c413](https://github.com/tanvincible/chassis/commit/a90c4130d10c3ef872f55ff54c35f975f9365217))

## [v0.6.0](https://github.com/tanvincible/chassis/compare/v0.5.0...v0.6.0) - 1 May 2026

### Merged
- bug: fix high space usage of .chassis files (≈1GB for ~10k vectors) [`#10`](https://github.com/tanvincible/chassis/pull/10)

### Added

- feat: add chassis_add_batch for row-major vector batches ([f799999](https://github.com/tanvincible/chassis/commit/f799999f2d6a24544c858e2289bfeb2e6b9b6c1e))

### Fixed

- fix: use real search API in pychassis/examples/batch_insert.py ([2595122](https://github.com/tanvincible/chassis/commit/25951229c5e1396926f1108a1c68ab1074e533d6))

## [v0.5.0](https://github.com/tanvincible/chassis/compare/v0.5.0-alpha...v0.5.0) - 27 January 2026

### Added

- feat: implement full python bindings and release v0.5.0 ([2796ad0](https://github.com/tanvincible/chassis/commit/2796ad0028d5e0ad4a48cf8ce114a5bc18239e4b))

## [v0.5.0-alpha](https://github.com/tanvincible/chassis/compare/v0.4.1-alpha...v0.5.0-alpha) - 27 January 2026

### Added

- feat: implement C-compatible FFI layer (v0.5.0-alpha) ([7e084e4](https://github.com/tanvincible/chassis/commit/7e084e4b18d9d18cbbb93e0a72d63bba20042ff8))

## [v0.4.1-alpha](https://github.com/tanvincible/chassis/compare/v0.4.0-alpha...v0.4.1-alpha) - 26 January 2026

## [v0.4.0-alpha](https://github.com/tanvincible/chassis/compare/v0.3.1-alpha...v0.4.0-alpha) - 26 January 2026

### Added

- feat: implement VectorIndex facade and crash-safe orchestration ([e1ad599](https://github.com/tanvincible/chassis/commit/e1ad599951212256f9ce1f713f31404ea229df8d))

## [v0.3.1-alpha](https://github.com/tanvincible/chassis/compare/v0.3.0-alpha...v0.3.1-alpha) - 24 January 2026

## [v0.3.0-alpha](https://github.com/tanvincible/chassis/compare/v0.2.0-alpha...v0.3.0-alpha) - 24 January 2026

### Added

- feat: implement SIMD acceleration & harden search ([e65348b](https://github.com/tanvincible/chassis/commit/e65348b39c0645b85aab366258dc7a0fe17e3f2b))

## [v0.2.0-alpha](https://github.com/tanvincible/chassis/compare/v0.1.0-alpha.1...v0.2.0-alpha) - 24 January 2026

### Merged
- Implement persistent graph header and direct mmap I/O for HNSW nodes [`#7`](https://github.com/tanvincible/chassis/pull/7)
- feat: fixed-width HNSW node records with O(1) [`#5`](https://github.com/tanvincible/chassis/pull/5)
- feat: Add zero-copy vector slice access via get_vector_slice() [`#4`](https://github.com/tanvincible/chassis/pull/4)

### Added

- feat: implement Graph I/O with persistent header and zero-allocation iteration ([fe46503](https://github.com/tanvincible/chassis/commit/fe46503cb6e6409c8e7a7add17150ac87ea123cc))
- feat: add bidirectional hnsw linking with diversity pruning ([d9962e3](https://github.com/tanvincible/chassis/commit/d9962e3c4c9119d549f2d384bff11e22fb4cca12))
- feat: add GraphHeader and graph I/O methods to HnswGraph and Storage ([fc5b465](https://github.com/tanvincible/chassis/commit/fc5b4655972b39981b696141cfce685d5cf308ee))

### Infrastructure

- build: implement precise changelog template and workflow ([194945a](https://github.com/tanvincible/chassis/commit/194945a38ba90646aa4e6015684cdb065aa8684c))
- build: harden quality standards with Rust 2024 and strict lints ([1f68b58](https://github.com/tanvincible/chassis/commit/1f68b58110a83fb8d3eb9357862e4a594bb01ff4))
- build: initialize chassis-ffi crate and cbindgen configuration ([dc24cdb](https://github.com/tanvincible/chassis/commit/dc24cdb4a87b08d3bd34c25abeef9132afe7f325))

### Performance

- perf: document official storage baseline and benchmark report ([3d37816](https://github.com/tanvincible/chassis/commit/3d37816a53959262ac7f042f7efb9828f25bb1b2))

## v0.1.0-alpha.1 - 18 January 2026

### Infrastructure

- build: implement precise changelog template and workflow ([a071044](https://github.com/tanvincible/chassis/commit/a0710441effdcf28cad5648e1a4ed64c3e7d197a))
- build: initialize project workspace and foundational structure ([cc08f6d](https://github.com/tanvincible/chassis/commit/cc08f6d86c75ea5fa882ebd07cb8b7610cb3cf41))
- build: configure git environment and initialize documentation ([d7bc0e5](https://github.com/tanvincible/chassis/commit/d7bc0e55c1f218c65ef0b4267acfcc269f8a1c94))
- build: harden quality standards with Rust 2024 and strict lints ([ac6bd96](https://github.com/tanvincible/chassis/commit/ac6bd9694b220f0fcdefa712285525fa7284ab0f))
- build: initialize chassis-ffi crate and cbindgen configuration ([bfae05b](https://github.com/tanvincible/chassis/commit/bfae05b0d35ea59afaf2190ca4bdd6acf17fa68a))
- build: lock development environment with rust-toolchain.toml ([277f6fd](https://github.com/tanvincible/chassis/commit/277f6fd3ea2f5ae7d9c0cc32f325ae26bb926918))
