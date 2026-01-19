# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.1/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/tanvincible/chassis/compare/v0.1.0-alpha.1...HEAD) - 19 January 2026

### Merged
- feat: fixed-width HNSW node records with O(1) [`#5`](https://github.com/tanvincible/chassis/pull/5)
- feat: Add zero-copy vector slice access via get_vector_slice() [`#4`](https://github.com/tanvincible/chassis/pull/4)

### Added

- feat: implement Graph I/O with persistent header and zero-allocation iteration ([fe46503](https://github.com/tanvincible/chassis/commit/fe46503cb6e6409c8e7a7add17150ac87ea123cc))
- feat: add GraphHeader and graph I/O methods to HnswGraph and Storage ([fc5b465](https://github.com/tanvincible/chassis/commit/fc5b4655972b39981b696141cfce685d5cf308ee))

### Documentation

- docs: add Contributor Covenant Code of Conduct ([486f7ae](https://github.com/tanvincible/chassis/commit/486f7ae4ca9991eae33b012958054ddc38b0d129))
- docs: add project contributing guidelines ([39f89b1](https://github.com/tanvincible/chassis/commit/39f89b1f70c19f3eb608f4802873dad1abb62473))

### Infrastructure

- chore: release v0.1.0-alpha.1 ([8c90ddf](https://github.com/tanvincible/chassis/commit/8c90ddf7338a24aaf65f82256d7165b7b6a7c643))
- chore: release v0.1.0-alpha.1 ([b75d694](https://github.com/tanvincible/chassis/commit/b75d69454056ce76aef695e33064a85061e3a8b4))
- build: implement precise changelog template and workflow ([194945a](https://github.com/tanvincible/chassis/commit/194945a38ba90646aa4e6015684cdb065aa8684c))
- build: harden quality standards with Rust 2024 and strict lints ([1f68b58](https://github.com/tanvincible/chassis/commit/1f68b58110a83fb8d3eb9357862e4a594bb01ff4))
- build: initialize chassis-ffi crate and cbindgen configuration ([dc24cdb](https://github.com/tanvincible/chassis/commit/dc24cdb4a87b08d3bd34c25abeef9132afe7f325))

### Performance

- perf: document official storage baseline and benchmark report ([3d37816](https://github.com/tanvincible/chassis/commit/3d37816a53959262ac7f042f7efb9828f25bb1b2))

## v0.1.0-alpha.1 - 18 January 2026

### Documentation

- docs: add project contributing guidelines ([e43dc9b](https://github.com/tanvincible/chassis/commit/e43dc9b86f01f4fa47a8f34fbb6cd0e5b379e3ea))
- docs: establish project identity and licensing ([f3cfed7](https://github.com/tanvincible/chassis/commit/f3cfed732d21aaf39123a85fec083bf7073ad2de))
- docs: revise manifesto and mdbook structure ([5b7f2e1](https://github.com/tanvincible/chassis/commit/5b7f2e1042a719e9ca92f2e0359850c1793edde3))

### Infrastructure

- chore: release v0.1.0-alpha.1 ([4b616fd](https://github.com/tanvincible/chassis/commit/4b616fdc0873e982ab03e4f04092460b69241d0f))
- build: implement precise changelog template and workflow ([a071044](https://github.com/tanvincible/chassis/commit/a0710441effdcf28cad5648e1a4ed64c3e7d197a))
- build: initialize project workspace and foundational structure ([cc08f6d](https://github.com/tanvincible/chassis/commit/cc08f6d86c75ea5fa882ebd07cb8b7610cb3cf41))
- build: configure git environment and initialize documentation ([d7bc0e5](https://github.com/tanvincible/chassis/commit/d7bc0e55c1f218c65ef0b4267acfcc269f8a1c94))
- build: harden quality standards with Rust 2024 and strict lints ([ac6bd96](https://github.com/tanvincible/chassis/commit/ac6bd9694b220f0fcdefa712285525fa7284ab0f))
- build: initialize chassis-ffi crate and cbindgen configuration ([bfae05b](https://github.com/tanvincible/chassis/commit/bfae05b0d35ea59afaf2190ca4bdd6acf17fa68a))
- build: lock development environment with rust-toolchain.toml ([277f6fd](https://github.com/tanvincible/chassis/commit/277f6fd3ea2f5ae7d9c0cc32f325ae26bb926918))
