# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.1/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/tanvincible/chassis/compare/v0.10.0...HEAD)

### Fixed

- `backup()` leaves what the index never wrote as holes: a backup takes the disk the index does,
  not its whole length (#124).

## [v0.10.0](https://github.com/tanvincible/chassis/compare/v0.9.1...v0.10.0) - 19 October 2026

Copy an index while it is in use, and search long vectors on huge pages after `warm`. Rust code
that sets `huge_pages` now passes `Some(true)` or `Some(false)`.
The file format doesn't change: v0.9.1 opens what v0.10.0 writes.

### Breaking

- Rust: `IndexOptions::huge_pages` is an `Option<bool>`, `None` by default (#121).

### Added

- `backup(path)`: a copy of the index, from its writer or, while it writes, from a reader (#116).
- `keep_small_pages()` and `chassis_keep_small_pages`: no huge pages, not even by `warm` (#121).

### Changed

- `warm()` asks for huge pages under vectors of 2 KB or more, on Linux, unless told not to (#121,
  #122).
- Python: `huge_pages` is `None` by default, leaving it to `warm` (#121).

### Performance

- Python `search` spends half as long outside the library, and `get` 30% less (#109).

## [v0.9.1](https://github.com/tanvincible/chassis/compare/v0.9.0...v0.9.1) - 17 October 2026

A write that fails partway leaves nothing behind for the next one to commit.
The file format doesn't change.

### Added

- `rollback()`: discards every add, delete and `user_version` since the last flush, and keeps the
  lock (#107).

### Fixed

- Database guide, skill and `langchain-chassisdb`: a write that failed before its flush left its
  changes in the index for the next flush, which could hide a row for good; they now roll back
  (#107).
- `langchain-chassisdb`: after such a failed add, every later add failed with `IdInUseError` (#107).
- Database guide, `langchain-chassisdb` and `chassisdb-mcp`: a `COMMIT` that failed and stayed open
  stopped every later write (#107).
- `chassisdb-mcp` and `langchain-chassisdb` 0.1.2; `langchain-chassisdb` needs `chassisdb` 0.9.1,
  for `rollback()`.

## [v0.9.0](https://github.com/tanvincible/chassis/compare/v0.8.0...v0.9.0) - 17 October 2026

Keep an index in step with a database, and check it.
The file format doesn't change: v0.8.0 opens what v0.9.0 writes.

### Breaking

- Rust: `Storage`, `HnswGraph` and other internals need the `internals` feature (#88).
- Rust: `SearchResult`s at the same distance order by id, and are equal only if ids are (#88).

### Added

- `user_version`: a number of yours that each flush commits with the data (#89).
- `check()`: lists vectors no search can reach, and vectors damaged into NaN (#102).
- Formats 3 and 4 are frozen: every later release reads them (#89).

### Performance

- A filtered search reads ids in ascending order in place, without copying them (#87).
- The database guide, the skill and `chassisdb-mcp` no longer pass every id to each search (#87).

### Fixed

- A file cut short, as a partial copy leaves it, is an error, not a SIGBUS (#90).
- Many exact copies of a vector no longer leave other vectors out of reach (#91).
- Database guide and skill: edits take a new vector id, and each write mends first, so a crash or
  a failed commit can't leave rows and vectors out of step for good (#99).
- `langchain-chassisdb`: a crash or a failed commit could hide a document from search for good;
  the store now mends itself (#100).
- `chassisdb-mcp`: a crash or a failed commit in `forget` could hide a memory until `reembed`; the
  server now mends itself, and each write flushes once, not twice (#101).
- `chassisdb-mcp` and `langchain-chassisdb` 0.1.1 need `chassisdb` 0.9, for `user_version`.

## [v0.8.0](https://github.com/tanvincible/chassis/compare/v0.7.2...v0.8.0) - 14 October 2026

One name everywhere: `chassisdb` is the crate, the Python package and its import. Look up vectors
and ids, and leave ids out of a search, in Rust, C and Python. Two new packages: `chassisdb-mcp`,
long-term memory for agents as an MCP server, and `langchain-chassisdb`, a LangChain vector
store. The file format doesn't change.

### Breaking

- The Rust crate is `chassisdb`, not `chassis-core`: in Cargo.toml, `chassisdb = "0.8"`, and in
  code, `use chassisdb::...` (#62).
- Python imports `chassisdb`, as it is installed. `import chassis` no longer works: PyPI's
  `chassis` is another project (#62).
- In Rust, a search with `k` = 0 returns an `InvalidArgument` error instead of an empty list, as
  C and Python already did (#59).
- In Python, a `with` block flushes when it ends without an exception, so its adds and deletes
  are kept (#59). `close()`, and dropping an index in Rust or C, still discard what wasn't
  flushed, as a crash would; the docs now say so.
- In Python, a read-only index's `options` report the file's metric and precision, and options
  that name another precision raise `OptionsMismatchError`, as another metric already did (#64).

### Added

- Look up a vector or an id, on the writer and on readers: `get(id)`, `contains(id)`, `ids()` and
  `deleted_count()` in Rust; `chassis_get`, `chassis_contains`, `chassis_ids` and
  `chassis_deleted_count` in C; `get(id)`, `id in index`, `ids()` and `deleted_count()` in
  Python (#61).
- Leave ids out of a search: `search(..., excluded=...)` in Python and `chassis_search_excluding`
  in C, so "more like this" can skip the vector it starts from (#61).
- The Python package installs on Python 3.11 and numpy 1.26 or later, instead of 3.13 and
  numpy 2.4 (#62).
- `chassisdb-mcp`: long-term memory for agents, as an MCP server and a command line, with the
  embedding model you pick (#65).
- `langchain-chassisdb`: a LangChain vector store on Chassis (#68).
- For coding agents: a Claude Code plugin with a skill, Cursor rules, `llms.txt`,
  `llms-full.txt` and a one-page API reference (#64).
- C names the metrics and precisions: `CHASSIS_METRIC_EUCLIDEAN`, `CHASSIS_METRIC_COSINE`,
  `CHASSIS_PRECISION_FULL`, `CHASSIS_PRECISION_HALF` (#59).

### Fixed

- Opening an index with another `max_connections` is an `OptionsMismatch` error that names both
  values, instead of an `Other` error that printed internal structs (#59).
- The error for a file with no valid header names the file (#64).
- On Windows, errors name the index's path as it was given, not its `\\?\` form (#64).
- On macOS, the C library's install name is `@rpath/libchassis_ffi.dylib`, so a program linked
  with `-rpath` finds it (#64).
- In Python, id lists that mix ids of 2\*\*63 and more with smaller ones are accepted; one id
  given as `excluded` or `allowed`, or `None` as a query, says what to pass instead (#64).
- The Python wheels carry the license files (#64).
- No compiler warnings when Chassis is built as a dependency (#59).

## [v0.7.2](https://github.com/tanvincible/chassis/compare/v0.7.1...v0.7.2) - 11 October 2026

`search_batch`: many queries in one call, in the C API and in Python. From Python, a loop of
`search` calls cost 15 to 38 µs a query on servers; one call doesn't. Nothing changes in
chassis-core.

### Added

- feat: search_batch, many queries in one call, in C and Python (#55) ([2b3ccb2](https://github.com/tanvincible/chassis/commit/2b3ccb2f0ca9442fcbe6863d1ee9a8b93b21dfa2))

## [v0.7.1](https://github.com/tanvincible/chassis/compare/v0.7.0...v0.7.1) - 10 October 2026

A README on chassis-core's crates.io page. Nothing changes in the library or the Python package.

### Infrastructure

- chore: the Python package's folder named chassisdb, as the package is (#51) ([e3a7d2d](https://github.com/tanvincible/chassis/commit/e3a7d2d2f45d2aeccb0c516ef2559e7a9806b5c5))
- ci: publish to crates.io through trusted publishing, or a token while one is set

## [v0.7.0](https://github.com/tanvincible/chassis/compare/v0.6.3...v0.7.0) - 10 October 2026

A new file format, with readers in other processes. Opening a file from v0.6.3 or earlier
for writing converts it, and v0.6.3 can't open the new files. The Python package is
`chassisdb`, imported as `chassis`.

### Breaking

- feat!: file format v3 and multi-process readers ([7c04de0](https://github.com/tanvincible/chassis/commit/7c04de072871586b72628ff6d4688a88150e0ee9))
- feat!: cosine distance (#14) ([4dfa6cf](https://github.com/tanvincible/chassis/commit/4dfa6cfa200f7f58a305078324e653a030f532e9))
- feat!: errors that say what was wrong and what to do (#41) ([6b5cbcc](https://github.com/tanvincible/chassis/commit/6b5cbcc8bbf4c375db0884fb3daff26fa5b01e88))

### Added

- feat: ids, deletes and crash-safe durability ([bdc9e34](https://github.com/tanvincible/chassis/commit/bdc9e34c297433ae4ca4220f1bbdf14d5e680433))
- feat: filtered search (#15) ([a5f3158](https://github.com/tanvincible/chassis/commit/a5f31588a75539ab48c32c4e0989408fb9956204))
- feat: parallel batch builds (#17) ([b64d4ad](https://github.com/tanvincible/chassis/commit/b64d4ad3c56c8344910c3fab050b9ec2c159e8e8))
- feat: compaction (#21) ([759a97d](https://github.com/tanvincible/chassis/commit/759a97dcf9337480217b9e330e6e2d39d3467d4b))
- feat: keep the vectors on huge pages, on request (#31) ([5f80578](https://github.com/tanvincible/chassis/commit/5f8057872dc071c9d6f9877357247dcb2312552d))
- feat: keep vectors in half precision, on request (#39) ([7518f52](https://github.com/tanvincible/chassis/commit/7518f526b9f58c089e06ab32f81dc78f4df5faf6))
- feat: warm(), reading an index into memory in the background (#40) ([a803de0](https://github.com/tanvincible/chassis/commit/a803de0162b28f4f3db44e09a3665750bf1755c1))

### Fixed

- fix: lock one byte on Windows and require Rust 1.88 ([f76a56f](https://github.com/tanvincible/chassis/commit/f76a56fe24359ba1eb6bd51f89b7fd9e60d7ab2d))
- fix: write committed nodes' lists back when a crash loses adds (#23) ([fe8c340](https://github.com/tanvincible/chassis/commit/fe8c340a29012b2f33e85673b05ee7fe1f92b1a7))

### Infrastructure

- ci: test the minimum Rust version for real (#13) ([5fb2159](https://github.com/tanvincible/chassis/commit/5fb21598216396cdd68ac517da0a4fb59e33ff9f))
- build(python): name the package chassisdb (#36) ([5210bc0](https://github.com/tanvincible/chassis/commit/5210bc0004d8a2a46a0c5bc0320ee959aaf2f55c))
- ci: build the C library and Python wheels on every platform (#44) ([48f8190](https://github.com/tanvincible/chassis/commit/48f819018ecaecad5a05fb01edefb2b6b3ffc641))
- ci: benchmark every pull request against main (#45) ([8b8480f](https://github.com/tanvincible/chassis/commit/8b8480f3a91b41e572fded06812f2070da1f103d))
- ci: build wheels with the stable toolchain as installed, without rust-toolchain.toml's extras (#47) ([0a18448](https://github.com/tanvincible/chassis/commit/0a18448b7c6c681c69b28ed7dab5ba489714acf6))

### Performance

- perf: prefetch neighbor vectors and reuse the visited filter (#20) ([85a1908](https://github.com/tanvincible/chassis/commit/85a1908fe3e3b2e6d08bd7bb4d16480a49dfd3ab))
- perf: prefetch into L2, a leaner search loop, and a smaller fill (#25) ([ce759f2](https://github.com/tanvincible/chassis/commit/ce759f2786622b7f249c7882aba99babd1c23667))
- perf: packed heap entries, squared distances, and a search loop that inlines (#27) ([bdb874d](https://github.com/tanvincible/chassis/commit/bdb874d025545d06b7d995046ee8226f31f08df1))
- perf: prefetch more of a long vector where the CPU does better with it (#30) ([7f4efbe](https://github.com/tanvincible/chassis/commit/7f4efbe35993cf439814727cd425fe7fce3792d7))
- perf: compute a search's distances in groups (#42) ([8d8b970](https://github.com/tanvincible/chassis/commit/8d8b9701696ef2e81a6759c93d278f0846015c42))

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
