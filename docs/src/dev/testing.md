# Running Tests

## Run All Tests

```bash
cargo test
```

This runs all unit tests and integration tests.

## Run Specific Tests

Run tests matching a pattern:

```bash
cargo test test_exclusive_locking
```

Run tests in a specific file:

```bash
cargo test --test storage_tests
```

## Test with Output

By default, `cargo test` captures stdout. To see print statements:

```bash
cargo test -- --nocapture
```

## Important Test Cases

### File Locking

The `test_exclusive_locking` test verifies that two processes cannot open the same file simultaneously. This prevents data corruption from concurrent writes.

### Page Alignment

The `test_file_growth_is_page_aligned` test ensures all file operations respect 4KB boundaries. This is critical for SSD performance and correctness on some filesystems.

### Lock Release

The `test_lock_released_on_drop` test verifies that file locks are properly released when the `Storage` object is dropped. Without this, files could remain locked even after the owning process exits abnormally.

### Persistence

The `test_persistence_across_reopens` test writes data, closes the file, reopens it, and verifies the data survived. This confirms the file format and commit logic work correctly.

## Platform Testing

Tests should pass on Linux, macOS, and Windows. File locking behavior differs slightly across platforms, so cross-platform testing is important.

ARM testing is done on actual hardware, not emulation. Alignment issues may not appear in QEMU.

## Adding Tests

All new features must include tests. Bug fixes should include a regression test that fails before the fix and passes after.

Tests should be isolated. Use `tempfile::NamedTempFile` to create temporary files that are automatically cleaned up.
