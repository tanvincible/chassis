# Storage Benchmarks

This document reports benchmark results for the **Chassis storage layer**.
The measurements focus exclusively on storage primitives (insert, read, growth, remapping) and do **not** include indexing or search logic.

All results were obtained on a single local machine. Absolute numbers will vary across hardware; the goal is to characterize **behavior, scaling, and cost structure**, not to present universal performance claims.

## Hardware

* **CPU:** Intel Core i7-12650H (10 cores, 16 threads, up to 4.7 GHz)
* **RAM:** 16 GB
* **Storage:** Intel SSDPEKNU512GZ (512 GB, consumer NVMe)
* **OS:** Manjaro Linux 6.16.8
* **Filesystem:** ext4, mounted with `noatime`
* **CPU governor:** `performance`
* **NVMe write cache:** disabled at the Linux block layer (`write through`)

> Note: Disabling the Linux write cache does not disable device-internal buffering or firmware-level behavior. Consumer NVMe devices typically lack power-loss protection.

## Test Configuration

* **Vector dimensions:** 768 (`f32`)
* **Vector size:** 3,072 bytes
* **Rust version:** 1.92.0
* **Build mode:** release (`cargo bench`)
* **Benchmark harness:** Criterion 0.8.1

## Running the Benchmarks

```bash
# Set CPU governor
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Run all benchmarks
cargo bench --bench storage

# View HTML report
open target/criterion/report/index.html
```

## Results

All values below are **median** unless stated otherwise.

### Raw Insert (No Commit)

Measures the cost of appending a vector without durability guarantees.

| Metric         | Value   |
| -------------- | ------- |
| Median latency | ~6.9 µs |
| Outliers       | ~7–8%   |

**Interpretation**

This represents the baseline cost of the storage layer: bounds checks, offset computation, and memory copy into an mmap’d region. No filesystem synchronization is performed.

The measured latency is consistent with copying ~3 KB of data plus minimal bookkeeping. Outliers are attributable to page faults and occasional remapping during file growth.

### Insert With Commit

Measures the cost of inserting a vector followed by `commit()`.

| Metric         | Value   |
| -------------- | ------- |
| Median latency | ~6.0 µs |
| Outliers       | ~12%    |

**Interpretation**

On this hardware, calling `commit()` does not introduce millisecond-scale latency. This indicates that the kernel and device acknowledge completion before data is guaranteed to be physically persisted to non-volatile media.

This behavior is typical for consumer NVMe devices without power-loss protection. It should not be interpreted as strong durability.

### Batch Insert

Measures amortized cost when multiple inserts are followed by a single commit.

| Batch size | Total time | Per insert |
| ---------- | ---------- | ---------- |
| 10         | ~54 µs     | ~5.4 µs    |
| 100        | ~606 µs    | ~6.1 µs    |
| 1000       | ~5.38 ms   | ~5.4 µs    |

**Interpretation**

Per-insert cost remains nearly constant across batch sizes, confirming that commit overhead is negligible on this hardware and that the insert path scales linearly.

On hardware with real persistence guarantees (e.g., enterprise NVMe with PLP), batching would be expected to amortize millisecond-scale sync costs.

### Hot Read (Cached)

Measures reading a vector already resident in CPU cache.

| Metric         | Value  |
| -------------- | ------ |
| Median latency | ~75 ns |

**Interpretation**

This reflects pure memory access and copy cost. The storage layer imposes no measurable overhead beyond pointer arithmetic and memcpy.

### Cold Read (Application Cold Start)

Measures reading a vector after reopening the file.

| Metric         | Value  |
| -------------- | ------ |
| Median latency | ~13 µs |

**Interpretation**

This includes file open, mmap, header validation, and the first data access. It approximates a realistic application cold-start read rather than raw NVMe latency.

### Sequential Read

Measures reading 1,000 vectors sequentially.

| Metric     | Value   |
| ---------- | ------- |
| Total time | ~139 µs |
| Per vector | ~139 ns |

**Interpretation**

Sequential access benefits from kernel readahead and cache locality. This pattern is relevant for index construction and graph traversal workloads.

### Remap Overhead

Measures growth from an empty file to 1,000 vectors, including all remaps.

| Metric     | Value              |
| ---------- | ------------------ |
| Total time | ~4.9 ms            |
| Amortized  | ~4.9 µs per insert |

**Interpretation**

Page-aligned growth keeps remapping costs bounded. The amortized overhead is comparable to raw insert cost and does not introduce performance cliffs.

## Key Observations

* **Insert cost:** ~6–7 µs per vector (memory-bound)
* **Read cost (hot):** sub-100 ns
* **Read cost (cold):** ~10–15 µs including file open
* **Growth behavior:** predictable and amortized
* **Variance:** low and explainable

## Durability Considerations

On this system, `fsync` completes in microseconds, indicating that completion is acknowledged before data is guaranteed to survive sudden power loss.

This is common behavior for consumer NVMe devices without power-loss protection. It implies:

* Commits are fast but not strongly durable
* Recent writes may be lost on sudden power failure
* This is a hardware property, not a software limitation

Chassis performs all software-visible durability steps correctly. Strong durability requires hardware support (e.g., PLP-enabled NVMe) or external protection such as a UPS.

## Scope and Comparisons

These benchmarks measure **storage primitives only**. They are not directly comparable to:

* In-memory ANN libraries (no persistence)
* Networked vector databases (different latency and consistency model)
* Search engines or query processors

The intent is to characterize the storage foundation on which indexing and search will be built.

## Reproducibility

To reproduce these results:

1. Clone the repository
2. Set CPU governor to `performance`
3. Ensure minimal background load
4. Record storage device model and filesystem
5. Run `cargo bench --bench storage`

When reporting results, include hardware, OS, filesystem, and governor settings. Differences in NVMe class (consumer vs enterprise) will significantly affect durability-related measurements.
