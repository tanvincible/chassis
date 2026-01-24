# ADR-0006: SIMD Acceleration

**Date:** 2026-01-24
**Status:** Accepted

## Context

As Chassis moves from a “graph builder” to a “search engine,” raw arithmetic throughput becomes the dominant bottleneck. Profiling reveals that searching high-dimensional vectors (e.g., 1536-dimensional OpenAI embeddings) is limited by the CPU’s ability to process floating-point operations sequentially.

* **The Bottleneck:** Scalar execution processes one dimension at a time. Modern CPUs have vector units (AVX2, NEON) capable of processing multiple floats per cycle, but compilers often fail to autovectorize complex distance loops optimally.
* **Dependency Chains:** Standard accumulation (`sum += diff * diff`) creates a dependency chain where each addition must wait for the previous one to complete. This exposes instruction latency (≈4 cycles) rather than throughput (≈0.5 cycles), leaving most execution units idle.
* **Target Performance:** To achieve sub-millisecond search latencies at scale, we must saturate the CPU’s floating-point execution pipelines.

## Decision

We will implement **hand-written SIMD kernels** for distance metrics using a **4-way unrolled accumulation** strategy.

### 1. Hardware Targeting

We explicitly target the two dominant server and edge architectures:

* **x86_64:** AVX2 with FMA (Fused Multiply-Add).
* **aarch64:** NEON (standard on all modern ARM cores, including Apple Silicon and AWS Graviton).

### 2. The “4-Way Unroll” Pattern

We do not rely on compiler autovectorization. Instead, we explicitly break dependency chains:

* Maintain **four independent vector accumulators** (`sum0`, `sum1`, `sum2`, `sum3`) inside the hot loop.
* This allows the CPU to issue multiple independent FMA instructions concurrently, hiding instruction latency behind pipeline throughput.
* **Result:** Effective throughput scales from one dependent FMA per latency window to four independent FMAs, saturating available execution ports.

### 3. Safety & Dispatch

* **Runtime Detection (x86):** Use `std::is_x86_feature_detected!("avx2")` to select the optimized kernel safely at runtime.
* **Compile-Time Detection (ARM):** NEON is guaranteed on `aarch64` and enabled via `#[cfg(target_arch = "aarch64")]`.
* **Scalar Fallback:** A pure Rust implementation is retained for unsupported hardware and correctness validation.

All unsafe SIMD kernels operate under the following invariants:

* Input slices have equal length.
* Bounds are validated before entering unsafe code.
* Unaligned loads are used deliberately and verified via benchmarks.

## Consequences

### Positive

#### Order-of-Magnitude Speedup

Benchmarks confirm a **~10× improvement** in raw distance calculation for 1536-dimensional vectors (≈750ns → ≈70ns). Throughput reaches ~23 Giga-elements/second (where one “element” corresponds to one float dimension processed), approaching the sustained hardware throughput ceiling on modern CPUs.

#### Search Latency Reduction

End-to-end search latency for 1536d vectors drops from ~330µs to ~35µs, enabling high-throughput workloads (>20k QPS per core on a single socket).

#### Stable P99 Latency

By replacing compiler-dependent loops with explicit intrinsics, performance becomes deterministic and robust across compiler versions and optimization flags.

### Negative

#### Unsafe Code Proliferation

SIMD intrinsics require `unsafe` blocks, increasing review burden and necessitating careful testing to avoid undefined behavior.

#### Maintenance Overhead

The distance implementation now has three parallel variants (Scalar, AVX2, NEON). Any new distance metric must be implemented and validated across all targets. To mitigate this, all SIMD dispatch is centralized in `distance.rs`.

## Compliance

* **Implementation:** `chassis-core/src/distance.rs` contains all SIMD kernels and the 4-way unrolled accumulation logic.
* **Dispatch:** The public `euclidean_distance` entry point selects the appropriate kernel before entering unsafe code.
* **Verification:** Benchmarks include explicit coverage for unaligned access and cross-implementation correctness.
