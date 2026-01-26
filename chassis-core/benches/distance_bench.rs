//! SIMD distance computation benchmarks.
//!
//! Measures:
//! - Absolute timings for SIMD vs scalar
//! - Scaling with vector dimensions
//! - Different vector patterns (aligned, unaligned, sparse)
//!
//! Expected results:
//! - AVX2: 4-6x speedup on 768-1536D
//! - NEON: 3-5x speedup on 768-1536D

use chassis_core::distance::{euclidean_distance, euclidean_distance_scalar};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

/// Benchmark: Pure distance computation at various dimensions
fn bench_distance_by_dimension(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_by_dimension");
    group.sample_size(1000); // High sample count for precise measurements

    for dims in [64, 128, 384, 768, 1536, 3072] {
        group.throughput(Throughput::Elements(dims));

        // Pre-generate vectors
        let a: Vec<f32> = (0..dims).map(|i| (i as f32).sin() * 0.5).collect();
        let b: Vec<f32> = (0..dims).map(|i| (i as f32).cos() * 0.5).collect();

        group.bench_with_input(BenchmarkId::new("simd", dims), &dims, |bench, _| {
            bench.iter(|| black_box(euclidean_distance(black_box(&a), black_box(&b))));
        });

        group.bench_with_input(BenchmarkId::new("scalar", dims), &dims, |bench, _| {
            bench.iter(|| black_box(euclidean_distance_scalar(black_box(&a), black_box(&b))));
        });
    }

    group.finish();
}

/// Benchmark: High-dimensional vectors (typical embeddings)
fn bench_high_dimensional(c: &mut Criterion) {
    let mut group = c.benchmark_group("high_dimensional");
    group.sample_size(500);

    // OpenAI ada-002 dimensions
    let dims = 1536;
    let a: Vec<f32> = (0..dims).map(|i| (i as f32 * 0.1).sin()).collect();
    let b: Vec<f32> = (0..dims).map(|i| (i as f32 * 0.1).cos()).collect();

    group.throughput(Throughput::Elements(dims));

    group.bench_function("simd_1536d", |bench| {
        bench.iter(|| black_box(euclidean_distance(&a, &b)));
    });

    group.bench_function("scalar_1536d", |bench| {
        bench.iter(|| black_box(euclidean_distance_scalar(&a, &b)));
    });

    group.finish();
}

/// Benchmark: Batch distance computation (common in search)
fn bench_batch_distances(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_distances");
    group.sample_size(100);

    let dims = 768;
    let batch_size = 100;

    let query: Vec<f32> = (0..dims).map(|i| (i as f32) * 0.01).collect();
    let vectors: Vec<Vec<f32>> =
        (0..batch_size).map(|j| (0..dims).map(|i| ((i + j) as f32) * 0.01).collect()).collect();

    group.throughput(Throughput::Elements(batch_size * dims));

    group.bench_function("simd_batch", |bench| {
        bench.iter(|| {
            for vec in &vectors {
                black_box(euclidean_distance(&query, vec));
            }
        });
    });

    group.bench_function("scalar_batch", |bench| {
        bench.iter(|| {
            for vec in &vectors {
                black_box(euclidean_distance_scalar(&query, vec));
            }
        });
    });

    group.finish();
}

/// Benchmark: Unaligned memory access patterns
fn bench_unaligned_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("unaligned_access");
    group.sample_size(500);

    let dims = 1024;

    // Create vectors with offset to test unaligned loads
    let mut buffer_a = vec![0.0f32; dims + 3];
    let mut buffer_b = vec![0.0f32; dims + 3];

    for i in 0..dims {
        buffer_a[i + 1] = (i as f32).sin();
        buffer_b[i + 2] = (i as f32).cos();
    }

    let a = &buffer_a[1..dims + 1]; // Offset by 4 bytes
    let b = &buffer_b[2..dims + 2]; // Offset by 8 bytes

    group.bench_function("simd_unaligned", |bench| {
        bench.iter(|| black_box(euclidean_distance(black_box(a), black_box(b))));
    });

    group.finish();
}

/// Benchmark: Small vectors (tail loop dominant)
fn bench_small_vectors(c: &mut Criterion) {
    let mut group = c.benchmark_group("small_vectors");
    group.sample_size(1000);

    for dims in [3, 7, 15, 31, 63] {
        let a: Vec<f32> = (0..dims).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..dims).map(|i| (i + 1) as f32).collect();

        group.throughput(Throughput::Elements(dims));

        group.bench_with_input(BenchmarkId::new("dims", dims), &dims, |bench, _| {
            bench.iter(|| black_box(euclidean_distance(&a, &b)));
        });
    }

    group.finish();
}

/// Benchmark: Sparse vectors (many zeros)
fn bench_sparse_vectors(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_vectors");
    group.sample_size(500);

    let dims = 1536;
    let mut a = vec![0.0; dims];
    let mut b = vec![0.0; dims];

    // Only 10% non-zero
    for i in (0..dims).step_by(10) {
        a[i] = (i as f32).sin();
        b[i] = (i as f32).cos();
    }

    group.bench_function("simd_sparse_10pct", |bench| {
        bench.iter(|| black_box(euclidean_distance(&a, &b)));
    });

    group.bench_function("scalar_sparse_10pct", |bench| {
        bench.iter(|| black_box(euclidean_distance_scalar(&a, &b)));
    });

    group.finish();
}

/// Benchmark: Dense vs sparse comparison
fn bench_density_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("density_impact");
    group.sample_size(500);

    let dims = 768;

    for density in [10, 50, 100] {
        let mut a = vec![0.0; dims];
        let mut b = vec![0.0; dims];

        let step = 100 / density;
        for i in (0..dims).step_by(step) {
            a[i] = (i as f32).sin();
            b[i] = (i as f32).cos();
        }

        group.bench_with_input(BenchmarkId::new("simd_density", density), &density, |bench, _| {
            bench.iter(|| black_box(euclidean_distance(&a, &b)));
        });
    }

    group.finish();
}

/// Benchmark: Realistic embedding patterns
fn bench_realistic_embeddings(c: &mut Criterion) {
    let mut group = c.benchmark_group("realistic_embeddings");
    group.sample_size(500);

    // Simulate realistic embedding distributions
    // Embeddings typically have:
    // - Mean ~0
    // - Std dev ~0.1-0.3
    // - Some structure (not pure random)

    let dims = 768;
    let a: Vec<f32> = (0..dims)
        .map(|i| {
            let base = (i as f32 * 0.1).sin() * 0.2;
            let noise = (i as f32 * 7.0).sin() * 0.1;
            base + noise
        })
        .collect();

    let b: Vec<f32> = (0..dims)
        .map(|i| {
            let base = (i as f32 * 0.1 + 0.5).cos() * 0.2;
            let noise = (i as f32 * 11.0).cos() * 0.1;
            base + noise
        })
        .collect();

    group.bench_function("simd_realistic_768d", |bench| {
        bench.iter(|| black_box(euclidean_distance(&a, &b)));
    });

    group.bench_function("scalar_realistic_768d", |bench| {
        bench.iter(|| black_box(euclidean_distance_scalar(&a, &b)));
    });

    group.finish();
}

/// Benchmark: Memory bandwidth test (very large vectors)
fn bench_memory_bandwidth(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_bandwidth");
    group.sample_size(50);

    // Large vectors to test memory bandwidth limits
    for dims in [4096, 8192, 16384] {
        let a: Vec<f32> = (0..dims).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..dims).map(|i| (i as f32) * 0.001 + 1.0).collect();

        group.throughput(Throughput::Bytes((dims * 8 * std::mem::size_of::<f32>()) as u64));

        group.bench_with_input(BenchmarkId::new("simd", dims), &dims, |bench, _| {
            bench.iter(|| black_box(euclidean_distance(&a, &b)));
        });
    }

    group.finish();
}

/// Benchmark: Cache effects with repeated computation
fn bench_cache_effects(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_effects");
    group.sample_size(200);

    let dims = 1024;
    let a: Vec<f32> = (0..dims).map(|i| (i as f32) * 0.01).collect();
    let b: Vec<f32> = (0..dims).map(|i| (i as f32) * 0.01 + 0.5).collect();

    // Compute many times (stays in L1 cache)
    group.bench_function("hot_cache_simd", |bench| {
        bench.iter(|| {
            for _ in 0..100 {
                black_box(euclidean_distance(&a, &b));
            }
        });
    });

    group.bench_function("hot_cache_scalar", |bench| {
        bench.iter(|| {
            for _ in 0..100 {
                black_box(euclidean_distance_scalar(&a, &b));
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_distance_by_dimension,
    bench_high_dimensional,
    bench_batch_distances,
    bench_unaligned_access,
    bench_small_vectors,
    bench_sparse_vectors,
    bench_density_impact,
    bench_realistic_embeddings,
    bench_memory_bandwidth,
    bench_cache_effects,
);

criterion_main!(benches);
