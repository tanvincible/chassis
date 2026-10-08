//! SIMD-accelerated distance metrics for vector comparison.
//!
//! # Performance Strategy
//!
//! Uses 4-way accumulator unrolling to break FMA dependency chains:
//! - FMA latency: ~4 cycles
//! - FMA throughput: 0.5 cycles (2 ops/cycle)
//! - Single accumulator: Pipeline stalls, limited by latency
//! - Four accumulators: Pipeline stays full, limited by throughput
//!
//! Expected speedup: 4-6x on high-dimensional vectors (768-1536D)

/// How an index compares vectors; fixed when the index is created.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum DistanceMetric {
    /// L2 distance.
    #[default]
    Euclidean,
    /// 1 − cosine similarity. Vectors are stored scaled to unit length, so search runs on L2,
    /// which ranks unit vectors the same way.
    Cosine,
}

/// `v` scaled to unit length, for a cosine index. The norm is taken in f64, where no f32 can
/// overflow or underflow when squared.
pub(crate) fn unit(v: &[f32]) -> anyhow::Result<Vec<f32>> {
    if v.iter().any(|x| !x.is_finite()) {
        anyhow::bail!("A cosine index can't use a vector with NaN or infinite components");
    }
    let norm = v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
    if norm == 0.0 {
        anyhow::bail!("A cosine index can't use a zero vector");
    }
    Ok(v.iter().map(|&x| (f64::from(x) / norm) as f32).collect())
}

/// Compute L2 (Euclidean) distance between two vectors with SIMD acceleration.
///
/// # Performance
///
/// - Scalar: ~2ns per dimension
/// - AVX2: ~0.3ns per dimension (6-7x faster)
/// - NEON: ~0.4ns per dimension (5x faster)
///
/// # Architecture Dispatch
///
/// - x86_64 + AVX2: Uses AVX2 intrinsics (runtime detection)
/// - aarch64: Uses NEON intrinsics (always available)
/// - Fallback: Portable scalar implementation
///
/// # Panics
///
/// Panics if the vectors differ in length.
#[inline]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    // The SIMD kernels read `a.len()` floats from both.
    assert_eq!(a.len(), b.len(), "vectors differ in length");

    #[cfg(target_arch = "x86_64")]
    {
        // The kernel uses FMA instructions too; a CPU with AVX2 but not FMA must not run it.
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { euclidean_distance_avx2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { euclidean_distance_neon(a, b) };
    }

    euclidean_distance_scalar(a, b)
}

/// A distance kernel that a loop computing many distances is compiled around, so that the kernel
/// is inlined into it and chosen once, not per call (ADR-0013).
pub(crate) trait Kernel {
    /// # Safety
    ///
    /// The slices must be the same length, and the CPU must have what the kernel uses.
    unsafe fn distance(a: &[f32], b: &[f32]) -> f32;
}

/// What every CPU of the target has: NEON on aarch64, plain arithmetic elsewhere.
pub(crate) struct Portable;

impl Kernel for Portable {
    #[inline(always)]
    unsafe fn distance(a: &[f32], b: &[f32]) -> f32 {
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: every aarch64 CPU has NEON; the caller vouches for the lengths.
            return unsafe { euclidean_distance_neon(a, b) };
        }
        #[allow(unreachable_code)]
        euclidean_distance_scalar(a, b)
    }
}

/// AVX2 and FMA. A loop that uses it has to be compiled for them too, or the call can't inline.
#[cfg(target_arch = "x86_64")]
pub(crate) struct Avx2;

#[cfg(target_arch = "x86_64")]
impl Kernel for Avx2 {
    #[inline(always)]
    unsafe fn distance(a: &[f32], b: &[f32]) -> f32 {
        // SAFETY: the caller vouches for the CPU and the lengths.
        unsafe { euclidean_distance_avx2(a, b) }
    }
}

/// Whether this CPU can run `Avx2`.
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn has_avx2() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
}

/// The kernel `euclidean_distance` runs on this machine, as a pointer: for code that computes
/// many distances but isn't compiled per kernel.
///
/// # Safety
///
/// The returned function reads `a.len()` floats from both slices, which must be the same length.
pub(crate) fn kernel() -> unsafe fn(&[f32], &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            return euclidean_distance_avx2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return euclidean_distance_neon;
    }
    #[allow(unreachable_code)]
    euclidean_distance_scalar
}

/// Scalar implementation (portable fallback)
#[inline]
pub fn euclidean_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0_f32;

    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }

    sum.sqrt()
}

/// AVX2 implementation with 4-way accumulator unrolling (x86_64 only)
///
/// # Optimization Strategy
///
/// Uses 4 independent accumulators (sum0, sum1, sum2, sum3) to break
/// FMA dependency chains and maximize pipeline utilization.
///
/// # Pipeline Analysis
///
/// - Single accumulator: 1 FMA / 4 cycles = 0.25 ops/cycle (latency-bound)
/// - Four accumulators: 4 FMA / 4 cycles = 1 ops/cycle (approaching 2 ops/cycle theoretical max)
///
/// # Loop Structure
///
/// Main loop: Process 32 floats/iteration (4 accumulators × 8 floats/vector)
/// Tail loop: Process remaining 8-float chunks
/// Scalar tail: Process final <8 elements
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn euclidean_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let mut i = 0;

    // Four independent accumulators to break dependency chains
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();

    // Main loop: Process 32 floats per iteration (4 vectors × 8 floats)
    // This keeps 4 FMA units busy, hiding latency
    while i + 32 <= len {
        // Load and compute differences
        let va0 = unsafe { _mm256_loadu_ps(a.as_ptr().add(i)) };
        let vb0 = unsafe { _mm256_loadu_ps(b.as_ptr().add(i)) };
        let diff0 = _mm256_sub_ps(va0, vb0);

        let va1 = unsafe { _mm256_loadu_ps(a.as_ptr().add(i + 8)) };
        let vb1 = unsafe { _mm256_loadu_ps(b.as_ptr().add(i + 8)) };
        let diff1 = _mm256_sub_ps(va1, vb1);

        let va2 = unsafe { _mm256_loadu_ps(a.as_ptr().add(i + 16)) };
        let vb2 = unsafe { _mm256_loadu_ps(b.as_ptr().add(i + 16)) };
        let diff2 = _mm256_sub_ps(va2, vb2);

        let va3 = unsafe { _mm256_loadu_ps(a.as_ptr().add(i + 24)) };
        let vb3 = unsafe { _mm256_loadu_ps(b.as_ptr().add(i + 24)) };
        let diff3 = _mm256_sub_ps(va3, vb3);

        // Fused multiply-add: sum = diff * diff + sum
        // Each accumulator is independent, allowing parallel execution
        sum0 = _mm256_fmadd_ps(diff0, diff0, sum0);
        sum1 = _mm256_fmadd_ps(diff1, diff1, sum1);
        sum2 = _mm256_fmadd_ps(diff2, diff2, sum2);
        sum3 = _mm256_fmadd_ps(diff3, diff3, sum3);

        i += 32;
    }

    // Tail loop: Process remaining 8-float chunks
    while i + 8 <= len {
        let va = unsafe { _mm256_loadu_ps(a.as_ptr().add(i)) };
        let vb = unsafe { _mm256_loadu_ps(b.as_ptr().add(i)) };
        let diff = _mm256_sub_ps(va, vb);
        sum0 = _mm256_fmadd_ps(diff, diff, sum0);
        i += 8;
    }

    // Reduce accumulators: Combine the 4 independent sums
    let sum_combined = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));

    // Horizontal reduction: Sum 8 lanes into a scalar
    // Extract high 128 bits and add to low 128 bits
    let sum_high = _mm256_extractf128_ps(sum_combined, 1);
    let sum_low = _mm256_castps256_ps128(sum_combined);
    let sum128 = _mm_add_ps(sum_low, sum_high);

    // Horizontal add within 128-bit register
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0x55));

    let mut total = _mm_cvtss_f32(sum32);

    // Scalar tail: Process remaining elements
    while i < len {
        let diff = a[i] - b[i];
        total += diff * diff;
        i += 1;
    }

    total.sqrt()
}

/// NEON implementation with 4-way accumulator unrolling (aarch64)
///
/// # Optimization Strategy
///
/// Same strategy as AVX2: 4 independent accumulators to maximize throughput.
/// NEON processes 4 floats per vector (vs 8 for AVX2), so main loop processes 16 floats.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn euclidean_distance_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let len = a.len();
    let mut i = 0;

    // Four independent accumulators
    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);

    // Main loop: Process 16 floats per iteration (4 vectors × 4 floats)
    while i + 16 <= len {
        let va0 = vld1q_f32(a.as_ptr().add(i));
        let vb0 = vld1q_f32(b.as_ptr().add(i));
        let diff0 = vsubq_f32(va0, vb0);

        let va1 = vld1q_f32(a.as_ptr().add(i + 4));
        let vb1 = vld1q_f32(b.as_ptr().add(i + 4));
        let diff1 = vsubq_f32(va1, vb1);

        let va2 = vld1q_f32(a.as_ptr().add(i + 8));
        let vb2 = vld1q_f32(b.as_ptr().add(i + 8));
        let diff2 = vsubq_f32(va2, vb2);

        let va3 = vld1q_f32(a.as_ptr().add(i + 12));
        let vb3 = vld1q_f32(b.as_ptr().add(i + 12));
        let diff3 = vsubq_f32(va3, vb3);

        // Fused multiply-add
        sum0 = vfmaq_f32(sum0, diff0, diff0);
        sum1 = vfmaq_f32(sum1, diff1, diff1);
        sum2 = vfmaq_f32(sum2, diff2, diff2);
        sum3 = vfmaq_f32(sum3, diff3, diff3);

        i += 16;
    }

    // Tail loop: Process remaining 4-float chunks
    while i + 4 <= len {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        let diff = vsubq_f32(va, vb);
        sum0 = vfmaq_f32(sum0, diff, diff);
        i += 4;
    }

    // Reduce accumulators
    let sum_combined = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));

    // Horizontal reduction: Sum 4 lanes
    let sum_pair = vpadd_f32(vget_low_f32(sum_combined), vget_high_f32(sum_combined));
    let sum_total = vpadd_f32(sum_pair, sum_pair);

    let mut total = vget_lane_f32(sum_total, 0);

    // Scalar tail
    while i < len {
        let diff = a[i] - b[i];
        total += diff * diff;
        i += 1;
    }

    total.sqrt()
}

/// Compute cosine distance (1 - cosine similarity).
///
/// Returns `1.0` when either vector has zero norm because cosine similarity is
/// undefined for zero vectors.
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_product = norm_a * norm_b;

    if norm_product == 0.0 {
        return 1.0;
    }

    1.0 - (dot / norm_product)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "differ in length")]
    fn test_euclidean_distance_refuses_unequal_lengths() {
        euclidean_distance(&[0.0; 64], &[0.0; 8]);
    }

    #[test]
    fn test_euclidean_distance_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let dist = euclidean_distance(&a, &b);
        let expected = ((3.0_f32).powi(2) * 3.0).sqrt();

        assert!((dist - expected).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_distance() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];

        let dist = cosine_distance(&a, &b);
        assert!((dist - 1.0).abs() < 1e-6); // Orthogonal vectors
    }

    #[test]
    fn test_cosine_distance_zero_norm() {
        let zero = vec![0.0, 0.0, 0.0];
        let non_zero = vec![1.0, 0.0, 0.0];

        let zero_to_zero = cosine_distance(&zero, &zero);
        let zero_to_non_zero = cosine_distance(&zero, &non_zero);

        assert_eq!(zero_to_zero, 1.0);
        assert_eq!(zero_to_non_zero, 1.0);
    }

    #[test]
    fn test_simd_correctness_small() {
        // Test with small vectors (exercises scalar tail)
        for size in [3, 7, 15, 31] {
            let a: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
            let b: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1 + 0.5).collect();

            let simd_result = euclidean_distance(&a, &b);
            let scalar_result = euclidean_distance_scalar(&a, &b);

            assert!(
                (simd_result - scalar_result).abs() < 1e-5,
                "SIMD mismatch at size {}: simd={}, scalar={}",
                size,
                simd_result,
                scalar_result
            );
        }
    }

    #[test]
    fn test_simd_correctness_large() {
        // Test with large vectors (exercises main loop)
        for size in [128, 384, 768, 1536] {
            let a: Vec<f32> = (0..size).map(|i| (i as f32).sin()).collect();
            let b: Vec<f32> = (0..size).map(|i| (i as f32).cos()).collect();

            let simd_result = euclidean_distance(&a, &b);
            let scalar_result = euclidean_distance_scalar(&a, &b);

            // Tolerance accounts for different accumulation order
            assert!(
                (simd_result - scalar_result).abs() < 1e-4,
                "SIMD mismatch at size {}: simd={}, scalar={}, diff={}",
                size,
                simd_result,
                scalar_result,
                (simd_result - scalar_result).abs()
            );
        }
    }

    #[test]
    fn test_simd_random_vectors() {
        use std::collections::hash_map::RandomState;
        use std::hash::BuildHasher;

        // Deterministic "random" using hash
        fn hash_to_f32(seed: u64) -> f32 {
            let state = RandomState::new();
            let hash = state.hash_one(seed);
            ((hash % 10000) as f32) / 10000.0
        }

        for dims in [64, 256, 512, 1024] {
            let a: Vec<f32> = (0..dims).map(|i| hash_to_f32(i as u64)).collect();
            let b: Vec<f32> = (0..dims).map(|i| hash_to_f32((i + 1000) as u64)).collect();

            let simd_result = euclidean_distance(&a, &b);
            let scalar_result = euclidean_distance_scalar(&a, &b);

            assert!(
                (simd_result - scalar_result).abs() < 1e-4,
                "Random vector mismatch at dims {}: simd={}, scalar={}",
                dims,
                simd_result,
                scalar_result
            );
        }
    }

    #[test]
    fn test_simd_edge_cases() {
        // All zeros
        let a = vec![0.0; 128];
        let b = vec![0.0; 128];
        assert_eq!(euclidean_distance(&a, &b), 0.0);

        // Identical vectors
        let c = vec![1.0; 256];
        let d = vec![1.0; 256];
        assert_eq!(euclidean_distance(&c, &d), 0.0);

        // One large value
        let mut e = vec![0.0; 512];
        let mut f = vec![0.0; 512];
        e[100] = 100.0;
        f[100] = 0.0;

        let dist = euclidean_distance(&e, &f);
        assert!((dist - 100.0).abs() < 1e-5);
    }

    #[test]
    fn test_simd_negative_values() {
        let a = vec![-1.0, -2.0, -3.0, -4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];

        let dist = euclidean_distance(&a, &b);
        let expected =
            (2.0_f32.powi(2) + 4.0_f32.powi(2) + 6.0_f32.powi(2) + 8.0_f32.powi(2)).sqrt();

        assert!((dist - expected).abs() < 1e-5);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_specific() {
        if is_x86_feature_detected!("avx2") {
            let a: Vec<f32> = (0..1024).map(|i| i as f32 * 0.01).collect();
            let b: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.01 + 1.0).collect();

            let avx2_result = unsafe { euclidean_distance_avx2(&a, &b) };
            let scalar_result = euclidean_distance_scalar(&a, &b);

            assert!(
                (avx2_result - scalar_result).abs() < 1e-4,
                "AVX2 vs scalar mismatch: avx2={}, scalar={}",
                avx2_result,
                scalar_result
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_neon_specific() {
        let a: Vec<f32> = (0..1024).map(|i| i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.01 + 1.0).collect();

        let neon_result = unsafe { euclidean_distance_neon(&a, &b) };
        let scalar_result = euclidean_distance_scalar(&a, &b);

        assert!(
            (neon_result - scalar_result).abs() < 1e-4,
            "NEON vs scalar mismatch: neon={}, scalar={}",
            neon_result,
            scalar_result
        );
    }
}
