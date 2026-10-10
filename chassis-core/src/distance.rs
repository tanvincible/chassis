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

use crate::half::Precision;

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
        crate::error::fail!(
            InvalidArgument,
            "A cosine index can't use a vector with NaN or infinite components\nhelp: check how \
             it was made; a division by zero gives NaN"
        );
    }
    let norm = v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
    if norm == 0.0 {
        crate::error::fail!(
            InvalidArgument,
            "A cosine index can't use a zero vector: it has no direction\nhelp: leave it out, or \
             use a euclidean index, where it is a point like any other"
        );
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
pub(crate) trait Kernel<E = f32> {
    /// The squared L2 distance from f32s to a stored vector, which orders vectors as the
    /// distance does and leaves the square root to whoever reports one.
    ///
    /// # Safety
    ///
    /// The slices must be the same length, and the CPU must have what the kernel uses.
    unsafe fn squared(a: &[f32], b: &[E]) -> f32;
}

/// A component of a stored vector: an `f32`, or a 16-bit float held in a `u16` (ADR-0018).
pub(crate) trait Element: Copy + Send + Sync + 'static {
    const PRECISION: Precision;

    /// The Euclidean distance from f32s to a stored vector on this machine, as a pointer: for
    /// code that computes many distances but isn't compiled per kernel.
    ///
    /// # Safety
    ///
    /// The returned function reads `a.len()` components from both slices, which must be the same
    /// length.
    fn kernel() -> unsafe fn(&[f32], &[Self]) -> f32;

    /// `stored` as f32s: itself, or widened into `scratch`.
    fn widened<'a>(stored: &'a [Self], scratch: &'a mut Vec<f32>) -> &'a [f32];
}

impl Element for f32 {
    const PRECISION: Precision = Precision::Full;

    fn kernel() -> unsafe fn(&[f32], &[f32]) -> f32 {
        kernel()
    }

    #[inline]
    fn widened<'a>(stored: &'a [f32], _: &'a mut Vec<f32>) -> &'a [f32] {
        stored
    }
}

impl Element for u16 {
    const PRECISION: Precision = Precision::Half;

    fn kernel() -> unsafe fn(&[f32], &[u16]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        if has_f16c() {
            return rooted::<F16c>;
        }
        rooted::<Portable>
    }

    fn widened<'a>(stored: &'a [u16], scratch: &'a mut Vec<f32>) -> &'a [f32] {
        crate::half::widen(stored, scratch);
        scratch
    }
}

/// The distance to a vector in half precision, with kernel `K`.
///
/// # Safety
///
/// As `Kernel::squared`.
unsafe fn rooted<K: Kernel<u16>>(a: &[f32], b: &[u16]) -> f32 {
    // SAFETY: the caller's.
    unsafe { K::squared(a, b) }.sqrt()
}

/// What every CPU of the target has: NEON on aarch64, plain arithmetic elsewhere.
pub(crate) struct Portable;

impl Kernel for Portable {
    #[inline(always)]
    unsafe fn squared(a: &[f32], b: &[f32]) -> f32 {
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: every aarch64 CPU has NEON; the caller vouches for the lengths.
            return unsafe { squared_neon(a, b) };
        }
        #[allow(unreachable_code)]
        squared_scalar(a, b)
    }
}

impl Kernel<u16> for Portable {
    #[inline(always)]
    unsafe fn squared(a: &[f32], b: &[u16]) -> f32 {
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: every aarch64 CPU has NEON; the caller vouches for the lengths.
            return unsafe { squared_half_neon(a, b) };
        }
        #[allow(unreachable_code)]
        squared_half_scalar(a, b)
    }
}

/// AVX2 and FMA. A loop that uses it has to be compiled for them too, or the call can't inline.
#[cfg(target_arch = "x86_64")]
pub(crate) struct Avx2;

#[cfg(target_arch = "x86_64")]
impl Kernel for Avx2 {
    #[inline(always)]
    unsafe fn squared(a: &[f32], b: &[f32]) -> f32 {
        // SAFETY: the caller vouches for the CPU and the lengths.
        unsafe { squared_avx2(a, b) }
    }
}

/// Whether this CPU can run `Avx2`.
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn has_avx2() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
}

/// `Avx2` for vectors in half precision, which F16C widens. Its loops are compiled for F16C too.
#[cfg(target_arch = "x86_64")]
pub(crate) struct F16c;

#[cfg(target_arch = "x86_64")]
impl Kernel<u16> for F16c {
    #[inline(always)]
    unsafe fn squared(a: &[f32], b: &[u16]) -> f32 {
        // SAFETY: the caller vouches for the CPU and the lengths.
        unsafe { squared_half_avx2(a, b) }
    }
}

/// Whether this CPU can run `F16c`.
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn has_f16c() -> bool {
    has_avx2() && is_x86_feature_detected!("f16c")
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
    squared_scalar(a, b).sqrt()
}

#[inline]
fn squared_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0_f32;

    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }

    sum
}

/// `squared_scalar` to a vector in half precision.
#[inline]
fn squared_half_scalar(a: &[f32], b: &[u16]) -> f32 {
    let mut sum = 0.0_f32;

    for i in 0..a.len() {
        let diff = a[i] - crate::half::decode(b[i]);
        sum += diff * diff;
    }

    sum
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
    // SAFETY: the caller's.
    unsafe { squared_avx2(a, b) }.sqrt()
}

/// `euclidean_distance_avx2` before the square root.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn squared_avx2(a: &[f32], b: &[f32]) -> f32 {
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

    total
}

/// `squared_avx2` to a vector in half precision: the same sums in the same order, over components
/// that F16C widens eight at a time.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn squared_half_avx2(a: &[f32], b: &[u16]) -> f32 {
    use std::arch::x86_64::*;

    // SAFETY (both): the loops below call them within the slices, which are the same length.
    let query = |i: usize| unsafe { _mm256_loadu_ps(a.as_ptr().add(i)) };
    let stored = |i: usize| unsafe { _mm256_cvtph_ps(_mm_loadu_si128(b.as_ptr().add(i).cast())) };

    let len = a.len();
    let mut i = 0;

    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();

    while i + 32 <= len {
        let diff0 = _mm256_sub_ps(query(i), stored(i));
        let diff1 = _mm256_sub_ps(query(i + 8), stored(i + 8));
        let diff2 = _mm256_sub_ps(query(i + 16), stored(i + 16));
        let diff3 = _mm256_sub_ps(query(i + 24), stored(i + 24));

        sum0 = _mm256_fmadd_ps(diff0, diff0, sum0);
        sum1 = _mm256_fmadd_ps(diff1, diff1, sum1);
        sum2 = _mm256_fmadd_ps(diff2, diff2, sum2);
        sum3 = _mm256_fmadd_ps(diff3, diff3, sum3);

        i += 32;
    }

    while i + 8 <= len {
        let diff = _mm256_sub_ps(query(i), stored(i));
        sum0 = _mm256_fmadd_ps(diff, diff, sum0);
        i += 8;
    }

    let sum_combined = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
    let sum_high = _mm256_extractf128_ps(sum_combined, 1);
    let sum_low = _mm256_castps256_ps128(sum_combined);
    let sum128 = _mm_add_ps(sum_low, sum_high);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0x55));

    let mut total = _mm_cvtss_f32(sum32);

    while i < len {
        let diff = a[i] - crate::half::decode(b[i]);
        total += diff * diff;
        i += 1;
    }

    total
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
    // SAFETY: the caller's.
    unsafe { squared_neon(a, b) }.sqrt()
}

/// `euclidean_distance_neon` before the square root.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn squared_neon(a: &[f32], b: &[f32]) -> f32 {
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

    total
}

/// `squared_neon` to a vector in half precision: the same sums in the same order, over components
/// widened four at a time.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn squared_half_neon(a: &[f32], b: &[u16]) -> f32 {
    use crate::half::widen4;
    use std::arch::aarch64::*;

    // SAFETY (both): the loops below call them within the slices, which are the same length.
    let query = |i: usize| unsafe { vld1q_f32(a.as_ptr().add(i)) };
    let stored = |i: usize| unsafe { widen4(b.as_ptr().add(i)) };

    let len = a.len();
    let mut i = 0;

    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);

    while i + 16 <= len {
        let diff0 = vsubq_f32(query(i), stored(i));
        let diff1 = vsubq_f32(query(i + 4), stored(i + 4));
        let diff2 = vsubq_f32(query(i + 8), stored(i + 8));
        let diff3 = vsubq_f32(query(i + 12), stored(i + 12));

        sum0 = vfmaq_f32(sum0, diff0, diff0);
        sum1 = vfmaq_f32(sum1, diff1, diff1);
        sum2 = vfmaq_f32(sum2, diff2, diff2);
        sum3 = vfmaq_f32(sum3, diff3, diff3);

        i += 16;
    }

    while i + 4 <= len {
        let diff = vsubq_f32(query(i), stored(i));
        sum0 = vfmaq_f32(sum0, diff, diff);
        i += 4;
    }

    let sum_combined = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
    let sum_pair = vpadd_f32(vget_low_f32(sum_combined), vget_high_f32(sum_combined));
    let sum_total = vpadd_f32(sum_pair, sum_pair);

    let mut total = vget_lane_f32(sum_total, 0);

    while i < len {
        let diff = a[i] - crate::half::decode(b[i]);
        total += diff * diff;
        i += 1;
    }

    total
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

    /// The half kernels are the f32 kernels with a widening load, so over the same values they
    /// must give the same bits: what an index finds can't depend on which one ran.
    #[test]
    fn test_half_kernels_give_the_bits_of_the_f32_kernels() {
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for len in (0..=70).chain([128, 960, 1536]) {
            // Any finite half, subnormals among them.
            let stored: Vec<u16> = (0..len)
                .map(|_| next() as u16)
                .map(|h| if h & 0x7c00 == 0x7c00 { h & 0x83ff } else { h })
                .collect();
            let query: Vec<f32> =
                (0..len).map(|_| (next() >> 40) as f32 / (1 << 20) as f32 - 8.0).collect();
            let widened: Vec<f32> = stored.iter().map(|&h| crate::half::decode(h)).collect();
            let bits = |d: f32| d.to_bits();

            assert_eq!(
                bits(squared_half_scalar(&query, &stored)),
                bits(squared_scalar(&query, &widened)),
                "scalar, {len}"
            );
            // SAFETY (here and below): the slices are the same length, and each kernel is used
            // only on a CPU that has it.
            let (half, full) = unsafe {
                (
                    <Portable as Kernel<u16>>::squared(&query, &stored),
                    <Portable as Kernel>::squared(&query, &widened),
                )
            };
            assert_eq!(bits(half), bits(full), "portable, {len}");
            #[cfg(target_arch = "x86_64")]
            if has_f16c() {
                let (half, full) =
                    unsafe { (F16c::squared(&query, &stored), Avx2::squared(&query, &widened)) };
                assert_eq!(bits(half), bits(full), "AVX2, {len}");
            }
            // The pointer may be to a kernel `euclidean_distance` doesn't use: on a CPU with
            // AVX2 and no F16C, the scalar one.
            let (half, full) = (
                unsafe { <u16 as Element>::kernel()(&query, &stored) },
                squared_scalar(&query, &widened).sqrt(),
            );
            assert!((half - full).abs() <= 1e-4 * full, "pointer, {len}: {half} and {full}");
        }
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
