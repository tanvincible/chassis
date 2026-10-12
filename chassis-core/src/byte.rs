//! Lab only, never merged: eight bits a component (LVQ-8), two kernels for it and their timing
//! against the half-precision kernel search uses. The plan is `bench/lab/byte-plan.md` on
//! `lab/x86`.
//!
//! A vector is kept as `x = c + lo + step * k`: `c` the center, the mean of the first vectors
//! added; `lo` and `step` its own; `k` one byte a component. A record is `lo`, `step` and
//! `|x - c|^2` as f32s, then the bytes. The query stays in f32 and is centered once. Dimensions
//! must be a multiple of 32: these kernels have no tails.

use crate::distance::Kernel;
use crate::prefetch::{Prefetch, group_width};
use std::array::from_fn;
use std::hint::black_box;
use std::time::Instant;

/// Bytes before a record's components.
pub const HEAD: usize = 12;

/// A query, and what the kernels need of it, computed once.
pub struct Query {
    full: Vec<f32>,
    centered: Vec<f32>,
    /// `|q - c|^2` and `sum(q - c)`.
    norm: f32,
    sum: f32,
}

impl Query {
    pub fn new(q: &[f32], center: &[f32]) -> Self {
        let centered: Vec<f32> = q.iter().zip(center).map(|(a, c)| a - c).collect();
        let norm = centered.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() as f32;
        let sum = centered.iter().map(|&x| f64::from(x)).sum::<f64>() as f32;
        Self { full: q.to_vec(), centered, norm, sum }
    }
}

/// `x` as a record, around `center`: the recall runs' rounding (`quant.py`'s `affine`).
pub fn encode(x: &[f32], center: &[f32], out: &mut [u8]) {
    let y: Vec<f32> = x.iter().zip(center).map(|(a, c)| a - c).collect();
    let lo = y.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = y.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let step = if hi > lo { (hi - lo) / 255.0 } else { 1.0 };
    let mut norm = 0f64;
    for (dst, &v) in out[HEAD..].iter_mut().zip(&y) {
        let k = ((v - lo) / step).round_ties_even().clamp(0.0, 255.0);
        *dst = k as u8;
        norm += (f64::from(lo) + f64::from(step) * f64::from(k)).powi(2);
    }
    out[0..4].copy_from_slice(&lo.to_le_bytes());
    out[4..8].copy_from_slice(&step.to_le_bytes());
    out[8..12].copy_from_slice(&(norm as f32).to_le_bytes());
}

#[inline(always)]
fn param(record: &[u8], at: usize) -> f32 {
    f32::from_le_bytes(record[4 * at..4 * at + 4].try_into().unwrap())
}

/// The squared distance in f64, from the query to what a record stands for.
pub fn reference(q: &[f32], center: &[f32], record: &[u8]) -> f64 {
    let (lo, step) = (f64::from(param(record, 0)), f64::from(param(record, 1)));
    q.iter()
        .zip(center)
        .zip(&record[HEAD..])
        .map(|((&q, &c), &k)| (f64::from(q) - (f64::from(c) + lo + step * f64::from(k))).powi(2))
        .sum()
}

/// A kernel as the timing loop calls it, over records (or half-precision vectors) as bytes.
pub trait Timed {
    const NAME: &'static str;
    /// Bytes a component, for the group width rule; the records' head is left out, as `main`'s
    /// rule counts components only.
    const COMPONENT: usize;
    /// # Safety
    ///
    /// Each record holds the query's dimensions, and the CPU has what the kernel uses.
    unsafe fn group<const W: usize>(q: &Query, v: [&[u8]; W]) -> [f32; W];
}

fn halves(v: &[u8]) -> &[u16] {
    // SAFETY: half-precision vectors are cut from a `Vec<u16>` at even strides.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), v.len() / 2) }
}

#[cfg(target_arch = "x86_64")]
type Half = crate::distance::F16c;
#[cfg(target_arch = "aarch64")]
type Half = crate::distance::Portable;

/// H: `main`'s half-precision kernel, alone under the width rule's 1, as search calls it.
pub struct H;
impl Timed for H {
    const NAME: &'static str = "H";
    const COMPONENT: usize = 2;
    #[inline(always)]
    unsafe fn group<const W: usize>(q: &Query, v: [&[u8]; W]) -> [f32; W] {
        let v = v.map(halves);
        // SAFETY: the caller's.
        unsafe {
            if W == 1 { [<Half as Kernel<u16>>::squared(&q.full, v[0]); W] } else { Half::squared_group(&q.full, v) }
        }
    }
}

/// H2: H again, timed as if it were another kernel. Planted: the same time.
pub struct H2;
impl Timed for H2 {
    const NAME: &'static str = "H2";
    const COMPONENT: usize = 2;
    #[inline(always)]
    unsafe fn group<const W: usize>(q: &Query, v: [&[u8]; W]) -> [f32; W] {
        // SAFETY: the caller's.
        unsafe { H::group(q, v) }
    }
}

/// HH: H twice on each vector. Planted: twice the time in cache.
pub struct HH;
impl Timed for HH {
    const NAME: &'static str = "HH";
    const COMPONENT: usize = 2;
    #[inline(always)]
    unsafe fn group<const W: usize>(q: &Query, v: [&[u8]; W]) -> [f32; W] {
        // SAFETY: the caller's.
        let (a, b) = unsafe { (H::group(q, v), H::group(q, black_box(v))) };
        from_fn(|j| a[j] + b[j])
    }
}

/// Z: the loop with no kernel, picking and asking ahead only: what the loop itself costs.
pub struct Z;
impl Timed for Z {
    const NAME: &'static str = "Z";
    const COMPONENT: usize = 2;
    #[inline(always)]
    unsafe fn group<const W: usize>(_: &Query, v: [&[u8]; W]) -> [f32; W] {
        black_box(v);
        [0.0; W]
    }
}

/// D, direct: `sum((q' - (lo + step * k))^2)`.
pub struct D;
impl Timed for D {
    const NAME: &'static str = "D";
    const COMPONENT: usize = 1;
    #[inline(always)]
    unsafe fn group<const W: usize>(q: &Query, v: [&[u8]; W]) -> [f32; W] {
        // SAFETY: the caller's.
        unsafe { direct(&q.centered, v) }
    }
}

/// E, expanded: `|q'|^2 - 2 lo sum(q') - 2 step sum(q' k) + |x - c|^2`.
pub struct E;
impl Timed for E {
    const NAME: &'static str = "E";
    const COMPONENT: usize = 1;
    #[inline(always)]
    unsafe fn group<const W: usize>(q: &Query, v: [&[u8]; W]) -> [f32; W] {
        // SAFETY: the caller's.
        let dots = unsafe { dots(&q.centered, v) };
        // Rounding can take a distance near 0 below it.
        from_fn(|j| {
            let (lo, step, norm) = (param(v[j], 0), param(v[j], 1), param(v[j], 2));
            (q.norm + norm - 2.0 * (lo * q.sum + step * dots[j])).max(0.0)
        })
    }
}

/// E in groups of four whatever the width rule says: recorded, not decided on.
pub struct E4;
impl Timed for E4 {
    const NAME: &'static str = "E4";
    const COMPONENT: usize = 1;
    #[inline(always)]
    unsafe fn group<const W: usize>(q: &Query, v: [&[u8]; W]) -> [f32; W] {
        // SAFETY: the caller's.
        unsafe { E::group(q, v) }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn widen8(p: *const u8) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    // SAFETY: the caller's: eight bytes at `p`.
    _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(unsafe { _mm_loadl_epi64(p.cast()) }))
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
fn hsum(s: [std::arch::x86_64::__m256; 4]) -> f32 {
    use std::arch::x86_64::*;
    let combined = _mm256_add_ps(_mm256_add_ps(s[0], s[1]), _mm256_add_ps(s[2], s[3]));
    let sum128 = _mm_add_ps(_mm256_castps256_ps128(combined), _mm256_extractf128_ps(combined, 1));
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    _mm_cvtss_f32(_mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0x55)))
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn direct<const W: usize>(q: &[f32], v: [&[u8]; W]) -> [f32; W] {
    use std::arch::x86_64::*;
    let lo: [__m256; W] = from_fn(|j| _mm256_set1_ps(param(v[j], 0)));
    let step: [__m256; W] = from_fn(|j| _mm256_set1_ps(param(v[j], 1)));
    let mut sums = [[_mm256_setzero_ps(); 4]; W];
    for i in (0..q.len()).step_by(32) {
        for m in 0..4 {
            // SAFETY: `i + 8 * m + 8` is within the query and, past the head, each record.
            let a = unsafe { _mm256_loadu_ps(q.as_ptr().add(i + 8 * m)) };
            for (j, sum) in sums.iter_mut().enumerate() {
                let k = unsafe { widen8(v[j].as_ptr().add(HEAD + i + 8 * m)) };
                let d = _mm256_sub_ps(a, _mm256_fmadd_ps(step[j], k, lo[j]));
                sum[m] = _mm256_fmadd_ps(d, d, sum[m]);
            }
        }
    }
    from_fn(|j| hsum(sums[j]))
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dots<const W: usize>(q: &[f32], v: [&[u8]; W]) -> [f32; W] {
    use std::arch::x86_64::*;
    let mut sums = [[_mm256_setzero_ps(); 4]; W];
    for i in (0..q.len()).step_by(32) {
        for m in 0..4 {
            // SAFETY: as in `direct`.
            let a = unsafe { _mm256_loadu_ps(q.as_ptr().add(i + 8 * m)) };
            for (j, sum) in sums.iter_mut().enumerate() {
                let k = unsafe { widen8(v[j].as_ptr().add(HEAD + i + 8 * m)) };
                sum[m] = _mm256_fmadd_ps(a, k, sum[m]);
            }
        }
    }
    from_fn(|j| hsum(sums[j]))
}

/// Sixteen bytes at `p` as four groups of four f32s.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn widen16(p: *const u8) -> [std::arch::aarch64::float32x4_t; 4] {
    use std::arch::aarch64::*;
    // SAFETY: the caller's: sixteen bytes at `p`.
    let b = unsafe { vld1q_u8(p) };
    let (low, high) = (vmovl_u8(vget_low_u8(b)), vmovl_high_u8(b));
    [
        vcvtq_f32_u32(vmovl_u16(vget_low_u16(low))),
        vcvtq_f32_u32(vmovl_high_u16(low)),
        vcvtq_f32_u32(vmovl_u16(vget_low_u16(high))),
        vcvtq_f32_u32(vmovl_high_u16(high)),
    ]
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
fn hsum(s: [std::arch::aarch64::float32x4_t; 4]) -> f32 {
    use std::arch::aarch64::*;
    let combined = vaddq_f32(vaddq_f32(s[0], s[1]), vaddq_f32(s[2], s[3]));
    let pair = vpadd_f32(vget_low_f32(combined), vget_high_f32(combined));
    vget_lane_f32(vpadd_f32(pair, pair), 0)
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn direct<const W: usize>(q: &[f32], v: [&[u8]; W]) -> [f32; W] {
    use std::arch::aarch64::*;
    let lo: [float32x4_t; W] = from_fn(|j| vdupq_n_f32(param(v[j], 0)));
    let step: [float32x4_t; W] = from_fn(|j| vdupq_n_f32(param(v[j], 1)));
    let mut sums = [[vdupq_n_f32(0.0); 4]; W];
    for i in (0..q.len()).step_by(16) {
        // SAFETY: `i + 16` is within the query and, past the head, each record.
        let a: [float32x4_t; 4] = from_fn(|m| unsafe { vld1q_f32(q.as_ptr().add(i + 4 * m)) });
        for (j, sum) in sums.iter_mut().enumerate() {
            let k = unsafe { widen16(v[j].as_ptr().add(HEAD + i)) };
            for m in 0..4 {
                let d = vsubq_f32(a[m], vfmaq_f32(lo[j], step[j], k[m]));
                sum[m] = vfmaq_f32(sum[m], d, d);
            }
        }
    }
    from_fn(|j| hsum(sums[j]))
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn dots<const W: usize>(q: &[f32], v: [&[u8]; W]) -> [f32; W] {
    use std::arch::aarch64::*;
    let mut sums = [[vdupq_n_f32(0.0); 4]; W];
    for i in (0..q.len()).step_by(16) {
        // SAFETY: as in `direct`.
        let a: [float32x4_t; 4] = from_fn(|m| unsafe { vld1q_f32(q.as_ptr().add(i + 4 * m)) });
        for (j, sum) in sums.iter_mut().enumerate() {
            let k = unsafe { widen16(v[j].as_ptr().add(HEAD + i)) };
            for m in 0..4 {
                sum[m] = vfmaq_f32(sum[m], a[m], k[m]);
            }
        }
    }
    from_fn(|j| hsum(sums[j]))
}

/// Something like an embedding: components about `1/sqrt(dims)`, two of them offset in every
/// vector, as ada-002's are.
struct Vectors(u64);
impl Vectors {
    fn next(&mut self, dims: usize, out: &mut [f32]) {
        let scale = 1.0 / (dims as f32).sqrt();
        for (i, x) in out.iter_mut().enumerate() {
            let mut u = 0f32;
            for _ in 0..4 {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                u += (self.0 >> 40) as f32 / (1u64 << 24) as f32;
            }
            *x = (u - 2.0) * 1.7 * scale + if i == 1 { -0.6 } else if i == 7 { 0.2 } else { 0.0 };
        }
    }
}

/// The checks before any timing: each kernel within 1e-4 of the f64 reference, and a group the
/// same bits as each of its vectors alone.
pub fn check() -> anyhow::Result<()> {
    for dims in [32usize, 96, 384, 768, 1536] {
        let mut vectors = Vectors(0x9E37_79B9_7F4A_7C15 ^ dims as u64);
        let (count, stride) = (1000, HEAD + dims);
        let mut raw = vec![0f32; count * dims];
        for x in raw.chunks_exact_mut(dims) {
            vectors.next(dims, x);
        }
        let mut center = vec![0f32; dims];
        for x in raw.chunks_exact(dims) {
            center.iter_mut().zip(x).for_each(|(c, v)| *c += v / count as f32);
        }
        let mut records = vec![0u8; count * stride];
        for (x, r) in raw.chunks_exact(dims).zip(records.chunks_exact_mut(stride)) {
            encode(x, &center, r);
        }
        let halves: Vec<u16> = raw.iter().map(|&x| crate::half::encode(x)).collect();
        let mut q = vec![0f32; dims];
        let mut worst = [0f64; 3];
        for _ in 0..10 {
            vectors.next(dims, &mut q);
            let query = Query::new(&q, &center);
            for v in 0..count - 4 {
                let r = |v: usize| &records[v * stride..(v + 1) * stride];
                let h = |v: usize| -> &[u8] {
                    // SAFETY: within `halves`.
                    unsafe { std::slice::from_raw_parts(halves.as_ptr().add(v * dims).cast(), 2 * dims) }
                };
                let want = reference(&q, &center, r(v));
                let half_want: f64 = q
                    .iter()
                    .zip(&halves[v * dims..(v + 1) * dims])
                    .map(|(&a, &b)| (f64::from(a) - f64::from(crate::half::decode(b))).powi(2))
                    .sum();
                // SAFETY: the checks run where the timing does, on the CPUs it needs.
                let (d1, e1, h1, d4, e4, h4) = unsafe {
                    (
                        run::<D, 1>(&query, [r(v)]),
                        run::<E, 1>(&query, [r(v)]),
                        run::<H, 1>(&query, [h(v)]),
                        run::<D, 4>(&query, from_fn(|j| r(v + j))),
                        run::<E, 4>(&query, from_fn(|j| r(v + j))),
                        run::<H, 4>(&query, from_fn(|j| h(v + j))),
                    )
                };
                for (k, (got, want)) in [(d1[0], want), (e1[0], want), (h1[0], half_want)].into_iter().enumerate() {
                    worst[k] = worst[k].max((f64::from(got) - want).abs() / want);
                }
                anyhow::ensure!(d4[0].to_bits() == d1[0].to_bits(), "D in a group: {} alone, {} in a group", d1[0], d4[0]);
                anyhow::ensure!(e4[0].to_bits() == e1[0].to_bits(), "E in a group: {} alone, {} in a group", e1[0], e4[0]);
                anyhow::ensure!(h4[0].to_bits() == h1[0].to_bits(), "H in a group: {} alone, {} in a group", h1[0], h4[0]);
            }
        }
        println!("bytecheck\t{dims}\tD\t{:.2e}\tE\t{:.2e}\tH\t{:.2e}", worst[0], worst[1], worst[2]);
        anyhow::ensure!(worst.iter().all(|&w| w <= 1e-4), "relative error past 1e-4 at {dims}: {worst:?}");
    }
    Ok(())
}

/// One group through the kernel, compiled for what it uses.
///
/// # Safety
///
/// As `Timed::group`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn run<K: Timed, const W: usize>(q: &Query, v: [&[u8]; W]) -> [f32; W] {
    // SAFETY: the caller's.
    unsafe { K::group(q, v) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn run<K: Timed, const W: usize>(q: &Query, v: [&[u8]; W]) -> [f32; W] {
    // SAFETY: the caller's.
    unsafe { K::group(q, v) }
}

/// Nanoseconds a distance: `rounds` times, 32 vectors picked at random, each asked for ahead,
/// then their distances in groups of `W`. Compiled for the kernel's instructions, so it inlines.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn timed<K: Timed, const W: usize>(q: &Query, data: &[u8], stride: usize, rounds: usize) -> f64 {
    // SAFETY: the caller's.
    unsafe { timing::<K, W>(q, data, stride, rounds) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn timed<K: Timed, const W: usize>(q: &Query, data: &[u8], stride: usize, rounds: usize) -> f64 {
    // SAFETY: the caller's.
    unsafe { timing::<K, W>(q, data, stride, rounds) }
}

#[inline(always)]
unsafe fn timing<K: Timed, const W: usize>(q: &Query, data: &[u8], stride: usize, rounds: usize) -> f64 {
    let count = data.len() / stride;
    let policy = Prefetch::detect().for_dims::<u8>(stride);
    let mut pick = 12345u64;
    let mut sum = 0f32;
    let start = Instant::now();
    for _ in 0..rounds {
        let ids: [usize; 32] = from_fn(|_| {
            pick = pick.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (pick >> 33) as usize % count
        });
        for &id in &ids {
            policy.vector(&data[id * stride..(id + 1) * stride]);
        }
        for chunk in ids.as_chunks::<W>().0 {
            let group = chunk.map(|id| &data[id * stride..(id + 1) * stride]);
            // SAFETY: the caller's.
            for d in unsafe { K::group(q, group) } {
                sum += d;
            }
        }
    }
    black_box(sum);
    start.elapsed().as_secs_f64() * 1e9 / (rounds * 32) as f64
}

fn best<K: Timed>(q: &Query, data: &[u8], stride: usize, width: usize, rounds: usize) -> f64 {
    // SAFETY: `bench` checked the CPU before any timing, and every record holds the query's
    // dimensions.
    unsafe {
        match width {
            8 => timed::<K, 8>(q, data, stride, rounds),
            4 => timed::<K, 4>(q, data, stride, rounds),
            _ => timed::<K, 1>(q, data, stride, rounds),
        }
    }
}

/// The checks, then each kernel timed in cache and out of it, as `byte-plan.md` says.
pub fn bench() -> anyhow::Result<()> {
    #[cfg(target_arch = "x86_64")]
    anyhow::ensure!(crate::distance::has_f16c(), "this CPU has no AVX2, FMA and F16C");
    check()?;
    for dims in [384usize, 768, 1536] {
        for (regime, bytes) in [("cache", 256usize << 10), ("memory", 1usize << 30)] {
            let count = bytes / (2 * dims);
            let stride = HEAD + dims;
            let mut vectors = Vectors(7 ^ dims as u64);
            let mut x = vec![0f32; dims];
            let mut center = vec![0f32; dims];
            let first = count.min(1000);
            for _ in 0..first {
                vectors.next(dims, &mut x);
                center.iter_mut().zip(&x).for_each(|(c, v)| *c += v / first as f32);
            }
            let mut vectors = Vectors(7 ^ dims as u64);
            let mut halves = vec![0u16; count * dims];
            let mut records = vec![0u8; count * stride];
            for (h, r) in halves.chunks_exact_mut(dims).zip(records.chunks_exact_mut(stride)) {
                vectors.next(dims, &mut x);
                h.iter_mut().zip(&x).for_each(|(h, &v)| *h = crate::half::encode(v));
                encode(&x, &center, r);
            }
            // SAFETY: the bytes of `halves`.
            let half_bytes = unsafe { std::slice::from_raw_parts(halves.as_ptr().cast::<u8>(), halves.len() * 2) };
            vectors.next(dims, &mut x);
            let query = Query::new(&x, &center);
            let rounds = 40_000_000 / dims / 32;
            let (wh, wb) = (group_width(dims * H::COMPONENT), group_width(dims * E::COMPONENT));
            let mut ns = [f64::MAX; 7];
            for _ in 0..3 {
                let times = [
                    best::<H>(&query, half_bytes, 2 * dims, wh, rounds),
                    best::<D>(&query, &records, stride, wb, rounds),
                    best::<E>(&query, &records, stride, wb, rounds),
                    best::<E4>(&query, &records, stride, 4, rounds),
                    best::<H2>(&query, half_bytes, 2 * dims, wh, rounds),
                    best::<HH>(&query, half_bytes, 2 * dims, wh, rounds),
                    best::<Z>(&query, half_bytes, 2 * dims, wh, rounds),
                ];
                ns.iter_mut().zip(times).for_each(|(n, t)| *n = n.min(t));
            }
            let names = [H::NAME, D::NAME, E::NAME, E4::NAME, H2::NAME, HH::NAME, Z::NAME];
            let widths = [wh, wb, wb, 4, wh, wh, wh];
            for ((name, width), time) in names.into_iter().zip(widths).zip(ns) {
                println!("bytebench\t{regime}\t{dims}\t{name}\t{width}\t{time:.1}\t{:.3}", ns[0] / time);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_kernels_agree_with_f64_and_groups_with_alone() {
        #[cfg(target_arch = "x86_64")]
        if !crate::distance::has_f16c() {
            return;
        }
        super::check().unwrap();
    }
}
