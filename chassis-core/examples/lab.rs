//! x86 lab harness (experiment branch only). Output lines are tab-separated:
//! `engine tag n what ef recall qps distances_per_query`.
//!
//! lab kernel <tag>
//! lab build <data> <dataset> <index> <n>           batch build of the first n vectors
//! lab seq <data> <dataset> <n> <tag>               one-thread build of the first n, timed
//! lab truth <data> <dataset> <n> <out>             exact top 10 among the first n
//! lab search <data> <dataset> <index> <truth> <tag>
//! lab flushes <data> <dataset> <index> <count> <tag>   add one vector and flush, count times
//! lab ceiling <data> <dataset> <n> <truth>         exact search over the data as loaded, against truth
//! lab halfbench                                    f32 and f16 kernels, in cache and out of it
//! lab cold <data> <dataset> <index> <truth> <tag>  open and search once through the queries,
//!                                                  timed from the start of the process
//!
//! LAB_ROUND16=1 rounds the stored vectors to half precision and back before anything but `truth`.

use chassis_core::{IndexOptions, IndexReader, SearchResult, VectorIndex, euclidean_distance};
use std::path::Path;
use std::time::Instant;

const K: usize = 10;
const PASSES: usize = 5;
const EF_SEARCH: [usize; 4] = [32, 64, 128, 256];

fn read<T>(path: &Path, parse: fn([u8; 4]) -> T) -> anyhow::Result<(usize, Vec<T>)> {
    let bytes = std::fs::read(path)?;
    let (header, values) = bytes.split_at(8);
    let cols = u32::from_le_bytes(header[4..8].try_into()?) as usize;
    Ok((cols, values.as_chunks::<4>().0.iter().map(|&c| parse(c)).collect()))
}

/// IEEE half precision in software: what an index storing 16-bit floats would keep of each value.
mod half16 {
    pub fn encode(x: f32) -> u16 {
        let bits = x.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32;
        let man = bits & 0x7f_ffff;
        if exp == 0xff {
            return sign | 0x7c00 | if man != 0 { 0x200 | (man >> 13) as u16 } else { 0 };
        }
        let e = exp - 127 + 15;
        if e >= 31 {
            return sign | 0x7c00;
        }
        if e <= 0 {
            if e < -10 {
                return sign;
            }
            let man = man | 0x80_0000;
            let shift = (14 - e) as u32;
            let mut half = (man >> shift) as u16;
            let (rem, halfway) = (man & ((1 << shift) - 1), 1 << (shift - 1));
            if rem > halfway || (rem == halfway && half & 1 == 1) {
                half += 1;
            }
            return sign | half;
        }
        let mut half = sign | ((e as u16) << 10) | (man >> 13) as u16;
        let rem = man & 0x1fff;
        if rem > 0x1000 || (rem == 0x1000 && half & 1 == 1) {
            half += 1;
        }
        half
    }

    pub fn decode(h: u16) -> f32 {
        let sign = u32::from(h & 0x8000) << 16;
        let exp = u32::from(h >> 10) & 0x1f;
        let mut man = u32::from(h & 0x3ff);
        f32::from_bits(match exp {
            0 if man == 0 => sign,
            0 => {
                let mut e = 0;
                while man & 0x400 == 0 {
                    man <<= 1;
                    e += 1;
                }
                sign | ((113 - e) << 23) | ((man & 0x3ff) << 13)
            }
            31 => sign | 0x7f80_0000 | (man << 13),
            _ => sign | ((exp + 112) << 23) | (man << 13),
        })
    }
}

/// Squared L2 between an f32 query and a stored vector, f32 or f16: the same loop for both, so
/// that the two differ only in what is loaded.
mod kernels {
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn sq_f32(a: &[f32], b: &[f32]) -> f32 {
        use std::arch::x86_64::*;
        let (mut s0, mut s1, mut s2, mut s3) =
            (_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps());
        let mut i = 0;
        while i + 32 <= a.len() {
            unsafe {
                let d0 = _mm256_sub_ps(
                    _mm256_loadu_ps(a.as_ptr().add(i)),
                    _mm256_loadu_ps(b.as_ptr().add(i)),
                );
                let d1 = _mm256_sub_ps(
                    _mm256_loadu_ps(a.as_ptr().add(i + 8)),
                    _mm256_loadu_ps(b.as_ptr().add(i + 8)),
                );
                let d2 = _mm256_sub_ps(
                    _mm256_loadu_ps(a.as_ptr().add(i + 16)),
                    _mm256_loadu_ps(b.as_ptr().add(i + 16)),
                );
                let d3 = _mm256_sub_ps(
                    _mm256_loadu_ps(a.as_ptr().add(i + 24)),
                    _mm256_loadu_ps(b.as_ptr().add(i + 24)),
                );
                s0 = _mm256_fmadd_ps(d0, d0, s0);
                s1 = _mm256_fmadd_ps(d1, d1, s1);
                s2 = _mm256_fmadd_ps(d2, d2, s2);
                s3 = _mm256_fmadd_ps(d3, d3, s3);
            }
            i += 32;
        }
        let v = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
        let mut out = [0f32; 8];
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), v) };
        out.iter().sum()
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn sq_f16(a: &[f32], b: &[u16]) -> f32 {
        use std::arch::x86_64::*;
        let (mut s0, mut s1, mut s2, mut s3) =
            (_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps());
        let mut i = 0;
        while i + 32 <= a.len() {
            unsafe {
                let h = |at: usize| _mm256_cvtph_ps(_mm_loadu_si128(b.as_ptr().add(at).cast()));
                let d0 = _mm256_sub_ps(_mm256_loadu_ps(a.as_ptr().add(i)), h(i));
                let d1 = _mm256_sub_ps(_mm256_loadu_ps(a.as_ptr().add(i + 8)), h(i + 8));
                let d2 = _mm256_sub_ps(_mm256_loadu_ps(a.as_ptr().add(i + 16)), h(i + 16));
                let d3 = _mm256_sub_ps(_mm256_loadu_ps(a.as_ptr().add(i + 24)), h(i + 24));
                s0 = _mm256_fmadd_ps(d0, d0, s0);
                s1 = _mm256_fmadd_ps(d1, d1, s1);
                s2 = _mm256_fmadd_ps(d2, d2, s2);
                s3 = _mm256_fmadd_ps(d3, d3, s3);
            }
            i += 32;
        }
        let v = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
        let mut out = [0f32; 8];
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), v) };
        out.iter().sum()
    }

    #[cfg(target_arch = "aarch64")]
    pub unsafe fn sq_f32(a: &[f32], b: &[f32]) -> f32 {
        use std::arch::aarch64::*;
        unsafe {
            let (mut s0, mut s1, mut s2, mut s3) =
                (vdupq_n_f32(0.0), vdupq_n_f32(0.0), vdupq_n_f32(0.0), vdupq_n_f32(0.0));
            let mut i = 0;
            while i + 16 <= a.len() {
                let d0 = vsubq_f32(vld1q_f32(a.as_ptr().add(i)), vld1q_f32(b.as_ptr().add(i)));
                let d1 =
                    vsubq_f32(vld1q_f32(a.as_ptr().add(i + 4)), vld1q_f32(b.as_ptr().add(i + 4)));
                let d2 =
                    vsubq_f32(vld1q_f32(a.as_ptr().add(i + 8)), vld1q_f32(b.as_ptr().add(i + 8)));
                let d3 =
                    vsubq_f32(vld1q_f32(a.as_ptr().add(i + 12)), vld1q_f32(b.as_ptr().add(i + 12)));
                s0 = vfmaq_f32(s0, d0, d0);
                s1 = vfmaq_f32(s1, d1, d1);
                s2 = vfmaq_f32(s2, d2, d2);
                s3 = vfmaq_f32(s3, d3, d3);
                i += 16;
            }
            vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)))
        }
    }

    #[cfg(target_arch = "aarch64")]
    pub unsafe fn sq_f16(a: &[f32], b: &[u16]) -> f32 {
        use std::arch::aarch64::*;
        use std::arch::asm;
        unsafe {
            // Half to single, four at a time: base ARMv8, but not yet a stable intrinsic.
            let h = |at: usize| -> float32x4_t {
                let half = vld1_u16(b.as_ptr().add(at));
                let single: float32x4_t;
                asm!("fcvtl {o:v}.4s, {i:v}.4h", i = in(vreg) half, o = lateout(vreg) single,
                     options(pure, nomem, nostack, preserves_flags));
                single
            };
            let (mut s0, mut s1, mut s2, mut s3) =
                (vdupq_n_f32(0.0), vdupq_n_f32(0.0), vdupq_n_f32(0.0), vdupq_n_f32(0.0));
            let mut i = 0;
            while i + 16 <= a.len() {
                let d0 = vsubq_f32(vld1q_f32(a.as_ptr().add(i)), h(i));
                let d1 = vsubq_f32(vld1q_f32(a.as_ptr().add(i + 4)), h(i + 4));
                let d2 = vsubq_f32(vld1q_f32(a.as_ptr().add(i + 8)), h(i + 8));
                let d3 = vsubq_f32(vld1q_f32(a.as_ptr().add(i + 12)), h(i + 12));
                s0 = vfmaq_f32(s0, d0, d0);
                s1 = vfmaq_f32(s1, d1, d1);
                s2 = vfmaq_f32(s2, d2, d2);
                s3 = vfmaq_f32(s3, d3, d3);
                i += 16;
            }
            vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)))
        }
    }

    /// Asks for a line into L2.
    #[inline(always)]
    pub fn hint(line: *const u8) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T1 }>(line.cast())
        };
        #[cfg(target_arch = "aarch64")]
        unsafe {
            std::arch::asm!("prfm pldl2keep, [{0}]", in(reg) line, options(nostack, preserves_flags, readonly))
        };
    }
}

/// Distances from one query to stored vectors picked at random, sixteen at a time with the first
/// lines of each asked for beforehand, as a search does: f32 against f16, on 4 KB and 2 MB pages,
/// from a set that fits in cache and from one that doesn't.
fn halfbench() -> anyhow::Result<()> {
    use memmap2::{Advice, MmapMut};
    for dims in [128usize, 384, 768, 960, 1536] {
        for (regime, bytes) in [("cache", 256usize << 10), ("memory", 1usize << 30)] {
            let count = bytes / (dims * 4);
            for pages in ["4k", "2m"] {
                let mut wide = MmapMut::map_anon(count * dims * 4)?;
                let mut narrow = MmapMut::map_anon(count * dims * 2)?;
                #[cfg(target_os = "linux")]
                for map in [&wide, &narrow] {
                    let _ = map.advise(if pages == "2m" {
                        Advice::HugePage
                    } else {
                        Advice::NoHugePage
                    });
                }
                #[cfg(not(target_os = "linux"))]
                if pages == "2m" {
                    continue;
                }
                let (full, half) = unsafe {
                    (
                        std::slice::from_raw_parts_mut(
                            wide.as_mut_ptr().cast::<f32>(),
                            count * dims,
                        ),
                        std::slice::from_raw_parts_mut(
                            narrow.as_mut_ptr().cast::<u16>(),
                            count * dims,
                        ),
                    )
                };
                let mut x = 0x9E37_79B9_7F4A_7C15u64;
                for (f, h) in full.iter_mut().zip(half.iter_mut()) {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    *f = (x >> 40) as f32 / (1u64 << 24) as f32;
                    *h = half16::encode(*f);
                }
                let query: Vec<f32> = (0..dims).map(|d| (d % 97) as f32 * 0.01).collect();
                // The f16 kernel must agree with the f32 one over the same values widened.
                for v in [0, count / 2, count - 1] {
                    let stored = &half[v * dims..(v + 1) * dims];
                    let widened: Vec<f32> = stored.iter().map(|&h| half16::decode(h)).collect();
                    let (a, b) = unsafe {
                        (kernels::sq_f16(&query, stored), kernels::sq_f32(&query, &widened))
                    };
                    anyhow::ensure!(a.to_bits() == b.to_bits(), "kernels disagree: {a} and {b}");
                }
                let rounds = (20_000_000 / dims).max(20_000) / 16;
                let lines = |bytes: usize| (bytes / 64).min(8);
                let mut best = [f64::MAX; 2];
                for _ in 0..3 {
                    for (kind, best) in best.iter_mut().enumerate() {
                        let mut pick = 12345u64;
                        let mut sum = 0f32;
                        let start = Instant::now();
                        for _ in 0..rounds {
                            let mut group = [0usize; 16];
                            for slot in &mut group {
                                pick = pick
                                    .wrapping_mul(6364136223846793005)
                                    .wrapping_add(1442695040888963407);
                                *slot = (pick >> 33) as usize % count;
                            }
                            for &v in &group {
                                let (start, bytes) = if kind == 0 {
                                    (full.as_ptr().wrapping_add(v * dims).cast::<u8>(), dims * 4)
                                } else {
                                    (half.as_ptr().wrapping_add(v * dims).cast::<u8>(), dims * 2)
                                };
                                for line in 0..lines(bytes) {
                                    kernels::hint(start.wrapping_add(line * 64));
                                }
                            }
                            for &v in &group {
                                sum += unsafe {
                                    if kind == 0 {
                                        kernels::sq_f32(&query, &full[v * dims..(v + 1) * dims])
                                    } else {
                                        kernels::sq_f16(&query, &half[v * dims..(v + 1) * dims])
                                    }
                                };
                            }
                        }
                        std::hint::black_box(sum);
                        *best =
                            best.min(start.elapsed().as_secs_f64() * 1e9 / (rounds * 16) as f64);
                    }
                }
                println!(
                    "halfbench	{regime}	{dims}	{pages}	f32_ns	{:.1}	f16_ns	{:.1}	ratio	{:.2}",
                    best[0],
                    best[1],
                    best[0] / best[1]
                );
            }
        }
    }
    Ok(())
}

fn options(ef_search: usize) -> IndexOptions {
    #[allow(unused_mut)]
    let mut options = IndexOptions {
        max_connections: 16,
        ef_construction: 200,
        ef_search,
        ..IndexOptions::default()
    };
    // LAB_HUGE=1 opens with huge pages, in a build that has them (`--cfg lab_hp`).
    #[cfg(lab_hp)]
    if std::env::var_os("LAB_HUGE").is_some() {
        options.huge_pages = true;
    }
    // LAB_HALF=1 creates and opens indexes in half precision, in a build that has it
    // (`--cfg lab_half`).
    #[cfg(lab_half)]
    if std::env::var_os("LAB_HALF").is_some() {
        options.precision = chassis_core::Precision::Half;
    }
    options
}

/// A writer's handle, or with LAB_READER=1 a reader's.
enum Handle {
    Writer(VectorIndex),
    Reader(IndexReader),
}

impl Handle {
    fn open(path: &str, dims: u32, options: IndexOptions) -> anyhow::Result<Self> {
        Ok(if std::env::var_os("LAB_READER").is_some() {
            Self::Reader(IndexReader::open(path, dims, options)?)
        } else {
            Self::Writer(VectorIndex::open(path, dims, options)?)
        })
    }

    fn search(&mut self, query: &[f32]) -> anyhow::Result<Vec<SearchResult>> {
        match self {
            Self::Writer(index) => index.search(query, K),
            Self::Reader(reader) => reader.search(query, K),
        }
    }

    fn len(&mut self) -> anyhow::Result<u64> {
        Ok(match self {
            Self::Writer(index) => index.len(),
            Self::Reader(reader) => {
                reader.refresh()?;
                reader.len()
            }
        })
    }
}

/// How much of this process is mapped with huge pages, to stderr.
fn huge_pages(tag: &str) {
    if let Ok(smaps) = std::fs::read_to_string("/proc/self/smaps_rollup") {
        let pick = |key: &str| {
            smaps
                .lines()
                .find(|l| l.starts_with(key))
                .map_or("?", |l| l[key.len()..].trim())
                .to_string()
        };
        eprintln!(
            "pages chassis {tag}: Rss {} AnonHuge {} FilePmd {} ShmemPmd {}",
            pick("Rss:"),
            pick("AnonHugePages:"),
            pick("FilePmdMapped:"),
            pick("ShmemPmdMapped:")
        );
    }
}

fn kernel(tag: &str) {
    for dims in [128usize, 960, 1536] {
        // 256 pairs: in L2 at every size here.
        let vectors: Vec<Vec<f32>> = (0..512)
            .map(|i| (0..dims).map(|d| ((i * 31 + d * 7) % 97) as f32 * 0.01).collect())
            .collect();
        let rounds = 40_000_000 / dims;
        let mut best = f64::MAX;
        for _ in 0..5 {
            let start = Instant::now();
            let mut sum = 0.0f32;
            for r in 0..rounds {
                let i = r % 256;
                sum += euclidean_distance(&vectors[i], &vectors[256 + i]);
            }
            std::hint::black_box(sum);
            best = best.min(start.elapsed().as_secs_f64() * 1e9 / rounds as f64);
        }
        println!("chassis\t{tag}\t{dims}\tkernel_ns\t0\t0\t{best:.2}\t0");
    }
}

/// Major faults, minor faults, peak resident megabytes and, on Linux, megabytes read from
/// storage, by this process so far.
#[cfg(unix)]
fn usage() -> (i64, i64, f64, f64) {
    // SAFETY: getrusage fills the struct it is given.
    let ru = unsafe {
        let mut ru: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut ru);
        ru
    };
    let resident = ru.ru_maxrss as f64 / if cfg!(target_os = "macos") { 1e6 } else { 1e3 };
    let read = std::fs::read_to_string("/proc/self/io")
        .ok()
        .and_then(|io| io.lines().find_map(|l| l.strip_prefix("read_bytes: ")?.parse::<f64>().ok()))
        .unwrap_or(0.0);
    (ru.ru_majflt as i64, ru.ru_minflt as i64, resident, read / 1e6)
}

/// From the start of the process: open the index, then each query once, with where the time went.
/// Lines are `cold engine tag metric value...`; an `at_N` line is milliseconds since the start once
/// N queries are answered, then `usage()`.
#[cfg(unix)]
fn cold(start: Instant, args: &[String]) -> anyhow::Result<()> {
    let (dir, name, tag) = (Path::new(&args[1]), &args[2], &args[5]);
    let (dims, test) = read(&dir.join(format!("{name}.test.f32")), f32::from_le_bytes)?;
    let queries: Vec<&[f32]> = test.chunks_exact(dims).collect();
    let (depth, gt) = read(Path::new(&args[4]), u32::from_le_bytes)?;
    let ef = std::env::var("LAB_EF").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
    let ms = |since: Instant| since.elapsed().as_secs_f64() * 1e3;
    let mark = |at: usize| {
        let (major, minor, resident, read) = usage();
        println!(
            "cold\tchassis\t{tag}\tat_{at}\t{:.3}\t{major}\t{minor}\t{resident:.1}\t{read:.1}",
            ms(start)
        );
    };

    let opening = Instant::now();
    let mut index = Handle::open(&args[3], dims as u32, options(ef))?;
    println!("cold\tchassis\t{tag}\topen_ms\t{:.3}", ms(opening));
    mark(0);
    let (mut each, mut hits) = (Vec::with_capacity(queries.len()), 0);
    for (i, query) in queries.iter().enumerate() {
        let asked = Instant::now();
        let found = index.search(query)?;
        each.push(ms(asked));
        let want = &gt[i * depth..i * depth + K];
        hits += found.iter().filter(|r| want.contains(&(r.id as u32))).count();
        if [1, 10, 100, 1000].contains(&(i + 1)) {
            mark(i + 1);
        }
    }
    let median = |of: &[f64]| {
        let mut sorted = of.to_vec();
        sorted.sort_by(f64::total_cmp);
        sorted.get(sorted.len() / 2).copied().unwrap_or(f64::NAN)
    };
    println!("cold\tchassis\t{tag}\tfirst_ms\t{:.3}", each[0]);
    for (from, to) in [(1, 10), (10, 100), (100, 1000)] {
        if each.len() >= to {
            println!(
                "cold\tchassis\t{tag}\tmedian_ms_{}_{to}\t{:.3}",
                from + 1,
                median(&each[from..to])
            );
        }
    }
    let again: Vec<f64> = queries
        .iter()
        .map(|q| {
            let asked = Instant::now();
            index.search(q).map(|_| ms(asked))
        })
        .collect::<Result<_, _>>()?;
    println!("cold\tchassis\t{tag}\tmedian_ms_again\t{:.3}", median(&again));
    println!("cold\tchassis\t{tag}\trecall\t{:.4}", hits as f64 / (queries.len() * K) as f64);
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let start = Instant::now();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |i: usize| args.get(i).cloned().unwrap_or_default();
    #[cfg(unix)]
    if arg(0) == "cold" {
        return cold(start, &args);
    }
    let _ = start;
    if arg(0) == "kernel" {
        kernel(&arg(1));
        return Ok(());
    }
    if arg(0) == "halfbench" {
        #[cfg(target_arch = "x86_64")]
        anyhow::ensure!(
            is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c"),
            "needs AVX2 and F16C"
        );
        return halfbench();
    }
    let dir = Path::new(&args[1]);
    let name = &args[2];
    let (dims, mut train) = read(&dir.join(format!("{name}.train.f32")), f32::from_le_bytes)?;
    if std::env::var("LAB_ROUND16").is_ok() && arg(0) != "truth" {
        for value in &mut train {
            *value = half16::decode(half16::encode(*value));
        }
    }
    let (_, test) = read(&dir.join(format!("{name}.test.f32")), f32::from_le_bytes)?;
    let queries: Vec<&[f32]> = test.chunks_exact(dims).collect();
    match arg(0).as_str() {
        "build" => {
            let n: usize = arg(4).parse()?;
            let _ = std::fs::remove_file(arg(3));
            let mut index = VectorIndex::open(arg(3), dims as u32, options(K))?;
            let start = Instant::now();
            index.add_batch(&train[..n * dims])?;
            index.flush()?;
            println!(
                "chassis\tbase\t{n}\tbuild_batch_s\t0\t0\t{:.1}\t0",
                start.elapsed().as_secs_f64()
            );
        }
        "seq" => {
            let n: usize = arg(3).parse()?;
            let path = dir.join(format!("{name}.seq.{}.chassis", arg(4).replace(['/', ','], "_")));
            let _ = std::fs::remove_file(&path);
            let mut index = VectorIndex::open(&path, dims as u32, options(K))?;
            let start = Instant::now();
            for vector in train[..n * dims].chunks_exact(dims) {
                index.add(vector)?;
            }
            index.flush()?;
            println!(
                "chassis\t{}\t{n}\tbuild_seq_s\t0\t0\t{:.1}\t0",
                arg(4),
                start.elapsed().as_secs_f64()
            );
            std::fs::remove_file(&path)?;
        }
        "ceiling" => {
            let n: usize = arg(3).parse()?;
            let (depth, gt) = read(Path::new(&arg(4)), u32::from_le_bytes)?;
            let (mut hits, mut top1) = (0usize, 0usize);
            for (query, row) in queries.iter().zip(gt.chunks_exact(depth)) {
                let mut all: Vec<(f32, u32)> = train[..n * dims]
                    .chunks_exact(dims)
                    .enumerate()
                    .map(|(id, v)| (euclidean_distance(query, v), id as u32))
                    .collect();
                all.select_nth_unstable_by(K, |a, b| a.0.total_cmp(&b.0));
                all.truncate(K);
                all.sort_by(|a, b| a.0.total_cmp(&b.0));
                hits += all.iter().filter(|(_, id)| row[..K].contains(id)).count();
                top1 += usize::from(all[0].1 == row[0]);
            }
            let largest = train[..n * dims].iter().fold(0f32, |m, v| m.max(v.abs()));
            let not_finite = train[..n * dims].iter().filter(|v| !v.is_finite()).count();
            println!(
                "ceiling\t{name}\t{n}\trecall10\t{:.5}\ttop1\t{:.5}\tlargest\t{largest}\tnot_finite\t{not_finite}",
                hits as f64 / (queries.len() * K) as f64,
                top1 as f64 / queries.len() as f64
            );
        }
        "truth" => {
            let n: usize = arg(3).parse()?;
            let mut out = Vec::new();
            out.extend_from_slice(&(queries.len() as u32).to_le_bytes());
            out.extend_from_slice(&(K as u32).to_le_bytes());
            for query in &queries {
                let mut all: Vec<(f32, u32)> = train[..n * dims]
                    .chunks_exact(dims)
                    .enumerate()
                    .map(|(id, v)| (euclidean_distance(query, v), id as u32))
                    .collect();
                all.select_nth_unstable_by(K, |a, b| a.0.total_cmp(&b.0));
                all.truncate(K);
                all.sort_by(|a, b| a.0.total_cmp(&b.0));
                out.extend(all.iter().flat_map(|(_, id)| id.to_le_bytes()));
            }
            std::fs::write(arg(4), out)?;
        }
        "search" => {
            let (depth, gt) = read(Path::new(&arg(4)), u32::from_le_bytes)?;
            let truth: Vec<&[u32]> = gt.chunks_exact(depth).map(|row| &row[..K]).collect();
            let tag = arg(5);
            // LAB_EF=<ef> searches at one ef only; LAB_PASSES=<n> times more passes (profiling).
            let only: Option<usize> = std::env::var("LAB_EF").ok().and_then(|v| v.parse().ok());
            let passes_wanted: usize =
                std::env::var("LAB_PASSES").ok().and_then(|v| v.parse().ok()).unwrap_or(PASSES);
            for ef_search in EF_SEARCH.into_iter().filter(|&ef| only.is_none_or(|o| o == ef)) {
                let mut index = Handle::open(&arg(3), dims as u32, options(ef_search))?;
                let n = index.len()?;
                #[cfg(all(lab, target_os = "linux"))]
                if std::env::var("LAB_MADV").is_ok()
                    && let Handle::Writer(index) = &index
                {
                    index.lab_huge();
                }
                // The first pass over a file that may not be in memory yet.
                let first = Instant::now();
                for query in &queries {
                    index.search(query)?;
                }
                let first = first.elapsed().as_secs_f64();
                println!("chassis\t{tag}\t{n}\tfirst_pass_s\t{ef_search}\t0\t{first:.3}\t0");
                #[cfg(lab)]
                let counted = chassis_core::lab::distances();
                #[cfg(lab)]
                let (hops, scans) = chassis_core::lab::hops_scans();
                let mut passes = Vec::with_capacity(passes_wanted);
                let mut results = Vec::new();
                for _ in 0..passes_wanted {
                    let start = Instant::now();
                    results = queries.iter().map(|q| index.search(q)).collect::<Result<_, _>>()?;
                    passes.push(queries.len() as f64 / start.elapsed().as_secs_f64());
                }
                let searches = (passes_wanted * queries.len()) as f64;
                #[cfg(lab)]
                let (per_query, hops, scans) = {
                    let (h, s) = chassis_core::lab::hops_scans();
                    (
                        (chassis_core::lab::distances() - counted) as f64 / searches,
                        (h - hops) as f64 / searches,
                        (s - scans) as f64 / searches,
                    )
                };
                #[cfg(not(lab))]
                let (per_query, hops, scans) = (0.0, 0.0, 0.0);
                passes.sort_by(f64::total_cmp);
                let hits: usize = results
                    .iter()
                    .zip(&truth)
                    .map(|(found, want)| {
                        found.iter().filter(|r| want.contains(&(r.id as u32))).count()
                    })
                    .sum();
                let recall = hits as f64 / (queries.len() * K) as f64;
                println!(
                    "chassis\t{tag}\t{n}\tsearch\t{ef_search}\t{recall:.4}\t{:.0}\t{per_query:.0}\t{hops:.0}\t{scans:.0}",
                    passes[passes_wanted / 2]
                );
                if ef_search == 256 {
                    huge_pages(&tag);
                }
            }
        }
        "flushes" => {
            let count: usize = arg(4).parse()?;
            let mut index = VectorIndex::open(arg(3), dims as u32, options(K))?;
            let n = index.len();
            let mut times = Vec::with_capacity(count);
            for i in 0..count {
                let start = Instant::now();
                index.add(queries[i % queries.len()])?;
                index.flush()?;
                times.push(start.elapsed().as_secs_f64() * 1e3);
            }
            times.sort_by(f64::total_cmp);
            let mean = times.iter().sum::<f64>() / count as f64;
            println!(
                "chassis\t{}\t{n}\tflush_ms\t0\t{:.3}\t{mean:.3}\t{:.3}",
                arg(5),
                times[count / 2],
                times[count * 9 / 10]
            );
        }
        other => anyhow::bail!("unknown mode {other}"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::half16::{decode, encode};

    #[test]
    fn half_round_trips_every_half_and_rounds_to_nearest_even() {
        for h in 0..=u16::MAX {
            let x = decode(h);
            if x.is_nan() {
                assert!(decode(encode(x)).is_nan());
            } else {
                assert_eq!(encode(x), h, "{h:#06x} {x}");
            }
        }
        // Halfway between two halves goes to the even one; just past it goes up.
        let (a, b) = (decode(0x3c00), decode(0x3c01));
        assert_eq!(encode((a + b) / 2.0), 0x3c00);
        assert_eq!(encode(f32::from_bits(((a + b) / 2.0).to_bits() + 1)), 0x3c01);
        let (a, b) = (decode(0x3c01), decode(0x3c02));
        assert_eq!(encode((a + b) / 2.0), 0x3c02);
        // Subnormal halves, the largest half, and past it.
        assert_eq!(encode(decode(0x0001) / 2.0), 0x0000);
        assert_eq!(encode(decode(0x0001) * 1.5), 0x0002);
        assert_eq!(encode(65504.0), 0x7bff);
        assert_eq!(encode(65520.0), 0x7c00);
        assert_eq!(encode(-1e9), 0xfc00);
        assert_eq!(encode(1e-9), 0);
    }
}
