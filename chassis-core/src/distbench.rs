//! Lab only: what one distance costs at 1,536 dimensions with AVX2 and with AVX-512, Chassis's
//! kernels and FAISS's (`bench/lab/avx512-plan.md`). Linux on x86 only.
//!
//! `distbench <dir> <libfaiss.so>`: the vectors' file goes in `dir`.

use crate::distance::{Avx2, Avx512, Kernel};
use crate::prefetch::{Prefetch, group_width};
use std::ffi::{CStr, c_void};
use std::hint::black_box;
use std::io::Write;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::time::Instant;

const DIMS: usize = 1536;
/// dbpedia's, as in the search runs: 608 MB.
const COUNT: usize = 99_000;
/// 240 KB, which L2 holds.
const IN_CACHE: usize = 40;
const NEIGHBORS: usize = 32;
const DISTANCES: usize = 2_000_000;
const HUGE: usize = 2 << 20;

/// FAISS's `fvec_L2sqr_batch_4`, which its HNSW search calls for four neighbors at a time.
type Batch4 = unsafe extern "C" fn(
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    usize,
    *mut f32,
    *mut f32,
    *mut f32,
    *mut f32,
);

/// Its AVX2 build (`SIMDLevel` 1) and its AVX-512 one (2), both in FAISS 1.15.1's `libfaiss.so`.
const FAISS_AVX2: &CStr =
    c"_ZN5faiss18fvec_L2sqr_batch_4ILNS_9SIMDLevelE1EEEvPKfS3_S3_S3_S3_mRfS4_S4_S4_";
const FAISS_AVX512: &CStr =
    c"_ZN5faiss18fvec_L2sqr_batch_4ILNS_9SIMDLevelE2EEEvPKfS3_S3_S3_S3_mRfS4_S4_S4_";

/// Vectors in one kind of memory.
struct Region {
    at: *const f32,
    count: usize,
    anon: Option<(*mut c_void, usize)>,
    _file: Option<memmap2::MmapMut>,
}

impl Region {
    /// Anonymous memory, as FAISS's vectors are in, with `advice` for its pages if any.
    fn anon(data: &[f32], advice: Option<libc::c_int>) -> Self {
        let bytes = std::mem::size_of_val(data);
        let len = bytes + HUGE;
        // SAFETY: a fresh private mapping; nothing else refers to it.
        let map = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(map, libc::MAP_FAILED, "mmap failed");
        let at = (map as usize).next_multiple_of(HUGE) as *mut f32;
        if let Some(advice) = advice {
            // SAFETY: inside the mapping.
            assert_eq!(unsafe { libc::madvise(at.cast(), bytes, advice) }, 0, "madvise failed");
        }
        // SAFETY: `bytes` fit after `at`, inside the mapping.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), at, data.len()) };
        Self { at, count: data.len() / DIMS, anon: Some((map, len)), _file: None }
    }

    /// A file mapped shared and writable, as a writer maps its index, read from the disk after
    /// `advise` and before any timing. With `huge`, advised as Chassis's `huge_pages` does.
    fn file(data: &[f32], dir: &Path, huge: bool) -> anyhow::Result<Self> {
        let path = dir.join("vectors.f32");
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;
        // SAFETY: the bytes of `data`.
        file.write_all(unsafe {
            std::slice::from_raw_parts(data.as_ptr().cast(), size_of_val(data))
        })?;
        file.sync_all()?;
        // Out of memory, so the pages come back under the mapping's advice.
        // SAFETY: a valid descriptor.
        unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
        // SAFETY: the lab owns the file and nothing else writes it.
        let map = unsafe { memmap2::MmapOptions::new().map_mut(&file)? };
        if huge {
            map.advise(memmap2::Advice::HugePage)?;
            map.advise(memmap2::Advice::Random)?;
        }
        let mut touched = 0u8;
        for page in (0..map.len()).step_by(4096) {
            touched ^= map[page];
        }
        black_box(touched);
        Ok(Self { at: map.as_ptr().cast(), count: data.len() / DIMS, anon: None, _file: Some(map) })
    }

    fn vector(&self, id: usize) -> &[f32] {
        assert!(id < self.count);
        // SAFETY: in bounds, and the memory lives as long as `self`.
        unsafe { std::slice::from_raw_parts(self.at.add(id * DIMS), DIMS) }
    }

    /// The share of this region's resident memory on huge pages, from `/proc/self/smaps`.
    fn huge_share(&self) -> f64 {
        let smaps = std::fs::read_to_string("/proc/self/smaps").unwrap_or_default();
        let at = self.at as usize;
        let (mut inside, mut rss, mut huge) = (false, 0u64, 0u64);
        let kb = |v: &str| v.trim().trim_end_matches("kB").trim().parse::<u64>().unwrap_or(0);
        for line in smaps.lines() {
            let range = line.split_whitespace().next().and_then(|r| r.split_once('-'));
            if let Some((Ok(lo), Ok(hi))) =
                range.map(|(lo, hi)| (usize::from_str_radix(lo, 16), usize::from_str_radix(hi, 16)))
            {
                inside = (lo..hi).contains(&at);
            } else if inside {
                if let Some(v) = line.strip_prefix("Rss:") {
                    rss += kb(v);
                } else if let Some(v) = line
                    .strip_prefix("AnonHugePages:")
                    .or_else(|| line.strip_prefix("FilePmdMapped:"))
                {
                    huge += kb(v);
                }
            }
        }
        huge as f64 / rss.max(1) as f64
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        if let Some((map, len)) = self.anon {
            // SAFETY: the mapping `anon` made, which nothing uses past here.
            unsafe { libc::munmap(map, len) };
        }
    }
}

/// A node's neighbors: `NEIGHBORS` vectors at random, the same sequence for every arm.
struct Picks(u64, usize);

impl Picks {
    fn next(&mut self) -> [usize; NEIGHBORS] {
        std::array::from_fn(|_| {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) as usize % self.1
        })
    }
}

/// One way of computing distances.
#[derive(Clone, Copy)]
enum Arm {
    Avx2,
    Avx512,
    Faiss(Batch4),
}

/// Nanoseconds a distance with Chassis's kernel `K`, as its search computes them: each neighbor
/// asked for ahead by `prefetch`, then their distances `W` at a time.
///
/// # Safety
///
/// The CPU has what `K` uses.
#[inline(always)]
unsafe fn chassis<K: Kernel, const W: usize>(q: &[f32], v: &Region, prefetch: Prefetch) -> f64 {
    let mut picks = Picks(12345, v.count);
    let mut sum = 0f32;
    let start = Instant::now();
    for _ in 0..DISTANCES / NEIGHBORS {
        let ids = picks.next();
        for &id in &ids {
            prefetch.vector(v.vector(id));
        }
        for group in ids.as_chunks::<W>().0 {
            // SAFETY: the caller's; every vector has the query's length.
            for d in unsafe { K::squared_group::<W>(q, group.map(|id| v.vector(id))) } {
                sum += d;
            }
        }
    }
    black_box(sum);
    start.elapsed().as_secs_f64() * 1e9 / DISTANCES as f64
}

/// `chassis` with `Avx2` inlined, as a search compiled for it runs it.
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn chassis_avx2<const W: usize>(q: &[f32], v: &Region, prefetch: Prefetch) -> f64 {
    // SAFETY: the caller's.
    unsafe { chassis::<Avx2, W>(q, v, prefetch) }
}

/// `chassis` with `Avx512` inlined.
#[target_feature(enable = "avx512f", enable = "avx2", enable = "fma")]
unsafe fn chassis_avx512<const W: usize>(q: &[f32], v: &Region, prefetch: Prefetch) -> f64 {
    // SAFETY: the caller's.
    unsafe { chassis::<Avx512, W>(q, v, prefetch) }
}

/// The same with FAISS's kernel, as its search computes them: four at a time, nothing asked for
/// ahead.
fn faiss(q: &[f32], v: &Region, batch4: Batch4) -> f64 {
    let mut picks = Picks(12345, v.count);
    let mut sum = 0f32;
    let start = Instant::now();
    for _ in 0..DISTANCES / NEIGHBORS {
        let ids = picks.next();
        for group in ids.as_chunks::<4>().0 {
            sum += faiss4(q, group.map(|id| v.vector(id)), batch4).iter().sum::<f32>();
        }
    }
    black_box(sum);
    start.elapsed().as_secs_f64() * 1e9 / DISTANCES as f64
}

fn faiss4(q: &[f32], v: [&[f32]; 4], batch4: Batch4) -> [f32; 4] {
    let [mut d0, mut d1, mut d2, mut d3] = [0f32; 4];
    // SAFETY: five vectors of `DIMS` and four outputs, as FAISS's signature takes them.
    unsafe {
        batch4(
            q.as_ptr(),
            v[0].as_ptr(),
            v[1].as_ptr(),
            v[2].as_ptr(),
            v[3].as_ptr(),
            DIMS,
            &mut d0,
            &mut d1,
            &mut d2,
            &mut d3,
        )
    };
    [d0, d1, d2, d3]
}

fn load_faiss(path: &str, name: &CStr) -> anyhow::Result<Batch4> {
    let path = std::ffi::CString::new(path)?;
    // SAFETY: a library path; its initializers are FAISS's own.
    let lib = unsafe { libc::dlopen(path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
    anyhow::ensure!(!lib.is_null(), "dlopen failed for {path:?}");
    // SAFETY: a symbol of the library just opened, which stays open.
    let symbol = unsafe { libc::dlsym(lib, name.as_ptr()) };
    anyhow::ensure!(!symbol.is_null(), "{name:?} isn't in {path:?}");
    // SAFETY: the symbol is that function, whose C++ references pass as pointers.
    Ok(unsafe { std::mem::transmute::<*mut c_void, Batch4>(symbol) })
}

/// Four distances by `arm`.
///
/// # Safety
///
/// The CPU has what `arm` uses.
unsafe fn four(arm: Arm, q: &[f32], v: [&[f32]; 4]) -> [f32; 4] {
    // SAFETY: the caller's.
    unsafe {
        match arm {
            Arm::Avx2 => Avx2::squared_group::<4>(q, v),
            Arm::Avx512 => Avx512::squared_group::<4>(q, v),
            Arm::Faiss(batch4) => faiss4(q, v, batch4),
        }
    }
}

/// Every arm against an f64 reference, on groups of four of `v`.
///
/// # Safety
///
/// As `four`, for every arm.
unsafe fn check(q: &[f32], v: &Region, arms: &[(&str, Arm)]) -> anyhow::Result<()> {
    let mut picks = Picks(777, v.count);
    for _ in 0..100 {
        let ids = picks.next();
        for group in ids.as_chunks::<4>().0 {
            let vectors = group.map(|id| v.vector(id));
            for &(name, arm) in arms {
                // SAFETY: the caller's.
                let got = unsafe { four(arm, q, vectors) };
                for (x, got) in vectors.iter().zip(got) {
                    let want: f64 =
                        q.iter().zip(*x).map(|(a, b)| (*a as f64 - *b as f64).powi(2)).sum();
                    let error = (got as f64 - want).abs() / want;
                    anyhow::ensure!(error <= 1e-4, "{name}: {got} against {want}, error {error}");
                }
            }
        }
    }
    Ok(())
}

/// Nanoseconds a distance by `arm`, `W` at a time for Chassis's.
///
/// # Safety
///
/// As `four`.
unsafe fn time<const W: usize>(arm: Arm, q: &[f32], v: &Region, prefetch: Prefetch) -> f64 {
    // SAFETY: the caller's.
    unsafe {
        match arm {
            Arm::Avx2 => chassis_avx2::<W>(q, v, prefetch),
            Arm::Avx512 => chassis_avx512::<W>(q, v, prefetch),
            Arm::Faiss(batch4) => faiss(q, v, batch4),
        }
    }
}

pub fn bench() -> anyhow::Result<()> {
    anyhow::ensure!(crate::distance::has_avx2(), "this CPU has no AVX2 and FMA");
    let avx512 = is_x86_feature_detected!("avx512f");
    let args: Vec<String> = std::env::args().skip(1).collect();
    anyhow::ensure!(args.len() == 2, "usage: distbench <dir> <libfaiss.so>");
    let dir = Path::new(&args[0]);
    let prefetch = Prefetch::detect().for_dims::<f32>(DIMS);
    let width = group_width(DIMS * 4);
    println!("distbench\tpolicy\t{prefetch:?}\twidth {width}\tavx512f {avx512}");

    let mut arms = vec![("avx2", Arm::Avx2)];
    if avx512 {
        arms.push(("avx512", Arm::Avx512));
    }
    arms.push(("faiss-avx2", Arm::Faiss(load_faiss(&args[1], FAISS_AVX2)?)));
    if avx512 {
        arms.push(("faiss-avx512", Arm::Faiss(load_faiss(&args[1], FAISS_AVX512)?)));
    }
    arms.push(("avx2-again", Arm::Avx2));

    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x >> 40) as f32 / (1u64 << 24) as f32
    };
    let data: Vec<f32> = (0..COUNT * DIMS).map(|_| next()).collect();
    let query: Vec<f32> = (0..DIMS).map(|_| next()).collect();

    for memory in ["cache", "anon", "file-huge"] {
        let region = match memory {
            "cache" => Region::anon(&data[..IN_CACHE * DIMS], None),
            "anon" => Region::anon(&data, None),
            _ => Region::file(&data, dir, true)?,
        };
        // SAFETY: only arms this CPU can run were added.
        unsafe { check(&query, &region, &arms)? };
        let share = region.huge_share();
        let mut best = vec![f64::MAX; arms.len()];
        for _ in 0..5 {
            for (&(_, arm), best) in arms.iter().zip(&mut best) {
                // SAFETY: as above.
                let ns = unsafe {
                    match width {
                        8 => time::<8>(arm, &query, &region, prefetch),
                        _ => time::<4>(arm, &query, &region, prefetch),
                    }
                };
                *best = best.min(ns);
            }
        }
        for ((name, _), ns) in arms.iter().zip(best) {
            println!("distbench\t{memory}\t{share:.3}\t{name}\t{ns:.1}");
        }
    }
    Ok(())
}
