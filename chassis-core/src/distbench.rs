//! Lab only: what one distance costs at 1,536 dimensions, by kernel, prefetch and the pages under
//! the vectors (`bench/lab/dist-plan.md`). Linux only.
//!
//! `distbench <dir> [libfaiss.so]`: the vectors' file goes in `dir`.

use crate::distance::Kernel;
use crate::prefetch::Prefetch;
use std::ffi::{CStr, c_void};
use std::hint::black_box;
use std::io::Write;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::time::Instant;

#[cfg(target_arch = "x86_64")]
type K = crate::distance::Avx2;
#[cfg(target_arch = "aarch64")]
type K = crate::distance::Portable;

const DIMS: usize = 1536;
/// dbpedia's, as in the search runs: 608 MB.
const COUNT: usize = 99_000;
/// 240 KB, in L2 everywhere.
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

/// The AVX2 build of it (`SIMDLevel` 1), the level FAISS runs at on Zen 3.
#[cfg(target_arch = "x86_64")]
const BATCH4: &CStr =
    c"_ZN5faiss18fvec_L2sqr_batch_4ILNS_9SIMDLevelE1EEEvPKfS3_S3_S3_S3_mRfS4_S4_S4_";
/// The NEON build (`SIMDLevel` 4), to try the harness on aarch64.
#[cfg(target_arch = "aarch64")]
const BATCH4: &CStr =
    c"_ZN5faiss18fvec_L2sqr_batch_4ILNS_9SIMDLevelE4EEEvPKfS3_S3_S3_S3_mRfS4_S4_S4_";

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

/// Nanoseconds a distance with Chassis's kernel, as its search computes them: each neighbor asked
/// for ahead by `prefetch`, then their distances four at a time.
///
/// # Safety
///
/// The CPU has what `K` uses.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2", enable = "fma"))]
unsafe fn chassis(q: &[f32], v: &Region, prefetch: Prefetch) -> f64 {
    let mut picks = Picks(12345, v.count);
    let mut sum = 0f32;
    let start = Instant::now();
    for _ in 0..DISTANCES / NEIGHBORS {
        let ids = picks.next();
        for &id in &ids {
            prefetch.vector(v.vector(id));
        }
        for group in ids.as_chunks::<4>().0 {
            // SAFETY: the caller's; every vector has the query's length.
            for d in unsafe { K::squared_group::<4>(q, group.map(|id| v.vector(id))) } {
                sum += d;
            }
        }
    }
    black_box(sum);
    start.elapsed().as_secs_f64() * 1e9 / DISTANCES as f64
}

/// The same with FAISS's kernel, as its search computes them: nothing asked for ahead.
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

fn load_faiss(path: &str) -> anyhow::Result<Batch4> {
    let path = std::ffi::CString::new(path)?;
    // SAFETY: a library path; its initializers are FAISS's own.
    let lib = unsafe { libc::dlopen(path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
    anyhow::ensure!(!lib.is_null(), "dlopen failed for {path:?}");
    // SAFETY: a symbol of the library just opened, which stays open.
    let symbol = unsafe { libc::dlsym(lib, BATCH4.as_ptr()) };
    anyhow::ensure!(!symbol.is_null(), "{BATCH4:?} isn't in {path:?}");
    // SAFETY: the symbol is that function, whose C++ references pass as pointers.
    Ok(unsafe { std::mem::transmute::<*mut c_void, Batch4>(symbol) })
}

/// Every kernel against an f64 reference, on groups of four of `v`.
fn check(q: &[f32], v: &Region, batch4: Option<Batch4>) -> anyhow::Result<()> {
    let mut picks = Picks(777, v.count);
    for _ in 0..100 {
        let ids = picks.next();
        for group in ids.as_chunks::<4>().0 {
            let vectors = group.map(|id| v.vector(id));
            // SAFETY: `bench` checked the CPU.
            let ours = unsafe { squared_group(q, vectors) };
            let theirs = batch4.map(|b| faiss4(q, vectors, b));
            for (j, x) in vectors.iter().enumerate() {
                let want: f64 =
                    q.iter().zip(*x).map(|(a, b)| (*a as f64 - *b as f64).powi(2)).sum();
                for got in [Some(ours[j]), theirs.map(|t| t[j])].into_iter().flatten() {
                    let error = (got as f64 - want).abs() / want;
                    anyhow::ensure!(error <= 1e-4, "{got} against {want}: relative error {error}");
                }
            }
        }
    }
    Ok(())
}

/// # Safety
///
/// As `chassis`.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2", enable = "fma"))]
unsafe fn squared_group(q: &[f32], v: [&[f32]; 4]) -> [f32; 4] {
    // SAFETY: the caller's.
    unsafe { K::squared_group::<4>(q, v) }
}

pub fn bench() -> anyhow::Result<()> {
    #[cfg(target_arch = "x86_64")]
    anyhow::ensure!(crate::distance::has_avx2(), "this CPU has no AVX2 and FMA");
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = Path::new(args.first().map_or(".", String::as_str)).to_path_buf();
    let batch4 = args.get(1).map(|p| load_faiss(p)).transpose()?;
    println!("distbench\tpolicy\t{:?}", Prefetch::detect().for_dims::<f32>(DIMS));

    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x >> 40) as f32 / (1u64 << 24) as f32
    };
    let data: Vec<f32> = (0..COUNT * DIMS).map(|_| next()).collect();
    let query: Vec<f32> = (0..DIMS).map(|_| next()).collect();

    let now = Prefetch::detect().for_dims::<f32>(DIMS);
    let mut arms: Vec<(String, Option<Prefetch>)> = vec![("now".into(), Some(now))];
    for lines in [0, 8, 16, 32, 96] {
        arms.push((format!("pf{lines}"), Some(Prefetch::into_l2(lines).for_dims::<f32>(DIMS))));
    }
    if batch4.is_some() {
        arms.push(("faiss".into(), None));
    }
    arms.push(("now-again".into(), Some(now)));

    for memory in ["cache", "anon", "anon-4k", "file", "file-huge"] {
        let region = match memory {
            "cache" => Region::anon(&data[..IN_CACHE * DIMS], None),
            "anon" => Region::anon(&data, None),
            "anon-4k" => Region::anon(&data, Some(libc::MADV_NOHUGEPAGE)),
            "file" => Region::file(&data, &dir, false)?,
            _ => Region::file(&data, &dir, true)?,
        };
        check(&query, &region, batch4)?;
        let share = region.huge_share();
        let mut best = vec![f64::MAX; arms.len()];
        for _ in 0..5 {
            for ((_, prefetch), best) in arms.iter().zip(&mut best) {
                let ns = match (prefetch, batch4) {
                    // SAFETY: checked above.
                    (Some(p), _) => unsafe { chassis(&query, &region, *p) },
                    (None, Some(b)) => faiss(&query, &region, b),
                    (None, None) => unreachable!(),
                };
                *best = best.min(ns);
            }
        }
        for ((arm, _), ns) in arms.iter().zip(best) {
            println!("distbench\t{memory}\t{share:.3}\t{arm}\t{ns:.1}");
        }
    }
    Ok(())
}
