//! Chassis at ten million vectors (bench/lab/scale-plan.md on lab/x86). Each command takes one
//! step with an index of synthetic vectors and prints what it found as `scale\t<step>\t<what>\t
//! <value>` lines. Vectors are computed from their ids, so the exact nearest neighbors never come
//! from the file under test.
//!
//! `cargo run --release --example scale -- <command> <path> [args]`:
//!
//! - `build <path> <n>`: ids 0..n, a million at a time, each flushed
//! - `exact <n> <churned> <out>`: the exact nearest live ids of the first queries, to `out`
//! - `search <path> <ef> <exact> <cold|warm|random> <queries>`: a reader's open, queries a second,
//!   recall, the disk's reads and a digest of the results; `random` advises random access first
//! - `graph <path>`: a digest of every slot's id and lists as a search reads them
//! - `check <path>`: a writer's open and `check`
//! - `delete <path> <n>`: deletes every hundredth id below `n`
//! - `add <path> <from> <to>`: adds ids `from..to`, ten thousand at a time, each flushed
//! - `backup <path> <dest>`
//! - `crash <path> <n>`: adds a million more without flushing, to be killed
//! - `verify <path> <len>`: a writer's open after the kill, its length and `check`
//! - `compact <path>`
//!
//! `SCALE_DIMS`, `SCALE_HALF=1`, `SCALE_HUGE=1` and `SCALE_CHUNK` change the dimensions, precision,
//! huge pages and vectors a build adds at a time.

use chassisdb::{
    HnswGraph, HnswParams, IndexOptions, IndexReader, Precision, Storage, VectorIndex,
    euclidean_distance,
};
use std::time::Instant;

const LATENT: usize = 16;
const CLUSTERS: u64 = 1000;
const K: usize = 10;
/// Queries whose exact neighbors `exact` finds: the first ones.
const EXACT: usize = 200;
const FIRST_QUERY: u64 = 1 << 40;
const ADDED: u64 = 100_000;
const CHUNK: u64 = 1_000_000;

fn setting(name: &str, default: u64) -> u64 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn dims() -> usize {
    setting("SCALE_DIMS", 128) as usize
}

fn options() -> IndexOptions {
    IndexOptions {
        precision: if setting("SCALE_HALF", 0) == 1 { Precision::Half } else { Precision::Full },
        huge_pages: (setting("SCALE_HUGE", 0) == 1).then_some(true),
        ..IndexOptions::default()
    }
}

const FNV: u64 = 0xCBF2_9CE4_8422_2325;

fn fnv(digest: u64, word: u64) -> u64 {
    (digest ^ word).wrapping_mul(0x100_0000_01B3)
}

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Gaussian numbers from a seed, by Box-Muller.
struct Gaussian(u64);

impl Gaussian {
    fn next(&mut self) -> f32 {
        let mut uniform = || {
            self.0 = mix(self.0);
            ((self.0 >> 11) as f64 + 1.0) / (1u64 << 53) as f64
        };
        let (a, b) = (uniform(), uniform());
        ((-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()) as f32
    }
}

/// Points around a thousand centers in 16 dimensions, projected to `dims()` with a little noise:
/// like embeddings, far fewer dimensions than they're stored in.
struct Data {
    projection: Vec<f32>,
    centers: Vec<f32>,
}

impl Data {
    fn new() -> Self {
        let mut g = Gaussian(0xC4A5_5150);
        let scale = 1.0 / (LATENT as f32).sqrt();
        Self {
            projection: (0..dims() * LATENT).map(|_| g.next() * scale).collect(),
            centers: (0..CLUSTERS as usize * LATENT).map(|_| g.next()).collect(),
        }
    }

    fn vector(&self, id: u64) -> Vec<f32> {
        let mut g = Gaussian(id);
        let center = (mix(id ^ 0x5CA1E) % CLUSTERS) as usize * LATENT;
        let z: Vec<f32> =
            self.centers[center..center + LATENT].iter().map(|c| c + 0.3 * g.next()).collect();
        let row = |r: &[f32]| r.iter().zip(&z).map(|(p, z)| p * z).sum::<f32>();
        self.projection.as_chunks::<LATENT>().0.iter().map(|r| row(r) + 0.05 * g.next()).collect()
    }
}

/// Whether `id` is in the index, before or after `churn`.
fn live(id: u64, n: u64, churned: bool) -> bool {
    match churned {
        false => id < n,
        true => (id < n && !id.is_multiple_of(100)) || (n..n + ADDED).contains(&id),
    }
}

fn out(step: &str, what: &str, value: impl std::fmt::Display) {
    println!("scale\t{step}\t{what}\t{value}");
}

fn secs(since: Instant) -> String {
    format!("{:.2}", since.elapsed().as_secs_f64())
}

/// The number after `<name>:` in `/proc/self/<file>`; 0 elsewhere than Linux.
fn proc(file: &str, name: &str) -> u64 {
    let text = std::fs::read_to_string(format!("/proc/self/{file}")).unwrap_or_default();
    let value =
        |l: &str| l.strip_prefix(name)?.strip_prefix(':')?.split_whitespace().next()?.parse().ok();
    text.lines().find_map(value).unwrap_or(0)
}

/// This process's memory, other than the file's pages: what it allocated.
fn anon_mb() -> u64 {
    proc("status", "RssAnon") / 1024
}

fn major_faults() -> u64 {
    // SAFETY: getrusage only writes the struct it is given.
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: as above.
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    usage.ru_majflt as u64
}

/// Advises random access on this process's mappings of `path`, so a fault reads no more than it
/// needs; returns the bytes advised. Linux only: it reads `/proc/self/maps`.
fn advise_random(path: &str) -> anyhow::Result<usize> {
    let path = std::fs::canonicalize(path)?.to_string_lossy().into_owned();
    let mut advised = 0;
    for line in std::fs::read_to_string("/proc/self/maps")?.lines().filter(|l| l.ends_with(&path)) {
        let (start, end) =
            line.split(' ').next().and_then(|r| r.split_once('-')).unwrap_or_default();
        let (start, end) = (usize::from_str_radix(start, 16)?, usize::from_str_radix(end, 16)?);
        // SAFETY: advice only, on a range this process maps.
        let r =
            unsafe { libc::madvise(start as *mut libc::c_void, end - start, libc::MADV_RANDOM) };
        anyhow::ensure!(r == 0, "madvise: {}", std::io::Error::last_os_error());
        advised += end - start;
    }
    anyhow::ensure!(advised > 0, "{path} isn't mapped");
    Ok(advised)
}

fn writer(path: &str) -> anyhow::Result<VectorIndex> {
    let start = Instant::now();
    let index = VectorIndex::open(path, dims() as u32, options())?;
    out("open", "writer_s", secs(start));
    Ok(index)
}

fn check(step: &str, index: &VectorIndex) -> anyhow::Result<()> {
    let start = Instant::now();
    let check = index.check()?;
    out(step, "check_s", secs(start));
    out(step, "unreachable", check.unreachable.len());
    out(step, "not_finite", check.not_finite.len());
    out(step, "len", index.len());
    Ok(())
}

fn exact(data: &Data, n: u64, churned: bool) -> Vec<Vec<u64>> {
    let queries: Vec<Vec<f32>> = (0..EXACT as u64).map(|q| data.vector(FIRST_QUERY + q)).collect();
    let top = if churned { n + ADDED } else { n };
    let threads = std::thread::available_parallelism().map_or(1, |t| t.get()) as u64;
    let parts: Vec<Vec<Vec<(f32, u64)>>> = std::thread::scope(|scope| {
        let queries = &queries;
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                scope.spawn(move || {
                    let mut best: Vec<Vec<(f32, u64)>> = vec![Vec::new(); EXACT];
                    for id in top * t / threads..top * (t + 1) / threads {
                        if !live(id, n, churned) {
                            continue;
                        }
                        let vector = data.vector(id);
                        for (best, query) in best.iter_mut().zip(queries) {
                            let d = euclidean_distance(query, &vector);
                            if best.len() < K || d < best[K - 1].0 {
                                let at = best.partition_point(|b| b.0 <= d);
                                best.insert(at, (d, id));
                                best.truncate(K);
                            }
                        }
                    }
                    best
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("a brute-force thread panicked")).collect()
    });
    (0..EXACT)
        .map(|q| {
            let mut all: Vec<(f32, u64)> = parts.iter().flat_map(|p| p[q].clone()).collect();
            all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            all.into_iter().take(K).map(|(_, id)| id).collect()
        })
        .collect()
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |i: usize| args.get(i).map(String::as_str).unwrap_or_default();
    let number = |i: usize| arg(i).parse::<u64>();
    let data = Data::new();
    match arg(0) {
        "build" => {
            let (path, n) = (arg(1), number(2)?);
            let mut index = writer(path)?;
            let (mut peak, chunk) = (0, setting("SCALE_CHUNK", CHUNK));
            #[cfg(lab)]
            let mut counted = [0u64; 3];
            for start in (0..n).step_by(chunk as usize) {
                let end = (start + chunk).min(n);
                let ids: Vec<u64> = (start..end).collect();
                let vectors: Vec<f32> = ids.iter().flat_map(|&id| data.vector(id)).collect();
                let at = Instant::now();
                index.add_batch_with_ids(&ids, &vectors)?;
                out("build", &format!("add_{end}_s"), secs(at));
                let at = Instant::now();
                index.flush()?;
                out("build", &format!("flush_{end}_s"), secs(at));
                out("build", &format!("page_tables_{end}_kb"), proc("status", "VmPTE"));
                // With `--cfg lab --cfg lab_count`, the distances and hops this chunk's adds took.
                #[cfg(lab)]
                {
                    let counts = chassisdb::lab::counts();
                    for (name, (now, before)) in
                        ["distances", "hops", "selected"].iter().zip(counts.iter().zip(counted))
                    {
                        out("build", &format!("{name}_{end}"), now - before);
                    }
                    counted = counts;
                }
                peak = peak.max(anon_mb());
            }
            out("build", "anon_peak_mb", peak);
            out("build", "huge_mapped_kb", proc("smaps_rollup", "FilePmdMapped"));
            out("build", "file_bytes", std::fs::metadata(path)?.len());
        }
        "exact" => {
            let (n, churned, to) = (number(1)?, arg(2) == "churned", arg(3));
            let start = Instant::now();
            let ids = exact(&data, n, churned);
            let lines: Vec<String> = ids
                .iter()
                .map(|ids| ids.iter().map(u64::to_string).collect::<Vec<_>>().join(" "))
                .collect();
            std::fs::write(to, lines.join("\n"))?;
            out("exact", "s", secs(start));
        }
        "search" => {
            let (path, ef, truth, mode) = (arg(1), number(2)? as usize, arg(3), arg(4));
            let count = number(5)?;
            let truth: Vec<Vec<u64>> = std::fs::read_to_string(truth)?
                .lines()
                .map(|l| l.split(' ').map(|id| id.parse().unwrap()).collect())
                .collect();
            let queries: Vec<Vec<f32>> = (0..count).map(|q| data.vector(FIRST_QUERY + q)).collect();
            let start = Instant::now();
            let options = IndexOptions { ef_search: ef, ..IndexOptions::default() };
            let mut reader = IndexReader::open(path, dims() as u32, options)?;
            out("search", "reader_open_s", secs(start));
            match mode {
                "warm" => reader.warm(),
                "random" => out("search", "random_bytes", advise_random(path)?),
                _ => {}
            }
            // Twice: the first pass reads the file in, unless it is already in memory.
            for pass in 1..=2 {
                let (mut digest, mut found) = (FNV, 0);
                let (start, faults, read) =
                    (Instant::now(), major_faults(), proc("io", "read_bytes"));
                for (q, query) in queries.iter().enumerate() {
                    let results = reader.search(query, K)?;
                    for r in &results {
                        digest = fnv(fnv(digest, r.id), u64::from(r.distance.to_bits()));
                    }
                    if let Some(truth) = truth.get(q) {
                        found += results.iter().filter(|r| truth.contains(&r.id)).count();
                    }
                }
                let qps = count as f64 / start.elapsed().as_secs_f64();
                out("search", &format!("pass{pass}_qps"), format!("{qps:.0}"));
                out("search", &format!("pass{pass}_major_faults"), major_faults() - faults);
                out("search", &format!("pass{pass}_read_bytes"), proc("io", "read_bytes") - read);
                out("search", &format!("pass{pass}_digest"), format!("{digest:016x}"));
                let recall = found as f64 / (truth.len().min(count as usize) * K) as f64;
                out("search", &format!("pass{pass}_recall"), format!("{recall:.4}"));
            }
            out("search", "anon_mb", anon_mb());
        }
        "check" => check("check", &writer(arg(1))?)?,
        "graph" => {
            // Each slot's id and lists as a search reads them, in order and as sets: a rollback
            // keeps an entry that is in both lists where it is, for readers, so order can change.
            let start = Instant::now();
            let graph =
                HnswGraph::open(Storage::open(arg(1), dims() as u32)?, HnswParams::default())?;
            let (mut ordered, mut sets) = (FNV, FNV);
            for slot in 0..graph.node_count() {
                let record = graph.read_node_record(slot)?;
                let header = [record.header.id, u64::from(record.header.layer_count)];
                (ordered, sets) =
                    (header.into_iter().fold(ordered, fnv), header.into_iter().fold(sets, fnv));
                for layer in 0..usize::from(record.header.layer_count) {
                    let mut ids = record.get_neighbors(layer);
                    ordered = ids.iter().copied().fold(fnv(ordered, ids.len() as u64), fnv);
                    ids.sort_unstable();
                    sets = ids.iter().copied().fold(fnv(sets, ids.len() as u64), fnv);
                }
            }
            out("graph", "slots", graph.node_count());
            out("graph", "ordered_digest", format!("{ordered:016x}"));
            out("graph", "set_digest", format!("{sets:016x}"));
            out("graph", "s", secs(start));
        }
        "delete" => {
            let (path, n) = (arg(1), number(2)?);
            let mut index = writer(path)?;
            let start = Instant::now();
            for id in (0..n).step_by(100) {
                anyhow::ensure!(index.delete(id)?, "{id} wasn't there to delete");
            }
            index.flush()?;
            out("delete", "s", secs(start));
        }
        "add" => {
            let (path, from, to) = (arg(1), number(2)?, number(3)?);
            let mut index = writer(path)?;
            let start = Instant::now();
            for from in (from..to).step_by(10_000) {
                let ids: Vec<u64> = (from..(from + 10_000).min(to)).collect();
                let vectors: Vec<f32> = ids.iter().flat_map(|&id| data.vector(id)).collect();
                index.add_batch_with_ids(&ids, &vectors)?;
                index.flush()?;
            }
            out("add", "s", secs(start));
            out("add", "anon_mb", anon_mb());
        }
        "backup" => {
            let (path, to) = (arg(1), arg(2));
            let mut index = writer(path)?;
            let start = Instant::now();
            index.backup(to)?;
            out("backup", "s", secs(start));
            out("backup", "anon_mb", anon_mb());
        }
        "crash" => {
            let (path, n) = (arg(1), number(2)?);
            let mut index = writer(path)?;
            out("crash", "flushed_len", index.len());
            let ids: Vec<u64> = (n + ADDED..n + ADDED + CHUNK).collect();
            let vectors: Vec<f32> = ids.iter().flat_map(|&id| data.vector(id)).collect();
            out("crash", "adding", ids.len());
            index.add_batch_with_ids(&ids, &vectors)?;
            out("crash", "finished_before_the_kill", 1);
        }
        "verify" => {
            let (path, len) = (arg(1), number(2)?);
            let index = writer(path)?;
            let found = index.len();
            anyhow::ensure!(found == len, "{found} vectors after the kill, not {len}");
            check("verify", &index)?;
        }
        "compact" => {
            let mut index = writer(arg(1))?;
            let start = Instant::now();
            index.compact()?;
            out("compact", "s", secs(start));
            out("compact", "anon_mb", anon_mb());
            check("compact", &index)?;
        }
        other => anyhow::bail!("unknown command {other:?}; the commands are in the file's docs"),
    }
    Ok(())
}
