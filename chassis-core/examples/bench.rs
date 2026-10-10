//! The pull-request benchmark (`.github/workflows/bench.yml`, `bench/compare.py`): the first `n`
//! vectors of a dataset written by `bench/ann/prepare.py`, built on every core and searched on
//! one thread. Prints one line of JSON per measurement.
//!
//! ```text
//! bench truth  <data_dir> <dataset> <n> <truth_out>
//! bench build  <data_dir> <dataset> <n> <index> <full|half>
//! bench search <data_dir> <dataset> <index> <truth> <ef>...
//! ```

use chassis_core::{IndexOptions, IndexReader, Precision, VectorIndex, euclidean_distance};
use std::path::Path;
use std::time::Instant;

const K: usize = 10;
/// Timed passes over the queries per `ef`; the median is reported.
const PASSES: usize = 3;

/// Reads a `prepare.py` file: u32 rows, u32 columns, then little-endian 4-byte values.
fn read<T>(path: &Path, parse: fn([u8; 4]) -> T) -> anyhow::Result<(usize, Vec<T>)> {
    let bytes = std::fs::read(path)?;
    let (header, values) = bytes.split_at(8);
    let cols = u32::from_le_bytes(header[4..8].try_into()?) as usize;
    Ok((cols, values.as_chunks::<4>().0.iter().map(|&c| parse(c)).collect()))
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |i: usize| args.get(i).map(String::as_str).unwrap_or_default();
    let (dir, name) = (Path::new(arg(1)), arg(2));
    let (dims, train) = read(&dir.join(format!("{name}.train.f32")), f32::from_le_bytes)?;
    let (_, test) = read(&dir.join(format!("{name}.test.f32")), f32::from_le_bytes)?;
    let queries: Vec<&[f32]> = test.chunks_exact(dims).collect();
    match arg(0) {
        "truth" => {
            let n: usize = arg(3).parse()?;
            let vectors: Vec<&[f32]> = train[..n * dims].chunks_exact(dims).collect();
            // Every query's K nearest by brute force, a share of the queries per core.
            let threads = std::thread::available_parallelism().map_or(1, |t| t.get());
            let nearest = |query: &[f32]| -> Vec<u32> {
                let mut all: Vec<(f32, u32)> = vectors
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (euclidean_distance(query, v), i as u32))
                    .collect();
                all.select_nth_unstable_by(K, |a, b| a.0.total_cmp(&b.0));
                all[..K].iter().map(|&(_, i)| i).collect()
            };
            let rows: Vec<Vec<u32>> = std::thread::scope(|scope| {
                let share = queries.len().div_ceil(threads);
                let handles: Vec<_> = queries
                    .chunks(share)
                    .map(|chunk| {
                        scope.spawn(move || chunk.iter().map(|q| nearest(q)).collect::<Vec<_>>())
                    })
                    .collect();
                handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
            });
            let mut out = Vec::new();
            out.extend_from_slice(&(rows.len() as u32).to_le_bytes());
            out.extend_from_slice(&(K as u32).to_le_bytes());
            for id in rows.iter().flatten() {
                out.extend_from_slice(&id.to_le_bytes());
            }
            std::fs::write(arg(4), out)?;
        }
        "build" => {
            let n: usize = arg(3).parse()?;
            let precision = if arg(5) == "half" { Precision::Half } else { Precision::Full };
            let _ = std::fs::remove_file(arg(4));
            let options = IndexOptions { precision, ..IndexOptions::default() };
            let mut index = VectorIndex::open(arg(4), dims as u32, options)?;
            let start = Instant::now();
            index.add_batch(&train[..n * dims])?;
            index.flush()?;
            println!(r#"{{"what": "build", "seconds": {:.3}}}"#, start.elapsed().as_secs_f64());
        }
        "search" => {
            let (_, truth) = read(Path::new(arg(4)), u32::from_le_bytes)?;
            for ef in args[5..].iter().map(|ef| ef.parse::<usize>()) {
                let ef = ef?;
                let options = IndexOptions { ef_search: ef, ..IndexOptions::default() };
                let mut index = IndexReader::open(arg(3), dims as u32, options)?;
                // A first pass brings the file in and is not timed.
                for query in &queries {
                    index.search(query, K)?;
                }
                let mut rates = Vec::with_capacity(PASSES);
                let mut hits = 0;
                for _ in 0..PASSES {
                    let start = Instant::now();
                    hits = 0;
                    for (query, want) in queries.iter().zip(truth.as_chunks::<K>().0) {
                        let found = index.search(query, K)?;
                        hits += found.iter().filter(|r| want.contains(&(r.id as u32))).count();
                    }
                    rates.push(queries.len() as f64 / start.elapsed().as_secs_f64());
                }
                rates.sort_by(f64::total_cmp);
                let recall = hits as f64 / (queries.len() * K) as f64;
                println!(
                    r#"{{"what": "search", "ef": {ef}, "recall": {recall:.4}, "qps": {:.1}}}"#,
                    rates[PASSES / 2]
                );
            }
        }
        other => anyhow::bail!("unknown mode {other:?}: truth, build or search"),
    }
    Ok(())
}
