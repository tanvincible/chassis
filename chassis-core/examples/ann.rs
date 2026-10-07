//! Runs Chassis on a dataset written by `bench/ann/prepare.py`: build time, file size, and
//! recall@10 against queries per second on one thread, across `ef_search` values. Queries per
//! second is the median of five passes over the queries.
//! `bench/ann/hnswlib_bench.py` runs hnswlib with the same settings.
//!
//! `cargo run --release --example ann -- <data_dir> <dataset>`

use chassis_core::{IndexOptions, VectorIndex};
use std::path::Path;
use std::time::Instant;

const K: usize = 10;
const PASSES: usize = 5;
const EF_SEARCH: [usize; 7] = [10, 16, 32, 64, 128, 256, 512];

/// Reads a `prepare.py` file: u32 rows, u32 columns, then little-endian 4-byte values.
fn read<T>(path: &Path, parse: fn([u8; 4]) -> T) -> anyhow::Result<(usize, Vec<T>)> {
    let bytes = std::fs::read(path)?;
    let (header, values) = bytes.split_at(8);
    let cols = u32::from_le_bytes(header[4..8].try_into()?) as usize;
    let values = values.as_chunks::<4>().0.iter().map(|&c| parse(c)).collect();
    Ok((cols, values))
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let (Some(dir), Some(name)) = (args.next(), args.next()) else {
        anyhow::bail!("usage: ann <data_dir> <dataset>");
    };
    let dir = Path::new(&dir);
    let (dims, train) = read(&dir.join(format!("{name}.train.f32")), f32::from_le_bytes)?;
    let (_, test) = read(&dir.join(format!("{name}.test.f32")), f32::from_le_bytes)?;
    let (depth, gt) = read(&dir.join(format!("{name}.gt.u32")), u32::from_le_bytes)?;

    let path = dir.join(format!("{name}.chassis"));
    let _ = std::fs::remove_file(&path);
    let options = |ef_search| IndexOptions { max_connections: 16, ef_construction: 200, ef_search };
    let dims_u32 = u32::try_from(dims)?;

    let mut index = VectorIndex::open(&path, dims_u32, options(K))?;
    let start = Instant::now();
    for (i, vector) in train.chunks_exact(dims).enumerate() {
        index.add(vector)?;
        if (i + 1) % 100_000 == 0 {
            eprintln!("  {} added, {:.0}s", i + 1, start.elapsed().as_secs_f64());
        }
    }
    index.flush()?;
    let build = start.elapsed().as_secs_f64();
    let size_mb = std::fs::metadata(&path)?.len() as f64 / 1e6;
    println!("chassis\t{name}\tbuild\t{build:.1}s\t{size_mb:.1} MB");
    drop(index);

    let queries: Vec<&[f32]> = test.chunks_exact(dims).collect();
    let truth: Vec<&[u32]> = gt.chunks_exact(depth).map(|row| &row[..K]).collect();
    for ef_search in EF_SEARCH {
        let index = VectorIndex::open(&path, dims_u32, options(ef_search))?;
        // Every open is a fresh mapping; without a warm-up pass the timed one pays its page faults.
        for query in &queries {
            index.search(query, K)?;
        }
        let mut passes = Vec::with_capacity(PASSES);
        let mut results = Vec::new();
        for _ in 0..PASSES {
            let start = Instant::now();
            results = queries.iter().map(|q| index.search(q, K)).collect::<Result<_, _>>()?;
            passes.push(queries.len() as f64 / start.elapsed().as_secs_f64());
        }
        passes.sort_by(f64::total_cmp);
        let qps = passes[PASSES / 2];
        let hits: usize = results
            .iter()
            .zip(&truth)
            .map(|(found, want)| found.iter().filter(|r| want.contains(&(r.id as u32))).count())
            .sum();
        let recall = hits as f64 / (queries.len() * K) as f64;
        println!("chassis\t{name}\t{ef_search}\t{recall:.3}\t{qps:.0}");
    }
    std::fs::remove_file(&path)?;
    Ok(())
}
