//! Filtered search on a dataset written by `bench/ann/prepare.py` (ADR-0009): latency and
//! recall@10 of `search_filtered` against an exact scan of the matching vectors, for filters
//! passing 30% down to 0.1% of the index. Filter shapes per fraction: random ids, looked up in a
//! `Vec<bool>` and, slower, in a `HashSet` as an allow-list would be; the Voronoi cell around the
//! query (matches near the query); another cell (matches far from it).
//!
//! `cargo run --release --example filtered -- <data_dir> <dataset> <index_path> <queries>`
//! builds the index at `<index_path>` unless it is already there.

use chassis_core::{IndexOptions, IndexReader, VectorIndex, euclidean_distance};
use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;

const K: usize = 10;

fn read<T>(path: &Path, parse: fn([u8; 4]) -> T) -> anyhow::Result<(usize, Vec<T>)> {
    let bytes = std::fs::read(path)?;
    let (header, values) = bytes.split_at(8);
    let cols = u32::from_le_bytes(header[4..8].try_into()?) as usize;
    Ok((cols, values.as_chunks::<4>().0.iter().map(|&c| parse(c)).collect()))
}

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Index of the nearest of `points` (row numbers into `data`) to `v`.
fn nearest(v: &[f32], data: &[f32], dims: usize, points: &[usize]) -> usize {
    let row = |i: usize| &data[points[i] * dims..(points[i] + 1) * dims];
    let distances = (0..points.len()).map(|i| (euclidean_distance(v, row(i)), i));
    distances.min_by(|a, b| a.0.total_cmp(&b.0)).expect("at least one point").1
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [dir, name, path, queries] = &args[..] else {
        anyhow::bail!("usage: filtered <data_dir> <dataset> <index_path> <queries>");
    };
    let (dir, path, queries) = (Path::new(dir), Path::new(path), queries.parse::<usize>()?);
    let (dims, train) = read(&dir.join(format!("{name}.train.f32")), f32::from_le_bytes)?;
    let (_, test) = read(&dir.join(format!("{name}.test.f32")), f32::from_le_bytes)?;
    let n = train.len() / dims;
    let vector = |i: usize| &train[i * dims..(i + 1) * dims];
    let options = |ef_search| IndexOptions {
        max_connections: 16,
        ef_construction: 200,
        ef_search,
        ..IndexOptions::default()
    };

    let mut index = VectorIndex::open(path, dims as u32, options(64))?;
    if index.len() != n as u64 {
        anyhow::ensure!(index.is_empty(), "{} holds another index", path.display());
        let start = Instant::now();
        for i in 0..n {
            index.add(vector(i))?;
        }
        index.flush()?;
        eprintln!("built {n} vectors in {:.0}s", start.elapsed().as_secs_f64());
    }
    drop(index);
    let mut readers = [
        IndexReader::open(path, dims as u32, options(64))?,
        IndexReader::open(path, dims as u32, options(256))?,
    ];

    println!("{name}, {n} vectors, {queries} queries; times are means in microseconds");
    println!("filter\tfraction\tmatches\texact_us\tef64_us\tef64_recall\tef256_us\tef256_recall");
    for fraction in [0.3, 0.1, 0.03, 0.01, 0.003, 0.001] {
        let random: Vec<bool> =
            (0..n).map(|i| (mix(i as u64) as f64) < fraction * u64::MAX as f64).collect();
        let set: HashSet<u64> = (0..n as u64).filter(|&i| random[i as usize]).collect();
        let cells = (1.0 / fraction).round() as usize;
        let pivots: Vec<usize> =
            (0..cells).map(|j| (mix(!(j as u64)) % n as u64) as usize).collect();
        let cell: Vec<usize> = (0..n).map(|i| nearest(vector(i), &train, dims, &pivots)).collect();

        for shape in ["random", "set", "near", "far"] {
            let (mut matches, mut exact_us, mut us, mut recall) = (0, 0.0, [0.0; 2], [0.0; 2]);
            for (q, query) in test.chunks_exact(dims).take(queries).enumerate() {
                let own = nearest(query, &train, dims, &pivots);
                let other = (own + 1 + mix(q as u64) as usize % (cells - 1).max(1)) % cells;
                let allow = |i: u64| match shape {
                    "random" => random[i as usize],
                    "set" => set.contains(&i),
                    "near" => cell[i as usize] == own,
                    _ => cell[i as usize] == other,
                };

                let start = Instant::now();
                let mut truth: Vec<(f32, u64)> = (0..n as u64)
                    .filter(|&i| allow(i))
                    .map(|i| (euclidean_distance(query, vector(i as usize)), i))
                    .collect();
                truth.sort_by(|a, b| a.0.total_cmp(&b.0));
                truth.truncate(K);
                exact_us += start.elapsed().as_secs_f64() * 1e6;
                matches += (0..n as u64).filter(|&i| allow(i)).count();

                for (r, reader) in readers.iter_mut().enumerate() {
                    let start = Instant::now();
                    let found = reader.search_filtered(query, K, allow)?;
                    us[r] += start.elapsed().as_secs_f64() * 1e6;
                    let hits = truth.iter().filter(|t| found.iter().any(|f| f.id == t.1)).count();
                    recall[r] += hits as f64 / truth.len().max(1) as f64;
                }
            }
            let mean = |total: f64| total / queries as f64;
            println!(
                "{shape}\t{fraction}\t{}\t{:.0}\t{:.0}\t{:.3}\t{:.0}\t{:.3}",
                matches / queries,
                mean(exact_us),
                mean(us[0]),
                mean(recall[0]),
                mean(us[1]),
                mean(recall[1]),
            );
        }
    }
    Ok(())
}
