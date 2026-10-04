//! Builds an index of uniform random vectors and reports build time, recall@10 and search latency,
//! then deletes 10% and 50% of it and reports them again.
//!
//! `cargo run --release --example recall -- [count] [dims]`

use chassis_core::{IndexOptions, VectorIndex, euclidean_distance};
use std::time::Instant;

const QUERIES: usize = 200;
const K: usize = 10;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1).map(|a| a.parse::<usize>());
    let count = args.next().transpose()?.unwrap_or(20_000);
    let dims = args.next().transpose()?.unwrap_or(128);

    let random = |n: usize| -> Vec<Vec<f32>> {
        (0..n).map(|_| (0..dims).map(|_| rand::random::<f32>() * 2.0 - 1.0).collect()).collect()
    };
    let data = random(count);
    let queries = random(QUERIES);

    let dir = tempfile::tempdir()?;
    let path = dir.path().join("recall.chassis");
    let dims = u32::try_from(dims)?;

    let mut index = VectorIndex::open(&path, dims, IndexOptions::default())?;
    let start = Instant::now();
    for v in &data {
        index.add(v)?;
    }
    index.flush()?;
    let build = start.elapsed().as_secs_f64();
    println!("{count} x {dims}d: build {build:.1}s ({:.0} inserts/s)", count as f64 / build);
    drop(index);

    let mut live = vec![true; count];
    for ef_search in [50, 200] {
        let index =
            VectorIndex::open(&path, dims, IndexOptions { ef_search, ..IndexOptions::default() })?;
        report(&format!("ef_search {ef_search:>3}"), &index, &data, &queries, &live)?;
    }

    let mut index = VectorIndex::open(&path, dims, IndexOptions::default())?;
    for percent in [10, 50] {
        // Delete evenly spread ids until `percent` of the index is gone.
        let target = count * percent / 100;
        let mut deleted = count - index.len() as usize;
        for id in (0..count).step_by(2).chain((1..count).step_by(2)) {
            if deleted == target {
                break;
            }
            if live[id] {
                index.delete(id as u64)?;
                live[id] = false;
                deleted += 1;
            }
        }
        let start = Instant::now();
        index.flush()?;
        let flush_ms = start.elapsed().as_secs_f64() * 1e3;
        report(
            &format!("{percent}% deleted, flush {flush_ms:.1} ms"),
            &index,
            &data,
            &queries,
            &live,
        )?;
    }
    Ok(())
}

/// Prints recall@K against brute force over the live vectors, and mean search latency.
fn report(
    label: &str,
    index: &VectorIndex,
    data: &[Vec<f32>],
    queries: &[Vec<f32>],
    live: &[bool],
) -> anyhow::Result<()> {
    let mut hits = 0;
    let mut search_secs = 0.0;
    for q in queries {
        let start = Instant::now();
        let results = index.search(q, K)?;
        search_secs += start.elapsed().as_secs_f64();

        let mut truth: Vec<(u64, f32)> = (0..)
            .zip(data)
            .filter(|&(id, _)| live[id as usize])
            .map(|(id, v)| (id, euclidean_distance(q, v)))
            .collect();
        truth.sort_by(|a, b| a.1.total_cmp(&b.1));
        hits += results.iter().filter(|r| truth[..K].iter().any(|t| t.0 == r.id)).count();
    }
    let recall = hits as f64 / (queries.len() * K) as f64;
    let latency_us = search_secs * 1e6 / queries.len() as f64;
    println!("  {label}: recall@{K} {recall:.3}, mean search {latency_us:.0} us");
    Ok(())
}
