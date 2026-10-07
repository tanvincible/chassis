//! What a crash that loses adds does to the vectors committed before it, and what `compact()`
//! restores (ADR-0011): recall@10 over 200 queries on random vectors, and how many of the `kept`
//! committed vectors a search for them finds, after committing them, after losing `lost` more to
//! a reopen without a flush, after adding those again, and after compacting. Layers are drawn at
//! random, so runs differ.
//!
//! `cargo run --release --example lost_adds -- <kept> <lost>`

use chassis_core::{IndexOptions, VectorIndex, euclidean_distance};

const DIMS: usize = 24;

fn vector(id: u64) -> Vec<f32> {
    let mut x = id.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..DIMS)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 40) as f32 / (1u64 << 24) as f32
        })
        .collect()
}

fn report(state: &str, index: &VectorIndex, n: u64, kept: u64) -> anyhow::Result<()> {
    let (mut hits, mut short) = (0, 0);
    for q in 0..200 {
        let query = vector(1_000_000 + q);
        let mut exact: Vec<(f32, u64)> =
            (0..n).map(|id| (euclidean_distance(&query, &vector(id)), id)).collect();
        exact.sort_by(|a, b| a.0.total_cmp(&b.0));
        let found = index.search(&query, 10)?;
        short += usize::from(found.len() < 10);
        hits += exact[..10].iter().filter(|(_, id)| found.iter().any(|r| r.id == *id)).count();
    }
    let mut found = 0;
    for id in 0..kept {
        found += u64::from(index.search(&vector(id), 10)?.iter().any(|r| r.id == id));
    }
    println!("{state}\t{:.3}\t{short}\t{found}", hits as f64 / 2000.0);
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [kept, lost] = &args[..] else { anyhow::bail!("usage: lost_adds <kept> <lost>") };
    let (kept, lost): (u64, u64) = (kept.parse()?, lost.parse()?);
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("index.chassis");
    let options = IndexOptions { ef_construction: 100, ef_search: 64, ..IndexOptions::default() };
    let flat = |ids: std::ops::Range<u64>| ids.flat_map(vector).collect::<Vec<f32>>();

    println!(
        "state\trecall@10\tsearches of 200 returning fewer than 10\tof {kept} committed, found"
    );
    let mut index = VectorIndex::open(&path, DIMS as u32, options.clone())?;
    index.add_batch(&flat(0..kept))?;
    index.flush()?;
    report("committed", &index, kept, kept)?;
    index.add_batch(&flat(kept..kept + lost))?;
    drop(index);

    let mut index = VectorIndex::open(&path, DIMS as u32, options)?;
    report("adds lost", &index, kept, kept)?;
    index.add_batch(&flat(kept..kept + lost))?;
    report("added again", &index, kept + lost, kept)?;
    index.compact()?;
    report("compacted", &index, kept + lost, kept)
}
