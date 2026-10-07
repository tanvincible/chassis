//! What deletes cost and what `compact()` gives back (ADR-0011), on a dataset written by
//! `bench/ann/prepare.py`: file size, recall@10 against the exact nearest live vectors, and
//! queries per second (the median of three passes), after deleting a random half, then 90%, of the
//! index, and after compacting.
//!
//! `cargo run --release --example churn -- <data_dir> <dataset>`

use chassis_core::{IndexOptions, IndexReader, VectorIndex, euclidean_distance};
use std::path::Path;
use std::time::Instant;

const K: usize = 10;
const EF_SEARCH: [usize; 2] = [64, 256];

fn read(path: &Path) -> anyhow::Result<(usize, Vec<f32>)> {
    let bytes = std::fs::read(path)?;
    let (header, values) = bytes.split_at(8);
    let cols = u32::from_le_bytes(header[4..8].try_into()?) as usize;
    Ok((cols, values.as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c)).collect()))
}

/// A stable pseudo-random number in 0..1 for `id`: vectors below a fraction are deleted.
fn draw(id: u64) -> f64 {
    let mut x = id.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (x ^ (x >> 31)) as f64 / u64::MAX as f64
}

/// Each query's `K` nearest ids among the vectors at or above `deleted`, by brute force.
fn exact(train: &[f32], dims: usize, queries: &[&[f32]], deleted: f64) -> Vec<Vec<u64>> {
    let nearest = |query: &&[f32]| {
        let mut best: Vec<(f32, u64)> = Vec::with_capacity(K + 1);
        for (id, vector) in train.chunks_exact(dims).enumerate() {
            if draw(id as u64) < deleted {
                continue;
            }
            let distance = euclidean_distance(query, vector);
            if best.len() < K || distance < best[K - 1].0 {
                let at = best.partition_point(|b| b.0 <= distance);
                best.insert(at, (distance, id as u64));
                best.truncate(K);
            }
        }
        best.into_iter().map(|(_, id)| id).collect()
    };
    std::thread::scope(|scope| {
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        let handles: Vec<_> = queries
            .chunks(queries.len().div_ceil(threads))
            .map(|chunk| scope.spawn(move || chunk.iter().map(nearest).collect::<Vec<_>>()))
            .collect();
        handles.into_iter().flat_map(|h| h.join().expect("a brute-force thread panicked")).collect()
    })
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [dir, name] = &args[..] else { anyhow::bail!("usage: churn <data_dir> <dataset>") };
    let dir = Path::new(dir);
    let (dims, train) = read(&dir.join(format!("{name}.train.f32")))?;
    let (_, test) = read(&dir.join(format!("{name}.test.f32")))?;
    let queries: Vec<&[f32]> = test.chunks_exact(dims).collect();
    let n = (train.len() / dims) as u64;
    let options = |ef_search| IndexOptions {
        max_connections: 16,
        ef_construction: 200,
        ef_search,
        ..IndexOptions::default()
    };

    let path = dir.join(format!("{name}.churn.chassis"));
    let _ = std::fs::remove_file(&path);
    let mut index = VectorIndex::open(&path, dims as u32, options(64))?;
    index.add_batch(&train)?;
    index.flush()?;

    println!("{name}, {n} vectors, {} queries, recall@{K} / queries per second", queries.len());
    println!("state\tlive\tfile_mb\tef64\tef256");
    let report = |state: &str, deleted: f64| -> anyhow::Result<()> {
        let truth = exact(&train, dims, &queries, deleted);
        let mut cells = Vec::new();
        for ef_search in EF_SEARCH {
            let mut reader = IndexReader::open(&path, dims as u32, options(ef_search))?;
            for query in &queries {
                reader.search(query, K)?;
            }
            // The median of three passes, as shared machines are noisy.
            let (mut passes, mut hits) = (Vec::new(), 0);
            for _ in 0..3 {
                let start = Instant::now();
                hits = 0;
                for (query, want) in queries.iter().zip(&truth) {
                    let found = reader.search(query, K)?;
                    hits += want.iter().filter(|id| found.iter().any(|r| r.id == **id)).count();
                }
                passes.push(queries.len() as f64 / start.elapsed().as_secs_f64());
            }
            passes.sort_by(f64::total_cmp);
            let recall = hits as f64 / (queries.len() * K) as f64;
            cells.push(format!("{recall:.3} / {:.0}", passes[1]));
        }
        let live = (0..n).filter(|&id| draw(id) >= deleted).count();
        let mb = std::fs::metadata(&path)?.len() as f64 / 1e6;
        println!("{state}\t{live}\t{mb:.0}\t{}", cells.join("\t"));
        Ok(())
    };

    report("built", 0.0)?;
    let mut deleted = 0.0;
    for fraction in [0.5, 0.9] {
        for id in (0..n).filter(|&id| (deleted..fraction).contains(&draw(id))) {
            index.delete(id)?;
        }
        index.flush()?;
        deleted = fraction;
        report(&format!("{:.0}% deleted", fraction * 100.0), deleted)?;
    }
    let start = Instant::now();
    index.compact()?;
    report(&format!("compacted in {:.0}s", start.elapsed().as_secs_f64()), deleted)?;
    drop(index);
    std::fs::remove_file(&path)?;
    Ok(())
}
