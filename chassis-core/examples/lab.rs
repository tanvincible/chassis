//! x86 lab harness (experiment branch only). Output lines are tab-separated:
//! `engine tag n what ef recall qps distances_per_query`.
//!
//! lab kernel <tag>
//! lab build <data> <dataset> <index> <n>           batch build of the first n vectors
//! lab seq <data> <dataset> <n> <tag>               one-thread build of the first n, timed
//! lab truth <data> <dataset> <n> <out>             exact top 10 among the first n
//! lab search <data> <dataset> <index> <truth> <tag>

use chassis_core::{IndexOptions, VectorIndex, euclidean_distance};
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

fn options(ef_search: usize) -> IndexOptions {
    IndexOptions { max_connections: 16, ef_construction: 200, ef_search, ..IndexOptions::default() }
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

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |i: usize| args.get(i).cloned().unwrap_or_default();
    if arg(0) == "kernel" {
        kernel(&arg(1));
        return Ok(());
    }
    let dir = Path::new(&args[1]);
    let name = &args[2];
    let (dims, train) = read(&dir.join(format!("{name}.train.f32")), f32::from_le_bytes)?;
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
            let path = dir.join(format!("{name}.seq.{}.chassis", arg(4)));
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
            for ef_search in EF_SEARCH {
                let index = VectorIndex::open(arg(3), dims as u32, options(ef_search))?;
                let n = index.len();
                for query in &queries {
                    index.search(query, K)?;
                }
                #[cfg(lab)]
                let counted = chassis_core::lab::distances();
                let mut passes = Vec::with_capacity(PASSES);
                let mut results = Vec::new();
                for _ in 0..PASSES {
                    let start = Instant::now();
                    results =
                        queries.iter().map(|q| index.search(q, K)).collect::<Result<_, _>>()?;
                    passes.push(queries.len() as f64 / start.elapsed().as_secs_f64());
                }
                #[cfg(lab)]
                let per_query = (chassis_core::lab::distances() - counted) as f64
                    / (PASSES * queries.len()) as f64;
                #[cfg(not(lab))]
                let per_query = 0.0;
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
                    "chassis\t{tag}\t{n}\tsearch\t{ef_search}\t{recall:.4}\t{:.0}\t{per_query:.0}",
                    passes[PASSES / 2]
                );
            }
        }
        other => anyhow::bail!("unknown mode {other}"),
    }
    Ok(())
}
