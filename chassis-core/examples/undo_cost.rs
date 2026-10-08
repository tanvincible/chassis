//! What saving committed nodes' lists costs (ADR-0012), on 128-dimensional random vectors: a
//! batch and then single adds onto `base` committed vectors, in wall and CPU time, the size of
//! the undo file, and an add followed by a flush, 200 times.
//!
//! `cargo run --release --example undo_cost -- <base> <batch> <singles>`

use chassis_core::{IndexOptions, VectorIndex};
use std::time::Instant;

const DIMS: usize = 128;

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

/// User plus system CPU seconds of this process so far, on all threads.
#[cfg(unix)]
fn cpu() -> f64 {
    // SAFETY: getrusage fills the struct it is given.
    let usage = unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut usage);
        usage
    };
    let seconds = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 / 1e6;
    seconds(usage.ru_utime) + seconds(usage.ru_stime)
}

#[cfg(not(unix))]
fn cpu() -> f64 {
    0.0
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [base, batch, singles] = &args[..] else {
        anyhow::bail!("usage: undo_cost <base> <batch> <singles>")
    };
    let (base, batch, singles): (u64, u64, u64) = (base.parse()?, batch.parse()?, singles.parse()?);
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("index.chassis");
    let options = IndexOptions { ef_construction: 100, ..IndexOptions::default() };
    let flat = |ids: std::ops::Range<u64>| ids.flat_map(vector).collect::<Vec<f32>>();
    let mut index = VectorIndex::open(&path, DIMS as u32, options)?;
    index.add_batch(&flat(0..base))?;
    index.flush()?;

    let vectors = flat(base..base + batch);
    let (wall, used) = (Instant::now(), cpu());
    index.add_batch(&vectors)?;
    println!(
        "batch of {batch}: {:.2} s, {:.1} s of CPU",
        wall.elapsed().as_secs_f64(),
        cpu() - used
    );
    index.flush()?;

    let vectors: Vec<Vec<f32>> = (base + batch..base + batch + singles).map(vector).collect();
    let (wall, used) = (Instant::now(), cpu());
    for vector in &vectors {
        index.add(vector)?;
    }
    let per_add = |seconds: f64| seconds * 1e6 / singles as f64;
    let undo = std::fs::metadata(dir.path().join("index.chassis.undo")).map_or(0, |m| m.len());
    println!(
        "{singles} single adds: {:.0} us each, {:.0} us of CPU each, undo file {:.1} MB",
        per_add(wall.elapsed().as_secs_f64()),
        per_add(cpu() - used),
        undo as f64 / 1e6
    );
    index.flush()?;

    let wall = Instant::now();
    for id in 0..200 {
        index.add(&vector(9_000_000 + id))?;
        index.flush()?;
    }
    println!("add then flush: {:.2} ms", wall.elapsed().as_secs_f64() * 1000.0 / 200.0);
    Ok(())
}
