//! What a backup costs (ADR-0024), on `n` random vectors of `dims` components: a backup from the
//! writer and one from a reader, timed, with each file's length and the disk it takes; then ten
//! backups from a reader while a writer adds ten vectors and flushes every `every_ms`, and how
//! many of them held a flush.
//!
//! `cargo run --release --example backup_cost -- <n> <dims> <every_ms>...`

use chassisdb::{ErrorKind, IndexOptions, IndexReader, VectorIndex};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

fn vector(id: u64, dims: usize) -> Vec<f32> {
    let mut x = id.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..dims)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 40) as f32 / (1u64 << 24) as f32
        })
        .collect()
}

/// The file's length and the disk it takes, in MB.
fn size(path: &Path) -> (f64, f64) {
    let meta = std::fs::metadata(path).unwrap();
    #[cfg(unix)]
    let disk = std::os::unix::fs::MetadataExt::blocks(&meta) * 512;
    #[cfg(not(unix))]
    let disk = meta.len();
    (meta.len() as f64 / 1e6, disk as f64 / 1e6)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (n, dims): (u64, usize) = (args[0].parse().unwrap(), args[1].parse().unwrap());
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index.chassis");
    let mut index = VectorIndex::open(&path, dims as u32, IndexOptions::default()).unwrap();
    for start in (0..n).step_by(10_000) {
        let batch: Vec<f32> =
            (start..n.min(start + 10_000)).flat_map(|i| vector(i, dims)).collect();
        index.add_batch(&batch).unwrap();
    }
    index.flush().unwrap();
    let (length, disk) = size(&path);
    println!("index: {n} vectors of {dims}, {length:.1} MB long, {disk:.1} MB on disk");

    let mut reader = IndexReader::open(&path, dims as u32, IndexOptions::default()).unwrap();
    let to = |name: &str| dir.path().join(name);
    let start = Instant::now();
    index.backup(to("writer.chassis")).unwrap();
    let writer_s = start.elapsed().as_secs_f64();
    let start = Instant::now();
    reader.backup(to("reader.chassis")).unwrap();
    let reader_s = start.elapsed().as_secs_f64();
    for (name, seconds) in [("writer.chassis", writer_s), ("reader.chassis", reader_s)] {
        let (length, disk) = size(&to(name));
        println!("{name}: {seconds:.3} s, {length:.1} MB long, {disk:.1} MB on disk");
    }

    let mut next = n;
    for every_ms in args[2..].iter().map(|a| a.parse::<u64>().unwrap()) {
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = stop.clone();
        let writer = std::thread::spawn(move || {
            while !stopping.load(Ordering::Relaxed) {
                for _ in 0..10 {
                    index.add(&vector(next, dims)).unwrap();
                    next += 1;
                }
                index.flush().unwrap();
                std::thread::sleep(Duration::from_millis(every_ms));
            }
            (index, next)
        });
        let (mut held, mut seconds) = (0, 0.0);
        for _ in 0..10 {
            let start = Instant::now();
            match reader.backup(to("during.chassis")) {
                Ok(()) => held += 1,
                Err(e) if e.kind() == ErrorKind::Other => {}
                Err(e) => panic!("{e}"),
            }
            seconds += start.elapsed().as_secs_f64();
        }
        stop.store(true, Ordering::Relaxed);
        (index, next) = writer.join().unwrap();
        println!(
            "a flush every {every_ms} ms: {held} of 10 held one, {:.3} s each",
            seconds / 10.0
        );
    }
}
