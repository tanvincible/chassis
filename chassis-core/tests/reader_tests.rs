//! Readers searching a file a writer is changing (ADR-0008, decision 5). Readers take no lock, so
//! most tests run the writer and the reader in one process; one runs the writer in a child.

use chassis_core::{IndexOptions, IndexReader, VectorIndex};
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use tempfile::tempdir;

const DIMS: u32 = 8;

/// Pseudo-random, so no two vectors tie (tied distances make exact-match searches flaky).
fn vector(id: u64) -> Vec<f32> {
    (0..u64::from(DIMS))
        .map(|d| {
            let mut x = (id * u64::from(DIMS) + d).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            x = (x ^ (x >> 31)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            (x >> 40) as f32 / (1u64 << 24) as f32
        })
        .collect()
}

fn options() -> IndexOptions {
    IndexOptions { ef_construction: 64, ef_search: 64, ..IndexOptions::default() }
}

fn top(reader: &mut IndexReader, id: u64) -> u64 {
    reader.search(&vector(id), 1).unwrap()[0].id
}

#[test]
fn test_reader_sees_only_committed_adds_and_deletes() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("index.chassis");
    let mut writer = VectorIndex::open(&path, DIMS, options()).unwrap();
    for id in 0..100 {
        writer.add(&vector(id)).unwrap();
    }
    writer.flush().unwrap();

    let mut reader = IndexReader::open(&path, DIMS, options()).unwrap();
    assert_eq!(top(&mut reader, 42), 42);
    assert_eq!(reader.len(), 100);

    for id in 100..200 {
        writer.add(&vector(id)).unwrap();
    }
    assert!(writer.delete(5).unwrap());
    for id in [5, 150] {
        let hits = reader.search(&vector(id), 10).unwrap();
        assert!(hits.iter().all(|hit| hit.id < 100), "unflushed adds are not returned");
        assert_eq!(hits.iter().any(|hit| hit.id == 5), id == 5, "unflushed deletes don't apply");
    }
    assert_eq!(reader.len(), 100);

    writer.flush().unwrap();
    assert_eq!(top(&mut reader, 150), 150);
    assert!(reader.search(&vector(5), 10).unwrap().iter().all(|hit| hit.id != 5));
    assert_eq!(reader.len(), 199);
}

#[test]
fn test_reader_returns_custom_ids_and_maps_new_segments() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("index.chassis");
    let mut writer = VectorIndex::open(&path, DIMS, options()).unwrap();
    writer.flush().unwrap();
    let mut reader = IndexReader::open(&path, DIMS, options()).unwrap();
    assert!(reader.is_empty());
    assert!(reader.search(&vector(0), 1).unwrap().is_empty());

    // 1,024-slot segments doubling: 5,000 slots cross several segments and heap chunks.
    for batch in 0..5 {
        for id in batch * 1000..(batch + 1) * 1000 {
            writer.add_with_id(id * 10 + 3, &vector(id)).unwrap();
        }
        writer.flush().unwrap();
        for id in (0..(batch + 1) * 1000).step_by(97) {
            assert_eq!(top(&mut reader, id), id * 10 + 3);
        }
    }
}

#[test]
fn test_reader_routes_through_an_unflushed_bulk_load() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("index.chassis");
    let mut writer = VectorIndex::open(&path, DIMS, options()).unwrap();
    for id in 0..300 {
        writer.add(&vector(id)).unwrap();
    }
    writer.flush().unwrap();
    let mut reader = IndexReader::open(&path, DIMS, options()).unwrap();
    // The committed nodes' lists now lead mostly to nodes added after the flush.
    for id in 300..3000 {
        writer.add(&vector(id)).unwrap();
    }
    // recall@10 among the committed vectors, for queries near them
    let mut hits = 0;
    for q in 0..100u64 {
        let query: Vec<f32> = vector(10_000 + q);
        let mut exact: Vec<(f32, u64)> = (0..300)
            .map(|id| (vector(id).iter().zip(&query).map(|(a, b)| (a - b) * (a - b)).sum(), id))
            .collect();
        exact.sort_by(|a, b| a.0.total_cmp(&b.0));
        let found = reader.search(&query, 10).unwrap();
        hits += exact[..10].iter().filter(|(_, id)| found.iter().any(|f| f.id == *id)).count();
    }
    let recall = hits as f64 / 1000.0;
    assert!(recall >= 0.95, "recall@10 {recall} over the committed vectors");
    for id in (300..3000).step_by(37) {
        assert!(reader.search(&vector(id), 10).unwrap().iter().all(|hit| hit.id < 300));
    }
}

#[test]
fn test_reader_survives_a_writer_crash_and_restart() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("index.chassis");
    let mut writer = VectorIndex::open(&path, DIMS, options()).unwrap();
    for id in 0..500 {
        writer.add(&vector(id)).unwrap();
    }
    writer.flush().unwrap();
    let mut reader = IndexReader::open(&path, DIMS, options()).unwrap();
    for id in 500..2500 {
        writer.add(&vector(id)).unwrap();
    }
    assert_eq!(top(&mut reader, 7), 7);
    drop(writer); // never flushed: slots 500.. are rolled back and reused

    let mut writer = VectorIndex::open(&path, DIMS, options()).unwrap();
    for id in 5000..5600 {
        assert_eq!(writer.add(&vector(id)).unwrap(), id - 4500);
    }
    // Edges the crashed writer pruned stay lost (ADR-0005), so a few committed vectors are hard to
    // reach; the reader must do no worse than the writer on the same graph.
    let (mut reader_misses, mut writer_misses) = (0, 0);
    for id in 0..500 {
        reader_misses += usize::from(top(&mut reader, id) != id);
        writer_misses += usize::from(writer.search(&vector(id), 1).unwrap()[0].id != id);
    }
    assert!(
        reader_misses <= writer_misses.max(10),
        "{reader_misses} misses, writer {writer_misses}"
    );
    writer.flush().unwrap();
    assert_eq!(reader.len(), 500);
    assert_eq!(top(&mut reader, 5300), 800);
    assert_eq!(reader.len(), 1100);
}

#[test]
fn test_reader_refuses_files_it_cannot_read() {
    let dir = tempdir().unwrap();
    let missing = dir.path().join("missing.chassis");
    assert!(IndexReader::open(&missing, DIMS, options()).is_err());
    assert!(!missing.exists(), "a reader never creates the file");

    let path = dir.path().join("index.chassis");
    VectorIndex::open(&path, DIMS, options()).unwrap().flush().unwrap();
    assert!(IndexReader::open(&path, DIMS + 1, options()).is_err());
    let other = IndexOptions { max_connections: 8, ..options() };
    assert!(IndexReader::open(&path, DIMS, other).is_err());

    let mut legacy = vec![0u8; 4096];
    legacy[..8].copy_from_slice(b"CHASSIS\0");
    legacy[8..12].copy_from_slice(&2u32.to_le_bytes());
    std::fs::write(&path, legacy).unwrap();
    let error = IndexReader::open(&path, DIMS, options()).unwrap_err().to_string();
    assert!(error.contains("migrate"), "{error}");
}

/// The child half: only does work when the parent test spawns it with a path.
#[test]
fn reader_test_writer() {
    let Ok(path) = std::env::var("CHASSIS_READER_PATH") else { return };
    let mut writer = VectorIndex::open(&path, DIMS, options()).unwrap();
    for batch in 0..u64::MAX {
        for id in batch * 200..(batch + 1) * 200 {
            writer.add(&vector(id)).unwrap();
        }
        if batch % 3 == 0 {
            writer.delete(batch * 200 + 1).unwrap();
        }
        writer.flush().unwrap();
        println!("flushed {batch}");
    }
}

#[test]
fn test_reader_in_another_process_than_the_writer() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("index.chassis");
    VectorIndex::open(&path, DIMS, options()).unwrap().flush().unwrap();
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["reader_test_writer", "--exact", "--nocapture"])
        .env("CHASSIS_READER_PATH", &path)
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();
    while !lines.next().unwrap().unwrap().starts_with("flushed") {}

    let mut reader = IndexReader::open(&path, DIMS, options()).unwrap();
    let (start, mut searches) = (Instant::now(), 0);
    let first = reader.len();
    let mut committed = first;
    // At least 3 s of searching, and until a later flush shows up: a loaded machine slows the writer.
    while start.elapsed() < Duration::from_secs(3)
        || (committed == first && start.elapsed() < Duration::from_secs(120))
    {
        // Every id below an earlier snapshot's count stays committed; only every 600th is deleted.
        let id = rand::random_range(0..committed.max(1));
        let hits = reader.search(&vector(id), 1).unwrap();
        if id % 600 != 1 {
            assert_eq!(hits[0].id, id, "committed vector {id} not found");
        }
        committed = reader.len();
        searches += 1;
    }
    child.kill().unwrap();
    child.wait().unwrap();
    assert!(committed > first, "the reader never saw a later flush");
    println!("{searches} searches while the writer reached {committed} vectors");
}

/// A reader that found the newest header copy torn adopts it once it reads whole, though its
/// sequence and checksum words never changed.
#[cfg(unix)]
#[test]
fn test_reader_adopts_a_header_copy_that_was_torn() {
    use std::os::unix::fs::FileExt;
    let dir = tempdir().unwrap();
    let path = dir.path().join("torn.chassis");
    let mut writer = VectorIndex::open(&path, DIMS, options()).unwrap();
    for batch in 0..2 {
        for id in 0..10 {
            writer.add(&vector(batch * 10 + id)).unwrap();
        }
        writer.flush().unwrap();
    }
    drop(writer);
    let file = std::fs::OpenOptions::new().read(true).write(true).open(&path).unwrap();
    let sequence = |at: u64| {
        let mut word = [0u8; 8];
        file.read_exact_at(&mut word, at + 24).unwrap();
        u64::from_le_bytes(word)
    };
    let newest = if sequence(0) > sequence(64 * 1024) { 0 } else { 64 * 1024 };
    let mut count = [0u8; 1];
    file.read_exact_at(&mut count, newest + 56).unwrap();
    file.write_all_at(&[count[0] ^ 1], newest + 56).unwrap();

    let mut reader = IndexReader::open(&path, DIMS, options()).unwrap();
    assert_eq!(reader.len(), 10);
    file.write_all_at(&count, newest + 56).unwrap();
    reader.refresh().unwrap();
    assert_eq!(reader.len(), 20);
}
