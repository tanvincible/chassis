//! Kills a writer process at random points; every reopen must show exactly the last flush.
//!
//! Process death only: a kill keeps the page cache, so this can't see fsync order or power loss.

use chassis_core::{IndexOptions, Storage, VectorIndex};
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::time::Duration;

/// Large vectors, so release runs cross graph moves (the first is at 2048 vectors).
const DIMS: u32 = 1024;
const BATCH: u64 = 50;
const DELETES_PER_BATCH: u64 = 10;
const ROUNDS: usize = 12;

fn options() -> IndexOptions {
    IndexOptions { ef_construction: 16, ..IndexOptions::default() }
}

fn vector(id: u64) -> Vec<f32> {
    vec![id as f32; DIMS as usize]
}

/// Ids that batch `b` deletes: every fifth id of the batch before it.
fn deleted_by(b: u64) -> Vec<u64> {
    if b == 0 { vec![] } else { (0..DELETES_PER_BATCH).map(|j| (b - 1) * BATCH + j * 5).collect() }
}

/// Live vectors after `batches` flushed batches.
fn live_after(batches: u64) -> u64 {
    batches * BATCH - batches.saturating_sub(1) * DELETES_PER_BATCH
}

/// The child half: only does work when the parent test spawns it with a path.
#[test]
fn crash_writer() {
    let Ok(path) = std::env::var("CHASSIS_CRASH_PATH") else { return };
    let mut index = VectorIndex::open(&path, DIMS, options()).unwrap();
    let mut batch = (0..).find(|&b| live_after(b) == index.len()).unwrap();
    loop {
        for id in batch * BATCH..(batch + 1) * BATCH {
            assert_eq!(index.add(&vector(id)).unwrap(), id);
        }
        for id in deleted_by(batch) {
            assert!(index.delete(id).unwrap());
        }
        index.flush().unwrap();
        batch += 1;
        println!("flushed {batch}");
    }
}

#[test]
fn test_kill_at_random_points_keeps_flushed_data() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("crash.chassis");
    let mut flushed = 0;

    for _ in 0..ROUNDS {
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args(["crash_writer", "--exact", "--nocapture"])
            .env("CHASSIS_CRASH_PATH", &path)
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        std::thread::sleep(Duration::from_millis(rand::random_range(50..500)));
        assert!(child.try_wait().unwrap().is_none(), "writer exited before it was killed");
        child.kill().unwrap();
        child.wait().unwrap();
        for line in BufReader::new(child.stdout.take().unwrap()).lines() {
            if let Some(n) = line.unwrap().strip_prefix("flushed ") {
                flushed = n.parse().unwrap();
            }
        }

        let index = VectorIndex::open(&path, DIMS, options()).unwrap();
        let len = index.len();
        // A kill between flush() and the println leaves one batch more than was reported. Any
        // other count means a flush kept its adds but not its deletes, or the reverse.
        flushed = [flushed, flushed + 1]
            .into_iter()
            .find(|&b| live_after(b) == len)
            .unwrap_or_else(|| panic!("len {len} is not a batch boundary near batch {flushed}"));
        if flushed > 0 {
            let last = flushed * BATCH - 1;
            assert_eq!(index.search(&vector(last), 5).unwrap()[0].id, last);
            for id in deleted_by(flushed - 1) {
                assert!(index.search(&vector(id), 5).unwrap().iter().all(|h| h.id != id));
            }
            // The next batch's deletes were never committed, so they must have rolled back.
            for id in deleted_by(flushed) {
                assert_eq!(index.search(&vector(id), 1).unwrap()[0].id, id, "id {id} rolled back");
            }
        }
        drop(index);

        let storage = Storage::open(&path, DIMS).unwrap();
        assert_eq!(storage.count(), flushed * BATCH);
        for id in 0..storage.count() {
            assert_eq!(storage.get_vector_slice(id).unwrap(), vector(id).as_slice(), "vector {id}");
        }
    }
    assert!(flushed > 0, "no round got as far as a flush");
}
