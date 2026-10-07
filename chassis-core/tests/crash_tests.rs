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

/// Opens the index once its lock is free. A process another test forks holds a copy of every
/// open handle until it execs, and with it the lock of an index this test has just closed.
fn unlocked<T>(open: impl Fn() -> anyhow::Result<T>) -> T {
    for _ in 0..400 {
        match open() {
            Err(e) if e.to_string().contains("already open") => {
                std::thread::sleep(Duration::from_millis(5));
            }
            opened => return opened.unwrap(),
        }
    }
    panic!("the index stayed locked for two seconds")
}

/// The child half: only does work when the parent test spawns it with a path.
#[test]
fn crash_writer() {
    let Ok(path) = std::env::var("CHASSIS_CRASH_PATH") else { return };
    let mut index = unlocked(|| VectorIndex::open(&path, DIMS, options()));
    let mut batch = (0..).find(|&b| live_after(b) == index.len()).unwrap();
    let add_batch = std::env::var("CHASSIS_CRASH_ADD_BATCH").is_ok();
    let compact = std::env::var("CHASSIS_CRASH_COMPACT").is_ok();
    loop {
        let ids = batch * BATCH..(batch + 1) * BATCH;
        if add_batch {
            let vectors: Vec<f32> = ids.clone().flat_map(vector).collect();
            assert_eq!(index.add_batch(&vectors).unwrap(), ids.collect::<Vec<_>>());
            println!("added with add_batch");
        } else {
            for id in ids {
                assert_eq!(index.add(&vector(id)).unwrap(), id);
            }
        }
        for id in deleted_by(batch) {
            assert!(index.delete(id).unwrap());
        }
        index.flush().unwrap();
        batch += 1;
        println!("flushed {batch}");
        if compact && batch % 2 == 0 {
            println!("compacting");
            index.compact().unwrap();
        }
    }
}

#[test]
fn test_kill_at_random_points_keeps_flushed_data() {
    kill_at_random_points(false, false);
}

/// The same, with each batch added by `add_batch`, so kills land while threads link it.
#[test]
fn test_kill_during_parallel_batches_keeps_flushed_data() {
    kill_at_random_points(true, false);
}

/// The same, compacting after every second flush, so kills land while the copy is built, flagged
/// and renamed. Compaction changes no vector or id, so every check on them still holds.
#[test]
fn test_kill_during_compaction_keeps_flushed_data() {
    kill_at_random_points(false, true);
}

fn kill_at_random_points(add_batch: bool, compact: bool) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("crash.chassis");
    let (mut flushed, mut batched, mut compacted) = (0, false, false);

    for _ in 0..ROUNDS {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command.args(["crash_writer", "--exact", "--nocapture"]).env("CHASSIS_CRASH_PATH", &path);
        if add_batch {
            command.env("CHASSIS_CRASH_ADD_BATCH", "1");
        }
        if compact {
            command.env("CHASSIS_CRASH_COMPACT", "1");
        }
        let mut child = command.stdout(Stdio::piped()).spawn().unwrap();
        std::thread::sleep(Duration::from_millis(rand::random_range(50..500)));
        assert!(child.try_wait().unwrap().is_none(), "writer exited before it was killed");
        child.kill().unwrap();
        child.wait().unwrap();
        for line in BufReader::new(child.stdout.take().unwrap()).lines() {
            let line = line.unwrap();
            batched |= line == "added with add_batch";
            compacted |= line == "compacting";
            if let Some(n) = line.strip_prefix("flushed ") {
                flushed = n.parse().unwrap();
            }
        }

        let index = unlocked(|| VectorIndex::open(&path, DIMS, options()));
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

        // Compaction moves vectors to other slots, so check them through the index instead.
        if compact {
            let index = unlocked(|| VectorIndex::open(&path, DIMS, options()));
            assert!(!dir.path().join("crash.chassis.compacting").exists(), "a killed copy remains");
            let deleted: Vec<u64> = (0..flushed).flat_map(deleted_by).collect();
            for id in (0..flushed * BATCH).step_by(7) {
                let found = index.search(&vector(id), 5).unwrap().iter().any(|h| h.id == id);
                assert_eq!(found, !deleted.contains(&id), "id {id} after {flushed} batches");
            }
            continue;
        }
        let storage = unlocked(|| Storage::open(&path, DIMS));
        assert_eq!(storage.count(), flushed * BATCH);
        for id in 0..storage.count() {
            assert_eq!(storage.get_vector_slice(id).unwrap(), vector(id).as_slice(), "vector {id}");
        }
    }
    assert!(flushed > 0, "no round got as far as a flush");
    assert_eq!(batched, add_batch, "the writer didn't add the way this test asked");
    assert_eq!(compacted, compact, "the writer didn't compact when this test asked");
}
