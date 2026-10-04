//! Power-loss simulation (ADR-0008, "Before Accepting" #4).
//!
//! In tests, `Storage::commit` hands this module the file's bytes just before and just after each
//! fsync. At every crash point each 512-byte sector of a crash image keeps either its last durable
//! version or its current one, and the file length is either. Every image must reopen to exactly
//! the last completed flush or the one in progress, and keep accepting writes.
//!
//! Limitation: a sector that was written twice since the last fsync is only ever tried in its
//! durable or its latest version, not an intermediate one.

use crate::{IndexOptions, VectorIndex};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

const SECTOR: usize = 512;
/// 1 KiB vectors, so a short workload outgrows the test-sized graph slack and moves the graph.
const DIMS: u32 = 256;
const BATCHES: u64 = 10;
const ADDS_PER_BATCH: u64 = 24;
const DELETES_PER_BATCH: usize = 3;
const IMAGES_PER_CRASH_POINT: usize = 4;

fn options() -> IndexOptions {
    IndexOptions { ef_construction: 32, ..IndexOptions::default() }
}

fn vector(id: u64) -> Vec<f32> {
    vec![id as f32; DIMS as usize]
}

/// A committed state: the ids that are live, and one past the largest id ever added.
#[derive(Clone, Debug, PartialEq)]
struct State {
    live: BTreeSet<u64>,
    next_id: u64,
}

struct Recorder {
    /// File bytes at the last completed fsync.
    durable: Vec<u8>,
    /// States a crash right now may recover to.
    accept: Vec<State>,
    image: PathBuf,
    rng: StdRng,
    crash_points: usize,
}

thread_local! {
    static RECORDER: RefCell<Option<Recorder>> = const { RefCell::new(None) };
}

pub(crate) fn before_fsync(bytes: &[u8]) {
    crash_point(bytes);
}

pub(crate) fn after_fsync(bytes: &[u8]) {
    RECORDER.with(|r| {
        if let Some(recorder) = r.borrow_mut().as_mut() {
            recorder.durable = bytes.to_vec();
        }
    });
}

/// Tries crash images of the current file against the accepted states.
fn crash_point(current: &[u8]) {
    // Taken out while it runs, so the fsyncs of the images it opens don't recurse into it.
    let Some(mut recorder) = RECORDER.with(|r| r.borrow_mut().take()) else { return };
    recorder.crash_points += 1;
    for variant in 0..IMAGES_PER_CRASH_POINT {
        let image = compose(&recorder.durable, current, variant, &mut recorder.rng);
        std::fs::write(&recorder.image, &image).unwrap();
        verify(&recorder.image, &recorder.accept, variant);
    }
    RECORDER.with(|r| *r.borrow_mut() = Some(recorder));
}

/// Variant 0 is all durable, 1 all current; the rest pick each sector and the length at random.
fn compose(durable: &[u8], current: &[u8], variant: usize, rng: &mut StdRng) -> Vec<u8> {
    let len = match variant {
        0 => durable.len(),
        1 => current.len(),
        _ => *[durable.len(), current.len()].get(rng.random_range(0..2)).unwrap(),
    };
    let sector_of = |bytes: &[u8], start: usize| -> Vec<u8> {
        let mut sector = vec![0; SECTOR.min(len - start)];
        if let Some(rest) = bytes.get(start..) {
            let available = rest.len().min(sector.len());
            sector[..available].copy_from_slice(&rest[..available]);
        }
        sector
    };
    let mut image = Vec::with_capacity(len);
    for start in (0..len).step_by(SECTOR) {
        let from_current = match variant {
            0 => false,
            1 => true,
            _ => rng.random_bool(0.5),
        };
        image.extend(sector_of(if from_current { current } else { durable }, start));
    }
    image
}

fn verify(path: &Path, accept: &[State], variant: usize) {
    let context = || format!("crash image variant {variant}, accepted states {accept:?}");
    let index = VectorIndex::open(path, DIMS, options())
        .unwrap_or_else(|e| panic!("reopen failed: {e:#}; {}", context()));
    let state = accept
        .iter()
        .find(|s| s.live.len() as u64 == index.len())
        .unwrap_or_else(|| panic!("len {} matches no accepted state; {}", index.len(), context()));

    for &id in &state.live {
        let hit = &index.search(&vector(id), 1).unwrap()[0];
        assert_eq!(hit.id, id, "live id {id} not found; {}", context());
    }
    for id in (0..state.next_id).filter(|id| !state.live.contains(id)) {
        let hits = index.search(&vector(id), 3).unwrap();
        assert!(hits.iter().all(|h| h.id != id), "deleted id {id} returned; {}", context());
    }

    // The recovered index must keep working: one more add, flush and reopen.
    drop(index);
    let mut index = VectorIndex::open(path, DIMS, options()).unwrap();
    let id = index.add(&vector(state.next_id)).unwrap();
    index.flush().unwrap();
    drop(index);
    let index = VectorIndex::open(path, DIMS, options()).unwrap();
    assert_eq!(index.len(), state.live.len() as u64 + 1, "after continuing; {}", context());
    assert_eq!(index.search(&vector(state.next_id), 1).unwrap()[0].id, id);
}

#[test]
fn test_power_loss_at_every_fsync_and_operation() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index.chassis");
    let mut index = VectorIndex::open(&path, DIMS, options()).unwrap();
    index.flush().unwrap();

    let mut committed = State { live: BTreeSet::new(), next_id: 0 };
    RECORDER.with(|r| {
        *r.borrow_mut() = Some(Recorder {
            durable: std::fs::read(&path).unwrap(),
            accept: vec![committed.clone()],
            image: dir.path().join("image.chassis"),
            rng: StdRng::seed_from_u64(7),
            crash_points: 0,
        });
    });
    let set_accept = |states: Vec<State>| {
        RECORDER.with(|r| r.borrow_mut().as_mut().unwrap().accept = states);
    };
    // Read through the index's own mapping: on Windows its file lock blocks other handles' reads.
    let between_operations = |index: &VectorIndex| crash_point(index.graph.storage.file_bytes());

    let mut rng = StdRng::seed_from_u64(11);
    for _ in 0..BATCHES {
        let mut next = committed.clone();
        for _ in 0..ADDS_PER_BATCH {
            let id = index.add(&vector(next.next_id)).unwrap();
            assert_eq!(id, next.next_id);
            next.live.insert(id);
            next.next_id += 1;
        }
        between_operations(&index);
        for _ in 0..DELETES_PER_BATCH {
            let live: Vec<u64> = next.live.iter().copied().collect();
            let id = live[rng.random_range(0..live.len())];
            assert!(index.delete(id).unwrap());
            next.live.remove(&id);
        }
        between_operations(&index);

        set_accept(vec![committed.clone(), next.clone()]);
        index.flush().unwrap();
        committed = next;
        set_accept(vec![committed.clone()]);
    }

    let recorder = RECORDER.with(|r| r.borrow_mut().take()).unwrap();
    // Two points between operations per batch, plus at least two fsyncs per flush.
    assert!(
        recorder.crash_points as u64 >= BATCHES * 4,
        "only {} crash points",
        recorder.crash_points
    );
    println!(
        "power loss: {} crash points, {} images",
        recorder.crash_points,
        recorder.crash_points * IMAGES_PER_CRASH_POINT
    );
}
