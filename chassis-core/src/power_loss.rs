//! Power-loss simulation (ADR-0008, "Before Accepting" #4).
//!
//! In tests, `Storage` hands this module the file's bytes just before and just after each fsync.
//! At every crash point each 512-byte sector of a crash image keeps its last durable version, its
//! current one, or is torn into a mix of the two's 8-byte words; the file length is either one.
//! Every image must reopen to exactly the last completed flush or the one in progress, then take
//! one more add-and-delete flush. Reopening an image may run recovery, whose own fsyncs are crash
//! points too. Before that, a reader opens the image as found, without recovery: it must show one
//! of those states and never fail a search.
//!
//! Limitations: a sector written twice since the last fsync is never tried in an intermediate
//! version, and a write never tears inside an aligned 8-byte word.

use crate::{IndexOptions, IndexReader, VectorIndex};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cell::{Cell, RefCell};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

const SECTOR: usize = 512;
/// 1 KiB vectors, so a short workload crosses the test-sized segments and heap chunks.
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
    /// Installed for one reopen, to crash inside the recovery it runs.
    in_recovery: bool,
}

thread_local! {
    static RECORDER: RefCell<Option<Recorder>> = const { RefCell::new(None) };
    static RECOVERY_CRASH_POINTS: Cell<usize> = const { Cell::new(0) };
    static SIMULATING: Cell<bool> = const { Cell::new(false) };
    /// Searches by readers of crash images, and the live vectors they didn't find first.
    static READER_SEARCHES: Cell<usize> = const { Cell::new(0) };
    static READER_MISSES: Cell<usize> = const { Cell::new(0) };
}

/// Whether this thread runs the simulation; `Storage` then calls the hooks instead of syncing.
pub(crate) fn simulating() -> bool {
    SIMULATING.get()
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
    if recorder.in_recovery {
        RECOVERY_CRASH_POINTS.set(RECOVERY_CRASH_POINTS.get() + 1);
    }
    // Inside recovery, all-durable is the image recovery started from: only random mixes are new.
    let first = if recorder.in_recovery { 2 } else { 0 };
    for variant in first..IMAGES_PER_CRASH_POINT {
        let image = compose(&recorder.durable, current, variant, &mut recorder.rng);
        std::fs::write(&recorder.image, &image).unwrap();
        verify(&recorder.image, &recorder.accept, variant, !recorder.in_recovery);
    }
    RECORDER.with(|r| *r.borrow_mut() = Some(recorder));
}

/// Variant 0 is all durable, 1 all current; the rest pick each sector and the length at random.
/// Torn sectors matter for headers: one fits in a sector, so whole-sector choices never tear it.
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
        let (old, new) = (sector_of(durable, start), sector_of(current, start));
        let sector = match variant {
            0 => old,
            1 => new,
            _ if old == new || rng.random_bool(0.8) => {
                if rng.random_bool(0.5) {
                    new
                } else {
                    old
                }
            }
            _ => {
                let words = old.chunks(8).zip(new.chunks(8));
                words.flat_map(|(o, n)| if rng.random_bool(0.5) { n } else { o }).copied().collect()
            }
        };
        image.extend(sector);
    }
    image
}

fn verify(path: &Path, accept: &[State], variant: usize, crash_in_recovery: bool) {
    let context = || format!("crash image variant {variant}, accepted states {accept:?}");
    match IndexReader::open(path, DIMS, options()) {
        Ok(reader) => check_reader(reader, accept, &context),
        // Only while the file is being created can there be no index yet.
        Err(_) if accept.iter().all(|s| s.next_id == 0) => {}
        Err(e) => panic!("reader open failed: {e:#}; {}", context()),
    }
    if crash_in_recovery {
        let recorder = Recorder {
            durable: std::fs::read(path).unwrap(),
            accept: accept.to_vec(),
            image: path.with_extension("recovering"),
            rng: StdRng::seed_from_u64(variant as u64),
            crash_points: 0,
            in_recovery: true,
        };
        RECORDER.with(|r| *r.borrow_mut() = Some(recorder));
    }
    let opened = VectorIndex::open(path, DIMS, options());
    RECORDER.with(|r| r.borrow_mut().take());
    let mut index = opened.unwrap_or_else(|e| panic!("reopen failed: {e:#}; {}", context()));
    let state = accept
        .iter()
        .find(|s| s.live.len() as u64 == index.len())
        .unwrap_or_else(|| panic!("len {} matches no accepted state; {}", index.len(), context()))
        .clone();
    check(&index, &state, &context);

    // The recovered index must keep working: one more add and delete, flush and reopen.
    let mut next = state.clone();
    let id = index.add(&vector(next.next_id)).unwrap();
    assert_eq!(id, next.next_id, "{}", context());
    next.live.insert(id);
    next.next_id += 1;
    let first = *next.live.first().unwrap();
    assert!(index.delete(first).unwrap());
    next.live.remove(&first);
    index.flush().unwrap();
    drop(index);
    let index = VectorIndex::open(path, DIMS, options()).unwrap();
    assert_eq!(index.len(), next.live.len() as u64, "after continuing; {}", context());
    check(&index, &next, &context);
}

/// A reader of the image as found, before any recovery (ADR-0008 "Before Accepting" 4b): it shows
/// an accepted state and returns only its live ids. It may route through slots the power loss
/// emptied, so it can miss some; those are counted, not failed.
fn check_reader(mut reader: IndexReader, accept: &[State], context: &dyn Fn() -> String) {
    let state = accept
        .iter()
        .find(|s| s.live.len() as u64 == reader.len())
        .unwrap_or_else(|| panic!("reader len {} matches no state; {}", reader.len(), context()));
    for &id in &state.live {
        let hits = reader
            .search(&vector(id), 3)
            .unwrap_or_else(|e| panic!("reader search failed: {e:#}; {}", context()));
        assert!(
            hits.iter().all(|h| state.live.contains(&h.id)),
            "reader returned {hits:?}; {}",
            context()
        );
        READER_SEARCHES.set(READER_SEARCHES.get() + 1);
        if hits.first().is_none_or(|h| h.id != id) {
            READER_MISSES.set(READER_MISSES.get() + 1);
        }
    }
}

/// Every live id is found, and no deleted one is returned.
fn check(index: &VectorIndex, state: &State, context: &dyn Fn() -> String) {
    for &id in &state.live {
        let hit = &index.search(&vector(id), 1).unwrap()[0];
        assert_eq!(hit.id, id, "live id {id} not found; {}", context());
    }
    for id in (0..state.next_id).filter(|id| !state.live.contains(id)) {
        let hits = index.search(&vector(id), 3).unwrap();
        assert!(hits.iter().all(|h| h.id != id), "deleted id {id} returned; {}", context());
    }
}

#[test]
fn test_power_loss_at_every_fsync_and_operation() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index.chassis");
    let mut index = VectorIndex::open(&path, DIMS, options()).unwrap();
    index.flush().unwrap();
    SIMULATING.set(true);

    let mut committed = State { live: BTreeSet::new(), next_id: 0 };
    RECORDER.with(|r| {
        *r.borrow_mut() = Some(Recorder {
            durable: std::fs::read(&path).unwrap(),
            accept: vec![committed.clone()],
            image: dir.path().join("image.chassis"),
            rng: StdRng::seed_from_u64(7),
            crash_points: 0,
            in_recovery: false,
        });
    });
    let set_accept = |states: Vec<State>| {
        RECORDER.with(|r| r.borrow_mut().as_mut().unwrap().accept = states);
    };
    // Read through the index's own mapping: on Windows its file lock blocks other handles' reads.
    let between_operations = |index: &VectorIndex| crash_point(&index.graph.storage.file_bytes());

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
        for n in 0..DELETES_PER_BATCH {
            let live: Vec<u64> = next.live.iter().copied().collect();
            // The first delete is of a vector added in this same flush, in the slot that an add
            // after a crash reuses first.
            let first_added = next.next_id - ADDS_PER_BATCH;
            let id = if n == 0 { first_added } else { live[rng.random_range(0..live.len())] };
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
    SIMULATING.set(false);
    let in_recovery = RECOVERY_CRASH_POINTS.get();
    // Two points between operations per batch, plus three fsyncs per flush with deletes.
    assert!(
        recorder.crash_points as u64 >= BATCHES * 5,
        "only {} crash points",
        recorder.crash_points
    );
    assert!(in_recovery > 0, "no crash image needed recovery");
    let images = recorder.crash_points * IMAGES_PER_CRASH_POINT + in_recovery * 2;
    println!(
        "power loss: {} crash points ({in_recovery} inside recovery), {images} images; readers of \
         them missed {} of {} live vectors",
        recorder.crash_points + in_recovery,
        READER_MISSES.get(),
        READER_SEARCHES.get()
    );
}

/// Creating a file: its one fsync can leave it empty, created, or with both header copies torn.
#[test]
fn test_power_loss_while_creating() {
    SIMULATING.set(true);
    for seed in 0..100 {
        let dir = tempfile::tempdir().unwrap();
        let empty = State { live: BTreeSet::new(), next_id: 0 };
        RECORDER.with(|r| {
            *r.borrow_mut() = Some(Recorder {
                durable: Vec::new(),
                accept: vec![empty],
                image: dir.path().join("image.chassis"),
                rng: StdRng::seed_from_u64(seed),
                crash_points: 0,
                in_recovery: false,
            });
        });
        VectorIndex::open(dir.path().join("index.chassis"), DIMS, options()).unwrap();
        assert_eq!(RECORDER.with(|r| r.borrow_mut().take()).unwrap().crash_points, 1);
    }
    SIMULATING.set(false);
}
