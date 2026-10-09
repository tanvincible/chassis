//! `IndexOptions::warm` (ADR-0019): asking changes no result, an index can go away while it is
//! being read in, and where a test can put a file out of memory, asking brings it back.

use chassis_core::{IndexOptions, IndexReader, SearchResult, VectorIndex};
use tempfile::tempdir;

const DIMS: u32 = 64;

/// Pseudo-random, so no two vectors tie.
fn vector(id: u64) -> Vec<f32> {
    (0..u64::from(DIMS))
        .map(|d| {
            let mut x = (id * u64::from(DIMS) + d).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            x = (x ^ (x >> 31)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            (x >> 40) as f32 / (1u64 << 24) as f32
        })
        .collect()
}

fn options(warm: bool) -> IndexOptions {
    IndexOptions { ef_construction: 16, ef_search: 32, warm, ..IndexOptions::default() }
}

fn found(results: Vec<SearchResult>) -> Vec<(u64, u32)> {
    results.iter().map(|r| (r.id, r.distance.to_bits())).collect()
}

#[test]
fn test_warming_changes_no_result() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("index.chassis");
    let mut index = VectorIndex::open(&path, DIMS, options(true)).unwrap();
    index.add_batch(&(0..2000).flat_map(vector).collect::<Vec<f32>>()).unwrap();
    // Asked for again while adds go on: what is there by then is read in.
    index.warm();
    for id in 2000..2100 {
        index.add(&vector(id)).unwrap();
    }
    index.flush().unwrap();

    let mut with = IndexReader::open(&path, DIMS, options(true)).unwrap();
    let mut without = IndexReader::open(&path, DIMS, options(false)).unwrap();
    with.warm();
    for id in (0..2100).step_by(41) {
        let wanted = found(without.search(&vector(id), 5).unwrap());
        assert_eq!(wanted[0].0, id);
        assert_eq!(found(with.search(&vector(id), 5).unwrap()), wanted);
        assert_eq!(found(index.search(&vector(id), 5).unwrap()), wanted);
    }

    // Deletes, a compaction and a reopen keep the vectors, and the option goes with them.
    // Windows can't replace a file a reader has open.
    drop((with, without));
    for id in (0..2100).step_by(3) {
        assert!(index.delete(id).unwrap());
    }
    index.compact().unwrap();
    drop(index);
    let index = VectorIndex::open(&path, DIMS, options(true)).unwrap();
    let mut reader = IndexReader::open(&path, DIMS, options(true)).unwrap();
    assert_eq!((index.len(), reader.len()), (1400, 1400));
    for id in [1, 1000, 2099] {
        assert_eq!(index.search(&vector(id), 1).unwrap()[0].id, id);
        assert_eq!(reader.search(&vector(id), 1).unwrap()[0].id, id);
    }
}

#[test]
fn test_an_index_can_go_away_while_it_is_read_in() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("index.chassis");
    let mut index = VectorIndex::open(&path, DIMS, options(false)).unwrap();
    index.add_batch(&(0..3000).flat_map(vector).collect::<Vec<f32>>()).unwrap();
    index.flush().unwrap();
    for round in 0..30 {
        let mut reader = IndexReader::open(&path, DIMS, options(true)).unwrap();
        if round % 2 == 0 {
            assert_eq!(reader.search(&vector(round), 1).unwrap()[0].id, round);
        }
    }
    // The writer too, with a read under way when it is replaced by its compacted copy.
    index.warm();
    index.compact().unwrap();
    assert_eq!(index.search(&vector(7), 1).unwrap()[0].id, 7);
}

/// The share of the file past its headers that is in memory, by the system's own account.
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn in_memory(path: &std::path::Path) -> f64 {
    let file = std::fs::File::open(path).unwrap();
    // SAFETY: mapped only to ask the system about its pages; nothing is read through it.
    let map = unsafe { memmap2::Mmap::map(&file) }.unwrap();
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    let mut pages = vec![0u8; map.len().div_ceil(page)];
    let asked =
        unsafe { libc::mincore(map.as_ptr() as *mut _, map.len(), pages.as_mut_ptr().cast()) };
    assert_eq!(asked, 0, "mincore failed");
    // Two header copies and the live page, 64 KiB each, are there as soon as anything opens it.
    let data = &pages[(3 * 64 * 1024) / page..];
    data.iter().filter(|&&p| p & 1 == 1).count() as f64 / data.len() as f64
}

/// Puts the file's pages out of memory, where the system lets a user.
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn put_out_of_memory(path: &std::path::Path) {
    let file = std::fs::File::open(path).unwrap();
    file.sync_all().unwrap();
    #[cfg(target_os = "linux")]
    {
        use std::os::fd::AsRawFd;
        // SAFETY: advice about an open file.
        unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
    }
    #[cfg(target_os = "macos")]
    {
        // SAFETY: the mapping is only named to the system.
        let map = unsafe { memmap2::Mmap::map(&file) }.unwrap();
        unsafe { libc::msync(map.as_ptr() as *mut _, map.len(), libc::MS_INVALIDATE) };
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn test_warming_reads_the_file_into_memory() {
    use std::time::{Duration, Instant};

    let dir = tempdir().unwrap();
    let path = dir.path().join("index.chassis");
    // Seven segments, each full, so the file has no unwritten end.
    let slots = 1024 * ((1 << 3) - 1);
    let mut index = VectorIndex::open(&path, DIMS, options(false)).unwrap();
    index.add_batch(&(0..slots).flat_map(vector).collect::<Vec<f32>>()).unwrap();
    index.flush().unwrap();
    drop(index);

    put_out_of_memory(&path);
    if in_memory(&path) > 0.2 {
        eprintln!("this system keeps the file in memory whatever it is told: nothing to test");
        return;
    }
    // Opening it without asking brings in little, however long it stays open.
    let reader = IndexReader::open(&path, DIMS, options(false)).unwrap();
    std::thread::sleep(Duration::from_millis(200));
    let unasked = in_memory(&path);
    assert!(unasked < 0.5, "{unasked} of the file in memory without asking");
    drop(reader);

    put_out_of_memory(&path);
    let _reader = IndexReader::open(&path, DIMS, options(true)).unwrap();
    let start = Instant::now();
    while in_memory(&path) < 0.85 {
        assert!(start.elapsed() < Duration::from_secs(30), "only {} read in", in_memory(&path));
        std::thread::sleep(Duration::from_millis(10));
    }
}
