//! A full disk is an error with a `help:` line, never a crash, and the index opens again to its last
//! flush once there's room (ADR-0008). It fills the small file system `CHASSIS_FULL_DISK` names,
//! which CI mounts: `cargo test -p chassisdb --test full_disk_tests -- --ignored`.

use chassisdb::{ErrorKind, IndexOptions, VectorIndex};
use std::path::Path;
use std::time::{Duration, Instant};

const DIMS: usize = 256;

fn vector(id: u64) -> Vec<f32> {
    (0..DIMS).map(|i| ((id as usize * 31 + i * 7) % 997) as f32 / 997.0).collect()
}

#[test]
#[ignore = "fills a file system: set CHASSIS_FULL_DISK to a small one"]
fn a_full_disk_is_an_error_and_the_last_flush_is_kept() {
    let dir = std::env::var("CHASSIS_FULL_DISK").expect("CHASSIS_FULL_DISK names a file system");
    let dir = Path::new(&dir);
    // Room to make once the disk is full: on APFS and Btrfs, which copy a page on every write,
    // putting the saved lists back takes new blocks.
    std::fs::write(dir.join("room"), vec![1u8; 24 << 20]).unwrap();
    let path = dir.join("full.chassis");
    let (mut flushed, mut next) = (0, 0);
    let error = {
        let mut index = VectorIndex::open(&path, DIMS as u32, IndexOptions::default()).unwrap();
        loop {
            let ids: Vec<u64> = (next..next + 1000).collect();
            let vectors: Vec<f32> = ids.iter().flat_map(|&id| vector(id)).collect();
            match index.add_batch_with_ids(&ids, &vectors).and_then(|()| index.flush()) {
                Ok(()) => (flushed, next) = (index.len(), next + 1000),
                Err(e) => break e,
            }
        }
    };
    assert_eq!(error.kind(), ErrorKind::Io, "{error}");
    assert!(error.to_string().contains("\nhelp: "), "{error}");
    assert!(flushed > 0, "the disk filled before the first flush: {error}");

    std::fs::remove_file(dir.join("room")).unwrap();
    // APFS can refuse a writer for a few seconds more after room is made.
    let start = Instant::now();
    let index = loop {
        match VectorIndex::open(&path, DIMS as u32, IndexOptions::default()) {
            Ok(index) => break index,
            Err(e) if e.kind() == ErrorKind::Io && start.elapsed() < Duration::from_secs(30) => {
                std::thread::sleep(Duration::from_secs(1));
            }
            Err(e) => panic!("{e}"),
        }
    };
    assert_eq!(index.len(), flushed);
    assert!(index.check().unwrap().not_finite.is_empty());
    for id in [0, flushed / 2, flushed - 1] {
        assert_eq!(index.get(id).unwrap(), Some(vector(id)), "vector {id}");
    }
}
