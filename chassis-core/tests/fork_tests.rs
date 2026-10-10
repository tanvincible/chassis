//! A forked child can drop an index its parent was reading in (ADR-0019). In a test binary of its
//! own: a child holds every file its process has open, other tests' locks among them, until it
//! exits, and a test reopening its index then would find it locked.
#![cfg(unix)]

use chassis_core::{IndexOptions, VectorIndex};

#[test]
fn test_a_forked_child_can_drop_an_index_being_read_in() {
    let dir = tempfile::tempdir().unwrap();
    let mut index =
        VectorIndex::open(dir.path().join("index.chassis"), 256, IndexOptions::default()).unwrap();
    let vectors: Vec<f32> = (0..256 * 2000).map(|i| (i as f32 * 0.37).sin()).collect();
    index.add_batch(&vectors).unwrap();
    index.flush().unwrap();
    index.warm();
    // SAFETY: the child drops one value and exits.
    let child = unsafe { libc::fork() };
    if child == 0 {
        let dropped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(index)));
        unsafe { libc::_exit(i32::from(dropped.is_err())) };
    }
    let mut status = -1;
    assert_eq!(unsafe { libc::waitpid(child, &mut status, 0) }, child);
    assert_eq!(status, 0, "the child panicked dropping the index");
}
