//! This test verifies that Rust's borrow checker prevents calling `commit()`
//! while a slice from `get_vector_slice()` is alive.
//!
//! If this code compiles, there is a bug in the API design.

use chassis_core::Storage;

fn main() {
    let mut storage = Storage::open("/tmp/test.chassis", 128).unwrap();
    storage.insert(&vec![1.0; 128]).unwrap();
    
    let slice = storage.get_vector_slice(0).unwrap();
    
    // ERROR: Cannot call commit() because it requires &mut self,
    // but `slice` is still borrowing &self
    storage.commit().unwrap();
    
    println!("{}", slice[0]);
}
