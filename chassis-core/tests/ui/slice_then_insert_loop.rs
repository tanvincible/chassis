//! This test verifies that Rust's borrow checker prevents calling `insert()`
//! in a loop while holding a slice from `get_vector_slice()`.
//!
//! If this code compiles, there is a bug in the API design.

use chassis_core::Storage;

fn main() {
    let mut storage = Storage::open("/tmp/test.chassis", 128).unwrap();
    storage.insert(&vec![1.0; 128]).unwrap();
    
    let slice = storage.get_vector_slice(0).unwrap();
    
    // ERROR: Cannot insert in a loop because insert() requires &mut self,
    // but `slice` is still borrowing &self
    for i in 0..10 {
        storage.insert(&vec![i as f32; 128]).unwrap();
    }
    
    println!("{}", slice[0]);
}
