//! This test verifies that a borrow of a slice extends through function returns,
//! preventing mutations even when the slice is returned.
//!
//! If this code compiles, there is a bug in the API design.

use chassis_core::Storage;

fn get_first_slice(storage: &Storage) -> &[f32] {
    storage.get_vector_slice(0).unwrap()
}

fn main() {
    let mut storage = Storage::open("/tmp/test.chassis", 128).unwrap();
    storage.insert(&vec![1.0; 128]).unwrap();
    
    let slice = get_first_slice(&storage);
    
    // ERROR: Cannot call insert() because the borrow from get_first_slice
    // extends through the return and `slice` is still borrowing &storage
    storage.insert(&vec![2.0; 128]).unwrap();
    
    println!("{}", slice[0]);
}
