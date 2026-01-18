//! This test verifies that a slice cannot outlive the Storage instance.
//!
//! If this code compiles, there is a bug in the API design.

use chassis_core::Storage;

fn main() {
    let slice = {
        let mut storage = Storage::open("/tmp/test.chassis", 128).unwrap();
        storage.insert(&vec![1.0; 128]).unwrap();
        
        // ERROR: Cannot return slice because it borrows from `storage`,
        // which will be dropped at the end of this block
        storage.get_vector_slice(0).unwrap()
    };
    
    println!("{}", slice[0]);
}
