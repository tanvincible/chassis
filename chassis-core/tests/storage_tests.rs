use chassis_core::Storage;
use tempfile::NamedTempFile;

#[test]
fn test_create_and_open() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();

    // Create new storage
    let storage = Storage::open(path, 128).unwrap();
    assert_eq!(storage.count(), 0);
    assert_eq!(storage.dimensions(), 128);
    drop(storage);

    // Reopen existing storage
    let storage = Storage::open(path, 128).unwrap();
    assert_eq!(storage.count(), 0);
    assert_eq!(storage.dimensions(), 128);
}

#[test]
fn test_dimension_mismatch() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();

    // Create with 128 dimensions
    let storage = Storage::open(path, 128).unwrap();
    drop(storage);

    // Try to open with different dimensions
    let result = Storage::open(path, 256);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("Dimension mismatch"));
}

#[test]
fn test_insert_and_retrieve() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();

    let mut storage = Storage::open(path, 128).unwrap();

    // Insert a vector
    let vector = vec![1.0, 2.0, 3.0].into_iter()
        .cycle()
        .take(128)
        .collect::<Vec<_>>();
    
    let id = storage.insert(&vector).unwrap();
    assert_eq!(id, 0);
    assert_eq!(storage.count(), 1);

    // Retrieve the vector
    let retrieved = storage.get_vector(0).unwrap();
    assert_eq!(retrieved, vector);
}

#[test]
fn test_insert_multiple() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();

    let mut storage = Storage::open(path, 128).unwrap();

    // Insert multiple vectors
    for i in 0..100 {
        let vector = vec![i as f32; 128];
        let id = storage.insert(&vector).unwrap();
        assert_eq!(id, i);
    }

    assert_eq!(storage.count(), 100);

    // Verify all vectors
    for i in 0..100 {
        let retrieved = storage.get_vector(i).unwrap();
        assert_eq!(retrieved, vec![i as f32; 128]);
    }
}

#[test]
fn test_commit_durability() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();

    {
        let mut storage = Storage::open(path, 128).unwrap();
        let vector = vec![42.0; 128];
        storage.insert(&vector).unwrap();
        
        // Explicit commit for durability
        storage.commit().unwrap();
    }

    // Reopen and verify data persisted
    let storage = Storage::open(path, 128).unwrap();
    assert_eq!(storage.count(), 1);
    let retrieved = storage.get_vector(0).unwrap();
    assert_eq!(retrieved, vec![42.0; 128]);
}

#[test]
fn test_exclusive_locking() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();

    // First instance owns the lock
    let _storage1 = Storage::open(path, 128).unwrap();

    // Second instance should fail to open the same file
    let storage2 = Storage::open(path, 128);
    assert!(storage2.is_err());
    assert!(storage2
        .unwrap_err()
        .to_string()
        .contains("already open"));
}

#[test]
fn test_lock_released_on_drop() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();

    {
        let _storage = Storage::open(path, 128).unwrap();
        // Lock held here
    } // Lock released when _storage is dropped

    // Should succeed because lock was released
    let storage2 = Storage::open(path, 128);
    assert!(storage2.is_ok(), "Lock should be released after drop");
}

#[test]
fn test_file_growth_is_page_aligned() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();

    let mut storage = Storage::open(path, 768).unwrap(); // 768 * 4 = 3072 bytes

    storage.insert(&vec![0.0; 768]).unwrap();
    storage.insert(&vec![0.0; 768]).unwrap();
    storage.commit().unwrap();

    let meta = std::fs::metadata(path).unwrap();
    assert_eq!(
        meta.len() % 4096,
        0,
        "File size must be page-aligned (4KB)"
    );
}

#[test]
fn test_out_of_bounds_read() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();

    let storage = Storage::open(path, 128).unwrap();

    // Try to read from empty storage
    let result = storage.get_vector(0);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("out of bounds"));
}

#[test]
fn test_wrong_dimension_insert() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();

    let mut storage = Storage::open(path, 128).unwrap();

    // Try to insert vector with wrong dimensions
    let wrong_vector = vec![1.0; 256];
    let result = storage.insert(&wrong_vector);
    
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("dimension mismatch"));
}

#[test]
fn test_persistence_across_reopens() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();

    // Insert data and close
    {
        let mut storage = Storage::open(path, 128).unwrap();
        for i in 0..10 {
            storage.insert(&vec![i as f32; 128]).unwrap();
        }
        storage.commit().unwrap();
    }

    // Reopen and verify
    {
        let storage = Storage::open(path, 128).unwrap();
        assert_eq!(storage.count(), 10);
        
        for i in 0..10 {
            let v = storage.get_vector(i).unwrap();
            assert_eq!(v, vec![i as f32; 128]);
        }
    }

    // Reopen again and add more
    {
        let mut storage = Storage::open(path, 128).unwrap();
        assert_eq!(storage.count(), 10);
        
        storage.insert(&vec![99.0; 128]).unwrap();
        storage.commit().unwrap();
    }

    // Final verification
    let storage = Storage::open(path, 128).unwrap();
    assert_eq!(storage.count(), 11);
    let last = storage.get_vector(10).unwrap();
    assert_eq!(last, vec![99.0; 128]);
}

#[test]
fn test_raw_fsync_cost() {
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::os::unix::io::AsRawFd;
    use std::time::Instant;
    
    let temp = tempfile::NamedTempFile::new().unwrap();
    let mut file = OpenOptions::new()
        .write(true)
        .open(temp.path())
        .unwrap();
    
    file.write_all(b"test").unwrap();
    
    let start = Instant::now();
    unsafe { libc::fsync(file.as_raw_fd()); }
    let elapsed = start.elapsed();
    
    println!("Raw fsync took: {:?}", elapsed);
}

// ===== Zero-Copy Vector Slice Tests =====

#[test]
fn test_get_vector_slice_returns_correct_data() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();
    
    let mut storage = Storage::open(path, 128).unwrap();
    
    // Insert a vector with known pattern
    let vector = (0..128).map(|i| i as f32).collect::<Vec<_>>();
    storage.insert(&vector).unwrap();
    
    // Get slice and verify data
    let slice = storage.get_vector_slice(0).unwrap();
    assert_eq!(slice.len(), 128);
    
    for (i, &val) in slice.iter().enumerate() {
        assert_eq!(val, i as f32, "Mismatch at index {}", i);
    }
}

#[test]
fn test_get_vector_slice_matches_get_vector() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();
    
    let mut storage = Storage::open(path, 768).unwrap();
    
    // Insert multiple vectors
    for i in 0..10 {
        let vector = vec![i as f32 * 1.5; 768];
        storage.insert(&vector).unwrap();
    }
    
    // Verify slice and owned vector match
    for i in 0..10 {
        let slice = storage.get_vector_slice(i).unwrap();
        let owned = storage.get_vector(i).unwrap();
        
        assert_eq!(slice.len(), owned.len());
        assert_eq!(slice, &owned[..]);
    }
}

#[test]
fn test_get_vector_slice_out_of_bounds() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();
    
    let storage = Storage::open(path, 128).unwrap();
    
    // Try to read from empty storage
    let result = storage.get_vector_slice(0);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("out of bounds"));
}

#[test]
fn test_get_vector_slice_out_of_bounds_with_data() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();
    
    let mut storage = Storage::open(path, 128).unwrap();
    storage.insert(&vec![1.0; 128]).unwrap();
    
    // Try to read beyond the last vector
    let result = storage.get_vector_slice(1);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("out of bounds"));
    
    // Verify we can still read the valid index
    let slice = storage.get_vector_slice(0).unwrap();
    assert_eq!(slice.len(), 128);
}

#[test]
fn test_get_vector_slice_correct_length() {
    let dimensions = [1, 64, 128, 384, 768, 1536];
    
    for dims in dimensions {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();
        
        let mut storage = Storage::open(path, dims).unwrap();
        let vector = vec![42.0; dims as usize];
        storage.insert(&vector).unwrap();
        
        let slice = storage.get_vector_slice(0).unwrap();
        assert_eq!(
            slice.len(), 
            dims as usize, 
            "Slice length mismatch for {} dimensions", 
            dims
        );
        
        // Verify all values
        for &val in slice {
            assert_eq!(val, 42.0);
        }
    }
}

#[test]
fn test_get_vector_slice_multiple_vectors() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();
    
    let mut storage = Storage::open(path, 128).unwrap();
    
    // Insert 1000 vectors with unique patterns
    for i in 0..1000 {
        let vector = vec![i as f32; 128];
        storage.insert(&vector).unwrap();
    }
    
    // Verify all vectors can be accessed correctly
    for i in 0..1000 {
        let slice = storage.get_vector_slice(i).unwrap();
        assert_eq!(slice.len(), 128);
        assert!(slice.iter().all(|&v| v == i as f32));
    }
}

#[test]
fn test_get_vector_slice_after_reopen() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();
    
    // Insert and commit
    {
        let mut storage = Storage::open(path, 256).unwrap();
        for i in 0..10 {
            let vector = vec![i as f32 + 0.5; 256];
            storage.insert(&vector).unwrap();
        }
        storage.commit().unwrap();
    }
    
    // Reopen and verify slices
    {
        let storage = Storage::open(path, 256).unwrap();
        for i in 0..10 {
            let slice = storage.get_vector_slice(i).unwrap();
            assert_eq!(slice.len(), 256);
            assert!(slice.iter().all(|&v| v == (i as f32 + 0.5)));
        }
    }
}

#[test]
fn test_get_vector_slice_no_allocation() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();
    
    let mut storage = Storage::open(path, 128).unwrap();
    let vector = vec![3.14; 128];
    storage.insert(&vector).unwrap();
    
    // Get two slices - they should point to same memory
    let slice1 = storage.get_vector_slice(0).unwrap();
    let slice2 = storage.get_vector_slice(0).unwrap();
    
    // Verify they point to the same memory address
    let ptr1 = slice1.as_ptr();
    let ptr2 = slice2.as_ptr();
    assert_eq!(ptr1, ptr2, "Slices should point to the same memory (zero-copy)");
    
    // Verify data is correct
    assert_eq!(slice1, slice2);
    assert!(slice1.iter().all(|&v| v == 3.14));
}

#[test]
fn test_get_vector_slice_alignment() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();
    
    let mut storage = Storage::open(path, 768).unwrap();
    
    for i in 0..10 {
        storage.insert(&vec![i as f32; 768]).unwrap();
    }
    
    // Verify all slices are properly aligned for f32 (4-byte alignment)
    for i in 0..10 {
        let slice = storage.get_vector_slice(i).unwrap();
        let ptr = slice.as_ptr() as usize;
        assert_eq!(
            ptr % 4, 
            0, 
            "Slice at index {} has misaligned pointer: 0x{:x}", 
            i, 
            ptr
        );
    }
}

#[test]
fn test_get_vector_slice_distance_calculation() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();
    
    let mut storage = Storage::open(path, 128).unwrap();
    
    // Insert two vectors for distance calculation
    let v1 = vec![1.0; 128];
    let v2 = vec![2.0; 128];
    storage.insert(&v1).unwrap();
    storage.insert(&v2).unwrap();
    
    // Use slices to calculate Euclidean distance (HNSW-style usage)
    let slice1 = storage.get_vector_slice(0).unwrap();
    let slice2 = storage.get_vector_slice(1).unwrap();
    
    let distance: f32 = slice1.iter()
        .zip(slice2.iter())
        .map(|(a, b)| {
            let diff = a - b;
            diff * diff
        })
        .sum::<f32>()
        .sqrt();
    
    // Expected distance: sqrt(128 * (1.0 - 2.0)^2) = sqrt(128) ≈ 11.31
    let expected = (128.0_f32).sqrt();
    assert!((distance - expected).abs() < 0.01, 
        "Distance calculation incorrect: expected {}, got {}", expected, distance);
}

#[test]
fn test_multiple_simultaneous_slices() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();
    
    let mut storage = Storage::open(path, 128).unwrap();
    
    for i in 0..5 {
        storage.insert(&vec![i as f32; 128]).unwrap();
    }
    
    // Get multiple slices simultaneously (all borrow &self immutably)
    let slice0 = storage.get_vector_slice(0).unwrap();
    let slice1 = storage.get_vector_slice(1).unwrap();
    let slice2 = storage.get_vector_slice(2).unwrap();
    let slice3 = storage.get_vector_slice(3).unwrap();
    let slice4 = storage.get_vector_slice(4).unwrap();
    
    // All slices should be valid and contain correct data
    assert!(slice0.iter().all(|&v| v == 0.0));
    assert!(slice1.iter().all(|&v| v == 1.0));
    assert!(slice2.iter().all(|&v| v == 2.0));
    assert!(slice3.iter().all(|&v| v == 3.0));
    assert!(slice4.iter().all(|&v| v == 4.0));
}

#[test]
fn test_slice_then_read_only_operations() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();
    
    let mut storage = Storage::open(path, 128).unwrap();
    storage.insert(&vec![1.0; 128]).unwrap();
    storage.insert(&vec![2.0; 128]).unwrap();
    
    // Hold a slice reference
    let slice = storage.get_vector_slice(0).unwrap();
    
    // Other read-only operations should work fine
    assert_eq!(storage.count(), 2);
    assert_eq!(storage.dimensions(), 128);
    
    let other_slice = storage.get_vector_slice(1).unwrap();
    let owned = storage.get_vector(1).unwrap();
    
    // All operations coexist
    assert_eq!(slice.len(), 128);
    assert_eq!(other_slice.len(), 128);
    assert_eq!(owned.len(), 128);
}
