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
