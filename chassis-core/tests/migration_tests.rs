//! Opening a format v1 or v2 file migrates it to v3 (ADR-0008, decision 8).

use chassis_core::{IndexOptions, MAGIC, NodeRecord, NodeRecordParams, VectorIndex};
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;
use tempfile::tempdir;

const DIMS: u32 = 4;

fn vector(slot: u64) -> Vec<f32> {
    vec![slot as f32; DIMS as usize]
}

/// A graph as releases up to 0.6.3 stored it.
struct Legacy {
    version: u32,
    graph_start: u64,
    /// Stored vectors; those past `records.len()` are leftovers of a crashed insert.
    vectors: u64,
    records: Vec<NodeRecord>,
    entry: u64,
    max_layer: u32,
    epoch: u32,
    flags: u8,
}

/// Writes `legacy` in the v1/v2 layout: a 4 KiB header, vectors, then a 64-byte graph header and
/// fixed-size records at `graph_start`.
fn write_legacy(path: &Path, legacy: &Legacy) {
    let params = NodeRecordParams::default();
    let mut head = vec![0u8; 4096];
    head[0..8].copy_from_slice(MAGIC);
    head[8..12].copy_from_slice(&legacy.version.to_le_bytes());
    head[12..16].copy_from_slice(&DIMS.to_le_bytes());
    head[16..24].copy_from_slice(&legacy.vectors.to_le_bytes());
    if legacy.version == 2 {
        head[24..32].copy_from_slice(b"CHLAYOUT");
        head[32..36].copy_from_slice(&1u32.to_le_bytes());
        head[40..48].copy_from_slice(&legacy.graph_start.to_le_bytes());
    }
    for slot in 0..legacy.vectors {
        head.extend(vector(slot).iter().flat_map(|x| x.to_le_bytes()));
    }

    let mut graph = vec![0u8; 64];
    graph[0..4].copy_from_slice(b"HNSW");
    graph[4..8].copy_from_slice(&1u32.to_le_bytes());
    graph[8..16].copy_from_slice(&legacy.entry.to_le_bytes());
    graph[16..24].copy_from_slice(&(legacy.records.len() as u64).to_le_bytes());
    graph[24..28].copy_from_slice(&legacy.max_layer.to_le_bytes());
    graph[28..30].copy_from_slice(&params.m.to_le_bytes());
    graph[30..32].copy_from_slice(&params.m0.to_le_bytes());
    graph[32] = params.max_layers;
    graph[33] = legacy.flags;
    graph[36..40].copy_from_slice(&legacy.epoch.to_le_bytes());
    for record in &legacy.records {
        graph.extend(record.to_bytes());
    }

    let mut file = std::fs::File::create(path).unwrap();
    file.write_all(&head).unwrap();
    file.seek(SeekFrom::Start(legacy.graph_start)).unwrap();
    file.write_all(&graph).unwrap();
}

/// Six nodes linked to each other on layer 0; node 3 is also on layer 1 and is the entry point.
fn records(ids: impl Fn(u64) -> u64) -> Vec<NodeRecord> {
    (0..6)
        .map(|slot| {
            let mut record =
                NodeRecord::new(slot, if slot == 3 { 2 } else { 1 }, Default::default());
            record.header.id = ids(slot);
            let others: Vec<u64> = (0..6).filter(|&n| n != slot).collect();
            record.set_neighbors(0, &others);
            record
        })
        .collect()
}

fn version_of(path: &Path) -> u32 {
    let bytes = std::fs::read(path).unwrap();
    u32::from_le_bytes(bytes[8..12].try_into().unwrap())
}

#[test]
fn test_v2_file_migrates_with_ids_and_committed_deletes_only() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("index.chassis");
    let mut records = records(|slot| 100 + slot);
    records[0].set_neighbors(0, &[1, 2, 9]); // 9: a backlink to a node a crash rolled back
    records[1].header.deleted_epoch = 1; // committed by the flush with epoch 1
    records[2].header.deleted_epoch = 2; // written by a flush that never committed
    write_legacy(
        &path,
        &Legacy {
            version: 2,
            graph_start: 8 << 20,
            vectors: 7,
            records,
            entry: 3,
            max_layer: 1,
            epoch: 1,
            flags: 2,
        },
    );
    // Left by an earlier migration that crashed.
    std::fs::write(dir.path().join("index.chassis.migrating"), b"junk").unwrap();

    let mut index = VectorIndex::open(&path, DIMS, IndexOptions::default()).unwrap();
    assert_eq!(version_of(&path), 3);
    assert!(!dir.path().join("index.chassis.migrating").exists());
    assert_eq!(index.len(), 5);
    for slot in [0, 2, 3, 4, 5] {
        assert_eq!(index.search(&vector(slot), 1).unwrap()[0].id, 100 + slot);
    }
    assert!(index.search(&vector(1), 6).unwrap().iter().all(|hit| hit.id != 101));

    assert_eq!(index.add(&vector(6)).unwrap(), 106);
    index.flush().unwrap();
    drop(index);
    let index = VectorIndex::open(&path, DIMS, IndexOptions::default()).unwrap();
    assert_eq!(index.len(), 6);
    assert_eq!(index.search(&vector(6), 1).unwrap()[0].id, 106);
}

#[test]
fn test_v1_file_with_graph_at_one_gib_migrates_to_a_small_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("index.chassis");
    let legacy = Legacy {
        version: 1,
        graph_start: 1 << 30,
        vectors: 6,
        records: records(|slot| slot),
        entry: 3,
        max_layer: 1,
        epoch: 0,
        flags: 0,
    };
    write_legacy(&path, &legacy);

    let mut index = VectorIndex::open(&path, DIMS, IndexOptions::default()).unwrap();
    assert_eq!(index.len(), 6);
    for slot in 0..6 {
        assert_eq!(index.search(&vector(slot), 1).unwrap()[0].id, slot);
    }
    assert_eq!(index.add(&vector(6)).unwrap(), 6);
    assert!(std::fs::metadata(&path).unwrap().len() < 100 << 20);
}

#[test]
fn test_corrupt_legacy_files_are_errors_and_stay_untouched() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("index.chassis");
    let cases = [
        // The graph claims more nodes than there are vectors.
        Legacy { vectors: 4, ..legacy_v2() },
        // Vectors, but the graph header was never written.
        Legacy { records: Vec::new(), entry: 0, ..legacy_v2() },
    ];
    for legacy in &cases {
        write_legacy(&path, legacy);
        if legacy.records.is_empty() {
            let mut bytes = std::fs::read(&path).unwrap();
            bytes[legacy.graph_start as usize..].fill(0);
            std::fs::write(&path, bytes).unwrap();
        }
        let before = std::fs::read(&path).unwrap();
        assert!(VectorIndex::open(&path, DIMS, IndexOptions::default()).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), before);
    }
}

fn legacy_v2() -> Legacy {
    Legacy {
        version: 2,
        graph_start: 64 << 10,
        vectors: 6,
        records: records(|slot| slot),
        entry: 3,
        max_layer: 1,
        epoch: 0,
        flags: 0,
    }
}

#[test]
fn test_migration_keeps_the_original_parameters() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("index.chassis");
    write_legacy(&path, &legacy_v2());

    let other = IndexOptions { max_connections: 8, ..IndexOptions::default() };
    assert!(VectorIndex::open(&path, DIMS, other).is_err());
    let index = VectorIndex::open(&path, DIMS, IndexOptions::default()).unwrap();
    assert_eq!(index.len(), 6);
}
