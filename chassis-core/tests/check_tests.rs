//! `check` finds what opening can't: live vectors no search reaches, and vectors that aren't
//! finite numbers.

use chassisdb::{Check, DistanceMetric, IndexOptions, IndexReader, Precision, VectorIndex};
use tempfile::TempDir;

const DIMS: usize = 16;

fn vector(i: u64) -> Vec<f32> {
    (0..DIMS).map(|d| ((i * 31 + d as u64 * 7) % 97) as f32 / 97.0).collect()
}

fn built(path: &std::path::Path, options: IndexOptions, count: u64) -> VectorIndex {
    let mut index = VectorIndex::open(path, DIMS as u32, options).unwrap();
    for i in 0..count {
        index.add(&vector(i)).unwrap();
    }
    index.flush().unwrap();
    index
}

#[test]
fn test_an_index_in_good_order_checks_clean() {
    let options = [
        IndexOptions::default(),
        IndexOptions { metric: DistanceMetric::Cosine, ..IndexOptions::default() },
        IndexOptions { precision: Precision::Half, ..IndexOptions::default() },
    ];
    for (i, options) in options.into_iter().enumerate() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join(format!("{i}.chassis"));
        let mut index = built(&path, options.clone(), 500);
        for id in (0..500).step_by(7) {
            index.delete(id).unwrap();
        }
        index.flush().unwrap();
        assert_eq!(index.check().unwrap(), Check::default(), "{options:?}");
        let mut reader = IndexReader::open(&path, DIMS as u32, options.clone()).unwrap();
        assert!(reader.check().unwrap().is_ok(), "{options:?}");
        drop(reader);
        index.compact().unwrap();
        assert!(index.check().unwrap().is_ok(), "{options:?} after compacting");
    }
}

#[test]
fn test_an_empty_index_checks_clean() {
    let dir = TempDir::new().unwrap();
    let index = built(&dir.path().join("empty.chassis"), IndexOptions::default(), 0);
    assert!(index.check().unwrap().is_ok());
}

#[test]
fn test_a_vector_damaged_into_nan_is_reported() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("nan.chassis");
    let marked = vec![1234.5678_f32; DIMS];
    {
        let mut index = built(&path, IndexOptions::default(), 50);
        assert_eq!(index.add(&marked).unwrap(), 50);
        index.flush().unwrap();
    }
    // Damage the file where it keeps the marked vector, as a bad disk could.
    let mut bytes = std::fs::read(&path).unwrap();
    let pattern: Vec<u8> = marked.iter().flat_map(|x| x.to_le_bytes()).collect();
    let at = bytes.windows(pattern.len()).position(|w| w == pattern).unwrap();
    bytes[at..at + 4].copy_from_slice(&f32::NAN.to_le_bytes());
    std::fs::write(&path, bytes).unwrap();

    let index = VectorIndex::open(&path, DIMS as u32, IndexOptions::default()).unwrap();
    assert_eq!(index.check().unwrap(), Check { unreachable: vec![], not_finite: vec![50] });
}

#[cfg(feature = "internals")]
#[test]
fn test_a_vector_nothing_links_to_is_unreachable_until_compacting() {
    use chassisdb::{HnswGraph, HnswParams, Storage};

    let dir = TempDir::new().unwrap();
    let path = dir.path().join("cut.chassis");
    drop(built(&path, IndexOptions::default(), 300));

    // Take one vector out of every neighbor list, as a bad linking rule could leave it.
    let storage = Storage::open(&path, DIMS as u32).unwrap();
    let mut graph = HnswGraph::open(storage, HnswParams::default()).unwrap();
    let cut = (0..300).find(|&slot| Some(slot) != graph.entry_point).unwrap();
    for slot in 0..graph.node_count() {
        let mut record = graph.read_node_record(slot).unwrap();
        for layer in 0..usize::from(record.header.layer_count) {
            let kept: Vec<u64> =
                record.get_neighbors(layer).into_iter().filter(|&n| n != cut).collect();
            record.set_neighbors(layer, &kept);
        }
        graph.write_node_record(&record).unwrap();
    }
    graph.commit().unwrap();
    drop(graph);

    let mut index = VectorIndex::open(&path, DIMS as u32, IndexOptions::default()).unwrap();
    assert_eq!(index.check().unwrap(), Check { unreachable: vec![cut], not_finite: vec![] });
    let mut reader = IndexReader::open(&path, DIMS as u32, IndexOptions::default()).unwrap();
    assert_eq!(reader.check().unwrap().unreachable, vec![cut]);
    drop(reader);
    index.compact().unwrap();
    assert!(index.check().unwrap().is_ok(), "compacting builds the graph again");
}
