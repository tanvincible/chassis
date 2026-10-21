//! Files written by earlier releases (ADR-0023): the current code opens each one, finds what the
//! release that wrote it found, and goes on writing it. The files are in `tests/compat`, made by
//! `tests/compat/generate.rs` (v0.7.0 and later) and `generate_format1.rs` (v0.6.3, format 1),
//! each run with the release named at the start of the file's name; `expected.txt` holds what
//! that release's searches returned.

use chassisdb::{DistanceMetric, IndexOptions, IndexReader, Precision, VectorIndex};
use flate2::read::GzDecoder;
use std::path::{Path, PathBuf};

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/compat");

/// The generators' vector `i`; their queries are `10_000` and up, and no file holds `20_000`.
fn vector(i: u64, dims: u32) -> Vec<f32> {
    let mut x = i.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..dims)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 40) as f32 / (1u64 << 24) as f32 + 0.01
        })
        .collect()
}

/// One line of `expected.txt`: the file, its dimensions, metric and precision, its length, its
/// live ids, then the ids of the 5 nearest to each of 5 queries.
struct Case {
    file: String,
    dims: u32,
    options: IndexOptions,
    len: u64,
    live: Vec<u64>,
    results: Vec<Vec<u64>>,
}

fn cases() -> Vec<Case> {
    let expected = std::fs::read_to_string(Path::new(DIR).join("expected.txt")).unwrap();
    let ids = |s: &str| s.split(',').map(|id| id.parse().unwrap()).collect::<Vec<u64>>();
    expected
        .lines()
        .map(|line| {
            let f: Vec<&str> = line.split_whitespace().collect();
            let metric =
                if f[2] == "cosine" { DistanceMetric::Cosine } else { DistanceMetric::Euclidean };
            let precision = if f[3] == "half" { Precision::Half } else { Precision::Full };
            Case {
                file: f[0].to_string(),
                dims: f[1].parse().unwrap(),
                options: IndexOptions {
                    metric: Some(metric),
                    precision: Some(precision),
                    ..IndexOptions::default()
                },
                len: f[4].parse().unwrap(),
                live: ids(f[5]),
                results: f[6..].iter().map(|s| ids(s)).collect(),
            }
        })
        .collect()
}

/// The case's file, uncompressed into `dir`.
fn unpack(case: &Case, dir: &Path) -> PathBuf {
    let path = dir.join(&case.file);
    let packed = std::fs::File::open(Path::new(DIR).join(format!("{}.gz", case.file))).unwrap();
    let mut file = std::fs::File::create(&path).unwrap();
    std::io::copy(&mut GzDecoder::new(packed), &mut file).unwrap();
    path
}

fn nearest(search: &mut dyn FnMut(&[f32]) -> Vec<u64>, dims: u32) -> Vec<Vec<u64>> {
    (0..5).map(|q| search(&vector(10_000 + q, dims))).collect()
}

#[test]
fn test_files_from_earlier_releases_open_and_search_as_they_did() {
    let cases = cases();
    assert_eq!(
        cases.len(),
        17,
        "one format 1 file, five each from v0.7.0 and v0.8.0, six from v0.9.1"
    );
    for case in &cases {
        let dir = tempfile::tempdir().unwrap();
        let path = unpack(case, dir.path());
        let (name, dims) = (&case.file, case.dims);

        // A reader opens format 3 and later as it is; a writer migrates older files first.
        if !name.contains("format1") {
            let mut reader = IndexReader::open(&path, dims, case.options.clone()).unwrap();
            assert_eq!(reader.len(), case.len, "{name}: reader");
            let mut search = |q: &[f32]| -> Vec<u64> {
                reader.search(q, 5).unwrap().iter().map(|r| r.id).collect()
            };
            assert_eq!(nearest(&mut search, dims), case.results, "{name}: reader");
        }

        let mut index = VectorIndex::open(&path, dims, case.options.clone()).unwrap();
        assert_eq!(index.len(), case.len, "{name}");
        let mut ids = index.ids().unwrap();
        ids.sort_unstable();
        assert_eq!(ids, case.live, "{name}");
        let mut search =
            |q: &[f32]| -> Vec<u64> { index.search(q, 5).unwrap().iter().map(|r| r.id).collect() };
        assert_eq!(nearest(&mut search, dims), case.results, "{name}");

        // And it goes on: one more vector, under an id never used before, found after a reopen.
        let id = index.add(&vector(20_000, dims)).unwrap();
        assert!(case.live.iter().all(|&live| live < id), "{name}: id {id} reused");
        index.flush().unwrap();
        drop(index);
        let index = VectorIndex::open(&path, dims, case.options.clone()).unwrap();
        assert_eq!(index.len(), case.len + 1, "{name}");
        assert_eq!(index.search(&vector(20_000, dims), 1).unwrap()[0].id, id, "{name}");
    }
}

/// A file cut off after its header, as a partial copy leaves it, is an error: mapped, the regions
/// the header names past the file's end read as SIGBUS, which fuzzing found.
#[test]
fn test_a_file_cut_short_is_an_error_not_a_crash() {
    let case = cases().into_iter().find(|c| c.file == "v0.8.0-deletes.chassis").unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = unpack(&case, dir.path());
    let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    file.set_len(4 * 64 * 1024).unwrap(); // The headers, the live page and one region's start
    drop(file);

    let check = |e: chassisdb::Error| {
        assert_eq!(e.kind(), chassisdb::ErrorKind::Corrupt, "{e}");
        assert!(e.to_string().contains("shorter than its header says"), "{e}");
    };
    check(IndexReader::open(&path, case.dims, case.options.clone()).unwrap_err());
    check(VectorIndex::open(&path, case.dims, case.options.clone()).unwrap_err());
}
