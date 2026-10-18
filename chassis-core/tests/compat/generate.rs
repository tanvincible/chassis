//! Writes the compatibility set's files for one release, and what that release's searches return.
//! Copy it into a checkout of the release as `chassis-core/examples/generate.rs` and run
//! `cargo run --release --example generate -- <dir> <release> >> expected.txt`; before v0.8.0
//! the crate is `chassis_core`, not `chassisdb`. Not compiled here.
use chassisdb::{DistanceMetric, IndexOptions, Precision, VectorIndex};
use std::fmt::Write as _;

const DIMS: u32 = 8;
const N: u64 = 48;

fn vector(i: u64) -> Vec<f32> {
    let mut x = i.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..DIMS)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 40) as f32 / (1u64 << 24) as f32 + 0.01
        })
        .collect()
}

fn record(out: &mut String, name: &str, index: &VectorIndex, metric: &str, precision: &str, live: &[u64]) {
    let ids: Vec<String> = live.iter().map(u64::to_string).collect();
    write!(out, "{name} {DIMS} {metric} {precision} {} {}", index.len(), ids.join(",")).unwrap();
    for q in 0..5 {
        let found = index.search(&vector(10_000 + q), 5).unwrap();
        let found: Vec<String> = found.iter().map(|r| r.id.to_string()).collect();
        write!(out, " {}", found.join(",")).unwrap();
    }
    out.push('\n');
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (dir, release) = (std::path::Path::new(&args[1]), args[2].as_str());
    let mut out = String::new();
    let open = |name: &str, metric, precision| {
        let options = IndexOptions { metric, precision, ..IndexOptions::default() };
        VectorIndex::open(dir.join(name), DIMS, options).unwrap()
    };
    let (e, c, full, half) = (DistanceMetric::Euclidean, DistanceMetric::Cosine, Precision::Full, Precision::Half);

    let name = format!("{release}-deletes.chassis");
    let mut index = open(&name, e, full);
    for i in 0..N { index.add(&vector(i)).unwrap(); }
    for id in [3, 10, 25] { index.delete(id).unwrap(); }
    index.flush().unwrap();
    let live: Vec<u64> = (0..N).filter(|i| ![3, 10, 25].contains(i)).collect();
    record(&mut out, &name, &index, "euclidean", "full", &live);

    let name = format!("{release}-ids.chassis");
    let mut index = open(&name, e, full);
    for i in 0..N { index.add_with_id(1000 + 7 * i, &vector(i)).unwrap(); }
    index.delete(1000 + 7 * 5).unwrap();
    index.flush().unwrap();
    let live: Vec<u64> = (0..N).filter(|&i| i != 5).map(|i| 1000 + 7 * i).collect();
    record(&mut out, &name, &index, "euclidean", "full", &live);

    let name = format!("{release}-cosine.chassis");
    let mut index = open(&name, c, full);
    for i in 0..N { index.add(&vector(i)).unwrap(); }
    index.flush().unwrap();
    record(&mut out, &name, &index, "cosine", "full", &(0..N).collect::<Vec<_>>());

    let name = format!("{release}-half.chassis");
    let mut index = open(&name, e, half);
    for i in 0..N { index.add(&vector(i)).unwrap(); }
    index.flush().unwrap();
    record(&mut out, &name, &index, "euclidean", "half", &(0..N).collect::<Vec<_>>());

    let name = format!("{release}-compacted.chassis");
    let mut index = open(&name, e, full);
    for i in 0..N { index.add(&vector(i)).unwrap(); }
    for id in 0..10 { index.delete(id).unwrap(); }
    index.compact().unwrap();
    for i in N..N + 2 { index.add(&vector(i)).unwrap(); }
    index.flush().unwrap();
    record(&mut out, &name, &index, "euclidean", "full", &(10..N + 2).collect::<Vec<_>>());

    // More vectors than the first segment's 1,024 slots, with ids and deletes in both segments.
    // From v0.9.1.
    let name = format!("{release}-segments.chassis");
    let mut index = open(&name, e, full);
    let (n, deleted) = (1_500, [5, 1_200]);
    for i in 0..n { index.add_with_id(1000 + 7 * i, &vector(i)).unwrap(); }
    for i in deleted { index.delete(1000 + 7 * i).unwrap(); }
    index.flush().unwrap();
    let live: Vec<u64> = (0..n).filter(|i| !deleted.contains(i)).map(|i| 1000 + 7 * i).collect();
    record(&mut out, &name, &index, "euclidean", "full", &live);

    print!("{out}");
}
