//! Writes the compatibility set's format 1 file, and what this release's searches return. Run it
//! as `generate.rs` says, in a checkout of v0.6.3. Not compiled here.
use chassis_core::{IndexOptions, VectorIndex};

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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (dir, release) = (std::path::Path::new(&args[1]), args[2].as_str());
    let name = format!("{release}-format1.chassis");
    let mut index = VectorIndex::open(dir.join(&name), DIMS, IndexOptions::default()).unwrap();
    for i in 0..N { index.add(&vector(i)).unwrap(); }
    index.flush().unwrap();
    let ids: Vec<String> = (0..N).map(|i| i.to_string()).collect();
    print!("{name} {DIMS} euclidean full {} {}", N, ids.join(","));
    for q in 0..5 {
        let found = index.search(&vector(10_000 + q), 5).unwrap();
        let found: Vec<String> = found.iter().map(|r| r.id.to_string()).collect();
        print!(" {}", found.join(","));
    }
    println!();
}
