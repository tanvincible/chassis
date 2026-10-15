#![no_main]
//! A file from the compatibility set with bytes changed or cut off: the first input byte picks
//! the file, then each 5 bytes are an offset and a value. An offset with its top bit set cuts the
//! file there instead.

use libfuzzer_sys::fuzz_target;
use std::io::Read;
use std::sync::OnceLock;

#[path = "open.rs"]
mod open;

fn files() -> &'static [Vec<u8>] {
    static FILES: OnceLock<Vec<Vec<u8>>> = OnceLock::new();
    FILES.get_or_init(|| {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../chassis-core/tests/compat");
        let mut paths: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.extension().is_some_and(|e| e == "gz"))
            .collect();
        paths.sort();
        let unpack = |path: &std::path::PathBuf| {
            let mut bytes = Vec::new();
            let packed = std::fs::File::open(path).unwrap();
            flate2::read::GzDecoder::new(packed).read_to_end(&mut bytes).unwrap();
            bytes
        };
        paths.iter().map(unpack).collect()
    })
}

fuzz_target!(|data: &[u8]| {
    let Some((&pick, patches)) = data.split_first() else { return };
    let mut file = files()[usize::from(pick) % files().len()].clone();
    for patch in patches.chunks_exact(5) {
        let at = u32::from_le_bytes([patch[0], patch[1], patch[2], patch[3]]) as usize;
        if at & 1 << 31 != 0 {
            file.truncate((at & !(1 << 31)) % (file.len() + 1));
        } else if !file.is_empty() {
            let i = at % file.len();
            file[i] = patch[4];
        }
    }
    open::exercise(&file);
});
