#![no_main]
//! Any bytes as a whole file: what reading a header makes of short, empty and foreign files.

use libfuzzer_sys::fuzz_target;

#[path = "open.rs"]
mod open;

fuzz_target!(|data: &[u8]| open::exercise(data));
