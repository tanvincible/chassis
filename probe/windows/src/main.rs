//! Probes the Windows file behavior that ADR-0008 ("Before Accepting" #1) depends on: growing and
//! shrinking a file that another process has mapped, byte-range locks against `ReadFile`, and
//! replacing a file that another process holds open or mapped.
//!
//! The parent runs every check and prints `RESULT <check>: <outcome>` lines; children, started by
//! re-running this binary with a role, hold files open or mapped, or try a single operation.

#[cfg(not(windows))]
fn main() {
    eprintln!("This probe only runs on Windows.");
}

#[cfg(windows)]
fn main() {
    windows::main();
}

#[cfg(windows)]
mod windows {
    use memmap2::MmapOptions;
    use std::ffi::OsStr;
    use std::fs::{self, File, OpenOptions};
    use std::os::windows::ffi::OsStrExt;
    use std::os::windows::fs::FileExt;
    use std::os::windows::io::AsRawHandle;
    use std::path::{Path, PathBuf};
    use std::process::{Child, Command};
    use std::time::{Duration, Instant};
    use windows_sys::Win32::Foundation::GetLastError;
    use windows_sys::Win32::Storage::FileSystem::{
        LOCKFILE_EXCLUSIVE_LOCK, LOCKFILE_FAIL_IMMEDIATELY, LockFileEx, MOVEFILE_REPLACE_EXISTING,
        MOVEFILE_WRITE_THROUGH, MoveFileExW, UnlockFileEx,
    };
    use windows_sys::Win32::System::IO::OVERLAPPED;

    const MIB: u64 = 1 << 20;

    pub fn main() {
        let args: Vec<String> = std::env::args().collect();
        match args.get(1).map(String::as_str) {
            Some("hold-map") => hold(&args[2], &args[3], true),
            Some("hold-open") => hold(&args[2], &args[3], false),
            Some("try-read") => try_read(&args[2]),
            Some("try-lock-whole") => try_lock_whole(&args[2]),
            _ => run_checks(),
        }
    }

    fn result(check: &str, outcome: impl std::fmt::Display) {
        println!("RESULT {check}: {outcome}");
    }

    fn outcome<T>(r: std::io::Result<T>) -> String {
        match r {
            Ok(_) => "ok".into(),
            Err(e) => format!("error {} ({e})", e.raw_os_error().unwrap_or(-1)),
        }
    }

    fn last_error() -> String {
        format!("error {}", unsafe { GetLastError() })
    }

    fn wide(path: &Path) -> Vec<u16> {
        OsStr::new(path).encode_wide().chain(Some(0)).collect()
    }

    fn overlapped(offset: u64) -> OVERLAPPED {
        let mut ov: OVERLAPPED = unsafe { std::mem::zeroed() };
        ov.Anonymous.Anonymous.Offset = offset as u32;
        ov.Anonymous.Anonymous.OffsetHigh = (offset >> 32) as u32;
        ov
    }

    fn lock(file: &File, offset: u64, len: u64) -> bool {
        let mut ov = overlapped(offset);
        let flags = LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY;
        let handle = file.as_raw_handle() as _;
        unsafe { LockFileEx(handle, flags, 0, len as u32, (len >> 32) as u32, &mut ov) != 0 }
    }

    fn unlock(file: &File, offset: u64, len: u64) -> bool {
        let mut ov = overlapped(offset);
        let handle = file.as_raw_handle() as _;
        unsafe { UnlockFileEx(handle, 0, len as u32, (len >> 32) as u32, &mut ov) != 0 }
    }

    fn wait_for(path: &Path) {
        let start = Instant::now();
        while !path.exists() {
            assert!(start.elapsed() < Duration::from_secs(30), "timed out waiting for {path:?}");
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// Starts a child that holds `path` open (and mapped, if `map`) until told to stop.
    fn spawn_holder(path: &Path, signals: &Path, map: bool) -> Child {
        fs::create_dir_all(signals).unwrap();
        let role = if map { "hold-map" } else { "hold-open" };
        let child = Command::new(std::env::current_exe().unwrap())
            .args([role, path.to_str().unwrap(), signals.to_str().unwrap()])
            .spawn()
            .unwrap();
        wait_for(&signals.join("ready"));
        child
    }

    fn stop_holder(mut child: Child, signals: &Path) -> String {
        fs::write(signals.join("stop"), b"").unwrap();
        let status = child.wait().unwrap();
        format!("holder exited {status}")
    }

    fn run_child(role: &str, path: &Path) -> String {
        let out = Command::new(std::env::current_exe().unwrap())
            .args([role, path.to_str().unwrap()])
            .output()
            .unwrap();
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    /// Child: maps the first 1 MiB (or just opens the file) and waits. If told to, it then maps a
    /// newly written range as a separate view and checks its bytes.
    fn hold(path: &str, signals: &str, map: bool) {
        let signals = Path::new(signals);
        let file = File::open(path).unwrap();
        let view = map.then(|| unsafe { MmapOptions::new().len(MIB as usize).map(&file).unwrap() });
        fs::write(signals.join("ready"), b"").unwrap();
        let mut ok = true;
        while !signals.join("stop").exists() {
            if let Some(view) = &view {
                ok &= view.iter().step_by(4096).all(|&b| b == 0xA1);
            }
            if let Ok(spec) = fs::read_to_string(signals.join("map-new")) {
                fs::remove_file(signals.join("map-new")).unwrap();
                let v: Vec<u64> = spec.split_whitespace().map(|x| x.parse().unwrap()).collect();
                let new_view = unsafe {
                    MmapOptions::new().offset(v[0]).len(v[1] as usize).map(&file).unwrap()
                };
                let good = new_view.iter().all(|&b| b == v[2] as u8);
                fs::write(signals.join("map-new-result"), if good { "ok" } else { "bad" }).unwrap();
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        let mut first = [0u8; 4];
        let read = file.seek_read(&mut first, 0);
        println!("holder: old view intact {ok}, first bytes via handle {first:?} ({read:?})");
        std::process::exit(if ok { 0 } else { 2 });
    }

    fn try_read(path: &str) {
        let file = File::open(path).unwrap();
        let mut buf = [0u8; 16];
        println!("{}", outcome(file.seek_read(&mut buf, 0)));
    }

    fn try_lock_whole(path: &str) {
        let file = File::open(path).unwrap();
        println!("{}", if lock(&file, 0, u64::MAX) { "ok".into() } else { last_error() });
    }

    fn run_checks() {
        let dir = PathBuf::from(std::env::var("RUNNER_TEMP").unwrap_or_else(|_| ".".into()))
            .join("chassis-windows-probe");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        println!("probe dir {dir:?}");
        grow_and_shrink(&dir);
        locks(&dir);
        replace(&dir);
    }

    /// (a) extend a file another process has mapped, then map the new range as a new view;
    /// (b) shrink it below that view.
    fn grow_and_shrink(dir: &Path) {
        let path = dir.join("grow.bin");
        fs::write(&path, vec![0xA1u8; MIB as usize]).unwrap();
        let signals = dir.join("grow-signals");
        let holder = spawn_holder(&path, &signals, true);

        let file = OpenOptions::new().read(true).write(true).open(&path).unwrap();
        result("a1 extend with SetEndOfFile under another process's view", outcome(file.set_len(2 * MIB)));
        let pattern = vec![0xB2u8; MIB as usize];
        result("a2 WriteFile into the extended range", outcome(file.seek_write(&pattern, MIB)));
        let tail = vec![0xC3u8; 64 * 1024];
        result("a3 extend with WriteFile past the end", outcome(file.seek_write(&tail, 3 * MIB)));
        result("a3 file length", file.metadata().unwrap().len());

        fs::write(signals.join("map-new"), format!("{} {} {}", MIB, MIB, 0xB2)).unwrap();
        wait_for(&signals.join("map-new-result"));
        result(
            "a4 other process maps the new range at a 64 KiB offset and sees the data",
            fs::read_to_string(signals.join("map-new-result")).unwrap(),
        );

        result("b1 shrink below the other process's view", outcome(file.set_len(MIB / 2)));
        result("b2 holder", stop_holder(holder, &signals));
    }

    /// (c) a whole-file lock against another process's ReadFile, a one-byte lock at 2^62, and
    /// whether a whole-range lock request (what fs2 and older releases take) overlaps it.
    fn locks(dir: &Path) {
        let path = dir.join("lock.bin");
        fs::write(&path, vec![7u8; 64 * 1024]).unwrap();
        let file = OpenOptions::new().read(true).write(true).open(&path).unwrap();

        result("c1 whole-range lock taken", lock(&file, 0, u64::MAX));
        result("c2 other process ReadFile under the whole-range lock", run_child("try-read", &path));
        result("c3 whole-range unlock", unlock(&file, 0, u64::MAX));

        result("c4 one-byte lock at 2^62 taken", lock(&file, 1 << 62, 1));
        result("c5 other process ReadFile under the one-byte lock", run_child("try-read", &path));
        result(
            "c6 other process's whole-range lock request while the one-byte lock is held",
            run_child("try-lock-whole", &path),
        );
        result("c7 one-byte unlock", unlock(&file, 1 << 62, 1));
    }

    /// (d) replace a file by rename while nothing, an open handle, or a mapped view holds it,
    /// with MoveFileExW and with std::fs::rename.
    fn replace(dir: &Path) {
        let fresh = |name: &str, byte: u8| {
            let p = dir.join(name);
            fs::write(&p, vec![byte; MIB as usize]).unwrap();
            p
        };
        let move_file = |from: &Path, to: &Path| {
            let flags = MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH;
            let ok = unsafe { MoveFileExW(wide(from).as_ptr(), wide(to).as_ptr(), flags) != 0 };
            if ok { "ok".to_string() } else { last_error() }
        };

        for (label, holder) in [("nothing", None), ("an open handle", Some(false)), ("a mapped view", Some(true))] {
            for api in ["MoveFileExW(REPLACE_EXISTING | WRITE_THROUGH)", "std::fs::rename"] {
                let orig = fresh("orig.bin", 0xA1);
                let new = fresh("new.bin", 0xD4);
                let signals = dir.join(format!("replace-{}-{}", label.len(), api.len()));
                let child = holder.map(|map| spawn_holder(&orig, &signals, map));
                let outcome = if api.starts_with("Move") {
                    move_file(&new, &orig)
                } else {
                    outcome(fs::rename(&new, &orig))
                };
                let now = fs::read(&orig).map(|b| b.first().copied());
                result(&format!("d replace while {label} holds the file, {api}"), outcome);
                result(&format!("d   path now starts with {now:?} (0xA1 = 161 old, 0xD4 = 212 new)"), "");
                if let Some(child) = child {
                    result("d   holder", stop_holder(child, &signals));
                }
                let _ = fs::remove_file(&new);
                let _ = fs::remove_file(&orig);
            }
        }
    }
}
