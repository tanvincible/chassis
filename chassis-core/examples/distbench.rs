//! Lab only: what one distance costs (`bench/lab/avx512-plan.md`). Build with `--cfg lab`, on
//! Linux on x86.
#[cfg(all(lab, target_os = "linux", target_arch = "x86_64"))]
fn main() -> anyhow::Result<()> {
    chassisdb::distbench::bench()
}

#[cfg(not(all(lab, target_os = "linux", target_arch = "x86_64")))]
fn main() {}
