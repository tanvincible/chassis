//! Lab only: what one distance costs (`bench/lab/dist-plan.md`). Build with `--cfg lab`, on Linux.
#[cfg(all(lab, target_os = "linux"))]
fn main() -> anyhow::Result<()> {
    chassisdb::distbench::bench()
}

#[cfg(not(all(lab, target_os = "linux")))]
fn main() {}
