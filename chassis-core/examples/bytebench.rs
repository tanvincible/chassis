//! Lab only: eight bits a component, the kernels' checks and timing (`bench/lab/byte-plan.md`).
fn main() -> anyhow::Result<()> {
    chassis_core::byte::bench()
}
