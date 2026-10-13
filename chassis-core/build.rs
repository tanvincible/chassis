//! Doctests the coding agents' skill (lib.rs) only where the repository is around the crate: the
//! published crate doesn't carry the skill.

fn main() {
    println!("cargo::rustc-check-cfg=cfg(chassis_skill)");
    if std::path::Path::new("../plugins/chassis/skills/chassis/SKILL.md").exists() {
        println!("cargo::rustc-cfg=chassis_skill");
    }
    println!("cargo::rerun-if-changed=build.rs");
}
