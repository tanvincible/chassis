//! Software prefetching for graph search (ADR-0013, ADR-0015). A search knows a node's neighbors
//! before it needs their vectors, so it asks the CPU to start loading all of them, and their cache
//! misses overlap instead of queuing. How many lines to ask for, and into which cache, depends on
//! the CPU.

use std::sync::OnceLock;

/// What to ask for ahead of a vector's distance computation: its first `lines` cache lines, the
/// first `near` of them into L1 and the rest into L2.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Prefetch {
    near: usize,
    lines: usize,
}

const LINE: usize = 64;

impl Prefetch {
    /// The policy for this machine, chosen once.
    pub(crate) fn detect() -> Self {
        static POLICY: OnceLock<Prefetch> = OnceLock::new();
        *POLICY.get_or_init(Self::for_this_cpu)
    }

    /// A node has up to 32 neighbors, so a search asks for up to 256 lines at once. Into L2 that
    /// was the best of what was tried on Zen 4, Emerald Rapids and Neoverse-N2, and on Zen 3 as
    /// good as anything. AMD cores before Zen 5 track only 24 misses to L1 at a time and drop the
    /// hints past that. Zen 5 tracks 124, and is a tenth faster or more with L1 (ADR-0013).
    #[cfg(target_arch = "x86_64")]
    fn for_this_cpu() -> Self {
        Self::for_x86(amd())
    }

    /// The policy for an x86 processor, given its CPUID family and model if it is AMD's. Eight
    /// lines are all of a 128-dimension vector. Of a longer one, Zen 4 does better with 32; Zen 3,
    /// Zen 5 and the Xeons measured do worse with more than eight (ADR-0015).
    #[cfg(any(target_arch = "x86_64", test))]
    fn for_x86(amd: Option<(u32, u32)>) -> Self {
        match amd {
            Some((ZEN_5.., _)) => Self { near: 8, lines: 8 },
            Some((ZEN_3_AND_4, model)) if is_zen_4(model) => Self { near: 0, lines: 64 },
            _ => Self { near: 0, lines: 8 },
        }
    }

    /// Apple's cores do far better with the whole vector on its way: 60% faster at 1,536
    /// dimensions than with eight lines, and 10 to 20% faster than with two thirds of it. The
    /// bound is 4,096 dimensions' worth (ADR-0015).
    #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
    fn for_this_cpu() -> Self {
        Self { near: 0, lines: 256 }
    }

    /// Sixteen lines was the setting that never lost on Neoverse-N2, at 960 and 1,536 dimensions
    /// (ADR-0015).
    #[cfg(all(target_arch = "aarch64", not(target_vendor = "apple")))]
    fn for_this_cpu() -> Self {
        Self { near: 0, lines: 64 }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn for_this_cpu() -> Self {
        Self { near: 0, lines: 0 }
    }

    /// The policy for vectors of `dims` floats: no more lines than one has. Worked out once per
    /// search, so that asking for a vector is two counted loops.
    pub(crate) fn for_dims(self, dims: usize) -> Self {
        let lines = self.lines.min((dims * size_of::<f32>()).div_ceil(LINE));
        Self { near: self.near.min(lines), lines }
    }

    /// Asks for the start of `vector`.
    #[inline(always)]
    pub(crate) fn vector(self, vector: &[f32]) {
        let start = vector.as_ptr().cast::<u8>();
        // A prefetch is only a hint and never faults, wherever it points.
        for line in 0..self.near {
            hint(start.wrapping_add(line * LINE), true);
        }
        for line in self.near..self.lines {
            hint(start.wrapping_add(line * LINE), false);
        }
    }
}

/// Asks for all of the `bytes` at `start`, into L1: a neighbor list about to be read.
#[inline(always)]
pub(crate) fn list(start: *const u8, bytes: usize) {
    for offset in (0..bytes).step_by(LINE) {
        hint(start.wrapping_add(offset), true);
    }
}

#[inline(always)]
fn hint(line: *const u8, near: bool) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: a prefetch reads nothing and never faults, whatever the address.
    unsafe {
        use std::arch::x86_64::{_MM_HINT_T0, _MM_HINT_T1, _mm_prefetch};
        if near {
            _mm_prefetch::<_MM_HINT_T0>(line.cast());
        } else {
            _mm_prefetch::<_MM_HINT_T1>(line.cast());
        }
    }
    #[cfg(target_arch = "aarch64")]
    // SAFETY: as above.
    unsafe {
        use std::arch::asm;
        if near {
            asm!("prfm pldl1keep, [{0}]", in(reg) line, options(nostack, preserves_flags, readonly));
        } else {
            asm!("prfm pldl2keep, [{0}]", in(reg) line, options(nostack, preserves_flags, readonly));
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = (line, near);
}

/// The CPUID family of Zen 5; its successors have higher ones.
#[cfg(any(target_arch = "x86_64", test))]
const ZEN_5: u32 = 0x1a;

/// The CPUID family Zen 3 and Zen 4 share.
#[cfg(any(target_arch = "x86_64", test))]
const ZEN_3_AND_4: u32 = 0x19;

/// Whether a model of family 19h is a Zen 4 core, by the ranges Linux tells them apart with.
#[cfg(any(target_arch = "x86_64", test))]
fn is_zen_4(model: u32) -> bool {
    matches!(model, 0x10..=0x1f | 0x60..=0xaf)
}

/// The CPUID family and model of an AMD processor.
#[cfg(target_arch = "x86_64")]
#[allow(unused_unsafe)] // `__cpuid` is safe from Rust 1.89 on.
fn amd() -> Option<(u32, u32)> {
    use std::arch::x86_64::__cpuid;
    // SAFETY: every x86-64 processor has CPUID.
    let (vendor, version) = unsafe { (__cpuid(0), __cpuid(1).eax) };
    // "AuthenticAMD", in EBX, EDX, ECX.
    let amd = (vendor.ebx, vendor.edx, vendor.ecx) == (0x6874_7541, 0x6974_6e65, 0x444d_4163);
    let family = ((version >> 8) & 0xf) + ((version >> 20) & 0xff);
    let model = ((version >> 4) & 0xf) | ((version >> 12) & 0xf0);
    amd.then_some((family, model))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefetching_past_the_end_of_the_data_is_harmless() {
        let policy = Prefetch::detect();
        for dims in [0, 1, 3, 16, 17, 128, 1536] {
            let vector = vec![1.0f32; dims];
            policy.vector(&vector);
            policy.for_dims(dims).vector(&vector);
            list(vector.as_ptr().cast(), dims * 4);
        }
    }

    #[test]
    fn test_a_short_vector_gets_no_more_lines_than_it_has() {
        let into_l1 = Prefetch { near: 8, lines: 8 };
        assert_eq!(into_l1.for_dims(0), Prefetch { near: 0, lines: 0 });
        assert_eq!(into_l1.for_dims(17), Prefetch { near: 2, lines: 2 });
        assert_eq!(into_l1.for_dims(1536), into_l1);
        assert_eq!(Prefetch { near: 0, lines: 8 }.for_dims(16), Prefetch { near: 0, lines: 1 });
    }

    #[test]
    fn test_each_measured_x86_processor_gets_its_policy() {
        let (into_l1, into_l2) = (Prefetch { near: 8, lines: 8 }, Prefetch { near: 0, lines: 8 });
        let deeper = Prefetch { near: 0, lines: 32 };
        // EPYC 7763 (Zen 3), 9V74 (Zen 4) and 9V45 (Zen 5), by family and model; then Intel.
        assert_eq!(Prefetch::for_x86(Some((0x19, 0x01))), into_l2);
        assert_eq!(Prefetch::for_x86(Some((0x19, 0x11))), deeper);
        assert_eq!(Prefetch::for_x86(Some((0x1a, 0x02))), into_l1);
        assert_eq!(Prefetch::for_x86(None), into_l2);
        // Not measured: desktop Zen 3 and Zen 4, earlier Zen cores, and whatever follows Zen 5.
        assert_eq!(Prefetch::for_x86(Some((0x19, 0x21))), into_l2);
        assert_eq!(Prefetch::for_x86(Some((0x19, 0x61))), deeper);
        assert_eq!(Prefetch::for_x86(Some((0x17, 0x31))), into_l2);
        assert_eq!(Prefetch::for_x86(Some((0x1b, 0x00))), into_l1);
        // A 128-dimension vector is eight lines whatever the policy.
        assert_eq!(deeper.for_dims(128), into_l2);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_amd_family_and_model_are_real() {
        let real = |(family, model)| (0xf..0x100).contains(&family) && model < 0x100;
        assert!(amd().is_none_or(real));
    }
}
