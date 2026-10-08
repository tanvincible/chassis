//! Software prefetching for graph search (ADR-0013). A search knows a node's neighbors before it
//! needs their vectors, so it asks the CPU to start loading all of them, and their cache misses
//! overlap instead of queuing. How many lines to ask for, and into which cache, depends on the CPU.

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
        *POLICY.get_or_init(|| {
            // Lab: LAB_PF=<near>,<lines> overrides the policy.
            #[cfg(lab)]
            if let Ok(spec) = std::env::var("LAB_PF")
                && let Some((near, lines)) = spec.split_once(',')
            {
                return Self { near: near.parse().unwrap(), lines: lines.parse().unwrap() };
            }
            Self::for_this_cpu()
        })
    }

    /// A node has up to 32 neighbors, so a search asks for up to 256 lines at once. Into L2 that
    /// was the best of what was tried on Zen 4, Emerald Rapids and Neoverse-N2, and on Zen 3 as
    /// good as anything. AMD cores before Zen 5 track only 24 misses to L1 at a time and drop the
    /// hints past that. Zen 5 tracks 124, and is a tenth faster or more with L1 (ADR-0013).
    #[cfg(target_arch = "x86_64")]
    fn for_this_cpu() -> Self {
        Self::for_x86(amd_family())
    }

    /// The policy for an x86 processor, given its CPUID family if it is AMD's.
    #[cfg(any(target_arch = "x86_64", test))]
    fn for_x86(amd_family: Option<u32>) -> Self {
        match amd_family {
            Some(ZEN_5..) => Self { near: 8, lines: 8 },
            _ => Self { near: 0, lines: 8 },
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn for_this_cpu() -> Self {
        Self { near: 0, lines: 8 }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn for_this_cpu() -> Self {
        Self { near: 0, lines: 0 }
    }

    /// Asks for the start of `vector`.
    #[inline(always)]
    pub(crate) fn vector(self, vector: &[f32]) {
        let bytes = std::mem::size_of_val(vector);
        self.range(vector.as_ptr().cast(), bytes);
    }

    /// Asks for the first lines of the `bytes` at `start`.
    #[inline(always)]
    fn range(self, start: *const u8, bytes: usize) {
        for (line, offset) in (0..bytes.min(self.lines * LINE)).step_by(LINE).enumerate() {
            // In bounds of the allocation; a prefetch is only a hint and never faults.
            hint(start.wrapping_add(offset), line < self.near);
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

/// The CPUID family of an AMD processor.
#[cfg(target_arch = "x86_64")]
#[allow(unused_unsafe)] // `__cpuid` is safe from Rust 1.89 on.
fn amd_family() -> Option<u32> {
    use std::arch::x86_64::__cpuid;
    // SAFETY: every x86-64 processor has CPUID.
    let (vendor, version) = unsafe { (__cpuid(0), __cpuid(1).eax) };
    // "AuthenticAMD", in EBX, EDX, ECX.
    let amd = (vendor.ebx, vendor.edx, vendor.ecx) == (0x6874_7541, 0x6974_6e65, 0x444d_4163);
    amd.then_some(((version >> 8) & 0xf) + ((version >> 20) & 0xff))
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
            list(vector.as_ptr().cast(), dims * 4);
        }
    }

    #[test]
    fn test_each_measured_x86_processor_gets_its_policy() {
        let (into_l1, into_l2) = (Prefetch { near: 8, lines: 8 }, Prefetch { near: 0, lines: 8 });
        // EPYC 7763 (Zen 3) and 9V74 (Zen 4) share a family; EPYC 9V45 is Zen 5; then Intel.
        assert_eq!(Prefetch::for_x86(Some(0x19)), into_l2);
        assert_eq!(Prefetch::for_x86(Some(0x1a)), into_l1);
        assert_eq!(Prefetch::for_x86(None), into_l2);
        // Not measured: earlier Zen cores, and whatever follows Zen 5.
        assert_eq!(Prefetch::for_x86(Some(0x17)), into_l2);
        assert_eq!(Prefetch::for_x86(Some(0x1b)), into_l1);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_amd_family_is_a_real_family() {
        assert!(amd_family().is_none_or(|family| (0xf..0x100).contains(&family)));
    }
}
