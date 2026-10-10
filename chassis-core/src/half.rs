//! Half precision (ADR-0018): vectors kept as IEEE 754 16-bit floats, half the size of f32s. A
//! query stays in f32, and a stored vector is widened as its distance is computed.

use anyhow::Result;

/// What an index keeps of each component of a vector; fixed when the index is created.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Precision {
    /// 32-bit floats, as given.
    #[default]
    Full,
    /// 16-bit floats: half the file and half the memory (ADR-0018). Each component is rounded to
    /// 11 significant bits, about three decimal digits, and a vector with a component of 65,520
    /// or more in magnitude is refused.
    Half,
}

impl Precision {
    /// Bytes one component takes in the file.
    pub(crate) fn bytes(self) -> usize {
        match self {
            Self::Full => size_of::<f32>(),
            Self::Half => size_of::<u16>(),
        }
    }
}

/// From here up a finite f32 rounds to infinity in half precision, whose largest value is 65,504.
const TOO_LARGE: f32 = 65_520.0;

/// Refuses vectors, `dims` components each, that half precision can't hold. NaN and infinity
/// pass here; callers refuse them for every precision.
pub(crate) fn check(vectors: &[f32], dims: usize) -> Result<()> {
    let Some(at) = vectors.iter().position(|x| x.is_finite() && x.abs() >= TOO_LARGE) else {
        return Ok(());
    };
    let which = match vectors.len() == dims {
        true => "The vector".to_string(),
        false => format!("Vector {} of the batch", at / dims),
    };
    crate::error::fail!(
        InvalidArgument,
        "{which} has {} at component {}, too large for half precision, which holds up to \
         65,504\nhelp: scale the vectors down, or keep them in an index in full precision",
        vectors[at],
        at % dims
    );
}

/// `x` as the nearest half, ties to the even one.
pub(crate) fn encode(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        // Infinity, or NaN with what fits of its payload and the quiet bit set.
        return sign | 0x7c00 | if mantissa != 0 { 0x200 | (mantissa >> 13) as u16 } else { 0 };
    }
    let exponent = exponent - 127 + 15;
    if exponent >= 31 {
        return sign | 0x7c00;
    }
    // The bits that don't fit decide the rounding; a carry out of the mantissa lands in the
    // exponent, which is what rounding up to the next power of two, or to infinity, means.
    let round = |half: u16, rest: u32, tie: u32| {
        half + u16::from(rest > tie || (rest == tie && half & 1 == 1))
    };
    if exponent <= 0 {
        // Subnormal in half precision, or too small for it.
        if exponent < -10 {
            return sign;
        }
        let (mantissa, shift) = (mantissa | 0x80_0000, (14 - exponent) as u32);
        return sign
            | round((mantissa >> shift) as u16, mantissa & ((1 << shift) - 1), 1 << (shift - 1));
    }
    sign | round(((exponent as u16) << 10) | (mantissa >> 13) as u16, mantissa & 0x1fff, 0x1000)
}

/// The f32 equal to the half `h`.
pub(crate) fn decode(h: u16) -> f32 {
    let sign = u32::from(h & 0x8000) << 16;
    let exponent = u32::from(h >> 10) & 0x1f;
    let mantissa = u32::from(h & 0x3ff);
    f32::from_bits(match exponent {
        0 if mantissa == 0 => sign,
        0 => {
            // Subnormal: shift the leading one up to where f32 implies it.
            let shift = mantissa.leading_zeros() - 21;
            sign | ((113 - shift) << 23) | (((mantissa << shift) & 0x3ff) << 13)
        }
        // Infinity, or NaN made quiet, as the instructions that widen do.
        31 if mantissa == 0 => sign | 0x7f80_0000,
        31 => sign | 0x7fc0_0000 | (mantissa << 13),
        _ => sign | ((exponent + 112) << 23) | (mantissa << 13),
    })
}

/// `vector` as halves, in the file's byte order, into `out`, which must be twice its length.
pub(crate) fn write(vector: &[f32], out: &mut [u8]) {
    for (dst, &x) in out.as_chunks_mut::<2>().0.iter_mut().zip(vector) {
        dst.copy_from_slice(&encode(x).to_le_bytes());
    }
}

/// `stored` as f32s, in `out`.
pub(crate) fn widen(stored: &[u16], out: &mut Vec<f32>) {
    out.clear();
    out.reserve(stored.len());
    // SAFETY: `out` has room for `stored.len()` floats, and the first `done` of them are written.
    let done = unsafe {
        let done = widen_groups(stored, out.as_mut_ptr());
        out.set_len(done);
        done
    };
    out.extend(stored[done..].iter().map(|&h| decode(h)));
}

/// Widens as many whole groups as this CPU's instructions take and returns how many components
/// that was.
///
/// # Safety
///
/// `out` must have room for `stored.len()` floats.
unsafe fn widen_groups(stored: &[u16], out: *mut f32) -> usize {
    #[cfg(target_arch = "x86_64")]
    if crate::distance::has_f16c() {
        // SAFETY: the CPU has F16C; the room is the caller's word.
        return unsafe { widen_f16c(stored, out) };
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: every aarch64 CPU has NEON; the room is the caller's word.
        return unsafe { widen_neon(stored, out) };
    }
    #[allow(unreachable_code)]
    {
        let _ = (stored, out);
        0
    }
}

/// Widens whole groups of eight and returns how many components that was.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn widen_f16c(stored: &[u16], out: *mut f32) -> usize {
    use std::arch::x86_64::*;
    let whole = stored.len() / 8 * 8;
    for i in (0..whole).step_by(8) {
        // SAFETY: `i + 8` is within `stored`, and within `out` by the caller's word.
        unsafe {
            let halves = _mm_loadu_si128(stored.as_ptr().add(i).cast());
            _mm256_storeu_ps(out.add(i), _mm256_cvtph_ps(halves));
        }
    }
    whole
}

/// Four halves at `halves` as f32s. Every ARMv8 CPU has the instruction; stable Rust has no
/// intrinsic for it yet.
///
/// # Safety
///
/// Four halves must be readable at `halves`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) unsafe fn widen4(halves: *const u16) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    // SAFETY: the caller's.
    let half = unsafe { vld1_u16(halves) };
    let single: float32x4_t;
    // SAFETY: reads one register and writes another.
    unsafe {
        std::arch::asm!(
            "fcvtl {single:v}.4s, {half:v}.4h",
            half = in(vreg) half,
            single = lateout(vreg) single,
            options(pure, nomem, nostack, preserves_flags)
        );
    }
    single
}

/// Widens whole groups of four and returns how many components that was.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn widen_neon(stored: &[u16], out: *mut f32) -> usize {
    use std::arch::aarch64::*;
    let whole = stored.len() / 4 * 4;
    for i in (0..whole).step_by(4) {
        // SAFETY: `i + 4` is within `stored`, and within `out` by the caller's word.
        unsafe { vst1q_f32(out.add(i), widen4(stored.as_ptr().add(i))) };
    }
    whole
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_every_half_survives_a_round_trip() {
        for h in 0..=u16::MAX {
            let x = decode(h);
            if x.is_nan() {
                assert!(decode(encode(x)).is_nan());
            } else {
                assert_eq!(encode(x), h, "{h:#06x} is {x}");
            }
        }
    }

    #[test]
    fn test_encode_rounds_to_the_nearest_half_and_ties_to_even() {
        let between = |a: u16, b: u16| (decode(a) + decode(b)) / 2.0;
        let after = |x: f32| f32::from_bits(x.to_bits() + 1);
        assert_eq!(encode(between(0x3c00, 0x3c01)), 0x3c00);
        assert_eq!(encode(after(between(0x3c00, 0x3c01))), 0x3c01);
        assert_eq!(encode(between(0x3c01, 0x3c02)), 0x3c02);
        // Rounding up carries into the exponent.
        assert_eq!(encode(between(0x3fff, 0x4000)), 0x4000);
        // Subnormal halves, and what is too small for any.
        assert_eq!(encode(decode(0x0001) / 2.0), 0x0000);
        assert_eq!(encode(decode(0x0001) * 1.5), 0x0002);
        assert_eq!(encode(between(0x03ff, 0x0400)), 0x0400);
        assert_eq!(encode(1e-9), 0);
        assert_eq!(encode(-1e-9), 0x8000);
    }

    #[test]
    fn test_what_is_too_large_is_exactly_what_rounds_to_infinity() {
        let below = f32::from_bits(TOO_LARGE.to_bits() - 1);
        assert_eq!(encode(below), 0x7bff);
        assert_eq!(decode(0x7bff), 65_504.0);
        assert_eq!(encode(TOO_LARGE), 0x7c00);
        assert_eq!(encode(-TOO_LARGE), 0xfc00);
        assert!(check(&[0.0, below, -below], 3).is_ok());
        assert!(check(&[0.0, TOO_LARGE], 2).is_err());
        assert!(check(&[-1e9], 1).is_err());
        assert!(check(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY], 3).is_ok());
        let error = check(&[0.0, 1.0, 2.0, 70_000.0], 2).unwrap_err().to_string();
        assert!(error.starts_with("Vector 1 of the batch has 70000 at component 1"), "{error}");
        assert_eq!(encode(f32::INFINITY), 0x7c00);
    }

    #[test]
    fn test_widen_agrees_with_decode_at_every_length() {
        let stored: Vec<u16> =
            (0..67u32).map(|i| (i.wrapping_mul(2654435761) >> 16) as u16).collect();
        let mut out = vec![1.0; 3];
        for len in 0..stored.len() {
            widen(&stored[..len], &mut out);
            let expected: Vec<f32> = stored[..len].iter().map(|&h| decode(h)).collect();
            assert_eq!(out.len(), len);
            assert!(out.iter().zip(&expected).all(|(a, b)| a.to_bits() == b.to_bits()), "{len}");
        }
    }
}
