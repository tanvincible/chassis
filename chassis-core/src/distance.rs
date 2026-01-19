/// Distance metric for vector comparison
#[derive(Debug, Clone, Copy)]
pub enum DistanceMetric {
    Euclidean,
    Cosine,
    DotProduct,
}

/// Compute L2 (Euclidean) distance between two vectors
#[inline]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { euclidean_distance_avx2(a, b) };
        }
    }
    
    euclidean_distance_scalar(a, b)
}

/// Scalar implementation (portable)
#[inline]
fn euclidean_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    
    sum.sqrt()
}

/// AVX2 implementation (x86_64 only)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn euclidean_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        use std::arch::x86_64::{_mm256_setzero_ps, _mm256_storeu_ps};

        let mut sum = _mm256_setzero_ps();
        let chunks = a.len() / 8;

        for i in 0..chunks {
            use std::arch::x86_64::{_mm256_fmadd_ps, _mm256_loadu_ps, _mm256_sub_ps};

            let offset = i * 8;
            let va = _mm256_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
            let diff = _mm256_sub_ps(va, vb);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }

        let mut result = [0.0f32; 8];
        _mm256_storeu_ps(result.as_mut_ptr(), sum);

        let mut total = result. iter().sum::<f32>();

        // Handle remaining elements
        for i in (chunks * 8)..a.len() {
            let diff = a[i] - b[i];
            total += diff * diff;
        }

        total.sqrt()
    }
}

/// Compute cosine distance (1 - cosine_similarity)
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    1.0 - (dot / (norm_a * norm_b))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_euclidean_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        
        let dist = euclidean_distance(&a, &b);
        let expected = ((3.0_f32).powi(2) * 3.0).sqrt();
        
        assert!((dist - expected).abs() < 1e-6);
    }
    
    #[test]
    fn test_cosine_distance() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        
        let dist = cosine_distance(&a, &b);
        assert!((dist - 1.0).abs() < 1e-6); // Orthogonal vectors
    }
}
