// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-nn.md, section "rope.rs"
// ============================================================================
//! Rotary position embeddings.

use midman_foundation::{Error, Result};
use midman_tensor::Tensor;

/// Rotates each pair `(x[i], x[i + half])` of the last axis by an angle taken
/// from `cos` and `sin`.
///
/// `x` is `[.., T, D]` with `D` even, and `cos` and `sin` are `[T, D / 2]`;
/// they broadcast over the leading axes. Rotation preserves the length of each
/// vector. This is the "rotate half" layout, not interleaved pairs.
///
/// # Errors
///
/// Returns an error if `D` is odd or the tables do not broadcast against `x`.
pub fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let Some(&dim) = x.shape().last() else {
        return Err(Error::shape("apply_rope", "the input has no axes"));
    };
    let half = dim / 2;
    if half * 2 != dim {
        return Err(Error::shape("apply_rope", format!("the last axis must be even, got {dim}")));
    }
    let x1 = x.narrow(-1, 0, half)?;
    let x2 = x.narrow(-1, half, half)?;
    let first = x1.mul(cos)?.sub(&x2.mul(sin)?)?;
    let second = x2.mul(cos)?.add(&x1.mul(sin)?)?;
    Tensor::concat(&[first, second], -1)
}

/// Precomputed rotation angles for every position up to a maximum length.
///
/// Position `p` and frequency index `i` use the angle `p * theta^(-i / half)`
/// where `half = head_dim / 2`.
#[derive(Debug, Clone)]
pub struct RotaryTables {
    cos: Tensor,
    sin: Tensor,
    head_dim: usize,
    max_seq_len: usize,
}

impl RotaryTables {
    /// Builds the tables, computing the angles in `f64`.
    ///
    /// # Errors
    ///
    /// Returns an error if `head_dim` is 0 or odd, `max_seq_len` is 0, or
    /// `theta` is not positive and finite.
    pub fn new(head_dim: usize, max_seq_len: usize, theta: f32) -> Result<RotaryTables> {
        let half = head_dim / 2;
        if head_dim == 0 || half * 2 != head_dim {
            return Err(Error::invalid(format!(
                "rope: head_dim must be even and positive, got {head_dim}"
            )));
        }
        if max_seq_len == 0 {
            return Err(Error::invalid("rope: max_seq_len must be at least 1"));
        }
        if !(theta.is_finite() && theta > 0.0) {
            return Err(Error::invalid(format!(
                "rope: theta must be positive and finite, got {theta}"
            )));
        }
        let mut cos = Vec::with_capacity(max_seq_len * half);
        let mut sin = Vec::with_capacity(max_seq_len * half);
        for pos in 0..max_seq_len {
            for i in 0..half {
                let frequency = f64::from(theta).powf(-(i as f64) / half as f64);
                let angle = pos as f64 * frequency;
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
        Ok(RotaryTables {
            cos: Tensor::from_vec(cos, &[max_seq_len, half])?,
            sin: Tensor::from_vec(sin, &[max_seq_len, half])?,
            head_dim,
            max_seq_len,
        })
    }

    /// The size of the last axis the tables rotate.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// The longest sequence the tables cover.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Rotates `x` of shape `[batch, heads, time, head_dim]`, treating its time
    /// axis as positions `0..time`.
    ///
    /// # Errors
    ///
    /// Returns an error if `x` is not rank 4, its last axis is not `head_dim`,
    /// or its time axis is longer than `max_seq_len`.
    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let shape = x.shape();
        if shape.len() != 4 || shape[3] != self.head_dim {
            return Err(Error::shape(
                "rope",
                format!("expected [batch, heads, time, {}], got {shape:?}", self.head_dim),
            ));
        }
        let time = shape[2];
        if time > self.max_seq_len {
            return Err(Error::shape(
                "rope",
                format!("sequence length {time} exceeds the table length {}", self.max_seq_len),
            ));
        }
        let cos = self.cos.narrow(0, 0, time)?;
        let sin = self.sin.narrow(0, 0, time)?;
        apply_rope(x, &cos, &sin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use midman_foundation::Rng;

    #[test]
    fn position_zero_is_not_rotated() {
        let tables = RotaryTables::new(4, 8, 10_000.0).unwrap();
        let x = Tensor::uniform(&[1, 2, 1, 4], -1.0, 1.0, &mut Rng::seed_from_u64(1)).unwrap();
        let y = tables.apply(&x).unwrap();
        assert_eq!(x.data(), y.data());
    }

    #[test]
    fn table_entries_follow_the_formula() {
        // head_dim 4 gives half = 2. Position 3: i = 0 has angle 3, i = 1 has
        // angle 3 / theta^(1/2) = 3 / 100 for theta = 10000.
        let tables = RotaryTables::new(4, 5, 10_000.0).unwrap();
        let cos = tables.cos.data();
        let sin = tables.sin.data();
        assert!((cos[3 * 2] - 3f64.cos() as f32).abs() < 1e-6);
        assert!((sin[3 * 2] - 3f64.sin() as f32).abs() < 1e-6);
        assert!((cos[3 * 2 + 1] - 0.03f64.cos() as f32).abs() < 1e-6);
        assert!((sin[3 * 2 + 1] - 0.03f64.sin() as f32).abs() < 1e-6);
    }

    #[test]
    fn rotation_preserves_vector_length() {
        let tables = RotaryTables::new(6, 16, 10_000.0).unwrap();
        let x = Tensor::uniform(&[2, 3, 9, 6], -1.0, 1.0, &mut Rng::seed_from_u64(2)).unwrap();
        let y = tables.apply(&x).unwrap();
        for (a, b) in x.data().chunks_exact(6).zip(y.data().chunks_exact(6)) {
            let na: f32 = a.iter().map(|v| v * v).sum();
            let nb: f32 = b.iter().map(|v| v * v).sum();
            assert!((na - nb).abs() < 1e-4, "{na} vs {nb}");
        }
    }

    #[test]
    fn dot_products_depend_only_on_the_distance_between_positions() {
        // The same vector at every position, so the rotation is the only thing
        // that differs. Score (m, n) must equal score (m + 1, n + 1).
        let (d, t) = (8, 7);
        let mut rng = Rng::seed_from_u64(3);
        let q: Vec<f32> = (0..d).map(|_| rng.uniform(-1.0, 1.0)).collect();
        let k: Vec<f32> = (0..d).map(|_| rng.uniform(-1.0, 1.0)).collect();
        let tile = |v: &[f32]| Tensor::from_vec(v.repeat(t), &[1, 1, t, d]).unwrap();
        let tables = RotaryTables::new(d, t, 10_000.0).unwrap();
        let rq = tables.apply(&tile(&q)).unwrap();
        let rk = tables.apply(&tile(&k)).unwrap();
        let score = |m: usize, n: usize| -> f32 {
            (0..d).map(|i| rq.data()[m * d + i] * rk.data()[n * d + i]).sum()
        };
        for m in 0..t - 1 {
            for n in 0..t - 1 {
                assert!((score(m, n) - score(m + 1, n + 1)).abs() < 1e-4, "m {m} n {n}");
            }
        }
        // And it is not constant in the distance, or the rotation did nothing.
        assert!((score(0, 0) - score(0, 3)).abs() > 1e-3);
    }

    #[test]
    fn shorter_sequences_use_the_first_positions() {
        let tables = RotaryTables::new(4, 8, 10_000.0).unwrap();
        let long = Tensor::uniform(&[1, 1, 6, 4], -1.0, 1.0, &mut Rng::seed_from_u64(4)).unwrap();
        let short = long.narrow(2, 0, 3).unwrap();
        let a = tables.apply(&long).unwrap();
        let b = tables.apply(&short).unwrap();
        assert_eq!(&a.data()[..12], b.data());
    }

    #[test]
    fn bad_arguments_are_errors() {
        assert!(RotaryTables::new(0, 4, 10_000.0).is_err());
        assert!(RotaryTables::new(3, 4, 10_000.0).is_err());
        assert!(RotaryTables::new(4, 0, 10_000.0).is_err());
        assert!(RotaryTables::new(4, 4, 0.0).is_err());
        assert!(RotaryTables::new(4, 4, f32::NAN).is_err());
        let tables = RotaryTables::new(4, 4, 10_000.0).unwrap();
        assert!(tables.apply(&Tensor::ones(&[1, 1, 5, 4]).unwrap()).is_err());
        assert!(tables.apply(&Tensor::ones(&[1, 1, 4, 6]).unwrap()).is_err());
        assert!(tables.apply(&Tensor::ones(&[1, 4, 4]).unwrap()).is_err());
        let odd = Tensor::ones(&[1, 3]).unwrap();
        assert!(apply_rope(&odd, &Tensor::ones(&[1, 1]).unwrap(), &Tensor::ones(&[1, 1]).unwrap())
            .is_err());
    }
}
