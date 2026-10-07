// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-optim.md, section "clip.rs"
// ============================================================================
//! Moving gradients out of the autograd result, and clipping them.

use midman_foundation::{Error, Result};
use midman_tensor::{Grads, Parameter};

/// Takes the gradient of every parameter out of `grads`, in the order of `params`.
///
/// # Errors
///
/// Returns an error naming the first parameter that has no gradient. Every
/// parameter of a model is used on every step, so a missing gradient means the
/// forward pass is disconnected from that parameter.
pub fn gather_gradients(params: &[&mut Parameter], grads: &mut Grads) -> Result<Vec<Vec<f32>>> {
    params
        .iter()
        .map(|p| {
            grads.take(&p.tensor()).ok_or_else(|| {
                Error::invalid(format!("parameter '{}' received no gradient", p.name()))
            })
        })
        .collect()
}

/// The L2 norm of all gradients taken together, accumulated in `f64`.
pub fn global_norm(grads: &[Vec<f32>]) -> f32 {
    let sum: f64 = grads.iter().flatten().map(|&g| f64::from(g) * f64::from(g)).sum();
    sum.sqrt() as f32
}

/// Scales every gradient by the same factor so their global norm is at most
/// `max_norm`, and returns the norm before clipping. Gradients already within
/// the limit are left untouched.
///
/// # Errors
///
/// Returns an error if `max_norm` is not positive and finite, or the gradient
/// norm is not finite (a diverged run), in which case nothing is modified.
pub fn clip_global_norm(grads: &mut [Vec<f32>], max_norm: f32) -> Result<f32> {
    if !(max_norm.is_finite() && max_norm > 0.0) {
        return Err(Error::invalid(format!(
            "clip: max_norm must be positive and finite, got {max_norm}"
        )));
    }
    let norm = global_norm(grads);
    if !norm.is_finite() {
        return Err(Error::invalid("clip: the gradient norm is not finite"));
    }
    if norm > max_norm {
        let factor = max_norm / (norm + 1e-6);
        for g in grads.iter_mut().flatten() {
            *g *= factor;
        }
    }
    Ok(norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use midman_tensor::Tensor;

    #[test]
    fn the_norm_spans_every_gradient() {
        let grads = vec![vec![3.0], vec![0.0, 4.0]];
        assert!((global_norm(&grads) - 5.0).abs() < 1e-6);
        assert_eq!(global_norm(&[]), 0.0);
    }

    #[test]
    fn clipping_rescales_to_the_limit_and_keeps_directions() {
        let mut grads = vec![vec![3.0, 0.0], vec![4.0]];
        let before = clip_global_norm(&mut grads, 1.0).unwrap();
        assert!((before - 5.0).abs() < 1e-6);
        assert!((global_norm(&grads) - 1.0).abs() < 1e-4);
        assert!((grads[0][0] / grads[1][0] - 0.75).abs() < 1e-5);
    }

    #[test]
    fn gradients_within_the_limit_are_untouched() {
        let mut grads = vec![vec![0.3, 0.4]];
        let before = clip_global_norm(&mut grads, 1.0).unwrap();
        assert!((before - 0.5).abs() < 1e-6);
        assert_eq!(grads, vec![vec![0.3, 0.4]]);
    }

    #[test]
    fn a_non_finite_norm_is_an_error_and_changes_nothing() {
        for bad in [f32::NAN, f32::INFINITY] {
            let mut grads = vec![vec![1.0, bad]];
            assert!(clip_global_norm(&mut grads, 1.0).is_err());
            assert_eq!(grads[0][0], 1.0);
        }
        assert!(clip_global_norm(&mut [vec![1.0]], 0.0).is_err());
        assert!(clip_global_norm(&mut [vec![1.0]], f32::NAN).is_err());
    }

    #[test]
    fn gathering_follows_the_parameter_order_and_empties_the_map() {
        let mut a = Parameter::new("a", Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap());
        let mut b = Parameter::new("b", Tensor::from_vec(vec![3.0], &[1]).unwrap());
        let loss = a
            .tensor()
            .mul(&a.tensor())
            .unwrap()
            .sum_all()
            .add(&b.tensor().scale(5.0).sum_all())
            .unwrap();
        let mut grads = loss.backward().unwrap();
        let params = [&mut b, &mut a];
        let gathered = gather_gradients(&params, &mut grads).unwrap();
        assert_eq!(gathered, vec![vec![5.0], vec![2.0, 4.0]]);
        assert!(grads.is_empty());
    }

    #[test]
    fn a_parameter_outside_the_graph_is_named_in_the_error() {
        let mut used = Parameter::new("used", Tensor::ones(&[1]).unwrap());
        let mut unused = Parameter::new("unused.weight", Tensor::ones(&[1]).unwrap());
        let mut grads = used.tensor().sum_all().backward().unwrap();
        let params = [&mut used, &mut unused];
        let err = gather_gradients(&params, &mut grads).unwrap_err();
        assert!(err.to_string().contains("unused.weight"), "{err}");
    }
}
