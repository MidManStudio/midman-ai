// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-optim.md, section "adamw.rs"
// ============================================================================
//! The AdamW optimizer.

use midman_foundation::{Error, Result};
use midman_tensor::Parameter;
use std::collections::{BTreeMap, BTreeSet};

/// First and second moment estimates for one parameter.
#[derive(Debug, Clone)]
struct Moments {
    m: Vec<f32>,
    v: Vec<f32>,
}

/// AdamW with decoupled weight decay, in the PyTorch formulation.
///
/// For step `t` (counting from 1), parameter `p`, gradient `g` and learning
/// rate `lr`:
///
/// ```text
/// p *= 1 - lr * weight_decay            (only if p has rank >= 2)
/// m  = beta1 * m + (1 - beta1) * g
/// v  = beta2 * v + (1 - beta2) * g * g
/// p -= lr / (1 - beta1^t) * m / (sqrt(v) / sqrt(1 - beta2^t) + eps)
/// ```
///
/// Moments are kept per parameter name, so parameter names must be unique.
#[derive(Debug, Clone)]
pub struct AdamW {
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    steps: u64,
    state: BTreeMap<String, Moments>,
}

impl AdamW {
    /// Creates an optimizer with no moment state.
    ///
    /// # Errors
    ///
    /// Returns an error unless both betas are in `[0, 1)`, `eps` is positive and
    /// finite, and `weight_decay` is finite and not negative.
    pub fn new(beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Result<AdamW> {
        for (name, beta) in [("beta1", beta1), ("beta2", beta2)] {
            if !(0.0..1.0).contains(&beta) {
                return Err(Error::invalid(format!("adamw: {name} must be in [0, 1), got {beta}")));
            }
        }
        if !(eps.is_finite() && eps > 0.0) {
            return Err(Error::invalid(format!(
                "adamw: eps must be positive and finite, got {eps}"
            )));
        }
        if !(weight_decay.is_finite() && weight_decay >= 0.0) {
            return Err(Error::invalid(format!(
                "adamw: weight_decay must be finite and not negative, got {weight_decay}"
            )));
        }
        Ok(AdamW { beta1, beta2, eps, weight_decay, steps: 0, state: BTreeMap::new() })
    }

    /// How many steps have been taken.
    pub fn step_count(&self) -> u64 {
        self.steps
    }

    /// Applies one update. `grads[i]` is the gradient of `params[i]`, and `lr`
    /// is the learning rate for this step (take it from a schedule).
    ///
    /// Everything is validated before anything changes, so a rejected step
    /// leaves the parameters, the moments and the step count as they were.
    ///
    /// # Errors
    ///
    /// Returns an error if the lists differ in length, a gradient does not match
    /// its parameter's size or contains a non-finite value, two parameters share
    /// a name, or `lr` is negative or not finite.
    pub fn step(
        &mut self,
        params: &mut [&mut Parameter],
        grads: &[Vec<f32>],
        lr: f32,
    ) -> Result<()> {
        if params.len() != grads.len() {
            return Err(Error::invalid(format!(
                "adamw: {} parameters but {} gradients",
                params.len(),
                grads.len()
            )));
        }
        if !(lr.is_finite() && lr >= 0.0) {
            return Err(Error::invalid(format!(
                "adamw: lr must be finite and not negative, got {lr}"
            )));
        }
        let mut seen = BTreeSet::new();
        for (p, g) in params.iter().zip(grads) {
            if !seen.insert(p.name().to_string()) {
                return Err(Error::invalid(format!(
                    "adamw: two parameters are named '{}'",
                    p.name()
                )));
            }
            if g.len() != p.numel() {
                return Err(Error::invalid(format!(
                    "adamw: parameter '{}' has {} values but its gradient has {}",
                    p.name(),
                    p.numel(),
                    g.len()
                )));
            }
            if g.iter().any(|v| !v.is_finite()) {
                return Err(Error::invalid(format!(
                    "adamw: gradient of '{}' is not finite",
                    p.name()
                )));
            }
            if let Some(old) = self.state.get(p.name()) {
                if old.m.len() != p.numel() {
                    return Err(Error::invalid(format!(
                        "adamw: parameter '{}' changed size since the last step",
                        p.name()
                    )));
                }
            }
        }

        self.steps += 1;
        let t = self.steps as f64;
        let bias1 = (1.0 - f64::from(self.beta1).powf(t)) as f32;
        let bias2_sqrt = (1.0 - f64::from(self.beta2).powf(t)).sqrt() as f32;
        let (beta1, beta2, eps) = (self.beta1, self.beta2, self.eps);
        let step_size = lr / bias1;

        for (p, g) in params.iter_mut().zip(grads) {
            let decay = if p.shape().len() >= 2 { 1.0 - lr * self.weight_decay } else { 1.0 };
            let moments = self
                .state
                .entry(p.name().to_string())
                .or_insert_with(|| Moments { m: vec![0.0; g.len()], v: vec![0.0; g.len()] });
            p.update(|data| {
                for (((x, &grad), m), v) in
                    data.iter_mut().zip(g).zip(&mut moments.m).zip(&mut moments.v)
                {
                    *x *= decay;
                    *m = beta1 * *m + (1.0 - beta1) * grad;
                    *v = beta2 * *v + (1.0 - beta2) * grad * grad;
                    *x -= step_size * *m / (v.sqrt() / bias2_sqrt + eps);
                }
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use midman_tensor::Tensor;

    fn params() -> (Parameter, Parameter) {
        (
            Parameter::new("w", Tensor::from_vec(vec![0.5, -0.5, 1.0, -1.0], &[2, 2]).unwrap()),
            Parameter::new("b", Tensor::from_vec(vec![0.1, 0.2, 0.3], &[3]).unwrap()),
        )
    }

    fn assert_close(got: &[f32], want: &[f32]) {
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-6, "{got:?} vs {want:?}");
        }
    }

    #[test]
    fn three_steps_match_an_independent_implementation() {
        // Reference values come from a separate float64 Python implementation of
        // the same update rule, with a matrix (decayed) and a vector (not decayed).
        let (mut w, mut b) = params();
        let mut opt = AdamW::new(0.9, 0.99, 1e-8, 0.1).unwrap();
        let steps: [(f32, [f32; 4], [f32; 3]); 3] = [
            (0.1, [0.1, -0.2, 0.3, -0.4], [1.0, -1.0, 0.5]),
            (0.05, [0.2, 0.1, -0.3, 0.4], [-0.5, 0.5, 1.0]),
            (0.02, [-0.1, 0.2, 0.3, 0.1], [0.0, 0.25, -0.75]),
        ];
        let after = [
            ([0.395, -0.395, 0.89, -0.89], [0.0, 0.3, 0.2]),
            ([0.3448313, -0.37969, 0.8881816, -0.8881816], [-0.013335, 0.313335, 0.1518063]),
            ([0.3357398, -0.3843846, 0.8796894, -0.8894384], [-0.0174675, 0.3145667, 0.1465412]),
        ];
        for ((lr, gw, gb), (want_w, want_b)) in steps.into_iter().zip(after) {
            opt.step(&mut [&mut w, &mut b], &[gw.to_vec(), gb.to_vec()], lr).unwrap();
            assert_close(w.data(), &want_w);
            assert_close(b.data(), &want_b);
        }
        assert_eq!(opt.step_count(), 3);
    }

    #[test]
    fn only_matrices_are_decayed() {
        // A zero gradient leaves just the decay: matrices shrink, vectors stay.
        let (mut w, mut b) = params();
        let mut opt = AdamW::new(0.9, 0.999, 1e-8, 0.5).unwrap();
        opt.step(&mut [&mut w, &mut b], &[vec![0.0; 4], vec![0.0; 3]], 0.1).unwrap();
        assert_close(w.data(), &[0.475, -0.475, 0.95, -0.95]);
        assert_eq!(b.data(), &[0.1, 0.2, 0.3]);
    }

    #[test]
    fn a_zero_learning_rate_changes_nothing_but_the_moments() {
        let (mut w, mut b) = params();
        let mut opt = AdamW::new(0.9, 0.999, 1e-8, 0.5).unwrap();
        opt.step(&mut [&mut w, &mut b], &[vec![1.0; 4], vec![1.0; 3]], 0.0).unwrap();
        assert_eq!(w.data(), &[0.5, -0.5, 1.0, -1.0]);
        assert_eq!(opt.step_count(), 1);
    }

    #[test]
    fn a_rejected_step_changes_nothing() {
        let (mut w, mut b) = params();
        let mut opt = AdamW::new(0.9, 0.999, 1e-8, 0.1).unwrap();
        let good = vec![vec![1.0; 4], vec![1.0; 3]];
        opt.step(&mut [&mut w, &mut b], &good, 0.1).unwrap();
        let (w_before, b_before) = (w.data().to_vec(), b.data().to_vec());

        let bad_lists: [Vec<Vec<f32>>; 4] = [
            vec![vec![1.0; 4]],
            vec![vec![1.0; 4], vec![1.0; 2]],
            vec![vec![1.0, f32::NAN, 1.0, 1.0], vec![1.0; 3]],
            vec![vec![1.0; 4], vec![f32::INFINITY, 1.0, 1.0]],
        ];
        for bad in &bad_lists {
            assert!(opt.step(&mut [&mut w, &mut b], bad, 0.1).is_err());
        }
        assert!(opt.step(&mut [&mut w, &mut b], &good, -0.1).is_err());
        assert!(opt.step(&mut [&mut w, &mut b], &good, f32::NAN).is_err());
        assert_eq!(w.data(), &w_before[..]);
        assert_eq!(b.data(), &b_before[..]);
        assert_eq!(opt.step_count(), 1);
    }

    #[test]
    fn duplicate_names_are_rejected() {
        let mut a = Parameter::new("same", Tensor::ones(&[2]).unwrap());
        let mut b = Parameter::new("same", Tensor::ones(&[2]).unwrap());
        let mut opt = AdamW::new(0.9, 0.999, 1e-8, 0.0).unwrap();
        let err = opt.step(&mut [&mut a, &mut b], &[vec![1.0; 2], vec![1.0; 2]], 0.1).unwrap_err();
        assert!(err.to_string().contains("same"), "{err}");
    }

    #[test]
    fn a_parameter_that_changes_size_is_rejected() {
        let mut opt = AdamW::new(0.9, 0.999, 1e-8, 0.0).unwrap();
        let mut a = Parameter::new("p", Tensor::ones(&[2]).unwrap());
        opt.step(&mut [&mut a], &[vec![1.0; 2]], 0.1).unwrap();
        let mut bigger = Parameter::new("p", Tensor::ones(&[3]).unwrap());
        assert!(opt.step(&mut [&mut bigger], &[vec![1.0; 3]], 0.1).is_err());
    }

    #[test]
    fn bad_hyperparameters_are_errors() {
        assert!(AdamW::new(1.0, 0.9, 1e-8, 0.0).is_err());
        assert!(AdamW::new(-0.1, 0.9, 1e-8, 0.0).is_err());
        assert!(AdamW::new(0.9, f32::NAN, 1e-8, 0.0).is_err());
        assert!(AdamW::new(0.9, 0.9, 0.0, 0.0).is_err());
        assert!(AdamW::new(0.9, 0.9, 1e-8, -1.0).is_err());
        assert!(AdamW::new(0.0, 0.0, 1e-8, 0.0).is_ok());
    }

    #[test]
    fn scaling_every_gradient_does_not_change_the_trajectory() {
        // Adam normalizes by the gradient's own size, so with a negligible epsilon a
        // gradient 1000 times larger moves the weights exactly as far.
        let run = |scale: f32| -> Vec<f32> {
            let mut w =
                Parameter::new("w", Tensor::from_vec(vec![0.5, -0.5, 1.0, -1.0], &[2, 2]).unwrap());
            let mut opt = AdamW::new(0.9, 0.99, 1e-20, 0.05).unwrap();
            for step in 0..6u32 {
                let g: Vec<f32> =
                    (0..4).map(|i| scale * ((step * 4 + i) as f32 * 0.37).sin()).collect();
                opt.step(&mut [&mut w], &[g], 0.05).unwrap();
            }
            w.data().to_vec()
        };
        assert_close(&run(1.0), &run(1000.0));
        assert_close(&run(1.0), &run(1e-3));
    }

    #[test]
    fn a_quadratic_converges_through_real_gradients() {
        // minimize sum((w - target)^2) using backward() for the gradients.
        let target = [2.0f32, -1.0, 0.5];
        let mut w = Parameter::new("w", Tensor::zeros(&[3]).unwrap());
        let mut opt = AdamW::new(0.9, 0.999, 1e-8, 0.0).unwrap();
        for _ in 0..300 {
            let diff = w.tensor().sub(&Tensor::from_vec(target.to_vec(), &[3]).unwrap()).unwrap();
            let mut grads = diff.mul(&diff).unwrap().sum_all().backward().unwrap();
            let mut params = [&mut w];
            let g = crate::gather_gradients(&params, &mut grads).unwrap();
            opt.step(&mut params, &g, 0.05).unwrap();
        }
        assert_close(w.data(), &target);
    }
}
