// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-tensor.md, section "param.rs"
// ============================================================================
//! [`Parameter`]: a named, trainable tensor that an optimizer can update in place.

use std::sync::Arc;

use midman_foundation::{Error, Result};

use crate::tensor::Tensor;

/// A named, trainable leaf tensor.
///
/// Models hold their weights as parameters. Each forward pass calls
/// [`Parameter::tensor`] to get a gradient-tracking handle, and the optimizer
/// changes the values with [`Parameter::update`] after the backward pass.
/// The parameter keeps the same [`Parameter::id`] for its whole life, so
/// gradients stay addressable across steps.
#[derive(Debug, Clone)]
pub struct Parameter {
    name: String,
    tensor: Tensor,
}

impl Parameter {
    /// Wraps `init` as a trainable parameter called `name`.
    pub fn new(name: impl Into<String>, init: Tensor) -> Parameter {
        Parameter { name: name.into(), tensor: init.leaf_requiring_grad() }
    }

    /// The parameter's name, for example `blocks.0.attn.wq`.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// A gradient-tracking handle to the current values, for use in a forward pass.
    pub fn tensor(&self) -> Tensor {
        self.tensor.clone()
    }

    /// The identifier gradients are keyed by. Stable across updates.
    pub fn id(&self) -> u64 {
        self.tensor.id()
    }

    /// The parameter's shape.
    pub fn shape(&self) -> &[usize] {
        self.tensor.shape()
    }

    /// Number of values.
    pub fn numel(&self) -> usize {
        self.tensor.numel()
    }

    /// The current values.
    pub fn data(&self) -> &[f32] {
        self.tensor.data()
    }

    /// Changes the values in place.
    ///
    /// If nothing else holds the parameter (the usual case once the step's
    /// graph has been dropped), the values are edited without copying. If a
    /// live graph still references the old values, they are copied first so
    /// that graph keeps seeing what it computed with.
    pub fn update(&mut self, f: impl FnOnce(&mut [f32])) {
        match Arc::get_mut(&mut self.tensor.0) {
            Some(inner) => {
                let values: &mut Vec<f32> = Arc::make_mut(&mut inner.data);
                f(values.as_mut_slice());
            }
            None => {
                let mut copy = self.tensor.0.data.as_ref().clone();
                f(&mut copy);
                self.tensor = Tensor::from_parts_with_id(
                    self.tensor.0.id,
                    self.tensor.0.shape.clone(),
                    Arc::new(copy),
                    true,
                    None,
                );
            }
        }
    }

    /// Replaces all values, for example when loading a checkpoint.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] if `values` has the wrong length.
    pub fn set_data(&mut self, values: &[f32]) -> Result<()> {
        if values.len() != self.numel() {
            return Err(Error::shape(
                "set_data",
                format!(
                    "parameter `{}` has {} values but {} were given",
                    self.name,
                    self.numel(),
                    values.len()
                ),
            ));
        }
        self.update(|data| data.copy_from_slice(values));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn param(values: &[f32]) -> Parameter {
        Parameter::new("w", Tensor::from_vec(values.to_vec(), &[values.len()]).unwrap())
    }

    #[test]
    fn a_new_parameter_tracks_gradients() {
        let p = param(&[1.0, 2.0]);
        assert_eq!(p.name(), "w");
        assert_eq!(p.shape(), &[2]);
        assert_eq!(p.numel(), 2);
        assert!(p.tensor().requires_grad());
    }

    #[test]
    fn update_edits_in_place_when_unshared() {
        let mut p = param(&[1.0, 2.0, 3.0]);
        let id = p.id();
        let before = p.data().as_ptr();
        p.update(|d| d.iter_mut().for_each(|v| *v += 10.0));
        assert_eq!(p.data(), &[11.0, 12.0, 13.0]);
        assert_eq!(p.id(), id);
        assert!(
            std::ptr::eq(p.data().as_ptr(), before),
            "an unshared update should not reallocate"
        );
    }

    #[test]
    fn update_leaves_a_live_graph_untouched() {
        let mut p = param(&[1.0, 2.0]);
        let held = p.tensor(); // stands in for a graph that still references the values
        let id = p.id();
        p.update(|d| d[0] = 99.0);
        assert_eq!(held.data(), &[1.0, 2.0], "the old handle must keep its values");
        assert_eq!(p.data(), &[99.0, 2.0]);
        assert_eq!(p.id(), id, "the id must survive the copy");
        assert!(p.tensor().requires_grad());
    }

    #[test]
    fn set_data_checks_the_length() {
        let mut p = param(&[1.0, 2.0]);
        p.set_data(&[5.0, 6.0]).unwrap();
        assert_eq!(p.data(), &[5.0, 6.0]);
        assert!(p.set_data(&[1.0]).is_err());
    }
}
