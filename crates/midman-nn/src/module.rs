// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-nn.md, section "module.rs"
// ============================================================================
//! The [`Module`] trait: access to a layer's trainable parameters.

use midman_tensor::Parameter;

/// Something that owns trainable parameters.
///
/// The order of [`Module::parameters`] is fixed (declaration order, a layer's
/// children in the order they run), and [`Module::parameters_mut`] returns the
/// same parameters in the same order. Optimizers and checkpoints rely on both.
pub trait Module {
    /// Every parameter this module owns, including those of its children.
    fn parameters(&self) -> Vec<&Parameter>;

    /// The same parameters, mutably, in the same order.
    fn parameters_mut(&mut self) -> Vec<&mut Parameter>;
}

/// The total number of scalar values across every parameter of `module`.
pub fn count_parameters<M: Module + ?Sized>(module: &M) -> usize {
    module.parameters().iter().map(|p| p.numel()).sum()
}
