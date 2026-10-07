// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-nn.md, section "embedding.rs"
// ============================================================================
//! A token embedding table.

use crate::init::check_std;
use crate::Module;
use midman_foundation::{Result, Rng};
use midman_tensor::{Parameter, Tensor};

/// A lookup table of shape `[num_embeddings, dim]`.
#[derive(Debug)]
pub struct Embedding {
    weight: Parameter,
}

impl Embedding {
    /// Creates a table named `"{name}.weight"` drawn from a normal distribution
    /// with standard deviation `std`.
    ///
    /// # Errors
    ///
    /// Returns an error if either size is 0 or `std` is not positive and finite.
    pub fn new(
        name: &str,
        num_embeddings: usize,
        dim: usize,
        std: f32,
        rng: &mut Rng,
    ) -> Result<Embedding> {
        check_std(name, std)?;
        let init = Tensor::randn(&[num_embeddings, dim], std, rng)?;
        Ok(Embedding { weight: Parameter::new(format!("{name}.weight"), init) })
    }

    /// Looks up one row per id. `ids` is row-major with shape `ids_shape`; the
    /// result has shape `ids_shape` followed by `dim`.
    ///
    /// # Errors
    ///
    /// Returns an error if `ids` does not match `ids_shape` or an id is out of range.
    pub fn forward(&self, ids: &[usize], ids_shape: &[usize]) -> Result<Tensor> {
        self.weight.tensor().embedding(ids, ids_shape)
    }

    /// The table. A model with tied embeddings also uses it as its output matrix.
    pub fn weight(&self) -> &Parameter {
        &self.weight
    }
}

impl Module for Embedding {
    fn parameters(&self) -> Vec<&Parameter> {
        vec![&self.weight]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        vec![&mut self.weight]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_returns_the_rows_in_order() {
        let mut table = Embedding::new("e", 3, 2, 0.1, &mut Rng::seed_from_u64(1)).unwrap();
        table.parameters_mut()[0].set_data(&[0.0, 1.0, 10.0, 11.0, 20.0, 21.0]).unwrap();
        let out = table.forward(&[2, 0, 1, 1], &[2, 2]).unwrap();
        assert_eq!(out.shape(), &[2, 2, 2]);
        assert_eq!(out.data(), &[20.0, 21.0, 0.0, 1.0, 10.0, 11.0, 10.0, 11.0]);
    }

    #[test]
    fn the_table_is_named_and_sized_from_the_arguments() {
        let table = Embedding::new("embed_tokens", 7, 4, 0.02, &mut Rng::seed_from_u64(2)).unwrap();
        assert_eq!(table.weight().name(), "embed_tokens.weight");
        assert_eq!(table.weight().shape(), &[7, 4]);
    }

    #[test]
    fn out_of_range_ids_and_bad_arguments_are_errors() {
        let mut rng = Rng::seed_from_u64(3);
        let table = Embedding::new("e", 3, 2, 0.1, &mut rng).unwrap();
        assert!(table.forward(&[3], &[1]).is_err());
        assert!(table.forward(&[0, 1], &[3]).is_err());
        assert!(Embedding::new("e", 0, 2, 0.1, &mut rng).is_err());
        assert!(Embedding::new("e", 3, 2, f32::INFINITY, &mut rng).is_err());
    }
}
