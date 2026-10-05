// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-tensor.md, section "autograd.rs"
// ============================================================================
//! Reverse-mode automatic differentiation.
//!
//! The graph is built eagerly as operations run. An operation whose inputs all
//! ignore gradients records nothing, so inference allocates no graph.
//! [`Tensor::backward`] orders the reachable nodes, seeds the output with 1.0,
//! and runs each node's backward closure from the output toward the leaves,
//! summing the gradients that reach a tensor from several uses.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use midman_foundation::{Error, Result};

use crate::tensor::{Grads, Node, Tensor};

/// Creates the result of an operation.
///
/// If any parent tracks gradients, the result becomes a graph node holding the
/// parents and `backward`. Otherwise `backward` is dropped and the result is a
/// plain constant. `backward` receives the output gradient and a flag per
/// parent saying whether that parent needs one, and returns one optional
/// gradient per parent, in parent order.
pub(crate) fn make_op<F>(
    shape: Vec<usize>,
    data: Arc<Vec<f32>>,
    parents: &[&Tensor],
    backward: F,
) -> Tensor
where
    F: Fn(&[f32], &[bool]) -> Vec<Option<Vec<f32>>> + Send + Sync + 'static,
{
    let node = parents.iter().any(|p| p.tracks_grad()).then(|| Node {
        parents: parents.iter().map(|p| (*p).clone()).collect(),
        backward: Box::new(backward),
    });
    Tensor::from_parts(shape, data, false, node)
}

/// Adds `grad` into the entry for `id`, creating it if absent.
fn accumulate(grads: &mut HashMap<u64, Vec<f32>>, id: u64, grad: Vec<f32>) {
    match grads.get_mut(&id) {
        Some(existing) => {
            debug_assert_eq!(existing.len(), grad.len(), "gradient length changed between uses");
            for (e, g) in existing.iter_mut().zip(&grad) {
                *e += g;
            }
        }
        None => {
            grads.insert(id, grad);
        }
    }
}

impl Tensor {
    /// Computes the gradient of this scalar with respect to every
    /// gradient-requiring leaf it depends on.
    ///
    /// Consumes the tensor, so the graph is released as soon as the gradients
    /// exist. If you still need the value, read it with [`Tensor::item`] first.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] if the tensor is not a single value, and
    /// [`Error::InvalidArgument`] if it does not track gradients.
    pub fn backward(self) -> Result<Grads> {
        if self.numel() != 1 {
            return Err(Error::shape(
                "backward",
                format!("needs a single value, got shape {:?}", self.shape()),
            ));
        }
        if !self.tracks_grad() {
            return Err(Error::invalid(
                "backward called on a tensor that does not track gradients",
            ));
        }

        // Iterative depth-first post-order: every node appears after all of its parents.
        let mut order: Vec<Tensor> = Vec::new();
        let mut visited: HashSet<u64> = HashSet::new();
        let mut stack: Vec<(Tensor, bool)> = vec![(self.clone(), false)];
        while let Some((tensor, expanded)) = stack.pop() {
            if expanded {
                order.push(tensor);
                continue;
            }
            if !visited.insert(tensor.id()) {
                continue;
            }
            stack.push((tensor.clone(), true));
            if let Some(node) = &tensor.0.node {
                for parent in &node.parents {
                    if parent.tracks_grad() && !visited.contains(&parent.id()) {
                        stack.push((parent.clone(), false));
                    }
                }
            }
        }

        let mut grads: HashMap<u64, Vec<f32>> = HashMap::new();
        grads.insert(self.id(), vec![1.0]);
        for tensor in order.iter().rev() {
            let Some(node) = &tensor.0.node else {
                continue; // a leaf keeps its accumulated gradient
            };
            let Some(grad_out) = grads.remove(&tensor.id()) else {
                continue; // nothing flowed into this node
            };
            let need: Vec<bool> = node.parents.iter().map(Tensor::tracks_grad).collect();
            let parent_grads = (node.backward)(&grad_out, &need);
            debug_assert_eq!(parent_grads.len(), node.parents.len());
            for (parent, grad) in node.parents.iter().zip(parent_grads) {
                if let (true, Some(grad)) = (parent.tracks_grad(), grad) {
                    debug_assert_eq!(grad.len(), parent.numel(), "gradient has the wrong length");
                    accumulate(&mut grads, parent.id(), grad);
                }
            }
        }
        Ok(Grads { map: grads })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaf(values: &[f32]) -> Tensor {
        Tensor::from_vec(values.to_vec(), &[values.len()]).unwrap().leaf_requiring_grad()
    }

    /// Builds `sum(a * b)` with hand-written backward closures, to test the engine
    /// itself independently of the real operations.
    fn dot(a: &Tensor, b: &Tensor) -> Tensor {
        let (a2, b2) = (a.clone(), b.clone());
        let value: f32 = a.data().iter().zip(b.data()).map(|(x, y)| x * y).sum();
        make_op(vec![], Arc::new(vec![value]), &[a, b], move |g, need| {
            vec![
                need[0].then(|| b2.data().iter().map(|y| g[0] * y).collect()),
                need[1].then(|| a2.data().iter().map(|x| g[0] * x).collect()),
            ]
        })
    }

    #[test]
    fn gradients_of_a_simple_product() {
        let a = leaf(&[1.0, 2.0, 3.0]);
        let b = leaf(&[4.0, 5.0, 6.0]);
        let grads = dot(&a, &b).backward().unwrap();
        assert_eq!(grads.get(&a).unwrap(), &[4.0, 5.0, 6.0]);
        assert_eq!(grads.get(&b).unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn a_tensor_used_twice_accumulates_both_gradients() {
        // f(a) = a . a, so df/da = 2a
        let a = leaf(&[1.0, -2.0, 3.0]);
        let grads = dot(&a, &a).backward().unwrap();
        assert_eq!(grads.get(&a).unwrap(), &[2.0, -4.0, 6.0]);
    }

    #[test]
    fn constants_get_no_gradient_and_are_not_asked_for_one() {
        let a = leaf(&[1.0, 2.0]);
        let c = Tensor::from_vec(vec![3.0, 4.0], &[2]).unwrap();
        let grads = dot(&a, &c).backward().unwrap();
        assert_eq!(grads.get(&a).unwrap(), &[3.0, 4.0]);
        assert!(grads.get(&c).is_none());
        assert_eq!(grads.len(), 1);
    }

    #[test]
    fn only_leaves_are_returned() {
        let a = leaf(&[1.0, 2.0]);
        let b = leaf(&[3.0, 4.0]);
        let out = dot(&a, &b);
        let out_id = out.id();
        let grads = out.backward().unwrap();
        assert!(grads.get_by_id(out_id).is_none());
        assert_eq!(grads.len(), 2);
    }

    #[test]
    fn a_diamond_is_ordered_correctly() {
        // y = (a . b) fed through two uses: z = y1 * y2 style via scalar graph.
        // z = (a.b) * (a.b) built from scalar nodes: dz/da = 2 (a.b) b
        let a = leaf(&[1.0, 2.0]);
        let b = leaf(&[3.0, 4.0]);
        let y = dot(&a, &b); // scalar 11
        let y_vec = Tensor::from_parts(
            vec![1],
            y.0.data.clone(),
            false,
            y.0.node.as_ref().map(|_| {
                let y2 = y.clone();
                Node {
                    parents: vec![y2],
                    backward: Box::new(|g: &[f32], _: &[bool]| vec![Some(g.to_vec())]),
                }
            }),
        );
        let z = dot(&y_vec, &y_vec); // 121
        assert_eq!(z.item().unwrap(), 121.0);
        let grads = z.backward().unwrap();
        // dz/dy = 2y = 22; dy/da = b; dy/db = a
        assert_eq!(grads.get(&a).unwrap(), &[66.0, 88.0]);
        assert_eq!(grads.get(&b).unwrap(), &[22.0, 44.0]);
    }

    #[test]
    fn backward_rejects_non_scalars_and_untracked_tensors() {
        let v = leaf(&[1.0, 2.0]);
        assert!(matches!(v.clone().backward(), Err(Error::Shape { .. })));
        let c = Tensor::scalar(1.0);
        assert!(matches!(c.backward(), Err(Error::InvalidArgument(_))));
    }

    #[test]
    fn an_op_on_untracked_inputs_builds_no_graph() {
        let a = Tensor::from_vec(vec![1.0], &[1]).unwrap();
        let b = Tensor::from_vec(vec![2.0], &[1]).unwrap();
        let out = dot(&a, &b);
        assert!(!out.tracks_grad());
        assert!(out.0.node.is_none());
    }

    #[test]
    fn deep_chains_do_not_overflow_the_stack() {
        // Chain of 20_000 identity-like nodes, then backward over it.
        let a = leaf(&[1.0]);
        let mut cur = a.clone();
        for _ in 0..20_000 {
            let parent = cur.clone();
            cur =
                make_op(vec![1], parent.0.data.clone(), &[&parent], |g, _| vec![Some(g.to_vec())]);
        }
        let s = make_op(vec![], Arc::new(vec![1.0]), &[&cur], |g, _| vec![Some(vec![g[0]])]);
        let grads = s.backward().unwrap();
        assert_eq!(grads.get(&a).unwrap(), &[1.0]);
        // Dropping the long chain must not recurse past the stack either.
        drop(cur);
    }
}
