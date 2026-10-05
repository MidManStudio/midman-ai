// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-tensor.md, section "tensor.rs"
// ============================================================================
//! The [`Tensor`] type, its constructors and accessors, and the [`Grads`] map.
//!
//! A tensor is an immutable, reference-counted block of `f32` values plus a
//! shape. Every operation returns a new tensor. When any input of an operation
//! tracks gradients, the result remembers its inputs and a backward closure, so
//! [`Tensor::backward`] can walk the graph in reverse.

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use midman_foundation::{Error, Result, Rng};

use crate::shape;

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// Backward pass of one graph node: given the gradient of the node's output and
/// which parents need a gradient, returns one optional gradient per parent.
pub(crate) type BackwardFn = Box<dyn Fn(&[f32], &[bool]) -> Vec<Option<Vec<f32>>> + Send + Sync>;

/// How a non-leaf tensor was produced.
pub(crate) struct Node {
    pub(crate) parents: Vec<Tensor>,
    pub(crate) backward: BackwardFn,
}

pub(crate) struct Inner {
    pub(crate) id: u64,
    pub(crate) shape: Vec<usize>,
    pub(crate) data: Arc<Vec<f32>>,
    /// True for leaf tensors whose gradient should be recorded.
    pub(crate) requires_grad: bool,
    /// Present on results of operations that had a gradient-tracking input.
    pub(crate) node: Option<Node>,
}

impl Drop for Inner {
    /// Releases the graph iteratively. The default recursive drop would overflow
    /// the stack on a very deep graph (a long unrolled sequence, for example).
    fn drop(&mut self) {
        let Some(node) = self.node.take() else {
            return;
        };
        let Node { parents, backward } = node;
        // The closure holds clones of the parents; releasing it first leaves the
        // `parents` list as their last owner (unless something else holds them).
        drop(backward);
        let mut stack = parents;
        while let Some(tensor) = stack.pop() {
            if let Ok(mut inner) = Arc::try_unwrap(tensor.0) {
                if let Some(node) = inner.node.take() {
                    drop(node.backward);
                    stack.extend(node.parents);
                }
            }
        }
    }
}

/// A dense, row-major `f32` tensor with optional gradient tracking.
///
/// Cloning is cheap: it copies a pointer, not the data. See the crate docs for
/// the autograd model.
#[derive(Clone)]
pub struct Tensor(pub(crate) Arc<Inner>);

impl Tensor {
    pub(crate) fn from_parts(
        shape: Vec<usize>,
        data: Arc<Vec<f32>>,
        requires_grad: bool,
        node: Option<Node>,
    ) -> Tensor {
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        Tensor::from_parts_with_id(id, shape, data, requires_grad, node)
    }

    pub(crate) fn from_parts_with_id(
        id: u64,
        shape: Vec<usize>,
        data: Arc<Vec<f32>>,
        requires_grad: bool,
        node: Option<Node>,
    ) -> Tensor {
        debug_assert_eq!(shape::numel(&shape), data.len(), "shape and data length disagree");
        Tensor(Arc::new(Inner { id, shape, data, requires_grad, node }))
    }

    /// Creates a tensor from values laid out row-major.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] if the shape has a zero-sized axis or does not
    /// match the number of values.
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Result<Tensor> {
        shape::validate_shape(shape)?;
        if shape::numel(shape) != data.len() {
            return Err(Error::shape(
                "from_vec",
                format!(
                    "shape {:?} needs {} values but {} were given",
                    shape,
                    shape::numel(shape),
                    data.len()
                ),
            ));
        }
        Ok(Tensor::from_parts(shape.to_vec(), Arc::new(data), false, None))
    }

    /// A rank-0 tensor holding one value.
    pub fn scalar(value: f32) -> Tensor {
        Tensor::from_parts(Vec::new(), Arc::new(vec![value]), false, None)
    }

    /// A tensor filled with `value`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] if the shape has a zero-sized axis.
    pub fn full(shape: &[usize], value: f32) -> Result<Tensor> {
        shape::validate_shape(shape)?;
        Tensor::from_vec(vec![value; shape::numel(shape)], shape)
    }

    /// A tensor of zeros. See [`Tensor::full`].
    pub fn zeros(shape: &[usize]) -> Result<Tensor> {
        Tensor::full(shape, 0.0)
    }

    /// A tensor of ones. See [`Tensor::full`].
    pub fn ones(shape: &[usize]) -> Result<Tensor> {
        Tensor::full(shape, 1.0)
    }

    /// A tensor of samples from the uniform distribution on `[lo, hi)`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] if the shape has a zero-sized axis.
    pub fn uniform(shape: &[usize], lo: f32, hi: f32, rng: &mut Rng) -> Result<Tensor> {
        shape::validate_shape(shape)?;
        let data = (0..shape::numel(shape)).map(|_| rng.uniform(lo, hi)).collect();
        Tensor::from_vec(data, shape)
    }

    /// A tensor of normal samples with mean 0 and the given standard deviation.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] if the shape has a zero-sized axis.
    pub fn randn(shape: &[usize], std: f32, rng: &mut Rng) -> Result<Tensor> {
        shape::validate_shape(shape)?;
        let data = (0..shape::numel(shape)).map(|_| std * rng.normal()).collect();
        Tensor::from_vec(data, shape)
    }

    /// The tensor's shape.
    pub fn shape(&self) -> &[usize] {
        &self.0.shape
    }

    /// Number of axes (0 for a scalar).
    pub fn rank(&self) -> usize {
        self.0.shape.len()
    }

    /// Number of elements.
    pub fn numel(&self) -> usize {
        self.0.data.len()
    }

    /// A process-unique identifier. Gradients are keyed by it.
    pub fn id(&self) -> u64 {
        self.0.id
    }

    /// The values, row-major.
    pub fn data(&self) -> &[f32] {
        self.0.data.as_slice()
    }

    /// Copies the values out.
    pub fn to_vec(&self) -> Vec<f32> {
        self.0.data.to_vec()
    }

    /// The single value of a one-element tensor.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] if the tensor has more than one element.
    pub fn item(&self) -> Result<f32> {
        if self.numel() == 1 {
            Ok(self.0.data[0])
        } else {
            Err(Error::shape(
                "item",
                format!("tensor of shape {:?} is not a single value", self.shape()),
            ))
        }
    }

    /// True if this is a leaf tensor created with [`Tensor::leaf_requiring_grad`].
    pub fn requires_grad(&self) -> bool {
        self.0.requires_grad
    }

    /// True if gradients flow through this tensor: it is a gradient-requiring
    /// leaf, or it was computed from one.
    pub fn tracks_grad(&self) -> bool {
        self.0.requires_grad || self.0.node.is_some()
    }

    /// A tensor sharing this one's data but cut off from the graph.
    pub fn detach(&self) -> Tensor {
        Tensor::from_parts(self.0.shape.clone(), self.0.data.clone(), false, None)
    }

    /// A new leaf tensor that shares this one's data and records gradients.
    /// Its gradient is available from [`Grads::get`] after [`Tensor::backward`].
    pub fn leaf_requiring_grad(&self) -> Tensor {
        Tensor::from_parts(self.0.shape.clone(), self.0.data.clone(), true, None)
    }

    /// True if every value is finite (no NaN or infinity).
    pub fn is_finite(&self) -> bool {
        self.data().iter().all(|v| v.is_finite())
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let preview: Vec<f32> = self.data().iter().copied().take(6).collect();
        let more = if self.numel() > 6 { ", ..." } else { "" };
        write!(
            f,
            "Tensor(id={}, shape={:?}, tracks_grad={}, data={:?}{})",
            self.id(),
            self.shape(),
            self.tracks_grad(),
            preview,
            more
        )
    }
}

/// Gradients of a scalar with respect to the leaf tensors that required them.
///
/// Produced by [`Tensor::backward`]. Intermediate gradients are not kept.
#[derive(Debug, Default)]
pub struct Grads {
    pub(crate) map: HashMap<u64, Vec<f32>>,
}

impl Grads {
    /// The gradient for a leaf tensor, or `None` if no gradient reached it.
    pub fn get(&self, tensor: &Tensor) -> Option<&[f32]> {
        self.map.get(&tensor.id()).map(Vec::as_slice)
    }

    /// The gradient for the tensor with the given id.
    pub fn get_by_id(&self, id: u64) -> Option<&[f32]> {
        self.map.get(&id).map(Vec::as_slice)
    }

    /// Removes and returns a leaf tensor's gradient.
    pub fn take(&mut self, tensor: &Tensor) -> Option<Vec<f32>> {
        self.map.remove(&tensor.id())
    }

    /// Number of gradients held.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// True if no gradients are held.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_vec_checks_the_shape() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.rank(), 2);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(Tensor::from_vec(vec![1.0, 2.0], &[3]).is_err());
        assert!(Tensor::from_vec(vec![], &[0]).is_err());
    }

    #[test]
    fn scalars_have_rank_zero() {
        let s = Tensor::scalar(2.5);
        assert_eq!(s.rank(), 0);
        assert_eq!(s.shape(), &[] as &[usize]);
        assert_eq!(s.item().unwrap(), 2.5);
        assert!(Tensor::ones(&[2]).unwrap().item().is_err());
    }

    #[test]
    fn filled_constructors() {
        assert_eq!(Tensor::zeros(&[2, 2]).unwrap().to_vec(), vec![0.0; 4]);
        assert_eq!(Tensor::ones(&[3]).unwrap().to_vec(), vec![1.0; 3]);
        assert_eq!(Tensor::full(&[2], 7.0).unwrap().to_vec(), vec![7.0, 7.0]);
        assert!(Tensor::zeros(&[2, 0]).is_err());
    }

    #[test]
    fn random_constructors_are_reproducible() {
        let a = Tensor::randn(&[4, 4], 0.5, &mut Rng::seed_from_u64(1)).unwrap();
        let b = Tensor::randn(&[4, 4], 0.5, &mut Rng::seed_from_u64(1)).unwrap();
        let c = Tensor::randn(&[4, 4], 0.5, &mut Rng::seed_from_u64(2)).unwrap();
        assert_eq!(a.to_vec(), b.to_vec());
        assert_ne!(a.to_vec(), c.to_vec());
        let u = Tensor::uniform(&[100], -1.0, 1.0, &mut Rng::seed_from_u64(3)).unwrap();
        assert!(u.data().iter().all(|v| (-1.0..1.0).contains(v)));
    }

    #[test]
    fn ids_are_unique_and_clones_share_data_and_id() {
        let a = Tensor::scalar(1.0);
        let b = Tensor::scalar(1.0);
        assert_ne!(a.id(), b.id());
        let c = a.clone();
        assert_eq!(a.id(), c.id());
        assert!(std::ptr::eq(a.data().as_ptr(), c.data().as_ptr()));
    }

    #[test]
    fn detach_shares_data_but_drops_tracking() {
        let leaf = Tensor::ones(&[2]).unwrap().leaf_requiring_grad();
        assert!(leaf.requires_grad() && leaf.tracks_grad());
        let d = leaf.detach();
        assert!(!d.tracks_grad());
        assert!(std::ptr::eq(leaf.data().as_ptr(), d.data().as_ptr()));
        assert_ne!(leaf.id(), d.id());
    }

    #[test]
    fn is_finite_spots_nan_and_infinity() {
        assert!(Tensor::ones(&[3]).unwrap().is_finite());
        assert!(!Tensor::from_vec(vec![1.0, f32::NAN], &[2]).unwrap().is_finite());
        assert!(!Tensor::from_vec(vec![f32::INFINITY], &[1]).unwrap().is_finite());
    }

    #[test]
    fn debug_output_is_compact() {
        let t = Tensor::from_vec((0..10).map(|v| v as f32).collect(), &[10]).unwrap();
        let text = format!("{t:?}");
        assert!(text.contains("shape=[10]") && text.contains("..."), "{text}");
    }
}
