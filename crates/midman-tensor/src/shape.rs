// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-tensor.md, section "shape.rs"
// ============================================================================
//! Shape arithmetic shared by all tensor operations.
//!
//! Tensors are dense and row-major, so a shape alone fixes the memory layout.
//! A rank-0 shape (`[]`) is a scalar with one element.

use midman_foundation::{Error, Result};

/// Number of elements in a tensor of the given shape (1 for a scalar).
pub fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Row-major strides, in elements: the distance between neighbours along each axis.
pub fn strides(shape: &[usize]) -> Vec<usize> {
    let mut out = vec![0; shape.len()];
    let mut acc = 1;
    for (stride, &dim) in out.iter_mut().zip(shape).rev() {
        *stride = acc;
        acc *= dim;
    }
    out
}

/// Broadcasts two shapes together using NumPy rules, or returns `None` if they
/// are incompatible. Shapes are aligned on their trailing axes, and an axis of
/// size 1 stretches to match the other shape.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let rank = a.len().max(b.len());
    let mut out = vec![0; rank];
    for k in 0..rank {
        let da = if k < a.len() { a[a.len() - 1 - k] } else { 1 };
        let db = if k < b.len() { b[b.len() - 1 - k] } else { 1 };
        out[rank - 1 - k] = if da == db || db == 1 {
            da
        } else if da == 1 {
            db
        } else {
            return None;
        };
    }
    Some(out)
}

/// Turns a possibly negative axis (counted from the end) into an index in `0..rank`.
///
/// # Errors
///
/// Returns [`Error::Shape`] if the axis is out of range.
pub fn normalize_axis(axis: isize, rank: usize) -> Result<usize> {
    let r = rank as isize;
    if axis >= -r && axis < r {
        Ok(if axis < 0 { (axis + r) as usize } else { axis as usize })
    } else {
        Err(Error::shape("axis", format!("axis {axis} is out of range for rank {rank}")))
    }
}

/// Checks that no axis has size zero. Zero-sized tensors are not supported.
///
/// # Errors
///
/// Returns [`Error::Shape`] if any axis is zero.
pub fn validate_shape(shape: &[usize]) -> Result<()> {
    if shape.contains(&0) {
        Err(Error::shape(
            "tensor",
            format!("shape {shape:?} has a zero-sized axis, which is not supported"),
        ))
    } else {
        Ok(())
    }
}

/// Strides for reading a tensor of `shape` as if it were broadcast to `out`:
/// zero on every axis that is stretched or missing.
pub(crate) fn broadcast_strides(shape: &[usize], out: &[usize]) -> Vec<usize> {
    let own = strides(shape);
    let lead = out.len() - shape.len();
    let mut result = vec![0; out.len()];
    for (i, (&dim, &stride)) in shape.iter().zip(&own).enumerate() {
        if dim != 1 {
            result[lead + i] = stride;
        }
    }
    result
}

/// Splits a shape around `axis` into `(outer, len, inner)`: the element at
/// `(o, k, i)` lives at `(o * len + k) * inner + i`.
pub(crate) fn axis_split(shape: &[usize], axis: usize) -> (usize, usize, usize) {
    (numel(&shape[..axis]), shape[axis], numel(&shape[axis + 1..]))
}

/// Calls `f(i, offset)` for every element `i` of `shape` in row-major order,
/// where `offset` follows the given strides (which may be zero).
pub(crate) fn for_each_offset(shape: &[usize], strides: &[usize], mut f: impl FnMut(usize, usize)) {
    let n = numel(shape);
    let rank = shape.len();
    let mut idx = vec![0usize; rank];
    let mut off = 0usize;
    for i in 0..n {
        f(i, off);
        let mut d = rank;
        while d > 0 {
            d -= 1;
            idx[d] += 1;
            off += strides[d];
            if idx[d] < shape[d] {
                break;
            }
            off -= strides[d] * shape[d];
            idx[d] = 0;
        }
    }
}

/// Like [`for_each_offset`] with two independent stride sets.
pub(crate) fn for_each_offset2(
    shape: &[usize],
    sa: &[usize],
    sb: &[usize],
    mut f: impl FnMut(usize, usize, usize),
) {
    let n = numel(shape);
    let rank = shape.len();
    let mut idx = vec![0usize; rank];
    let (mut oa, mut ob) = (0usize, 0usize);
    for i in 0..n {
        f(i, oa, ob);
        let mut d = rank;
        while d > 0 {
            d -= 1;
            idx[d] += 1;
            oa += sa[d];
            ob += sb[d];
            if idx[d] < shape[d] {
                break;
            }
            oa -= sa[d] * shape[d];
            ob -= sb[d] * shape[d];
            idx[d] = 0;
        }
    }
}

/// Sums `grad` (laid out as `from`) down to the broadcast-compatible shape `to`.
/// This is the adjoint of broadcasting.
pub(crate) fn sum_to_shape(grad: &[f32], from: &[usize], to: &[usize]) -> Vec<f32> {
    if from == to {
        return grad.to_vec();
    }
    let mut out = vec![0.0f32; numel(to)];
    let to_strides = broadcast_strides(to, from);
    for_each_offset(from, &to_strides, |i, off| out[off] += grad[i]);
    out
}

/// Reorders `src` (laid out as `shape`) so output axis `i` is input axis
/// `dims[i]`. Returns the new shape and the new data.
pub(crate) fn permute_data(src: &[f32], shape: &[usize], dims: &[usize]) -> (Vec<usize>, Vec<f32>) {
    let src_strides = strides(shape);
    let out_shape: Vec<usize> = dims.iter().map(|&d| shape[d]).collect();
    let walk: Vec<usize> = dims.iter().map(|&d| src_strides[d]).collect();
    let mut out = vec![0.0f32; src.len()];
    for_each_offset(&out_shape, &walk, |i, off| out[i] = src[off]);
    (out_shape, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numel_and_strides() {
        assert_eq!(numel(&[]), 1);
        assert_eq!(numel(&[2, 3, 4]), 24);
        assert_eq!(strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(strides(&[]), Vec::<usize>::new());
        assert_eq!(strides(&[5]), vec![1]);
    }

    #[test]
    fn broadcasting_follows_numpy_rules() {
        assert_eq!(broadcast_shapes(&[2, 3], &[2, 3]), Some(vec![2, 3]));
        assert_eq!(broadcast_shapes(&[2, 3], &[3]), Some(vec![2, 3]));
        assert_eq!(broadcast_shapes(&[2, 1, 4], &[3, 1]), Some(vec![2, 3, 4]));
        assert_eq!(broadcast_shapes(&[], &[2, 2]), Some(vec![2, 2]));
        assert_eq!(broadcast_shapes(&[2, 3], &[2]), None);
        assert_eq!(broadcast_shapes(&[4, 3], &[2, 3]), None);
    }

    #[test]
    fn axes_can_be_negative() {
        assert_eq!(normalize_axis(0, 3).unwrap(), 0);
        assert_eq!(normalize_axis(-1, 3).unwrap(), 2);
        assert_eq!(normalize_axis(-3, 3).unwrap(), 0);
        assert!(normalize_axis(3, 3).is_err());
        assert!(normalize_axis(-4, 3).is_err());
        assert!(normalize_axis(0, 0).is_err());
    }

    #[test]
    fn zero_sized_axes_are_rejected() {
        assert!(validate_shape(&[2, 0]).is_err());
        assert!(validate_shape(&[]).is_ok());
        assert!(validate_shape(&[1, 5]).is_ok());
    }

    #[test]
    fn axis_split_matches_the_index_formula() {
        assert_eq!(axis_split(&[2, 3, 4], 1), (2, 3, 4));
        assert_eq!(axis_split(&[2, 3, 4], 0), (1, 2, 12));
        assert_eq!(axis_split(&[2, 3, 4], 2), (6, 4, 1));
    }

    #[test]
    fn offsets_walk_in_row_major_order_with_broadcast_strides() {
        // A [3] vector broadcast over a [2, 3] output revisits the same three offsets.
        let out = [2usize, 3];
        let s = broadcast_strides(&[3], &out);
        assert_eq!(s, vec![0, 1]);
        let mut seen = Vec::new();
        for_each_offset(&out, &s, |_, off| seen.push(off));
        assert_eq!(seen, vec![0, 1, 2, 0, 1, 2]);
        // A [2, 1] column broadcast over [2, 3] repeats each offset three times.
        let s = broadcast_strides(&[2, 1], &out);
        let mut seen = Vec::new();
        for_each_offset(&out, &s, |_, off| seen.push(off));
        assert_eq!(seen, vec![0, 0, 0, 1, 1, 1]);
    }

    #[test]
    fn two_stride_walk_tracks_both_offsets() {
        let out = [2usize, 2];
        let sa = strides(&out);
        let sb = broadcast_strides(&[2], &out);
        let mut seen = Vec::new();
        for_each_offset2(&out, &sa, &sb, |i, a, b| seen.push((i, a, b)));
        assert_eq!(seen, vec![(0, 0, 0), (1, 1, 1), (2, 2, 0), (3, 3, 1)]);
    }

    #[test]
    fn sum_to_shape_reduces_broadcast_axes() {
        let g = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // shape [2, 3]
        assert_eq!(sum_to_shape(&g, &[2, 3], &[3]), vec![5.0, 7.0, 9.0]);
        assert_eq!(sum_to_shape(&g, &[2, 3], &[2, 1]), vec![6.0, 15.0]);
        assert_eq!(sum_to_shape(&g, &[2, 3], &[]), vec![21.0]);
        assert_eq!(sum_to_shape(&g, &[2, 3], &[2, 3]), g.to_vec());
    }

    #[test]
    fn permute_data_transposes_and_reorders() {
        // [[1, 2, 3], [4, 5, 6]] transposed is [[1, 4], [2, 5], [3, 6]].
        let (shape, data) = permute_data(&[1., 2., 3., 4., 5., 6.], &[2, 3], &[1, 0]);
        assert_eq!(shape, vec![3, 2]);
        assert_eq!(data, vec![1., 4., 2., 5., 3., 6.]);
        // A 3-axis rotation: out[i][j][k] = in[k][i][j] for dims [2, 0, 1].
        let src: Vec<f32> = (0..24).map(|v| v as f32).collect();
        let (shape, data) = permute_data(&src, &[2, 3, 4], &[2, 0, 1]);
        assert_eq!(shape, vec![4, 2, 3]);
        // out[1][0][2] = in[0][2][1] = 0*12 + 2*4 + 1 = 9, stored at index (1*2 + 0)*3 + 2 = 8
        assert_eq!(data[8], 9.0);
    }
}
