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
///
/// The operations no longer use this: it moves the index once per element, which
/// is slow. It stays in test builds as the reference that [`for_each_row`] and
/// [`for_each_row2`] are checked against.
#[cfg(test)]
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

/// Like [`for_each_offset`] with two independent stride sets. Test builds only.
#[cfg(test)]
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

/// Calls `f(start, oa, ob)` once per row of `shape`. A row is a run of
/// `shape[last]` consecutive output elements: `start` is the output index of its
/// first element, and `oa` and `ob` are the offsets of that element under the two
/// stride sets. A rank-0 shape is one row of one element.
///
/// The rows come in the same order, and with the same offsets, as the elements
/// of `for_each_offset2` that start a row. The index is moved once per row and
/// not once per element, which leaves the caller a plain loop over the row.
pub(crate) fn for_each_row2(
    shape: &[usize],
    sa: &[usize],
    sb: &[usize],
    mut f: impl FnMut(usize, usize, usize),
) {
    let n = numel(shape);
    if n == 0 {
        return;
    }
    let inner = shape.last().copied().unwrap_or(1);
    let outer = shape.len().saturating_sub(1);
    let mut idx = vec![0usize; outer];
    let (mut oa, mut ob) = (0usize, 0usize);
    for row in 0..n / inner {
        f(row * inner, oa, ob);
        let mut d = outer;
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

/// Like [`for_each_row2`] with one stride set: calls `f(start, offset)` per row.
pub(crate) fn for_each_row(shape: &[usize], strides: &[usize], mut f: impl FnMut(usize, usize)) {
    let unused = vec![0usize; shape.len()];
    for_each_row2(shape, strides, &unused, |start, off, _| f(start, off));
}

/// Sums `grad` (laid out as `from`) down to the broadcast-compatible shape `to`.
/// This is the adjoint of broadcasting.
///
/// Every output element receives its contributions in row-major order of `grad`,
/// exactly as when the walk visits one element at a time, so the sums are
/// bit-for-bit what they were.
pub(crate) fn sum_to_shape(grad: &[f32], from: &[usize], to: &[usize]) -> Vec<f32> {
    if from == to {
        return grad.to_vec();
    }
    let mut out = vec![0.0f32; numel(to)];
    let to_strides = broadcast_strides(to, from);
    let inner = from.last().copied().unwrap_or(1);
    match to_strides.last().copied().unwrap_or(0) {
        // The last axis is kept: each row adds elementwise into a row of the output.
        1 => for_each_row(from, &to_strides, |i, off| {
            for (o, &g) in out[off..off + inner].iter_mut().zip(&grad[i..i + inner]) {
                *o += g;
            }
        }),
        // The last axis is summed away: the whole row goes into one output element.
        0 => for_each_row(from, &to_strides, |i, off| {
            let mut total = out[off];
            for &g in &grad[i..i + inner] {
                total += g;
            }
            out[off] = total;
        }),
        stride => for_each_row(from, &to_strides, |i, off| {
            for (j, &g) in grad[i..i + inner].iter().enumerate() {
                out[off + j * stride] += g;
            }
        }),
    }
    out
}

/// Reorders `src` (laid out as `shape`) so output axis `i` is input axis
/// `dims[i]`. Returns the new shape and the new data.
pub(crate) fn permute_data(src: &[f32], shape: &[usize], dims: &[usize]) -> (Vec<usize>, Vec<f32>) {
    let src_strides = strides(shape);
    let out_shape: Vec<usize> = dims.iter().map(|&d| shape[d]).collect();
    let walk: Vec<usize> = dims.iter().map(|&d| src_strides[d]).collect();
    let mut out = vec![0.0f32; src.len()];
    let inner = out_shape.last().copied().unwrap_or(1);
    match walk.last().copied().unwrap_or(0) {
        // The last axis stays last: rows are contiguous in the source and are copied whole.
        1 => for_each_row(&out_shape, &walk, |i, off| {
            out[i..i + inner].copy_from_slice(&src[off..off + inner]);
        }),
        stride => for_each_row(&out_shape, &walk, |i, off| {
            for (j, o) in out[i..i + inner].iter_mut().enumerate() {
                *o = src[off + j * stride];
            }
        }),
    }
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

    /// Values of mixed sign and magnitude, so the order of float additions shows in the bits.
    fn noise(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let unit = (state >> 40) as f32 / (1u64 << 24) as f32;
                let sign = if state & 1 == 0 { 1.0 } else { -1.0 };
                sign * (0.01 + unit * 4.0) * [1.0, 10.0, 0.1][(state >> 8) as usize % 3]
            })
            .collect()
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    /// Output shapes with an operand shape that broadcasts to each, covering every mix of
    /// kept and stretched last axes.
    const BROADCASTS: &[(&[usize], &[usize])] = &[
        (&[2, 3, 4], &[4]),
        (&[2, 3, 4], &[3, 1]),
        (&[2, 3, 4], &[2, 1, 1]),
        (&[2, 3, 4], &[1, 3, 4]),
        (&[2, 3, 4], &[2, 3, 1]),
        (&[2, 3, 4], &[]),
        (&[3, 5], &[5]),
        (&[6], &[1]),
        (&[5, 1], &[1]),
        (&[1, 5, 1], &[3, 1, 1]),
        (&[2, 2, 2, 2], &[2, 1, 2, 1]),
        (&[1], &[1]),
        (&[], &[]),
    ];

    #[test]
    fn rows_visit_the_same_offsets_as_the_per_element_walk() {
        for &(out, operand) in BROADCASTS {
            for other in [operand, out] {
                let sa = broadcast_strides(operand, out);
                let sb = broadcast_strides(other, out);
                let mut reference = Vec::new();
                for_each_offset2(out, &sa, &sb, |i, a, b| reference.push((i, a, b)));
                let inner = out.last().copied().unwrap_or(1);
                let (la, lb) = (sa.last().copied().unwrap_or(0), sb.last().copied().unwrap_or(0));
                let mut seen = Vec::new();
                for_each_row2(out, &sa, &sb, |start, a, b| {
                    for j in 0..inner {
                        seen.push((start + j, a + j * la, b + j * lb));
                    }
                });
                assert_eq!(seen, reference, "out {out:?}, operands {operand:?} and {other:?}");
                let mut one = Vec::new();
                for_each_row(out, &sa, |start, a| one.push((start, a)));
                let starts: Vec<_> =
                    reference.iter().step_by(inner).map(|&(i, a, _)| (i, a)).collect();
                assert_eq!(one, starts, "single stride set, out {out:?}");
            }
        }
    }

    #[test]
    fn a_row_walk_over_nothing_visits_nothing() {
        let mut calls = 0;
        for_each_row(&[3, 0], &[0, 0], |_, _| calls += 1);
        assert_eq!(calls, 0);
    }

    #[test]
    fn sum_to_shape_adds_in_exactly_the_per_element_order() {
        for (n, &(from, to)) in BROADCASTS.iter().enumerate() {
            let grad = noise(numel(from), 40 + n as u64);
            let mut want = vec![0.0f32; numel(to)];
            let to_strides = broadcast_strides(to, from);
            for_each_offset(from, &to_strides, |i, off| want[off] += grad[i]);
            let got = sum_to_shape(&grad, from, to);
            assert_eq!(bits(&got), bits(&want), "from {from:?} to {to:?}");
        }
    }

    #[test]
    fn permute_data_gathers_exactly_as_the_per_element_walk() {
        let cases: &[(&[usize], &[usize])] = &[
            (&[2, 3, 4], &[0, 1, 2]),
            (&[2, 3, 4], &[2, 0, 1]),
            (&[2, 3, 4], &[0, 2, 1]),
            (&[2, 3, 4], &[1, 0, 2]),
            (&[3, 5], &[1, 0]),
            (&[4, 3, 2, 5], &[0, 2, 1, 3]),
            (&[4, 3, 2, 5], &[3, 1, 0, 2]),
            (&[7], &[0]),
            (&[], &[]),
        ];
        for (n, &(shape, dims)) in cases.iter().enumerate() {
            let src = noise(numel(shape), 70 + n as u64);
            let strides_in = strides(shape);
            let out_shape: Vec<usize> = dims.iter().map(|&d| shape[d]).collect();
            let walk: Vec<usize> = dims.iter().map(|&d| strides_in[d]).collect();
            let mut want = vec![0.0f32; src.len()];
            for_each_offset(&out_shape, &walk, |i, off| want[i] = src[off]);
            let (got_shape, got) = permute_data(&src, shape, dims);
            assert_eq!(got_shape, out_shape);
            assert_eq!(bits(&got), bits(&want), "shape {shape:?}, dims {dims:?}");
        }
    }
}
