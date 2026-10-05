// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-tensor.md, section "ops_shape.rs"
// ============================================================================
//! Operations that move data without arithmetic: reshape, permute, slicing,
//! concatenation, repetition and embedding lookup.

use std::sync::Arc;

use midman_foundation::{Error, Result};

use crate::autograd::make_op;
use crate::shape::{axis_split, normalize_axis, numel, permute_data, validate_shape};
use crate::tensor::Tensor;

impl Tensor {
    /// Views the same values under a new shape. No data is copied.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] if the new shape has a different element count
    /// or a zero-sized axis.
    pub fn reshape(&self, shape: &[usize]) -> Result<Tensor> {
        validate_shape(shape)?;
        if numel(shape) != self.numel() {
            return Err(Error::shape(
                "reshape",
                format!(
                    "cannot reshape {:?} ({} values) into {:?} ({} values)",
                    self.shape(),
                    self.numel(),
                    shape,
                    numel(shape)
                ),
            ));
        }
        Ok(make_op(shape.to_vec(), Arc::clone(&self.0.data), &[self], |g, _| {
            vec![Some(g.to_vec())]
        }))
    }

    /// Reorders the axes: axis `i` of the result is axis `dims[i]` of the input.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] if `dims` is not a permutation of `0..rank`.
    pub fn permute(&self, dims: &[usize]) -> Result<Tensor> {
        let rank = self.rank();
        let mut seen = vec![false; rank];
        let valid = dims.len() == rank
            && dims.iter().all(|&d| {
                let fresh = d < rank && !seen[d];
                if fresh {
                    seen[d] = true;
                }
                fresh
            });
        if !valid {
            return Err(Error::shape(
                "permute",
                format!("{dims:?} is not a permutation of the {rank} axes of {:?}", self.shape()),
            ));
        }
        let (out_shape, out) = permute_data(self.data(), self.shape(), dims);
        let mut inverse = vec![0usize; rank];
        for (i, &d) in dims.iter().enumerate() {
            inverse[d] = i;
        }
        let shape_back = out_shape.clone();
        Ok(make_op(out_shape, Arc::new(out), &[self], move |g, _| {
            let (_, dx) = permute_data(g, &shape_back, &inverse);
            vec![Some(dx)]
        }))
    }

    /// Swaps two axes. Negative axes count from the end.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] if either axis is out of range.
    pub fn transpose(&self, axis_a: isize, axis_b: isize) -> Result<Tensor> {
        let a = normalize_axis(axis_a, self.rank())?;
        let b = normalize_axis(axis_b, self.rank())?;
        let mut dims: Vec<usize> = (0..self.rank()).collect();
        dims.swap(a, b);
        self.permute(&dims)
    }

    /// Takes `len` consecutive entries starting at `start` along one axis.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] if `len` is zero or the range runs past the axis.
    pub fn narrow(&self, axis: isize, start: usize, len: usize) -> Result<Tensor> {
        let axis = normalize_axis(axis, self.rank())?;
        let (outer, full, inner) = axis_split(self.shape(), axis);
        if len == 0 || start + len > full {
            return Err(Error::shape(
                "narrow",
                format!("range {start}..{} does not fit axis {axis} of size {full}", start + len),
            ));
        }
        let mut out = Vec::with_capacity(outer * len * inner);
        for block in self.data().chunks_exact(full * inner) {
            out.extend_from_slice(&block[start * inner..(start + len) * inner]);
        }
        let mut out_shape = self.shape().to_vec();
        out_shape[axis] = len;
        Ok(make_op(out_shape, Arc::new(out), &[self], move |g, _| {
            let mut dx = vec![0.0f32; outer * full * inner];
            for (dblock, gblock) in
                dx.chunks_exact_mut(full * inner).zip(g.chunks_exact(len * inner))
            {
                dblock[start * inner..(start + len) * inner].copy_from_slice(gblock);
            }
            vec![Some(dx)]
        }))
    }

    /// Joins tensors along an existing axis. All other axes must match.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] if `parts` is empty or the shapes disagree
    /// anywhere except along `axis`.
    pub fn concat(parts: &[Tensor], axis: isize) -> Result<Tensor> {
        let first =
            parts.first().ok_or_else(|| Error::shape("concat", "needs at least one tensor"))?;
        let axis = normalize_axis(axis, first.rank())?;
        for p in parts {
            let same = p.rank() == first.rank()
                && p.shape()
                    .iter()
                    .zip(first.shape())
                    .enumerate()
                    .all(|(i, (a, b))| i == axis || a == b);
            if !same {
                return Err(Error::shape(
                    "concat",
                    format!(
                        "shape {:?} does not match {:?} outside axis {axis}",
                        p.shape(),
                        first.shape()
                    ),
                ));
            }
        }
        let lens: Vec<usize> = parts.iter().map(|p| p.shape()[axis]).collect();
        let total: usize = lens.iter().sum();
        let (outer, _, inner) = axis_split(first.shape(), axis);
        let mut out = Vec::with_capacity(outer * total * inner);
        for o in 0..outer {
            for (p, &len) in parts.iter().zip(&lens) {
                out.extend_from_slice(&p.data()[o * len * inner..(o + 1) * len * inner]);
            }
        }
        let mut out_shape = first.shape().to_vec();
        out_shape[axis] = total;
        let refs: Vec<&Tensor> = parts.iter().collect();
        Ok(make_op(out_shape, Arc::new(out), &refs, move |g, need| {
            let mut offset = 0;
            let mut grads = Vec::with_capacity(lens.len());
            for (&len, &wanted) in lens.iter().zip(need) {
                grads.push(wanted.then(|| {
                    let mut d = Vec::with_capacity(outer * len * inner);
                    for gblock in g.chunks_exact(total * inner) {
                        d.extend_from_slice(&gblock[offset * inner..(offset + len) * inner]);
                    }
                    d
                }));
                offset += len;
            }
            grads
        }))
    }

    /// Repeats each entry along an axis `times` times in a row, so `[a, b]`
    /// with `times = 2` becomes `[a, a, b, b]`. This is how grouped-query
    /// attention shares one key/value head among several query heads.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] if the axis is out of range or `times` is zero.
    pub fn repeat_interleave(&self, axis: isize, times: usize) -> Result<Tensor> {
        if times == 0 {
            return Err(Error::shape("repeat_interleave", "times must be at least 1"));
        }
        let axis = normalize_axis(axis, self.rank())?;
        let (outer, len, inner) = axis_split(self.shape(), axis);
        let out_len = len * times;
        let mut out = vec![0.0f32; outer * out_len * inner];
        for (src, dst) in
            self.data().chunks_exact(len * inner).zip(out.chunks_exact_mut(out_len * inner))
        {
            for (k, row) in src.chunks_exact(inner).enumerate() {
                for r in 0..times {
                    let at = (k * times + r) * inner;
                    dst[at..at + inner].copy_from_slice(row);
                }
            }
        }
        let mut out_shape = self.shape().to_vec();
        out_shape[axis] = out_len;
        Ok(make_op(out_shape, Arc::new(out), &[self], move |g, _| {
            let mut dx = vec![0.0f32; outer * len * inner];
            for (dblock, gblock) in
                dx.chunks_exact_mut(len * inner).zip(g.chunks_exact(out_len * inner))
            {
                for (k, drow) in dblock.chunks_exact_mut(inner).enumerate() {
                    for r in 0..times {
                        let at = (k * times + r) * inner;
                        for (d, &gv) in drow.iter_mut().zip(&gblock[at..at + inner]) {
                            *d += gv;
                        }
                    }
                }
            }
            vec![Some(dx)]
        }))
    }

    /// Looks up rows of this `[V, D]` table. The result has shape
    /// `ids_shape + [D]`. Gradients scatter-add back into the table.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] if the table is not rank 2 or `ids_shape` does
    /// not match the number of ids, and [`Error::IndexOutOfRange`] if an id is
    /// not below `V`.
    pub fn embedding(&self, ids: &[usize], ids_shape: &[usize]) -> Result<Tensor> {
        if self.rank() != 2 {
            return Err(Error::shape(
                "embedding",
                format!("the table must have shape [V, D], got {:?}", self.shape()),
            ));
        }
        validate_shape(ids_shape)?;
        if numel(ids_shape) != ids.len() {
            return Err(Error::shape(
                "embedding",
                format!(
                    "ids_shape {:?} describes {} ids but {} were given",
                    ids_shape,
                    numel(ids_shape),
                    ids.len()
                ),
            ));
        }
        let (v, d) = (self.shape()[0], self.shape()[1]);
        if let Some(&bad) = ids.iter().find(|&&id| id >= v) {
            return Err(Error::index("token id", bad, v));
        }
        let table = self.data();
        let mut out = Vec::with_capacity(ids.len() * d);
        for &id in ids {
            out.extend_from_slice(&table[id * d..(id + 1) * d]);
        }
        let mut out_shape = ids_shape.to_vec();
        out_shape.push(d);
        let ids = ids.to_vec();
        Ok(make_op(out_shape, Arc::new(out), &[self], move |g, _| {
            let mut dtable = vec![0.0f32; v * d];
            for (&id, grow) in ids.iter().zip(g.chunks_exact(d)) {
                for (t, &gv) in dtable[id * d..(id + 1) * d].iter_mut().zip(grow) {
                    *t += gv;
                }
            }
            vec![Some(dtable)]
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(values: &[f32], shape: &[usize]) -> Tensor {
        Tensor::from_vec(values.to_vec(), shape).unwrap()
    }

    fn seq(shape: &[usize]) -> Tensor {
        t(&(0..numel(shape)).map(|v| v as f32).collect::<Vec<_>>(), shape)
    }

    #[test]
    fn reshape_shares_the_data_and_checks_the_count() {
        let a = seq(&[2, 3]);
        let b = a.reshape(&[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert!(std::ptr::eq(a.data().as_ptr(), b.data().as_ptr()));
        assert!(a.reshape(&[4]).is_err());
        assert!(a.reshape(&[6, 0]).is_err());
        assert!(a.reshape(&[]).is_err());
    }

    #[test]
    fn permute_and_transpose() {
        let a = seq(&[2, 3]);
        assert_eq!(a.transpose(0, 1).unwrap().to_vec(), vec![0., 3., 1., 4., 2., 5.]);
        assert_eq!(a.transpose(-1, -2).unwrap().shape(), &[3, 2]);
        let cube = seq(&[2, 3, 4]);
        let p = cube.permute(&[2, 0, 1]).unwrap();
        assert_eq!(p.shape(), &[4, 2, 3]);
        assert_eq!(p.data()[8], 9.0); // out[1][0][2] = in[0][2][1]
        assert!(cube.permute(&[0, 1]).is_err());
        assert!(cube.permute(&[0, 1, 1]).is_err());
        assert!(cube.permute(&[0, 1, 3]).is_err());
    }

    #[test]
    fn narrow_slices_along_any_axis() {
        let a = seq(&[2, 4]);
        assert_eq!(a.narrow(1, 1, 2).unwrap().to_vec(), vec![1., 2., 5., 6.]);
        assert_eq!(a.narrow(0, 1, 1).unwrap().to_vec(), vec![4., 5., 6., 7.]);
        assert_eq!(a.narrow(-1, 2, 2).unwrap().shape(), &[2, 2]);
        assert!(a.narrow(1, 3, 2).is_err());
        assert!(a.narrow(1, 0, 0).is_err());
    }

    #[test]
    fn narrow_gradient_lands_in_the_slice() {
        let a = seq(&[2, 4]).leaf_requiring_grad();
        let grads = a.narrow(1, 1, 2).unwrap().sum_all().backward().unwrap();
        assert_eq!(grads.get(&a).unwrap(), &[0., 1., 1., 0., 0., 1., 1., 0.]);
    }

    #[test]
    fn concat_joins_along_an_axis() {
        let a = t(&[1., 2., 3., 4.], &[2, 2]);
        let b = t(&[5., 6.], &[2, 1]);
        let c = Tensor::concat(&[a.clone(), b], 1).unwrap();
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(c.to_vec(), vec![1., 2., 5., 3., 4., 6.]);
        let d = Tensor::concat(&[a.clone(), a], 0).unwrap();
        assert_eq!(d.to_vec(), vec![1., 2., 3., 4., 1., 2., 3., 4.]);
        assert!(Tensor::concat(&[], 0).is_err());
        assert!(Tensor::concat(&[seq(&[2, 2]), seq(&[3, 3])], 0).is_err());
        assert!(Tensor::concat(&[seq(&[2, 2]), seq(&[2])], 0).is_err());
    }

    #[test]
    fn concat_splits_the_gradient_back_to_each_part() {
        let a = t(&[1., 2.], &[1, 2]).leaf_requiring_grad();
        let b = t(&[3.], &[1, 1]).leaf_requiring_grad();
        let weights = t(&[10., 20., 30.], &[1, 3]);
        let loss =
            Tensor::concat(&[a.clone(), b.clone()], 1).unwrap().mul(&weights).unwrap().sum_all();
        let grads = loss.backward().unwrap();
        assert_eq!(grads.get(&a).unwrap(), &[10., 20.]);
        assert_eq!(grads.get(&b).unwrap(), &[30.]);
    }

    #[test]
    fn repeat_interleave_repeats_each_entry_in_place() {
        let a = t(&[1., 2., 3., 4.], &[2, 2]);
        assert_eq!(
            a.repeat_interleave(0, 2).unwrap().to_vec(),
            vec![1., 2., 1., 2., 3., 4., 3., 4.]
        );
        assert_eq!(
            a.repeat_interleave(1, 2).unwrap().to_vec(),
            vec![1., 1., 2., 2., 3., 3., 4., 4.]
        );
        assert!(a.repeat_interleave(1, 0).is_err());
    }

    #[test]
    fn repeat_interleave_gradient_sums_the_copies() {
        let a = t(&[1., 2.], &[2]).leaf_requiring_grad();
        let w = t(&[1., 10., 100., 1000.], &[4]);
        let grads =
            a.repeat_interleave(0, 2).unwrap().mul(&w).unwrap().sum_all().backward().unwrap();
        assert_eq!(grads.get(&a).unwrap(), &[11., 1100.]);
    }

    #[test]
    fn embedding_gathers_rows() {
        let table = t(&[0., 1., 10., 11., 20., 21.], &[3, 2]);
        let e = table.embedding(&[2, 0, 2, 1], &[2, 2]).unwrap();
        assert_eq!(e.shape(), &[2, 2, 2]);
        assert_eq!(e.to_vec(), vec![20., 21., 0., 1., 20., 21., 10., 11.]);
        let single = table.embedding(&[1], &[]).unwrap();
        assert_eq!(single.shape(), &[2]);
    }

    #[test]
    fn embedding_gradient_scatter_adds_repeated_ids() {
        let table = t(&[0.; 6], &[3, 2]).leaf_requiring_grad();
        let grads = table.embedding(&[2, 0, 2], &[3]).unwrap().sum_all().backward().unwrap();
        assert_eq!(grads.get(&table).unwrap(), &[1., 1., 0., 0., 2., 2.]);
    }

    #[test]
    fn embedding_validates_ids_and_shapes() {
        let table = t(&[0.; 6], &[3, 2]);
        assert!(matches!(table.embedding(&[3], &[1]), Err(Error::IndexOutOfRange { .. })));
        assert!(matches!(table.embedding(&[0, 1], &[3]), Err(Error::Shape { .. })));
        assert!(matches!(t(&[0.; 3], &[3]).embedding(&[0], &[1]), Err(Error::Shape { .. })));
    }
}
