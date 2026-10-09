// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-nn.md, section "attention.rs"
// ============================================================================
//! Causal grouped-query self-attention.

use crate::init::Init;
use crate::{Linear, Module, RotaryTables};
use midman_foundation::{Error, Result, Rng};
use midman_tensor::{Parameter, Tensor};

/// `a` is an exact multiple of `b`. Written without `%` on purpose; see the
/// design notes in `docs/midman-nn.md`.
fn divides(a: usize, b: usize) -> bool {
    b != 0 && (a / b) * b == a
}

/// An additive causal mask of shape `[t, t]`: 0 on and below the diagonal and a
/// large negative number above it, so a softmax gives future positions weight 0.
///
/// # Errors
///
/// Returns an error if `t` is 0.
pub fn causal_mask(t: usize) -> Result<Tensor> {
    let data = (0..t * t).map(|i| if i % t > i / t { -1.0e9 } else { 0.0 }).collect();
    Tensor::from_vec(data, &[t, t])
}

/// `softmax(q k^T / sqrt(head_dim) + causal_mask) v` with grouped-query sharing.
///
/// `q` is `[batch, n_heads, time, head_dim]`; `k` and `v` are
/// `[batch, n_kv_heads, time, head_dim]` where `n_heads` is a multiple of
/// `n_kv_heads`. Consecutive groups of `n_heads / n_kv_heads` query heads share
/// one key/value head. The result is `[batch, n_heads, time, head_dim]`.
///
/// # Errors
///
/// Returns an error if the shapes do not have that form.
pub fn causal_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let (qs, ks, vs) = (q.shape(), k.shape(), v.shape());
    if qs.len() != 4 || ks.len() != 4 || ks != vs {
        return Err(Error::shape(
            "causal_attention",
            format!(
                "expected q [B, H, T, D] and k, v [B, Hkv, T, D]; got {qs:?}, {ks:?} and {vs:?}"
            ),
        ));
    }
    if qs[0] != ks[0] || qs[2] != ks[2] || qs[3] != ks[3] {
        return Err(Error::shape(
            "causal_attention",
            format!("q {qs:?} and k {ks:?} disagree on batch, time or head size"),
        ));
    }
    let (n_heads, n_kv_heads) = (qs[1], ks[1]);
    if !divides(n_heads, n_kv_heads) {
        return Err(Error::shape(
            "causal_attention",
            format!("{n_heads} query heads is not a multiple of {n_kv_heads} key/value heads"),
        ));
    }
    let groups = n_heads / n_kv_heads;
    let (k, v) = if groups > 1 {
        (k.repeat_interleave(1, groups)?, v.repeat_interleave(1, groups)?)
    } else {
        (k.clone(), v.clone())
    };
    let scale = 1.0 / (qs[3] as f32).sqrt();
    let scores = q.matmul(&k.transpose(-1, -2)?)?.scale(scale).add(&causal_mask(qs[2])?)?;
    scores.softmax(-1)?.matmul(&v)
}

/// The four projection matrices of an attention layer, as plain tensors.
#[derive(Debug, Clone)]
pub struct AttentionWeights {
    /// Query projection, `[d_model, n_heads * head_dim]`.
    pub q: Tensor,
    /// Key projection, `[d_model, n_kv_heads * head_dim]`.
    pub k: Tensor,
    /// Value projection, `[d_model, n_kv_heads * head_dim]`.
    pub v: Tensor,
    /// Output projection, `[n_heads * head_dim, d_model]`.
    pub o: Tensor,
}

/// The full attention computation over explicit weights: project, split into
/// heads, rotate queries and keys, attend, merge heads, project out.
///
/// `x` is `[batch, time, d_model]` and so is the result. The head size is taken
/// from `rope`.
///
/// # Errors
///
/// Returns an error if `x` is not rank 3 or the weights do not fit the head
/// counts and head size.
pub fn attention_forward(
    x: &Tensor,
    weights: &AttentionWeights,
    n_heads: usize,
    n_kv_heads: usize,
    rope: &RotaryTables,
) -> Result<Tensor> {
    let &[batch, time, _] = x.shape() else {
        return Err(Error::shape(
            "attention",
            format!("expected [batch, time, d_model], got {:?}", x.shape()),
        ));
    };
    let head_dim = rope.head_dim();
    let heads = |projected: Tensor, count: usize| -> Result<Tensor> {
        projected.reshape(&[batch, time, count, head_dim])?.permute(&[0, 2, 1, 3])
    };
    let q = rope.apply(&heads(x.matmul(&weights.q)?, n_heads)?)?;
    let k = rope.apply(&heads(x.matmul(&weights.k)?, n_kv_heads)?)?;
    let v = heads(x.matmul(&weights.v)?, n_kv_heads)?;
    let context = causal_attention(&q, &k, &v)?;
    context.permute(&[0, 2, 1, 3])?.reshape(&[batch, time, n_heads * head_dim])?.matmul(&weights.o)
}

/// A causal self-attention layer with grouped-query sharing and no biases.
#[derive(Debug)]
pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    /// Creates the layer. Its weights are named `"{name}.q_proj.weight"`,
    /// `k_proj`, `v_proj` and `o_proj`; the output projection uses `init.out_std`.
    ///
    /// # Errors
    ///
    /// Returns an error if `d_model` is not a multiple of `n_heads`, `n_heads`
    /// is not a multiple of `n_kv_heads`, or any size is 0.
    pub fn new(
        name: &str,
        d_model: usize,
        n_heads: usize,
        n_kv_heads: usize,
        init: Init,
        rng: &mut Rng,
    ) -> Result<Attention> {
        if !divides(d_model, n_heads) || !divides(n_heads, n_kv_heads) {
            return Err(Error::invalid(format!(
                "{name}: d_model {d_model} must be a multiple of n_heads {n_heads}, \
                 and n_heads a multiple of n_kv_heads {n_kv_heads}"
            )));
        }
        let head_dim = d_model / n_heads;
        let kv_dim = n_kv_heads * head_dim;
        Ok(Attention {
            q_proj: Linear::new(&format!("{name}.q_proj"), d_model, d_model, init.std, rng)?,
            k_proj: Linear::new(&format!("{name}.k_proj"), d_model, kv_dim, init.std, rng)?,
            v_proj: Linear::new(&format!("{name}.v_proj"), d_model, kv_dim, init.std, rng)?,
            o_proj: Linear::new(&format!("{name}.o_proj"), d_model, d_model, init.out_std, rng)?,
            n_heads,
            n_kv_heads,
            head_dim,
        })
    }

    /// Attends over `x` of shape `[batch, time, d_model]`.
    ///
    /// # Errors
    ///
    /// Returns an error if `x` does not have that shape, or `rope` was built for
    /// a different head size.
    pub fn forward(&self, x: &Tensor, rope: &RotaryTables) -> Result<Tensor> {
        if rope.head_dim() != self.head_dim {
            return Err(Error::shape(
                "attention",
                format!(
                    "rope tables are for head size {}, the layer uses {}",
                    rope.head_dim(),
                    self.head_dim
                ),
            ));
        }
        let weights = AttentionWeights {
            q: self.q_proj.weight().tensor(),
            k: self.k_proj.weight().tensor(),
            v: self.v_proj.weight().tensor(),
            o: self.o_proj.weight().tensor(),
        };
        attention_forward(x, &weights, self.n_heads, self.n_kv_heads, rope)
    }

    /// The size of one attention head.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
}

impl Module for Attention {
    fn parameters(&self) -> Vec<&Parameter> {
        let mut all = self.q_proj.parameters();
        all.extend(self.k_proj.parameters());
        all.extend(self.v_proj.parameters());
        all.extend(self.o_proj.parameters());
        all
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        let mut all = self.q_proj.parameters_mut();
        all.extend(self.k_proj.parameters_mut());
        all.extend(self.v_proj.parameters_mut());
        all.extend(self.o_proj.parameters_mut());
        all
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_t(shape: &[usize], seed: u64) -> Tensor {
        Tensor::uniform(shape, -1.0, 1.0, &mut Rng::seed_from_u64(seed)).unwrap()
    }

    #[test]
    fn the_mask_is_zero_on_and_below_the_diagonal() {
        let mask = causal_mask(3).unwrap();
        let big = -1.0e9;
        assert_eq!(mask.data(), &[0.0, big, big, 0.0, 0.0, big, 0.0, 0.0, 0.0]);
        assert!(causal_mask(0).is_err());
    }

    #[test]
    fn a_single_position_attends_only_to_itself() {
        let (q, k, v) =
            (rand_t(&[1, 2, 1, 4], 1), rand_t(&[1, 2, 1, 4], 2), rand_t(&[1, 2, 1, 4], 3));
        let out = causal_attention(&q, &k, &v).unwrap();
        for (a, b) in out.data().iter().zip(v.data()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn a_known_answer_pins_down_the_score_scaling() {
        // head size 4, so scores are divided by 2. Position 1 has q = [2, 0, 0, 0] and
        // keys k0 = [1, 0, 0, 0], k1 = 0: scaled scores [1, 0], so the weights are
        // [e / (1 + e), 1 / (1 + e)] = [sigmoid(1), 1 - sigmoid(1)]. Without the
        // scaling they would be [sigmoid(2), 1 - sigmoid(2)].
        let t = |d: Vec<f32>| Tensor::from_vec(d, &[1, 1, 2, 4]).unwrap();
        let q = t(vec![2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0]);
        let k = t(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let v = t(vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let out = causal_attention(&q, &k, &v).unwrap();
        let w0 = 1.0 / (1.0 + (-1.0f32).exp());
        let expected = [1.0, 0.0, 0.0, 0.0, w0, 1.0 - w0, 0.0, 0.0];
        for (got, want) in out.data().iter().zip(expected) {
            assert!((got - want).abs() < 1e-6, "{:?} vs {expected:?}", out.data());
        }
    }

    #[test]
    fn the_future_does_not_leak_into_the_past() {
        let (q, k, v) =
            (rand_t(&[1, 2, 5, 3], 4), rand_t(&[1, 2, 5, 3], 5), rand_t(&[1, 2, 5, 3], 6));
        let base = causal_attention(&q, &k, &v).unwrap();
        // Change the last position of k and v for every head.
        let bump = |t: &Tensor| {
            let mut data = t.to_vec();
            for head in 0..2 {
                for d in 0..3 {
                    data[(head * 5 + 4) * 3 + d] += 10.0;
                }
            }
            Tensor::from_vec(data, t.shape()).unwrap()
        };
        let changed = causal_attention(&q, &bump(&k), &bump(&v)).unwrap();
        for head in 0..2 {
            let start = head * 5 * 3;
            assert_eq!(base.data()[start..start + 12], changed.data()[start..start + 12]);
            assert_ne!(base.data()[start + 12..start + 15], changed.data()[start + 12..start + 15]);
        }
    }

    #[test]
    fn grouped_attention_equals_attention_over_copied_heads() {
        // 4 query heads over 2 key/value heads: copy each kv head twice by hand.
        let q = rand_t(&[2, 4, 3, 2], 7);
        let k = rand_t(&[2, 2, 3, 2], 8);
        let v = rand_t(&[2, 2, 3, 2], 9);
        let grouped = causal_attention(&q, &k, &v).unwrap();
        let copy = |t: &Tensor| {
            let parts: Vec<Tensor> = (0..2)
                .flat_map(|h| {
                    let one = t.narrow(1, h, 1).unwrap();
                    [one.clone(), one]
                })
                .collect();
            Tensor::concat(&parts, 1).unwrap()
        };
        let full = causal_attention(&q, &copy(&k), &copy(&v)).unwrap();
        assert_eq!(grouped.data(), full.data());
    }

    #[test]
    fn multi_query_attention_with_one_kv_head_works() {
        let out = causal_attention(
            &rand_t(&[1, 4, 3, 2], 10),
            &rand_t(&[1, 1, 3, 2], 11),
            &rand_t(&[1, 1, 3, 2], 12),
        );
        assert_eq!(out.unwrap().shape(), &[1, 4, 3, 2]);
    }

    #[test]
    fn huge_scores_stay_finite() {
        // Scores in the thousands would overflow a softmax that skips the row maximum.
        let big = |seed| rand_t(&[1, 2, 4, 4], seed).scale(300.0);
        let out = causal_attention(&big(40), &big(41), &rand_t(&[1, 2, 4, 4], 42)).unwrap();
        assert!(out.is_finite());
        // Each output row is a convex combination of value rows, so it stays within their range.
        assert!(out.data().iter().all(|v| v.abs() <= 1.0 + 1e-5));
    }

    #[test]
    fn batch_entries_are_independent() {
        let q = rand_t(&[3, 2, 4, 3], 43);
        let (k, v) = (rand_t(&[3, 2, 4, 3], 44), rand_t(&[3, 2, 4, 3], 45));
        let together = causal_attention(&q, &k, &v).unwrap();
        let one = |t: &Tensor| t.narrow(0, 1, 1).unwrap();
        let alone = causal_attention(&one(&q), &one(&k), &one(&v)).unwrap();
        let size = 2 * 4 * 3;
        assert_eq!(&together.data()[size..2 * size], alone.data());
    }

    #[test]
    fn bad_shapes_are_errors() {
        let q = rand_t(&[1, 4, 3, 2], 13);
        let ok = rand_t(&[1, 2, 3, 2], 14);
        assert!(
            causal_attention(&q, &rand_t(&[1, 3, 3, 2], 15), &rand_t(&[1, 3, 3, 2], 16)).is_err()
        );
        assert!(causal_attention(&q, &ok, &rand_t(&[1, 1, 3, 2], 17)).is_err());
        assert!(
            causal_attention(&q, &rand_t(&[1, 2, 4, 2], 18), &rand_t(&[1, 2, 4, 2], 19)).is_err()
        );
        assert!(
            causal_attention(&q, &rand_t(&[1, 2, 3, 4], 20), &rand_t(&[1, 2, 3, 4], 21)).is_err()
        );
        assert!(
            causal_attention(&q, &rand_t(&[2, 2, 3, 2], 22), &rand_t(&[2, 2, 3, 2], 23)).is_err()
        );
        assert!(causal_attention(&rand_t(&[4, 3, 2], 24), &ok, &ok).is_err());
    }

    fn layer(n_heads: usize, n_kv_heads: usize) -> Attention {
        let mut rng = Rng::seed_from_u64(30);
        Attention::new("attn", 8, n_heads, n_kv_heads, Init::uniform(0.5), &mut rng).unwrap()
    }

    #[test]
    fn the_layer_keeps_the_input_shape() {
        let rope = RotaryTables::new(4, 16, 10_000.0).unwrap();
        let out = layer(2, 1).forward(&rand_t(&[3, 5, 8], 31), &rope).unwrap();
        assert_eq!(out.shape(), &[3, 5, 8]);
        assert!(out.is_finite());
    }

    #[test]
    fn the_layer_is_causal() {
        let rope = RotaryTables::new(4, 16, 10_000.0).unwrap();
        let attn = layer(2, 2);
        let x = rand_t(&[1, 6, 8], 32);
        let base = attn.forward(&x, &rope).unwrap();
        let mut data = x.to_vec();
        for v in &mut data[5 * 8..] {
            *v += 3.0;
        }
        let changed = attn.forward(&Tensor::from_vec(data, &[1, 6, 8]).unwrap(), &rope).unwrap();
        for (a, b) in base.data()[..5 * 8].iter().zip(&changed.data()[..5 * 8]) {
            assert!((a - b).abs() < 1e-6);
        }
        assert_ne!(base.data()[5 * 8..], changed.data()[5 * 8..]);
    }

    #[test]
    fn weights_are_named_and_sized_for_grouped_heads() {
        let attn = layer(4, 2);
        let params = attn.parameters();
        let names: Vec<&str> = params.iter().map(|p| p.name()).collect();
        assert_eq!(
            names,
            [
                "attn.q_proj.weight",
                "attn.k_proj.weight",
                "attn.v_proj.weight",
                "attn.o_proj.weight"
            ]
        );
        assert_eq!(params[0].shape(), &[8, 8]);
        assert_eq!(params[1].shape(), &[8, 4]);
        assert_eq!(params[2].shape(), &[8, 4]);
        assert_eq!(params[3].shape(), &[8, 8]);
        assert_eq!(attn.head_dim(), 2);
    }

    #[test]
    fn mutable_and_shared_parameters_come_in_the_same_order() {
        let mut attn = layer(2, 1);
        let shared: Vec<String> = attn.parameters().iter().map(|p| p.name().to_string()).collect();
        let mutable: Vec<String> =
            attn.parameters_mut().iter().map(|p| p.name().to_string()).collect();
        assert_eq!(shared, mutable);
    }

    #[test]
    fn bad_arguments_are_errors() {
        let mut rng = Rng::seed_from_u64(33);
        let init = Init::uniform(0.1);
        assert!(Attention::new("a", 8, 3, 1, init, &mut rng).is_err());
        assert!(Attention::new("a", 8, 4, 3, init, &mut rng).is_err());
        assert!(Attention::new("a", 8, 0, 1, init, &mut rng).is_err());
        assert!(Attention::new("a", 8, 4, 0, init, &mut rng).is_err());
        let wrong_rope = RotaryTables::new(2, 16, 10_000.0).unwrap();
        assert!(layer(2, 1).forward(&rand_t(&[1, 3, 8], 34), &wrong_rope).is_err());
        let rope = RotaryTables::new(4, 16, 10_000.0).unwrap();
        assert!(layer(2, 1).forward(&rand_t(&[3, 8], 35), &rope).is_err());
    }
}
