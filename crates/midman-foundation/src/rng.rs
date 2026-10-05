// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-foundation.md, section "rng.rs"
// ============================================================================
//! A small, seedable random number generator with no dependencies.
//!
//! Training runs must be reproducible, so every source of randomness in MidMan
//! (weight initialization, data shuffling, sampling) draws from this generator.
//! The algorithm is xoshiro256** seeded through SplitMix64.
//!
//! The integer outputs ([`Rng::next_u64`], [`Rng::below`], [`Rng::shuffle`]) are
//! identical on every platform. [`Rng::normal`] uses `ln`, `sin` and `cos` from
//! the platform math library, so its last bits can differ between platforms.

/// One step of SplitMix64, used to expand a 64-bit seed into the generator state.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A seedable pseudo-random number generator (xoshiro256**).
///
/// Not suitable for cryptography.
#[derive(Debug, Clone)]
pub struct Rng {
    state: [u64; 4],
    spare_normal: Option<f32>,
}

impl Rng {
    /// Creates a generator from a 64-bit seed. The same seed always gives the
    /// same sequence.
    pub fn seed_from_u64(seed: u64) -> Self {
        let mut sm = seed;
        let state =
            [splitmix64(&mut sm), splitmix64(&mut sm), splitmix64(&mut sm), splitmix64(&mut sm)];
        Rng { state, spare_normal: None }
    }

    /// Returns the next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.state;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Returns a uniform `f32` in `[0, 1)` with 24 random bits.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / 16_777_216.0)
    }

    /// Returns a uniform `f64` in `[0, 1)` with 53 random bits.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Returns a uniform `f32` in `[lo, hi)`.
    pub fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }

    /// Returns a standard normal sample (mean 0, variance 1) using the
    /// Box-Muller transform. Every second call returns a cached value.
    pub fn normal(&mut self) -> f32 {
        if let Some(z) = self.spare_normal.take() {
            return z;
        }
        let u1 = 1.0 - self.next_f64(); // in (0, 1], so ln is finite
        let u2 = self.next_f64();
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        self.spare_normal = Some((radius * theta.sin()) as f32);
        (radius * theta.cos()) as f32
    }

    /// Returns a uniform integer in `0..n` without modulo bias.
    ///
    /// # Panics
    ///
    /// Panics if `n` is 0.
    pub fn below(&mut self, n: usize) -> usize {
        assert!(n > 0, "Rng::below requires n > 0");
        let n = n as u64;
        // Lemire's nearly-divisionless method.
        let mut product = u128::from(self.next_u64()) * u128::from(n);
        let mut low = product as u64;
        if low < n {
            let threshold = n.wrapping_neg() % n;
            while low < threshold {
                product = u128::from(self.next_u64()) * u128::from(n);
                low = product as u64;
            }
        }
        (product >> 64) as usize
    }

    /// Shuffles a slice in place (Fisher-Yates).
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i + 1);
            items.swap(i, j);
        }
    }

    /// Creates an independent generator for a numbered stream, for example one
    /// per worker or per layer. Advances this generator.
    pub fn fork(&mut self, stream: u64) -> Rng {
        Rng::seed_from_u64(self.next_u64() ^ stream.wrapping_mul(0x9E37_79B9_7F4A_7C15))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_answers_match_an_independent_reference() {
        // Reference values come from a separate Python implementation of
        // SplitMix64 seeding plus xoshiro256**.
        let mut r = Rng::seed_from_u64(0);
        assert_eq!(r.next_u64(), 0x99ec_5f36_cb75_f2b4);
        assert_eq!(r.next_u64(), 0xbf6e_1f78_4956_452a);
        assert_eq!(r.next_u64(), 0x1a5f_849d_4933_e6e0);
        assert_eq!(r.next_u64(), 0x6aa5_94f1_262d_2d2c);

        let mut r = Rng::seed_from_u64(42);
        assert_eq!(r.next_u64(), 0x1578_0b2e_0c2e_c716);
        assert_eq!(r.next_u64(), 0x6104_d986_6d11_3a7e);
        assert_eq!(r.next_u64(), 0xae17_5332_39e4_99a1);
        assert_eq!(r.next_u64(), 0xecb8_ad47_03b3_60a1);
    }

    #[test]
    fn below_matches_the_reference_sequence() {
        let mut r = Rng::seed_from_u64(7);
        let got: Vec<usize> = (0..8).map(|_| r.below(10)).collect();
        assert_eq!(got, vec![7, 2, 8, 9, 9, 8, 0, 1]);
    }

    #[test]
    fn next_f32_matches_the_reference_values() {
        let mut r = Rng::seed_from_u64(5);
        let expected = [0.288_411_2_f32, 0.602_082_3, 0.649_546_7];
        for e in expected {
            assert!((r.next_f32() - e).abs() < 1e-7);
        }
    }

    #[test]
    fn same_seed_same_stream_different_seed_different_stream() {
        let mut a = Rng::seed_from_u64(1234);
        let mut b = Rng::seed_from_u64(1234);
        let mut c = Rng::seed_from_u64(1235);
        let xs: Vec<u64> = (0..16).map(|_| a.next_u64()).collect();
        let ys: Vec<u64> = (0..16).map(|_| b.next_u64()).collect();
        let zs: Vec<u64> = (0..16).map(|_| c.next_u64()).collect();
        assert_eq!(xs, ys);
        assert_ne!(xs, zs);
    }

    #[test]
    fn floats_stay_in_range() {
        let mut r = Rng::seed_from_u64(9);
        for _ in 0..10_000 {
            let x = r.next_f32();
            assert!((0.0..1.0).contains(&x));
            let y = r.next_f64();
            assert!((0.0..1.0).contains(&y));
            let u = r.uniform(-2.0, 3.0);
            assert!((-2.0..3.0).contains(&u));
        }
    }

    #[test]
    fn normal_has_mean_zero_and_unit_variance() {
        let mut r = Rng::seed_from_u64(2024);
        let n = 200_000;
        let samples: Vec<f64> = (0..n).map(|_| f64::from(r.normal())).collect();
        let mean = samples.iter().sum::<f64>() / n as f64;
        let var = samples.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.01, "mean {mean}");
        assert!((var - 1.0).abs() < 0.02, "variance {var}");
    }

    #[test]
    fn below_is_roughly_uniform_and_in_range() {
        let mut r = Rng::seed_from_u64(77);
        let mut counts = [0usize; 6];
        let n = 60_000;
        for _ in 0..n {
            counts[r.below(6)] += 1;
        }
        for (face, &c) in counts.iter().enumerate() {
            let expected = n / 6;
            assert!(c.abs_diff(expected) < expected / 20, "face {face} got {c}");
        }
        assert_eq!(r.below(1), 0);
    }

    #[test]
    #[should_panic(expected = "n > 0")]
    fn below_zero_panics() {
        Rng::seed_from_u64(0).below(0);
    }

    #[test]
    fn shuffle_is_a_permutation_and_depends_on_the_seed() {
        let original: Vec<u32> = (0..50).collect();
        let mut a = original.clone();
        let mut b = original.clone();
        Rng::seed_from_u64(3).shuffle(&mut a);
        Rng::seed_from_u64(4).shuffle(&mut b);
        let mut sorted = a.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, original);
        assert_ne!(a, original);
        assert_ne!(a, b);
    }

    #[test]
    fn forks_are_independent_of_each_other_and_the_parent() {
        let mut parent = Rng::seed_from_u64(11);
        let mut f0 = parent.fork(0);
        let mut f1 = parent.fork(1);
        let a: Vec<u64> = (0..8).map(|_| f0.next_u64()).collect();
        let b: Vec<u64> = (0..8).map(|_| f1.next_u64()).collect();
        let c: Vec<u64> = (0..8).map(|_| parent.next_u64()).collect();
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_ne!(b, c);
    }
}
