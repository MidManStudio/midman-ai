// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-data.md, section "hash.rs"
// ============================================================================
//! SHA-256, written from the specification (FIPS 180-4).

const K: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];

const INITIAL_STATE: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

const BLOCK_BYTES: usize = 64;

/// Incremental SHA-256.
///
/// Feed bytes with [`Sha256::update`] in any number of pieces and finish with
/// [`Sha256::finalize`]. The result does not depend on how the input was split.
///
/// The message length is counted in a `u64` of bytes and taken modulo 2^64 bits, as the
/// standard says; inputs near 2 exabytes are not a concern.
#[derive(Clone)]
pub struct Sha256 {
    state: [u32; 8],
    block: [u8; BLOCK_BYTES],
    filled: usize,
    total_bytes: u64,
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256 {
    /// Starts a new hash.
    pub fn new() -> Self {
        Sha256 { state: INITIAL_STATE, block: [0; BLOCK_BYTES], filled: 0, total_bytes: 0 }
    }

    /// Adds `data` to the message.
    pub fn update(&mut self, mut data: &[u8]) {
        self.total_bytes = self.total_bytes.wrapping_add(data.len() as u64);
        if self.filled > 0 {
            let take = (BLOCK_BYTES - self.filled).min(data.len());
            self.block[self.filled..self.filled + take].copy_from_slice(&data[..take]);
            self.filled += take;
            data = &data[take..];
            if self.filled < BLOCK_BYTES {
                return;
            }
            compress(&mut self.state, &self.block);
            self.filled = 0;
        }
        while data.len() >= BLOCK_BYTES {
            compress(&mut self.state, &data[..BLOCK_BYTES]);
            data = &data[BLOCK_BYTES..];
        }
        if !data.is_empty() {
            self.block[..data.len()].copy_from_slice(data);
            self.filled = data.len();
        }
    }

    /// Ends the message and returns the 32-byte digest.
    pub fn finalize(mut self) -> [u8; 32] {
        let bit_length = self.total_bytes.wrapping_mul(8);
        // One 0x80 byte, then zeros until 8 bytes are left in the block, then the length.
        let zeros = if self.filled < 56 { 55 - self.filled } else { 119 - self.filled };
        let mut padding = [0u8; 128];
        padding[0] = 0x80;
        self.update(&padding[..1 + zeros]);
        self.update(&bit_length.to_be_bytes());
        debug_assert_eq!(self.filled, 0);
        let mut digest = [0u8; 32];
        for (out, word) in digest.chunks_mut(4).zip(self.state) {
            out.copy_from_slice(&word.to_be_bytes());
        }
        digest
    }
}

/// The SHA-256 digest of `data`.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize()
}

/// Lowercase hexadecimal text for a byte string.
pub fn to_hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut text = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        text.push(char::from(DIGITS[usize::from(byte >> 4)]));
        text.push(char::from(DIGITS[usize::from(byte & 15)]));
    }
    text
}

/// One application of the compression function to a 64-byte block.
fn compress(state: &mut [u32; 8], block: &[u8]) {
    debug_assert_eq!(block.len(), BLOCK_BYTES);
    let mut w = [0u32; 64];
    for (word, bytes) in w.iter_mut().zip(block.chunks(4)) {
        *word = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for (&k, &word) in K.iter().zip(w.iter()) {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let choose = (e & f) ^ (!e & g);
        let t1 = h.wrapping_add(s1).wrapping_add(choose).wrapping_add(k).wrapping_add(word);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let majority = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(majority);
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }
    for (slot, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
        *slot = slot.wrapping_add(value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex_of(data: &[u8]) -> String {
        to_hex(&sha256(data))
    }

    #[test]
    fn nist_short_messages() {
        assert_eq!(hex_of(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(
            hex_of(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            hex_of(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        assert_eq!(
            hex_of(
                b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"
            ),
            "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1"
        );
    }

    #[test]
    fn one_million_a() {
        let mut hasher = Sha256::new();
        let chunk = [b'a'; 1000];
        for _ in 0..1000 {
            hasher.update(&chunk);
        }
        assert_eq!(
            to_hex(&hasher.finalize()),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn padding_boundaries_agree_with_one_shot_in_every_split() {
        // Lengths around the points where the padding needs one or two blocks.
        for len in [0usize, 1, 54, 55, 56, 57, 63, 64, 65, 119, 120, 127, 128, 129, 200] {
            let data: Vec<u8> = (0..len).map(|i| (i * 31 + 7) as u8).collect();
            let whole = sha256(&data);
            for split in 0..=len {
                let mut hasher = Sha256::new();
                hasher.update(&data[..split]);
                hasher.update(&data[split..]);
                assert_eq!(hasher.finalize(), whole, "length {len}, split at {split}");
            }
        }
    }

    #[test]
    fn many_small_updates_agree_with_one_shot() {
        let data: Vec<u8> = (0..1000).map(|i| (i * 13 + 5) as u8).collect();
        let mut hasher = Sha256::new();
        for piece in data.chunks(7) {
            hasher.update(piece);
        }
        assert_eq!(hasher.finalize(), sha256(&data));
    }

    #[test]
    fn empty_updates_change_nothing() {
        let mut hasher = Sha256::new();
        hasher.update(b"");
        hasher.update(b"abc");
        hasher.update(b"");
        assert_eq!(hasher.finalize(), sha256(b"abc"));
    }

    #[test]
    fn a_clone_continues_independently() {
        let mut first = Sha256::new();
        first.update(b"ab");
        let mut second = first.clone();
        first.update(b"c");
        second.update(b"d");
        assert_eq!(first.finalize(), sha256(b"abc"));
        assert_eq!(second.finalize(), sha256(b"abd"));
    }

    #[test]
    fn hex_is_lowercase_and_two_digits_per_byte() {
        assert_eq!(to_hex(&[]), "");
        assert_eq!(to_hex(&[0x00, 0x0f, 0xa0, 0xff]), "000fa0ff");
    }
}
