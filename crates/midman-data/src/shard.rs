// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-data.md, section "shard.rs"
// ============================================================================
//! The token shard: a file of token ids ready for training.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use midman_foundation::{Error, Result};

use crate::hash::sha256;

/// The version of the shard file format this build reads and writes.
pub const SHARD_VERSION: u32 = 1;

const WHAT: &str = "token shard";
const MAGIC: [u8; 8] = *b"MMTOKENS";

/// Magic, version, token width, vocabulary size, end-of-text id (four bytes each) and the
/// token count (eight bytes).
const HEADER_BYTES: usize = 8 + 4 + 4 + 4 + 4 + 8;
const CHECKSUM_BYTES: usize = 32;

/// The number of bytes per token the format uses for a vocabulary: 2 when every id fits in
/// 16 bits, otherwise 4. There is one correct width for each vocabulary size, and the reader
/// insists on it, so equal shards are equal files.
fn width_for(vocab_size: u32) -> usize {
    if vocab_size <= 65_536 {
        2
    } else {
        4
    }
}

/// A run of token ids: documents one after another, each followed by the end-of-text token.
///
/// A shard is the unit of storage. The file is a 32-byte header, the ids as little-endian
/// integers of 2 or 4 bytes, and a SHA-256 of everything before it; the layout is in
/// `docs/midman-data.md`. The reader checks the header, the exact length, the checksum and
/// that every id is below the vocabulary size.
///
/// ```
/// use midman_data::TokenShard;
///
/// let shard = TokenShard::new(300, 256, vec![5, 6, 7, 256, 9, 256]).unwrap();
/// let bytes = shard.to_bytes();
/// assert_eq!(TokenShard::from_bytes(&bytes).unwrap(), shard);
/// assert_eq!(bytes.len(), 32 + 6 * 2 + 32);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenShard {
    vocab_size: u32,
    eos: u32,
    tokens: Vec<u32>,
}

impl TokenShard {
    /// A shard of `tokens` for a vocabulary of `vocab_size` ids whose end-of-text token is
    /// `eos`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] if the vocabulary is empty, if `eos` is not below
    /// `vocab_size`, or if some token is not (the message gives its position).
    pub fn new(vocab_size: u32, eos: u32, tokens: Vec<u32>) -> Result<TokenShard> {
        if vocab_size == 0 {
            return Err(Error::invalid("the vocabulary size is 0"));
        }
        if eos >= vocab_size {
            return Err(Error::invalid(format!(
                "the end-of-text id {eos} is not below the vocabulary size {vocab_size}"
            )));
        }
        if let Some(at) = tokens.iter().position(|&t| t >= vocab_size) {
            return Err(Error::invalid(format!(
                "token {} at position {at} is not below the vocabulary size {vocab_size}",
                tokens[at]
            )));
        }
        Ok(TokenShard { vocab_size, eos, tokens })
    }

    /// The number of ids in the vocabulary the tokens come from.
    pub fn vocab_size(&self) -> u32 {
        self.vocab_size
    }

    /// The id of the end-of-text token.
    pub fn eos(&self) -> u32 {
        self.eos
    }

    /// The token ids.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// The number of tokens.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether the shard holds no tokens.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// The bytes per token in the file: 2 for a vocabulary of up to 65,536 ids, else 4.
    pub fn token_width(&self) -> usize {
        width_for(self.vocab_size)
    }

    /// The file contents.
    pub fn to_bytes(&self) -> Vec<u8> {
        let width = self.token_width();
        let mut out = Vec::with_capacity(HEADER_BYTES + self.tokens.len() * width + CHECKSUM_BYTES);
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&SHARD_VERSION.to_le_bytes());
        out.extend_from_slice(&(width as u32).to_le_bytes());
        out.extend_from_slice(&self.vocab_size.to_le_bytes());
        out.extend_from_slice(&self.eos.to_le_bytes());
        out.extend_from_slice(&(self.tokens.len() as u64).to_le_bytes());
        for &token in &self.tokens {
            if width == 2 {
                // The vocabulary has at most 65,536 ids, so every id fits in 16 bits.
                out.extend_from_slice(&(token as u16).to_le_bytes());
            } else {
                out.extend_from_slice(&token.to_le_bytes());
            }
        }
        let checksum = sha256(&out);
        out.extend_from_slice(&checksum);
        out
    }

    /// Reads a shard from its file contents.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Parse`] for a wrong magic, a width that is not the one for the
    /// vocabulary, an end-of-text id outside the vocabulary, a length that does not match the
    /// token count, a checksum that does not match, or a token outside the vocabulary; and
    /// [`Error::UnsupportedVersion`] for a version other than 1.
    pub fn from_bytes(bytes: &[u8]) -> Result<TokenShard> {
        let fail = |message: String| Err(Error::parse(WHAT, 0, message));
        if bytes.len() < HEADER_BYTES + CHECKSUM_BYTES {
            return fail(format!(
                "{} bytes is shorter than the smallest shard, {}",
                bytes.len(),
                HEADER_BYTES + CHECKSUM_BYTES
            ));
        }
        if bytes[..8] != MAGIC {
            return fail("the file does not start with `MMTOKENS`".to_string());
        }
        let word = |at: usize| {
            u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
        };
        let version = word(8);
        if version != SHARD_VERSION {
            return Err(Error::UnsupportedVersion {
                what: WHAT.to_string(),
                found: version,
                supported: SHARD_VERSION,
            });
        }
        let width = word(12) as usize;
        let vocab_size = word(16);
        let eos = word(20);
        let count = u64::from_le_bytes([
            bytes[24], bytes[25], bytes[26], bytes[27], bytes[28], bytes[29], bytes[30], bytes[31],
        ]);
        if vocab_size == 0 {
            return fail("the vocabulary size is 0".to_string());
        }
        if width != width_for(vocab_size) {
            return fail(format!(
                "token width {width} is not the one for a vocabulary of {vocab_size} ids ({})",
                width_for(vocab_size)
            ));
        }
        if eos >= vocab_size {
            return fail(format!(
                "the end-of-text id {eos} is not below the vocabulary size {vocab_size}"
            ));
        }
        let expected = usize::try_from(count)
            .ok()
            .and_then(|n| n.checked_mul(width))
            .and_then(|n| n.checked_add(HEADER_BYTES + CHECKSUM_BYTES));
        if expected != Some(bytes.len()) {
            return fail(format!(
                "{count} tokens of {width} bytes do not make a file of {} bytes",
                bytes.len()
            ));
        }
        let body = bytes.len() - CHECKSUM_BYTES;
        if sha256(&bytes[..body]) != bytes[body..] {
            return fail("the checksum does not match: the file is damaged".to_string());
        }
        let tokens: Vec<u32> = bytes[HEADER_BYTES..body]
            .chunks(width)
            .map(|chunk| {
                if width == 2 {
                    u32::from(u16::from_le_bytes([chunk[0], chunk[1]]))
                } else {
                    u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                }
            })
            .collect();
        if let Some(at) = tokens.iter().position(|&t| t >= vocab_size) {
            return fail(format!(
                "token {} at position {at} is not below the vocabulary size {vocab_size}",
                tokens[at]
            ));
        }
        Ok(TokenShard { vocab_size, eos, tokens })
    }

    /// Writes the shard to `path` without ever leaving a half-written file there: the bytes
    /// go to `<path>.tmp`, are flushed to disk, and are renamed over `path`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] with the path if the file cannot be written; the temporary file
    /// is removed.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let mut tmp_name = path.as_os_str().to_owned();
        tmp_name.push(".tmp");
        let tmp = PathBuf::from(tmp_name);
        let result = self.write_then_rename(&tmp, path);
        if result.is_err() {
            let _ = fs::remove_file(&tmp);
        }
        result.map_err(|err| Error::Io(format!("{}: {err}", path.display())))
    }

    fn write_then_rename(&self, tmp: &Path, path: &Path) -> std::io::Result<()> {
        let mut file = fs::File::create(tmp)?;
        file.write_all(&self.to_bytes())?;
        file.flush()?;
        file.sync_all()?;
        drop(file);
        fs::rename(tmp, path)
    }

    /// Reads a shard from a file.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] with the path if the file cannot be read, or the error of
    /// [`TokenShard::from_bytes`].
    pub fn load(path: impl AsRef<Path>) -> Result<TokenShard> {
        let path = path.as_ref();
        let bytes =
            fs::read(path).map_err(|err| Error::Io(format!("{}: {err}", path.display())))?;
        TokenShard::from_bytes(&bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::to_hex;

    fn shard(vocab: u32, tokens: &[u32]) -> TokenShard {
        TokenShard::new(vocab, 0, tokens.to_vec()).unwrap()
    }

    /// File contents with every header field and the payload given, and a correct checksum.
    fn file(version: u32, width: u32, vocab: u32, eos: u32, count: u64, payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&MAGIC);
        for word in [version, width, vocab, eos] {
            out.extend_from_slice(&word.to_le_bytes());
        }
        out.extend_from_slice(&count.to_le_bytes());
        out.extend_from_slice(payload);
        let checksum = sha256(&out);
        out.extend_from_slice(&checksum);
        out
    }

    fn message(bytes: &[u8]) -> String {
        TokenShard::from_bytes(bytes).unwrap_err().to_string()
    }

    #[test]
    fn the_file_has_exactly_this_layout() {
        // vocabulary 300, end-of-text 256, tokens 1 2 3 256: two bytes each, little endian.
        let shard = TokenShard::new(300, 256, vec![1, 2, 3, 256]).unwrap();
        let bytes = shard.to_bytes();
        let body = &bytes[..bytes.len() - 32];
        assert_eq!(
            to_hex(body),
            "4d4d544f4b454e53\
             01000000\
             02000000\
             2c010000\
             00010000\
             0400000000000000\
             0100020003000001"
        );
        assert_eq!(&bytes[bytes.len() - 32..], &sha256(body));
        assert_eq!(bytes.len(), 32 + 8 + 32);
    }

    #[test]
    fn four_byte_ids_are_used_above_65536() {
        let bytes = TokenShard::new(65_537, 0, vec![65_536, 7]).unwrap().to_bytes();
        assert_eq!(&bytes[12..16], &4u32.to_le_bytes());
        assert_eq!(bytes.len(), 32 + 2 * 4 + 32);
        assert_eq!(&bytes[32..36], &65_536u32.to_le_bytes());
        let bytes = TokenShard::new(65_536, 0, vec![65_535, 7]).unwrap().to_bytes();
        assert_eq!(&bytes[12..16], &2u32.to_le_bytes());
        assert_eq!(bytes.len(), 32 + 2 * 2 + 32);
        assert_eq!(&bytes[32..34], &65_535u16.to_le_bytes());
    }

    #[test]
    fn round_trips_in_both_widths_and_empty() {
        for (vocab, tokens) in [
            (256u32, vec![0u32, 255, 1, 2, 3]),
            (65_536, vec![0, 65_535, 40_000]),
            (65_537, vec![0, 65_536, 40_000]),
            (4_000_000_000, vec![0, 3_999_999_999, 1_234_567_890]),
            (300, vec![]),
        ] {
            let original = shard(vocab, &tokens);
            let again = TokenShard::from_bytes(&original.to_bytes()).unwrap();
            assert_eq!(again, original, "vocabulary {vocab}");
            assert_eq!(again.tokens(), tokens.as_slice());
            assert_eq!((again.vocab_size(), again.eos(), again.len()), (vocab, 0, tokens.len()));
            assert_eq!(again.is_empty(), tokens.is_empty());
        }
    }

    #[test]
    fn construction_checks_the_vocabulary() {
        assert!(matches!(TokenShard::new(0, 0, vec![]), Err(Error::InvalidArgument(_))));
        assert!(matches!(TokenShard::new(10, 10, vec![]), Err(Error::InvalidArgument(_))));
        let err = TokenShard::new(10, 3, vec![1, 2, 10, 4]).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
        assert!(err.to_string().contains("token 10 at position 2"), "{err}");
        assert!(TokenShard::new(10, 9, vec![9, 0]).is_ok());
    }

    #[test]
    fn the_reader_refuses_a_wrong_header_each_with_its_reason() {
        let good = file(1, 2, 300, 256, 2, &[1, 0, 2, 0]);
        assert!(TokenShard::from_bytes(&good).is_ok());
        // Magic.
        let mut bad = good.clone();
        bad[0] = b'X';
        assert!(message(&bad).contains("MMTOKENS"));
        // Version.
        let bad = file(2, 2, 300, 256, 2, &[1, 0, 2, 0]);
        assert!(matches!(
            TokenShard::from_bytes(&bad),
            Err(Error::UnsupportedVersion { found: 2, supported: 1, .. })
        ));
        assert!(matches!(
            TokenShard::from_bytes(&file(0, 2, 300, 256, 2, &[1, 0, 2, 0])),
            Err(Error::UnsupportedVersion { found: 0, .. })
        ));
        // A width that is not the one for the vocabulary, or not a width at all.
        for width in [0, 1, 3, 4, 8] {
            let bad = file(1, width, 300, 256, 2, &[1, 0, 2, 0, 0, 0, 0, 0]);
            assert!(message(&bad).contains("token width"), "width {width}: {}", message(&bad));
        }
        assert!(message(&file(1, 2, 65_537, 0, 1, &[1, 0])).contains("token width"));
        // Vocabulary and end-of-text.
        assert!(message(&file(1, 2, 0, 0, 0, &[])).contains("vocabulary size is 0"));
        assert!(message(&file(1, 2, 300, 300, 0, &[])).contains("end-of-text"));
        assert!(TokenShard::from_bytes(&file(1, 2, 300, 299, 0, &[])).is_ok());
    }

    #[test]
    fn the_reader_refuses_a_wrong_length_for_every_cut_and_extension() {
        let good = shard(300, &[1, 2, 3, 4, 5]).to_bytes();
        for cut in 0..good.len() {
            assert!(
                TokenShard::from_bytes(&good[..cut]).is_err(),
                "a file cut to {cut} bytes was accepted"
            );
        }
        let mut longer = good.clone();
        longer.push(0);
        assert!(message(&longer).contains("do not make a file"));
        // A count that promises more than the file holds, and one that cannot fit in memory.
        let lying = file(1, 2, 300, 0, 3, &[1, 0, 2, 0]);
        assert!(message(&lying).contains("3 tokens"));
        for count in [u64::MAX, u64::MAX / 2, 1 << 62] {
            let huge = file(1, 2, 300, 0, count, &[]);
            assert!(message(&huge).contains("do not make a file"), "{count}");
        }
    }

    #[test]
    fn the_reader_refuses_damage_anywhere_in_the_file() {
        let good = shard(300, &[1, 2, 3, 4, 5]).to_bytes();
        // Flip each bit of the body and of the checksum: always an error, never a wrong shard.
        for index in 0..good.len() {
            for bit in [0u8, 7] {
                let mut bad = good.clone();
                bad[index] ^= 1 << bit;
                assert!(
                    TokenShard::from_bytes(&bad).is_err(),
                    "flipping bit {bit} of byte {index} was not noticed"
                );
            }
        }
        let mut bad = good.clone();
        bad[34] ^= 0x10;
        assert!(message(&bad).contains("checksum"), "{}", message(&bad));
    }

    #[test]
    fn a_token_outside_the_vocabulary_is_refused_even_with_a_valid_checksum() {
        // Ids 1, 300 (not below 300), 2.
        let bad = file(1, 2, 300, 0, 3, &[1, 0, 44, 1, 2, 0]);
        let text = message(&bad);
        assert!(text.contains("token 300 at position 1"), "{text}");
    }

    #[test]
    fn save_and_load_round_trip_and_replace_atomically() {
        let dir = std::env::temp_dir().join(format!("midman-data-shard-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.mmtokens");
        let first = shard(300, &[1, 2, 3]);
        first.save(&path).unwrap();
        assert_eq!(TokenShard::load(&path).unwrap(), first);
        let second = shard(300, &[4, 5, 6, 7]);
        second.save(&path).unwrap();
        assert_eq!(TokenShard::load(&path).unwrap(), second);
        assert!(!dir.join("a.mmtokens.tmp").exists());
        // I/O errors name the path.
        let err = first.save(dir.join("nope").join("b.mmtokens")).unwrap_err();
        assert!(matches!(err, Error::Io(_)) && err.to_string().contains("nope"), "{err}");
        let err = TokenShard::load(dir.join("absent.mmtokens")).unwrap_err();
        assert!(
            matches!(err, Error::Io(_)) && err.to_string().contains("absent.mmtokens"),
            "{err}"
        );
        // A damaged file on disk is a parse error, not a panic.
        let mut bytes = fs::read(&path).unwrap();
        bytes[40] ^= 1;
        fs::write(&path, bytes).unwrap();
        assert!(matches!(TokenShard::load(&path), Err(Error::Parse { .. })));
        fs::remove_dir_all(&dir).unwrap();
    }
}
