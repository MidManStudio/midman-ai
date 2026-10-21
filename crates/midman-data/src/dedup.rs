// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-data.md, section "dedup.rs"
// ============================================================================
//! Exact deduplication: a document is a duplicate if its bytes were seen before.

use std::collections::HashSet;
use std::hash::{BuildHasherDefault, Hash, Hasher};

use crate::hash::sha256;

/// A SHA-256 digest used as a set key.
///
/// The digest is already uniformly distributed, so hashing it again would be wasted work:
/// the set uses its first eight bytes as the hash.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Digest([u8; 32]);

impl Hash for Digest {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let d = &self.0;
        state.write_u64(u64::from_le_bytes([d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]]));
    }
}

/// A hasher that passes through the one `u64` it is given.
#[derive(Default)]
struct PassThrough(u64);

impl Hasher for PassThrough {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        // Only reached by a key type other than `Digest`; kept correct, not fast.
        for &byte in bytes {
            self.0 = self.0.rotate_left(8) ^ u64::from(byte);
        }
    }

    fn write_u64(&mut self, value: u64) {
        self.0 = value;
    }
}

/// Remembers the SHA-256 of every document it has accepted.
///
/// ```
/// use midman_data::ExactDedup;
///
/// let mut dedup = ExactDedup::new();
/// assert!(dedup.insert(b"fn main() {}"));
/// assert!(!dedup.insert(b"fn main() {}"));
/// assert_eq!((dedup.unique(), dedup.duplicates()), (1, 1));
/// ```
///
/// Two documents are the same only if every byte is the same; a trailing newline or a
/// different line ending makes a different document. Memory use is 32 bytes per distinct
/// document plus the set's own overhead.
#[derive(Default)]
pub struct ExactDedup {
    seen: HashSet<Digest, BuildHasherDefault<PassThrough>>,
    checked: u64,
    duplicates: u64,
}

impl ExactDedup {
    /// An empty filter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records `content` and returns `true` if it is new, `false` if it was seen before.
    pub fn insert(&mut self, content: &[u8]) -> bool {
        self.insert_digest(sha256(content))
    }

    /// Like [`ExactDedup::insert`] for a SHA-256 digest computed elsewhere.
    pub fn insert_digest(&mut self, digest: [u8; 32]) -> bool {
        self.checked += 1;
        let new = self.seen.insert(Digest(digest));
        if !new {
            self.duplicates += 1;
        }
        new
    }

    /// Whether `content` has been seen, without recording it or counting it.
    pub fn contains(&self, content: &[u8]) -> bool {
        self.seen.contains(&Digest(sha256(content)))
    }

    /// Number of distinct documents seen.
    pub fn unique(&self) -> usize {
        self.seen.len()
    }

    /// Number of documents that were duplicates of an earlier one.
    pub fn duplicates(&self) -> u64 {
        self.duplicates
    }

    /// Number of documents offered to [`ExactDedup::insert`] and [`ExactDedup::insert_digest`].
    pub fn checked(&self) -> u64 {
        self.checked
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_sight_is_new_and_repeats_are_not() {
        let mut dedup = ExactDedup::new();
        assert!(dedup.insert(b"alpha"));
        assert!(dedup.insert(b"beta"));
        assert!(!dedup.insert(b"alpha"));
        assert!(!dedup.insert(b"alpha"));
        assert!(!dedup.insert(b"beta"));
        assert_eq!(dedup.unique(), 2);
        assert_eq!(dedup.duplicates(), 3);
        assert_eq!(dedup.checked(), 5);
    }

    #[test]
    fn the_empty_document_is_a_document() {
        let mut dedup = ExactDedup::new();
        assert!(dedup.insert(b""));
        assert!(!dedup.insert(b""));
    }

    #[test]
    fn one_byte_of_difference_is_a_different_document() {
        let mut dedup = ExactDedup::new();
        assert!(dedup.insert(b"fn main() {}\n"));
        assert!(dedup.insert(b"fn main() {}"));
        assert!(dedup.insert(b"fn main() {}\r\n"));
        assert!(dedup.insert(b"Fn main() {}\n"));
        assert_eq!(dedup.unique(), 4);
        assert_eq!(dedup.duplicates(), 0);
    }

    #[test]
    fn contains_neither_records_nor_counts() {
        let mut dedup = ExactDedup::new();
        assert!(!dedup.contains(b"x"));
        assert_eq!((dedup.unique(), dedup.checked()), (0, 0));
        dedup.insert(b"x");
        assert!(dedup.contains(b"x"));
        assert!(!dedup.contains(b"y"));
        assert_eq!((dedup.unique(), dedup.checked()), (1, 1));
    }

    #[test]
    fn a_digest_and_its_content_are_the_same_key() {
        let mut dedup = ExactDedup::new();
        assert!(dedup.insert_digest(sha256(b"doc")));
        assert!(!dedup.insert(b"doc"));
        assert!(dedup.insert(b"other"));
        assert!(!dedup.insert_digest(sha256(b"other")));
    }

    #[test]
    fn many_documents_with_a_few_repeats_are_counted_exactly() {
        let mut dedup = ExactDedup::new();
        let mut new = 0;
        for i in 0..20_000u32 {
            // Documents 0..10_000 appear twice.
            let document = (i % 10_000).to_le_bytes();
            if dedup.insert(&document) {
                new += 1;
            }
        }
        assert_eq!(new, 10_000);
        assert_eq!(dedup.unique(), 10_000);
        assert_eq!(dedup.duplicates(), 10_000);
        assert_eq!(dedup.checked(), 20_000);
    }

    #[test]
    fn digests_that_share_their_first_eight_bytes_stay_distinct() {
        // The set hashes only the first eight bytes; equality still compares all 32.
        let mut first = [7u8; 32];
        let mut second = [7u8; 32];
        first[31] = 1;
        second[31] = 2;
        let mut dedup = ExactDedup::new();
        assert!(dedup.insert_digest(first));
        assert!(dedup.insert_digest(second));
        assert!(!dedup.insert_digest(first));
        assert_eq!(dedup.unique(), 2);
    }
}
