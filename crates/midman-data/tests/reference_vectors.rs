// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-data.md, section "tests/reference_vectors.rs"
// ============================================================================
//! SHA-256 against digests made by Python's `hashlib`.
//!
//! `fixtures/sha256_vectors.txt` holds `length digest` lines. The messages are rebuilt here
//! from the formula in `fixtures/generate_fixtures.py`: byte `j` of the message of length
//! `n` is `(31 * j + 7 * n + 5) mod 256`.

use midman_data::{sha256, to_hex, Sha256};

fn message(length: usize) -> Vec<u8> {
    (0..length).map(|j| ((31 * j + 7 * length + 5) % 256) as u8).collect()
}

fn vectors() -> Vec<(usize, &'static str)> {
    include_str!("fixtures/sha256_vectors.txt")
        .lines()
        .map(|line| {
            let (length, digest) = line.split_once(' ').expect("`length digest`");
            (length.parse().expect("a length"), digest)
        })
        .collect()
}

#[test]
fn the_fixture_covers_every_small_length_and_some_large_ones() {
    let lengths: Vec<usize> = vectors().iter().map(|v| v.0).collect();
    assert!((0..=300).all(|n| lengths.contains(&n)));
    assert!(lengths.contains(&100_003));
    assert_eq!(lengths.len(), 309);
}

#[test]
fn every_vector_matches_in_one_shot() {
    for (length, digest) in vectors() {
        assert_eq!(to_hex(&sha256(&message(length))), digest, "length {length}");
    }
}

#[test]
fn every_vector_matches_when_fed_in_uneven_pieces() {
    for (length, digest) in vectors() {
        let data = message(length);
        for piece in [1usize, 3, 63, 64, 65, 1000] {
            let mut hasher = Sha256::new();
            for part in data.chunks(piece) {
                hasher.update(part);
            }
            assert_eq!(to_hex(&hasher.finalize()), digest, "length {length}, pieces of {piece}");
        }
    }
}
