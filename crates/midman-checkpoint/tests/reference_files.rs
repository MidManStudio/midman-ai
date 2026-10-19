// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-checkpoint.md, section "tests/reference_files.rs"
// ============================================================================
//! Tests against files written by the reference `safetensors` package.
//!
//! The fixtures in `tests/fixtures/` are hex dumps of real files, made by
//! `generate_fixtures.py`. They show that this crate reads what the reference
//! implementation writes and writes the very same bytes.

use midman_checkpoint::TensorFile;

fn decode(hex: &str) -> Vec<u8> {
    let digits: Vec<u8> = hex.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    assert_eq!(digits.len() % 2, 0, "odd number of hex digits");
    digits
        .chunks(2)
        .map(|pair| {
            let text = std::str::from_utf8(pair).unwrap();
            u8::from_str_radix(text, 16).unwrap()
        })
        .collect()
}

macro_rules! fixture {
    ($name:literal) => {
        decode(include_str!(concat!("fixtures/", $name, ".hex")))
    };
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// The values `generate_fixtures.py` puts in `basic.hex`, in file order.
fn basic_expected() -> Vec<(&'static str, Vec<usize>, Vec<f32>)> {
    vec![
        ("b", vec![3], vec![0.1, 0.2, 0.3]),
        ("empty", vec![0], vec![]),
        ("scalar", vec![], vec![7.0]),
        (
            "special",
            vec![6],
            vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0, f32::MIN_POSITIVE, 1e-45],
        ),
        ("w", vec![2, 3], vec![1.5, -2.0, 0.0, 3.25, 1e-30, 1e30]),
    ]
}

fn expected_file(tensors: &[(&str, Vec<usize>, Vec<f32>)]) -> TensorFile {
    let mut file = TensorFile::new();
    for (name, shape, data) in tensors {
        file.push(*name, shape, data).unwrap();
    }
    file
}

#[test]
fn reads_the_reference_file_with_every_dtype_shape_and_special_value() {
    let file = TensorFile::from_bytes(&fixture!("basic")).unwrap();
    assert_eq!(file.metadata().get("format").map(String::as_str), Some("pt"));
    let expected = basic_expected();
    assert_eq!(file.tensors().len(), expected.len());
    for (tensor, (name, shape, data)) in file.tensors().iter().zip(&expected) {
        assert_eq!(tensor.name(), *name);
        assert_eq!(tensor.shape(), shape.as_slice(), "{name}");
        assert_eq!(bits(tensor.data()), bits(data), "{name}");
    }
}

#[test]
fn writes_exactly_the_bytes_the_reference_writes() {
    let mut file = expected_file(&basic_expected());
    file.set_metadata("format", "pt");
    assert_eq!(file.to_bytes().unwrap(), fixture!("basic"));
}

#[test]
fn names_and_metadata_with_quotes_tabs_newlines_and_unicode_match_byte_for_byte() {
    let bytes = fixture!("unicode_names");
    let file = TensorFile::from_bytes(&bytes).unwrap();
    assert_eq!(file.get("layer.0.é日本").unwrap().data(), [1.0, 2.0]);
    assert_eq!(file.get("quote\"back\\slash\ttab").unwrap().data(), [3.0]);
    assert_eq!(file.metadata().get("note").map(String::as_str), Some("café \"quoted\" 日本\nline"));

    let mut again = TensorFile::new();
    again.push("layer.0.é日本", &[2], &[1.0, 2.0]).unwrap();
    again.push("quote\"back\\slash\ttab", &[1], &[3.0]).unwrap();
    again.set_metadata("note", "café \"quoted\" 日本\nline");
    assert_eq!(again.to_bytes().unwrap(), bytes);
}

#[test]
fn the_empty_file_matches_the_reference_both_ways() {
    let bytes = fixture!("empty");
    let file = TensorFile::from_bytes(&bytes).unwrap();
    assert!(file.tensors().is_empty() && file.metadata().is_empty());
    assert_eq!(TensorFile::new().to_bytes().unwrap(), bytes);
}

#[test]
fn several_metadata_entries_read_back_whatever_order_the_reference_wrote_them_in() {
    // The reference writes the entries in an arbitrary order, so only the content is
    // compared here. This crate always writes them sorted by key.
    let file = TensorFile::from_bytes(&fixture!("multi_metadata")).unwrap();
    let pairs: Vec<(&str, &str)> =
        file.metadata().iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    assert_eq!(pairs, [("alpha", "1"), ("beta", "2"), ("gamma", "3")]);
    assert_eq!(file.get("x").unwrap().data(), [1.0]);

    let rewritten = file.to_bytes().unwrap();
    let text = String::from_utf8_lossy(&rewritten);
    let alpha = text.find("\"alpha\"").unwrap();
    let beta = text.find("\"beta\"").unwrap();
    let gamma = text.find("\"gamma\"").unwrap();
    assert!(alpha < beta && beta < gamma);
    let back = TensorFile::from_bytes(&rewritten).unwrap();
    assert_eq!(back.metadata(), file.metadata());
}

#[test]
fn a_reference_file_with_another_dtype_is_rejected_by_name() {
    let message = TensorFile::from_bytes(&fixture!("mixed_dtypes")).unwrap_err().to_string();
    assert!(message.contains("`h`") && message.contains("F16"), "{message}");
}

#[test]
fn no_prefix_of_a_file_is_accepted_and_none_panics() {
    let bytes = fixture!("basic");
    for len in 0..bytes.len() {
        assert!(TensorFile::from_bytes(&bytes[..len]).is_err(), "prefix of {len} bytes");
    }
    assert!(TensorFile::from_bytes(&bytes).is_ok());
}

#[test]
fn flipping_any_single_bit_never_panics() {
    // A flipped bit may leave a valid file (a changed value) or break it; either is
    // fine. What matters is that reading never panics or allocates wildly.
    for name_bytes in [fixture!("basic"), fixture!("unicode_names")] {
        for position in 0..name_bytes.len() {
            for bit in 0..8 {
                let mut corrupted = name_bytes.clone();
                corrupted[position] ^= 1 << bit;
                let _ = TensorFile::from_bytes(&corrupted);
            }
        }
    }
}

#[test]
fn trailing_bytes_after_the_data_are_rejected() {
    let mut bytes = fixture!("basic");
    bytes.push(0);
    assert!(TensorFile::from_bytes(&bytes).is_err());
}
