// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-api.md, section "tests/c_abi.rs"
// ============================================================================
//! Tests of the C ABI, written the way a C caller uses it: through the exported
//! functions, with raw pointers, reading results back through the exported
//! buffers. The same functions are what the browser build exposes.

// Calling the exported functions needs `unsafe`; it lives in the small helpers
// below, each with its safety argument.
#![allow(unsafe_code)]

use midman_api::ffi::*;

const TINY: [f64; 10] = [12.0, 16.0, 2.0, 2.0, 1.0, 32.0, 16.0, 10_000.0, 1e-5, 1.0];
const HYPER: [f64; 8] = [0.03, 0.003, 15.0, 150.0, 0.01, 1.0, 0.9, 0.99];

fn text() -> String {
    // SAFETY: the pointer and length come from the library's own text buffer, which
    // stays valid until the next library call, and the bytes are copied out here.
    let bytes =
        unsafe { std::slice::from_raw_parts(midman_text_ptr(), midman_text_len() as usize) };
    String::from_utf8(bytes.to_vec()).expect("the text result is UTF-8")
}

fn error() -> String {
    // SAFETY: as in `text`, for the error buffer.
    let bytes =
        unsafe { std::slice::from_raw_parts(midman_error_ptr(), midman_error_len() as usize) };
    String::from_utf8(bytes.to_vec()).expect("the error message is UTF-8")
}

fn floats() -> Vec<f32> {
    // SAFETY: as in `text`, for the numeric buffer, which is 4-byte aligned.
    unsafe { std::slice::from_raw_parts(midman_floats_ptr(), midman_floats_len() as usize) }
        .to_vec()
}

fn config_report(config: &[f64; 10]) -> i32 {
    // SAFETY: the array has the 10 aligned f64 values the function reads.
    unsafe { midman_config_report(config.as_ptr()) }
}

fn create(config: &[f64; 10], hyper: &[f64; 8], seed: u32, batch: u32, seq: u32) -> u32 {
    // SAFETY: the arrays have the 10 and 8 aligned f64 values the function reads.
    unsafe { midman_lab_create(config.as_ptr(), hyper.as_ptr(), seed, 0, batch, seq) }
}

fn tiny_lab() -> u32 {
    let handle = create(&TINY, &HYPER, 2024, 2, 12);
    assert_ne!(handle, 0, "creating the lab failed: {}", error());
    handle
}

#[test]
fn the_abi_version_is_one() {
    assert_eq!(midman_abi_version(), 1);
}

#[test]
fn allocated_buffers_are_aligned_writable_and_freeable() {
    for len in [0u32, 1, 7, 8, 4096] {
        let ptr = midman_alloc(len);
        assert!(!ptr.is_null());
        assert_eq!(ptr as usize % 8, 0, "len {len}");
        // SAFETY: the buffer has at least `max(len, 1)` bytes, freed once with its length.
        unsafe {
            ptr.write(0xAB);
            assert_eq!(ptr.read(), 0xAB);
            midman_free(ptr, len);
        }
    }
    // SAFETY: freeing null is documented to do nothing.
    unsafe { midman_free(std::ptr::null_mut(), 16) };
}

#[test]
fn describe_lists_the_abi_the_field_orders_and_every_preset() {
    assert_eq!(midman_describe(), STATUS_OK);
    let json = text();
    assert!(json.starts_with("{\"abi\":1,"), "{json}");
    assert!(json.contains("\"config_fields\":[\"vocab_size\",\"d_model\""), "{json}");
    assert!(json.contains("\"hyper_fields\":[\"max_lr\""), "{json}");
    assert!(json.contains("\"max_params\":8000000"), "{json}");
    for name in ["smoke", "tiny", "small", "base", "medium", "large"] {
        assert!(json.contains(&format!("\"name\":\"{name}\"")), "{name} missing: {json}");
    }
    assert!(json.contains("\"params\":117056"), "the smoke preset's exact count: {json}");
}

#[test]
fn a_preset_config_round_trips_through_the_report() {
    let mut values = [0.0f64; 10];
    // SAFETY: `values` has room for the 10 aligned f64 values written.
    assert_eq!(unsafe { midman_preset_config(0, values.as_mut_ptr()) }, STATUS_OK);
    assert_eq!(values[1], 64.0, "smoke has width 64");
    assert_eq!(config_report(&values), STATUS_OK);
    let json = text();
    assert!(json.contains("\"valid\":true"), "{json}");
    assert!(json.contains("\"total\":117056"), "{json}");
    assert!(json.contains("\"head_dim\":16"), "{json}");
}

#[test]
fn an_invalid_config_is_an_answer_not_a_failure() {
    let mut bad = TINY;
    bad[3] = 3.0; // 3 heads do not divide the width of 16
    assert_eq!(config_report(&bad), STATUS_OK);
    let json = text();
    assert!(json.contains("\"valid\":false"), "{json}");
    assert!(json.contains("\"error\":\""), "{json}");
}

#[test]
fn malformed_arguments_are_rejected_with_a_message() {
    // SAFETY: null is checked for and rejected before anything is read.
    assert_eq!(unsafe { midman_config_report(std::ptr::null()) }, ERR_ARGUMENT);
    assert!(error().contains("null"), "{}", error());
    for (index, value) in [(0usize, 12.5f64), (1, -1.0), (2, f64::NAN), (3, 5e9)] {
        let mut bad = TINY;
        bad[index] = value;
        assert_eq!(config_report(&bad), ERR_ARGUMENT, "field {index} = {value}");
        assert!(error().contains(["vocab_size", "d_model", "n_layers", "n_heads"][index]));
    }
    let mut bad = TINY;
    bad[7] = f64::INFINITY;
    assert_eq!(config_report(&bad), ERR_ARGUMENT);
    // SAFETY: the null output pointer is rejected before it is written.
    assert_eq!(unsafe { midman_preset_config(0, std::ptr::null_mut()) }, ERR_ARGUMENT);
    let mut out = [0.0f64; 10];
    // SAFETY: `out` has room for 10 values; the bad index is rejected first.
    assert_eq!(unsafe { midman_preset_config(99, out.as_mut_ptr()) }, ERR_ARGUMENT);
}

#[test]
fn a_lab_runs_from_creation_to_destruction() {
    let handle = tiny_lab();
    assert_eq!(handle, 1);

    assert_eq!(midman_lab_step(handle, 5), STATUS_OK);
    let records = floats();
    assert_eq!(records.len(), 15, "three values per step");
    assert!(records.chunks_exact(3).all(|r| r[0].is_finite() && r[1] > 0.0 && r[2] > 0.0));
    assert!((records[0] - 12f32.ln()).abs() < 0.3, "the first loss is near ln(vocab)");

    assert_eq!(midman_lab_eval(handle), STATUS_OK);
    let json = text();
    assert!(json.contains("\"step\":5"), "{json}");
    assert!(json.contains("\"accuracy\":"), "{json}");
    assert!(json.contains("\"batch\":2,\"seq\":12"), "{json}");

    assert_eq!(midman_lab_stats(handle), STATUS_OK);
    let json = text();
    assert!(json.contains("\"name\":\"embed_tokens.weight\",\"shape\":[12,16]"), "{json}");
    assert!(json.contains("\"num_parameters\":"), "{json}");
    assert!(
        !json.contains("\"grad_norm\":null"),
        "a step has run, so every gradient norm is known: {json}"
    );

    assert_eq!(midman_lab_attention(handle, 1), STATUS_OK);
    assert_eq!(text(), "{\"layers\":2,\"heads\":2,\"seq\":12}");
    assert_eq!(floats().len(), 2 * 2 * 12 * 12);

    assert_eq!(midman_lab_embedding(handle), STATUS_OK);
    assert_eq!(text(), "{\"rows\":12,\"cols\":16}");
    assert_eq!(floats().len(), 12 * 16);

    assert_eq!(midman_lab_destroy(handle), STATUS_OK);
    assert_eq!(midman_lab_step(handle, 1), ERR_ARGUMENT);
    assert!(error().contains("not a live lab handle"), "{}", error());
    assert_eq!(midman_lab_destroy(handle), ERR_ARGUMENT);
}

#[test]
fn gradient_norms_are_null_before_the_first_step() {
    let handle = tiny_lab();
    assert_eq!(midman_lab_stats(handle), STATUS_OK);
    let json = text();
    assert!(json.contains("\"last_loss\":null"), "{json}");
    assert!(json.contains("\"grad_norm\":null"), "{json}");
}

#[test]
fn a_destroyed_slot_is_reused_and_labs_do_not_share_state() {
    let first = tiny_lab();
    let second = tiny_lab();
    assert_eq!((first, second), (1, 2));
    assert_eq!(midman_lab_step(first, 3), STATUS_OK);
    assert_eq!(midman_lab_stats(second), STATUS_OK);
    assert!(text().contains("\"step\":0"), "stepping one lab must not move the other");
    assert_eq!(midman_lab_destroy(first), STATUS_OK);
    assert_eq!(tiny_lab(), 1, "the freed slot is reused");
    assert_eq!(midman_lab_stats(second), STATUS_OK, "the other lab survives");
}

#[test]
fn invented_handles_are_errors_not_crashes() {
    for handle in [0u32, 7, u32::MAX] {
        assert_eq!(midman_lab_step(handle, 1), ERR_ARGUMENT, "handle {handle}");
        assert_eq!(midman_lab_eval(handle), ERR_ARGUMENT);
        assert_eq!(midman_lab_stats(handle), ERR_ARGUMENT);
        assert_eq!(midman_lab_attention(handle, 0), ERR_ARGUMENT);
        assert_eq!(midman_lab_embedding(handle), ERR_ARGUMENT);
        assert_eq!(midman_lab_destroy(handle), ERR_ARGUMENT);
        assert!(error().contains("handle"), "{}", error());
    }
}

#[test]
fn a_bad_creation_returns_zero_and_says_why() {
    let mut huge = TINY;
    huge[0] = 100_000.0;
    huge[1] = 128.0;
    huge[3] = 4.0;
    huge[4] = 4.0;
    huge[5] = 256.0;
    assert_eq!(create(&huge, &HYPER, 1, 2, 12), 0);
    assert!(error().contains("up to"), "{}", error());
    assert_eq!(create(&TINY, &HYPER, 1, 0, 12), 0, "no sequences");
    assert_eq!(create(&TINY, &HYPER, 1, 2, 17), 0, "longer than max_seq_len");
    let mut bad_hyper = HYPER;
    bad_hyper[6] = 1.0;
    assert_eq!(create(&TINY, &bad_hyper, 1, 2, 12), 0, "beta1 of 1");
    // SAFETY: null pointers are rejected before anything is read.
    assert_eq!(unsafe { midman_lab_create(std::ptr::null(), HYPER.as_ptr(), 1, 0, 2, 12) }, 0);
}

#[test]
fn attention_for_a_missing_sequence_is_an_error() {
    let handle = tiny_lab();
    assert_ne!(midman_lab_attention(handle, 2), STATUS_OK);
    assert!(error().contains("does not exist"), "{}", error());
}

#[test]
fn training_that_blows_up_reports_the_steps_it_completed() {
    let mut hyper = HYPER;
    hyper[0] = 1e30;
    hyper[1] = 1e30;
    hyper[2] = 0.0;
    let handle = create(&TINY, &hyper, 2024, 2, 12);
    assert_ne!(handle, 0, "{}", error());
    assert_eq!(midman_lab_step(handle, 50), STATUS_STOPPED);
    assert!(error().contains("finite"), "{}", error());
    let records = floats();
    assert!(
        !records.is_empty() && records.len() < 150 && (records.len() / 3) * 3 == records.len(),
        "{}",
        records.len()
    );
}

#[test]
fn each_call_replaces_the_previous_results() {
    let handle = tiny_lab();
    assert_eq!(midman_lab_embedding(handle), STATUS_OK);
    assert!(!floats().is_empty() && !text().is_empty());
    assert_eq!(midman_lab_step(handle, 0), STATUS_OK);
    assert!(floats().is_empty(), "a call with no numeric result must not leave an old one behind");
    assert!(text().is_empty() && error().is_empty());
}

#[test]
fn the_same_seed_gives_the_same_training_through_the_abi() {
    let run = |seed: u32| {
        let handle = create(&TINY, &HYPER, seed, 2, 12);
        assert_eq!(midman_lab_step(handle, 8), STATUS_OK);
        let records = floats();
        assert_eq!(midman_lab_destroy(handle), STATUS_OK);
        records.iter().map(|v| v.to_bits()).collect::<Vec<u32>>()
    };
    assert_eq!(run(5), run(5));
    assert_ne!(run(5), run(6));
}

#[test]
fn the_rope_table_is_cosines_then_sines() {
    assert_eq!(midman_rope_table(8, 6, 10_000.0), STATUS_OK);
    assert_eq!(text(), "{\"positions\":6,\"half\":4}");
    let table = floats();
    assert_eq!(table.len(), 2 * 6 * 4);
    assert_eq!(&table[..4], &[1.0; 4], "position 0 has cos 1");
    assert_eq!(&table[24..28], &[0.0; 4], "and sin 0");
    assert!(midman_rope_table(7, 6, 10_000.0) < 0, "an odd head size has no rotary table");
    assert!(!error().is_empty());
    assert_eq!(midman_rope_table(8, 6, f64::NAN), ERR_ARGUMENT);
}
