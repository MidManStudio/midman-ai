// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-checkpoint.md, section "tests/model_roundtrip.rs"
// ============================================================================
//! The end-to-end test: save a real model, load it into another one, and check the two
//! compute exactly the same thing.

use std::collections::BTreeMap;
use std::path::PathBuf;

use midman_checkpoint::{load_parameters, parameters_to_file, save_parameters, TensorFile};
use midman_foundation::Rng;
use midman_model::MidManModel;
use midman_model_config::Preset;
use midman_nn::Module;

fn model(preset: Preset, seed: u64) -> MidManModel {
    let config = preset.default_config();
    MidManModel::new(&config, &mut Rng::seed_from_u64(seed)).unwrap()
}

fn tokens(model: &MidManModel, n: usize) -> Vec<usize> {
    let vocab = model.config().vocab_size;
    (0..n).map(|i| (i * 7 + 3) % vocab).collect()
}

fn logits(model: &MidManModel) -> Vec<u32> {
    let ids = tokens(model, 2 * 16);
    model.forward(&ids, 2, 16).unwrap().data().iter().map(|v| v.to_bits()).collect()
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("midman-checkpoint-it-{}-{name}", std::process::id()))
}

#[test]
fn a_saved_model_loads_into_another_and_computes_the_same_logits_bit_for_bit() {
    let source = model(Preset::Smoke, 1);
    let mut target = model(Preset::Smoke, 2);
    assert_ne!(logits(&source), logits(&target), "two seeds should differ");

    let path = temp_path("smoke.safetensors");
    let mut metadata = BTreeMap::new();
    metadata.insert("step".to_string(), "0".to_string());
    save_parameters(&path, &source.parameters(), &metadata).unwrap();

    let file = TensorFile::load(&path).unwrap();
    assert_eq!(file.metadata().get("step").map(String::as_str), Some("0"));
    assert_eq!(file.tensors().len(), source.parameters().len());
    load_parameters(&file, &mut target.parameters_mut()).unwrap();
    std::fs::remove_file(&path).unwrap();

    assert_eq!(logits(&source), logits(&target));
}

#[test]
fn the_checkpoint_holds_exactly_the_models_parameter_count() {
    let m = model(Preset::Smoke, 1);
    let file = parameters_to_file(&m.parameters(), &BTreeMap::new()).unwrap();
    let values: usize = file.tensors().iter().map(|t| t.data().len()).sum();
    assert_eq!(values, m.num_parameters());
}

#[test]
fn a_checkpoint_of_a_different_size_is_refused_and_changes_nothing() {
    let smoke = model(Preset::Smoke, 1);
    let mut tiny = model(Preset::Tiny, 2);
    let before = logits(&tiny);
    let file = parameters_to_file(&smoke.parameters(), &BTreeMap::new()).unwrap();
    assert!(load_parameters(&file, &mut tiny.parameters_mut()).is_err());
    assert_eq!(logits(&tiny), before);
}

#[test]
fn saving_twice_gives_identical_files() {
    let m = model(Preset::Smoke, 1);
    let a = parameters_to_file(&m.parameters(), &BTreeMap::new()).unwrap().to_bytes().unwrap();
    let b = parameters_to_file(&m.parameters(), &BTreeMap::new()).unwrap().to_bytes().unwrap();
    assert_eq!(a, b);
}
