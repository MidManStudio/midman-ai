// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-model-config.md, section "tests/preset_files.rs"
// ============================================================================
//! Keeps `configs/models/*.toml` in step with the presets in code.

use std::fs;
use std::path::PathBuf;

use midman_model_config::{ModelConfig, Preset};

fn config_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../configs/models")
}

/// The exact text a preset's file must contain.
fn expected_file_text(preset: Preset) -> String {
    let cfg = preset.default_config();
    format!(
        "# MidMan model config preset: {name}\n# Parameters at vocab_size {vocab}: {count}\n{body}",
        name = preset.name(),
        vocab = cfg.vocab_size,
        count = cfg.param_count().unwrap(),
        body = cfg.to_toml_string(),
    )
}

#[test]
fn every_preset_has_an_up_to_date_file() {
    for preset in Preset::ALL {
        let path = config_dir().join(format!("{}.toml", preset.name()));
        let on_disk = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        assert_eq!(
            on_disk,
            expected_file_text(preset),
            "{} is out of date with the {} preset in code",
            path.display(),
            preset
        );
    }
}

#[test]
fn every_preset_file_parses_back_to_the_preset() {
    for preset in Preset::ALL {
        let path = config_dir().join(format!("{}.toml", preset.name()));
        let text = fs::read_to_string(&path).unwrap();
        let parsed = ModelConfig::from_toml_str(&text).unwrap();
        assert_eq!(parsed, preset.default_config(), "{preset}");
    }
}

#[test]
fn there_are_no_unlisted_model_files() {
    let known: Vec<String> = Preset::ALL.iter().map(|p| format!("{}.toml", p.name())).collect();
    for entry in fs::read_dir(config_dir()).unwrap() {
        let name = entry.unwrap().file_name().to_string_lossy().into_owned();
        assert!(known.contains(&name), "unexpected file configs/models/{name}");
    }
}
