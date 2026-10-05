# configs

Versioned model, training, and evaluation configuration.

- `models/` holds one TOML file per model-size preset (`smoke`, `tiny`, `small`, `base`, `medium`, `large`). The file is the readable copy of the preset in `crates/midman-model-config`, and the test `tests/preset_files.rs` in that crate fails if a file drifts from the code.
- Training and evaluation configs: nothing here yet.
