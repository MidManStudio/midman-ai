# midman-model-config

## Overview
The architecture configuration of a MidMan model. `ModelConfig` describes a decoder-only transformer, validates it, counts its parameters exactly, and reads and writes a strict TOML-compatible text format. `Preset` names six standard sizes from about 0.1M to about 1B parameters.

The v0 architecture is a Llama-style decoder: pre-normalization with RMSNorm, rotary position embeddings, grouped-query attention, a gated (SwiGLU) MLP, and no bias terms. These are defaults to start from, not a final design. `ModelConfig` is the single source of truth for the model's shape, and the model crate must build exactly the parameters it counts.

The only dependency is `midman-foundation`.

## Modules

### `lib.rs`
**What it does:** Crate root. Re-exports `ModelConfig`, `ParamBreakdown`, `CONFIG_SCHEMA_VERSION` and `Preset`. The crate docs include a runnable example.

### `config.rs`
**What it does:** `ModelConfig` with ten fields (`vocab_size`, `d_model`, `n_layers`, `n_heads`, `n_kv_heads`, `d_ff`, `max_seq_len`, `rope_theta`, `norm_eps`, `tie_embeddings`), `validate`, `head_dim`, `kv_dim`, `param_breakdown` and `param_count`.

**Parameter formula** (`d` is `d_model`, `kv` is `n_kv_heads * head_dim`, `L` is `n_layers`):
- embedding: `vocab_size * d`
- attention per layer: `2*d*d + 2*d*kv` (query, key, value and output projections)
- MLP per layer: `3 * d * d_ff` (gate, up and down)
- norms: `2*d` per layer, plus `d` for the final norm
- output head: 0 when embeddings are tied, otherwise `vocab_size * d`

**Design notes:**
- Validation names the first bad field. It checks that `d_model` is divisible by `n_heads`, that `n_heads` is divisible by `n_kv_heads`, that the head dimension is even (rotary embeddings rotate pairs), and that the floats are finite and positive.
- Counting uses checked 64-bit arithmetic, so an absurd config is an error instead of a silent wraparound.
- Divisibility is written as `(a / b) * b == a` instead of `a % b == 0`. Newer clippy versions flag the modulo form on unsigned integers and ask for a method this workspace's older toolchains do not have.
- The counts are tested against a hand calculation for the smoke config and against totals computed separately in Python for every preset.

### `text.rs`
**What it does:** `ModelConfig::to_toml_string` and `ModelConfig::from_toml_str`.

**Format:** a flat TOML subset. It supports `key = value` lines, integers with optional `_` separators, floats, booleans, blank lines and `#` comments. There are no tables and no strings. Example:

```
schema_version = 1
vocab_size = 32000
d_model = 384
n_layers = 6
n_heads = 6
n_kv_heads = 6
d_ff = 1024
max_seq_len = 512
rope_theta = 10000.0
norm_eps = 1e-5
tie_embeddings = true
```

**Design notes:**
- Files are strict. Every key is required, and unknown or duplicate keys are errors, so a config never silently falls back to a default.
- The schema version is read first, so a file written by a newer format reports the version mismatch instead of an unknown-key error.
- Floats are written in the shortest form that reads back to the same bits, and a test checks the round trip bit for bit.
- Parse errors carry the line number. Parsed values go through `validate`.

### `presets.rs`
**What it does:** `Preset` (`Smoke`, `Tiny`, `Small`, `Base`, `Medium`, `Large`) with `config(vocab_size)`, `default_config`, `default_vocab_size`, `name`, `Display` and `FromStr`.

| Preset | Layers | d_model | Heads | KV heads | d_ff | Default vocab | Parameters |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `smoke` | 2 | 64 | 4 | 4 | 176 | 256 | 117,056 |
| `tiny` | 4 | 128 | 4 | 4 | 344 | 4,096 | 1,315,968 |
| `small` | 6 | 384 | 6 | 6 | 1,024 | 32,000 | 22,909,824 |
| `base` | 12 | 768 | 12 | 12 | 2,048 | 32,000 | 109,529,856 |
| `medium` | 24 | 1,024 | 16 | 8 | 2,816 | 32,000 | 315,933,696 |
| `large` | 20 | 2,048 | 16 | 8 | 5,632 | 32,000 | 1,009,338,368 |

All presets tie the input and output embeddings. Changing only the vocabulary size changes only the embedding count.

**Design notes:** The sizes are starting points chosen so the parameter counts land near round numbers. They are not tuned. The vocabulary is a parameter because it depends on the tokenizer, which does not exist yet.

### `tests/preset_files.rs`
**What it does:** Keeps `configs/models/*.toml` in step with the presets in code. It checks that each preset's file has exactly the expected text, that each file parses back to the preset, and that no unlisted files sit in the directory.

**Design notes:** The files are the human-readable copy of the presets. If a preset changes in code and the file does not, this test fails and names the file.

## CI and Workflows
The workspace workflows in `.github/workflows` cover this crate through the package dropdown. See `docs/ci-and-workflows.md`.

## Fixes and Problems
None yet.
