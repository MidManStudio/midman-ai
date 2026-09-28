# MidMan AI

A Rust-first research and development workspace for a foundation model and software/game-development assistant, with first-class Rust, Ubel, and Mid Engine workflows.

## Workspace status

This is an architecture and crate-layout scaffold. Crates are intentionally stubs; no model implementation, training data, weights, or credentials are included.

## Workspace

- `crates/`: reusable model, data, training, inference, agent, sandbox, and integration libraries.
- `apps/`: CLI, server, and trainer entry points.
- `configs/`: versioned model/training/evaluation configuration examples.
- `docs/`: architecture, data governance, training, sandbox, and integration notes.
- `data/`: manifests and instructions only; do not commit bulk datasets.
- `tests/`, `benches/`, `scripts/`: regression suites, benchmarks, and developer utilities.

## Initial commands

```sh
cargo metadata --workspace --no-deps
cargo test -p midman-foundation
cargo fmt --all -- --check
```

The workspace's `default-members` deliberately avoids assuming that all hardware, integration, or application targets are buildable on every host.

## Source and licensing policy

Every external training or evaluation source must be recorded with origin URL, immutable revision where applicable, license/terms, retrieval date, transformations, and exclusion status. Public availability alone is not permission to train on a source. Do not ingest secrets, credentials, personal data, or material whose license/terms are incompatible with the project.

## Existing ecosystem

- Ubel language/compiler/runtime: https://github.com/MidManStudio/ubel_stratum
- Mid Engine: https://github.com/Mid-D-Man/mid-engine
- DixScript: https://github.com/Mid-D-Man/DixScript-Rust
- AI Toolkit: https://github.com/MidManStudio/midmanstudio-ai-toolkit

Treat those repositories as their own sources of truth. Integrate through versioned dependencies or pinned revisions; do not copy their implementations into this workspace without review.
