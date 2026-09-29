# Workspace Cargo configuration

## Overview
Notes on the root `Cargo.toml`, `rust-toolchain.toml`, `rustfmt.toml`, and `.cargo/config.toml`. The layout follows the conventions used in the Mid Engine workspace.

## Members
18 library crates under `crates/` and 3 binary crates under `apps/`, listed explicitly. The root manifest is virtual, so it has no package of its own.

## [workspace.lints] and [workspace.dependencies]
`unsafe_code` is `deny` and `missing_docs` is `warn`, with clippy's `undocumented_unsafe_blocks` at `warn`. Every crate opts in with `[lints] workspace = true`. A crate that needs unsafe code opts back out with `#![allow(unsafe_code)]` at its crate root, so the exception is visible in one line.

`[workspace.dependencies]` is empty. Shared dependency versions go there once any exist.

## Profiles
`[profile.release]` and `[profile.bench]` are set explicitly because `cargo bench` does not inherit `[profile.release]`.

## Toolchain
`rust-toolchain.toml` uses the `stable` channel. No minimum supported Rust version is recorded yet.

## Versioning
Every crate stays at `0.0.1` during active development. The first official release jumps straight to `1.0.0`.

## Not decided yet
- `[workspace.package]` is not used, and no crate sets a `license` field. Add both once a license is chosen.
- Whether any crate needs `unsafe` code.

## Fixes and Problems
None yet.
