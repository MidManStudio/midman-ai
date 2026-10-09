# Workspace Cargo configuration

## Overview
Notes on the root `Cargo.toml`, `rust-toolchain.toml`, `rustfmt.toml`, and `.cargo/config.toml`. The layout follows the conventions used in the Mid Engine workspace.

## Members
18 library crates under `crates/`, 3 binary crates under `apps/` and 1 benchmark package under `benches/`, listed explicitly. The root manifest is virtual, so it has no package of its own.

## [workspace.lints] and [workspace.dependencies]
`unsafe_code` is `deny` and `missing_docs` is `warn`, with clippy's `undocumented_unsafe_blocks` at `warn`. Every crate opts in with `[lints] workspace = true`. A crate that needs unsafe code opts back out with `#![allow(unsafe_code)]` at its crate root, so the exception is visible in one line.

`[workspace.dependencies]` holds `criterion` (version 0.5 with `html_reports`), which only `benches/midman-bench` uses. Every other crate still has no external dependencies. Shared dependency versions go in this table, and crates opt in with `criterion = { workspace = true }`.

The benchmark package's bench files open with `#![allow(missing_docs)]`, because `criterion_group!` expands to an undocumented `pub fn`.

## Profiles
`[profile.release]` and `[profile.bench]` are set explicitly because `cargo bench` does not inherit `[profile.release]`.

## Toolchain
`rust-toolchain.toml` uses the `stable` channel. No minimum supported Rust version is recorded yet. Every crate except the benchmark package was checked to build and test on Rust 1.75, and everything was checked on Rust 1.91.

## Toolchain walls
`benches/midman-bench` depends on criterion, which depends on `clap_builder`, and current `clap_builder` needs a Cargo that understands edition 2024 (about 1.85 or newer). Cargo resolves a package's whole manifest, dev-dependencies included, before it builds any target from it. That is why criterion lives in its own package: a criterion dev-dependency inside a library crate would raise the requirement for that crate's tests too.

What was checked on Rust 1.75 with the lockfile that contains criterion's 90 packages:
- `cargo test -p <crate>` works for each of the six crates that have code (`midman-foundation`, `midman-model-config`, `midman-tensor`, `midman-nn`, `midman-model`, `midman-optim`). Resolution is scoped to the package's own dependency closure, so the benchmark package is not touched.
- A bare `cargo test --workspace`, and `cargo check -p midman-bench --benches`, fail with "feature `edition2024` is required", because they build targets that need criterion. A plain `cargo check -p midman-bench` passes, because nothing it builds without `--benches` uses criterion.
- All 21 other packages pass `cargo check -p <package> --locked`.

CI uses the latest stable, so it is not affected. On a local Cargo older than 1.85, use `-p`.

`Cargo.lock` is version 3. Cargo 1.83 and newer write version 4 for new lockfiles, and Cargo 1.91 rewrote this repository's version 3 lockfile as version 4 when criterion's packages were added. Cargo 1.75 cannot read version 4, which would have extended the toolchain requirement from the benchmark package to every crate. The version line was set back to 3 by hand, and Cargo 1.91 keeps it on later commands. After a `cargo update` with a newer Cargo, check that the line still reads `version = 3`.

## Versioning
Every crate stays at `0.0.1` during active development. The first official release jumps straight to `1.0.0`.

## Not decided yet
- `[workspace.package]` is not used, and no crate sets a `license` field. Add both once a license is chosen.
- Whether any crate needs `unsafe` code.
- Whether to record a minimum supported Rust version. The benchmark package needs about 1.85 and the rest built on 1.75 when checked.

## Fixes and Problems

### `Cargo.toml`
- **`missing_docs` fired on criterion's macro output.** See `docs/midman-bench.md`; the allow lives in the bench files, not in this manifest.

### `Cargo.lock`
- **Adding criterion bumped the lockfile to version 4.** Described under "Toolchain walls". The version line was restored to 3, and `cargo audit` over the 90-package lockfile found no vulnerabilities and no warnings.
