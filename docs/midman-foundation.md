# midman-foundation

## Overview
Shared errors, format-version checks and the deterministic random number generator used across MidMan. The crate uses only the standard library, so every other crate can depend on it without pulling anything else in.

The crate description also mentions shared IDs and core traits. Nothing needs them yet, so they are not implemented.

## Modules

### `lib.rs`
**What it does:** Crate root. Declares the modules and re-exports `Error`, `Result`, `Rng`, `check_schema_version` and `VERSION`. The crate docs include a runnable example.

### `error.rs`
**What it does:** The `Error` enum used by every MidMan crate, and the `Result` alias. The variants are `Config`, `Shape`, `IndexOutOfRange`, `Parse`, `UnsupportedVersion`, `InvalidArgument` and `Io`. Short constructors such as `Error::config` and `Error::shape` keep call sites readable.

**Design notes:**
- Variants hold text instead of wrapped source errors, so the type is `Clone` and `Eq` and test assertions can compare errors directly.
- The enum is `#[non_exhaustive]`, so adding a variant later does not break code that has a wildcard arm.
- Messages name the thing that is wrong (the config field, the operation, the line number), because most errors here are programmer or data mistakes that a person has to read.

### `version.rs`
**What it does:** `VERSION`, the crate version from the manifest, and `check_schema_version`, which accepts a file's format version only if it equals the version this build reads.

**Design notes:** There are no format migrations yet, so an exact match is the only rule. When the first migration is written, this is the function that changes.

### `rng.rs`
**What it does:** `Rng`, a seedable xoshiro256** generator whose 64-bit seed is expanded with SplitMix64. It provides raw 64-bit output, uniform `f32` and `f64`, normal samples (Box-Muller), unbiased bounded integers (Lemire's method), in-place shuffling, and forked streams for independent consumers.

**Design notes:**
- A reproducible training run needs every source of randomness (weight initialization, data order, sampling) to come from one generator that never changes underneath it. There is no dependency on the `rand` crate for that reason.
- The integer outputs are identical on every platform. Known-answer tests pin the first outputs for two seeds, plus a bounded-integer sequence and a float sequence, against a separate Python implementation of the same algorithm.
- `Rng::normal` calls the platform's `ln`, `sin` and `cos`, so its last bits can differ between platforms.

**Limits:** Not cryptographically secure.

## CI and Workflows
The workspace workflows in `.github/workflows` cover this crate through the package dropdown. See `docs/ci-and-workflows.md`.

## Fixes and Problems
None yet.
