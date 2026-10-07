# CI and workflows

## Overview
Five GitHub Actions workflows and six helper scripts cover formatting, linting, tests, docs, security scans, benchmarks and a training smoke test for the whole workspace.

Every workflow is manual only (`workflow_dispatch`). There are no push or pull_request triggers, so every run is one you chose to start. Each run writes a job summary and uploads its raw logs and a JSON result as an artifact.

The workflows call the scripts, so `scripts/` has to be in the repository before a workflow can pass. Apply the tarball first, then add the workflow files.

## Conventions
- Manual dispatch only. Enum-like inputs are `choice` dropdowns. The package dropdown in `ci.yml` and `benchmarks.yml` is a fixed list, so a new crate has to be added to both. `train-smoke.yml` has no dropdown.
- Every step that pipes into `tee` starts with `set -o pipefail`, so a failing command is not hidden by the pipe.
- Steps that can fail use `continue-on-error: true` and an `id`, so every check runs. The final step reads the step outcomes and fails the job if any check failed.
- The job summary comes from a script that parses the raw log. Summaries stay well under GitHub's 1024 KiB limit.
- Cache keys start with `cargo-registry-midman-<workflow>-` plus `runner.os`, so they cannot collide with other workflows or other repositories. `~/.cargo/bin` is never cached. `target/` is not cached yet, because the workspace has no dependencies whose compile time is worth the cache quota.
- Action versions are the current Node 24 majors: `actions/checkout@v7`, `actions/cache@v6`, `actions/upload-artifact@v7`. The v4 versions used in mid-engine declare Node 20 in their `action.yml`.
- Permissions are `contents: read` and checkouts do not persist credentials.

## Workflows

### `ci.yml`
**What it does:** Runs `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo build --all-targets` and `cargo test --no-fail-fast`, for the whole workspace or one package.

**Inputs:** `package` (dropdown: `workspace`, or any crate or app name).

**Output:** job summary, plus artifact `ci-results-<run>` with the four raw logs and `ci-results.json` (7 days).

**Script:** `scripts/ci_summary.py`.

### `docs.yml`
**What it does:** Runs `scripts/check_docs.py`, then builds rustdoc for the workspace with `RUSTDOCFLAGS=-D warnings`.

**Inputs:** `require_notice_headers` (boolean, default off). When on, a `.rs` file with no NOTICE header fails the run.

**Output:** job summary, artifact `rustdoc-html-<run>` (the HTML) and artifact `docs-results-<run>` (raw log and JSON), 7 days.

The HTML is not deployed anywhere. There is no docs hosting target for this repository yet.

### `security.yml`
**What it does:** Three independent jobs.
- `audit`: `cargo audit --json` against `Cargo.lock`. Known vulnerabilities fail the job. Unmaintained, unsound and yanked warnings are listed but do not fail it. `cargo-audit` is installed with `taiki-e/install-action@v2`.
- `secrets`: gitleaks over the full git history, with values redacted. The gitleaks CLI is downloaded at a pinned version and checked against a pinned SHA-256. The gitleaks GitHub Action is not used, because it needs a license key for organization repositories.
- `hygiene`: runs `scripts/repo_hygiene.py`.

**Output:** a job summary per job, plus artifacts `audit-<run>`, `gitleaks-<run>` and `hygiene-<run>` (7 days).

**Scripts:** `scripts/security_summary.py`, `scripts/repo_hygiene.py`.

### `benchmarks.yml`
**What it does:** Runs `cargo bench` for the whole workspace or one package and summarizes criterion results.

**Inputs:** `package` (dropdown), `bench_filter` (optional regex passed after `--`), `baseline_note` (free text shown in the summary).

**Output:** job summary, plus artifact `bench-results-<run>` with the raw log and `bench-results.json` (30 days).

Nothing in the workspace has a `[[bench]]` target yet, so the run currently finishes green and says no results were found. Runs on shared runners are noisy, so compare like with like. There is no ISA matrix; add one when a crate gets dispatched SIMD or GPU backends that need separate numbers.

### `train-smoke.yml`
**What it does:** Runs the ignored test `the_smoke_preset_memorizes_random_sequences` from `crates/midman-model/tests/overfit.rs` in a release build: the 117K-parameter Smoke preset trains on 4 random sequences of 32 tokens for 300 steps and must memorize them (final loss under a tenth of the starting loss, next-token accuracy above 95%).

**Inputs:** none.

**Output:** job summary with the start and end loss and the accuracy, plus artifact `train-smoke-<run>` with the raw log and `train-smoke-results.json` (30 days).

A smaller version of the same test runs in every `cargo test`, so ordinary CI covers the training path. This workflow exists because the larger one is too slow in a debug build. It uses the workspace release profile (fat LTO, one codegen unit), so most of its time is compiling. The workflow file was checked with actionlint 1.7.12 and shellcheck. The test itself was run on Rust 1.75 with LTO turned off, where it took about 5 seconds; the workflow itself has not been run on GitHub.

## Scripts
All scripts use only the Python standard library. `check_docs.py` and `repo_hygiene.py` need Python 3.11 or newer for `tomllib`.

### `ci_summary.py`
**What it does:** Parses the fmt, clippy, build and test logs from `ci.yml` into `ci-results.json` and a markdown summary. It trims backtraces from failing tests, shows compile errors when the test binaries did not build, and never fails the job itself.

### `check_docs.py`
**What it does:** Checks that every workspace member has `docs/<package>.md`, that every NOTICE header points at an existing doc and a matching heading, and that a crate doc ends with "Fixes and Problems". Files with no NOTICE header and em dashes in docs are reported but do not fail the run. Exits 1 on a hard failure.

### `repo_hygiene.py`
**What it does:** Reads the tracked files from git and fails on model weights, datasets, archives, build directories, credential-looking file names, files over 5 MiB, a workspace member without `[lints] workspace = true`, and a crate directory that is not a workspace member. Paths under `.mdix/` and `.github/` are skipped, because the replacements flow stores archives there. It also lists every use of `unsafe`.

### `bench_summary.py`
**What it does:** Parses criterion `time:` lines from the `cargo bench` log into a table grouped by criterion group, with outlier counts and the change verdict against the previous run. It reports common problems such as "took zero time", and shows the end of the raw log when it finds no results.

### `security_summary.py`
**What it does:** Turns `cargo audit --json` and the gitleaks JSON report into markdown. It never prints secret values.

### `train_smoke_summary.py`
**What it does:** Reads the `loss A -> B, accuracy C` line and the `test result:` line from the train-smoke log and writes `train-smoke-results.json` and a markdown summary. If the line is missing or the run failed, the summary shows the end of the raw log. It never fails the job.

## Not decided yet
- `cargo-deny` is not used. It needs a license policy, and the workspace license is not chosen.
- No GPU or self-hosted runner jobs. The workflows run on `ubuntu-latest`.
- The per-crate docs still say "None yet" under CI and Workflows. These workflows are workspace-wide and take a package input, so they cover every crate.

## Fixes and Problems
None yet.
