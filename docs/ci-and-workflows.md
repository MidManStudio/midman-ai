# CI and workflows

## Overview
Four GitHub Actions workflows and five helper scripts cover formatting, linting, tests, docs, security scans and benchmarks for the whole workspace.

Every workflow is manual only (`workflow_dispatch`). There are no push or pull_request triggers, so every run is one you chose to start. Each run writes a job summary and uploads its raw logs and a JSON result as an artifact.

The workflows call the scripts, so `scripts/` has to be in the repository before a workflow can pass. Apply the tarball first, then add the workflow files.

## Conventions
- Manual dispatch only. Enum-like inputs are `choice` dropdowns. The package dropdown in `ci.yml` and `benchmarks.yml` is a fixed list, so a new crate has to be added to both.
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

## Not decided yet
- `cargo-deny` is not used. It needs a license policy, and the workspace license is not chosen.
- No GPU or self-hosted runner jobs. The workflows run on `ubuntu-latest`.
- The per-crate docs still say "None yet" under CI and Workflows. These workflows are workspace-wide and take a package input, so they cover every crate.

## Fixes and Problems
None yet.
