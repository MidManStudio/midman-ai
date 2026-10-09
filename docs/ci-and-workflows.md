# CI and workflows

## Overview
Five GitHub Actions workflows and six helper scripts cover formatting, linting, tests, docs, security scans, benchmarks and a training smoke test for the whole workspace. Two of the scripts have self-tests that run against real captured output.

Every workflow is manual only (`workflow_dispatch`). There are no push or pull_request triggers, so every run is one you chose to start. Each run writes a job summary and uploads its raw logs and a JSON result as an artifact.

The workflows call the scripts, so `scripts/` has to be in the repository before a workflow can pass. Apply the tarball first, then add the workflow files.

## Conventions
- Manual dispatch only. Enum-like inputs are `choice` dropdowns. The package dropdown in `ci.yml` and `benchmarks.yml` is a fixed list, so a new crate has to be added to both. `train-smoke.yml` has no dropdown.
- Every step that pipes into `tee` starts with `set -o pipefail`, so a failing command is not hidden by the pipe.
- Steps that can fail use `continue-on-error: true` and an `id`, so every check runs. The final step reads the step outcomes and fails the job if any check failed.
- The job summary comes from a script that parses the raw log. Summaries stay well under GitHub's 1024 KiB limit.
- Cache keys start with `cargo-registry-midman-<workflow>-` plus `runner.os`, so they cannot collide with other workflows or other repositories. `~/.cargo/bin` is never cached. `target/` is not cached yet. The only external code is criterion's dependency tree, which is built only for the benchmark package.
- Action versions are the current Node 24 majors: `actions/checkout@v7`, `actions/cache@v6`, `actions/upload-artifact@v7`. The v4 versions used in mid-engine declare Node 20 in their `action.yml`.
- Permissions are `contents: read` and checkouts do not persist credentials.
- `benchmarks.yml` and `train-smoke.yml` start with a parser self-test step (`python3 -m unittest discover -s scripts -p 'test_*.py'`), so a broken summary script fails in seconds instead of after a long run. They also take a `platforms` input (`ubuntu-only` by default, or `all` for `ubuntu-latest`, `macos-latest` and `ubuntu-24.04-arm`) and a free-text `baseline_note`, and their summaries name the runner.
- Raw log text embedded in a summary starts at cargo's own `Finished` line, so build noise is dropped, and is capped in size.

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
**What it does:** Runs `cargo bench` for the criterion package `midman-bench` (the default), one crate, or the whole workspace, and summarizes the criterion results.

**Inputs:** `package` (dropdown), `bench_filter` (optional regex passed after `--`), `baseline_note` (free text shown in the summary), `platforms` (`ubuntu-only` or `all`).

**Output:** a job summary per runner, plus artifact `bench-results-<run>-<runner>` with the raw log and `bench-results.json` (30 days).

Criterion needs a Cargo that understands edition 2024, so the workflow uses the latest stable. See "Toolchain walls" in `docs/workspace-cargo.md`. Runs on shared runners are noisy, so compare like with like and use `baseline_note` to say what changed. `platforms: all` is for checking that a number is not an artifact of one runner type. There is no ISA matrix, because the kernels are plain Rust with no dispatched SIMD or GPU backends to compare; add one when a crate gets those.

### `train-smoke.yml`
**What it does:** Runs the ignored test `the_smoke_preset_memorizes_random_sequences` from `crates/midman-model/tests/overfit.rs` in a release build: the 117K-parameter Smoke preset trains on 4 random sequences of 32 tokens for 300 steps and must memorize them (final loss under a tenth of the starting loss, next-token accuracy above 95%).

**Inputs:** `baseline_note` (free text shown in the summary), `platforms` (`ubuntu-only` or `all`). The random normal draws use the platform's `ln`, `sin` and `cos`, so the final loss can differ in its last digits between platforms; `platforms: all` puts the three results side by side.

**Output:** a job summary per runner with the start and end loss and the accuracy, plus artifact `train-smoke-<run>-<runner>` with the raw log and `train-smoke-results.json` (30 days).

A smaller version of the same test runs in every `cargo test`, so ordinary CI covers the training path. This workflow exists because the larger one is too slow in a debug build. It uses the workspace release profile (fat LTO, one codegen unit), so most of its time is compiling. The workflow file was checked with actionlint 1.7.12 and shellcheck. Its first run on GitHub (`ubuntu-latest`, Rust 1.99.0) passed, with loss 5.5674 to 0.0005 and 100% accuracy, the same as the local run on Rust 1.75 without LTO. The matrix and the new inputs have not been run on GitHub yet.

## Scripts
All scripts use only the Python standard library. `check_docs.py` and `repo_hygiene.py` need Python 3.11 or newer for `tomllib`.

### `ci_summary.py`
**What it does:** Parses the fmt, clippy, build and test logs from `ci.yml` into `ci-results.json` and a markdown summary. It trims backtraces from failing tests, shows compile errors when the test binaries did not build, and never fails the job itself.

### `check_docs.py`
**What it does:** Checks that every workspace member has `docs/<package>.md`, that every NOTICE header points at an existing doc and a matching heading, and that a crate doc ends with "Fixes and Problems". Files with no NOTICE header and em dashes in docs are reported but do not fail the run. Exits 1 on a hard failure.

### `repo_hygiene.py`
**What it does:** Reads the tracked files from git and fails on model weights, datasets, archives, build directories, credential-looking file names, files over 5 MiB, a workspace member without `[lints] workspace = true`, and a crate directory that is not a workspace member. Paths under `.mdix/` and `.github/` are skipped, because the replacements flow stores archives there. It also lists every use of `unsafe`.

### `bench_summary.py`
**What it does:** Parses criterion's `time:` and `thrpt:` lines from the `cargo bench` log into a table per group, with the estimate, its 95% interval, throughput and outlier count. An id is split at its final `/` into group and variant, and groups keep the order of the log. In a group, every variant gets a ratio against the `baseline-*` variant with a badge (parity or faster up to 1.05 times, then warning up to 1.5, error up to 5, and flagged beyond that). If there is no baseline it uses the `unit-*` variant and shows a plain multiple with no badge. It lists criterion warnings and panics as diagnostics, and for a failed run or a log with no results it embeds the end of the raw log from cargo's `Finished` line, capped at 200,000 bytes. It never fails the job.

**Decisions:**
- Time is labelled as criterion's estimate, not a median. The middle number in `time: [low estimate high]` is the estimate of the typical time, not the median of the samples.
- The throughput pattern allows several spaces between a number and its unit, because criterion prints `729.04  elem/s` with two for a unit that has no SI prefix.

### `test_bench_summary.py`
**What it does:** Self-tests for `bench_summary.py`, 27 tests. They run against `scripts/fixtures/criterion_real_run.txt`, 140 lines of real criterion output from a local run (13 results in 5 groups, with ids on their own line and on the same line as `time:`, `change:` blocks, outlier lines, and three throughput unit shapes), plus synthetic logs for cases the real output cannot show on demand: both cargo `Finished` formats, size caps, warnings, a missing log. Run with `python3 -m unittest discover -s scripts -p 'test_*.py'`.

### `security_summary.py`
**What it does:** Turns `cargo audit --json` and the gitleaks JSON report into markdown. It never prints secret values.

### `train_smoke_summary.py`
**What it does:** Reads the `loss A -> B, accuracy C` line and the `test result:` line from the train-smoke log and writes `train-smoke-results.json` and a markdown summary that names the runner and shows the note. It counts panics as a diagnostic. If the line is missing or the run failed, the summary shows the end of the raw log from cargo's `Finished` line, capped by size and by lines. It never fails the job.

### `test_train_smoke_summary.py`
**What it does:** Self-tests for `train_smoke_summary.py`. They run against `scripts/fixtures/train_smoke_real_run.txt`, a real smoke-test log in the older cargo format, plus synthetic logs for the modern format, failures and a missing log.

## Not decided yet
- `cargo-deny` is not used. It needs a license policy, and the workspace license is not chosen.
- No GPU or self-hosted runner jobs. The workflows run on `ubuntu-latest`.
- The per-crate docs still say "None yet" under CI and Workflows. These workflows are workspace-wide and take a package input, so they cover every crate.

## Fixes and Problems

### `bench_summary.py`
- **It labelled criterion's estimate as "Median".** Criterion's middle number is the estimate of the typical time (the slope or the mean), not a median. The column now says what it is.
- **It grouped by the first path component.** A group is everything before the final `/`, so `a/b/c` is group `a/b` and variant `c`. Groups also used to be sorted alphabetically, which put `matmul_square_192` before `matmul_square_32`; they now keep the order of the log.
- **It had never been run on real criterion output**, because no crate had a benchmark. The rewrite was tested against a real run, which showed that `thrpt:` lines carry two spaces before a unit without a prefix, and that criterion prints a `change:` block with `time:   [+17.1% ...]` lines inside it after a second run. Those lines do not match the result pattern, because every number in them ends in `%`, and a test now pins that.
- **The `Finished` marker only matched the newer cargo format.** Cargo before about 1.80 prints `Finished release [optimized] ...` without a backtick. The embed now matches a line that starts with `Finished` in either form. Found by running the train-smoke script on a log from cargo 1.75.
- **A forward-versus-backward ratio got a failure badge.** With the first benchmarks, `forward-backward` against `baseline-forward` showed `4.06x` with a failure badge, though backward costing about four forward passes is expected. The `unit-*` convention exists for this.
- A mutation run over the parser found two survivors, and both were equivalent mutants (the change lines cannot match the result pattern whatever the number pattern is, and the throughput mutant had changed a place where one space is correct). Every other injected bug was caught.

### `benchmarks.yml`
- **The dropdown had no package that has benchmarks.** `midman-bench` is added to it and to the dropdown in `ci.yml`, and is now the default instead of `workspace`.
- **Missing `platforms` input, and no runner name in the summary.** Both are added, with the matrix, runner-scoped cache keys and per-runner artifact names.
- **A broken parser would only have shown after a full benchmark run.** The parser self-test now runs first.

### `train-smoke.yml`
- **First version had no `baseline_note` or `platforms` input, no diagnostics, and embedded an unstripped log tail.** All four are fixed to match `benchmarks.yml`.

