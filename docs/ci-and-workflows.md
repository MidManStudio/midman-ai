# CI and workflows

## Overview
Eight GitHub Actions workflows and nine helper scripts cover formatting, linting, tests, docs, security scans, benchmarks, a training smoke test, a data pipeline smoke test and the C ABI for the whole workspace. Five of the scripts have self-tests that run against real captured output.

Every workflow is manual only (`workflow_dispatch`). There are no push or pull_request triggers, so every run is one you chose to start. Each run writes a job summary and uploads its raw logs and a JSON result as an artifact.

The workflows call the scripts, so `scripts/` has to be in the repository before a workflow can pass. Apply the tarball first, then add the workflow files.

## Conventions
- Manual dispatch only. Enum-like inputs are `choice` dropdowns. The package dropdown in `ci.yml` and `benchmarks.yml` is a fixed list, so a new crate has to be added to both. `train-smoke.yml` and `midman-api-ffi.yml` have no package dropdown.
- Every step that pipes into `tee` starts with `set -o pipefail`, so a failing command is not hidden by the pipe.
- Steps that can fail use `continue-on-error: true` and an `id`, so every check runs. The final step reads the step outcomes and fails the job if any check failed.
- The job summary comes from a script that parses the raw log. Summaries stay well under GitHub's 1024 KiB limit.
- Cache keys start with `cargo-registry-midman-<workflow>-` plus `runner.os`, so they cannot collide with other workflows or other repositories. `~/.cargo/bin` is never cached. `target/` is not cached yet. The only external code is criterion's dependency tree and the `cc` crate, which are built only for the benchmark package.
- Action versions are the current Node 24 majors: `actions/checkout@v7`, `actions/cache@v6`, `actions/upload-artifact@v7`. The v4 versions used in mid-engine declare Node 20 in their `action.yml`.
- Permissions are `contents: read` and checkouts do not persist credentials.
- `benchmarks.yml`, `train-smoke.yml` and `midman-api-ffi.yml` start with a parser self-test step (`python3 -m unittest discover -s scripts -p 'test_*.py'`), so a broken summary script fails in seconds instead of after a long run. They also take a `platforms` input (`ubuntu-only` by default, or `all` for `ubuntu-latest`, `macos-latest` and `ubuntu-24.04-arm`) and a free-text `baseline_note`, and their summaries name the runner.
- Raw log text embedded in a summary starts at cargo's own `Finished` line, so build noise is dropped, and is capped in size.

## Workflows

### `ci.yml`
**What it does:** Runs `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo build --all-targets` and `cargo test --no-fail-fast`, for the whole workspace or one package.

**Inputs:** `package` (dropdown: `workspace`, or any crate or app name).

**Output:** job summary, plus artifact `ci-results-<run>` with the four raw logs and `ci-results.json` (7 days).

**Script:** `scripts/ci_summary.py`.

`midman-tensor` has an optional `avx` feature, and the Clippy and Test steps cover both configurations. A workspace run gets the feature switched on by `midman-bench`, because Cargo unifies features across the packages of one build, so the extra commands lint and test `midman-tensor` alone, where it is off, and append their output to the same raw logs. A run for `midman-tensor` alone gets the feature off, so there the extra commands add `--features avx`. Other packages get no extra command. Without these, the default build of the crate would not be checked by a workspace run.

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

Criterion needs a Cargo that understands edition 2024, so the workflow uses the latest stable. See "Toolchain walls" in `docs/workspace-cargo.md`. Runs on shared runners are noisy, so compare like with like and use `baseline_note` to say what changed. `platforms: all` is for checking that a number is not an artifact of one runner type. There is no ISA matrix. `midman-tensor`'s AVX kernels are chosen at run time from what the CPU reports, not by a compile-time target feature, so one build covers every tier, and the `-scalar` twins in the matrix-product and step groups show the AVX kernels and the scalar ones in the same run. The runners do not have AVX-512 and no such kernel exists. Add an ISA matrix when a crate gets backends selected by compile-time flags.

The benchmark package also compiles plain C versions of several kernels (`benches/midman-bench/c/`) with the `cc` crate, so the runner needs a C compiler: `gcc` on the Ubuntu runners and `clang` on macOS, both preinstalled. The job summary header names the compiler, read from a line the benches print at the start of the run, and a group that has `baseline-c-*` variants gets a `vs fastest C` column. The workflow file itself did not change for this; the macOS and ARM legs have not been run with the C kernels yet.

### `train-smoke.yml`
**What it does:** Runs the ignored test `the_smoke_preset_memorizes_random_sequences` from `crates/midman-model/tests/overfit.rs` in a release build: the 117K-parameter Smoke preset trains on 4 random sequences of 32 tokens for 300 steps and must memorize them (final loss under a tenth of the starting loss, next-token accuracy above 95%).

**Inputs:** `baseline_note` (free text shown in the summary), `platforms` (`ubuntu-only` or `all`). The random normal draws use the platform's `ln`, `sin` and `cos`, so the final loss can differ in its last digits between platforms; `platforms: all` puts the three results side by side.

**Output:** a job summary per runner with the start and end loss and the accuracy, plus artifact `train-smoke-<run>-<runner>` with the raw log and `train-smoke-results.json` (30 days).

A smaller version of the same test runs in every `cargo test`, so ordinary CI covers the training path. This workflow exists because the larger one is too slow in a debug build. It uses the workspace release profile (fat LTO, one codegen unit), so most of its time is compiling. The workflow file was checked with actionlint 1.7.12 and shellcheck. Its first run on GitHub (`ubuntu-latest`, Rust 1.99.0) passed, with loss 5.5674 to 0.0005 and 100% accuracy, the same as the local run on Rust 1.75 without LTO. The matrix and the new inputs have not been run on GitHub yet.

On x86_64 the step runs the test twice, the second time with `--features midman-tensor/avx`, and prints a line naming each run into the raw log. The AVX kernels are bit-identical to the scalar ones, so the summary's Run 2 row must show the same loss and accuracy as Run 1; the summary does not label which is which, so Run 1 is the scalar build and Run 2 the AVX one. The ARM legs run only the scalar build, because the feature does nothing there. The second run has not been run on GitHub.

### `data-smoke.yml`
**What it does:** Runs the whole data path on the runner, over this repository's own text files (every tracked file under 1 MB that is valid UTF-8 and has no space in its name, in sorted order): builds the tools, trains a vocabulary of 1,024 ids with the eight agreed special tokens, runs `build_shards` (scrub, deduplicate, tokenize, pack into shards of about 200,000 tokens, read them back, hash the first 100 batches), and times the scrubber on the same files. On x86_64 Linux it also downloads gitleaks (the version and checksum `security.yml` pins), generates the made-up corpus of 288 secrets of 24 kinds from `crates/midman-data/tests/fixtures/oracle_corpus.py`, and compares gitleaks and the scrubber on it. The scrubber has to find every sample of every kind and nothing outside them.

**Inputs:** `baseline_note` (free text shown in the summary), `platforms` (`ubuntu-only` or `all`).

**Output:** a job summary per runner with the pipeline numbers, the SHA-256 of the vocabulary file, of all the shards and of the first batches, and the comparison table; artifact `data-smoke-<run>-<runner>` with the raw log, `data-smoke-results.json`, the vocabulary file and the shards (30 days). With `platforms: all` a last job, `compare`, downloads the three results and fails unless the vocabulary, every shard, the digest of the first batches and the counts are byte for byte the same on `ubuntu-latest`, `macos-latest` and `ubuntu-24.04-arm`. The corpus is the same files at the same commit and every step is integer arithmetic or a fixed algorithm (the trainer, SHA-256, the shuffle); the scrubber's entropy test uses the platform's `log2`, so a difference in a rare borderline case would point there. `deploy-site.yml` publishes the latest result on the Tests page.

`ci.yml` already runs the unit tests of `midman-data` and `midman-tokenizer`; this workflow adds the command-line path, real numbers on GitHub's machines, the second tool and the second machine. It uses the workspace release profile, so most of its time is compiling. The workflow file was checked with actionlint 1.7.12 and shellcheck, and its steps were run in order in the sandbox on x86_64 Linux with Rust 1.91, where every step passed, including the gitleaks download and checksum and the comparison: 256 files, 760 merges, 713,006 tokens at 2.25 bytes per token, 4 shards, 5,568 sequences, the scrubber at 72 MB/s (sandbox figures, one noisy core). The macOS and ARM legs, the `compare` job and the workflow as a whole have not run on GitHub yet.

### `midman-api-ffi.yml`
**What it does:** Tests the C ABI of `midman-api` the way a C or C++ program uses it, in this order: the crate's Rust tests (`cargo test -p midman-api`); `crates/midman-api/ffi-smoke-test/test.c` compiled as C11 with `-Wall -Wextra -Wpedantic -Werror` and run against the shared library, then against the static library; `cxx_check.cpp`, which includes the header from C++17; `test.c` rebuilt with AddressSanitizer and UndefinedBehaviorSanitizer; and, on Linux, `test.c` under valgrind with a leak check for definitely lost blocks.

**Inputs:** `baseline_note` (free text shown in the summary), `platforms` (`ubuntu-only`, or `all` for `ubuntu-latest`, `macos-latest` and `ubuntu-24.04-arm`), and `profile` (`debug` or `release`, which build of the library the C programs use).

**Output:** a job summary per runner with a table of suites and every failed check by name, plus artifact `midman-api-ffi-<run>-<runner>` with the raw logs and `midman-api-ffi-results.json` (30 days).

The static library is built on request with `cargo rustc --crate-type staticlib`, because the crate's own crate types are `rlib` and `cdylib`, and the system libraries a C linker needs after the archive are read from `rustc --print native-static-libs` and not guessed. The C and C++ programs run in one step with a log each, so one failure does not hide the others, and a program that was never built leaves the compile log in its place. The last step runs `scripts/ffi_summary.py gate`, which fails the run if any check failed or none were collected. Leak detection is off in the sanitizer run on macOS, where it is not supported, and valgrind is Linux-only.

The workflow file was checked with actionlint 1.7.12 and shellcheck. Every command in it was run in a sandbox with gcc 13.3, Rust 1.91 and valgrind 3.22, in both profiles, with 90 passing checks from `test.c` in each of the shared, static, sanitizer and valgrind runs. The workflow as a whole, the macOS and ARM legs and the `native-static-libs` output on those platforms have not run on GitHub yet.

### `deploy-site.yml`
**What it does:** Publishes the site to Cloudflare Pages. Manual dispatch only: it never runs on a push. It builds the WebAssembly engine (a SIMD build and a plain build), runs the engine crate's tests, the Node tests (engine against the real binary, charts, the lab page and the two result pages in a headless DOM) and the script tests, builds the docs book with a pinned mdBook v0.4.40, pulls the latest results from the most recent successful `ci.yml`, `benchmarks.yml`, `train-smoke.yml` and `data-smoke.yml` runs, assembles `dist/`, makes sure the Pages project exists, and deploys.

**Inputs:** `run_tests` (default true), `project_name` (default `midman-ai`).

**Output:** the deployed site, a summary of the engine builds and the results published, and artifact `site-<run>` with `dist/` and the raw test logs (14 days).

Missing results never block a deploy; the pages say that none are published yet. The repository needs the `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` secrets. The workflow file was checked with actionlint and shellcheck, and every step that can run in a sandbox was run there (the engine builds, all tests, the book, the assembly); the Cloudflare steps and the workflow as a whole have not run on GitHub yet.

## Scripts
All scripts use only the Python standard library. `check_docs.py` and `repo_hygiene.py` need Python 3.11 or newer for `tomllib`.

### `ci_summary.py`
**What it does:** Parses the fmt, clippy, build and test logs from `ci.yml` into `ci-results.json` and a markdown summary. It trims backtraces from failing tests, shows compile errors when the test binaries did not build, and never fails the job itself.

### `check_docs.py`
**What it does:** Checks that every workspace member has `docs/<package>.md`, that every NOTICE header points at an existing doc and a matching heading, and that a crate doc ends with "Fixes and Problems". Files with no NOTICE header and em dashes in docs are reported but do not fail the run. Exits 1 on a hard failure.

### `repo_hygiene.py`
**What it does:** Reads the tracked files from git and fails on model weights, datasets, archives, build directories, credential-looking file names, files over 5 MiB, a workspace member without `[lints] workspace = true`, and a crate directory that is not a workspace member. Paths under `.mdix/` and `.github/` are skipped, because the replacements flow stores archives there. It also lists every use of `unsafe`.

### `bench_summary.py`
**What it does:** Parses criterion's `time:` and `thrpt:` lines from the `cargo bench` log into a table per group, with the estimate, its 95% interval, throughput and outlier count. An id is split at its final `/` into group and variant, and groups keep the order of the log. In a group, every variant gets a ratio against the `baseline-*` variant with a badge (parity or faster up to 1.05 times, then warning up to 1.5, error up to 5, and flagged beyond that). If there is no baseline it uses the `unit-*` variant and shows a plain multiple with no badge. If a group also has `baseline-c-*` variants (the same operation in plain C), every variant also gets a ratio against the fastest of them, shown as `vs fastest C` with the same badges, and the Rust baseline stays the group's denominator because it is registered first. The C compiler is read from the line `C baselines compiled with: ...` that the benches print. It lists criterion warnings and panics as diagnostics, and for a failed run or a log with no results it embeds the end of the raw log from cargo's `Finished` line, capped at 200,000 bytes. It never fails the job.

**Decisions:**
- Time is labelled as criterion's estimate, not a median. The middle number in `time: [low estimate high]` is the estimate of the typical time, not the median of the samples.
- The throughput pattern allows several spaces between a number and its unit, because criterion prints `729.04  elem/s` with two for a unit that has no SI prefix.
- The C reference is the fastest `baseline-c-*` variant, not the first. For the matrix product that is the i-k-j loop, the best plain C a person would write, and not the textbook loop that the Rust baseline also uses. The `baseline-c-` prefix is matched whole, so a variant called `baseline-cache` is not a C variant.
- The JSON gains `meta.c_compiler`, `c_reference` per group (a variant name or null) and `ratio_c` per row (null in a group without C variants). The existing fields are unchanged, so the site's benchmarks page reads the new files as before and does not show the new column.

### `test_bench_summary.py`
**What it does:** Self-tests for `bench_summary.py`, 36 tests. They run against `scripts/fixtures/criterion_real_run.txt`, 140 lines of real criterion output from a local run (13 results in 5 groups, with ids on their own line and on the same line as `time:`, `change:` blocks, outlier lines, and three throughput unit shapes), plus synthetic logs for cases the real output cannot show on demand: both cargo `Finished` formats, size caps, warnings, a missing log. Nine of them cover the C comparison: the fastest C variant is the reference, the ratios, the Rust baseline staying the denominator, groups without C variants, a variant that only resembles the prefix, the markdown column, the compiler line (read once, only at the start of a line, absent when unknown) and the JSON. Run with `python3 -m unittest discover -s scripts -p 'test_*.py'`.

### `ffi_summary.py`
**What it does:** Two commands. `parse` turns the logs of `midman-api-ffi.yml` into `midman-api-ffi-results.json` and a markdown summary: a suite for the Rust tests (one check per `test name ... ok` line, doctests included), a suite per C or C++ program (one check per `ok:` or `FAIL:` line), the same plus three sanitizer checks for the sanitizer run, and the same plus three valgrind checks for the valgrind run. `gate` exits 1 if any check failed or none were collected. `parse` never fails the job.

**Decisions:**
- A C log with no closing `=== N check(s) failed ===` line is one failed check, "program ran to completion", whatever else it contains. A crash and a link error print no `FAIL:` lines, and would otherwise read as zero failures.
- A closing line that disagrees with the number of `FAIL:` lines, and a log with a closing line and no checks, are failed checks too.
- A Rust log with no tests in it (cargo missing, a compile error, a filter that matched nothing) is one failed check, not a suite that quietly is not there. A valgrind log with no `ERROR SUMMARY` line is a failed check as well.
- The sanitizer checks look for `ERROR: AddressSanitizer`, `ERROR: LeakSanitizer` and `runtime error:` in the log, and the undefined-behavior check carries the first three messages as its detail.
- The valgrind checks read the `ERROR SUMMARY` line and the `definitely lost` line, and also check the exit status that `--error-exitcode=99` turns valgrind's findings into.
- The raw log of a failed suite is embedded in the summary from its end, capped at 20,000 characters, and only for suites that failed.
- The command line takes `NAME=PATH` pairs, and a name may contain commas and spaces.

### `test_ffi_summary.py`
**What it does:** Self-tests for `ffi_summary.py`, 26 tests, run against real captured output in `scripts/fixtures/`: `ffi_c_real_run.txt` (a passing run of `test.c`, 90 checks), `ffi_c_real_fail.txt` (a run with two failed checks), `ffi_c_real_crash.txt` (the library aborted on purpose partway through, 25 checks and no closing line), `ffi_asan_real_use_after_free.txt`, `ffi_ubsan_real_overflow.txt` and `ffi_lsan_real_leak.txt` (real sanitizer reports from a small faulty program), `ffi_valgrind_real_clean.txt` and `ffi_valgrind_real_leak.txt`, and `ffi_cargo_real_run.txt` (the test lines of a real `cargo test -p midman-api`, 41 tests). Synthetic logs cover the cases real output cannot show on demand. The tests pass the `parse` command a full set of logs and check the JSON, the markdown and the gate's exit status.

### `security_summary.py`
**What it does:** Turns `cargo audit --json` and the gitleaks JSON report into markdown. It never prints secret values.

### `train_smoke_summary.py`
**What it does:** Reads the `loss A -> B, accuracy C` line and the `test result:` line from the train-smoke log and writes `train-smoke-results.json` and a markdown summary that names the runner and shows the note. It counts panics as a diagnostic. If the line is missing or the run failed, the summary shows the end of the raw log from cargo's `Finished` line, capped by size and by lines. It never fails the job.

### `test_train_smoke_summary.py`
**What it does:** Self-tests for `train_smoke_summary.py`. They run against `scripts/fixtures/train_smoke_real_run.txt`, a real smoke-test log in the older cargo format, plus synthetic logs for the modern format, failures and a missing log.

### `data_smoke_summary.py`
**What it does:** Reads the log of `data-smoke.yml`, which has one section per step starting with a line `=== name ===`, and writes `data-smoke-results.json` and a markdown summary: the pipeline numbers (corpus, vocabulary, files read, duplicates, secrets replaced by rule, documents, tokens, shards, sequences, speed), the SHA-256 of the vocabulary file, of every shard and of all the shards together (the files are hashed here, not in the log), and the scrubber-against-gitleaks table. In the JSON the digests are `vocabulary`, `shards` (a plain list, in file-name order) and `combined` (over the names and digests of all the shards). The JSON has `ok`, which is true only if every step that ran succeeded (the gitleaks step may be skipped), the numbers and the batch digest were found in the log, and an oracle that ran agreed. It never fails the job; the last workflow step reads `ok`. With `--compare DIR` it reads every `data-smoke-results.json` below `DIR` and checks that the vocabulary, every shard, the first-batches digest and the counts are the same on every runner; it exits 1 if they differ, if a value is missing, or if fewer than two results are found. If a run failed, the summary shows the end of the raw log from cargo's `Finished` line, capped by size and by lines.

### `test_data_smoke_summary.py`
**What it does:** Thirty-eight self-tests for `data_smoke_summary.py`. They run against `scripts/fixtures/data-smoke-ubuntu.txt`, a log captured by running the workflow's own `run:` blocks in order, and against logs made from it with one thing changed: every failed or skipped step, a missed sample, a finding outside the samples, a missing digest or speed line, ANSI colour codes, and empty and garbage logs. They also cover the file hashes (including that the combined digest depends on every shard), the markdown, the command line, and the comparison of runners: identical runners agree, each kind of difference is caught, a single shard that differs or is missing is named, and the exit codes. `scripts/fixtures/data-smoke-results.json` is the result of that run, made by running the script over the run's saved outputs (it was made again when the digest layout changed, see Fixes and Problems); `web/tests/test/fixtures/` holds a copy for the page tests.

### `assemble_site.py`
**What it does:** Puts the site's parts into `dist/`: the landing page at the root, the lab, tests and benchmarks pages (without their tests and fixtures), the built book as `docs/`, both engine builds into `lab/`, the result files into `data/`, and the headers file. It fails naming what is missing if either engine build or the book is absent, because a site published without them would look fine and not work; missing result files are not an error.

### `test_assemble_site.py`
**What it does:** Seven self-tests for `assemble_site.py`: a complete site lands in the right places, tests and fixtures are not published, a missing engine is an error that names the file and assembles nothing, missing data is fine, an old output is replaced, and the command line exits non-zero on a problem.

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

- **Groups with a C baseline needed a second ratio.** The Rust baseline stays the denominator, so on its own the summary could not say how `midman-tensor` compares with plain C. The `vs fastest C` column and the compiler line were added, with nine tests. Four injected bugs (slowest instead of fastest, an inverted ratio, a prefix that matches too much, an unanchored compiler pattern) were each caught.

### `ffi_summary.py`
- **The cargo pattern missed doctests.** A doctest line is `test crates/midman-api/src/lib.rs - (line 11) ... ok`, and a name with spaces did not match a pattern that wanted one word. Found by testing against a real log, which has 41 tests where the first version counted 40.
- Ten injected bugs were each caught by a test (a missing closing line passing, a count disagreement ignored, no checks passing, the doctest pattern narrowed, the gate passing zero checks, lost bytes ignored, the leak check dropped, the sanitizer detail dropped, a zero total counted as a pass, a missing cargo log passing).

### `midman-api-ffi.yml`
- **Shellcheck SC2010.** Checking the library name with `ls | grep` was replaced with a loop over the two names.
- **The first design had no way to show why a program was missing.** If the compile step failed, the run step would have found no binary and printed only "No such file". A program that was not built now leaves the compile log in its place.

### `benchmarks.yml`
- **The dropdown had no package that has benchmarks.** `midman-bench` is added to it and to the dropdown in `ci.yml`, and is now the default instead of `workspace`.
- **Missing `platforms` input, and no runner name in the summary.** Both are added, with the matrix, runner-scoped cache keys and per-runner artifact names.
- **A broken parser would only have shown after a full benchmark run.** The parser self-test now runs first.

### `train-smoke.yml`
- **First version had no `baseline_note` or `platforms` input, no diagnostics, and embedded an unstripped log tail.** All four are fixed to match `benchmarks.yml`.

### `data_smoke_summary.py`
- **Gitleaks read the first result file as secrets.** The JSON had `"tokenizer": "<64 hex digits>"` and one `"shard-00000.mmtokens": "<64 hex digits>"` entry per shard, and the generic rule of gitleaks takes a key with `token` in its name and a high-entropy value for a secret. Committed once, those lines would have failed the Security workflow, which scans the whole history, for good. It was found by running the pinned gitleaks over the new files before any commit. The vocabulary digest is now under the key `vocabulary`, the shard digests are a plain list, and the shard names are covered by the combined digest.

### `data-smoke.yml`
- **Written to avoid two portability traps.** A summary script that put a backslash escape inside an f-string expression fails on Python before 3.12, and `mapfile` does not exist in the bash 3.2 of the macOS runner. The script computes such values first, and the workflow lists the corpus files into a plain variable.
- **Open:** the macOS and ARM legs and the `compare` job have not run on GitHub, so whether the three runners really produce the same bytes is not yet confirmed. The first run with `platforms: all` is the test.

### `deploy-site.yml`
- **A scratch directory collided with a tracked one.** The first draft fetched results into `data/`, which is the repository's own tracked directory (it holds `data/README.md`). It uses `site-data/` now, and the scratch directories are in `.gitignore`.
- **Shellcheck style note.** Listing files by piping `ls` was replaced with `find`.
