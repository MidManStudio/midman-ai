# midman-data

## Overview
Dataset ingestion, source manifests, filtering, deduplication, packing, sampling and batch construction.

Status: the first part, which decides what text may be used and cleans it. A license policy that reads SPDX expressions, a provenance manifest, a secret scrubber and exact deduplication, with the SHA-256 they need. Not built yet: ingestion (fetching sources), quality filters (very long lines, binary and generated files), near-duplicate detection, packing documents into training sequences, sampling and batch construction. The Cargo spelling of a license (`MIT/Apache-2.0`) is not read; only SPDX expressions are.

Dependencies: `midman-foundation`, for the shared `Error`. There are no external crates and the tests use no other crate.

The one decision here that is the user's: the data license policy is permissive plus weak copyleft (the MPL and LGPL families), chosen over permissive only and over any license. Everything else in this document, including which licenses count as permissive and every threshold of the secret scrubber, is the project's own default and has not been confirmed.

The intended order for a document is: the license check on its source (through the manifest), then scrubbing, then deduplication, then tokenizing and packing. Scrubbing comes before deduplication so that two documents that differ only by a secret count as the same document; `tests/pipeline.rs` shows it.

Limits:
- A source is judged by one license string for the whole source. The manifest does not know about files inside a repository that carry another license, and nothing checks that the string is true. It comes from wherever the source was listed.
- The scrubber finds the shapes listed under `scrub.rs`, not secrets in general. It misses a secret in a shape it does not know, one split over lines or encoded again (a token in base64, say), and a password written in prose. It works on `&str`, so text has to be valid UTF-8 before it gets here.
- It cannot tell a real key from the made-up one in documentation, so documentation placeholders are replaced too. That is measured below and is the intended behavior: a wrongly replaced string costs a token, a missed key costs more.
- Exact deduplication compares whole documents byte for byte, through their SHA-256. A trailing newline or a different line ending makes a different document. It keeps 32 bytes per distinct document in memory plus the set's own overhead, and the set is not saved between runs.
- Everything works on one document at a time on one thread.

## Modules

### `lib.rs`
**What it does:** Crate root. Declares the five modules and re-exports their public items: `LicensePolicy`, `LicenseExpr`, `Verdict` and the two license lists; `Manifest`, `Source`, `Partition`, `Rejected` and the version constant; `Scrubber`, `Scrubbed`, `Finding` and `REDACTION`; `ExactDedup`; `sha256`, `Sha256` and `to_hex`.

### `hash.rs`
**What it does:** `Sha256`, an incremental SHA-256 (`new`, `update`, `finalize`), the one-call `sha256`, and `to_hex`.

**Decisions:**
- It is written from FIPS 180-4 because the foundational crates stay standard-library only and the standard library has no hash. It is SHA-256, not a faster hash, because exact deduplication treats equal digests as equal documents, which needs a hash for which an accidental collision is not a risk, and because Python's `hashlib` gives an independent answer for every message length tried.
- The message length is counted in bytes in a `u64` and used modulo 2^64 bits, as the standard says.
- Speed was measured once on the sandbox CPU (a virtualized Xeon, one noisy core), so read it as a rough figure: 64 MiB in 0.28 s, about 235 MB/s on one thread, with no SHA instructions used. At that speed 100 GB takes about seven minutes. It is not tuned and has no criterion benchmark yet, so there is no benchmark record; any intrinsics would have to follow `platform-optimization.md` like the matrix kernels.

**Tests:** Seven tests in the file. The NIST messages (empty, `abc`, the 448-bit and 896-bit ones) and one million `a` bytes; fifteen lengths around the points where padding needs one or two blocks, each fed to the hasher split at every possible position; many small updates, empty updates, a cloned hasher continuing on its own, and the hex form.

### `dedup.rs`
**What it does:** `ExactDedup`, which remembers the SHA-256 of every document it has accepted. `insert` and `insert_digest` return whether the document is new; `contains` asks without recording; `unique`, `duplicates` and `checked` count.

**Decisions:**
- The set stores whole 32-byte digests and compares all 32 bytes, but hashes only the first eight, because a digest is already uniformly distributed and hashing it again would be wasted work. A test inserts digests that share their first eight bytes and checks that they stay distinct.
- A digest computed elsewhere can be inserted directly, so a pipeline that already hashed a document does not hash it twice.
- There is no removal and no saving to disk yet.
- With 2 KB documents it runs at about 225 MB/s, which is the speed of the hash.

**Tests:** Seven tests in the file: first sight and repeats with the counters, the empty document, one byte of difference (trailing newline, CRLF, case), `contains` not recording or counting, a digest and its content being the same key, 20,000 documents with 10,000 repeated, and digests with a shared prefix.

### `license.rs`
**What it does:** `LicenseExpr`, a parsed SPDX license expression (`parse`, and `Display` that writes it back with the parentheses it needs); `LicensePolicy`, the set of allowed licenses (`empty`, `permissive_and_weak_copyleft`, `allow`, `allowed_ids`, `check`, `check_expr`); `Verdict` (`Accepted` or `Rejected` with a reason); and the two lists `PERMISSIVE` and `WEAK_COPYLEFT`.

**Decisions:**
- The grammar is SPDX 2.3: identifiers, `ID+`, `ID WITH EXCEPTION`, `AND`, `OR` and parentheses, with `WITH` binding tighter than `AND` and `AND` tighter than `OR`. The operators must be in capitals, which is what the specification requires; a lowercase `or` is an error that says so. Identifiers match without regard to case. They are not looked up in the SPDX list: the policy decides what is allowed, and an unknown identifier is simply not on the list. Input is limited to 1,024 bytes and 16 levels of parentheses, so hostile text is an error and not a stack overflow.
- A syntax error is an `Err`; a valid expression for a license that is not allowed is `Verdict::Rejected`. That keeps "this source is badly described" apart from "this source may not be used".
- The default policy is `PERMISSIVE` plus `WEAK_COPYLEFT`. Weak copyleft means exactly what the user named, the MPL and LGPL families (MPL-1.0, 1.1, 2.0 and 2.0-no-copyleft-exception; LGPL 2.0, 2.1 and 3.0 in their `-only` and `-or-later` forms). The EPL, CDDL and similar licenses are weak copyleft too but were not named, so they are not in the list; `allow` adds them. Which licenses count as permissive is the project's list of 25: 0BSD, Apache-2.0, BlueOak-1.0.0, BSD-1-Clause, BSD-2-Clause, BSD-3-Clause, BSD-3-Clause-Clear, BSL-1.0, CC0-1.0, curl, HPND, ISC, Libpng, MIT, MIT-0, NCSA, PSF-2.0, Python-2.0, Unicode-3.0, Unicode-DFS-2016, Unlicense, UPL-1.0, WTFPL, X11 and Zlib. All 35 identifiers are current, non-deprecated entries of SPDX license list 3.29.0, which a test checks.
- The old GNU spellings are read as the SPDX list defines them: `LGPL-2.1` alone is `LGPL-2.1-only` and `LGPL-2.1+` is `LGPL-2.1-or-later`, and likewise for the GPL, AGPL and GFDL. They are real entries of the official list, so they matter. For any other identifier a trailing `+` is ignored.
- `A OR B` is accepted if either is, `A AND B` only if both are. An exception (`WITH`) never turns a license that is not on the list into one that is: `Apache-2.0 WITH LLVM-exception` is accepted because Apache-2.0 is, `GPL-2.0-only WITH Classpath-exception-2.0` is rejected because GPL-2.0-only is not. A test runs every official license with a spread of exceptions to check that an exception changes nothing.
- `NOASSERTION` and `NONE` are rejected with their own reason ("no license was identified"). A `LicenseRef-...` is rejected unless the policy allows it by name, which is how in-house data gets in.

**Tests:** Twenty-two tests in the file. Single identifiers, `+`, `WITH` and references; precedence and parentheses; `Display` reading back as the same expression for ten expressions; many kinds of malformed input; the capital-letter error; the exact size and depth limits (1,024 bytes pass, 1,025 do not; depth 16 passes, 17 does not; a thousand open parentheses is an error); the default list accepted entry by entry; sixteen strong-copyleft or unknown licenses rejected; case; the old GNU spellings against policies that list only one form; `+` on other identifiers; `OR` and `AND` over mixed parts; exceptions; `NOASSERTION` and `NONE`; the wording of rejection reasons; allowing a reference; the empty policy; and the canonical form of the allowed list. Six more tests in `tests/spdx_list.rs`.

### `manifest.rs`
**What it does:** `Manifest`, the sources that make up a corpus, and `Source`, one entry: `id`, `url`, `revision`, `license`, `fetched` and `category`. `push` checks and adds a source; `get`, `sources`, `len` and `is_empty` read; `partition` splits the sources by a license policy into `accepted` and `rejected` (with the reason); `to_text` and `from_text` write and read the file format; `save` and `load` do it through a file.

**Decisions:**
- A manifest is always valid. `push` refuses a source whose field breaks its rule or whose id is already there, and the reader goes through the same check, so there is no way to hold or load an invalid one.
- The rules: the id is up to 128 characters of letters, digits and `. _ / -`; the URL looks like `scheme://...` with no spaces and at most 512 bytes; the revision is a commit id, 40 or 64 lowercase hex digits; the license parses as an SPDX expression; the fetch date is a real calendar date from 1970, written `YYYY-MM-DD`; the category is up to 32 characters of `a-z 0-9 _ -`. No field may be empty, hold a control character, or have a space at either end. Requiring a commit id and not a branch name is the project's choice: a name like `main` is not provenance, because it points somewhere else tomorrow.
- The text format is a header line `midman-data-manifest 1`, then for each source a blank line and six `key: value` lines in a fixed order. The reader is strict: it rejects a wrong header, a version other than 1 (with the unsupported-version error), a missing final newline, a carriage return, a blank line in the wrong place, a missing, repeated, unknown or misordered key, and an empty value, and it names the line. The sources keep the order they were added in, so a manifest file diffs cleanly.
- The project's own data (terminal sessions made here, say) uses the license `LicenseRef-in-house`, which passes only in a policy that allows that name.
- `partition` would treat a license that no longer parses as a refusal with the parse error as the reason; this cannot happen for a manifest built through `push`.
- `save` writes `<path>.tmp`, flushes and syncs it, then renames it over the target, as the checkpoint writer does. The containing directory is not synced.

**Tests:** Thirteen tests in the file. `push`, `get` and `len`; a duplicate id adding nothing; every kind of bad field value refused by field name; the edge values that are accepted (a leap day, the century leap rules, a 64-digit revision, the exact 128, 512 and 32 limits, a `file://` URL, a reference as license); the exact text shape and round trips; the empty manifest; the reader refusing a wrong frame and wrong records, each with the line number; `partition` over six licenses; the in-house reference; and saving, loading, atomic replacement, and I/O errors that name the path and leave no temporary file.

### `scrub.rs`
**What it does:** `Scrubber`, which finds secrets in a `&str` (`find`, returning `Finding`s with a rule name and byte offsets) and replaces them with `<redacted>` (`scrub`, returning `Scrubbed`). `without_generic` turns off the rule most likely to catch something that is not a secret.

**The rules.** Each secret has a rule name; the ones for tokens with a fixed prefix need the prefix at the start of a word and at least a minimum number of characters after it, and replace the whole run:
- `github-token`: `ghp_`, `gho_`, `ghu_`, `ghs_`, `ghr_` and 36 letters or digits; `github_pat_` and 50 more.
- `gitlab-token`: `glpat-` and 20. `aws-access-key-id`: `AKIA`, `ASIA`, `ABIA`, `ACCA` and 16 capitals or digits 2 to 7.
- `slack-token`: `xoxb-`, `xoxa-`, `xoxp-`, `xoxr-`, `xoxs-`, `xapp-` and 20. `slack-webhook`: the hooks URL and 20.
- `stripe-key`: `sk_live_`, `sk_test_`, `rk_live_`, `rk_test_` and 10. `google-api-key`: `AIza` and 35.
- `anthropic-key`: `sk-ant-` and 20. `openai-key`: `sk-proj-`, `sk-svcacct-`, `sk-admin-` and 20, or `sk-` and 40 letters or digits.
- `npm-token`: `npm_` and 36. `pypi-token`: `pypi-AgEI` and 50. `sendgrid-key`: `SG.` and 60. `huggingface-token`: `hf_` and 30.
- `jwt`: two parts that start `eyJ` and have at least 17 characters each, then a dot and a signature that may be empty.
- `private-key`: from a BEGIN line whose label contains `PRIVATE KEY` through the matching END line, including keys whose line breaks are written as the two characters `\n`.
- `url-password`: only the password in `scheme://user:password@host`; the user may be empty.
- `auth-header`: the credential in `Authorization: Bearer`, `Basic` or `Token`, if it is at least 16 characters of token text.
- `generic-secret`: the value after `=`, `==`, `===`, `:`, `:=` or `=>` when the key contains a keyword (`secret`, `password`, `passwd`, `pwd`, `token`, `apikey`, `api_key`, `api-key`, `access_key`, `accesskey`, `private_key`, `privatekey`).

**Decisions:**
- It is written by hand because there is no regex crate here. It makes two passes over the bytes: one for the fixed-prefix tokens, the JWT, the private key and the URL password, one over identifiers for the header and keyword rules.
- It leans toward finding too much. A run longer than the minimum is replaced whole, and a longer match at the same place beats a shorter one, so a quoted value that holds a token plus more is replaced entirely. Where rules overlap, the earliest start wins, then the longer match, then a rule for a known shape over the generic one.
- Only the secret is replaced, with `<redacted>`, so the rest of the line is kept and a scrubbed file still reads like code. The replacement itself matches no rule, so scrubbing is idempotent; a randomized test checks that, and that the findings are sorted, do not overlap and sit on character boundaries.
- The generic rule is the one with thresholds: a value of 16 to 256 characters with a digit and a letter and a Shannon entropy of at least 3.5 bits per character (pinned by tests on both sides: 3.45 is refused, 3.58 accepted). It skips values that look like paths (a leading `.` or `~`, or a leading `/` together with any of `. - _ ~` or no capital letter), URLs and templates (`://`, `${`, `{{`, `<%`) and lowercase snake_case names. An unquoted value that is followed by `( . [ :` or `{`, or by `, ) ] }`, is code (a call, a field, an argument or parameter list), so a type annotation like `private_key: SomeType,` is left alone. A comparison is read like an assignment, because a secret compared against in code is written in the code.
- `<redacted>` and not a rule-specific marker, so the tokenizer learns one short pattern. The finding keeps the rule name for the counts.
- A private key body is searched for up to 16 KiB; a key whose END line is missing is replaced through its run of base64-looking text. A mention of the BEGIN line in documentation does not swallow what follows: the body must start with a run of 40 base64 characters (or a `Proc-Type:` header) and hold 64 in all.

**Measured, against gitleaks 8.30.1 (the version `security.yml` pins) on a made-up corpus.** `tests/fixtures/oracle_corpus.py` writes 288 samples, 12 of each of 24 kinds, with random values. The scrubber found all 288 and nothing outside them. Gitleaks found 274 findings and covered every kind except URL passwords (0 of 12); it also missed one of 12 AWS secret keys and one of 12 quoted generic secrets, which the scrubber found. The scrubber found everything gitleaks did on this corpus. The corpus is built from the kinds the scrubber was written for, so this shows agreement on those shapes and says nothing about the roughly two hundred other services gitleaks knows.

**Measured, false positives, on text that should hold no secrets.** Three sets: the user's repositories plus the skills repository (6,014 files, 48.4 MB), the source of the crates in the Cargo registry (2,337 files, 28.1 MB), and the Python standard library and installed packages (1,702 files, 23.4 MB). The Cargo registry sources gave 0 findings. Of the 95 in the repositories, none is in midman-ai, mid-engine, mdix-scaffold or ubel_stratum; 4 are in DixScript-Rust (a test password in a fixture and three placeholder URLs), and the rest are in the skills repository's documentation and tests: documented placeholders such as AWS's published example key, `FAKE` keys, `mySecretPassword123`, and `postgres:postgres@` URLs. Python gave 12, all example values from OAuth specifications and tests, plus a placeholder password in a URL. None of them is code that was wrongly changed, and the scrubber cannot tell a placeholder from a real key; that is the limit named above. One genuine false positive turned up during the work, a parameter annotation, and is fixed (see Fixes and Problems).

**Measured, speed.** On the same sandbox CPU, one thread, release build: 81 MB/s on the repositories, 99 MB/s on the Rust sources, 87 MB/s on the Python sources; 100 GB would take about 20 minutes. A profile (callgrind, a sample of 400 files) before the last change showed 48% of all instructions in `memcmp`, from comparing every prefix rule at every word start; a table of the bytes a rule can start with removed that. Before it the speed was 27 MB/s. The findings were identical before and after on all three sets of files and on the corpus.

**Mutation testing:** 80 injected bugs across the five modules in the final run. The first run found 12 that no test caught. One was an equivalent change (see Fixes and Problems for `hash.rs`), one was a condition that was simpler to delete, and ten were real gaps: exact boundaries (a 1,024-byte expression, a 512-byte URL, the 30-day months, the 16 KiB key limit), the longer-match rule, `::` paths, quoted paths that start with a dot, `:/` against `://`, `Bearer:`, and the key label characters. Tests were added for them; the final run caught 79 of 80, and the survivor is the equivalent one.

**Tests:** Thirty-six tests in the file. Every prefix rule finds a token of its minimum length in three surroundings and refuses one character fewer; longer runs; the word boundary (what may and may not come before a prefix); the specific-before-general order of `sk-` prefixes; webhooks; the JWT shapes; private keys of five labels, with escaped line breaks, with CRLF, with PEM headers, without an END line, and with the documentation mentions that must not match; passwords in URLs and the template forms that must stay; authorization headers and their negatives; fifteen assignment syntaxes (comparisons included) and twenty-five lines of ordinary code that must stay; the exact length limits of the JWT, the credential, the key and the generic rule; the entropy threshold on both sides; arguments, parameters and paths; overlap resolution; offsets and ordering; idempotence; a scan of this crate's own sources, which mention every prefix and keyword and must produce no finding; and 400 random mixtures of secret fragments and multi-byte text checked for panics, boundaries, overlap, preserved text and idempotence. Every test secret is assembled at run time from pieces, and every PEM line from a constant, so that no line of this repository looks like a credential to a scanner: gitleaks reads the whole history, so one committed line that matches would fail the Security workflow for good.

### `tests/reference_vectors.rs`
**What it does:** SHA-256 against digests made by Python's `hashlib`.

**Tests:** Three tests. `fixtures/sha256_vectors.txt` holds a digest for the messages of every length from 0 to 300 and for 511, 512, 513, 1,000, 4,096, 65,535, 65,536 and 100,003 bytes, 309 in all. The messages are rebuilt here from a formula (byte `j` of a message of length `n` is `(31 j + 7 n + 5) mod 256`) that `fixtures/generate_fixtures.py` shares, so the messages are not stored. Every digest matches in one call and when the message is fed in pieces of 1, 3, 63, 64, 65 and 1,000 bytes.

### `tests/spdx_list.rs`
**What it does:** The parser and the default policy against the official SPDX list.

**Tests:** Six tests. `fixtures/spdx_ids.txt` (740 licenses with their deprecated flag) and `fixtures/spdx_exceptions.txt` (86 exceptions) are written from SPDX's `license-list-data`, release 3.29.0, by `generate_fixtures.py`. Every official identifier and exception parses, alone, with `+`, with an exception and inside a compound expression. The default list holds only current identifiers. The default policy accepts exactly the 35 listed licenses plus the six deprecated LGPL spellings and nothing else out of the 740. No identifier of the GPL, AGPL, GFDL, SSPL, EUPL, OSL, non-commercial, no-derivatives or similar families is accepted. An exception changes the verdict for none of the 740 licenses.

### `tests/pipeline.rs`
**What it does:** The parts together.

**Tests:** Two tests. A manifest of four sources is written, read back and partitioned: the MIT-or-Apache and the MPL sources are used, the GPL one and the unlicensed one are refused with reasons. Then scrubbing before deduplication: two documents that differ only by a token become one, the document that was seen twice is counted, and the token is in none of the kept text.

### `examples/scan.rs`
**What it does:** Runs the scrubber over files and reports. `lines FILE` prints `line`, rule and the start of the match for each finding; `files` does the same for the paths on standard input and ends with totals; `speed` times the scan, best of five; `--no-generic` turns off the generic rule. Matches are cut to 48 characters, so a report never holds a whole secret. It is what produced the comparison, false-positive and speed figures above.

**Fixtures:** `tests/fixtures/generate_fixtures.py` writes the SHA-256 vectors and, given the SPDX data directory, the two SPDX files, and says how to rerun it. `tests/fixtures/oracle_corpus.py generate DIR` writes the secret corpus and `compare DIR GITLEAKS_JSON OURS_TSV` prints the table of kinds; its header gives the gitleaks and `scan` commands. The corpus is not committed.

## CI and Workflows
`ci.yml` can select this crate by name. Nothing else runs it specially. It has not been run on GitHub yet. `security.yml` runs gitleaks over the repository history with the pinned version; the new files were checked with that version and report nothing.

## Fixes and Problems

### `scrub.rs`
- **The module's first name was refused.** It was `secrets.rs`, and `scripts/repo_hygiene.py` rejects any tracked file that matches `secrets.*` as a credential file. The module was renamed instead of the script's rule being loosened.
- **Slow prefix check.** The first version compared every prefix rule at every word start with a library call, and a profile put 48% of all instructions there; the speed was 27 MB/s. A table of the bytes a rule can start with brought it to 81 to 99 MB/s with identical findings. Not tuned further.
- **A secret that starts with `/`.** The generic rule first rejected every value that began with `/` as a path. About one real AWS secret key in 64 starts with `/` (base64 has it in its alphabet), and the made-up corpus caught it. The rule now treats a leading `/` as a path only with one of `. - _ ~` in the value or no capital letter.
- **A parameter annotation caught as a secret.** `private_key: PKCS7PrivateKeyTypes,` in a Python package matched the generic rule. A first fix, skipping values that read like names, was dropped after measuring: on random 16-character text it misjudged 3% as names, and 7% in a tighter version, which loses real secrets. The fix is structural and costs no recall: an unquoted value followed by a comma or a closing bracket is an argument or a parameter, not a literal.
- **Gitleaks flagged five lines of the first tests.** PEM header and footer text written out in test strings, and a `secret:` field with a made-up value. The tests now build those from a dashes constant and `format!`. Because gitleaks reads history, this had to be right before the first commit.
- **The crate's own text matched its own rules.** The self-scan test caught URL examples in doc comments and in test strings; they were reworded or built at run time.
- **Comparisons were excluded.** `==` was first not read as an assignment, which would have missed a secret compared against in code. It is now, and `===` too.
- **One condition removed.** The check that a BEGIN line is not preceded by a dash did nothing useful and is gone.
- **Open:** the shape list has 21 rules; gitleaks has about two hundred. More services are added when the real corpus shows which matter. The 16 KiB key limit leaves the rest of a longer key unreplaced, which no real key reaches.

### `license.rs`
- Found by rereading before the first build: the identifier check had two identical branches and accepted an empty `LicenseRef-`; both fixed.
- Not handled: the legacy `MIT/Apache-2.0` slash spelling used by Cargo manifests. A source listed that way is a parse error until a reader for it exists.

### `hash.rs`
- Mutation testing: replacing `>=` with `>` in the full-block loop of `update` changes nothing observable. A full block is then kept in the buffer and compressed by the next call or by `finalize`, and the digest is the same. The mutant is equivalent, so no test can tell it apart, and it is the one survivor of the 80.

### `manifest.rs`
- Nothing found by use yet.
