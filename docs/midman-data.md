# midman-data

## Overview
Dataset ingestion, source manifests, filtering, deduplication, packing, sampling and batch construction.

Status: the path from text to batches. A license policy that reads SPDX expressions, a provenance manifest, a secret scrubber and exact deduplication (with the SHA-256 they need) decide what text may be used and clean it; a packer turns it into token shards on disk with an end-of-text token after every document; and a batcher cuts the shards into training sequences and hands them out in a seeded order that a stopped run can resume. A first cut of ingestion lives next to the crate: `scripts/ingest.py` lists repositories, pins each to a commit and writes the manifest, and `examples/check_manifest.rs` applies the license policy to it and names the sources `ingest.py fetch` may take. The manifest of the corpus is `data/manifest.txt`; the `midman-ai Ingest` workflow reads it. Not built yet: the final source list, quality filters (very long lines, binary and generated files), near-duplicate detection, mixing sources by weight, fill-in-the-middle and terminal-session formatting, and a reader for data larger than memory. The Cargo spelling of a license (`MIT/Apache-2.0`) is not read; only SPDX expressions are.

Dependencies: `midman-foundation`, for the shared `Error` and `Rng`, and `midman-tokenizer`, for packing. There are no external crates. One test also uses `midman-model` and `midman-model-config`, to give a batch to the model.

Two decisions here are the user's. The data license policy is permissive plus weak copyleft (the MPL and LGPL families), chosen over permissive only and over any license. The special-token set is the one proposed, with `<|endoftext|>` ending every document (see `docs/midman-tokenizer.md`). Everything else in this document, including which licenses count as permissive, every threshold of the secret scrubber, and the way documents are packed and batched, is the project's own default and has not been confirmed.

The order for a document is: the license check on its source (through the manifest), then scrubbing, then deduplication, then tokenizing and packing into shards; training then reads the shards through the batcher. Scrubbing comes before deduplication so that two documents that differ only by a secret count as the same document; `tests/pipeline.rs` shows it, and `examples/build_shards.rs` runs the whole path over files.

Limits:
- A source is judged by one license string for the whole source. The manifest does not know about files inside a repository that carry another license, and nothing checks that the string is true. It comes from wherever the source was listed.
- The scrubber finds the shapes listed under `scrub.rs`, not secrets in general. It misses a secret in a shape it does not know, one split over lines or encoded again (a token in base64, say), and a password written in prose. It works on `&str`, so text has to be valid UTF-8 before it gets here.
- It cannot tell a real key from the made-up one in documentation, so documentation placeholders are replaced too. That is measured below and is the intended behavior: a wrongly replaced string costs a token, a missed key costs more.
- Exact deduplication compares whole documents byte for byte, through their SHA-256. A trailing newline or a different line ending makes a different document. It keeps 32 bytes per distinct document in memory plus the set's own overhead, and the set is not saved between runs.
- Everything works on one document at a time on one thread. A shard is read whole into memory: 100 million tokens is 200 MB on disk and 400 MB in memory.
- Documents are packed one after another and are not kept apart in a training sequence: a window can end one document and start the next, and attention is not masked at the boundary. Where a shard ends with fewer tokens than a whole window, those tokens are not used.
- Batches draw uniformly from all sequences. Mixing sources by weight (more Rust than shell, say) is not built; it needs the sources kept in separate shard sets.

## Modules

### `lib.rs`
**What it does:** Crate root. Declares the modules and re-exports their public items: `LicensePolicy`, `LicenseExpr`, `Verdict` and the two license lists; `Manifest`, `Source`, `Partition`, `Rejected` and the version constant; `Scrubber`, `Scrubbed`, `Finding` and `REDACTION`; `ExactDedup`; `sha256`, `Sha256` and `to_hex`; `TokenShard` and `SHARD_VERSION`; `Packer`; `Sequences`, `Batcher` and `Batch`.

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
**What it does:** `LicenseExpr`, a parsed SPDX license expression (`parse`, and `Display` that writes it back with the parentheses it needs); `LicensePolicy`, the set of allowed licenses (`empty`, `permissive_and_weak_copyleft`, `allow`, `allow_unlicensed`, `allows_unlicensed`, `allowed_ids`, `check`, `check_expr`); `Verdict` (`Accepted` or `Rejected` with a reason); and the two lists `PERMISSIVE` and `WEAK_COPYLEFT`.

**Decisions:**
- The grammar is SPDX 2.3: identifiers, `ID+`, `ID WITH EXCEPTION`, `AND`, `OR` and parentheses, with `WITH` binding tighter than `AND` and `AND` tighter than `OR`. The operators must be in capitals, which is what the specification requires; a lowercase `or` is an error that says so. Identifiers match without regard to case. They are not looked up in the SPDX list: the policy decides what is allowed, and an unknown identifier is simply not on the list. Input is limited to 1,024 bytes and 16 levels of parentheses, so hostile text is an error and not a stack overflow.
- A syntax error is an `Err`; a valid expression for a license that is not allowed is `Verdict::Rejected`. That keeps "this source is badly described" apart from "this source may not be used".
- The default policy is `PERMISSIVE` plus `WEAK_COPYLEFT`. Weak copyleft means exactly what the user named, the MPL and LGPL families (MPL-1.0, 1.1, 2.0 and 2.0-no-copyleft-exception; LGPL 2.0, 2.1 and 3.0 in their `-only` and `-or-later` forms). The EPL, CDDL and similar licenses are weak copyleft too but were not named, so they are not in the list; `allow` adds them. Which licenses count as permissive is the project's list of 25: 0BSD, Apache-2.0, BlueOak-1.0.0, BSD-1-Clause, BSD-2-Clause, BSD-3-Clause, BSD-3-Clause-Clear, BSL-1.0, CC0-1.0, curl, HPND, ISC, Libpng, MIT, MIT-0, NCSA, PSF-2.0, Python-2.0, Unicode-3.0, Unicode-DFS-2016, Unlicense, UPL-1.0, WTFPL, X11 and Zlib. All 35 identifiers are current, non-deprecated entries of SPDX license list 3.29.0, which a test checks.
- The old GNU spellings are read as the SPDX list defines them: `LGPL-2.1` alone is `LGPL-2.1-only` and `LGPL-2.1+` is `LGPL-2.1-or-later`, and likewise for the GPL, AGPL and GFDL. They are real entries of the official list, so they matter. For any other identifier a trailing `+` is ignored.
- `A OR B` is accepted if either is, `A AND B` only if both are. An exception (`WITH`) never turns a license that is not on the list into one that is: `Apache-2.0 WITH LLVM-exception` is accepted because Apache-2.0 is, `GPL-2.0-only WITH Classpath-exception-2.0` is rejected because GPL-2.0-only is not. A test runs every official license with a spread of exceptions to check that an exception changes nothing.
- `NONE` and `NOASSERTION` are different things and are told apart. `NONE` is what the listing says when it found no license at all (GitHub gives `license: null`); `NOASSERTION` is a license file the listing could not identify (GitHub gives the id `NOASSERTION`, as for a custom or unusual license text), which is not the same as having none and may be a restriction. By default both are rejected, `NONE` with "no license was identified" and `NOASSERTION` with "the license could not be identified".
- **2026-10-25, the project's rule (the user's decision):** a source with no license is fair game unless something in it specifically says otherwise. `LicensePolicy::allow_unlicensed` accepts `NONE`; `NOASSERTION` stays refused. The default policy is unchanged and the flag is off, so the library is conservative unless the caller asks; `check_manifest` asks, unless it is given `--require-license`. The second half of the rule, the look for a statement against use, is not in this crate: it needs the files of the source, so it is `ingest.py notices`. The flag is a choice about the corpus, not a statement of law: code without a license is not licensed by default.
- A `LicenseRef-...` is rejected unless the policy allows it by name, which is how in-house data gets in.

**Tests:** Twenty-three tests in the file. Single identifiers, `+`, `WITH` and references; precedence and parentheses; `Display` reading back as the same expression for ten expressions; many kinds of malformed input; the capital-letter error; the exact size and depth limits (1,024 bytes pass, 1,025 do not; depth 16 passes, 17 does not; a thousand open parentheses is an error); the default list accepted entry by entry; sixteen strong-copyleft or unknown licenses rejected; case; the old GNU spellings against policies that list only one form; `+` on other identifiers; `OR` and `AND` over mixed parts; exceptions; `NOASSERTION` and `NONE`, each with its own reason; `NONE` accepted only when the policy allows it, `NOASSERTION` never, inside `AND`, `OR` and `WITH`, and the flag leaving everything else alone; the wording of rejection reasons; allowing a reference; the empty policy; and the canonical form of the allowed list. Six more tests in `tests/spdx_list.rs`. Six changes to the unlicensed flag were each caught: ignoring it, having it on by default, accepting `NOASSERTION` with it, never setting it, not matching `NONE`, and the wording of the `NOASSERTION` refusal.

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

### `shard.rs`
**What it does:** `TokenShard`, a run of token ids: documents one after another, each followed by the end-of-text token. `new` checks and builds one; `vocab_size`, `eos`, `tokens`, `len`, `is_empty` and `token_width` read it; `to_bytes` and `from_bytes` write and read the file contents; `save` and `load` do it through a file.

**The file (version 1).** All integers are little endian. Eight bytes `MMTOKENS`; the version, the token width (2 or 4), the vocabulary size and the end-of-text id, four bytes each; the token count in eight bytes; the tokens, each in the stated width; and the SHA-256 of everything before it, 32 bytes. The header is 32 bytes. The files are named `.mmtokens`.

**Decisions:**
- There is one correct encoding. The width is 2 bytes when the vocabulary has at most 65,536 ids and 4 otherwise, and the reader insists on it, so equal shards are equal files, and the planned vocabularies of 4,096 and 32,000 ids cost two bytes per token.
- The header carries the vocabulary size and the end-of-text id, so a shard made with one tokenizer cannot quietly be trained on with another: `Sequences` refuses a mixture, and a model's configuration can be checked against the shard.
- The reader checks, in order: the length, the magic, the version, the vocabulary size, the width against the vocabulary, the end-of-text id, that the count and the width account for every byte of the file (counts are checked without overflow, so a count of 2^64-1 is an error and not an allocation), the checksum, and then that every id is below the vocabulary size. Any bit flipped anywhere is an error.
- `save` writes `<path>.tmp`, flushes and syncs it, then renames it over the target, as the manifest does.
- A shard is read whole. Reading by pieces or mapping it into memory is not built.

**Tests:** Nine tests in the file. The exact layout of a small shard, written out byte by byte from the specification; the widths on both sides of 65,536 ids; round trips for five vocabulary sizes up to 4,000,000,000 ids and for an empty shard; construction checks; every header field wrong in turn, each with its own message; every possible cut of a file, an extra byte, wrong counts including 2^64-1, 2^63-1 and 2^62; every one of the first and last bit of every byte flipped; a token outside the vocabulary under a correct checksum; and saving, loading, atomic replacement, I/O errors that name the path, and a damaged file on disk.

### `pack.rs`
**What it does:** `Packer` turns documents into shards. `new` takes the vocabulary size, the end-of-text id and a target shard size; `for_tokenizer` takes them from a tokenizer; `add_document` takes token ids and `add_text` takes text and a tokenizer; both return the shard they completed, if any; `finish` returns what is left; `documents` and `tokens` count.

**Decisions:**
- A document is its tokens followed by the end-of-text token. There is no start-of-document token.
- A shard closes at the end of the first document that brings it to the target size or past it, so documents are never cut and shards are a little larger than the target. A document longer than the target is a shard of its own. The last shard, from `finish`, is smaller.
- Empty documents are ignored and not counted.
- It is strict. An id outside the vocabulary, or the end-of-text id inside a document (it would end the document early), is an error and nothing from that document is kept.
- `add_text` always reads special-token text as ordinary text (`SpecialMode::AsText`), so a document that contains the characters `<|endoftext|>` cannot end itself or inject any control token; the test feeds it exactly that. It also refuses a tokenizer whose vocabulary or end-of-text differ from the packer's.
- `for_tokenizer` needs the tokenizer to have `<|endoftext|>`. A `Tokenizer::bytes_only()` vocabulary (the Smoke preset's 256) has none and is refused, so Smoke-size runs on real shards use a vocabulary that has the special tokens (264 ids at least).

**Tests:** Ten tests in the file. End-of-text after each document and shards closing at the target; a document longer than a shard; closing exactly at the target and not one token before; empty documents; refused documents changing nothing; construction checks; the counters adding up to what the shards hold; a packer made from a tokenizer; text unable to inject control tokens; and a tokenizer without end-of-text or a different one being refused.

### `batch.rs`
**What it does:** `Sequences`, the training windows in a list of shards, and `Batcher`, which hands them out in batches. `Sequences::window` returns a window of `seq_len + 1` tokens. `Batcher::next_batch` returns a `Batch` of `inputs` and `targets` (each `batch * seq` ids as `usize`, the shape `MidManModel::loss` takes, with the epoch and position it came from); `epoch`, `position` and `seek` save and restore where a run was.

**Decisions:**
- A sequence is a window of `seq_len + 1` consecutive tokens of one shard, starting every `seq_len` tokens. The first `seq_len` are the inputs and the last `seq_len` the targets, so each target is the token after its input. Neighboring windows share one token, so every token is a target exactly once, and no window crosses from one shard to the next.
- Each pass over the data visits the sequences in a fresh random order, `batch` at a time. The order of a pass depends only on the seed and the pass number (it is a shuffle made from `Rng::fork(epoch)` of a generator seeded with the seed), so a run that stopped is continued exactly by creating a batcher with the same seed and data and calling `seek(epoch, position)` with the two numbers it saved. A test stops a run after each of thirty batch counts, including the ends of passes, and checks the continuation against the uninterrupted run.
- When the number of sequences is not a multiple of the batch size, the few left over are skipped in that pass; a different few are skipped in the next, and another test checks that every sequence is used within forty passes.
- A pass is rolled over lazily, at the next call, so the state after the last batch of a pass is (that pass, all batches done) and `seek` to it behaves.
- The batcher is endless. There is no separate held-out set yet.

**Tests:** Thirteen tests in the file. Windows start every `seq_len` tokens and overlap by one; every token is a target exactly once and the remainder is dropped; the exact count at the edges; windows never crossing shards; no shards and bad arguments; targets being the next tokens in every row; one pass using each sequence at most once; every sequence being used over enough passes; the order depending on seed and pass alone; resuming after every stopping point; `seek` checks and going backwards; construction checks; and a batch of one sequence per pass.

**Mutation testing:** 45 bugs were injected into `shard.rs`, `pack.rs` and this file (every boundary, counter and check named above, the shuffle's seed and epoch, `seek`, the shift between inputs and targets, and `AsText`); all 45 were caught on the first run.

### `tests/reference_vectors.rs`
**What it does:** SHA-256 against digests made by Python's `hashlib`.

**Tests:** Three tests. `fixtures/sha256_vectors.txt` holds a digest for the messages of every length from 0 to 300 and for 511, 512, 513, 1,000, 4,096, 65,535, 65,536 and 100,003 bytes, 309 in all. The messages are rebuilt here from a formula (byte `j` of a message of length `n` is `(31 j + 7 n + 5) mod 256`) that `fixtures/generate_fixtures.py` shares, so the messages are not stored. Every digest matches in one call and when the message is fed in pieces of 1, 3, 63, 64, 65 and 1,000 bytes.

### `tests/spdx_list.rs`
**What it does:** The parser and the default policy against the official SPDX list.

**Tests:** Six tests. `fixtures/spdx_ids.txt` (740 licenses with their deprecated flag) and `fixtures/spdx_exceptions.txt` (86 exceptions) are written from SPDX's `license-list-data`, release 3.29.0, by `generate_fixtures.py`. Every official identifier and exception parses, alone, with `+`, with an exception and inside a compound expression. The default list holds only current identifiers. The default policy accepts exactly the 35 listed licenses plus the six deprecated LGPL spellings and nothing else out of the 740. No identifier of the GPL, AGPL, GFDL, SSPL, EUPL, OSL, non-commercial, no-derivatives or similar families is accepted. An exception changes the verdict for none of the 740 licenses.

### `tests/pipeline.rs`
**What it does:** The parts together.

**Tests:** Two tests. A manifest of four sources is written, read back and partitioned: the MIT-or-Apache and the MPL sources are used, the GPL one and the unlicensed one are refused with reasons. Then scrubbing before deduplication: two documents that differ only by a token become one, the document that was seen twice is counted, and the token is in none of the kept text.

### `tests/ingest_manifest.rs`
**What it does:** Reads a manifest written by `scripts/ingest.py` with this crate's reader and judges it with the license policy. `tests/fixtures/ingest_manifest.txt` is a copy of `scripts/fixtures/ingest-manifest.txt`: four sources (hashbrown, syn, memchr, unicode-ident) from a real run of `ingest.py crates` and `ingest.py manifest` on 2026-10-25, each a repository at a commit id with the license crates.io lists for it.

**Decisions:** The script has its own writer of the manifest format, so there are two writers and one reader. This test is what keeps them from drifting apart: the Rust reader must accept the script's file, and the Rust writer must give the same bytes back.

**Tests:** Five tests. The reader accepts the file and finds the four ids in order with category `rust`, the fetch date, a 40-digit revision and a GitHub URL; writing the manifest again gives the file byte for byte; the policy accepts all four licenses, among them `(MIT OR Apache-2.0) AND Unicode-3.0` and `Unlicense OR MIT`; the same file with one license changed to `GPL-3.0-only` is valid and has exactly that source refused; and the same file with one license changed to `NONE` has that source refused by the default policy ("no license was identified") and accepted by the policy that allows unlicensed sources, while `NOASSERTION` is refused by both ("the license could not be identified").

### `examples/scan.rs`
**What it does:** Runs the scrubber over files and reports. `lines FILE` prints `line`, rule and the start of the match for each finding; `files` does the same for the paths on standard input and ends with totals; `speed` times the scan, best of five; `--no-generic` turns off the generic rule. Matches are cut to 48 characters, so a report never holds a whole secret. It is what produced the comparison, false-positive and speed figures above.

**Fixtures:** `tests/fixtures/generate_fixtures.py` writes the SHA-256 vectors and, given the SPDX data directory, the two SPDX files, and says how to rerun it. `tests/fixtures/oracle_corpus.py generate DIR` writes the secret corpus and `compare DIR GITLEAKS_JSON OURS_TSV` prints the table of kinds; its header gives the gitleaks and `scan` commands. The corpus is not committed.

### `tests/shards_end_to_end.rs`
**What it does:** The whole path from text to a model's loss, and a larger check of packing.

**Tests:** Two tests. The first scrubs and deduplicates a corpus of forty distinct functions, ten exact repeats and four scripts that differ only by a token; trains a vocabulary with the eight agreed special tokens; packs into shards of at least 400 tokens; writes them to disk and reads them back; checks that the stream is exactly the kept documents, in order, each ended by end-of-text, with no secret in it; cuts sequences, checks every batch (ids inside the vocabulary, targets being the next tokens); and gives a batch to a Smoke-size model, whose loss at the start has to be within 0.5 of the natural log of the vocabulary. The second packs 2,000 documents of random length (some empty) into shards of 5,000 tokens and checks that they come back exactly, that every closed shard ends with end-of-text and has at least 5,000 tokens, and that the sequence count is what the formula gives.

### `examples/build_shards.rs`
**What it does:** A command-line tool that does the whole data path over files: `build_shards --tokenizer FILE --out DIR [--shard-tokens N] [--seq-len N] [--no-scrub] < list-of-paths`. Each path is a document. A file that is not valid UTF-8, and one that is empty, is skipped and counted. The rest is scrubbed, compared by hash with the documents already kept (the first copy wins), tokenized and written as `shard-00000.mmtokens`, `shard-00001.mmtokens` and so on. It then reads every shard back, which checks its checksum, and reports the number of sequences of `--seq-len` tokens and a SHA-256 of the first 100 batches the batcher makes from them (batches of 4, seed 1; each id as four little-endian bytes, inputs then targets), which is the same on every platform if the whole path is. It refuses a directory that already holds shards. It does not check licenses: pass it only the paths of sources the manifest and the policy accepted.

**Run on real files.** The files were this project's neighbors on the sandbox: the user's repositories, the skills repository and the source of the crates in the Cargo registry, 8,424 files. A vocabulary of 4,096 ids with the eight special tokens was trained on 7,571 of them (92 MB; the other 853 held out by a hash of the path) with `midman-tokenizer`'s `train`: 3,832 merges, which with 256 bytes and 8 special tokens is exactly 4,096, in 22 seconds. On the held-out files it reached 2.02 bytes per token, but six of them are binary files (2.2 MB) that the shard builder skips; on the 845 UTF-8 files it reached 2.99. `build_shards` on all of the files, with shards of 5 million tokens and sequences of 256: 73 files not UTF-8, 6 empty, 169 duplicates after scrubbing, 96 secrets replaced (59 URL passwords, 17 generic, 8 authorization headers, 4 each of Anthropic keys and AWS key ids, 2 GitHub tokens, 1 each of OpenAI and Stripe; one of the 59 is the example URL in this document's own source, which the scrubber reads as a password), 8,176 documents kept, 75.6 MB, 25.7 million tokens at 2.94 bytes per token, in six shards (five of about 10.0 MB and one of 1.3 MB, two bytes per token) holding 100,319 sequences. It took 7.4 s, 10.3 MB/s for the whole path on one thread, which is mostly the tokenizer (about 11 MB/s on its own, with no cache). These are figures for whatever text was on this machine, not for the training corpus: that is not chosen yet, and the production vocabulary has to be trained on it.

**Does a model learn from it?** A one-off check outside the repository: a model of the Smoke preset's size with a vocabulary of 4,096 ids, trained for 300 steps on batches of 8 sequences of 128 tokens from these shards, through the batcher. Its loss began at 8.325, the natural log of 4,096 being 8.318, and fell to a mean of 6.3 over the last 49 steps. That shows that shards, batcher and model fit together, and nothing about model quality.

### `examples/check_manifest.rs`
**What it does:** `check_manifest MANIFEST [--accepted FILE] [--require-license]`. Reads the manifest with `Manifest::load`, which refuses a file that is not valid and names the line, and judges every source with the permissive-plus-weak-copyleft policy, plus the project's rule that a source with no license (`NONE`) is accepted. `--require-license` takes that rule away, so `NONE` is refused like any license that is not allowed. A license that exists but could not be identified (`NOASSERTION`) is refused either way. The report on standard output has one line per source, `accepted` or `rejected`, with id, category and license (and the reason for a refusal); the totals by category go to standard error, with a line saying how many of the accepted have no license. With `--accepted FILE` the accepted ids are written there, one per line, in manifest order. The exit status says whether the manifest was valid, not what the policy decided.

**Decisions:** The rule about unlicensed sources is on by default here because it is the project's rule, and the library's own default stays strict. The policy lives only in this crate. `scripts/ingest.py` records the license a listing gives and decides nothing about it, so there is no second copy of the lists to keep in step. The accepted list is a plain file of ids because `ingest.py fetch` reads exactly that.

**Tests:** None of its own. The same calls are tested in `manifest.rs`, `license.rs` and `tests/ingest_manifest.rs`; the run on the 77-source manifest of 2026-10-25 (66 accepted, 5 of them with no license, and 11 refused: 6 for licenses not on the list, 5 for a license that could not be identified) was checked by hand.

## CI and Workflows
`ci.yml` can select this crate by name and runs its unit tests and clippy. `data-smoke.yml` runs the command-line path (`train`, `build_shards` and `scan` over the repository's own files), compares the scrubber with gitleaks on x86_64 Linux, and with `platforms: all` checks that the vocabulary, every shard and the digest of the first batches are byte for byte the same on the three runners; its result is published on the site's Tests page. Neither has been run on GitHub yet. `security.yml` runs gitleaks over the repository history with the pinned version; the new files were checked with that version and report nothing.

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
- **Open:** the shape list has 18 rules (13 of them for tokens with a fixed prefix, 32 prefixes in all); gitleaks has about two hundred. An earlier version of this line said 21, which was a miscount. More services are added when the real corpus shows which matter. The 16 KiB key limit leaves the rest of a longer key unreplaced, which no real key reaches.

### `license.rs`
- Found by rereading before the first build: the identifier check had two identical branches and accepted an empty `LicenseRef-`; both fixed.
- Not handled: the legacy `MIT/Apache-2.0` slash spelling used by Cargo manifests. A source listed that way is a parse error until a reader for it exists. `scripts/ingest.py` rewrites that spelling to `MIT OR Apache-2.0` before it writes a manifest, so the reader never sees it.

### `shard.rs`
- **Open:** a shard is read whole into memory, so a shard of 100 million tokens needs 400 MB. Reading by pieces or mapping the file is not built, and shards are kept below that size for now.

### `pack.rs`
- **Open:** there is no start-of-document token and documents are not kept apart in a sequence, so a window can hold the end of one document and the start of the next. Masking attention at the boundary is not built and has not been chosen.

### `batch.rs`
- **Open:** batches draw uniformly from every sequence, so the mix of sources is whatever the shards hold; weighting needs the sources in separate shard sets. There is no held-out split yet.

### `hash.rs`
- Mutation testing: replacing `>=` with `>` in the full-block loop of `update` changes nothing observable. A full block is then kept in the buffer and compressed by the next call or by `finalize`, and the digest is the same. The mutant is equivalent, so no test can tell it apart, and it is the one survivor of the 80.

### `manifest.rs`
- Nothing found by use yet.
