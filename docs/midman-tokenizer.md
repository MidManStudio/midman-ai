# midman-tokenizer

## Overview
A byte-level byte-pair-encoding (BPE) tokenizer, written from scratch: a rule-based pre-tokenizer, a trainer that learns merges from text, and a tokenizer that encodes, decodes, handles special tokens and saves itself as a small text file. Every byte string encodes and decodes exactly, whether or not it is valid UTF-8, because every byte has its own token.

Status: the machinery is built and tested, and the settings of the pre-tokenizer were chosen by measurement on a mixed corpus (see Measurements). Not built yet: the production vocabulary, which has to be trained on the real corpus from `midman-data` before it is fixed; special-token templates for chat and agent turns; any speed work (no cache, no threads); benchmarks in `midman-bench`; an exporter to other tools' tokenizer formats.

Dependencies: `midman-foundation`, for the shared error type. Nothing else.

How it fits the model presets: `ModelConfig::vocab_size` must equal `Tokenizer::vocab_size()`, which counts the 256 bytes, the merges and the special tokens. The Smoke preset's 256 is `Tokenizer::bytes_only()`. The Tiny preset's 4,096 and the larger presets' 32,000 are what `TrainerConfig::new` is given, with any special tokens taken out of the merge budget (eight special tokens leave 3,832 merges for a 4,096 vocabulary).

Limits:
- The trainer is single-threaded and keeps one entry per distinct piece of the corpus in memory, not the corpus itself. Memory use was not measured.
- The encoder has no cache and no threads. It encodes about 11 MB/s on the sandbox's one noisy CPU.
- No piece is longer than 128 bytes. A longer run (a base64 blob, a minified line) is cut into pieces of that size.
- Where text is split depends on the Unicode tables built into the Rust standard library, and a saved tokenizer does not record which version. A newer Rust could classify a newly assigned character differently.
- There is no streaming decoder: decoding a token at a time can produce half of a multi-byte character, so decode whole sequences.

## Measurements
The pre-tokenizer's settings decide what merges can exist, so eight combinations were trained and compared before a default was chosen. This is the "measure before the vocabulary is fixed" step for the shell and terminal data.

**The corpus** (66.4 MB, built in the sandbox and not committed; 90 percent of files for training and 10 percent, chosen by a hash of the file path, for testing):

| Category | Held-out docs | Training bytes | What it is |
| --- | --- | --- | --- |
| rust | 248 | 29.0 MB | crates from the Cargo registry (a random 18 MB sample) and the Rust files of the MidMan, Mid Engine, DixScript and Ubel repositories |
| diff | 124 | 10.5 MB | commits from `git log -p` of the five repositories |
| docs | 117 | 8.7 MB | Markdown from the repositories, license texts and package docs |
| code_other | 81 | 6.1 MB | JavaScript, TypeScript, Python, C, HTML, CSS and C# from the repositories |
| shell | 107 | 1.9 MB | shell scripts from the system, bash completions, repository scripts and the `run:` blocks of workflow files |
| terminal | 21 | 1.7 MB | command sessions generated here: `ls`, `git`, `grep`, `cargo build` and `clippy` on code with deliberate errors, compiler errors from `gcc` and Python, `--help` pages, `dpkg`, `ps`, with 110,000 real ANSI escape sequences |
| config | 54 | 2.0 MB | YAML, TOML, JSON and `.conf` files |
| ubel | 30 | 0.2 MB | the 260 `.ubl` files and samples in the Ubel repository |

For training, shell was repeated 4 times, terminal 6, config 2 and ubel 10, so that small categories count for something. For the single score ("weighted") each category's bytes per token were weighted 28 percent rust, 12 diff, 12 docs, 10 code_other, 12 shell, 14 terminal, 6 config and 6 ubel. Both the repeats and the weights are my choices, not measured from any real usage.

**The eight settings** (a vocabulary of 32,000 was trained for each, and smaller vocabularies are its first merges):

| Name | ansi | underscore in words | prefix | digit run | punct absorbs newlines | whitespace |
| --- | --- | --- | --- | --- | --- | --- |
| V0 | no | no | space | unlimited | no | gpt2 |
| V1 | no | no | any one | 3 | yes | cl100k |
| V2 | yes | yes | space | unlimited | no | gpt2 |
| V3 | yes | yes | any one | 3 | yes | cl100k |
| V4 | yes | yes | any one | 1 | yes | cl100k |
| **V5 (chosen)** | yes | yes | any one | 3 | yes | gpt2 |
| V6 | yes | yes | space | 3 | yes | cl100k |
| V7 | yes | no | any one | 3 | yes | gpt2 |

**Bytes per token on held-out files** (higher is better):

| | V0 | V1 | V2 | V3 | V4 | V5 | V6 | V7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| weighted, vocab 4,096 | 2.773 | 2.862 | 2.879 | 2.946 | 2.911 | **2.962** | 2.902 | 2.961 |
| weighted, vocab 32,000 | 3.437 | 3.745 | 3.650 | 3.883 | 3.788 | **3.915** | 3.709 | 3.892 |
| terminal only, vocab 32,000 | 2.532 | 2.774 | 3.286 | 3.612 | 3.394 | 3.628 | 3.314 | 3.607 |

The chosen setting's full table:

| vocab | rust | diff | docs | code_other | shell | terminal | config | ubel | weighted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4,096 | 3.133 | 2.885 | 2.379 | 2.994 | 2.964 | 3.006 | 3.264 | 3.020 | 2.962 |
| 8,192 | 3.473 | 3.301 | 2.812 | 3.413 | 3.363 | 3.269 | 3.789 | 3.314 | 3.335 |
| 16,384 | 3.761 | 3.646 | 3.283 | 3.797 | 3.734 | 3.472 | 4.206 | 3.530 | 3.663 |
| 32,000 | 3.972 | 3.900 | 3.738 | 4.067 | 4.024 | 3.628 | 4.484 | 3.656 | 3.915 |

What the numbers say (at 32,000 unless stated):
- **ANSI escape sequences matter most for terminal text.** Treating each sequence as one piece takes terminal text from 2.774 to 3.612 bytes per token (V1 to V3, 30 percent), and a colored rustc error message from 69 tokens to 36.
- **Letting a word take any one character before it beats space only by 4.5 percent** (V3 against V6, which differ in nothing else). Paths such as `/usr`, flags such as `--release` and fields such as `.name` become one piece each.
- **Cutting numbers into single digits costs 2.4 percent** (V4 against V3). It may help a model learn arithmetic; that was not measured, and the cost is for the user to weigh. The default cuts every three digits.
- **Joining a line break with the next line's indentation beats ending the run at the last line break by 0.8 percent** (V5 against V3), and **keeping underscores inside words gains 0.6 percent** overall, 1.7 percent on Rust and shell about 1 percent, but loses 1.2 percent on prose docs (V5 against V7). These two are small.
- **Vocabulary size:** from 4,096 to 32,000 the weighted score rises from 2.962 to 3.915, about 32 percent. Each doubling gives less: 12.6 percent for 4,096 to 8,192, 9.8 percent for 8,192 to 16,384.
- **Scored separately on the even and odd held-out documents, the order of the settings did not change:** V5 3.914 and 3.916, V7 3.889 and 3.894, V3 3.882 and 3.885, V4 3.796 and 3.784, V1 3.722 and 3.768, V6 3.706 and 3.712. The halves share the training run, so this shows the test files are not the cause, not that another corpus would agree. Differences under 1 percent should be read as ties.

Tokens used by single lines at 32,000 with the chosen setting (a line of N bytes costs about N/4 tokens in prose, less in code): a colored compile line 22 tokens for 90 bytes, a path 20 for 71, a URL 22 for 79, a timestamped log line 34 for 72, a Rust function header 37 for 118, a `cargo test` command line 26 for 101, a diff header 69 for 217, an indented 8-space block 22 for 104, but a git hash 40 tokens for 48 bytes, a UUID 24 for 40, a hex dump line 50 for 78 and a line of numbers 31 for 56. Hashes and numbers do not compress; that is their content, not a flaw of the tokenizer.

Caveats. The Rust text is mostly crates.io code, not a sample of GitHub. The terminal sessions were generated here, are heavy on `ls`, and are not real developer sessions. The shell sample is small. The Ubel held-out set is 30 files and 37 KB. Each setting was trained once. **The production vocabulary must be trained again on the real corpus before it is fixed.**

Speed in the sandbox (one virtualized Xeon, noisy): reading and counting 78 MB took 1.4 to 1.8 seconds, learning 31,744 merges took 2.3 to 4.0 seconds (the settings with ANSI handling have the most distinct pieces, about 290,000, and took 3.4 to 4.0), and encoding the 6.2 MB held-out set took 0.50 to 0.58 seconds, which is where the 11 MB/s comes from. These are single runs, not benchmarks.

## Proposed special tokens
Not decided, and nothing in the code depends on this list; the trainer takes any list. For discussion, the set suggested by the work this project is meant to do:
- `<|endoftext|>`: the end of a document or sequence.
- `<|pad|>`: padding in a batch.
- `<|fim_prefix|>`, `<|fim_middle|>`, `<|fim_suffix|>`: fill-in-the-middle code completion.
- `<|command|>`, `<|output|>`, `<|exit_status|>`: a terminal session, where the exit status is written as text after its marker.

Chat and agent turn markers (system, user, assistant, tool call, tool result) are left out until the format is chosen. Every token added takes one id from the merge budget of a fixed-size vocabulary, so a later addition either needs a retrained embedding table or room reserved now.

## Modules

### `lib.rs`
**What it does:** Crate root. Declares the modules and re-exports the public types: `Tokenizer`, `SpecialMode`, `Trainer`, `TrainerConfig`, `TrainStats`, `PreTokenizerConfig`, `WordPrefix`, `WhitespaceRule`, `Pieces`, and the constants `MAX_PIECE_BYTES`, `FIRST_MERGE_ID` and `FORMAT_VERSION`.

### `pretokenize.rs`
**What it does:** Splits text into pieces that merges never cross. `PreTokenizerConfig::pieces` returns an iterator of slices that concatenate back to the input exactly, none empty and none over 128 bytes.

At each position the first of these rules that matches decides the piece:
1. An ANSI escape sequence (`ESC [`, parameter bytes `0-?`, intermediate bytes ` -/`, a final byte `@-~`), or a lone `ESC`, when `ansi` is on.
2. A word: letters, and `_` when `underscore_in_words` is on, with an optional prefix. The prefix is one space, or with `WordPrefix::AnyOne` any one character that is not a letter, digit or line break.
3. Digits, at most `digit_run` at a time (0 for no limit), with an optional space prefix under `WordPrefix::Space`.
4. A run of punctuation and symbols with an optional space prefix, and with `punct_newlines` the line breaks right after it.
5. Whitespace: under `WhitespaceRule::Gpt2` a run is one piece except that its last character is left to prefix the next word, so a line break and the indentation after it are one piece; under `Cl100k` a run containing a line break ends at its last line break.

A letter is a Unicode alphabetic character, a digit is an ASCII digit, whitespace is the Unicode `White_Space` property, and a byte that is not valid UTF-8 counts as punctuation, one byte at a time.

**Decisions:**
- The rules are the GPT-2 and `cl100k_base` patterns reduced to six switches, so that variants can be trained and compared on data instead of argued. The switches are stored in a saved tokenizer, because encoding must split text the way the vocabulary was trained.
- The default, `PreTokenizerConfig::code_shell()`, is V5 from the measurements. `gpt2_like()` and `cl100k_like()` are kept as named baselines.
- There is no rule for English contractions (`'s`, `'t`), which both reference patterns have. This is a tokenizer for code and terminal text; what it costs prose was not measured.
- Whitespace characters other than space, tab, line breaks, vertical tab and form feed (no-break space, ideographic space) are whitespace that cannot prefix a word.
- ANSI handling covers control sequences (CSI) only. Operating system commands (`ESC ]` ...), such as terminal titles, are a lone `ESC` followed by ordinary text.
- The 128-byte limit bounds the work the trainer and encoder do per piece. The cut moves back by up to three bytes so it does not fall inside a multi-byte character, and stays put in bytes that are not UTF-8.
- Digits are ASCII only; other Unicode digits count as symbols.

**Tests:** Sixteen unit tests: the GPT-2 rules, the line-break and indentation rules, the `cl100k` rules, ANSI sequences on and off, an incomplete escape, underscores, digit runs, Unicode classes, invalid UTF-8, vertical tab and form feed, the default, the cut at the limit and at character boundaries (including a four-byte letter, which needs the full three-byte step back), determinism, and that for all 96 combinations of settings the pieces always concatenate to the input. Independently, the committed fixtures hold 410 cases that an unrelated implementation cut the same way (see `tests/reference_oracle.rs`).

### `tokenizer.rs`
**What it does:** `Tokenizer`, a vocabulary with `encode`, `encode_bytes`, `decode`, `decode_bytes`, special-token handling, `bytes_only`, and the saved text format (`to_text`, `from_text`, `save`, `load`).

Ids are in three blocks. Ids 0 to 255 are the bytes, so a small id is its own byte. Merge `r` joins two earlier tokens into token `256 + r`. The special tokens come last.

**Decisions:**
- Encoding a piece applies the earliest-learned merge that applies anywhere in it, to every non-overlapping occurrence from left to right, and repeats. That is exactly the order the trainer applied the merges, so encoding text the tokenizer was trained on gives the same pieces the trainer ended with (a test checks this for every distinct piece).
- `SpecialMode::AsText` encodes special-token text as ordinary bytes, and `Recognize` turns it into the token. Use `AsText` for training data and anything from outside, so that text reading `<|endoftext|>` cannot inject a control token; use `Recognize` for text the program builds. At one position the longest special token wins.
- Special tokens come last so that merges keep their ids whatever specials are added; the cost is that a special token's id depends on the merge count.
- A special token may not be empty, repeated, contain a line break or start or end with whitespace, because the saved format is one line per token.
- The saved format is plain text: a first line naming the format and version, a line of pre-tokenizer settings, one `special` line per special token and one `merge` line per merge. It is strict: unknown or repeated settings, blank lines, specials after merges and merges that use ids that do not exist yet are errors with the line number. A different version is the unsupported-version error. A text format was chosen so it can be read and diffed in a pull request from a phone.
- Two merges can produce the same bytes by different routes (`a` + `bc` and `ab` + `c`). They are kept, as in the reference algorithm, so a few ids can be redundant.
- Token ids are `u32`.

**Tests:** Fourteen unit tests: a bytes-only tokenizer, merge order, repeated pairs, merges not crossing piece boundaries, special tokens (last ids, recognition, edges, overlaps, longest match), decoding and its error, lossy text decoding against exact bytes, construction errors, the exact saved layout and its round trip, twenty-two kinds of malformed file with line numbers, a future version, saving to a file, and a random test over 40 random vocabularies and 1,000 random inputs that compares the encoder with a deliberately slow obvious one and checks that decoding returns the input.

### `trainer.rs`
**What it does:** `Trainer` counts the distinct pieces of the text it is given (`add`, or `add_counted` for pieces already counted), and `finish` learns merges from the counts alone and returns the `Tokenizer` with `TrainStats`. `TrainerConfig` holds the vocabulary size, pre-tokenizer settings, special tokens and the minimum frequency.

Each step merges the most frequent adjacent pair, with the count taken at every position, so in `aaa` the pair `aa` counts twice although only one merge can apply. Ties go to the pair with the smaller left id and then the smaller right id. Pair counts are kept exact by recounting each word a merge touches, found through an index from pairs to words, with a priority queue of pairs whose stale entries are corrected when they surface.

**Decisions:**
- The result does not depend on the order text was added, on hash order or on the number of runs; a test trains with the documents in two orders and twice. The tie rule is what makes this so.
- The minimum frequency is 2 by default. A corpus that runs out of pairs seen that often stops early, and `TrainStats::stopped_early` says so; the vocabulary is then smaller than asked.
- The vocabulary size counts the bytes, merges and special tokens together, and is an error if it cannot hold the bytes and the specials.
- `TrainStats::merge_counts` records each merge's count, which never increases; a last merge seen only a handful of times means the corpus was too small for the vocabulary.
- It is single-threaded, and uses the standard library's hash map with its default hasher. Counting shards in parallel and combining them with `add_counted` would work without changing the algorithm; that was not needed at this scale.

**Tests:** Twelve unit tests, among them the textbook example (low, lower, newest, widest) whose eight merges and counts were worked out by hand, determinism, early stopping, the minimum frequency, special tokens in the budget, configuration errors, counts that never increase, training text that round-trips, more merges never giving more tokens, the encoder reproducing the trainer's final segmentation of every piece, and the overlapping-pair case (`aaaaa`).

### `examples/train.rs`
**What it does:** A command-line tool: `cargo run --release -p midman-tokenizer --example train -- --vocab 32000 --out tokenizer.txt [--special TEXT]... [--min-frequency N] [--pretokenizer default|gpt2|cl100k] [--eval PATH]... PATH...`. Every file under the training paths is a document. It trains, saves the tokenizer, prints how many merges it learned and whether it stopped early, and, for the evaluation paths, prints bytes per token and counts files that fail to encode and decode back to their own bytes.

**Decisions:** Directories are walked in sorted order with dotfiles and symbolic links skipped, so the same inputs give the same tokenizer. It exists so the user can train a tokenizer from repositories in a workflow without writing code.

**Tests:** None in code. Run by hand on this repository's own files: a 4,096 vocabulary with two special tokens learned 3,838 merges from 394 KB, and scored 3.076 bytes per token on 13 held-out files with no round-trip failures.

### `tests/reference_oracle.rs`
**What it does:** Checks the Rust code against an independent implementation, `tests/fixtures/reference.py`, which has a regular-expression pre-tokenizer and the textbook BPE loop that recounts every pair at every step, and shares no code or structure with this crate.

**Tests:** Four tests. The pre-tokenizer cuts all 410 golden cases the same way, over 8 of the settings (edge cases such as lone and incomplete escapes, runs over the 128-byte limit including cuts inside two-, three- and four-byte characters, Unicode whitespace, control characters, and random mixtures). Training on the 5 KB corpus in `tests/fixtures/bpe_corpus.txt` learns the same 300 merges as the reference, and encodes eight held-out strings to the same ids, under two settings, one without and one with ANSI handling.

**Fixtures:** `reference.py` writes the fixtures; run it from `tests/fixtures/` with `pip install regex`. The corpus file writes the escape character as the four characters `\x1b`, restored on reading. The committed files came from `regex` 2026.9.29. Both sides use Unicode properties, so the fixtures keep to long-stable characters.

Outside CI, the same reference was compared with the Rust code at a much larger scale: 118,368 pre-tokenizer cases (random mixtures of every character class, ANSI variants, runs over the limit, and 14 real files) over all 96 combinations of settings with no mismatch; and training on 425 KB of real text from five repositories for five settings with 400 merges each, plus one setting with 1,500 merges, gave identical merges and identical encodings of eleven held-out strings, in 11 to 44 milliseconds against 7 to 30 seconds for the reference.

## CI and Workflows
`ci.yml` can select this crate by name. Nothing else runs it specially. It has not been run on GitHub yet. There are no benchmarks for it in `midman-bench` yet.

## Fixes and Problems

### `pretokenize.rs`
- **Mutation testing found two weak tests.** Two of 35 injected bugs survived the first run. A cut that moves back by at most two bytes instead of three passed, because my first four-byte test used an emoji, which is not a letter and so forms its own piece, never straddling the cut; a four-byte letter is needed. Treating the vertical tab as punctuation passed, because in the case I had written it split the same way either way; it needs a punctuation mark next to it. Both tests were fixed and the mutants are caught.
- **Open: contractions, operating system commands and Unicode versions.** See the decisions and limits above.

### `tokenizer.rs`
- **Open: no cache, no threads.** The encoder does the work for every piece every time. Words repeat a great deal in real text, so a cache is likely the first speed-up, but it has not been measured.
- **Open: no streaming decode, and redundant ids.** See the decisions above.

### `trainer.rs`
- **Open: the window-count quirk.** Overlapping occurrences of one pair are counted at every position though only some can merge. The reference algorithm counts the same way, which is why the two agree, but it can rank a pair slightly above its real count in text with long runs of one symbol.
- **Open: memory and the hash function.** Neither was measured, and the stale entries in the pair-to-word index are never pruned.

### Testing method
Three kinds of check, besides the unit tests. The independent reference above. Mutation testing: thirty-five plausible bugs were injected one at a time (ANSI byte ranges, the piece limit and the cut, whitespace rules, newline and digit rules, character classes, prefix rules, merge order, special-token order and filtering, decoding errors, version and specials validation, duplicate and impossible merges, tie order, the minimum frequency, index registration, statistics, the merge step, and configuration checks) and, after the two tests above were strengthened, all are caught. And an interoperability measurement on a real corpus, which is where the defaults come from.
