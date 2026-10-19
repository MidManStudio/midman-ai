# midman-checkpoint

## Overview
Saving and loading model weights as standard safetensors files, with a reader and writer written in this crate and helpers that save and load a model's parameters by name.

Status: weights only. A checkpoint holds one `f32` tensor per model parameter plus string metadata, and loading is strict and all or nothing. Not built yet: optimizer state, training progress (step, tokens seen, random generator state), version migrations, the `F16` and `BF16` dtypes, and reading without loading the whole file into memory.

Dependencies: `midman-foundation` and `midman-tensor`. The tests also use `midman-model`, `midman-model-config` and `midman-nn`. The crate has no external dependencies, including for the JSON in the file header.

Limits:
- A file is read into memory whole, and its values are then copied into `f32` vectors, so loading needs about twice the file size for a moment. There is no memory mapping or streaming.
- Only the `F32` dtype exists. A file with any other dtype is refused by name, not converted.
- Names and layouts are MidMan's own. Linear weights are `[in, out]`, the transpose of PyTorch, and the rotary embedding rotates halves; see `docs/midman-nn.md`. A checkpoint is readable by any safetensors reader, but giving it to a PyTorch model would need those conversions, which do not exist and are untested.

## Modules

### `lib.rs`
**What it does:** Crate root. Declares the modules and re-exports `TensorFile`, `NamedTensor`, and the parameter helpers with their constants.

### `json.rs`
**What it does:** A small JSON reader and string writer, only as much as a safetensors header needs. The reader accepts objects, arrays, strings and non-negative integers, and rejects everything else.

**Decisions:**
- It is written here, not taken from a crate, because the foundational crates stay standard-library only.
- It is strict on purpose. Booleans, `null`, negative numbers, fractions, exponents, leading zeros, duplicate keys, trailing commas, trailing data, control characters in strings and unpaired surrogates are all errors, so a corrupt or hostile header is not half understood. Nesting deeper than 8 levels is refused; a header needs 3.
- The writer escapes only what JSON requires (quote, backslash, control characters, with `\b` and `\f` and `\n` `\r` `\t` as short escapes) and writes other text, including non-ASCII, as it is. These are the escapes the reference library writes, which is what makes the output byte-identical to it.

**Tests:** Seven tests: the shapes a header uses, empty containers and the largest integer, every escape including surrogate pairs and raw multi-byte text, twenty-three kinds of rejected input, duplicate keys and the exact nesting boundary, error messages that name the header and the byte, and the writer round trip.

### `safetensors.rs`
**What it does:** `TensorFile`, the contents of a safetensors file (named `f32` tensors and string metadata), with `push`, `get`, `to_bytes`, `write_to`, `save`, `from_bytes` and `load`. A file is an 8-byte little-endian header length, a JSON header, then raw little-endian data. The header maps each name to its dtype, shape and byte range, and may carry a `__metadata__` object of strings.

**Decisions:**
- It is the standard format, so other tools can read a checkpoint. Two checks against the reference `safetensors` package (version 0.8.0, installed from PyPI): a real Tiny checkpoint written by this crate (5.3 MB, 38 tensors, 1,315,968 values) loads in the reference reader with the same names, shapes and metadata, the same sums in 64-bit arithmetic and the same first value of every tensor; and for the fixtures the writer produces the same bytes as the reference writer.
- The writer sorts tensors by name and pads the header with spaces to a multiple of eight bytes. The reference sorts by dtype and then name, which is the same thing for one dtype. The reference writes several metadata entries in an arbitrary order (the fixture came out as `beta`, `alpha`, `gamma`), so byte-identical output can only be promised for at most one metadata entry. This crate writes them sorted by key, so its own output is deterministic.
- The reader checks everything the format promises before it trusts it: the length prefix, the 100 MB header limit, that the header starts with `{` and is valid UTF-8 and a JSON object, every shape against its byte range, and that the ranges tile the data with no gap, overlap or trailing bytes. Allocation is bounded by the length of the file, never by what the header claims; a header that claims a billion elements over four bytes of data is refused before anything is allocated.
- It is strict about fields. An entry with a field other than `dtype`, `shape` and `data_offsets`, or a `__metadata__` value that is not a string, is an error. This could refuse a file written by some tool that adds fields; no such tool is known.
- Reading orders tensors by their position in the data. Tensors with a zero-length range (a shape with a zero in it) are allowed, as in the format.
- `save` writes to `<path>.tmp`, flushes and syncs it, then renames it over the target, so a crash leaves the old file or the new one. The containing directory is not synced, so surviving a power cut at the moment of the rename is not guaranteed.
- Errors use the shared type: `Parse` for malformed content, `Io` for failures with the path in the message, `Shape` and `InvalidArgument` for bad calls to `push`.

**Tests:** Seventeen tests in the file. Bit-exact round trips including NaN payloads, signed zeros, infinities and subnormals; names, shapes, scalars, empty tensors and Unicode; the exact header text and padding, including a walk over every header length modulo eight; the empty file; reading in data order and with any whitespace padding; twenty-four kinds of malformed file, each rejected; overlap with a total that still fits; a header that lies about size; the limit, and the message at the limit; a writer that fails partway; saving, replacing, a missing directory and a missing file.

### `params.rs`
**What it does:** `parameters_to_file`, `save_parameters`, `check_header` and `load_parameters`, which turn a model's `Parameter`s into a checkpoint and back, matching by name.

**Decisions:**
- A checkpoint carries `format = midman-checkpoint` and `checkpoint_version = 1` in its metadata. The writer owns those two keys and refuses a caller who sets them.
- Loading is strict and all or nothing. Every parameter must be in the file with the same shape, the file must hold nothing the model lacks, two parameters must not share a name, and the header check must pass. All of that is checked before any value is copied, so a failed load changes nothing. The errors name the offending tensors, up to five and then a count.
- Only version 1 is accepted. A file with an older or newer version is refused with the unsupported-version error. Migrations will replace this when a version 2 exists.
- Matching is by name, not by position, so the order of parameters does not matter and a renamed layer shows up as one missing and one unexpected tensor.

**Tests:** Nine tests in the file: the metadata written, the reserved keys, duplicate names on save and load, loading in any order, a failed load changing nothing, a shape mismatch naming both shapes, extra tensors, the name list cut after five, and the header check on foreign, missing, wrong, older and newer files.

### `tests/reference_files.rs`
**What it does:** Tests against files written by the reference `safetensors` package, stored in `tests/fixtures/`.

**Tests:** Nine tests. The reference file with five tensors (scalar, empty, special values, a matrix) is read with every value compared bit for bit, and rewritten to the same bytes. Names and metadata with quotes, backslashes, tabs, newlines and non-ASCII text read back and rewrite byte for byte. The empty file matches both ways. Several metadata entries read back whatever their order. A file with an `F16` tensor is refused naming it. No prefix of a file is accepted, a single flipped bit anywhere in two files never panics (about 5,000 variants), and a trailing byte is rejected.

**Fixtures:** The `.hex` files in `tests/fixtures/` are hex dumps of real reference files, because the repository does not allow weight files (`repo_hygiene.py` refuses the `.safetensors` extension). `generate_fixtures.py` writes them and says how to rerun it; the values in it must match the ones in the test. The committed files came from `safetensors` 0.8.0 and `numpy` 2.4.4.

### `tests/model_roundtrip.rs`
**What it does:** The end-to-end test with a real model.

**Tests:** Four tests. A Smoke model saved to a file and loaded into a differently seeded one produces the same logits, bit for bit, on 32 tokens, where before loading the two differ. The checkpoint holds exactly the model's parameter count. A Smoke checkpoint is refused by a Tiny model, which is left unchanged. Saving the same model twice gives identical bytes.

## CI and Workflows
`ci.yml` can select this crate by name. Nothing else runs it specially. It has not been run on GitHub yet.

## Fixes and Problems

### `json.rs`
- **Reviewing the mutants found the nesting boundary untested.** A limit written as `depth > MAX_DEPTH + 1` would have passed every test, because the test nested two levels past the limit. The test now nests exactly one past it, and the mutant is caught.

### `safetensors.rs`
- **A mutation test found aligned headers untested.** Padding written as `8 - len % 8` instead of `(8 - len % 8) % 8` adds eight spaces when a header is already aligned. None of the fixtures happened to be aligned, so every test passed. A test now walks the header through every length modulo eight.
- **A mutation test found the header limit untested.** A limit that was twice too high still gave an error, because the file was too short, so the test could not tell which check stopped it. The test now checks the message, and checks that a header exactly at the limit is stopped for a different reason.
- **The reference does not order several metadata entries.** Its output for three entries was `beta`, `alpha`, `gamma`. The test for several entries therefore compares content, and the claim of byte-identical output is limited to one metadata entry.
- **Open: the cleanup of the temporary file after a failed write has no test.** The failures that can be produced in a test happen before the file exists. The cleanup is a best-effort `remove_file`.
- **Open: no directory sync, no `F16`, no `BF16`, no memory mapping.** See the limits above.

### `params.rs`
- **Reviewing the mutants found older versions untested.** A check written as `found > CHECKPOINT_VERSION` would have accepted version 0 and passed the tests, which only tried a newer version. The header test now tries version 0 as well, and the mutant is caught.

### Testing method
The crate was mutation-tested: thirty-three plausible bugs were injected one at a time (a wrong padding, no sorting, accepting a gap, an overlap or trailing bytes, not checking a shape against its range, accepting other dtypes, a missing start-byte check, big-endian in the reader and in the writer, accepting duplicate or reserved names, not checking value counts, accepting unknown fields, loosened limits, no rename on save, loose metadata types, wrong surrogate arithmetic, accepting leading zeros, duplicate keys, deep nesting, control characters, wrapped overflow, trailing data, unescaped quotes, loose version checks, a format name that is not compared, and accepting extra tensors, missing tensors, shape mismatches, reserved keys and duplicate parameter names). Two survived the runs (aligned header padding, and the header limit) and two gaps were found by reading the mutant list before running it (the nesting boundary and older versions); each led to the test described above, and all thirty-three mutants are now caught.
