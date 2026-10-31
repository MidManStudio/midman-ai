# MidMan AI

> A coding and game-development model, built and trained from scratch in Rust.

MidMan AI is a Rust workspace for a coding and game-development model, the pipeline that trains it, and an agent runtime around it. Nothing here starts from pretrained weights or a machine-learning framework: the tensors, the automatic differentiation, the kernels, the tokenizer and the data pipeline are written in this repository. Rust and [Ubel Stratum](https://github.com/MidManStudio/ubel_stratum) are the core code domains, shell and terminal material is in scope, and the model is meant to be useful for [Mid Engine](https://github.com/Mid-D-Man/mid-engine) work.

It is early. A first small model (the Tiny preset, 1,315,968 parameters) has been trained on a sample of the real corpus on GitHub's machines: 4.1 million tokens, with a held-out loss of 3.914 against 8.318 for a model that knows nothing. It is a test of the path, not a useful model yet. The table below says what exists.

---

## Crates

Status as of 2026-10-31. Working means it has code, tests and a document in `docs/`. In progress means it works but has known gaps listed in its document. Planned means the crate is an empty stub.

| Crate | Role | Status |
|---|---|---|
| `midman-foundation` | Shared error type and a seeded random number generator | 🟢 Working |
| `midman-model-config` | Model architecture description, exact parameter counts, presets from `smoke` (117,056) to `large` (1,009,338,368) | 🟢 Working |
| `midman-tensor` | CPU `f32` tensors, reverse-mode autograd, matrix products (with AVX kernels behind a Cargo feature) | 🟢 Working |
| `midman-nn` | Linear, embedding, RMSNorm, rotary embeddings, causal attention, SwiGLU, decoder block | 🟢 Working |
| `midman-model` | The decoder-only model: forward pass and loss | 🟢 Working |
| `midman-optim` | AdamW (its state can be saved and restored), warmup-and-cosine schedule, gradient clipping | 🟢 Working |
| `midman-checkpoint` | Weights in the standard safetensors format; the state of a training run is stored in the same format | 🟢 Working |
| `midman-tokenizer` | Byte-level BPE with a trainer, special tokens and a code-and-shell pre-tokenizer | 🟢 Working |
| `midman-data` | License policy, source manifest, secret scrubber, exact deduplication, token shards, batcher | 🟡 In progress (ingestion works on 480 sources; no quality filters or near-duplicate detection yet) |
| `midman-api` | Safe Rust API and a C ABI, also compiled to WebAssembly for the website | 🟢 Working |
| `benches/midman-bench` | Benchmarks with plain C twins as baselines | 🟢 Working |
| `midman-training` | The training loop: steps, held-out loss, a state that can be saved and resumed | 🟡 In progress (one thread; no gradient accumulation) |
| `midman-quant`, `midman-inference` | Quantization, inference | 🔵 Planned |
| `midman-agent`, `midman-tools`, `midman-sandbox` | Agent runtime, tools, restricted execution | 🔵 Planned |
| `midman-ubel`, `midman-game` | Ubel Stratum and game-engine integration | 🔵 Planned |
| `apps/midman-cli`, `apps/midman-server`, `apps/midman-trainer` | Command line, server, trainer | 🔵 Planned |

Each crate has a document in `docs/` that holds what it does, the decisions behind it, its benchmarks and tests, and a list of fixes and open problems by file.

## Crate dependency order

```
midman-foundation    (no workspace dependencies)
midman-model-config  (uses foundation)
midman-tensor        (uses foundation)
midman-tokenizer     (uses foundation)
midman-nn            (uses foundation, tensor)
midman-optim         (uses foundation, tensor)
midman-checkpoint    (uses foundation, tensor)
midman-model         (uses foundation, model-config, nn, tensor)
midman-data          (uses foundation, tokenizer)
midman-training      (uses foundation, tensor, nn, model, model-config, optim, checkpoint, data)
midman-api           (uses foundation, model, model-config, nn, optim, tensor)
```

`midman-api` is also what the website runs in the browser.

## Getting started

Any recent stable Rust toolchain builds the workspace. The benchmark package needs Cargo 1.85 or newer; the other crates are also tested on Rust 1.75, per crate with `-p`.

```bash
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo fmt --all -- --check
python3 -m unittest discover -s scripts -p 'test_*.py'
```

The training path can be seen on the smallest size: a model that must learn one fixed batch. That test is ignored by default because it is slow in a debug build:

```bash
cargo test --release -p midman-model --locked -- --ignored
```

Train a tokenizer, build token shards, and check a source manifest:

```bash
cargo run --release -p midman-tokenizer --example train -- \
    --vocab 4096 --out tokenizer.txt --special '<|endoftext|>' training-dir

cargo run --release -p midman-data --example build_shards -- \
    --tokenizer tokenizer.txt --out shards/ < list-of-paths

cargo run --release -p midman-data --example check_manifest -- \
    data/manifest.txt --accepted accepted.txt
```

Train a model on those shards (the state is saved every `--save-every` steps, `--minutes` stops on a time budget, `--resume` continues a stopped run):

```bash
cargo run --release --features avx -p midman-training --example train -- \
    --shards shards/train --eval-shards shards/held --out run-out \
    --preset tiny --steps 2000 --batch 8 --seq 256
```

The commands of the ingestion script are listed at the top of `scripts/ingest.py`.

## The training corpus

The corpus is built from public repositories listed in [`data/manifest.txt`](data/README.md): each source is pinned to a commit, so the corpus can be rebuilt exactly. The rules:

- A source is used if its license is on the allow list: permissive licenses (MIT, Apache-2.0, BSD and the like) and weak copyleft (MPL and LGPL). The lists are in `crates/midman-data/src/license.rs`.
- A source with **no license** is used too, unless its own top-level files say otherwise (all rights reserved, no commercial use, no AI training and similar). Sources that the scan flags are left out; ones that were read and kept go in `data/notices-reviewed.txt`. This is a choice the project made. It is not a statement that code without a license is free to use.
- A source whose license exists but could not be identified is refused.
- Secrets are replaced before anything is tokenized, and documents that are identical are kept once.

Documentation (Markdown and reStructuredText) is a separate kind from code and is sampled on its own. The manifest has 545 sources, of which 480 pass the policy (377 Rust and 103 shell).

No license has been chosen for this repository itself yet.

## Training so far

One model, the Tiny preset, has been trained: 2,000 steps of 8 x 256 tokens (4.1 million tokens) on a sample of the corpus, on one CPU thread with the AVX kernels on a GitHub runner, at about 2,000 to 2,300 tokens a second. The loss on repositories that were held out of training went from 8.318 (a model that knows nothing) to 5.296 at step 250 and 3.914 at step 2,000, and it was still falling. A run that is stopped and resumed from its saved state gives the same held-out loss at every evaluation as one that is not stopped, to the three decimals shown (Train runs 1 and 4, the second continuing run 2 from step 438). None of the settings is tuned.

## Continuous integration

Ten GitHub Actions workflows, all started by hand (`workflow_dispatch`):

| Workflow | What it does |
|---|---|
| `ci.yml` | Format, clippy, build and test, for the whole workspace or one package |
| `docs.yml` | Check that the docs and the source agree, then build rustdoc |
| `security.yml` | Dependency advisories (`cargo audit`), a secret scan over the full history (gitleaks) and repository hygiene |
| `benchmarks.yml` | Criterion benchmarks and the plain C baselines |
| `train-smoke.yml` | A tiny model must learn a fixed batch, on several platforms |
| `data-smoke.yml` | The data pipeline must give byte-identical shards and batches on every platform |
| `ingest.yml` | Check the manifest against the license policy, fetch the sources and measure them, or propose a new manifest |
| `train.yml` | Build a corpus sample, train the vocabulary and the model within a time budget, write the losses to the summary, and continue a stopped run |
| `midman-api-ffi.yml` | Test the C ABI from C and C++ |
| `deploy-site.yml` | Build the website and deploy it to Cloudflare Pages |

Details and the history of every fix are in `docs/ci-and-workflows.md`.

## Layout

```
crates/       the library crates (midman-*)
apps/         command line, server and trainer (stubs)
benches/      midman-bench: criterion benchmarks and C baselines
web/          the website: landing page, lab, test and benchmark pages, docs book
configs/      one TOML file per model-size preset
data/         the source manifest and its notes (no datasets are committed)
docs/         one document per crate, plus architecture, workspace and CI notes
scripts/      the helper scripts that CI uses, with self-tests and real captured fixtures
examples/     workspace-level examples (none yet)
tests/        workspace-level tests (none yet)
```

## Working rules

- Versions stay `0.0.1` during development; the first official release is `1.0.0`.
- `unsafe_code` is denied in the workspace. The crates that need it (`midman-api` for the C ABI, `midman-bench` for the C baselines, `midman-tensor` for the AVX kernels) say so at their crate roots, and the `undocumented_unsafe_blocks` lint is on, so every `unsafe` block needs a safety comment.
- Every source file starts with a NOTICE header that points to its section in `docs/`, and `scripts/check_docs.py` checks it.
- Profile before optimizing. A speed-up is kept only with a release-build measurement, and results stay bit-identical unless a change says otherwise.
- Apart from the benchmark package, the crates use only the standard library and each other.
- No weights, datasets or credentials are committed. `scripts/repo_hygiene.py` enforces it, and the secret scan covers the whole history.

## Related projects

- [Ubel Stratum](https://github.com/MidManStudio/ubel_stratum): the programming language the model is meant to learn.
- [Mid Engine](https://github.com/Mid-D-Man/mid-engine): the modular Rust game engine whose documentation and benchmarking conventions this repository follows.
- [DixScript-Rust](https://github.com/Mid-D-Man/DixScript-Rust): the `.mdix` configuration language.
- [mdix-scaffold](https://github.com/Mid-D-Man/mdix-scaffold): the replacement tooling used to deliver changes to this repository.
