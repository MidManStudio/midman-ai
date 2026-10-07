# midman-model

## Overview
`MidManModel`, the decoder-only transformer: a token embedding, `n_layers` pre-norm decoder blocks from `midman-nn`, a final RMSNorm, and an output projection that is either the embedding table transposed (tied) or its own matrix. It is built from a `ModelConfig`, and construction fails unless the parameters it created add up to exactly `ModelConfig::param_count`.

Dependencies: `midman-foundation`, `midman-model-config`, `midman-nn` and `midman-tensor`. `midman-optim` is a dev-dependency used only by the training smoke test. It moves to `midman-training` when that crate has a training loop.

Test status when this was written: 6 unit tests, 9 whole-model tests, 1 training smoke test that runs by default and 1 that is ignored by default, 1 doctest. Checked on Rust 1.75 with `fmt`, `clippy -D warnings` and `rustdoc -D warnings`. Newer clippy versions were not available when the code was written.

## Modules

### `lib.rs`
**What it does:** Crate root. Re-exports `MidManModel` and `DEFAULT_INIT_STD`. The crate docs include a runnable example.

### `model.rs`
**What it does:** `MidManModel` with `new`, `with_init_std`, `config`, `num_parameters`, `forward` (token ids to logits of shape `[batch, seq, vocab_size]`), `loss` (mean next-token cross-entropy) and a `Module` implementation.

**Parameter names and order:** `embed_tokens.weight`, then for each block `blocks.{i}.attn_norm`, `blocks.{i}.attn.{q,k,v,o}_proj`, `blocks.{i}.mlp_norm` and `blocks.{i}.mlp.{gate,up,down}_proj`, then `final_norm.weight`, and `lm_head.weight` only when embeddings are untied. Every name but the norms ends in `.weight`.

**Design notes:**
- Every weight is drawn from one `Rng` in a fixed order, so a single seed fixes the whole model.
- Ordinary weights use `std` (0.02 by default). The attention output and MLP down projections use `std / sqrt(2 * n_layers)`. Norm weights start at 1. These are defaults to start from, not tuned.
- With tied embeddings the logits are `x * embedding^T`, so the embedding table gets a gradient from both of its uses.
- `forward` checks `batch`, `seq`, `seq <= max_seq_len` and the id count before running anything. There is no KV cache and no start-position argument yet.
- Gradient tapes hold every intermediate tensor, so the logits and loss must be dropped (or consumed by `backward`) before an optimizer step, or `Parameter::update` has to copy the weights instead of editing them in place. `backward` consumes the loss, so the usual loop does this naturally.

### `tests/wiring.rs`
**What it does:** Whole-model checks.
- **Parameter accounting:** for the Smoke and Tiny presets and a grouped-query model in both head modes, the built parameters equal `ModelConfig::param_breakdown` in total and per category (embedding, attention, MLP, norms, head). The larger presets are covered by the config crate's own tests, because building a billion-parameter model in a unit test is not sensible.
- **Names:** unique, in running order, with the expected first and last names.
- **Determinism:** one seed gives one model; another seed gives another.
- **Causality:** changing the token at any position leaves every earlier position's logits unchanged, and sequences in a batch do not affect each other.
- **Init scale:** each weight's measured spread is within 30% of the documented standard deviation.
- **Gradients:** a finite-difference check of the loss with respect to six random entries of every parameter, with the weights drawn at standard deviation 0.3 so the gradients are far above `f32` noise (default-scale gradients would be too small for a finite difference to resolve). In both head modes the overall relative error was about 3e-4 or better and the worst single parameter about 2e-3, so the test asserts below 5e-3 overall and below 3e-2 per parameter, and asserts that each sampled gradient is large enough to mean something.

### `tests/overfit.rs`
**What it does:** Training smoke tests. A small model must memorize random token sequences, which needs the embeddings, attention (a repeated token has to be told apart by what came before it), the MLPs, the loss, `backward`, the gradient plumbing and AdamW to all work together.
- **`a_tiny_model_memorizes_random_sequences`** (runs in `cargo test`): vocabulary 12, width 16, 2 layers, 2 sequences of 12 tokens, 150 steps. Observed: loss 2.49 to 0.0006 and 100% next-token accuracy, in about 1.5 seconds in a debug build. It asserts the start is near `ln(V)`, the loss falls below a tenth of its start, and accuracy is above 95%.
- **`the_smoke_preset_memorizes_random_sequences`** (ignored by default): the 117K-parameter Smoke preset, 4 sequences of 32 tokens, 300 steps. Observed in a release build with LTO off: loss 5.567 to 0.0005 and 100% accuracy in about 5 seconds. Run it with `cargo test --release -p midman-model --test overfit -- --ignored`; the `train-smoke` workflow does.

## CI and Workflows
`train-smoke.yml` runs the ignored smoke-preset test in release mode. See `docs/ci-and-workflows.md`. The default test run covers the rest.

## Fixes and Problems
- **The init-scale test first failed on sampling noise.** `v_proj` in the small test model has 128 weights, so its measured spread can be 17% off the true value without anything being wrong. The tolerance was widened from 15% to 30%. The bug the test exists for (residual projections not scaled, a factor of about 2.4 here) is far outside that.
- **A mutation run found a gap here.** Replacing `Init::scaled` with `Init::uniform` in the model survived the first version of the tests, because nothing measured the weight spread. The init-scale test above was added. See `docs/midman-nn.md` for the other four gaps from the same run.
- **Clippy `needless_range_loop` in the gradient-check helper.** It looped over an index to read a vector; it now iterates with `enumerate`.
