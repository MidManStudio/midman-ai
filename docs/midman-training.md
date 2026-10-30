# midman-training

## Overview
The training loop: a `Trainer` that owns a model, an optimizer, a schedule and a batcher and takes training steps, evaluation on held-out data, and saving and resuming a run.

One step takes the next batch from the data, runs the forward pass and the loss, goes backward, clips the gradient, asks the schedule for a learning rate and updates the weights. What a run depends on is the model's weights, the optimizer's moments and the place in the data, and `Trainer::save_state` keeps exactly that in one file. A run stopped after step `k` and resumed gives, bit for bit, what a run that never stopped gives. `Trainer::export_weights` writes the weights alone as an ordinary checkpoint.

Dependencies: `midman-foundation`, `midman-tensor`, `midman-nn`, `midman-model`, `midman-model-config`, `midman-optim`, `midman-checkpoint` and `midman-data`. A Cargo feature `avx` turns on the AVX matrix kernels of `midman-tensor` (results are the same; see `midman-tensor`).

Not built yet: gradient accumulation over several batches, more than one thread, a mix of several data sources by weight, a learning-rate schedule that can be changed in a running run, and anything distributed. The crate's description in `Cargo.toml` lists some of these; they are the plan, not what exists.

Test status when this was written: 21 unit tests, 6 integration tests and 1 doctest. Checked with `fmt`, `clippy -D warnings` and `rustdoc -D warnings` on Rust 1.91 and 1.75, and mutation-tested (see "Fixes and Problems").

## Modules

### `lib.rs`
**What it does:** Crate root: the documentation and a doctest that trains a tiny model on a repeating pattern until its loss has fallen, and the exports (`TrainConfig`, `Trainer`, `StepReport`, `EvalReport`, `Progress`, `load_state`, `progress_of`, `state_to_file`, `STATE_FORMAT`, `STATE_VERSION`).

**Tests:** The doctest.

### `config.rs`
**What it does:** `TrainConfig`, the settings of a run that are not the model or the data: batch size, sequence length, the seed of the order of the data, the learning-rate schedule (`max_lr`, `min_lr`, `warmup_steps`, `total_steps`), AdamW's betas, epsilon and weight decay, and the clipping norm. `new(batch, seq, steps)` gives defaults (`max_lr` 1e-3, `min_lr` 1e-4, warmup a twentieth of the steps, betas 0.9 and 0.95, epsilon 1e-8, weight decay 0.1, clip 1.0). `validate` checks them. `schedule` and `optimizer` build the schedule and the optimizer. `to_metadata` and `check_matches` write the settings into a state file and check, on resume, that they are the same.

**Decisions:**
- The settings are saved with the state and a resume with other settings is refused, naming every setting that differs (`train.max_lr was 0.001 and is now 0.002`). A run that goes on with another schedule is not the same run, and continuing it silently would make a result that nobody can reproduce. This includes `total_steps`: the cosine schedule is made for it.
- A float is saved as the shortest text that reads back as the same number, so the comparison is exact and not "close".
- None of the defaults has been tuned. They are the usual values for a small decoder, written down so that a run states what it used.

**Tests:** Five unit tests: the defaults are valid for step counts from 1 to 1000; each of fourteen bad settings is refused; the metadata reads back as the same numbers (a seed of `u64::MAX` and a float with many digits); every difference and every missing setting is named, and extra entries in the file are not a difference; the schedule and the optimizer follow the settings.

### `state.rs`
**What it does:** The state of a run in one file, in the safetensors container of `midman-checkpoint` with the format name `midman-training-state` and version 1. Each parameter is a tensor `param/<name>`, its two AdamW moments are `adamw.m/<name>` and `adamw.v/<name>` (none before the first step), and the metadata holds the settings, the step, the pass over the data and the batch within it, and whatever the caller adds. `state_to_file` builds the file, `progress_of` reads and checks the header without touching anything, and `load_state` puts weights and moments into the parameters and the optimizer.

**Decisions:**
- One file, so that saving is atomic: `TensorFile::save` writes a temporary file and renames it, and a crash cannot leave weights from one step with moments from another. The weights alone are exported separately (`Trainer::export_weights`), because `load_parameters` in `midman-checkpoint` refuses tensors the model does not have, so the moments cannot share a plain checkpoint.
- A load changes nothing unless the whole of it can succeed: the header, the settings, the optimizer's step count, every parameter and moment present with the right shape, no tensor that nobody wants, and no moments in a run that has taken no step. The optimizer's own refusal (`AdamW::restore`) comes before any weight is written.
- The place in the data is the pass and the batch within it, which is all the batcher needs (its order depends only on the seed and the pass). No random generator state is saved, because a step uses none.
- A state saved before the first step holds the weights and no moments, and loads.

**Tests:** Eight unit tests: a round trip through bytes (names in order, metadata, weights and moments bit for bit); a state before the first step; what the writer refuses (a step count that is not the optimizer's, two parameters with one name, with the message, moments for a parameter that is not given, moments of another size, each reserved key); a load being refused, each time leaving the parameters and the optimizer untouched, for a missing parameter, a missing first or second moment, a tensor nobody wants, a tensor of another kind, another format, another version, a step that is not a number, other settings, a wrong shape, and moments in a run with no step; moments that the optimizer refuses; two parameters with one name on load; the list of missing names cut after five; the progress read on its own.

### `trainer.rs`
**What it does:** `Trainer`. `new(model, config, data)` checks that the data's windows are the run's sequence length, that the sequence is not longer than the model allows, that the data's vocabulary is the model's, and that there is at least one batch. `step` takes one step and returns a `StepReport` (step, loss before the update, gradient norm before clipping, learning rate, tokens, pass). `evaluate(data, max_batches)` gives the mean loss on held-out windows. `save_state`, `resume` and `export_weights` write and read the files above. `done`, `step_count`, `tokens_seen`, `progress`, `model`, `optimizer` and `config` read the state.

**Decisions:**
- A step that fails changes nothing, including the place in the data: the batcher is put back where it was. The loss not being a number (the weights have blown up, or a token the model cannot take) is an error before anything is updated, not a step that spreads NaN into the weights.
- A report's pass is the pass its batch came from. The batcher moves on to a new pass only when the first batch of it is asked for, so the position before the call would name the old one.
- Evaluation takes sequences at even distances through the data (`k * len / rows`), not the first ones, so that it samples the whole of it, and the same data gives the same sequences every time. A batch is the run's size or all there is if the data holds fewer sequences. Evaluation builds the graph the loss needs and never goes backward; it leaves the run untouched (tested by comparing the saved state before and after).
- A resume also checks the number of sequences and the vocabulary of the data against what the state was saved with, because the place in the data means nothing on other data. It does not check that the data is the same bytes; the workflow rebuilds it from the same manifest, sample and seed, and the number of sequences is the check that catches a different corpus.
- Training is single threaded and takes whole batches; there is no gradient accumulation yet.

**Tests:** Eight unit tests: what `new` checks (zero batch, a sequence length that is not the data's, longer than the model's, a larger and a smaller vocabulary, fewer sequences than a batch); what a step reports and how the counters move; the loss falling from about ln(12) to under half on a pattern it can learn; a failing step (a weight set to NaN) changing nothing and the run going on afterwards; the pass in each report across two pass boundaries; evaluation being the mean over evenly spread sequences (checked against a hand computation, and cut to what the data holds); evaluation refusing what does not fit; evaluation not training.

### `tests/resume.rs`
**What it does:** Six integration tests of stopping and continuing a run.

**Tests:**
- A resumed run is the run that never stopped: a 12 step run is compared, step by step (loss, gradient norm, learning rate and pass, bit for bit) and in its final state file (byte for byte), with runs that were stopped, saved and resumed in a new trainer with a model initialised some other way, at seven cut points: before the first step, early, one batch before the end of a pass, exactly at the end of a pass, just after it, and late. The data has seven batches a pass, so the 12 steps go through the end of a pass.
- Resuming twice is the same as resuming once.
- The exported weights are an ordinary checkpoint: no optimizer tensors, the step and the tokens in the metadata, and `load_parameters` puts them into a new model that then gives the same loss bit for bit.
- A resume that is refused leaves the run alone, for a missing file, junk, a weights checkpoint, other settings, a different model width, different data, a place beyond the end of a pass and an incomplete state; the run then goes on and a good file still resumes.
- A saved state replaces the earlier one whole, and no temporary file is left.
- The held-out loss falls with training.

### `examples/train.rs`
**What it does:** A command line that trains a model on token shards (`--shards`, the `.mmtokens` files of `midman-data`) for `--steps` steps or `--minutes` minutes, with the preset's model and the shards' vocabulary, a held-out set (`--eval-shards`), logging and evaluation intervals, and `--resume`. In `--out` it keeps `state.safetensors` (replaced every `--save-every` steps and at the end), `weights.safetensors` (at the end), `train.tsv` (step, mean loss since the last line, gradient norm, learning rate, tokens, seconds), `eval.tsv` and `summary.txt` (steps, total steps, tokens, finished). `--features avx` turns on the AVX kernels.

**Decisions:**
- The time budget stops the run, saves and exits normally; the schedule stays the one for `--steps`, so a run that stopped early continues with `--resume` and the same settings. The seconds in `train.tsv` start again at 0 at a resume, and `train_summary.py` allows for that.
- The loss on a line of `train.tsv` is the mean over the steps since the previous line, not the last step's, which is too noisy to read. After a resume the first line covers only the steps since the resume, so it can differ from the line of a run that never stopped (7.403 against 7.398 at step 20 in the real run below) while every step is the same.
- The example has the same name as the tokenizer's (`train`), so both binaries have one path; they are always started with `cargo run -p <crate> --example train`, never from `target/`.

**Tests:** None of its own. It was run for real on the shards built from a corpus of 3,705 files (8.6 million tokens, vocabulary 4,096): the Tiny preset (1,315,968 parameters) for 100 steps of 8 by 256 tokens, stopped on a 0.6 minute budget at step 16 and resumed to step 100, and the loss on held-out sources went from 7.195 to 6.785 (ln 4096 is 8.318). A run that did not stop gave the same gradient norm, learning rate and, at step 25, loss as the resumed one at steps 5, 10, 15, 20 and 25. Step 20 has a gradient norm of 5.109, a spike that is in the run without a stop too and so is not a fault of the resume.

## CI and Workflows
`train.yml` runs the whole path from the manifest to a trained model: see `ci-and-workflows.md`.

## Fixes and Problems

### `lib.rs`
- **Open:** the description in `Cargo.toml` still lists gradient accumulation and distributed training, which do not exist.

### `config.rs`
- No defect found. The defaults are not tuned.

### `state.rs`
- **Mutation testing** (see `trainer.rs`): the two checks for two parameters with one name were hidden by the container, which refuses a duplicate tensor name with a message of its own, so a test that only asked for an error passed without them. The tests now read the message.

### `trainer.rs`
- **Mutation testing** (32 changes to `config.rs`, `state.rs` and `trainer.rs`, one at a time: bounds, the sequence length and vocabulary checks, the counters, the rollback of a failed step, the learning rate's step, the pass in a report, each part of evaluation, each check of a resume, each check of a load, the order of the moments, the version): the first run left two alive and one that did not compile. The vocabulary check was loosened to "not larger" by one change and the tests only had a larger vocabulary, so a smaller one is now tested; the pass in a report differed from the one before the call only on the first batch of a pass, which only the resume test crossed and compared between two runs of the same code, so a test now checks the passes in order across two boundaries. After that every change is caught.
- **Open:** a loss that is infinite (and not NaN) is also refused as "not a number", but only NaN is tested.
- **Open:** one thread, and a resume copies the moments into `Vec<f32>`s; for the sizes that can be trained on a CPU this does not matter.

### `tests/resume.rs`
- No defect found.

### `examples/train.rs`
- **The same example name in two crates.** `midman-tokenizer` and `midman-training` both have `examples/train.rs`, so a build of both puts one binary over the other in `target/release/examples/train`. Found when the shard step could have run the wrong binary; always use `cargo run -p`.
- **Fixed:** the end of a run evaluated again when the last step had just been evaluated, writing the same line twice to `eval.tsv`; it now skips the second. The real log of that run, with the duplicate line, is a test fixture of `train_summary.py`, which keeps one line per step.
- **Measured:** the Tiny preset at 8 by 256 tokens ran at about 850 tokens a second with the scalar kernels and 1,282 with `--features avx` on a 2.1 GHz sandbox CPU, a 1.5 times gain for the whole step with identical losses.
