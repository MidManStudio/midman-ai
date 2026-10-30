// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-training.md, section "tests/resume.rs"
// ============================================================================
//! Saving a run, stopping, and continuing it must give the run that never stopped.

use midman_checkpoint::{load_parameters, TensorFile};
use midman_data::{Sequences, TokenShard};
use midman_foundation::Rng;
use midman_model::MidManModel;
use midman_model_config::ModelConfig;
use midman_nn::Module;
use midman_training::{StepReport, TrainConfig, Trainer};
use std::path::PathBuf;

fn model_config(d_model: usize) -> ModelConfig {
    ModelConfig {
        vocab_size: 12,
        d_model,
        n_layers: 1,
        n_heads: 2,
        n_kv_heads: 1,
        d_ff: 2 * d_model,
        max_seq_len: 8,
        rope_theta: 10_000.0,
        norm_eps: 1e-5,
        tie_embeddings: true,
    }
}

/// A repeating pattern of five tokens; `tokens` of them make `(tokens - 1) / 8` sequences.
fn data(tokens: usize) -> Sequences {
    let ids: Vec<u32> = (0..tokens).map(|i| ((i * 7 + i / 5) % 5) as u32).collect();
    Sequences::new(vec![TokenShard::new(12, 11, ids).unwrap()], 8).unwrap()
}

fn config(steps: u64) -> TrainConfig {
    let mut config = TrainConfig::new(2, 8, steps);
    config.max_lr = 1e-2;
    config.seed = 11;
    config
}

/// 14 sequences, 7 batches a pass: a run of 12 steps goes through the end of a pass.
fn trainer(init_seed: u64, steps: u64) -> Trainer {
    let model = MidManModel::new(&model_config(16), &mut Rng::seed_from_u64(init_seed)).unwrap();
    Trainer::new(model, config(steps), data(113)).unwrap()
}

struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Scratch {
        let dir =
            std::env::temp_dir().join(format!("midman-training-it-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }

    fn file(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn same_report(a: &StepReport, b: &StepReport) -> bool {
    a.step == b.step
        && a.loss.to_bits() == b.loss.to_bits()
        && a.grad_norm.to_bits() == b.grad_norm.to_bits()
        && a.lr.to_bits() == b.lr.to_bits()
        && a.tokens == b.tokens
        && a.epoch == b.epoch
}

fn state_bytes(t: &Trainer) -> Vec<u8> {
    t.state_file().unwrap().to_bytes().unwrap()
}

#[test]
fn a_resumed_run_is_the_run_that_never_stopped() {
    let scratch = Scratch::new("resume");
    let total = 12;
    let mut straight = trainer(3, total);
    let reports: Vec<StepReport> = (0..total).map(|_| straight.step().unwrap()).collect();
    let final_bytes = state_bytes(&straight);
    // The run did go through the end of a pass over the data.
    assert_eq!(reports[0].epoch, 0);
    assert_eq!(reports[11].epoch, 1);

    // Cut points: before the first step, early, just before the end of a pass, exactly at
    // the end of a pass, just after it, and late.
    for cut in [0u64, 1, 5, 6, 7, 8, 11] {
        let path = scratch.file(&format!("state-{cut}.safetensors"));
        let mut first = trainer(3, total);
        for _ in 0..cut {
            first.step().unwrap();
        }
        first.save_state(&path).unwrap();
        drop(first);

        // A new process would start from a model that is initialised some other way.
        let mut second = trainer(99, total);
        second.resume(&path).unwrap();
        assert_eq!(second.step_count(), cut);
        for step in cut..total {
            let report = second.step().unwrap();
            assert!(
                same_report(&report, &reports[step as usize]),
                "cut {cut}: step {} gave {report:?}, the run that never stopped gave {:?}",
                step + 1,
                reports[step as usize]
            );
        }
        assert_eq!(state_bytes(&second), final_bytes, "cut {cut}: the final state differs");
    }
}

#[test]
fn resuming_twice_is_the_same_as_resuming_once() {
    let scratch = Scratch::new("twice");
    let total = 12;
    let mut straight = trainer(3, total);
    for _ in 0..total {
        straight.step().unwrap();
    }

    let mut run = trainer(3, total);
    for stop in [3, 8] {
        while run.step_count() < stop {
            run.step().unwrap();
        }
        let path = scratch.file(&format!("s{stop}.safetensors"));
        run.save_state(&path).unwrap();
        run = trainer(50 + stop, total);
        run.resume(&path).unwrap();
    }
    while !run.done() {
        run.step().unwrap();
    }
    assert_eq!(state_bytes(&run), state_bytes(&straight));
}

#[test]
fn the_exported_weights_are_an_ordinary_checkpoint() {
    let scratch = Scratch::new("export");
    let mut t = trainer(3, 10);
    for _ in 0..4 {
        t.step().unwrap();
    }
    let path = scratch.file("weights.safetensors");
    t.export_weights(&path).unwrap();

    let file = TensorFile::load(&path).unwrap();
    assert_eq!(file.metadata()["step"], "4");
    assert_eq!(file.metadata()["tokens_seen"], "64");
    // No optimizer state in it.
    assert!(file.tensors().iter().all(|tensor| !tensor.name().starts_with("adamw")));

    let mut fresh = MidManModel::new(&model_config(16), &mut Rng::seed_from_u64(77)).unwrap();
    let mut params = fresh.parameters_mut();
    load_parameters(&file, &mut params).unwrap();

    let inputs: Vec<usize> = (0..16).map(|i| i % 5).collect();
    let targets: Vec<usize> = (0..16).map(|i| (i + 1) % 5).collect();
    let a = t.model().loss(&inputs, &targets, 2, 8).unwrap().item().unwrap();
    let b = fresh.loss(&inputs, &targets, 2, 8).unwrap().item().unwrap();
    assert_eq!(a.to_bits(), b.to_bits());
}

#[test]
fn a_resume_that_is_refused_leaves_the_run_alone() {
    let scratch = Scratch::new("refused");
    let mut source = trainer(3, 12);
    for _ in 0..4 {
        source.step().unwrap();
    }
    let good = scratch.file("good.safetensors");
    source.save_state(&good).unwrap();

    let mut target = trainer(3, 12);
    for _ in 0..2 {
        target.step().unwrap();
    }
    let before = state_bytes(&target);

    let mut refuse = |name: &str, path: &PathBuf, mention: &str| {
        let err = target.resume(path).expect_err(name);
        assert!(err.to_string().contains(mention), "{name}: {err}");
        assert_eq!(state_bytes(&target), before, "{name}: the run changed");
    };

    // Not there, and not a state at all.
    refuse("missing file", &scratch.file("nope.safetensors"), "");
    let junk = scratch.file("junk.safetensors");
    std::fs::write(&junk, b"this is not a tensor file").unwrap();
    refuse("junk", &junk, "");

    // A checkpoint of weights is not the state of a run.
    let weights = scratch.file("weights.safetensors");
    source.export_weights(&weights).unwrap();
    refuse("weights only", &weights, "format");

    // Other settings.
    let other_settings = {
        let mut run = {
            let model = MidManModel::new(&model_config(16), &mut Rng::seed_from_u64(3)).unwrap();
            let mut cfg = config(12);
            cfg.max_lr = 2e-2;
            Trainer::new(model, cfg, data(113)).unwrap()
        };
        run.step().unwrap();
        let path = scratch.file("other-settings.safetensors");
        run.save_state(&path).unwrap();
        path
    };
    refuse("other settings", &other_settings, "train.max_lr");

    // A different model.
    let wide = {
        let model = MidManModel::new(&model_config(24), &mut Rng::seed_from_u64(3)).unwrap();
        let mut run = Trainer::new(model, config(12), data(113)).unwrap();
        run.step().unwrap();
        let path = scratch.file("wide.safetensors");
        run.save_state(&path).unwrap();
        path
    };
    refuse("other model", &wide, "shape");

    // Other data.
    let other_data = {
        let model = MidManModel::new(&model_config(16), &mut Rng::seed_from_u64(3)).unwrap();
        let mut run = Trainer::new(model, config(12), data(201)).unwrap();
        run.step().unwrap();
        let path = scratch.file("other-data.safetensors");
        run.save_state(&path).unwrap();
        path
    };
    refuse("other data", &other_data, "data.sequences");

    // A place beyond the end of a pass.
    let beyond = {
        let mut file = TensorFile::load(&good).unwrap();
        file.set_metadata("position", "999");
        let path = scratch.file("beyond.safetensors");
        file.save(&path).unwrap();
        path
    };
    refuse("beyond the pass", &beyond, "999 batches");

    // A state with a tensor missing.
    let incomplete = {
        let file = TensorFile::load(&good).unwrap();
        let mut cut = TensorFile::new();
        for (key, value) in file.metadata() {
            cut.set_metadata(key.as_str(), value.as_str());
        }
        for tensor in file.tensors().iter().skip(1) {
            cut.push(tensor.name(), tensor.shape(), tensor.data()).unwrap();
        }
        let path = scratch.file("incomplete.safetensors");
        cut.save(&path).unwrap();
        path
    };
    refuse("incomplete", &incomplete, "missing");

    // And the run still goes on, and a good file still resumes.
    target.step().unwrap();
    target.resume(&good).unwrap();
    assert_eq!(target.step_count(), 4);
}

#[test]
fn a_saved_state_is_replaced_whole_or_not_at_all() {
    let scratch = Scratch::new("atomic");
    let path = scratch.file("state.safetensors");
    let mut t = trainer(3, 10);
    t.step().unwrap();
    t.save_state(&path).unwrap();
    let first = std::fs::read(&path).unwrap();
    t.step().unwrap();
    t.save_state(&path).unwrap();
    let second = std::fs::read(&path).unwrap();
    assert_ne!(first, second);
    assert_eq!(second, state_bytes(&t));
    // No temporary file is left behind.
    let names: Vec<String> = std::fs::read_dir(&scratch.0)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(names, ["state.safetensors"]);
}

#[test]
fn held_out_loss_falls_with_training() {
    let mut t = trainer(3, 40);
    let held = data(401);
    let before = t.evaluate(&held, 4).unwrap();
    while !t.done() {
        t.step().unwrap();
    }
    let after = t.evaluate(&held, 4).unwrap();
    assert_eq!(before.tokens, after.tokens);
    assert!(after.loss < 0.5 * before.loss, "{} -> {}", before.loss, after.loss);
}
