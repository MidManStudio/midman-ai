// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-training.md, section "trainer.rs"
// ============================================================================
//! The trainer: one run of a model over a stream of batches.

use crate::state::{load_state, progress_of, state_to_file, Progress};
use crate::TrainConfig;
use midman_checkpoint::{save_parameters, TensorFile};
use midman_data::{Batcher, Sequences};
use midman_foundation::{Error, Result};
use midman_model::MidManModel;
use midman_nn::Module;
use midman_optim::{clip_global_norm, gather_gradients, AdamW, WarmupCosine};
use std::collections::BTreeMap;
use std::path::Path;

/// The metadata key that holds how many sequences the data had when the run was saved.
const SEQUENCES_KEY: &str = "data.sequences";
/// The metadata key that holds the vocabulary size.
const VOCAB_KEY: &str = "data.vocab_size";

/// What one training step did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepReport {
    /// Steps taken, this one included.
    pub step: u64,
    /// The loss on this step's batch, before the update.
    pub loss: f32,
    /// The global norm of the gradient, before clipping.
    pub grad_norm: f32,
    /// The learning rate used.
    pub lr: f32,
    /// Tokens the model was trained on in this step (`batch_size * seq_len`).
    pub tokens: u64,
    /// The pass over the data this step's batch came from.
    pub epoch: u64,
}

/// The loss of the model on held-out data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvalReport {
    /// The mean loss over the batches, in nats per token.
    pub loss: f32,
    /// How many batches were evaluated.
    pub batches: usize,
    /// How many tokens were predicted.
    pub tokens: u64,
}

/// A model, an optimizer, a schedule and a batcher, and the step they are at.
pub struct Trainer {
    model: MidManModel,
    optimizer: AdamW,
    schedule: WarmupCosine,
    batcher: Batcher,
    config: TrainConfig,
    step: u64,
    vocab_size: u32,
    sequences: usize,
}

impl std::fmt::Debug for Trainer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Trainer")
            .field("step", &self.step)
            .field("epoch", &self.batcher.epoch())
            .field("position", &self.batcher.position())
            .field("sequences", &self.sequences)
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl Trainer {
    /// Starts a run at step 0 with a new optimizer.
    ///
    /// # Errors
    ///
    /// Returns an error if the settings are not valid (see [`TrainConfig::validate`]), if
    /// `data` has windows of another length than `config.seq_len`, if the sequence length
    /// is longer than the model allows, if the data's vocabulary is not the model's, or if
    /// there are fewer sequences than one batch.
    pub fn new(model: MidManModel, config: TrainConfig, data: Sequences) -> Result<Trainer> {
        config.validate()?;
        if data.seq_len() != config.seq_len {
            return Err(Error::invalid(format!(
                "the data has windows for sequences of {} tokens but the run uses {}",
                data.seq_len(),
                config.seq_len
            )));
        }
        let model_config = model.config();
        if config.seq_len > model_config.max_seq_len {
            return Err(Error::invalid(format!(
                "sequences of {} tokens are longer than the model's {}",
                config.seq_len, model_config.max_seq_len
            )));
        }
        let vocab_size = match data.vocab_size() {
            Some(v) if v as usize == model_config.vocab_size => v,
            Some(v) => {
                return Err(Error::invalid(format!(
                    "the data has a vocabulary of {v} and the model {}",
                    model_config.vocab_size
                )));
            }
            None => return Err(Error::invalid("the data has no tokens".to_string())),
        };
        let sequences = data.len();
        let optimizer = config.optimizer()?;
        let schedule = config.schedule()?;
        let batcher = Batcher::new(data, config.batch_size, config.seed)?;
        Ok(Trainer { model, optimizer, schedule, batcher, config, step: 0, vocab_size, sequences })
    }

    /// Steps taken so far.
    pub fn step_count(&self) -> u64 {
        self.step
    }

    /// Whether the steps the settings asked for have all been taken.
    pub fn done(&self) -> bool {
        self.step >= self.config.total_steps
    }

    /// Tokens trained on so far.
    pub fn tokens_seen(&self) -> u64 {
        self.step * (self.config.batch_size * self.config.seq_len) as u64
    }

    /// The model.
    pub fn model(&self) -> &MidManModel {
        &self.model
    }

    /// The optimizer.
    pub fn optimizer(&self) -> &AdamW {
        &self.optimizer
    }

    /// The settings of the run.
    pub fn config(&self) -> &TrainConfig {
        &self.config
    }

    /// Where the run is: the step and the place in the data.
    pub fn progress(&self) -> Progress {
        Progress { step: self.step, epoch: self.batcher.epoch(), position: self.batcher.position() }
    }

    /// Takes one step: the next batch, the loss, the gradient, clipping and the update.
    ///
    /// A step that fails changes nothing: the weights, the optimizer, the step count and the
    /// place in the data stay as they were.
    ///
    /// # Errors
    ///
    /// Returns an error if the loss is not finite (the weights have blown up or the data
    /// holds a token the model cannot take), or if the model, the backward pass or the
    /// optimizer refuse, for example on a gradient that is not finite.
    pub fn step(&mut self) -> Result<StepReport> {
        let (epoch, position) = (self.batcher.epoch(), self.batcher.position());
        let batch = self.batcher.next_batch();
        match self.update(&batch.inputs, &batch.targets, batch.batch, batch.seq) {
            Ok((loss, grad_norm, lr)) => {
                self.step += 1;
                Ok(StepReport {
                    step: self.step,
                    loss,
                    grad_norm,
                    lr,
                    tokens: (batch.batch * batch.seq) as u64,
                    epoch: batch.epoch,
                })
            }
            Err(err) => {
                // The batcher has moved on; put it back where it was.
                self.batcher.seek(epoch, position)?;
                Err(err)
            }
        }
    }

    fn update(
        &mut self,
        inputs: &[usize],
        targets: &[usize],
        batch: usize,
        seq: usize,
    ) -> Result<(f32, f32, f32)> {
        let loss = self.model.loss(inputs, targets, batch, seq)?;
        let value = loss.item()?;
        if !value.is_finite() {
            return Err(Error::invalid(format!(
                "the loss at step {} is {value}, not a number",
                self.step + 1
            )));
        }
        let mut grads = loss.backward()?;
        let mut params = self.model.parameters_mut();
        let mut gradients = gather_gradients(&params, &mut grads)?;
        let grad_norm = clip_global_norm(&mut gradients, self.config.clip_norm)?;
        let lr = self.schedule.lr(self.step);
        self.optimizer.step(&mut params, &gradients, lr)?;
        Ok((value, grad_norm, lr))
    }

    /// The loss of the model on `data`, over at most `max_batches` batches of the run's
    /// batch size, taken at even distances through the sequences so that the whole of the
    /// data is sampled. The same data gives the same sequences every time.
    ///
    /// # Errors
    ///
    /// Returns an error if `data` has windows of another length than the run's or another
    /// vocabulary than the model's, if it is empty, if `max_batches` is zero, or if the
    /// model refuses.
    pub fn evaluate(&self, data: &Sequences, max_batches: usize) -> Result<EvalReport> {
        if data.seq_len() != self.config.seq_len {
            return Err(Error::invalid(format!(
                "the evaluation data has windows for {} tokens and the run uses {}",
                data.seq_len(),
                self.config.seq_len
            )));
        }
        if data.vocab_size() != Some(self.vocab_size) {
            return Err(Error::invalid(format!(
                "the evaluation data has a vocabulary of {:?} and the model {}",
                data.vocab_size(),
                self.vocab_size
            )));
        }
        if max_batches == 0 || data.is_empty() {
            return Err(Error::invalid("nothing to evaluate".to_string()));
        }
        let seq = self.config.seq_len;
        // A batch is the run's size, or all there is if the data holds fewer sequences.
        let batch = self.config.batch_size.min(data.len());
        let batches = max_batches.min(data.len() / batch);
        let rows = batches * batch;
        let mut total = 0.0f64;
        for b in 0..batches {
            let mut inputs = Vec::with_capacity(batch * seq);
            let mut targets = Vec::with_capacity(batch * seq);
            for r in 0..batch {
                // Even steps through the sequences: row `k` of `rows` is sequence `k * len / rows`.
                let at = (b * batch + r) * data.len() / rows;
                let window = data.window(at).ok_or_else(|| {
                    Error::invalid(format!("the evaluation data has no sequence {at}"))
                })?;
                inputs.extend(window[..seq].iter().map(|&t| t as usize));
                targets.extend(window[1..].iter().map(|&t| t as usize));
            }
            let loss = self.model.loss(&inputs, &targets, batch, seq)?;
            total += f64::from(loss.item()?);
        }
        Ok(EvalReport {
            loss: (total / batches as f64) as f32,
            batches,
            tokens: (rows * seq) as u64,
        })
    }

    /// The state of the run as a file: weights, optimizer moments, settings and place.
    ///
    /// # Errors
    ///
    /// Returns an error if [`state_to_file`] does.
    pub fn state_file(&self) -> Result<TensorFile> {
        let params = self.model.parameters();
        let mut extra = BTreeMap::new();
        extra.insert(SEQUENCES_KEY.to_string(), self.sequences.to_string());
        extra.insert(VOCAB_KEY.to_string(), self.vocab_size.to_string());
        state_to_file(&params, &self.optimizer, self.progress(), &self.config, &extra)
    }

    /// Saves the state of the run to `path`, replacing a file there atomically.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be built or written.
    pub fn save_state(&self, path: impl AsRef<Path>) -> Result<()> {
        self.state_file()?.save(path)
    }

    /// Continues a saved run: loads the weights, the optimizer state and the place in the
    /// data from `path`. Nothing changes unless all of it can be loaded.
    ///
    /// The file has to come from a run with these settings, on this model, over data with
    /// the same number of sequences and the same vocabulary.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read, is not a state file, was saved with other
    /// settings or for another model or data, or is not complete.
    pub fn resume(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let file = TensorFile::load(path)?;
        let progress = progress_of(&file)?;
        for (key, now) in
            [(SEQUENCES_KEY, self.sequences.to_string()), (VOCAB_KEY, self.vocab_size.to_string())]
        {
            match file.metadata().get(key) {
                Some(then) if *then == now => {}
                Some(then) => {
                    return Err(Error::invalid(format!(
                        "the saved run had `{key}` {then} and this run has {now}: it is not the same data"
                    )));
                }
                None => {
                    return Err(Error::invalid(format!("the saved run has no `{key}` entry")));
                }
            }
        }
        if progress.position > self.batcher.batches_per_epoch() {
            return Err(Error::invalid(format!(
                "the saved run is {} batches into a pass of {}",
                progress.position,
                self.batcher.batches_per_epoch()
            )));
        }
        let mut params = self.model.parameters_mut();
        let loaded = load_state(&file, &mut params, &mut self.optimizer, &self.config)?;
        self.step = loaded.step;
        self.batcher.seek(loaded.epoch, loaded.position)
    }

    /// Writes the weights alone to `path` as an ordinary checkpoint (see `midman-checkpoint`),
    /// with the step in its metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if the checkpoint cannot be written.
    pub fn export_weights(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut metadata = BTreeMap::new();
        metadata.insert("step".to_string(), self.step.to_string());
        metadata.insert("tokens_seen".to_string(), self.tokens_seen().to_string());
        save_parameters(path, &self.model.parameters(), &metadata)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use midman_data::TokenShard;
    use midman_foundation::Rng;
    use midman_model_config::ModelConfig;

    pub(crate) fn model_config() -> ModelConfig {
        ModelConfig {
            vocab_size: 12,
            d_model: 16,
            n_layers: 1,
            n_heads: 2,
            n_kv_heads: 1,
            d_ff: 32,
            max_seq_len: 8,
            rope_theta: 10_000.0,
            norm_eps: 1e-5,
            tie_embeddings: true,
        }
    }

    pub(crate) fn data(tokens: usize) -> Sequences {
        let ids: Vec<u32> = (0..tokens).map(|i| (i % 5) as u32).collect();
        Sequences::new(vec![TokenShard::new(12, 11, ids).unwrap()], 8).unwrap()
    }

    pub(crate) fn trainer(steps: u64) -> Trainer {
        let model = MidManModel::new(&model_config(), &mut Rng::seed_from_u64(3)).unwrap();
        let mut config = TrainConfig::new(2, 8, steps);
        config.max_lr = 1e-2;
        config.seed = 5;
        Trainer::new(model, config, data(400)).unwrap()
    }

    #[test]
    fn the_run_checks_what_it_is_given() {
        let model = || MidManModel::new(&model_config(), &mut Rng::seed_from_u64(3)).unwrap();
        let good = TrainConfig::new(2, 8, 10);
        assert!(Trainer::new(model(), good.clone(), data(400)).is_ok());

        let mut bad = good.clone();
        bad.batch_size = 0;
        assert!(Trainer::new(model(), bad, data(400)).is_err());

        // The window length of the data is not the run's.
        let err = Trainer::new(model(), TrainConfig::new(2, 4, 10), data(400)).unwrap_err();
        assert!(err.to_string().contains("windows for sequences of 8"), "{err}");

        // Longer sequences than the model allows.
        let long =
            Sequences::new(vec![TokenShard::new(12, 11, vec![1; 100]).unwrap()], 16).unwrap();
        let err = Trainer::new(model(), TrainConfig::new(2, 16, 10), long).unwrap_err();
        assert!(err.to_string().contains("longer than the model's 8"), "{err}");

        // Another vocabulary, larger or smaller: only the model's own is accepted.
        let other =
            Sequences::new(vec![TokenShard::new(20, 19, vec![1; 100]).unwrap()], 8).unwrap();
        let err = Trainer::new(model(), good.clone(), other).unwrap_err();
        assert!(err.to_string().contains("vocabulary of 20 and the model 12"), "{err}");
        let smaller =
            Sequences::new(vec![TokenShard::new(8, 7, vec![1; 100]).unwrap()], 8).unwrap();
        let err = Trainer::new(model(), good.clone(), smaller).unwrap_err();
        assert!(err.to_string().contains("vocabulary of 8 and the model 12"), "{err}");

        // Fewer sequences than a batch.
        let few = Sequences::new(vec![TokenShard::new(12, 11, vec![1; 12]).unwrap()], 8).unwrap();
        assert!(Trainer::new(model(), TrainConfig::new(2, 8, 10), few).is_err());
    }

    #[test]
    fn a_step_reports_what_it_did_and_moves_the_run_on() {
        let mut t = trainer(10);
        assert_eq!((t.step_count(), t.tokens_seen(), t.done()), (0, 0, false));
        assert_eq!(t.progress(), Progress { step: 0, epoch: 0, position: 0 });
        let r = t.step().unwrap();
        assert_eq!(r.step, 1);
        assert_eq!(r.tokens, 16);
        assert_eq!(r.epoch, 0);
        assert!(r.loss.is_finite() && r.loss > 1.0, "{}", r.loss);
        assert!(r.grad_norm > 0.0);
        // Warmup: the first step already moves, at max_lr / warmup_steps.
        assert_eq!(r.lr, t.config().schedule().unwrap().lr(0));
        assert_eq!(t.step_count(), 1);
        assert_eq!(t.tokens_seen(), 16);
        assert_eq!(t.progress().position, 1);
        assert_eq!(t.optimizer().step_count(), 1);
        for _ in 1..10 {
            t.step().unwrap();
        }
        assert!(t.done());
        assert_eq!(t.tokens_seen(), 160);
    }

    #[test]
    fn a_report_says_which_pass_over_the_data_its_batch_came_from() {
        // 14 sequences and batches of 2: seven batches a pass. The first batch of the second
        // pass is the eighth step, and its report has to say pass 1 although the batcher
        // only moves on when that batch is asked for.
        let model = MidManModel::new(&model_config(), &mut Rng::seed_from_u64(3)).unwrap();
        let mut config = TrainConfig::new(2, 8, 20);
        config.max_lr = 1e-2;
        let mut t = Trainer::new(model, config, data(113)).unwrap();
        let epochs: Vec<u64> = (0..16).map(|_| t.step().unwrap().epoch).collect();
        assert_eq!(epochs, [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2]);
        assert_eq!(t.progress().epoch, 2);
        assert_eq!(t.progress().position, 2);
    }

    #[test]
    fn the_loss_falls_on_a_pattern_it_can_learn() {
        let mut t = trainer(40);
        let first = t.step().unwrap().loss;
        let mut last = first;
        while !t.done() {
            last = t.step().unwrap().loss;
        }
        assert!((first - 12f32.ln()).abs() < 0.5, "starts near ln(12), got {first}");
        assert!(last < 0.5 * first, "{first} -> {last}");
    }

    #[test]
    fn a_step_that_fails_changes_nothing() {
        let mut t = trainer(10);
        t.step().unwrap();
        let before = t.state_file().unwrap().to_bytes().unwrap();
        let progress = t.progress();
        // A weight that is not a number makes the loss not a number.
        let mut params = t.model.parameters_mut();
        let first = &mut params[0];
        let poisoned = vec![f32::NAN; first.numel()];
        let saved = first.data().to_vec();
        first.set_data(&poisoned).unwrap();
        let err = t.step().unwrap_err();
        assert!(err.to_string().contains("not a number"), "{err}");
        let mut params = t.model.parameters_mut();
        params[0].set_data(&saved).unwrap();
        assert_eq!(t.progress(), progress);
        assert_eq!(t.step_count(), 1);
        assert_eq!(t.state_file().unwrap().to_bytes().unwrap(), before);
        // And the run goes on as if nothing had happened.
        t.step().unwrap();
        assert_eq!(t.step_count(), 2);
    }

    #[test]
    fn evaluation_is_the_mean_loss_on_evenly_spread_sequences() {
        let t = trainer(10);
        let held = data(800);
        let report = t.evaluate(&held, 3).unwrap();
        assert_eq!(report.batches, 3);
        assert_eq!(report.tokens, 3 * 2 * 8);
        // The same call gives the same number.
        assert_eq!(t.evaluate(&held, 3).unwrap(), report);
        // The mean of the per-batch losses, computed by hand for the same sequences.
        let rows = 6;
        let mut sum = 0.0f64;
        for b in 0..3 {
            let (mut inputs, mut targets) = (Vec::new(), Vec::new());
            for r in 0..2 {
                let at = (b * 2 + r) * held.len() / rows;
                let w = held.window(at).unwrap();
                inputs.extend(w[..8].iter().map(|&x| x as usize));
                targets.extend(w[1..].iter().map(|&x| x as usize));
            }
            sum += f64::from(t.model().loss(&inputs, &targets, 2, 8).unwrap().item().unwrap());
        }
        assert_eq!(report.loss, (sum / 3.0) as f32);
        // More batches than the data holds is cut to what it holds: 12 sequences, batch 2.
        let small =
            Sequences::new(vec![TokenShard::new(12, 11, vec![1; 12 * 8 + 1]).unwrap()], 8).unwrap();
        assert_eq!(small.len(), 12);
        let report = t.evaluate(&small, 100).unwrap();
        assert_eq!((report.batches, report.tokens), (6, 12 * 8));
    }

    #[test]
    fn evaluation_refuses_what_does_not_fit() {
        let t = trainer(10);
        let held = data(800);
        assert!(t.evaluate(&held, 0).is_err());
        let other_len =
            Sequences::new(vec![TokenShard::new(12, 11, vec![1; 100]).unwrap()], 4).unwrap();
        assert!(t.evaluate(&other_len, 2).is_err());
        let other_vocab =
            Sequences::new(vec![TokenShard::new(30, 29, vec![1; 100]).unwrap()], 8).unwrap();
        assert!(t.evaluate(&other_vocab, 2).is_err());
    }

    #[test]
    fn evaluation_does_not_train() {
        let t = trainer(10);
        let before = t.state_file().unwrap().to_bytes().unwrap();
        t.evaluate(&data(800), 4).unwrap();
        assert_eq!(t.state_file().unwrap().to_bytes().unwrap(), before);
    }
}
