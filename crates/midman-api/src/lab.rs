// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-api.md, section "lab.rs"
// ============================================================================
//! The training lab: a model, an optimizer and a fixed task, stepped from outside
//! so a caller can watch the loss, the gradients and the attention as it learns.
//!
//! The task is memorization. `batch` random token sequences of `seq + 1` tokens
//! are drawn once from the seed; inputs are the first `seq` tokens of each and
//! targets the last `seq`. It is the same task the training smoke test uses, so
//! what a caller sees here is what that test checks.

use midman_foundation::{Error, Result, Rng};
use midman_model::MidManModel;
use midman_model_config::ModelConfig;
use midman_nn::Module;
use midman_optim::{clip_global_norm, gather_gradients, AdamW, WarmupCosine};

/// The largest model the lab will build, in parameters. A lab holds the weights,
/// two Adam moments and the gradients, about four copies in all, and runs in a
/// browser tab, so the limit keeps a phone from running out of memory.
pub const MAX_PARAMS: u64 = 8_000_000;

/// The most tokens (`batch * seq`) one lab task may have.
pub const MAX_TOKENS: usize = 4096;

/// Optimizer, schedule and clipping settings.
#[derive(Debug, Clone, PartialEq)]
pub struct Hyper {
    /// Peak learning rate, reached after warmup.
    pub max_lr: f32,
    /// Learning rate the cosine decay ends at.
    pub min_lr: f32,
    /// Steps of linear warmup.
    pub warmup_steps: u64,
    /// Step at which the schedule reaches `min_lr`.
    pub total_steps: u64,
    /// AdamW weight decay, applied to matrices only.
    pub weight_decay: f32,
    /// Global gradient-norm limit.
    pub clip_norm: f32,
    /// AdamW first-moment decay.
    pub beta1: f32,
    /// AdamW second-moment decay.
    pub beta2: f32,
}

impl Default for Hyper {
    fn default() -> Hyper {
        Hyper {
            max_lr: 0.02,
            min_lr: 0.002,
            warmup_steps: 20,
            total_steps: 300,
            weight_decay: 0.01,
            clip_norm: 1.0,
            beta1: 0.9,
            beta2: 0.99,
        }
    }
}

/// Everything needed to build a [`Lab`].
#[derive(Debug, Clone)]
pub struct LabConfig {
    /// The model architecture.
    pub model: ModelConfig,
    /// Seeds the weights and the task data.
    pub seed: u64,
    /// Number of sequences in the task.
    pub batch: usize,
    /// Tokens per sequence (inputs and targets each).
    pub seq: usize,
    /// Optimizer settings.
    pub hyper: Hyper,
}

/// What one training step measured.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepRecord {
    /// The loss before the update.
    pub loss: f32,
    /// The global gradient norm before clipping.
    pub grad_norm: f32,
    /// The learning rate used for the update.
    pub lr: f32,
}

/// The result of [`Lab::step`].
#[derive(Debug, Clone, PartialEq)]
pub struct StepReport {
    /// One record per completed step.
    pub records: Vec<StepRecord>,
    /// Why training stopped before all requested steps, if it did (for example a
    /// loss or gradient that is not a finite number).
    pub stopped: Option<String>,
}

/// The model's loss and predictions on the task.
#[derive(Debug, Clone, PartialEq)]
pub struct Eval {
    /// Mean next-token cross-entropy.
    pub loss: f32,
    /// Fraction of target tokens the model's top prediction gets right.
    pub accuracy: f32,
    /// The top prediction at every position, `batch * seq` ids.
    pub predictions: Vec<usize>,
}

/// Attention probabilities of every layer and head for one sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionMaps {
    /// Number of layers.
    pub layers: usize,
    /// Number of query heads per layer.
    pub heads: usize,
    /// Sequence length; each map is `seq` by `seq`.
    pub seq: usize,
    /// `layers * heads * seq * seq` values, layer-major then head, row-major.
    pub data: Vec<f32>,
}

/// A summary of one parameter tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct ParamStat {
    /// The parameter's name, such as `blocks.0.attn.q_proj.weight`.
    pub name: String,
    /// Its shape.
    pub shape: Vec<usize>,
    /// Root mean square of its values.
    pub rms: f32,
    /// Its gradient norm at the last step before clipping, once a step has run.
    pub grad_norm: Option<f32>,
}

/// A model being trained on a fixed task.
#[derive(Debug)]
pub struct Lab {
    config: LabConfig,
    model: MidManModel,
    optimizer: AdamW,
    schedule: WarmupCosine,
    inputs: Vec<usize>,
    targets: Vec<usize>,
    step: u64,
    last_loss: Option<f32>,
    last_grad_norms: Vec<f32>,
}

impl Lab {
    /// Builds the model and the task.
    ///
    /// # Errors
    ///
    /// Returns an error if the model config is invalid or larger than
    /// [`MAX_PARAMS`], `batch` or `seq` is 0, `seq` exceeds the model's
    /// `max_seq_len`, `batch * seq` exceeds [`MAX_TOKENS`], or a hyperparameter
    /// is out of range.
    pub fn new(config: LabConfig) -> Result<Lab> {
        config.model.validate()?;
        let params = config.model.param_count()?;
        if params > MAX_PARAMS {
            return Err(Error::invalid(format!(
                "the lab builds models up to {MAX_PARAMS} parameters, this one has {params}"
            )));
        }
        let (batch, seq) = (config.batch, config.seq);
        if batch == 0 || seq == 0 {
            return Err(Error::invalid(
                "the lab needs at least one sequence of at least one token",
            ));
        }
        if seq > config.model.max_seq_len {
            return Err(Error::invalid(format!(
                "sequence length {seq} exceeds the model's max_seq_len {}",
                config.model.max_seq_len
            )));
        }
        if !matches!(batch.checked_mul(seq), Some(tokens) if tokens <= MAX_TOKENS) {
            return Err(Error::invalid(format!(
                "the lab task is limited to {MAX_TOKENS} tokens, got {batch} x {seq}"
            )));
        }
        let h = &config.hyper;
        if !(h.clip_norm.is_finite() && h.clip_norm > 0.0) {
            return Err(Error::invalid(format!(
                "clip_norm must be positive and finite, got {}",
                h.clip_norm
            )));
        }
        let schedule = WarmupCosine::new(h.max_lr, h.min_lr, h.warmup_steps, h.total_steps)?;
        let optimizer = AdamW::new(h.beta1, h.beta2, 1e-8, h.weight_decay)?;

        let mut rng = Rng::seed_from_u64(config.seed);
        let model = MidManModel::new(&config.model, &mut rng)?;
        let vocab = config.model.vocab_size;
        let mut inputs = Vec::with_capacity(batch * seq);
        let mut targets = Vec::with_capacity(batch * seq);
        for _ in 0..batch {
            let tokens: Vec<usize> = (0..=seq).map(|_| rng.below(vocab)).collect();
            inputs.extend_from_slice(&tokens[..seq]);
            targets.extend_from_slice(&tokens[1..]);
        }
        Ok(Lab {
            config,
            model,
            optimizer,
            schedule,
            inputs,
            targets,
            step: 0,
            last_loss: None,
            last_grad_norms: Vec::new(),
        })
    }

    /// The configuration the lab was built from.
    pub fn config(&self) -> &LabConfig {
        &self.config
    }

    /// How many steps have run.
    pub fn step_count(&self) -> u64 {
        self.step
    }

    /// The loss measured at the most recent step, once one has run.
    pub fn last_loss(&self) -> Option<f32> {
        self.last_loss
    }

    /// The task's input ids, `batch * seq` of them, one sequence after another.
    pub fn inputs(&self) -> &[usize] {
        &self.inputs
    }

    /// The task's target ids, aligned with [`Lab::inputs`].
    pub fn targets(&self) -> &[usize] {
        &self.targets
    }

    /// The total number of parameters.
    pub fn num_parameters(&self) -> usize {
        self.model.num_parameters()
    }

    /// Runs up to `steps` training steps and reports what each measured.
    ///
    /// Training stops early, with the completed steps still reported, if a loss or
    /// gradient is not a finite number; [`StepReport::stopped`] says why.
    pub fn step(&mut self, steps: usize) -> StepReport {
        let mut records = Vec::with_capacity(steps);
        let mut stopped = None;
        for _ in 0..steps {
            match self.step_once() {
                Ok(record) => records.push(record),
                Err(error) => {
                    stopped = Some(error.to_string());
                    break;
                }
            }
        }
        StepReport { records, stopped }
    }

    fn step_once(&mut self) -> Result<StepRecord> {
        let (batch, seq) = (self.config.batch, self.config.seq);
        let lr = self.schedule.lr(self.step);
        let loss = self.model.loss(&self.inputs, &self.targets, batch, seq)?;
        let loss_value = loss.item()?;
        if !loss_value.is_finite() {
            return Err(Error::invalid(format!("the loss is {loss_value}, not a finite number")));
        }
        let mut grads = loss.backward()?;
        let mut params = self.model.parameters_mut();
        let mut gradients = gather_gradients(&params, &mut grads)?;
        let norms: Vec<f32> = gradients.iter().map(|g| l2_norm(g)).collect();
        let grad_norm = clip_global_norm(&mut gradients, self.config.hyper.clip_norm)?;
        self.optimizer.step(&mut params, &gradients, lr)?;
        self.last_grad_norms = norms;
        self.step += 1;
        self.last_loss = Some(loss_value);
        Ok(StepRecord { loss: loss_value, grad_norm, lr })
    }

    /// Measures the loss and the top prediction at every position, without training.
    ///
    /// # Errors
    ///
    /// Returns an error if the forward pass fails.
    pub fn eval(&self) -> Result<Eval> {
        let (batch, seq) = (self.config.batch, self.config.seq);
        let vocab = self.config.model.vocab_size;
        let logits = self.model.forward(&self.inputs, batch, seq)?;
        let loss = logits.reshape(&[batch * seq, vocab])?.cross_entropy(&self.targets)?.item()?;
        let predictions: Vec<usize> = logits
            .data()
            .chunks_exact(vocab)
            .map(|row| {
                row.iter().enumerate().fold(0, |best, (i, &v)| if v > row[best] { i } else { best })
            })
            .collect();
        let correct = predictions.iter().zip(&self.targets).filter(|(p, t)| p == t).count();
        Ok(Eval { loss, accuracy: correct as f32 / predictions.len() as f32, predictions })
    }

    /// The attention probabilities of every layer and head for one of the task's
    /// sequences.
    ///
    /// # Errors
    ///
    /// Returns an error if `sequence` is not below `batch`.
    pub fn attention(&self, sequence: usize) -> Result<AttentionMaps> {
        let (batch, seq) = (self.config.batch, self.config.seq);
        if sequence >= batch {
            return Err(Error::invalid(format!(
                "sequence {sequence} does not exist, the task has {batch}"
            )));
        }
        let ids = &self.inputs[sequence * seq..(sequence + 1) * seq];
        let (_, probs) = self.model.forward_traced(ids, 1, seq)?;
        let layers = probs.len();
        let heads = self.config.model.n_heads;
        let mut data = Vec::with_capacity(layers * heads * seq * seq);
        for layer in &probs {
            data.extend_from_slice(layer.data());
        }
        Ok(AttentionMaps { layers, heads, seq, data })
    }

    /// The token embedding table as `(rows, columns, values)`, one row per token.
    pub fn embedding(&self) -> (usize, usize, Vec<f32>) {
        let table = self
            .model
            .parameters()
            .into_iter()
            .find(|p| p.name() == "embed_tokens.weight")
            .map(|p| p.data().to_vec())
            .unwrap_or_default();
        (self.config.model.vocab_size, self.config.model.d_model, table)
    }

    /// A summary of every parameter, in the model's fixed order.
    pub fn param_stats(&self) -> Vec<ParamStat> {
        self.model
            .parameters()
            .into_iter()
            .enumerate()
            .map(|(i, p)| ParamStat {
                name: p.name().to_string(),
                shape: p.shape().to_vec(),
                rms: rms(p.data()),
                grad_norm: self.last_grad_norms.get(i).copied(),
            })
            .collect()
    }
}

fn l2_norm(values: &[f32]) -> f32 {
    values.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>().sqrt() as f32
}

fn rms(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    (values.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / values.len() as f64).sqrt()
        as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_model() -> ModelConfig {
        ModelConfig {
            vocab_size: 12,
            d_model: 16,
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 1,
            d_ff: 32,
            max_seq_len: 16,
            rope_theta: 10_000.0,
            norm_eps: 1e-5,
            tie_embeddings: true,
        }
    }

    fn config() -> LabConfig {
        let hyper = Hyper {
            max_lr: 0.03,
            min_lr: 0.003,
            warmup_steps: 15,
            total_steps: 150,
            ..Hyper::default()
        };
        LabConfig { model: tiny_model(), seed: 2024, batch: 2, seq: 12, hyper }
    }

    #[test]
    fn a_lab_builds_with_the_task_the_config_describes() {
        let lab = Lab::new(config()).unwrap();
        assert_eq!(lab.inputs().len(), 24);
        assert_eq!(lab.targets().len(), 24);
        assert!(lab.inputs().iter().all(|&t| t < 12));
        // Targets are the inputs shifted by one inside each sequence.
        assert_eq!(&lab.inputs()[1..12], &lab.targets()[..11]);
        assert_eq!(&lab.inputs()[13..24], &lab.targets()[12..23]);
        assert_eq!(lab.step_count(), 0);
        assert_eq!(lab.last_loss(), None);
        assert_eq!(lab.num_parameters() as u64, tiny_model().param_count().unwrap());
    }

    #[test]
    fn training_learns_the_task() {
        let mut lab = Lab::new(config()).unwrap();
        let before = lab.eval().unwrap();
        assert!(before.accuracy < 0.5, "an untrained model should not know the task");
        let report = lab.step(150);
        assert!(report.stopped.is_none());
        assert_eq!(report.records.len(), 150);
        assert_eq!(lab.step_count(), 150);
        let after = lab.eval().unwrap();
        assert!(after.loss < 0.1 * report.records[0].loss, "loss only fell to {}", after.loss);
        assert!(after.accuracy > 0.95, "accuracy {}", after.accuracy);
        assert_eq!(after.predictions.len(), 24);
        assert_eq!(lab.last_loss(), Some(report.records[149].loss));
    }

    #[test]
    fn each_record_follows_the_schedule_and_measures_real_numbers() {
        let mut lab = Lab::new(config()).unwrap();
        let report = lab.step(20);
        let schedule = WarmupCosine::new(0.03, 0.003, 15, 150).unwrap();
        for (i, r) in report.records.iter().enumerate() {
            assert_eq!(r.lr, schedule.lr(i as u64));
            assert!(r.loss.is_finite() && r.loss > 0.0);
            assert!(r.grad_norm.is_finite() && r.grad_norm > 0.0);
        }
        // Warmup: the rate rises. And the first loss is near the uniform guess.
        assert!(report.records[0].lr < report.records[5].lr);
        assert!((report.records[0].loss - 12f32.ln()).abs() < 0.3);
    }

    #[test]
    fn stepping_in_pieces_equals_stepping_at_once() {
        let mut whole = Lab::new(config()).unwrap();
        let mut pieces = Lab::new(config()).unwrap();
        let all = whole.step(12).records;
        let mut split = pieces.step(5).records;
        split.extend(pieces.step(7).records);
        assert_eq!(all, split);
    }

    #[test]
    fn one_seed_gives_one_training_run_bit_for_bit() {
        let run = |seed: u64| {
            let mut c = config();
            c.seed = seed;
            let mut lab = Lab::new(c).unwrap();
            lab.step(10).records.iter().map(|r| r.loss.to_bits()).collect::<Vec<u32>>()
        };
        assert_eq!(run(5), run(5));
        assert_ne!(run(5), run(6));
    }

    #[test]
    fn parameter_stats_gain_gradient_norms_after_a_step() {
        let mut lab = Lab::new(config()).unwrap();
        let before = lab.param_stats();
        assert_eq!(before.len(), 1 + 2 * 9 + 1);
        assert!(before.iter().all(|p| p.grad_norm.is_none() && p.rms > 0.0));
        assert_eq!(before[0].name, "embed_tokens.weight");
        assert_eq!(before[0].shape, vec![12, 16]);
        lab.step(1);
        let after = lab.param_stats();
        assert!(after.iter().all(|p| p.grad_norm.is_some_and(|g| g.is_finite() && g >= 0.0)));
    }

    #[test]
    fn attention_maps_have_the_documented_layout_and_are_causal() {
        let lab = Lab::new(config()).unwrap();
        let maps = lab.attention(1).unwrap();
        assert_eq!((maps.layers, maps.heads, maps.seq), (2, 2, 12));
        assert_eq!(maps.data.len(), 2 * 2 * 12 * 12);
        for matrix in maps.data.chunks_exact(12 * 12) {
            for (i, row) in matrix.chunks_exact(12).enumerate() {
                assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-5);
                assert!(row[i + 1..].iter().all(|&p| p == 0.0));
            }
        }
        assert!(lab.attention(2).is_err());
    }

    #[test]
    fn the_embedding_has_one_row_per_token() {
        let lab = Lab::new(config()).unwrap();
        let (rows, cols, values) = lab.embedding();
        assert_eq!((rows, cols, values.len()), (12, 16, 12 * 16));
    }

    #[test]
    fn training_that_blows_up_stops_and_says_why() {
        let mut c = config();
        c.hyper.max_lr = 1e30;
        c.hyper.min_lr = 1e30;
        c.hyper.warmup_steps = 0;
        let mut lab = Lab::new(c).unwrap();
        let report = lab.step(50);
        assert!(report.records.len() < 50, "it should have stopped");
        let message = report.stopped.expect("a reason");
        assert!(message.contains("finite"), "{message}");
        // The steps before the failure are still reported.
        assert!(!report.records.is_empty());
    }

    #[test]
    fn bad_configurations_are_errors() {
        let bad = |edit: &dyn Fn(&mut LabConfig)| {
            let mut c = config();
            edit(&mut c);
            Lab::new(c).is_err()
        };
        assert!(bad(&|c| c.batch = 0));
        assert!(bad(&|c| c.seq = 0));
        assert!(bad(&|c| c.seq = 17));
        assert!(bad(&|c| c.batch = 1000));
        assert!(bad(&|c| c.model.n_heads = 3));
        assert!(bad(&|c| c.hyper.clip_norm = 0.0));
        assert!(bad(&|c| c.hyper.max_lr = -1.0));
        assert!(bad(&|c| c.hyper.warmup_steps = 150));
        assert!(bad(&|c| c.hyper.beta1 = 1.0));
    }

    #[test]
    fn a_model_beyond_the_size_limit_is_refused_before_it_is_built() {
        let mut c = config();
        c.model.vocab_size = 100_000;
        c.model.d_model = 128;
        c.model.n_heads = 4;
        c.model.n_kv_heads = 4;
        c.model.d_ff = 256;
        let message = Lab::new(c).unwrap_err().to_string();
        assert!(message.contains("up to"), "{message}");
    }
}
