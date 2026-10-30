// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-training.md, section "examples/train.rs"
// ============================================================================
//! Trains a model on token shards, for a number of steps or a number of minutes, saving
//! its state so that the run can be stopped and continued.
//!
//! ```text
//! cargo run --release -p midman-training --example train -- \
//!     --shards DIR --out DIR --preset tiny --steps 20000 --batch 8 --seq 256 \
//!     [--eval-shards DIR] [--minutes M] [--resume] [--seed N] [--max-lr X] [--min-lr X] \
//!     [--warmup N] [--weight-decay X] [--clip X] [--save-every N] [--eval-every N] \
//!     [--eval-batches N] [--log-every N]
//! ```
//!
//! `--shards` holds the `*.mmtokens` files of the training data (see `build_shards` in
//! `midman-data`); `--eval-shards` the same for held-out data. The model is the preset's,
//! with the vocabulary of the shards. In `--out` the run keeps
//!
//! - `state.safetensors`: the state of the run (weights, optimizer, place in the data),
//!   replaced every `--save-every` steps and at the end;
//! - `weights.safetensors`: the weights alone, written at the end;
//! - `train.tsv`: one line per `--log-every` steps (step, mean loss since the last line,
//!   last gradient norm, learning rate, tokens seen, seconds);
//! - `eval.tsv`: one line per evaluation (step, held-out loss, tokens).
//!
//! `--minutes` stops the run when that much time has passed, after saving; the schedule is
//! still the one for `--steps`, so a run that stopped early continues with `--resume` and
//! the same settings. `--resume` fails if there is no state to continue.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use midman_data::{Sequences, TokenShard};
use midman_foundation::Rng;
use midman_model::MidManModel;
use midman_model_config::Preset;
use midman_training::{TrainConfig, Trainer};

const USAGE: &str = "usage: train --shards DIR --out DIR --preset NAME --steps N --batch N --seq N [--eval-shards DIR] [--minutes M] [--resume] [--seed N] [--max-lr X] [--min-lr X] [--warmup N] [--weight-decay X] [--clip X] [--save-every N] [--eval-every N] [--eval-batches N] [--log-every N]";

struct Args {
    shards: PathBuf,
    eval_shards: Option<PathBuf>,
    out: PathBuf,
    preset: Preset,
    steps: u64,
    batch: usize,
    seq: usize,
    minutes: Option<f64>,
    resume: bool,
    seed: u64,
    max_lr: Option<f32>,
    min_lr: Option<f32>,
    warmup: Option<u64>,
    weight_decay: Option<f32>,
    clip: Option<f32>,
    save_every: u64,
    eval_every: u64,
    eval_batches: usize,
    log_every: u64,
}

fn parse_args() -> Result<Args, String> {
    let mut args = std::env::args().skip(1);
    let mut shards = None;
    let mut eval_shards = None;
    let mut out = None;
    let mut preset = None;
    let (mut steps, mut batch, mut seq) = (None, None, None);
    let mut found = Args {
        shards: PathBuf::new(),
        eval_shards: None,
        out: PathBuf::new(),
        preset: Preset::Smoke,
        steps: 0,
        batch: 0,
        seq: 0,
        minutes: None,
        resume: false,
        seed: 0,
        max_lr: None,
        min_lr: None,
        warmup: None,
        weight_decay: None,
        clip: None,
        save_every: 500,
        eval_every: 500,
        eval_batches: 16,
        log_every: 10,
    };
    while let Some(arg) = args.next() {
        if arg == "--resume" {
            found.resume = true;
            continue;
        }
        let value = args.next().ok_or_else(|| format!("{arg} needs a value\n{USAGE}"))?;
        let bad = |what: &str| format!("`{value}` is not {what} for {arg}");
        match arg.as_str() {
            "--shards" => shards = Some(PathBuf::from(&value)),
            "--eval-shards" => eval_shards = Some(PathBuf::from(&value)),
            "--out" => out = Some(PathBuf::from(&value)),
            "--preset" => preset = Some(value.parse::<Preset>().map_err(|e| e.to_string())?),
            "--steps" => steps = Some(value.parse().map_err(|_| bad("a number"))?),
            "--batch" => batch = Some(value.parse().map_err(|_| bad("a number"))?),
            "--seq" => seq = Some(value.parse().map_err(|_| bad("a number"))?),
            "--minutes" => found.minutes = Some(value.parse().map_err(|_| bad("a number"))?),
            "--seed" => found.seed = value.parse().map_err(|_| bad("a number"))?,
            "--max-lr" => found.max_lr = Some(value.parse().map_err(|_| bad("a number"))?),
            "--min-lr" => found.min_lr = Some(value.parse().map_err(|_| bad("a number"))?),
            "--warmup" => found.warmup = Some(value.parse().map_err(|_| bad("a number"))?),
            "--weight-decay" => {
                found.weight_decay = Some(value.parse().map_err(|_| bad("a number"))?);
            }
            "--clip" => found.clip = Some(value.parse().map_err(|_| bad("a number"))?),
            "--save-every" => found.save_every = value.parse().map_err(|_| bad("a number"))?,
            "--eval-every" => found.eval_every = value.parse().map_err(|_| bad("a number"))?,
            "--eval-batches" => found.eval_batches = value.parse().map_err(|_| bad("a number"))?,
            "--log-every" => found.log_every = value.parse().map_err(|_| bad("a number"))?,
            other => return Err(format!("unknown argument `{other}`\n{USAGE}")),
        }
    }
    let need = |name: &str| format!("{name} is required\n{USAGE}");
    found.shards = shards.ok_or_else(|| need("--shards"))?;
    found.eval_shards = eval_shards;
    found.out = out.ok_or_else(|| need("--out"))?;
    found.preset = preset.ok_or_else(|| need("--preset"))?;
    found.steps = steps.ok_or_else(|| need("--steps"))?;
    found.batch = batch.ok_or_else(|| need("--batch"))?;
    found.seq = seq.ok_or_else(|| need("--seq"))?;
    if found.save_every == 0 || found.log_every == 0 || found.eval_every == 0 {
        return Err("--save-every, --eval-every and --log-every must be above zero".to_string());
    }
    Ok(found)
}

/// Every `*.mmtokens` file of a directory, in name order, as windows of `seq` tokens.
fn load_sequences(dir: &Path, seq: usize) -> Result<Sequences, String> {
    let mut paths: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|ext| ext == "mmtokens"))
        .collect();
    paths.sort();
    if paths.is_empty() {
        return Err(format!("{}: no .mmtokens files", dir.display()));
    }
    let mut shards = Vec::new();
    for path in &paths {
        shards.push(TokenShard::load(path).map_err(|e| format!("{}: {e}", path.display()))?);
    }
    Sequences::new(shards, seq).map_err(|e| format!("{}: {e}", dir.display()))
}

fn append(path: &Path, line: &str) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    writeln!(file, "{line}").map_err(|e| format!("{}: {e}", path.display()))
}

fn run(args: &Args) -> Result<(), String> {
    fs::create_dir_all(&args.out).map_err(|e| format!("{}: {e}", args.out.display()))?;
    let train = load_sequences(&args.shards, args.seq)?;
    let vocab = train.vocab_size().ok_or("the training data has no tokens")? as usize;
    let eval = match &args.eval_shards {
        Some(dir) => Some(load_sequences(dir, args.seq)?),
        None => None,
    };
    eprintln!(
        "train: {} sequences of {} tokens, vocabulary {vocab}{}",
        train.len(),
        args.seq,
        eval.as_ref().map_or(String::new(), |e| format!(", {} held-out sequences", e.len()))
    );

    let mut config = TrainConfig::new(args.batch, args.seq, args.steps);
    config.seed = args.seed;
    if let Some(v) = args.max_lr {
        config.max_lr = v;
        config.min_lr = config.min_lr.min(v);
    }
    if let Some(v) = args.min_lr {
        config.min_lr = v;
    }
    if let Some(v) = args.warmup {
        config.warmup_steps = v;
    }
    if let Some(v) = args.weight_decay {
        config.weight_decay = v;
    }
    if let Some(v) = args.clip {
        config.clip_norm = v;
    }

    let model_config = args.preset.config(vocab);
    let mut rng = Rng::seed_from_u64(args.seed);
    let model = MidManModel::new(&model_config, &mut rng).map_err(|e| e.to_string())?;
    eprintln!("train: preset {}, {} parameters", args.preset.name(), model.num_parameters());
    let mut trainer = Trainer::new(model, config, train).map_err(|e| e.to_string())?;

    let state_path = args.out.join("state.safetensors");
    let weights_path = args.out.join("weights.safetensors");
    let train_log = args.out.join("train.tsv");
    let eval_log = args.out.join("eval.tsv");
    if args.resume {
        trainer.resume(&state_path).map_err(|e| format!("cannot resume: {e}"))?;
        eprintln!("train: resumed at step {}", trainer.step_count());
    } else {
        if !train_log.exists() {
            append(&train_log, "step\tloss\tgrad_norm\tlr\ttokens\tseconds")?;
        }
        if !eval_log.exists() && eval.is_some() {
            append(&eval_log, "step\teval_loss\ttokens")?;
        }
    }

    let budget = args.minutes.map(|m| Duration::from_secs_f64(m * 60.0));
    let started = Instant::now();
    let steps_at_start = trainer.step_count();
    let mut window_loss = 0.0f64;
    let mut window_steps = 0u64;
    let mut stopped_early = false;
    let mut last_eval_step = None;

    while !trainer.done() {
        let report =
            trainer.step().map_err(|e| format!("step {}: {e}", trainer.step_count() + 1))?;
        window_loss += f64::from(report.loss);
        window_steps += 1;
        let step = report.step;
        let seconds = started.elapsed().as_secs_f64();
        if step % args.log_every == 0 {
            let mean = window_loss / window_steps as f64;
            let rate =
                (step - steps_at_start) as f64 * (args.batch * args.seq) as f64 / seconds.max(1e-9);
            eprintln!(
                "step {step}/{} loss {mean:.4} grad {:.3} lr {:.2e} {rate:.0} tok/s {seconds:.0}s",
                args.steps, report.grad_norm, report.lr
            );
            append(
                &train_log,
                &format!(
                    "{step}\t{mean:.6}\t{:.6}\t{:.6e}\t{}\t{seconds:.1}",
                    report.grad_norm,
                    report.lr,
                    trainer.tokens_seen()
                ),
            )?;
            window_loss = 0.0;
            window_steps = 0;
        }
        if let Some(eval) = &eval {
            if step % args.eval_every == 0 {
                let result =
                    trainer.evaluate(eval, args.eval_batches).map_err(|e| e.to_string())?;
                eprintln!(
                    "step {step}: held-out loss {:.4} over {} tokens",
                    result.loss, result.tokens
                );
                append(&eval_log, &format!("{step}\t{:.6}\t{}", result.loss, result.tokens))?;
                last_eval_step = Some(step);
            }
        }
        if step % args.save_every == 0 {
            trainer.save_state(&state_path).map_err(|e| e.to_string())?;
        }
        if budget.is_some_and(|b| started.elapsed() >= b) && !trainer.done() {
            stopped_early = true;
            break;
        }
    }

    trainer.save_state(&state_path).map_err(|e| e.to_string())?;
    trainer.export_weights(weights_path).map_err(|e| e.to_string())?;
    if let Some(eval) = eval.as_ref().filter(|_| last_eval_step != Some(trainer.step_count())) {
        let result = trainer.evaluate(eval, args.eval_batches).map_err(|e| e.to_string())?;
        eprintln!("train: final held-out loss {:.4} over {} tokens", result.loss, result.tokens);
        append(
            &eval_log,
            &format!("{}\t{:.6}\t{}", trainer.step_count(), result.loss, result.tokens),
        )?;
    }
    eprintln!(
        "train: {} at step {} of {}, {} tokens, {:.0}s; state in {}",
        if stopped_early { "stopped on the time budget" } else { "finished" },
        trainer.step_count(),
        args.steps,
        trainer.tokens_seen(),
        started.elapsed().as_secs_f64(),
        args.out.display()
    );
    // A last line a script can read.
    let mut summary =
        File::create(args.out.join("summary.txt")).map_err(|e| format!("summary.txt: {e}"))?;
    writeln!(
        summary,
        "steps {}\ntotal_steps {}\ntokens {}\nfinished {}",
        trainer.step_count(),
        args.steps,
        trainer.tokens_seen(),
        !stopped_early
    )
    .map_err(|e| format!("summary.txt: {e}"))
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::from(2);
        }
    };
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("train: {message}");
            ExitCode::FAILURE
        }
    }
}
