// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-tokenizer.md, section "examples/train.rs"
// ============================================================================
//! Trains a tokenizer from files and reports how well it compresses held-out files.
//!
//! ```text
//! cargo run --release -p midman-tokenizer --example train -- \
//!     --vocab 32000 --out tokenizer.txt \
//!     --special '<|endoftext|>' --eval held-out-dir  training-dir another-file
//! ```
//!
//! Every file under the training paths is one document. Directories are walked in sorted
//! order, and names starting with a dot are skipped. The evaluation also checks that
//! every file encodes and decodes back to exactly its own bytes.

use std::fs;
use std::path::{Path, PathBuf};

use midman_tokenizer::{PreTokenizerConfig, SpecialMode, Tokenizer, Trainer, TrainerConfig};

const USAGE: &str = "usage: train --vocab N --out FILE [--special TEXT]... [--min-frequency N]
             [--pretokenizer default|gpt2|cl100k] [--eval PATH]... PATH...";

struct Args {
    vocab: usize,
    out: PathBuf,
    specials: Vec<String>,
    min_frequency: u64,
    pretokenizer: PreTokenizerConfig,
    eval: Vec<PathBuf>,
    inputs: Vec<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut args = std::env::args().skip(1);
    let mut vocab = None;
    let mut out = None;
    let mut specials = Vec::new();
    let mut min_frequency = 2;
    let mut pretokenizer = PreTokenizerConfig::default();
    let mut eval = Vec::new();
    let mut inputs = Vec::new();
    while let Some(arg) = args.next() {
        let mut value =
            |name: &str| args.next().ok_or_else(|| format!("{name} needs a value\n{USAGE}"));
        match arg.as_str() {
            "--vocab" => {
                let text = value("--vocab")?;
                vocab =
                    Some(text.parse().map_err(|_| format!("`{text}` is not a vocabulary size"))?);
            }
            "--out" => out = Some(PathBuf::from(value("--out")?)),
            "--special" => specials.push(value("--special")?),
            "--min-frequency" => {
                let text = value("--min-frequency")?;
                min_frequency = text.parse().map_err(|_| format!("`{text}` is not a number"))?;
            }
            "--pretokenizer" => {
                pretokenizer = match value("--pretokenizer")?.as_str() {
                    "default" => PreTokenizerConfig::default(),
                    "gpt2" => PreTokenizerConfig::gpt2_like(),
                    "cl100k" => PreTokenizerConfig::cl100k_like(),
                    other => return Err(format!("unknown pre-tokenizer `{other}`\n{USAGE}")),
                };
            }
            "--eval" => eval.push(PathBuf::from(value("--eval")?)),
            "--help" | "-h" => return Err(USAGE.to_string()),
            flag if flag.starts_with("--") => {
                return Err(format!("unknown flag `{flag}`\n{USAGE}"))
            }
            path => inputs.push(PathBuf::from(path)),
        }
    }
    let vocab = vocab.ok_or_else(|| format!("--vocab is required\n{USAGE}"))?;
    let out = out.ok_or_else(|| format!("--out is required\n{USAGE}"))?;
    if inputs.is_empty() {
        return Err(format!("give at least one file or directory to train on\n{USAGE}"));
    }
    Ok(Args { vocab, out, specials, min_frequency, pretokenizer, eval, inputs })
}

/// Collects the files under `path`, directories in sorted order, skipping dotfiles and
/// symbolic links.
fn collect(path: &Path, files: &mut Vec<PathBuf>) -> Result<(), String> {
    let meta = fs::symlink_metadata(path).map_err(|err| format!("{}: {err}", path.display()))?;
    if meta.file_type().is_symlink() {
        return Ok(());
    }
    if meta.is_file() {
        files.push(path.to_path_buf());
        return Ok(());
    }
    let mut entries: Vec<PathBuf> = fs::read_dir(path)
        .map_err(|err| format!("{}: {err}", path.display()))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| !p.file_name().is_some_and(|n| n.to_string_lossy().starts_with('.')))
        .collect();
    entries.sort();
    for entry in entries {
        collect(&entry, files)?;
    }
    Ok(())
}

fn files_under(paths: &[PathBuf]) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    for path in paths {
        collect(path, &mut files)?;
    }
    Ok(files)
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let mut config = TrainerConfig::new(args.vocab);
    config.pretokenizer = args.pretokenizer;
    config.special_tokens = args.specials;
    config.min_frequency = args.min_frequency;
    let mut trainer = Trainer::new(config).map_err(|err| err.to_string())?;

    let files = files_under(&args.inputs)?;
    let mut skipped = 0usize;
    for file in &files {
        match fs::read(file) {
            Ok(bytes) => trainer.add(&bytes),
            Err(_) => skipped += 1,
        }
    }
    let (tokenizer, stats) = trainer.finish().map_err(|err| err.to_string())?;
    tokenizer.save(&args.out).map_err(|err| err.to_string())?;
    println!(
        "trained on {} files ({} unreadable): {} bytes, {} pieces, {} distinct",
        files.len() - skipped,
        skipped,
        stats.bytes_seen,
        stats.pieces_seen,
        stats.unique_pieces
    );
    println!(
        "learned {} of {} merges, vocabulary {}, the last merge was seen {} times; saved to {}",
        stats.merges,
        stats.requested_merges,
        tokenizer.vocab_size(),
        stats.merge_counts.last().copied().unwrap_or(0),
        args.out.display()
    );
    if stats.stopped_early() {
        println!(
            "warning: ran out of pairs seen at least {} times before reaching the vocabulary",
            args.min_frequency
        );
    }
    if !args.eval.is_empty() {
        evaluate(&tokenizer, &files_under(&args.eval)?);
    }
    Ok(())
}

fn evaluate(tokenizer: &Tokenizer, files: &[PathBuf]) {
    let (mut bytes, mut tokens, mut failures, mut read) = (0usize, 0usize, 0usize, 0usize);
    for file in files {
        let Ok(data) = fs::read(file) else { continue };
        read += 1;
        let ids = tokenizer.encode_bytes(&data, SpecialMode::AsText);
        if tokenizer.decode_bytes(&ids).map(|back| back != data).unwrap_or(true) {
            failures += 1;
        }
        bytes += data.len();
        tokens += ids.len();
    }
    let per_token = if tokens == 0 { 0.0 } else { bytes as f64 / tokens as f64 };
    println!("evaluated {read} files: {bytes} bytes in {tokens} tokens, {per_token:.3} bytes per token, {failures} round-trip failures");
}

fn main() {
    if let Err(message) = run() {
        eprintln!("{message}");
        std::process::exit(2);
    }
}
