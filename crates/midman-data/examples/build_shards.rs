// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-data.md, section "examples/build_shards.rs"
// ============================================================================
//! Builds token shards from text files: scrub, deduplicate, tokenize, pack.
//!
//! ```text
//! cargo run --release -p midman-data --example build_shards -- \
//!     --tokenizer tokenizer.txt --out shards/ [--shard-tokens 100000000] [--seq-len 2048] \
//!     [--no-scrub]  < list-of-paths
//! ```
//!
//! Each path on standard input is one document. A file that is not valid UTF-8, and one that
//! is empty, is skipped and counted. Text is scrubbed of secrets first (unless `--no-scrub`),
//! then compared by SHA-256 with the documents already kept: the first copy wins. What
//! remains is tokenized with the special-token text read as ordinary text, followed by the
//! end-of-text token, and written as `shard-00000.mmtokens`, `shard-00001.mmtokens` and so
//! on. A report goes to standard error, ending with a SHA-256 of the first 100 batches the
//! batcher makes from the shards (batches of 4, seed 1), which is the same on every platform
//! if the whole path is. The license check is not done here: pass only paths of sources the
//! manifest and the license policy accepted.

use std::collections::BTreeMap;
use std::fs;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::time::Instant;

use midman_data::{to_hex, Batcher, ExactDedup, Packer, Scrubber, Sequences, Sha256, TokenShard};
use midman_tokenizer::Tokenizer;

/// Batches hashed for the report digest, and their shape.
const DIGEST_BATCHES: usize = 100;
const DIGEST_BATCH_SIZE: usize = 4;
const DIGEST_SEED: u64 = 1;

const USAGE: &str = "usage: build_shards --tokenizer FILE --out DIR [--shard-tokens N] [--seq-len N] [--no-scrub]  < paths";

struct Args {
    tokenizer: PathBuf,
    out: PathBuf,
    shard_tokens: usize,
    seq_len: usize,
    scrub: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut args = std::env::args().skip(1);
    let (mut tokenizer, mut out) = (None, None);
    let (mut shard_tokens, mut seq_len, mut scrub) = (100_000_000usize, 2048usize, true);
    while let Some(arg) = args.next() {
        let mut value =
            |name: &str| args.next().ok_or_else(|| format!("{name} needs a value\n{USAGE}"));
        let number = |name: &str, text: String| {
            text.parse::<usize>().map_err(|_| format!("`{text}` is not a number for {name}"))
        };
        match arg.as_str() {
            "--tokenizer" => tokenizer = Some(PathBuf::from(value("--tokenizer")?)),
            "--out" => out = Some(PathBuf::from(value("--out")?)),
            "--shard-tokens" => shard_tokens = number("--shard-tokens", value("--shard-tokens")?)?,
            "--seq-len" => seq_len = number("--seq-len", value("--seq-len")?)?,
            "--no-scrub" => scrub = false,
            other => return Err(format!("unknown argument `{other}`\n{USAGE}")),
        }
    }
    Ok(Args {
        tokenizer: tokenizer.ok_or_else(|| format!("--tokenizer is required\n{USAGE}"))?,
        out: out.ok_or_else(|| format!("--out is required\n{USAGE}"))?,
        shard_tokens,
        seq_len,
        scrub,
    })
}

fn shard_path(dir: &Path, n: usize) -> PathBuf {
    dir.join(format!("shard-{n:05}.mmtokens"))
}

fn write_shard(shard: &TokenShard, dir: &Path, written: &mut usize) -> Result<(), String> {
    shard.save(shard_path(dir, *written)).map_err(|e| e.to_string())?;
    *written += 1;
    Ok(())
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let tokenizer = Tokenizer::load(&args.tokenizer).map_err(|e| e.to_string())?;
    let mut packer =
        Packer::for_tokenizer(&tokenizer, args.shard_tokens).map_err(|e| e.to_string())?;
    fs::create_dir_all(&args.out).map_err(|e| format!("{}: {e}", args.out.display()))?;
    if shard_path(&args.out, 0).exists() {
        return Err(format!("{} already holds shards; use an empty directory", args.out.display()));
    }

    let scrubber = Scrubber::new();
    let mut dedup = ExactDedup::new();
    let (mut read, mut not_utf8, mut empty, mut bytes_kept) = (0u64, 0u64, 0u64, 0u64);
    let mut secrets: BTreeMap<&str, u64> = BTreeMap::new();
    let mut written = 0usize;
    let start = Instant::now();

    for path in std::io::stdin().lock().lines().map_while(Result::ok) {
        let Ok(raw) = fs::read(&path) else {
            return Err(format!("cannot read {path}"));
        };
        read += 1;
        let Ok(text) = String::from_utf8(raw) else {
            not_utf8 += 1;
            continue;
        };
        let text = if args.scrub {
            let scrubbed = scrubber.scrub(&text);
            for finding in &scrubbed.findings {
                *secrets.entry(finding.rule).or_default() += 1;
            }
            scrubbed.text
        } else {
            text
        };
        if text.is_empty() {
            empty += 1;
            continue;
        }
        if !dedup.insert(text.as_bytes()) {
            continue;
        }
        bytes_kept += text.len() as u64;
        if let Some(shard) = packer.add_text(&tokenizer, &text).map_err(|e| e.to_string())? {
            write_shard(&shard, &args.out, &mut written)?;
        }
    }
    let documents = packer.documents();
    let tokens = packer.tokens();
    if let Some(shard) = packer.finish() {
        write_shard(&shard, &args.out, &mut written)?;
    }

    // Read every shard back (which checks its checksum) and count the training sequences.
    let mut shards = Vec::with_capacity(written);
    for n in 0..written {
        shards.push(TokenShard::load(shard_path(&args.out, n)).map_err(|e| e.to_string())?);
    }
    let sequences = Sequences::new(shards, args.seq_len).map_err(|e| e.to_string())?;
    let batch_digest = batch_digest(&sequences);
    let seconds = start.elapsed().as_secs_f64();
    let secret_total: u64 = secrets.values().sum();
    eprintln!(
        "files read {read}: not UTF-8 {not_utf8}, empty {empty}, duplicates {}",
        dedup.duplicates()
    );
    eprintln!("secrets replaced {secret_total} {secrets:?}");
    eprintln!(
        "documents kept {documents}: {bytes_kept} bytes, {tokens} tokens ({:.2} bytes per token)",
        bytes_kept as f64 / tokens.max(1) as f64
    );
    eprintln!(
        "{written} shards in {}; {} sequences of {} tokens; {seconds:.1} s, {:.1} MB/s",
        args.out.display(),
        sequences.len(),
        args.seq_len,
        bytes_kept as f64 / 1e6 / seconds.max(1e-9)
    );
    match batch_digest {
        Some((batches, digest)) => eprintln!(
            "first {batches} batches ({DIGEST_BATCH_SIZE} x {} tokens, seed {DIGEST_SEED}): sha256 {digest}",
            args.seq_len
        ),
        None => eprintln!("batch digest: fewer than {DIGEST_BATCH_SIZE} sequences, no batches"),
    }
    Ok(())
}

/// The SHA-256 of the inputs and targets of the first batches the batcher makes, each id as
/// four little-endian bytes, and how many batches that was.
fn batch_digest(sequences: &Sequences) -> Option<(usize, String)> {
    let mut batcher = Batcher::new(sequences.clone(), DIGEST_BATCH_SIZE, DIGEST_SEED).ok()?;
    let batches = DIGEST_BATCHES.min(batcher.batches_per_epoch());
    let mut hasher = Sha256::new();
    for _ in 0..batches {
        let batch = batcher.next_batch();
        for &id in batch.inputs.iter().chain(&batch.targets) {
            hasher.update(&(id as u32).to_le_bytes());
        }
    }
    Some((batches, to_hex(&hasher.finalize())))
}

fn main() {
    if let Err(message) = run() {
        eprintln!("{message}");
        std::process::exit(2);
    }
}
