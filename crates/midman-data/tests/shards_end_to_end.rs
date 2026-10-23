// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-data.md, section "tests/shards_end_to_end.rs"
// ============================================================================
//! The whole path from text to a model's loss: scrub, deduplicate, tokenize with the agreed
//! special tokens, pack into shards on disk, read them back, cut sequences, batch, and give
//! a batch to the model.

use std::fs;

use midman_data::{Batcher, ExactDedup, Packer, Scrubber, Sequences, TokenShard, REDACTION};
use midman_foundation::Rng;
use midman_model::MidManModel;
use midman_model_config::Preset;
use midman_tokenizer::{
    default_special_tokens, SpecialMode, Tokenizer, Trainer, TrainerConfig, END_OF_TEXT,
};

/// A key-looking value built at run time, so that no line of this file looks like a credential.
fn token(seed: u32) -> String {
    let body: String = (0..36)
        .map(|i| {
            let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
            char::from(alphabet[(seed as usize * 7 + i * 13 + i * i) % alphabet.len()])
        })
        .collect();
    format!("{}{body}", ["gh", "p_"].concat())
}

/// Forty different small functions, ten exact repeats of earlier ones, and four scripts that
/// differ only by the token they hold.
fn corpus() -> Vec<String> {
    let mut docs: Vec<String> = (0..40)
        .map(|n| {
            format!(
                "fn item_{n}(x: u32) -> u32 {{\n    let y = x * {n} + {};\n    y\n}}\n",
                n * 3 + 1
            )
        })
        .collect();
    for n in 0..10 {
        docs.push(docs[n * 3].clone());
    }
    for n in 0..4 {
        docs.push(format!("export GITHUB_TOKEN={}\ncargo publish --registry main\n", token(n)));
    }
    docs
}

fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("midman-data-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn text_to_shards_to_batches_to_the_models_loss() {
    let scrubber = Scrubber::new();
    let scrubbed: Vec<String> = corpus().iter().map(|d| scrubber.scrub(d).text).collect();

    // The four scripts become one document once their token is gone.
    let mut dedup = ExactDedup::new();
    let kept: Vec<&String> = scrubbed.iter().filter(|d| dedup.insert(d.as_bytes())).collect();
    assert_eq!(kept.len(), 40 + 1);
    assert_eq!(dedup.duplicates(), 10 + 3);
    assert!(kept.last().unwrap().contains(REDACTION));

    // A real trained vocabulary with the agreed special tokens.
    let mut config = TrainerConfig::new(300);
    config.special_tokens = default_special_tokens();
    let mut trainer = Trainer::new(config).unwrap();
    for doc in &kept {
        trainer.add(doc.as_bytes());
    }
    let (tokenizer, _) = trainer.finish().unwrap();
    let vocab = tokenizer.vocab_size();
    assert!(vocab > 264, "some merges were learned: {vocab}");
    let eos = tokenizer.special_id(END_OF_TEXT).unwrap();

    // Pack into small shards, write them, read them back.
    let mut packer = Packer::for_tokenizer(&tokenizer, 400).unwrap();
    let mut shards = Vec::new();
    for doc in &kept {
        shards.extend(packer.add_text(&tokenizer, doc).unwrap());
    }
    let packed_tokens = packer.tokens();
    shards.extend(packer.finish());
    assert!(shards.len() >= 2, "{} shards", shards.len());
    assert!(shards[..shards.len() - 1].iter().all(|s| s.len() >= 400));
    assert_eq!(shards.iter().map(|s| s.len() as u64).sum::<u64>(), packed_tokens);

    let dir = temp_dir("end-to-end");
    for (n, shard) in shards.iter().enumerate() {
        shard.save(dir.join(format!("shard-{n:05}.mmtokens"))).unwrap();
    }
    let loaded: Vec<TokenShard> = (0..shards.len())
        .map(|n| TokenShard::load(dir.join(format!("shard-{n:05}.mmtokens"))).unwrap())
        .collect();
    assert_eq!(loaded, shards);

    // The stream is exactly the kept documents, in order, each ended by end-of-text, with no
    // secret in it.
    let stream: Vec<u32> = loaded.iter().flat_map(|s| s.tokens().iter().copied()).collect();
    let decoded: Vec<String> = stream
        .split(|&t| t == eos)
        .filter(|part| !part.is_empty())
        .map(|part| tokenizer.decode(part).unwrap())
        .collect();
    assert_eq!(decoded.len(), kept.len());
    for (got, want) in decoded.iter().zip(&kept) {
        assert_eq!(got, *want);
    }
    assert_eq!(stream.iter().filter(|&&t| t == eos).count(), kept.len());
    assert!(!decoded.join("").contains("ghp_"));

    // Sequences and batches.
    let seq_len = 32;
    let sequences = Sequences::new(loaded, seq_len).unwrap();
    assert!(sequences.len() >= 8, "{} sequences", sequences.len());
    let mut batcher = Batcher::new(sequences, 4, 1234).unwrap();
    let mut first = None;
    for _ in 0..batcher.batches_per_epoch() * 2 {
        let batch = batcher.next_batch();
        assert!(batch.inputs.iter().chain(&batch.targets).all(|&t| t < vocab));
        for row in 0..batch.batch {
            let inputs = &batch.inputs[row * seq_len..(row + 1) * seq_len];
            let targets = &batch.targets[row * seq_len..(row + 1) * seq_len];
            assert_eq!(&inputs[1..], &targets[..seq_len - 1]);
        }
        first.get_or_insert(batch);
    }

    // The model takes the batch as it is: an untrained model's loss is about ln(vocabulary).
    let batch = first.unwrap();
    let model_config = Preset::Smoke.config(vocab);
    let model = MidManModel::new(&model_config, &mut Rng::seed_from_u64(7)).unwrap();
    let loss = model.loss(&batch.inputs, &batch.targets, batch.batch, batch.seq).unwrap();
    let loss = loss.item().unwrap();
    let uniform = (vocab as f32).ln();
    assert!(loss.is_finite() && (loss - uniform).abs() < 0.5, "loss {loss}, ln(vocab) {uniform}");
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn many_documents_of_every_length_survive_packing_exactly() {
    let tokenizer =
        Tokenizer::new(Default::default(), Vec::new(), default_special_tokens()).unwrap();
    let mut rng = Rng::seed_from_u64(99);
    let docs: Vec<String> = (0..2000)
        .map(|_| {
            let length = rng.below(300);
            (0..length).map(|_| char::from(b'a' + rng.below(26) as u8)).collect()
        })
        .collect();

    let mut packer = Packer::for_tokenizer(&tokenizer, 5000).unwrap();
    let mut shards = Vec::new();
    for doc in &docs {
        shards.extend(packer.add_text(&tokenizer, doc).unwrap());
    }
    let non_empty = docs.iter().filter(|d| !d.is_empty()).count();
    assert_eq!(packer.documents(), non_empty as u64);
    shards.extend(packer.finish());

    let eos = tokenizer.special_id(END_OF_TEXT).unwrap();
    let stream: Vec<u32> = shards.iter().flat_map(|s| s.tokens().iter().copied()).collect();
    let got: Vec<String> = stream
        .split(|&t| t == eos)
        .filter(|p| !p.is_empty())
        .map(|p| tokenizer.decode(p).unwrap())
        .collect();
    let want: Vec<&String> = docs.iter().filter(|d| !d.is_empty()).collect();
    assert_eq!(got.len(), want.len());
    assert!(got.iter().zip(want).all(|(g, w)| g == w));

    // Whatever the document sizes, shards close at the first document end at or past 5000.
    for shard in &shards[..shards.len() - 1] {
        assert!(shard.len() >= 5000);
        assert_eq!(*shard.tokens().last().unwrap(), eos);
    }
    // And the sequences cover every token but the few left at the end of each shard.
    let sequences = Sequences::new(shards.clone(), 64).unwrap();
    let expected: usize =
        shards.iter().map(|s| if s.len() > 64 { (s.len() - 1) / 64 } else { 0 }).sum();
    assert_eq!(sequences.len(), expected);
    // Recognizing special tokens would have changed the ids: the data path uses AsText.
    let injected = tokenizer.encode("a<|endoftext|>b", SpecialMode::Recognize);
    assert!(injected.contains(&eos));
}
