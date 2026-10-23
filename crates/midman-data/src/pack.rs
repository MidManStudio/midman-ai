// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-data.md, section "pack.rs"
// ============================================================================
//! Packing documents into token shards.

use midman_foundation::{Error, Result};
use midman_tokenizer::{SpecialMode, Tokenizer, END_OF_TEXT};

use crate::shard::TokenShard;

/// Turns documents into [`TokenShard`]s: every document's tokens followed by the end-of-text
/// token, one document after another.
///
/// ```
/// use midman_data::Packer;
///
/// // A vocabulary of 300 ids whose end-of-text token is 256, and shards of about 6 tokens.
/// let mut packer = Packer::new(300, 256, 6).unwrap();
/// assert!(packer.add_document(&[1, 2, 3]).unwrap().is_none());
/// let shard = packer.add_document(&[4, 5]).unwrap().expect("6 tokens: the shard is full");
/// assert_eq!(shard.tokens(), [1, 2, 3, 256, 4, 5, 256]);
/// assert!(packer.finish().is_none());
/// ```
///
/// A shard is closed at the end of the first document that brings it to the target size or
/// beyond, so documents are never cut and a shard is a little larger than the target. The
/// last, smaller shard comes from [`Packer::finish`].
#[derive(Debug)]
pub struct Packer {
    vocab_size: u32,
    eos: u32,
    shard_tokens: usize,
    buffer: Vec<u32>,
    documents: u64,
    tokens: u64,
}

impl Packer {
    /// A packer for a vocabulary of `vocab_size` ids with `eos` as the end-of-text id, closing
    /// a shard once it holds `shard_tokens` tokens or more.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] if `eos` is not below `vocab_size` or if
    /// `shard_tokens` is 0.
    pub fn new(vocab_size: u32, eos: u32, shard_tokens: usize) -> Result<Packer> {
        if eos >= vocab_size {
            return Err(Error::invalid(format!(
                "the end-of-text id {eos} is not below the vocabulary size {vocab_size}"
            )));
        }
        if shard_tokens == 0 {
            return Err(Error::invalid("a shard must hold at least one token"));
        }
        Ok(Packer { vocab_size, eos, shard_tokens, buffer: Vec::new(), documents: 0, tokens: 0 })
    }

    /// A packer for `tokenizer`: its vocabulary size and its `<|endoftext|>` token.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] if the tokenizer has no `<|endoftext|>` token (a
    /// tokenizer made with `Tokenizer::bytes_only` has none), if its vocabulary does not fit
    /// in 32 bits, or if `shard_tokens` is 0.
    pub fn for_tokenizer(tokenizer: &Tokenizer, shard_tokens: usize) -> Result<Packer> {
        let eos = tokenizer.special_id(END_OF_TEXT).ok_or_else(|| {
            Error::invalid(format!(
                "the tokenizer has no `{END_OF_TEXT}` token to end documents with"
            ))
        })?;
        let vocab_size = u32::try_from(tokenizer.vocab_size())
            .map_err(|_| Error::invalid("the vocabulary does not fit in 32 bits"))?;
        Packer::new(vocab_size, eos, shard_tokens)
    }

    /// Adds one document given as token ids. An empty document is ignored. Returns the shard
    /// this document completed, if it did.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`], and adds nothing, if a token is not below the
    /// vocabulary size or if the end-of-text id appears inside the document (it would end it
    /// early; text encoded with `SpecialMode::AsText` never contains it).
    pub fn add_document(&mut self, ids: &[u32]) -> Result<Option<TokenShard>> {
        if ids.is_empty() {
            return Ok(None);
        }
        if let Some(at) = ids.iter().position(|&t| t >= self.vocab_size) {
            return Err(Error::invalid(format!(
                "token {} at position {at} of the document is not below the vocabulary size {}",
                ids[at], self.vocab_size
            )));
        }
        if let Some(at) = ids.iter().position(|&t| t == self.eos) {
            return Err(Error::invalid(format!(
                "the end-of-text token {} appears inside a document, at position {at}",
                self.eos
            )));
        }
        self.buffer.extend_from_slice(ids);
        self.buffer.push(self.eos);
        self.documents += 1;
        self.tokens += ids.len() as u64 + 1;
        if self.buffer.len() >= self.shard_tokens {
            return self.close().map(Some);
        }
        Ok(None)
    }

    /// Encodes `text` with `tokenizer` and adds it as one document. Special-token text inside
    /// `text` is read as ordinary text (`SpecialMode::AsText`), so a document can never
    /// inject a control token. Returns the shard this document completed, if it did.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] if `tokenizer` does not have this packer's
    /// vocabulary size and end-of-text id.
    pub fn add_text(&mut self, tokenizer: &Tokenizer, text: &str) -> Result<Option<TokenShard>> {
        if tokenizer.vocab_size() as u64 != u64::from(self.vocab_size)
            || tokenizer.special_id(END_OF_TEXT) != Some(self.eos)
        {
            return Err(Error::invalid(
                "the tokenizer is not the one this packer was made for (vocabulary or end-of-text differ)",
            ));
        }
        let ids = tokenizer.encode(text, SpecialMode::AsText);
        self.add_document(&ids)
    }

    /// Ends the packing and returns the tokens still waiting, if there are any.
    pub fn finish(mut self) -> Option<TokenShard> {
        if self.buffer.is_empty() {
            None
        } else {
            self.close().ok()
        }
    }

    /// The number of documents added, not counting empty ones.
    pub fn documents(&self) -> u64 {
        self.documents
    }

    /// The number of tokens added so far, end-of-text tokens included.
    pub fn tokens(&self) -> u64 {
        self.tokens
    }

    fn close(&mut self) -> Result<TokenShard> {
        TokenShard::new(self.vocab_size, self.eos, std::mem::take(&mut self.buffer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use midman_tokenizer::{default_special_tokens, PreTokenizerConfig};

    fn tokenizer() -> Tokenizer {
        Tokenizer::new(PreTokenizerConfig::default(), Vec::new(), default_special_tokens()).unwrap()
    }

    fn packer(shard_tokens: usize) -> Packer {
        Packer::new(300, 256, shard_tokens).unwrap()
    }

    #[test]
    fn documents_are_followed_by_end_of_text_and_shards_close_at_the_target() {
        let mut p = packer(5);
        assert!(p.add_document(&[1, 2]).unwrap().is_none());
        let first = p.add_document(&[3, 4]).unwrap().unwrap();
        // 2+1 then 2+1 = 6 tokens, the first time the total reaches 5 or more.
        assert_eq!(first.tokens(), [1, 2, 256, 3, 4, 256]);
        assert!(p.add_document(&[5]).unwrap().is_none());
        let rest = p.finish().unwrap();
        assert_eq!(rest.tokens(), [5, 256]);
    }

    #[test]
    fn a_document_longer_than_the_target_is_one_shard_by_itself() {
        let mut p = packer(3);
        let shard = p.add_document(&[1, 2, 3, 4, 5, 6, 7]).unwrap().unwrap();
        assert_eq!(shard.tokens(), [1, 2, 3, 4, 5, 6, 7, 256]);
        assert!(p.finish().is_none());
    }

    #[test]
    fn a_shard_closes_exactly_at_the_target_too() {
        let mut p = packer(4);
        let shard = p.add_document(&[1, 2, 3]).unwrap().unwrap();
        assert_eq!(shard.len(), 4);
        let mut p = packer(5);
        assert!(p.add_document(&[1, 2, 3]).unwrap().is_none());
    }

    #[test]
    fn empty_documents_are_ignored_and_not_counted() {
        let mut p = packer(100);
        assert!(p.add_document(&[]).unwrap().is_none());
        p.add_document(&[1]).unwrap();
        p.add_document(&[]).unwrap();
        assert_eq!((p.documents(), p.tokens()), (1, 2));
        assert_eq!(p.finish().unwrap().tokens(), [1, 256]);
        assert!(packer(10).finish().is_none());
    }

    #[test]
    fn bad_documents_are_refused_and_change_nothing() {
        let mut p = packer(100);
        p.add_document(&[1, 2]).unwrap();
        let err = p.add_document(&[3, 300, 4]).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
        assert!(err.to_string().contains("token 300 at position 1"), "{err}");
        let err = p.add_document(&[3, 4, 256]).unwrap_err();
        assert!(err.to_string().contains("inside a document, at position 2"), "{err}");
        assert_eq!((p.documents(), p.tokens()), (1, 3));
        assert_eq!(p.finish().unwrap().tokens(), [1, 2, 256]);
    }

    #[test]
    fn construction_checks_its_arguments() {
        assert!(Packer::new(300, 300, 10).is_err());
        assert!(Packer::new(300, 0, 0).is_err());
        assert!(Packer::new(300, 299, 1).is_ok());
        assert!(Packer::new(0, 0, 1).is_err());
    }

    #[test]
    fn the_counters_add_up_to_what_the_shards_hold() {
        let mut p = packer(50);
        let mut in_shards = 0usize;
        for n in 1..60u32 {
            let doc: Vec<u32> = (0..n % 7 + 1).map(|i| (i + n) % 256).collect();
            if let Some(shard) = p.add_document(&doc).unwrap() {
                in_shards += shard.len();
            }
        }
        let tokens = p.tokens();
        let documents = p.documents();
        in_shards += p.finish().map_or(0, |s| s.len());
        assert_eq!(in_shards as u64, tokens);
        assert_eq!(documents, 59);
    }

    #[test]
    fn a_packer_for_a_tokenizer_uses_its_vocabulary_and_endoftext() {
        let tokenizer = tokenizer();
        let mut p = Packer::for_tokenizer(&tokenizer, 4).unwrap();
        let shard = p.add_text(&tokenizer, "ab").unwrap();
        assert!(shard.is_none());
        let shard = p.add_text(&tokenizer, "cd").unwrap().unwrap();
        assert_eq!((shard.vocab_size(), shard.eos()), (264, 256));
        assert_eq!(shard.tokens(), [97, 98, 256, 99, 100, 256]);
    }

    #[test]
    fn text_cannot_inject_control_tokens() {
        let tokenizer = tokenizer();
        let mut p = Packer::for_tokenizer(&tokenizer, 1000).unwrap();
        p.add_text(&tokenizer, "before <|endoftext|> after <|pad|>").unwrap();
        let shard = p.finish().unwrap();
        let ends = shard.tokens().iter().filter(|&&t| t == 256).count();
        assert_eq!(ends, 1, "only the packer's own end-of-text");
        // Only plain bytes and the packer's own end-of-text: no other control token got in.
        assert!(shard.tokens().iter().all(|&t| t <= 256));
        let text = tokenizer.decode(&shard.tokens()[..shard.len() - 1]).unwrap();
        assert_eq!(text, "before <|endoftext|> after <|pad|>");
    }

    #[test]
    fn a_tokenizer_without_endoftext_or_a_different_one_is_refused() {
        let plain = Tokenizer::bytes_only();
        assert!(Packer::for_tokenizer(&plain, 10)
            .unwrap_err()
            .to_string()
            .contains("<|endoftext|>"));
        let tokenizer = tokenizer();
        let mut p = Packer::new(264, 257, 10).unwrap();
        assert!(p.add_text(&tokenizer, "x").is_err(), "end-of-text differs");
        let mut p = Packer::new(300, 256, 10).unwrap();
        assert!(p.add_text(&tokenizer, "x").is_err(), "vocabulary differs");
        assert_eq!(p.documents(), 0);
        assert!(Packer::for_tokenizer(&tokenizer, 0).is_err());
    }
}
