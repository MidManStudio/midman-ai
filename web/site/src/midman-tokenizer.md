# midman-tokenizer

**What it does:** a byte-level BPE tokenizer and its trainer, written from scratch. The pre-tokenizer is made for code and terminal text: an ANSI colour sequence is one piece, identifiers keep their underscores, and numbers are cut three digits at a time. Eight special tokens (end of text, padding, three for fill-in-the-middle, three for terminal sessions) are fixed in `special.rs`. Text is read as ordinary text by default, so a document cannot inject one.

**Example:** `Trainer::new(config)?`, then `trainer.add(bytes)` for each file and `trainer.finish()?` to get the tokenizer; `tokenizer.encode(text, SpecialMode::AsText)` to use it.

**Status:** done and tested against an independent Python implementation. A vocabulary of 4,096 ids trained on 92 MB of code in 22 seconds. The production vocabulary is not trained yet: it has to be trained on the real corpus.
