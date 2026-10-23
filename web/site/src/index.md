# MidMan AI

A coding and game-development language model built from scratch in Rust: its own tensors, its own autograd, its own training loop. No pretrained weights and no ML framework.

**Try it:** the [lab](lab.md) runs the real model in your browser, compiled to WebAssembly.

**What works today:** the tensor library with reverse-mode autodiff, the transformer layers, the model, AdamW, an end-to-end training loop that memorizes a task, saving and loading model weights as standard safetensors files, a byte-level BPE tokenizer with its trainer, and the data path from text to batches: a license policy, a source manifest, secret scrubbing, exact deduplication, packing into token shards and a resumable batcher. **Not started:** fetching sources, mixing them by weight, saving optimizer state and training progress, and everything after them.

Each crate page below says what the crate does, shows how to use it, and states its status. The longer design notes live in the repository's `docs/` folder.
