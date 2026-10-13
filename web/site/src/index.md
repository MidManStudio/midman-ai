# MidMan AI

A coding and game-development language model built from scratch in Rust: its own tensors, its own autograd, its own training loop. No pretrained weights and no ML framework.

**Try it:** the [lab](lab.md) runs the real model in your browser, compiled to WebAssembly.

**What works today:** the tensor library with reverse-mode autodiff, the transformer layers, the model, AdamW, and an end-to-end training loop that memorizes a task. **Not started:** the tokenizer, the data pipeline, checkpoints and everything after them.

Each crate page below says what the crate does, shows how to use it, and states its status. The longer design notes live in the repository's `docs/` folder.
