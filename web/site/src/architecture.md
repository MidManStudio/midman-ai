# Workspace layout

18 library crates under `crates/`, 3 binaries under `apps/`, and one benchmark package under `benches/`. Nine library crates have code today, plus `midman-api`; the other eight are stubs with their documentation already written.

```text
midman-foundation -> midman-model-config -> midman-tensor -> midman-nn -> midman-model -> midman-optim
                                                                               \-> midman-api (C ABI, WebAssembly)
midman-foundation, midman-tensor -> midman-checkpoint                  (weights files)
midman-foundation -> midman-tokenizer -> midman-data                   (text to shards and batches)
```

**Status:** current. Dependencies point one way, and nothing depends on `midman-bench` or `midman-api`.
