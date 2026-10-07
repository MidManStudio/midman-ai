# MidMan AI architecture

## Overview

MidMan AI is a Rust workspace for a coding and game-development model, its training pipeline, and an agent runtime around it. The model is trained from scratch in Rust, with Rust and Ubel Stratum as the core code domains.

The workspace keeps three systems separate but interoperable:

- MidMan AI: the model, training pipeline, and model-specific runtime.
- MidMan agent runtime: tools, planning, context, Ubel execution, and task orchestration.
- Sandbox and execution layer: restricted code execution, filesystem and network policies, resource limits, and process isolation.

## Crates

### Foundation model and numerical core

| Crate | Responsibility |
| --- | --- |
| `midman-foundation` | Shared IDs, errors, versioning, core traits and common model abstractions. |
| `midman-model-config` | Serializable architecture configurations, model dimensions, attention settings and validation. |
| `midman-tensor` | Tensor representations, shapes, strides, dtype abstractions and numerical operations. |
| `midman-nn` | Neural network layers, embeddings, normalization, attention, feed-forward blocks and transformer architecture. |
| `midman-model` | MidMan model definition, forward pass, parameter registration and model construction. |
| `midman-optim` | Optimizers, learning-rate schedules, gradient clipping and training-state updates. |
| `midman-tokenizer` | Tokenization, vocabulary construction, special tokens, encoding and decoding, and Ubel-aware token handling. |
| `midman-quant` | Weight and activation quantization, low-bit representations, conversion and quantization evaluation. |

### Training and data pipeline

| Crate | Responsibility |
| --- | --- |
| `midman-data` | Dataset ingestion, source manifests, filtering, deduplication, packing, sampling and batch construction. |
| `midman-training` | Training loop, loss computation, gradient accumulation, evaluation, distributed-training interfaces and run management. |
| `midman-checkpoint` | Saving and loading model weights, optimizer state, training progress, metadata and checkpoint version migrations. |
| `midman-inference` | Text generation, sampling, KV cache, batching, streaming output and inference sessions. |

### Agent and execution layer

| Crate | Responsibility |
| --- | --- |
| `midman-agent` | Conversation state, task planning, context assembly, tool orchestration and the agent loop. |
| `midman-tools` | Typed tool interfaces, tool registry, schemas, validation, dispatch and results. |
| `midman-sandbox` | Restricted execution, permissions, resource quotas, filesystem isolation, process supervision and audit events. |
| `midman-ubel` | Ubel Stratum integration: parsing, diagnostics, compiler interfaces, language-aware context and execution adapters. |
| `midman-game` | Game-development workflows, Mid Engine integration, scene and entity context, asset tooling and simulation-related tools. |
| `midman-api` | Stable library interfaces for applications embedding MidMan. |

## Apps

| App | Responsibility |
| --- | --- |
| `midman-cli` | Command-line entry point for MidMan. |
| `midman-server` | Server entry point for MidMan. |
| `midman-trainer` | Training entry point for MidMan model runs. |

## Status
Real code exists in six crates. The rest are stubs.

| Crate | State |
| --- | --- |
| `midman-foundation` | Errors, format-version check, deterministic RNG. |
| `midman-model-config` | Architecture config, exact parameter counts, text format, six presets. |
| `midman-tensor` | CPU tensors, reverse-mode autodiff, every operation a decoder block needs, gradient checker. |
| `midman-nn` | Linear, embedding, RMSNorm, rotary embeddings, causal grouped-query attention, SwiGLU MLP, decoder block. |
| `midman-model` | `MidManModel`: the decoder-only transformer built from a `ModelConfig`, with the parameter total checked against the config. |
| `midman-optim` | AdamW, warmup and cosine schedule, global-norm gradient clipping. |

A small model trains end to end on the CPU: the smoke test in `midman-model` memorizes random sequences through the real forward pass, `backward`, clipping and AdamW.

Dependencies so far: `midman-model-config`, `midman-tensor` and `midman-optim` (through `midman-tensor`) depend on `midman-foundation`. `midman-nn` depends on `midman-foundation` and `midman-tensor`. `midman-model` depends on `midman-foundation`, `midman-model-config`, `midman-nn` and `midman-tensor`, and has `midman-optim` as a dev-dependency for the smoke test. Nothing else depends on anything.

## Decisions
- **Trained from scratch, in Rust.** No pretrained weights. Training data is public open-source code plus in-house data.
- **Tensors, autograd and kernels are written in-house.** The model stack does not build on a framework such as Burn. A CPU reference implementation comes first, because it can be tested completely and every later backend has to match it. GPU access has not been arranged, and the GPU backend is not designed yet.
- **v0 model architecture:** a Llama-style decoder (RMSNorm, rotary embeddings, grouped-query attention, SwiGLU MLP, no biases, tied embeddings). These are defaults to start from, set in `midman-model-config`, and can change.

## Not settled yet
- How the tensor storage is generalized for a GPU backend, and where the GPU kernels come from.
- The rest of the crate dependency graph.
- The final model architecture beyond the v0 defaults.
- The sandbox execution approach.
