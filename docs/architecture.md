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

## Not settled yet
- The crate dependency graph. No crate depends on another one yet.
- The compute backend strategy.
- The final model architecture.
- The sandbox execution approach.

## Related projects
- Ubel Stratum: https://github.com/MidManStudio/ubel_stratum
- Mid Engine: https://github.com/Mid-D-Man/mid-engine
- DixScript-Rust: https://github.com/Mid-D-Man/DixScript-Rust

## Fixes and Problems
None yet.
