# midman-optim

**What it does:** AdamW with decoupled weight decay, a warmup-and-cosine learning-rate schedule, and global gradient-norm clipping.

**Example:** `clip_global_norm(&mut grads, 1.0)?` then `optimizer.step(&mut params, &grads, lr)?`.

**Status:** done and tested against an independent implementation.
