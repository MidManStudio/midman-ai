# midman-nn

**What it does:** the layers of a decoder-only transformer: linear, embedding, RMSNorm, rotary embeddings, causal grouped-query attention, a SwiGLU MLP and a decoder block. Each layer's math is also a plain function so it can be gradient-checked, and attention can return its probabilities for visualization.

**Example:** `Attention::forward_traced` returns the output and the `[batch, heads, time, time]` attention probabilities.

**Status:** done and tested. RMSNorm is much slower than a plain loop and is on the list to fix.
