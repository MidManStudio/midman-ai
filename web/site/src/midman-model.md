# midman-model

**What it does:** `MidManModel`, the decoder-only transformer built from a `ModelConfig`. Construction fails unless the parameters add up to exactly what the config counts.

**Example:** `model.loss(&inputs, &targets, batch, seq)?.backward()?` gives gradients for every parameter. `forward_traced` also returns every layer's attention.

**Status:** done and tested, including a gradient check through the whole network and bit-for-bit reproducible training.
