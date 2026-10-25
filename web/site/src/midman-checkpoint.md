# midman-checkpoint

**What it does:** saves a model's weights to a standard safetensors file and loads them back. The reader and writer are the crate's own and handle `F32` tensors. Loading is strict (a missing, extra or differently shaped tensor is an error) and all or nothing, and saving writes a temporary file and renames it, so a crash never leaves a half-written checkpoint.

**Example:** `save_parameters("weights.safetensors", &params, &metadata)?`, and later `load_parameters(&TensorFile::load("weights.safetensors")?, &mut params)?`.

**Status:** done and tested, including against the reference Python `safetensors` library, which reads our files and gets the same values. Optimizer state and training progress are not saved yet.
