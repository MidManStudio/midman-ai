# midman-tensor

**What it does:** dense `f32` tensors on the CPU with reverse-mode automatic differentiation, every operation a transformer needs, and a finite-difference gradient checker.

**Example:** build `x.matmul(&w).silu().sum_all()`, call `backward()`, and read the gradient of `w`.

**Status:** done and tested. Single-threaded; broadcast and softmax are measurably slower than a plain loop (see [midman-bench](midman-bench.md)).
