# midman-bench

**What it does:** criterion benchmarks for the crates above, with a plain reference implementation beside every kernel so a result means something.

**Example:** open the [benchmarks page](../benchmarks/) for the latest run.

**Status:** current. Open findings: RMSNorm, broadcast add, softmax and the AdamW step are slower than their plain-loop references.
