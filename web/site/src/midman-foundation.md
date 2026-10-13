# midman-foundation

**What it does:** the error type, a format-version check, and a deterministic random number generator (xoshiro256**) used for every random choice in the project.

**Example:** `Rng::seed_from_u64(7).next_f32()` always gives the same value on every platform.

**Status:** done and tested.
