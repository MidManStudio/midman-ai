// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-bench.md, section "build.rs"
// ============================================================================
//! Compiles `c/baseline_kernels.c` into a static library linked into this package.

fn main() {
    println!("cargo:rerun-if-changed=c/baseline_kernels.c");
    println!("cargo:rerun-if-changed=c/baseline_kernels.h");
    println!("cargo:rerun-if-changed=build.rs");

    cc::Build::new()
        .file("c/baseline_kernels.c")
        .std("c11")
        .warnings(true)
        .flag_if_supported("-Wpedantic")
        // Same arithmetic as the Rust baselines: no fused multiply-add, no fast-math.
        .flag_if_supported("-ffp-contract=off")
        // Rust's `sqrt` has no errno side effect; without this flag the C one does,
        // and the compiler cannot vectorize it.
        .flag_if_supported("-fno-math-errno")
        .compile("midman_c_baselines");
}
