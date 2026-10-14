// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-bench.md, section "baseline_kernels.h"
// ============================================================================
// Plain C versions of the kernels that benches/midman-bench times, so the Rust
// code can be compared with what a straightforward C implementation does.
//
// Hand-written, not generated. The Rust side declares the same prototypes in
// src/lib.rs, and its tests call every function here, so a signature that
// drifts shows up as a test failure rather than as silent stack corruption.
//
// Every function writes only to its output arguments, never allocates, and
// assumes the buffers it is given do not overlap (the definitions use
// `restrict`). Lengths are element counts, not bytes.
#ifndef MIDMAN_BASELINE_KERNELS_H
#define MIDMAN_BASELINE_KERNELS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// The compiler that built this file, as the compiler reports it (`__VERSION__`).
// Never NULL; the string is static.
const char *mmb_compiler(void);

// c = a * b for square n x n row-major matrices, textbook i-j-k loop order.
void mmb_matmul_ijk(const float *a, const float *b, float *c, size_t n);

// The same product with the loops ordered i-k-j, so the inner loop walks both
// b and c along a row and can be vectorized. Every c[i][j] is still summed in
// the same order as mmb_matmul_ijk, so the two give bit-identical results.
void mmb_matmul_ikj(const float *a, const float *b, float *c, size_t n);

// Softmax over each row of a rows x cols matrix: subtract the row maximum,
// exponentiate, divide by the sum. cols == 0 does nothing.
void mmb_softmax_rows(const float *x, float *out, size_t rows, size_t cols);

// out[i] = x[i] / (1 + exp(-x[i])).
void mmb_silu(const float *x, float *out, size_t len);

// out[r][c] = matrix[r][c] + row[c] for a rows x cols matrix and one row of cols.
void mmb_broadcast_add(const float *matrix, const float *row, float *out,
                       size_t rows, size_t cols);

// RMSNorm over each row of dim values: out = x / sqrt(mean(x^2) + eps) * weight.
// dim == 0 does nothing.
void mmb_rms_norm(const float *x, const float *weight, float *out,
                  size_t rows, size_t dim, float eps);

// One AdamW step over len values, in place. `step` counts from 1. The update
// rule and the order of the floating-point operations match
// benches/midman-bench/benches/optim_step.rs's PlainAdamW.
void mmb_adamw_update(float *weights, const float *grads, float *m, float *v,
                      size_t len, int step, float lr, float beta1, float beta2,
                      float eps, float weight_decay);

#ifdef __cplusplus
}
#endif

#endif
