// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-bench.md, section "baseline_kernels.c"
// ============================================================================
#include "baseline_kernels.h"

#include <math.h>

const char *mmb_compiler(void) {
#ifdef __VERSION__
    return __VERSION__;
#else
    return "unknown C compiler";
#endif
}

void mmb_matmul_ijk(const float *restrict a, const float *restrict b,
                    float *restrict c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            float acc = 0.0f;
            for (size_t p = 0; p < n; p++) {
                acc += a[i * n + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
}

void mmb_matmul_ikj(const float *restrict a, const float *restrict b,
                    float *restrict c, size_t n) {
    for (size_t i = 0; i < n * n; i++) {
        c[i] = 0.0f;
    }
    for (size_t i = 0; i < n; i++) {
        for (size_t p = 0; p < n; p++) {
            const float aip = a[i * n + p];
            for (size_t j = 0; j < n; j++) {
                c[i * n + j] += aip * b[p * n + j];
            }
        }
    }
}

void mmb_softmax_rows(const float *restrict x, float *restrict out,
                      size_t rows, size_t cols) {
    if (cols == 0) {
        return;
    }
    for (size_t r = 0; r < rows; r++) {
        const float *row = x + r * cols;
        float *dst = out + r * cols;
        float max = row[0];
        for (size_t c = 1; c < cols; c++) {
            if (row[c] > max) {
                max = row[c];
            }
        }
        float sum = 0.0f;
        for (size_t c = 0; c < cols; c++) {
            dst[c] = expf(row[c] - max);
            sum += dst[c];
        }
        for (size_t c = 0; c < cols; c++) {
            dst[c] = dst[c] / sum;
        }
    }
}

void mmb_silu(const float *restrict x, float *restrict out, size_t len) {
    for (size_t i = 0; i < len; i++) {
        out[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

void mmb_broadcast_add(const float *restrict matrix, const float *restrict row,
                       float *restrict out, size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < cols; c++) {
            out[r * cols + c] = matrix[r * cols + c] + row[c];
        }
    }
}

void mmb_rms_norm(const float *restrict x, const float *restrict weight,
                  float *restrict out, size_t rows, size_t dim, float eps) {
    if (dim == 0) {
        return;
    }
    for (size_t r = 0; r < rows; r++) {
        const float *row = x + r * dim;
        float *dst = out + r * dim;
        float sum_squares = 0.0f;
        for (size_t c = 0; c < dim; c++) {
            sum_squares += row[c] * row[c];
        }
        const float mean_square = sum_squares / (float)dim;
        const float inv = 1.0f / sqrtf(mean_square + eps);
        for (size_t c = 0; c < dim; c++) {
            dst[c] = row[c] * inv * weight[c];
        }
    }
}

void mmb_adamw_update(float *restrict weights, const float *restrict grads,
                      float *restrict m, float *restrict v, size_t len,
                      int step, float lr, float beta1, float beta2, float eps,
                      float weight_decay) {
    const float bias1 = 1.0f - powf(beta1, (float)step);
    const float bias2_sqrt = sqrtf(1.0f - powf(beta2, (float)step));
    const float decay = 1.0f - lr * weight_decay;
    for (size_t i = 0; i < len; i++) {
        weights[i] *= decay;
        m[i] = beta1 * m[i] + (1.0f - beta1) * grads[i];
        v[i] = beta2 * v[i] + (1.0f - beta2) * grads[i] * grads[i];
        weights[i] -= lr / bias1 * m[i] / (sqrtf(v[i]) / bias2_sqrt + eps);
    }
}
