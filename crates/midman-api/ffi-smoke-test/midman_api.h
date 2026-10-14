// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-api.md, section "midman_api.h"
// ============================================================================
// The C ABI of midman-api (crates/midman-api/src/ffi.rs), version 1.
//
// Hand-written to match ffi.rs exactly; there is no header generator in the build.
// If a signature in ffi.rs changes, change it here in the same commit.
// ffi-smoke-test/test.c (C11), cxx_check.cpp (C++17) and the Rust tests in
// tests/c_abi.rs all call these functions, and .github/workflows/midman-api-ffi.yml
// runs the first two against the real library.
//
// Rules of the boundary (the full reasoning is in ffi.rs):
//
//   Handles, not pointers.  A lab lives inside the library; you hold a small
//                           integer. A stale or invented handle is an error
//                           status, never undefined behaviour.
//   You own your inputs.    Arrays you pass in are yours. Allocate them with
//                           midman_alloc (or any 8-byte-aligned memory) and free
//                           them with midman_free using the same length.
//   It owns its results.    Results are left in library buffers read through the
//                           midman_text_*, midman_floats_* and midman_error_*
//                           accessors. Read what you need (copy it out) before
//                           the next call into the library; the next call that is
//                           not one of those accessors replaces all three buffers.
//   Status codes only.      Functions return 0 for success, 1 for "stopped early"
//                           (midman_lab_step only) or a negative code, with the
//                           message in the error buffer. Nothing unwinds across
//                           the boundary; a panic is reported as MIDMAN_ERR_PANIC.
//   One thread.             State is thread-local. A handle is only valid on the
//                           thread that created it.
//
// Arrays of numbers are doubles. A model config is 10 values, in the order of
// enum midman_config_field. Hyperparameters are 8 values, in the order of enum
// midman_hyper_field. midman_describe() reports the same orders as JSON.
#ifndef MIDMAN_API_H
#define MIDMAN_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define MIDMAN_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define MIDMAN_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif

// The arrays and results below are 32-bit floats and 64-bit doubles.
MIDMAN_STATIC_ASSERT(sizeof(float) == 4, "midman_api.h: float must be 32 bits");
MIDMAN_STATIC_ASSERT(sizeof(double) == 8, "midman_api.h: double must be 64 bits");

#ifdef __cplusplus
extern "C" {
#endif

// The version of this ABI. It changes when a function is removed or its meaning
// changes, not when one is added. Compare midman_abi_version() against it at startup.
#define MIDMAN_ABI_VERSION 1u

// ---- Status codes ----------------------------------------------------------
#define MIDMAN_OK            0    // The call succeeded.
#define MIDMAN_STOPPED       1    // midman_lab_step stopped before every requested step.
#define MIDMAN_ERR_ARGUMENT -1    // A null pointer, an invalid handle, a number out of range.
#define MIDMAN_ERR_CONFIG   -2    // A model configuration problem.
#define MIDMAN_ERR_SHAPE    -3    // Tensor shapes did not fit.
#define MIDMAN_ERR_INDEX    -4    // An index (such as a token id) was out of range.
#define MIDMAN_ERR_OTHER    -5    // Any other library error.
#define MIDMAN_ERR_PANIC   -99    // A panic was caught inside the library.

// ---- Array layouts ---------------------------------------------------------
#define MIDMAN_CONFIG_LEN 10      // doubles in a model config
#define MIDMAN_HYPER_LEN   8      // doubles in a hyperparameter array

enum midman_config_field {
    MIDMAN_CONFIG_VOCAB_SIZE = 0,
    MIDMAN_CONFIG_D_MODEL,
    MIDMAN_CONFIG_N_LAYERS,
    MIDMAN_CONFIG_N_HEADS,
    MIDMAN_CONFIG_N_KV_HEADS,
    MIDMAN_CONFIG_D_FF,
    MIDMAN_CONFIG_MAX_SEQ_LEN,
    MIDMAN_CONFIG_ROPE_THETA,
    MIDMAN_CONFIG_NORM_EPS,
    MIDMAN_CONFIG_TIE_EMBEDDINGS  // 0 or non-zero
};

enum midman_hyper_field {
    MIDMAN_HYPER_MAX_LR = 0,
    MIDMAN_HYPER_MIN_LR,
    MIDMAN_HYPER_WARMUP_STEPS,
    MIDMAN_HYPER_TOTAL_STEPS,
    MIDMAN_HYPER_WEIGHT_DECAY,
    MIDMAN_HYPER_CLIP_NORM,
    MIDMAN_HYPER_BETA1,
    MIDMAN_HYPER_BETA2
};

MIDMAN_STATIC_ASSERT((int)MIDMAN_CONFIG_TIE_EMBEDDINGS == MIDMAN_CONFIG_LEN - 1,
                     "midman_api.h: enum midman_config_field must have MIDMAN_CONFIG_LEN entries");
MIDMAN_STATIC_ASSERT((int)MIDMAN_HYPER_BETA2 == MIDMAN_HYPER_LEN - 1,
                     "midman_api.h: enum midman_hyper_field must have MIDMAN_HYPER_LEN entries");

// ---- Memory ----------------------------------------------------------------

// The version of this ABI. Compare with MIDMAN_ABI_VERSION.
uint32_t midman_abi_version(void);

// Allocates `len` bytes (at least 1), aligned to 8, for you to fill with input
// data. Returns NULL if the allocation fails. Free it with midman_free using the
// same `len`.
uint8_t *midman_alloc(uint32_t len);

// Frees a buffer from midman_alloc. `len` must be the value passed to midman_alloc.
// A NULL `ptr` is ignored.
void midman_free(uint8_t *ptr, uint32_t len);

// ---- Results ---------------------------------------------------------------
// Each call that produces a result replaces all three buffers. When a length is 0
// the pointer must not be dereferenced (it may be non-NULL but points at nothing).

// The text result: UTF-8 JSON, not NUL-terminated. Read `midman_text_len()` bytes.
const uint8_t *midman_text_ptr(void);
uint32_t midman_text_len(void);

// The numeric result: 4-byte-aligned floats.
const float *midman_floats_ptr(void);
uint32_t midman_floats_len(void);

// The error message of the last call: UTF-8, not NUL-terminated, empty on success.
const uint8_t *midman_error_ptr(void);
uint32_t midman_error_len(void);

// ---- Stateless calls -------------------------------------------------------

// Text result: the ABI version, the field order of the config and hyperparameter
// arrays, the lab's size limits, and every preset with its config values and exact
// parameter count.
int32_t midman_describe(void);

// Writes the 10 config values of preset `index` (0 is smoke) to `out`, which must
// point to at least MIDMAN_CONFIG_LEN writable, 8-byte-aligned doubles.
// MIDMAN_ERR_ARGUMENT for a bad index or a null or misaligned `out`.
int32_t midman_preset_config(uint32_t index, double *out);

// Validates a model config and counts its parameters without building anything.
// `config` must point to MIDMAN_CONFIG_LEN readable, 8-byte-aligned doubles.
// Text result: {"valid":true,"head_dim":..,"kv_dim":..,"params":{...,"total":..}} or
// {"valid":false,"error":".."}. Both return MIDMAN_OK: an invalid config is an
// answer, not a failure of the call. A malformed argument (null pointer, a count
// that is not a whole number) is MIDMAN_ERR_ARGUMENT.
int32_t midman_config_report(const double *config);

// The rotary rotation tables. Numeric result: the cosines then the sines, each
// max_seq_len * head_dim / 2 values, one row per position. Text result:
// {"positions":..,"half":..}.
int32_t midman_rope_table(uint32_t head_dim, uint32_t max_seq_len, double theta);

// ---- Labs ------------------------------------------------------------------

// Creates a lab and returns its handle (1 or more), or 0 on failure with the
// reason in the error buffer. `config` is MIDMAN_CONFIG_LEN and `hyper` is
// MIDMAN_HYPER_LEN readable, 8-byte-aligned doubles. The seed is the 64-bit value
// (seed_hi << 32) | seed_lo. The task is `batch` random sequences of `seq` tokens.
uint32_t midman_lab_create(const double *config, const double *hyper,
                           uint32_t seed_lo, uint32_t seed_hi,
                           uint32_t batch, uint32_t seq);

// Runs `steps` training steps. Numeric result: three values per completed step
// (loss before the update, global gradient norm before clipping, learning rate).
// Returns MIDMAN_OK if every step ran, or MIDMAN_STOPPED if a loss or gradient was
// not finite: the numeric result then holds the steps that did complete and the
// error buffer says why.
int32_t midman_lab_step(uint32_t handle, uint32_t steps);

// Measures loss and predictions without training. Text result:
// {"step":..,"loss":..,"accuracy":..,"batch":..,"seq":..,"inputs":[[..]],
// "targets":[[..]],"predictions":[[..]]}.
int32_t midman_lab_eval(uint32_t handle);

// Summarizes every parameter. Text result: {"step":..,"last_loss":..|null,
// "num_parameters":..,"params":[{"name":..,"shape":[..],"rms":..,"grad_norm":..|null}]}.
int32_t midman_lab_stats(uint32_t handle);

// The attention probabilities of one task sequence. Numeric result: layers * heads
// * seq * seq values, layer-major then head, row-major. Text result:
// {"layers":..,"heads":..,"seq":..}.
int32_t midman_lab_attention(uint32_t handle, uint32_t sequence);

// The token embedding table. Numeric result: rows * cols values, one row per
// token. Text result: {"rows":..,"cols":..}.
int32_t midman_lab_embedding(uint32_t handle);

// Destroys a lab and frees its memory. The handle becomes invalid.
int32_t midman_lab_destroy(uint32_t handle);

#ifdef __cplusplus
}
#endif

#endif
