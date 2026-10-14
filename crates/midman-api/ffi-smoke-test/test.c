// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-api.md, section "test.c"
// ============================================================================
// A real C program (C11) that calls the midman-api C ABI through midman_api.h and
// links against the compiled library, shared or static. It checks the same
// behaviour as tests/c_abi.rs, plus what only a C caller can get wrong: buffers
// from midman_alloc used as inputs, a misaligned pointer, and a result that must
// be copied out before the next call.
//
// Output: one line per check, "ok:   <message>" or "FAIL: <message> (line N)",
// then a last line "=== N check(s) failed ===". scripts/ffi_summary.py reads those
// lines, and treats a log without the last line (a crash or a link error) as a
// failure. Exit status is 1 if any check failed.
//
// Every lab a test creates is destroyed before it returns, so the slot numbers
// below (the first lab is 1) hold and a leak checker sees a clean exit.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "midman_api.h"

static int failures = 0;

// The last results, copied out of the library's buffers by grab().
static char g_text[1 << 16];
static char g_error[1 << 12];
static float g_floats[4096];
static uint32_t g_floats_n;

static void grab(void);

// A failed check prints what the library last said, so the log explains itself.
// No library call happens between the condition and that copy.
#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (cond) {                                                            \
            printf("ok:   %s\n", msg);                                         \
        } else {                                                               \
            grab();                                                            \
            printf("FAIL: %s (line %d)\n", msg, __LINE__);                     \
            printf("      text: %.200s\n", g_text);                            \
            printf("      error: %.200s\n", g_error);                          \
            failures++;                                                        \
        }                                                                      \
    } while (0)

static const double TINY[MIDMAN_CONFIG_LEN] = {12.0, 16.0, 2.0, 2.0, 1.0, 32.0, 16.0, 10000.0, 1e-5, 1.0};
static const double HYPER[MIDMAN_HYPER_LEN] = {0.03, 0.003, 15.0, 150.0, 0.01, 1.0, 0.9, 0.99};

// Copies the three result buffers into the g_* arrays. A result too big for them
// is reported as a failure rather than silently cut.
static void grab(void) {
    uint32_t text_len = midman_text_len();
    uint32_t error_len = midman_error_len();
    g_floats_n = midman_floats_len();
    int fits = text_len < sizeof g_text && error_len < sizeof g_error &&
               g_floats_n <= sizeof g_floats / sizeof g_floats[0];
    if (!fits) {
        printf("FAIL: a result is too big for this test's buffers (line %d)\n", __LINE__);
        failures++;
        text_len = 0;
        error_len = 0;
        g_floats_n = 0;
    }
    if (text_len > 0) {
        memcpy(g_text, midman_text_ptr(), text_len);
    }
    g_text[text_len] = '\0';
    if (error_len > 0) {
        memcpy(g_error, midman_error_ptr(), error_len);
    }
    g_error[error_len] = '\0';
    if (g_floats_n > 0) {
        memcpy(g_floats, midman_floats_ptr(), g_floats_n * sizeof(float));
    }
}

static int has(const char *haystack, const char *needle) {
    return strstr(haystack, needle) != NULL;
}

static uint32_t tiny_lab(void) {
    uint32_t handle = midman_lab_create(TINY, HYPER, 2024, 0, 2, 12);
    grab();
    if (handle == 0) {
        printf("FAIL: creating the tiny lab failed: %s (line %d)\n", g_error, __LINE__);
        failures++;
    }
    return handle;
}

static void destroy(uint32_t handle) {
    int32_t status = midman_lab_destroy(handle);
    grab();
    if (status != MIDMAN_OK) {
        printf("FAIL: destroying lab %u returned %d: %s (line %d)\n", (unsigned)handle, (int)status,
               g_error, __LINE__);
        failures++;
    }
}

static void test_version_and_layout(void) {
    CHECK(midman_abi_version() == 1u, "the ABI version is 1");
    CHECK(midman_abi_version() == MIDMAN_ABI_VERSION, "the library and the header agree on the ABI version");
    CHECK(MIDMAN_CONFIG_LEN == 10 && MIDMAN_HYPER_LEN == 8, "the header's array lengths are 10 and 8");
    CHECK(MIDMAN_CONFIG_ROPE_THETA == 7 && MIDMAN_HYPER_BETA2 == 7, "the header's field indices are where the docs say");
}

static void test_alloc(void) {
    static const uint32_t lens[] = {0, 1, 7, 8, 4096};
    int all_ok = 1;
    for (size_t i = 0; i < sizeof lens / sizeof lens[0]; i++) {
        uint8_t *ptr = midman_alloc(lens[i]);
        if (ptr == NULL || ((uintptr_t)ptr % 8) != 0) {
            all_ok = 0;
            continue;
        }
        memset(ptr, 0xAB, lens[i] > 0 ? lens[i] : 1);
        if (ptr[0] != 0xAB) {
            all_ok = 0;
        }
        midman_free(ptr, lens[i]);
    }
    CHECK(all_ok, "midman_alloc returns writable 8-aligned memory for lengths 0, 1, 7, 8 and 4096");
    midman_free(NULL, 16);
    CHECK(1, "midman_free(NULL, n) is a no-op");
}

static void test_describe(void) {
    CHECK(midman_describe() == MIDMAN_OK, "describe succeeds");
    grab();
    CHECK(strncmp(g_text, "{\"abi\":1,", 9) == 0, "describe starts with the ABI version");
    CHECK(has(g_text, "\"config_fields\":[\"vocab_size\",\"d_model\",\"n_layers\",\"n_heads\","
                      "\"n_kv_heads\",\"d_ff\",\"max_seq_len\",\"rope_theta\",\"norm_eps\",\"tie_embeddings\"]"),
          "describe lists the config fields in the order of enum midman_config_field");
    CHECK(has(g_text, "\"hyper_fields\":[\"max_lr\",\"min_lr\",\"warmup_steps\",\"total_steps\","
                      "\"weight_decay\",\"clip_norm\",\"beta1\",\"beta2\"]"),
          "describe lists the hyperparameter fields in the order of enum midman_hyper_field");
    CHECK(has(g_text, "\"max_params\":8000000"), "describe reports the parameter limit");
    CHECK(has(g_text, "\"name\":\"smoke\"") && has(g_text, "\"name\":\"tiny\"") && has(g_text, "\"name\":\"small\"") &&
              has(g_text, "\"name\":\"base\"") && has(g_text, "\"name\":\"medium\"") && has(g_text, "\"name\":\"large\""),
          "describe lists all six presets");
    CHECK(has(g_text, "\"params\":117056"), "describe gives the smoke preset's exact parameter count");
}

static void test_presets_and_reports(void) {
    double values[MIDMAN_CONFIG_LEN] = {0};
    CHECK(midman_preset_config(0, values) == MIDMAN_OK, "preset 0 can be read");
    grab();
    CHECK(values[MIDMAN_CONFIG_D_MODEL] == 64.0, "the smoke preset has width 64");
    CHECK(midman_config_report(values) == MIDMAN_OK, "a preset's config validates");
    grab();
    CHECK(has(g_text, "\"valid\":true") && has(g_text, "\"total\":117056") && has(g_text, "\"head_dim\":16"),
          "the report of the smoke preset has its exact count and head size");

    double bad[MIDMAN_CONFIG_LEN];
    memcpy(bad, TINY, sizeof bad);
    bad[MIDMAN_CONFIG_N_HEADS] = 3.0;  // 3 heads do not divide the width of 16
    CHECK(midman_config_report(bad) == MIDMAN_OK, "an invalid config is an answer, not a failure");
    grab();
    CHECK(has(g_text, "\"valid\":false") && has(g_text, "\"error\":\""), "the answer says valid:false and why");
}

static void test_malformed_arguments(void) {
    CHECK(midman_config_report(NULL) == MIDMAN_ERR_ARGUMENT, "a NULL config is MIDMAN_ERR_ARGUMENT");
    grab();
    CHECK(has(g_error, "null"), "and the message says the pointer was null");

    static const char *const names[] = {"vocab_size", "d_model", "n_layers", "n_heads"};
    static const double bad_values[] = {12.5, -1.0, NAN, 5e9};
    int all_rejected = 1;
    for (size_t i = 0; i < 4; i++) {
        double bad[MIDMAN_CONFIG_LEN];
        memcpy(bad, TINY, sizeof bad);
        bad[i] = bad_values[i];
        int32_t status = midman_config_report(bad);
        grab();
        if (status != MIDMAN_ERR_ARGUMENT || !has(g_error, names[i])) {
            all_rejected = 0;
        }
    }
    CHECK(all_rejected, "a fractional, negative, NaN or too large count is rejected, naming the field");

    double inf_theta[MIDMAN_CONFIG_LEN];
    memcpy(inf_theta, TINY, sizeof inf_theta);
    inf_theta[MIDMAN_CONFIG_ROPE_THETA] = INFINITY;
    CHECK(midman_config_report(inf_theta) == MIDMAN_ERR_ARGUMENT, "an infinite rope_theta is rejected");

    CHECK(midman_preset_config(0, NULL) == MIDMAN_ERR_ARGUMENT, "a NULL output pointer is rejected");
    double out[MIDMAN_CONFIG_LEN];
    CHECK(midman_preset_config(99, out) == MIDMAN_ERR_ARGUMENT, "a preset index that does not exist is rejected");

    // A misaligned address is rejected before anything is read from it. The pointer
    // is built from an integer and never dereferenced here.
    _Alignas(8) unsigned char raw[MIDMAN_CONFIG_LEN * sizeof(double) + 8];
    memset(raw, 0, sizeof raw);
    const double *misaligned = (const double *)((uintptr_t)raw + 1u);
    CHECK(midman_config_report(misaligned) == MIDMAN_ERR_ARGUMENT, "a misaligned config pointer is rejected");
    grab();
    CHECK(has(g_error, "aligned"), "and the message says it was not aligned");

    CHECK(midman_describe() == MIDMAN_OK, "describe again so there is text to lose");
    grab();
    CHECK(strlen(g_text) > 0, "describe left text behind");
    CHECK(midman_config_report(NULL) == MIDMAN_ERR_ARGUMENT, "a failing call");
    grab();
    CHECK(strlen(g_text) == 0 && strlen(g_error) > 0, "a failing call leaves no stale text, only its error");
}

static void test_caller_owned_buffers(void) {
    // The documented way to pass arrays: memory from midman_alloc, filled by the caller.
    double *config = (double *)midman_alloc(sizeof TINY);
    double *hyper = (double *)midman_alloc(sizeof HYPER);
    CHECK(config != NULL && hyper != NULL, "midman_alloc gives room for a config and hyperparameters");
    if (config == NULL || hyper == NULL) {
        return;
    }
    memcpy(config, TINY, sizeof TINY);
    memcpy(hyper, HYPER, sizeof HYPER);
    CHECK(midman_config_report(config) == MIDMAN_OK, "a config built in midman_alloc memory validates");
    uint32_t handle = midman_lab_create(config, hyper, 7, 0, 2, 12);
    grab();
    CHECK(handle != 0, "a lab can be created from midman_alloc memory");
    midman_free((uint8_t *)config, (uint32_t)sizeof TINY);
    midman_free((uint8_t *)hyper, (uint32_t)sizeof HYPER);
    if (handle != 0) {
        CHECK(midman_lab_step(handle, 2) == MIDMAN_OK, "the lab keeps working after its inputs are freed");
        grab();
        destroy(handle);
    }
}

static void test_lab_lifecycle(void) {
    uint32_t handle = tiny_lab();
    CHECK(handle == 1, "the first lab gets handle 1");

    CHECK(midman_lab_step(handle, 5) == MIDMAN_OK, "five steps run");
    grab();
    CHECK(g_floats_n == 15, "three numbers per step");
    int sane = g_floats_n == 15;
    for (uint32_t i = 0; sane && i + 2 < g_floats_n; i += 3) {
        sane = isfinite(g_floats[i]) && g_floats[i + 1] > 0.0f && g_floats[i + 2] > 0.0f;
    }
    CHECK(sane, "every step has a finite loss, a positive gradient norm and a positive learning rate");
    CHECK(fabsf(g_floats[0] - logf(12.0f)) < 0.3f, "the first loss is near ln(vocab size)");

    CHECK(midman_lab_eval(handle) == MIDMAN_OK, "eval succeeds");
    grab();
    CHECK(has(g_text, "\"step\":5") && has(g_text, "\"accuracy\":") && has(g_text, "\"batch\":2,\"seq\":12"),
          "eval reports the step, the accuracy and the task shape");

    CHECK(midman_lab_stats(handle) == MIDMAN_OK, "stats succeed");
    grab();
    CHECK(has(g_text, "\"name\":\"embed_tokens.weight\",\"shape\":[12,16]") && has(g_text, "\"num_parameters\":"),
          "stats list the parameters with their shapes");
    CHECK(!has(g_text, "\"grad_norm\":null"), "after a step every gradient norm is known");

    CHECK(midman_lab_attention(handle, 1) == MIDMAN_OK, "attention of sequence 1 succeeds");
    grab();
    CHECK(strcmp(g_text, "{\"layers\":2,\"heads\":2,\"seq\":12}") == 0, "attention reports layers, heads and seq");
    CHECK(g_floats_n == 2 * 2 * 12 * 12, "attention has layers * heads * seq * seq values");

    CHECK(midman_lab_embedding(handle) == MIDMAN_OK, "the embedding table can be read");
    grab();
    CHECK(strcmp(g_text, "{\"rows\":12,\"cols\":16}") == 0 && g_floats_n == 12 * 16, "it has vocab x width values");

    CHECK(midman_lab_destroy(handle) == MIDMAN_OK, "the lab is destroyed");
    CHECK(midman_lab_step(handle, 1) == MIDMAN_ERR_ARGUMENT, "its handle is then an error");
    grab();
    CHECK(has(g_error, "not a live lab handle"), "and the message says so");
    CHECK(midman_lab_destroy(handle) == MIDMAN_ERR_ARGUMENT, "destroying twice is an error, not a crash");
}

static void test_gradient_norms_before_a_step(void) {
    uint32_t handle = tiny_lab();
    CHECK(midman_lab_stats(handle) == MIDMAN_OK, "stats of a fresh lab succeed");
    grab();
    CHECK(has(g_text, "\"last_loss\":null") && has(g_text, "\"grad_norm\":null"),
          "the last loss and gradient norms are null before the first step");
    destroy(handle);
}

static void test_slots(void) {
    uint32_t first = tiny_lab();
    uint32_t second = tiny_lab();
    CHECK(first == 1 && second == 2, "two labs get handles 1 and 2");
    CHECK(midman_lab_step(first, 3) == MIDMAN_OK, "stepping the first lab");
    CHECK(midman_lab_stats(second) == MIDMAN_OK, "reading the second lab");
    grab();
    CHECK(has(g_text, "\"step\":0"), "stepping one lab does not move the other");
    destroy(first);
    uint32_t again = tiny_lab();
    CHECK(again == 1, "the freed slot is reused");
    CHECK(midman_lab_stats(second) == MIDMAN_OK, "the other lab survives");
    destroy(again);
    destroy(second);
}

static void test_invented_handles(void) {
    static const uint32_t handles[] = {0u, 7u, UINT32_MAX};
    int all_errors = 1;
    for (size_t i = 0; i < 3; i++) {
        uint32_t h = handles[i];
        int ok = midman_lab_step(h, 1) == MIDMAN_ERR_ARGUMENT && midman_lab_eval(h) == MIDMAN_ERR_ARGUMENT &&
                 midman_lab_stats(h) == MIDMAN_ERR_ARGUMENT && midman_lab_attention(h, 0) == MIDMAN_ERR_ARGUMENT &&
                 midman_lab_embedding(h) == MIDMAN_ERR_ARGUMENT && midman_lab_destroy(h) == MIDMAN_ERR_ARGUMENT;
        grab();
        all_errors = all_errors && ok && has(g_error, "handle");
    }
    CHECK(all_errors, "handles 0, 7 and UINT32_MAX are errors for every lab call, not crashes");
}

static void test_bad_creation(void) {
    double huge[MIDMAN_CONFIG_LEN];
    memcpy(huge, TINY, sizeof huge);
    huge[MIDMAN_CONFIG_VOCAB_SIZE] = 100000.0;
    huge[MIDMAN_CONFIG_D_MODEL] = 128.0;
    huge[MIDMAN_CONFIG_N_HEADS] = 4.0;
    huge[MIDMAN_CONFIG_N_KV_HEADS] = 4.0;
    huge[MIDMAN_CONFIG_D_FF] = 256.0;
    CHECK(midman_lab_create(huge, HYPER, 1, 0, 2, 12) == 0, "a model over the size limit is refused with handle 0");
    grab();
    CHECK(has(g_error, "up to"), "and the message states the limit");
    CHECK(midman_lab_create(TINY, HYPER, 1, 0, 0, 12) == 0, "a task with no sequences is refused");
    CHECK(midman_lab_create(TINY, HYPER, 1, 0, 2, 17) == 0, "a sequence longer than max_seq_len is refused");
    double bad_hyper[MIDMAN_HYPER_LEN];
    memcpy(bad_hyper, HYPER, sizeof bad_hyper);
    bad_hyper[MIDMAN_HYPER_BETA1] = 1.0;
    CHECK(midman_lab_create(TINY, bad_hyper, 1, 0, 2, 12) == 0, "a beta1 of 1 is refused");
    CHECK(midman_lab_create(NULL, HYPER, 1, 0, 2, 12) == 0, "a NULL config is refused");
    CHECK(midman_lab_create(TINY, NULL, 1, 0, 2, 12) == 0, "NULL hyperparameters are refused");
}

static void test_missing_sequence(void) {
    uint32_t handle = tiny_lab();
    CHECK(midman_lab_attention(handle, 2) != MIDMAN_OK, "attention for a sequence that does not exist is an error");
    grab();
    CHECK(has(g_error, "does not exist"), "and the message says so");
    destroy(handle);
}

static void test_blow_up(void) {
    double hyper[MIDMAN_HYPER_LEN];
    memcpy(hyper, HYPER, sizeof hyper);
    hyper[MIDMAN_HYPER_MAX_LR] = 1e30;
    hyper[MIDMAN_HYPER_MIN_LR] = 1e30;
    hyper[MIDMAN_HYPER_WARMUP_STEPS] = 0.0;
    uint32_t handle = midman_lab_create(TINY, hyper, 2024, 0, 2, 12);
    grab();
    CHECK(handle != 0, "a lab with an absurd learning rate can still be created");
    if (handle == 0) {
        return;
    }
    CHECK(midman_lab_step(handle, 50) == MIDMAN_STOPPED, "training that blows up returns MIDMAN_STOPPED");
    grab();
    CHECK(has(g_error, "finite"), "and the message says a value was not finite");
    CHECK(g_floats_n > 0 && g_floats_n < 150 && g_floats_n % 3 == 0, "the steps that did complete are reported");
    destroy(handle);
}

static void test_results_are_replaced(void) {
    uint32_t handle = tiny_lab();
    CHECK(midman_lab_embedding(handle) == MIDMAN_OK, "a call with text and numbers");
    grab();
    CHECK(g_floats_n > 0 && strlen(g_text) > 0, "leaves both results");
    CHECK(midman_lab_step(handle, 0) == MIDMAN_OK, "a call with neither");
    grab();
    CHECK(g_floats_n == 0 && strlen(g_text) == 0 && strlen(g_error) == 0, "replaces both, leaving nothing stale behind");
    destroy(handle);
}

static void test_seeds(void) {
    float runs[3][24];
    static const uint32_t seeds[3] = {5, 5, 6};
    for (int i = 0; i < 3; i++) {
        uint32_t handle = midman_lab_create(TINY, HYPER, seeds[i], 0, 2, 12);
        grab();
        if (handle == 0) {
            printf("FAIL: creating a seeded lab failed (line %d)\n", __LINE__);
            failures++;
            return;
        }
        int32_t status = midman_lab_step(handle, 8);
        grab();
        memcpy(runs[i], g_floats, sizeof runs[i]);
        if (status != MIDMAN_OK || g_floats_n != 24) {
            printf("FAIL: the seeded run did not complete (line %d)\n", __LINE__);
            failures++;
        }
        destroy(handle);
    }
    CHECK(memcmp(runs[0], runs[1], sizeof runs[0]) == 0, "the same seed gives bit-identical training");
    CHECK(memcmp(runs[0], runs[2], sizeof runs[0]) != 0, "a different seed gives different training");
}

static void test_rope_table(void) {
    CHECK(midman_rope_table(8, 6, 10000.0) == MIDMAN_OK, "a rope table for head size 8 and 6 positions");
    CHECK(((uintptr_t)midman_floats_ptr() % 4) == 0, "the numeric result is 4-byte aligned");
    grab();
    CHECK(strcmp(g_text, "{\"positions\":6,\"half\":4}") == 0, "its text names the positions and half size");
    CHECK(g_floats_n == 2 * 6 * 4, "it holds the cosines then the sines");
    int cos0 = 1;
    int sin0 = 1;
    for (int i = 0; i < 4; i++) {
        cos0 = cos0 && g_floats[i] == 1.0f;
        sin0 = sin0 && g_floats[24 + i] == 0.0f;
    }
    CHECK(cos0 && sin0, "position 0 has cosine 1 and sine 0");
    CHECK(midman_rope_table(7, 6, 10000.0) < 0, "an odd head size has no rotary table");
    grab();
    CHECK(strlen(g_error) > 0, "and the error says why");
    CHECK(midman_rope_table(8, 6, NAN) == MIDMAN_ERR_ARGUMENT, "a NaN theta is rejected");
}

int main(void) {
    // Line-buffered, so every check reaches the log before the next call runs. With a
    // block buffer, a crash would lose all the output printed so far.
    setvbuf(stdout, NULL, _IOLBF, 0);
    printf("=== midman-api C ABI: a C11 program calling the compiled library ===\n\n");
    test_version_and_layout();
    test_alloc();
    test_describe();
    test_presets_and_reports();
    test_malformed_arguments();
    test_caller_owned_buffers();
    test_lab_lifecycle();
    test_gradient_norms_before_a_step();
    test_slots();
    test_invented_handles();
    test_bad_creation();
    test_missing_sequence();
    test_blow_up();
    test_results_are_replaced();
    test_seeds();
    test_rope_table();
    printf("\n=== %d check(s) failed ===\n", failures);
    return failures == 0 ? 0 : 1;
}
