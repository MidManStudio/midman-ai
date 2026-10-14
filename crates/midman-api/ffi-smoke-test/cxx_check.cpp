// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-api.md, section "cxx_check.cpp"
// ============================================================================
// Includes midman_api.h from C++17 and calls the library, to prove the header
// parses as C++ with its extern "C" guard and that the symbols link without
// mangling. The full behaviour is checked by test.c; this is only the C++ side.
//
// Same output convention as test.c ("ok:", "FAIL:", a last "=== N check(s) failed ===").
#include <cstdint>
#include <cstdio>

#include "midman_api.h"

static_assert(MIDMAN_CONFIG_LEN == 10 && MIDMAN_HYPER_LEN == 8, "array lengths");
static_assert(MIDMAN_ERR_PANIC == -99 && MIDMAN_STOPPED == 1, "status codes");

static int failures = 0;

static void check(bool condition, const char *message, int line) {
    if (condition) {
        std::printf("ok:   %s\n", message);
    } else {
        std::printf("FAIL: %s (line %d)\n", message, line);
        failures++;
    }
}

#define CHECK(cond, msg) check((cond), (msg), __LINE__)

int main() {
    // Line-buffered, so a crash does not lose the output printed so far.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::printf("=== midman-api C ABI: C++17 including the header ===\n\n");
    CHECK(midman_abi_version() == MIDMAN_ABI_VERSION, "the library and the header agree on the ABI version");
    CHECK(midman_describe() == MIDMAN_OK && midman_text_len() > 0, "describe gives text");

    alignas(8) double config[MIDMAN_CONFIG_LEN] = {};
    CHECK(midman_preset_config(0, config) == MIDMAN_OK, "a preset can be read");
    CHECK(midman_config_report(config) == MIDMAN_OK, "and validated");

    static const double hyper[MIDMAN_HYPER_LEN] = {0.03, 0.003, 15.0, 150.0, 0.01, 1.0, 0.9, 0.99};
    const std::uint32_t lab = midman_lab_create(config, hyper, 1, 0, 2, 8);
    CHECK(lab != 0, "a lab can be created");
    CHECK(midman_lab_step(lab, 2) == MIDMAN_OK && midman_floats_len() == 6, "it runs two steps");
    CHECK(midman_lab_destroy(lab) == MIDMAN_OK, "and is destroyed");
    CHECK(midman_lab_step(lab, 1) == MIDMAN_ERR_ARGUMENT, "its handle is then an error");

    std::printf("\n=== %d check(s) failed ===\n", failures);
    return failures == 0 ? 0 : 1;
}
