#!/usr/bin/env python3
# ============================================================================
# NOTICE: Full documentation, design decisions, and fix history for this file
# live in docs/ci-and-workflows.md, section "test_ffi_summary.py"
# ============================================================================
"""Self-tests for ffi_summary.py, run against real captured output (scripts/fixtures/ffi_*.txt):
a passing and a failing run of crates/midman-api/ffi-smoke-test/test.c, a run that crashed,
real AddressSanitizer, UndefinedBehaviorSanitizer, LeakSanitizer and valgrind reports, and a
real `cargo test -p midman-api` log. Run with:

    python3 -m unittest discover -s scripts -p 'test_*.py'
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ffi_summary as fs  # noqa: E402

FIXTURES = os.path.join(HERE, "fixtures")
SCRIPT = os.path.join(HERE, "ffi_summary.py")


def fixture(name):
    return fs.read(os.path.join(FIXTURES, name))


def names(checks, status):
    return [c["name"] for c in checks if c["status"] == status]


class CChecks(unittest.TestCase):
    def test_a_real_passing_run_is_all_passed(self):
        checks = fs.c_checks(fixture("ffi_c_real_run.txt"))
        self.assertEqual(len(checks), 90)
        self.assertEqual(names(checks, "failed"), [])
        self.assertIn("the ABI version is 1", names(checks, "passed"))

    def test_a_real_failing_run_names_each_failed_check_with_its_line(self):
        checks = fs.c_checks(fixture("ffi_c_real_fail.txt"))
        failed = [c for c in checks if c["status"] == "failed"]
        self.assertEqual(len(failed), 2)
        self.assertEqual(failed[0]["name"], "training that blows up returns MIDMAN_STOPPED")
        self.assertRegex(failed[0]["detail"], r"^line \d+$")
        self.assertEqual(len([c for c in checks if c["status"] == "passed"]), 88)

    def test_the_text_and_error_lines_under_a_failure_are_not_checks(self):
        text = "FAIL: a thing (line 4)\n      text: ok: not a check\n      error: FAIL: nor this (line 9)\n"
        text += "=== 1 check(s) failed ===\n"
        checks = fs.c_checks(text)
        self.assertEqual([c["name"] for c in checks], ["a thing"])

    def test_a_real_crash_is_a_failed_run_even_though_no_check_failed(self):
        text = fixture("ffi_c_real_crash.txt")
        self.assertNotIn("check(s) failed", text)
        checks = fs.c_checks(text)
        self.assertEqual(names(checks, "failed"), ["program ran to completion"])
        self.assertEqual(len(names(checks, "passed")), 25)  # what ran before the crash is kept

    def test_a_missing_or_empty_log_is_a_failed_run(self):
        for text in (fs.read("/nonexistent/log.txt"), ""):
            self.assertEqual(names(fs.c_checks(text), "failed"), ["program ran to completion"])

    def test_a_closing_line_that_disagrees_with_the_fail_lines_is_caught(self):
        checks = fs.c_checks("FAIL: x (line 1)\n=== 0 check(s) failed ===\n")
        self.assertIn("the failure count agrees with the FAIL lines", names(checks, "failed"))

    def test_a_program_that_printed_no_checks_does_not_pass(self):
        checks = fs.c_checks("=== 0 check(s) failed ===\n")
        self.assertEqual(names(checks, "failed"), ["ran at least one check"])

    def test_color_codes_are_stripped(self):
        self.assertEqual(len(fs.c_checks(fs.strip_ansi("\x1b[32mok:   fine\x1b[0m\n=== 0 check(s) failed ===\n"))), 1)


class Sanitizers(unittest.TestCase):
    def test_a_clean_run_passes_all_three(self):
        checks = fs.sanitizer_checks(fixture("ffi_c_real_run.txt"))
        self.assertEqual(len(checks), 3)
        self.assertEqual(names(checks, "failed"), [])

    def test_a_real_use_after_free_fails_the_address_check_and_the_run(self):
        text = fixture("ffi_asan_real_use_after_free.txt")
        self.assertEqual(names(fs.sanitizer_checks(text), "failed"), ["no AddressSanitizer errors"])
        self.assertEqual(names(fs.c_checks(text), "failed"), ["program ran to completion"])

    def test_a_real_undefined_behavior_report_fails_and_says_what(self):
        checks = fs.sanitizer_checks(fixture("ffi_ubsan_real_overflow.txt"))
        failed = [c for c in checks if c["status"] == "failed"]
        self.assertEqual([c["name"] for c in failed], ["no UndefinedBehaviorSanitizer reports"])
        self.assertIn("signed integer overflow", failed[0]["detail"])

    def test_a_real_leak_report_fails_only_the_leak_check(self):
        text = fixture("ffi_lsan_real_leak.txt")
        self.assertEqual(names(fs.sanitizer_checks(text), "failed"), ["no LeakSanitizer reports"])
        self.assertEqual(names(fs.c_checks(text), "failed"), [])  # the program itself finished


class Valgrind(unittest.TestCase):
    def test_a_real_clean_run_passes(self):
        text = fixture("ffi_valgrind_real_clean.txt")
        self.assertEqual(names(fs.valgrind_checks(text, 0), "failed"), [])
        self.assertEqual(len(fs.valgrind_checks(text, 0)), 3)
        self.assertEqual(names(fs.c_checks(text), "failed"), [])  # the program's stdout is in the same log

    def test_a_real_leak_fails_all_three(self):
        checks = fs.valgrind_checks(fixture("ffi_valgrind_real_leak.txt"), 99)
        self.assertEqual(len(names(checks, "failed")), 3)
        detail = {c["name"]: c["detail"] for c in checks}
        self.assertEqual(detail["nothing definitely lost"], "100 bytes")
        self.assertEqual(detail["valgrind exit status is 0"], "exit status 99")

    def test_a_log_without_a_summary_is_a_failed_run(self):
        self.assertEqual(names(fs.valgrind_checks("valgrind: command not found\n", 127), "failed"),
                         ["valgrind ran to completion"])

    def test_a_bad_exit_status_alone_fails_one_check(self):
        text = fixture("ffi_valgrind_real_clean.txt")
        self.assertEqual(names(fs.valgrind_checks(text, 1), "failed"), ["valgrind exit status is 0"])


class Cargo(unittest.TestCase):
    def test_a_real_log_gives_one_check_per_test_including_doctests(self):
        checks = fs.cargo_checks(fixture("ffi_cargo_real_run.txt"))
        self.assertEqual(len(checks), 41)
        self.assertEqual(names(checks, "failed"), [])
        self.assertTrue(any("(line" in c["name"] for c in checks))  # the doctest, whose name has spaces

    def test_a_failed_test_is_a_failed_check(self):
        checks = fs.cargo_checks("test a::b ... ok\ntest a::c ... FAILED\ntest a::d ... ignored\n")
        self.assertEqual(names(checks, "failed"), ["a::c"])
        self.assertEqual(len(checks), 2)  # an ignored test is not a check

    def test_no_tests_at_all_is_a_failure_not_an_empty_pass(self):
        for text in ("", "error: could not compile `midman-api`\n", "/bin/sh: cargo: not found\n"):
            self.assertEqual(names(fs.cargo_checks(text), "failed"), ["collected at least one Rust test"])


class Arguments(unittest.TestCase):
    def test_names_may_contain_commas_and_spaces(self):
        self.assertEqual(fs.split_named("C11 test, shared library=ffi-so-raw.txt"),
                         ("C11 test, shared library", "ffi-so-raw.txt"))

    def test_a_value_without_a_name_is_rejected(self):
        for bad in ("just-a-path.txt", "=path.txt"):
            with self.assertRaises(Exception):
                fs.split_named(bad)


class EndToEnd(unittest.TestCase):
    def run_parse(self, tmp, c_log, extra=()):
        out, md = os.path.join(tmp, "r.json"), os.path.join(tmp, "s.md")
        with open(md, "w", encoding="utf-8") as fh:
            fh.write("EXISTING\n")
        env = dict(os.environ, BASELINE_NOTE="a note", RUNNER_LABEL="test-runner")
        proc = subprocess.run(
            [sys.executable, SCRIPT, "parse", "--out", out, "--md", md, "--build", "9", "--branch", "main",
             "--commit", "0123456789abcdef", "--rust-version", "rustc test", "--cc-version", "gcc 13",
             "--cxx-version", "g++ 13",
             "--cargo-raw", os.path.join(FIXTURES, "ffi_cargo_real_run.txt"),
             "--c-log", f"C11 test, shared library={os.path.join(FIXTURES, c_log)}",
             "--sanitizer-log", f"C11 test under ASan and UBSan={os.path.join(FIXTURES, 'ffi_c_real_run.txt')}",
             "--valgrind-log", f"C11 test under valgrind={os.path.join(FIXTURES, 'ffi_valgrind_real_clean.txt')}",
             "--valgrind-status", "0", *extra],
            env=env, capture_output=True, text=True, check=True)
        with open(out, encoding="utf-8") as fh:
            data = json.load(fh)
        with open(md, encoding="utf-8") as fh:
            return data, fh.read(), out, proc

    def test_a_clean_run_passes_the_gate_and_reads_well(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, md, out, proc = self.run_parse(tmp, "ffi_c_real_run.txt")
            self.assertEqual(data["summary"], {"total": 41 + 90 + 93 + 93, "passed": 41 + 90 + 93 + 93, "failed": 0})
            self.assertEqual([s["name"] for s in data["suites"]],
                             ["Rust tests (cargo test -p midman-api)", "C11 test, shared library",
                              "C11 test under ASan and UBSan", "C11 test under valgrind"])
            self.assertTrue(md.startswith("EXISTING\n"))  # appended, not overwritten
            self.assertIn("\u2705 pass", md)
            self.assertIn("Note: a note", md)
            self.assertIn("C: `gcc 13`", md)
            self.assertNotIn("Failed checks", md)
            self.assertNotIn("<details>", md)
            self.assertNotIn('"log"', json.dumps(data))  # raw logs stay out of the JSON
            self.assertEqual(subprocess.run([sys.executable, SCRIPT, "gate", out], capture_output=True).returncode, 0)

    def test_a_failing_run_fails_the_gate_and_the_summary_says_which_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, md, out, _ = self.run_parse(tmp, "ffi_c_real_fail.txt")
            self.assertEqual(data["summary"]["failed"], 2)
            self.assertIn("\u274c FAIL", md)
            self.assertIn("### Failed checks", md)
            self.assertIn("training that blows up returns MIDMAN_STOPPED", md)
            self.assertIn("End of the log: C11 test, shared library", md)
            self.assertEqual(subprocess.run([sys.executable, SCRIPT, "gate", out], capture_output=True).returncode, 1)

    def test_the_gate_fails_with_no_file_and_with_no_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = subprocess.run([sys.executable, SCRIPT, "gate", os.path.join(tmp, "nope.json")],
                                     capture_output=True, text=True)
            self.assertEqual(missing.returncode, 1)
            self.assertIn("No results file", missing.stdout)
            empty = os.path.join(tmp, "empty.json")
            with open(empty, "w", encoding="utf-8") as fh:
                json.dump({"summary": {"total": 0, "passed": 0, "failed": 0}}, fh)
            none = subprocess.run([sys.executable, SCRIPT, "gate", empty], capture_output=True, text=True)
            self.assertEqual(none.returncode, 1)
            self.assertIn("No checks were collected", none.stdout)

    def test_the_log_tail_in_the_summary_is_capped(self):
        suites = [fs.suite("big", [fs.check("x", False)], "z" * (fs.MAX_TAIL_CHARS * 3))]
        meta = {"build": "1", "branch": "b", "commit": "c" * 10, "runner": "r", "rust_version": "", "cc": "",
                "cxx": "", "note": ""}
        md = fs.build_md(meta, suites, fs.summarize(suites))
        self.assertLess(len(md), fs.MAX_TAIL_CHARS + 2000)

    def test_nothing_collected_is_never_a_pass(self):
        meta = {"build": "1", "branch": "b", "commit": "c" * 10, "runner": "r", "rust_version": "", "cc": "",
                "cxx": "", "note": ""}
        md = fs.build_md(meta, [], fs.summarize([]))
        self.assertIn("\u274c FAIL", md)
        self.assertIn("No checks were collected", md)


if __name__ == "__main__":
    unittest.main()
