#!/usr/bin/env python3
# ============================================================================
# NOTICE: Full documentation, design decisions, and fix history for this file
# live in docs/ci-and-workflows.md, section "test_train_smoke_summary.py"
# ============================================================================
"""Self-tests for train_smoke_summary.py, run against a real captured smoke-test
log (scripts/fixtures/train_smoke_real_run.txt, cargo 1.75 format) and synthetic
logs for the modern cargo format and for failures. Run with:

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
import train_smoke_summary as ts  # noqa: E402

FIXTURE = os.path.join(HERE, "fixtures", "train_smoke_real_run.txt")


def fixture_text():
    with open(FIXTURE, encoding="utf-8") as fh:
        return fh.read()


class ParseRealOutput(unittest.TestCase):
    def test_the_result_line_and_the_totals_are_found(self):
        runs, totals = ts.parse(fixture_text())
        self.assertEqual(runs, [(5.5674, 0.0005, 1.0)])
        self.assertEqual(totals, {"status": "ok", "passed": 1, "failed": 0, "ignored": 0})

    def test_ansi_colour_codes_do_not_hide_the_line(self):
        coloured = fixture_text().replace("loss 5.5674", "\x1b[32mloss\x1b[0m 5.5674")
        self.assertEqual(ts.parse(coloured)[0], [(5.5674, 0.0005, 1.0)])

    def test_several_runs_keep_their_order(self):
        text = "loss 5.0000 -> 0.1000, accuracy 0.900\nloss 6.0000 -> 0.2000, accuracy 0.800\n"
        self.assertEqual([r[1] for r in ts.parse(text)[0]], [0.1, 0.2])

    def test_a_failed_run_reports_its_totals(self):
        text = "test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured\n"
        self.assertEqual(ts.parse(text)[1]["status"], "FAILED")


class RawEmbed(unittest.TestCase):
    NOISE = "".join(f"warning: lint noise {i}\n" for i in range(5000))

    def test_the_older_cargo_format_drops_the_build_output(self):
        out = ts.raw_embed(fixture_text())
        self.assertTrue(out.lstrip().startswith("Finished release"))
        self.assertNotIn("Compiling", out)
        self.assertIn("accuracy 1.000", out)

    def test_the_modern_cargo_format_drops_build_noise(self):
        out = ts.raw_embed(self.NOISE + "    Finished `release` profile [optimized] in 2s\nrunning 1 test\n")
        self.assertTrue(out.lstrip().startswith("Finished `release`"))
        self.assertNotIn("lint noise", out)

    def test_without_the_marker_the_log_is_kept_but_bounded(self):
        out = ts.raw_embed("error: could not compile\n" + self.NOISE)
        self.assertLessEqual(len(out.splitlines()), ts.TAIL_LINES)
        self.assertLessEqual(len(out), ts.MAX_RAW_EMBED)

    def test_a_huge_single_line_is_capped(self):
        self.assertLessEqual(len(ts.raw_embed("x" * (ts.MAX_RAW_EMBED * 3))), ts.MAX_RAW_EMBED)


class CommandLine(unittest.TestCase):
    def run_cli(self, log, outcome, env_extra=None):
        with tempfile.TemporaryDirectory() as tmp:
            json_path, md_path = os.path.join(tmp, "r.json"), os.path.join(tmp, "s.md")
            env = dict(os.environ, **(env_extra or {}))
            subprocess.run(
                [sys.executable, os.path.join(HERE, "train_smoke_summary.py"), "--log", log,
                 "--outcome", outcome, "--build", "3", "--commit", "0123456789abcdef", "--json", json_path,
                 "--md", md_path], env=env, check=True, capture_output=True, text=True)
            with open(json_path, encoding="utf-8") as fh:
                data = json.load(fh)
            with open(md_path, encoding="utf-8") as fh:
                return data, fh.read()

    def test_a_passing_real_run(self):
        data, md = self.run_cli(FIXTURE, "success", {"RUNNER_LABEL": "ubuntu-latest", "BASELINE_NOTE": "n1"})
        self.assertEqual(data["runs"][0]["accuracy"], 1.0)
        self.assertEqual(data["meta"]["runner"], "ubuntu-latest")
        self.assertIn("| 1 | 5.5674 | 0.0005 | 100.0% |", md)
        self.assertIn("Note: n1", md)
        self.assertIn("Runner: `ubuntu-latest`", md)
        self.assertNotIn("End of the raw log", md)

    def test_a_failing_run_shows_the_log_and_counts_panics(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "bad.txt")
            with open(log, "w", encoding="utf-8") as fh:
                fh.write("    Finished `release` profile\nthread 'x' panicked at src/a.rs:1:1\n"
                         "test result: FAILED. 0 passed; 1 failed; 0 ignored\n")
            data, md = self.run_cli(log, "failure")
        self.assertEqual(data["panics"], 1)
        self.assertIn("❌ fail", md)
        self.assertIn("| panic | 1 |", md)
        self.assertIn("End of the raw log", md)
        self.assertIn("No `loss", md)

    def test_a_missing_log_is_not_an_error(self):
        data, md = self.run_cli("/nonexistent/log.txt", "failure")
        self.assertEqual(data["runs"], [])
        self.assertIn("No `loss", md)


if __name__ == "__main__":
    unittest.main()
