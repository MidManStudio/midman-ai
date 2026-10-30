#!/usr/bin/env python3
# ============================================================================
# NOTICE: Full documentation, design decisions, and fix history for this file
# live in docs/ci-and-workflows.md, section "test_train_summary.py"
# ============================================================================
"""Self-tests for train_summary.py, on the real logs of a training run.

Run with `python3 -m unittest discover -s scripts -p 'test_*.py'`.
"""
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import train_summary as ts  # noqa: E402

FIXTURES = os.path.join(HERE, "fixtures")


def fixture(name):
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as handle:
        return handle.read()


class ParseTests(unittest.TestCase):
    def test_the_real_training_log(self):
        rows = ts.parse_train(fixture("train-run-train.tsv"))
        self.assertEqual(len(rows), 20)
        self.assertEqual(rows[0], {"step": 5, "loss": 8.230307, "grad_norm": 1.775724, "lr": 0.002,
                                   "tokens": 10240, "seconds": 12.7})
        self.assertEqual(rows[-1]["step"], 100)
        self.assertEqual(rows[-1]["tokens"], 204800)
        self.assertTrue(all(isinstance(r["step"], float) for r in rows))

    def test_the_real_evaluation_log_keeps_one_line_per_step(self):
        # The file has the line of step 100 twice (the loop and the end of the run both
        # evaluated); the report has to show it once.
        text = fixture("train-run-eval.tsv")
        self.assertEqual(text.count("\n100\t"), 2)
        rows = ts.parse_eval(text)
        self.assertEqual([r["step"] for r in rows], [16, 25, 50, 75, 100])
        self.assertEqual(rows[-1]["eval_loss"], 6.784648)

    def test_the_real_summary(self):
        self.assertEqual(ts.parse_summary(fixture("train-run-summary.txt")),
                         {"steps": 100, "total_steps": 100, "tokens": 204800, "finished": True})
        stopped = ts.parse_summary("steps 16\ntotal_steps 100\ntokens 32768\nfinished false\n")
        self.assertFalse(stopped["finished"])
        self.assertEqual(stopped["steps"], 16)

    def test_logs_that_are_not_what_train_writes(self):
        good = fixture("train-run-train.tsv")
        for name, text in {
            "no header": good.split("\n", 1)[1],
            "other header": good.replace("grad_norm", "gradient", 1),
            "short line": good + "101\t5.0\n",
            "long line": good + "101\t5.0\t1\t1\t1\t1\t1\n",
            "not a number": good + "101\tx\t1\t1\t1\t1\n",
            "empty": "",
        }.items():
            with self.assertRaises(ts.SummaryError, msg=name):
                ts.parse_train(text)
        with self.assertRaises(ts.SummaryError):
            ts.parse_eval("step\tloss\ttokens\n1\t2\t3\n")
        for text in ("steps 1\n", "steps 1\ntotal_steps 2\ntokens 3\nfinished maybe\n",
                     "steps x\ntotal_steps 2\ntokens 3\nfinished true\n", "steps 1 2\n", "steps\n"):
            with self.assertRaises(ts.SummaryError, msg=text):
                ts.parse_summary(text)

    def test_a_step_logged_twice_keeps_the_later_line(self):
        text = "step\teval_loss\ttokens\n5\t2.0\t10\n6\t1.5\t10\n5\t1.0\t20\n"
        self.assertEqual(ts.parse_eval(text), [
            {"step": 5, "eval_loss": 1.0, "tokens": 20}, {"step": 6, "eval_loss": 1.5, "tokens": 10}])

    def test_blank_lines_are_skipped(self):
        text = "step\teval_loss\ttokens\n\n1\t2.5\t30\n\n"
        self.assertEqual(ts.parse_eval(text), [{"step": 1, "eval_loss": 2.5, "tokens": 30}])


class RowsAndSpeedTests(unittest.TestCase):
    def test_pick_rows_spreads_evenly_and_keeps_the_ends(self):
        rows = list(range(100))
        picked = ts.pick_rows(rows, 5)
        self.assertEqual(picked, [0, 25, 50, 74, 99])
        self.assertEqual(ts.pick_rows(rows, 2), [0, 99])
        self.assertEqual(ts.pick_rows(rows[:5], 5), rows[:5])
        self.assertEqual(ts.pick_rows(rows[:3], 12), rows[:3])
        self.assertEqual(ts.pick_rows([], 4), [])
        self.assertEqual(len(ts.pick_rows(rows, 12)), 12)
        with self.assertRaises(ts.SummaryError):
            ts.pick_rows(rows, 1)

    def test_speed_ignores_the_restart_of_the_clock_at_a_resume(self):
        rows = ts.parse_train(fixture("train-run-train.tsv"))
        # Between step 15 (35.7 s) and step 20 (10.3 s) the clock started again.
        self.assertLess(rows[3]["seconds"], rows[2]["seconds"])
        speed = ts.tokens_per_second(rows)
        self.assertTrue(800 < speed < 1000, speed)

    def test_speed_is_the_median_of_the_intervals(self):
        def row(tokens, seconds):
            return {"tokens": tokens, "seconds": seconds}

        rows = [row(0, 0), row(100, 1), row(1100, 2), row(1300, 4), row(1300, 5), row(1300, 6),
                row(1300, 7), row(1400, 7)]
        # Intervals: 100, 1000, 100 tokens a second; the others have no tokens or no time.
        self.assertEqual(ts.tokens_per_second(rows), 100.0)
        self.assertIsNone(ts.tokens_per_second([row(0, 0)]))
        self.assertIsNone(ts.tokens_per_second([]))


class ReportTests(unittest.TestCase):
    def real(self):
        return (ts.parse_train(fixture("train-run-train.tsv")),
                ts.parse_eval(fixture("train-run-eval.tsv")),
                ts.parse_summary(fixture("train-run-summary.txt")))

    def test_the_report_of_the_real_run(self):
        train, evals, summary = self.real()
        text = ts.report(train, evals, summary, "a note", 4096)
        self.assertTrue(text.startswith("## midman-ai train\n\na note\n\nfinished: 100 of 100 steps, 204,800 tokens trained on.\n"))
        self.assertIn("Speed: about ", text)
        self.assertIn("ln(4096) = 8.318 nats per token.", text)
        self.assertIn("| 16 | 7.195 | 16,384 |", text)
        self.assertIn("| 100 | 6.785 | 16,384 |", text)
        self.assertEqual(text.count("| 100 | 6.785"), 1)
        self.assertIn("Held-out loss went from 7.195 to 6.785.", text)
        self.assertIn("| 5 | 8.230 | 1.776 | 2.00e-03 | 10,240 |", text)
        self.assertIn("Largest gradient norm logged: 5.109 at step 20.", text)
        # 12 rows at most in the training table, and every evaluation (there are five).
        body = text.split("### Training loss")[1]
        self.assertEqual(body.count("\n| ") - 1, 12)
        self.assertTrue(text.endswith("\n"))

    def test_a_run_that_was_stopped_says_so_and_a_missing_piece_is_left_out(self):
        train, evals, summary = self.real()
        summary = dict(summary, finished=False, steps=16)
        text = ts.report(train[:3], [], summary)
        self.assertIn("stopped on the time budget at step 16 of 100 (the state can be resumed)", text)
        self.assertNotIn("Held-out", text)
        self.assertNotIn("ln(", text)
        self.assertNotIn("Held-out loss went", ts.report(train, evals[:1], summary))
        empty = ts.report([], [], summary)
        self.assertIn("The training log has no lines yet.", empty)
        self.assertNotIn("Speed", empty)
        self.assertNotIn("note", empty.split("\n")[2] if len(empty.split("\n")) > 2 else "")

    def test_the_rows_shown_can_be_set(self):
        train, evals, summary = self.real()
        text = ts.report(train, evals, summary, rows=4)
        self.assertEqual(text.split("### Training loss")[1].count("\n| ") - 1, 4)
        self.assertEqual(text.split("### Training loss")[0].count("\n| ") - 1, 4)


class CommandTests(unittest.TestCase):
    def make_run(self, root, with_eval=True):
        for name in ("train.tsv", "summary.txt"):
            with open(os.path.join(root, name), "w", encoding="utf-8") as handle:
                handle.write(fixture({"train.tsv": "train-run-train.tsv", "summary.txt": "train-run-summary.txt"}[name]))
        if with_eval:
            with open(os.path.join(root, "eval.tsv"), "w", encoding="utf-8") as handle:
                handle.write(fixture("train-run-eval.tsv"))

    def run_main(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = ts.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_print_and_append(self):
        with tempfile.TemporaryDirectory() as root:
            self.make_run(root)
            code, out, err = self.run_main(["--run-dir", root, "--vocab", "4096"])
            self.assertEqual((code, err), (0, ""))
            self.assertTrue(out.startswith("## midman-ai train"))
            self.assertIn("### Held-out loss", out)
            summary = os.path.join(root, "s.md")
            with open(summary, "w", encoding="utf-8") as handle:
                handle.write("before\n")
            code, out, err = self.run_main(["--run-dir", root, "--markdown", summary, "--note", "n"])
            self.assertEqual((code, out), (0, ""))
            with open(summary, encoding="utf-8") as handle:
                text = handle.read()
            self.assertTrue(text.startswith("before\n## midman-ai train\n\nn\n"))

    def test_without_an_evaluation_log(self):
        with tempfile.TemporaryDirectory() as root:
            self.make_run(root, with_eval=False)
            code, out, _err = self.run_main(["--run-dir", root])
            self.assertEqual(code, 0)
            self.assertNotIn("Held-out", out)

    def test_a_missing_or_broken_log_is_an_error_with_a_message(self):
        with tempfile.TemporaryDirectory() as root:
            code, _out, err = self.run_main(["--run-dir", root])
            self.assertEqual(code, 1)
            self.assertIn("train_summary:", err)
            self.make_run(root)
            with open(os.path.join(root, "summary.txt"), "w", encoding="utf-8") as handle:
                handle.write("steps 1\n")
            code, _out, err = self.run_main(["--run-dir", root])
            self.assertEqual(code, 1)
            self.assertIn("total_steps", err)


if __name__ == "__main__":
    unittest.main()
