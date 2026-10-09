#!/usr/bin/env python3
# ============================================================================
# NOTICE: Full documentation, design decisions, and fix history for this file
# live in docs/ci-and-workflows.md, section "test_bench_summary.py"
# ============================================================================
"""Self-tests for bench_summary.py, run against real captured criterion output
(scripts/fixtures/criterion_real_run.txt) and a few synthetic logs for the
cases real output cannot show on demand. Run with:

    python3 -m unittest discover -s scripts -p 'test_*.py'
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_summary as bs  # noqa: E402

FIXTURE = os.path.join(HERE, "fixtures", "criterion_real_run.txt")


def fixture_text():
    return bs.read(FIXTURE)


class ParseRealOutput(unittest.TestCase):
    def setUp(self):
        self.text = fixture_text()
        self.results = bs.parse(self.text)
        self.by_name = {r["name"]: r for r in self.results}

    def test_every_result_line_is_found_and_nothing_else(self):
        # Count independently: lines that open a result are `time: [<digit>`.
        # The `time:   [+17.1% ...]` lines inside `change:` blocks must not count.
        expected = sum(1 for line in self.text.splitlines() if re.search(r"time:\s+\[\d", line))
        self.assertEqual(expected, 13)
        self.assertEqual(len(self.results), expected)
        self.assertTrue(any(line.strip() == "change:" for line in self.text.splitlines()))
        self.assertTrue(all(r["mid_ns"] > 0 for r in self.results))

    def test_names_come_from_the_own_line_form_and_the_same_line_form(self):
        self.assertIn("train_step_smoke_b4_t32/unit-forward", self.by_name)  # id on its own line
        self.assertIn("model_build/smoke", self.by_name)  # id on the same line as `time:`
        self.assertIn("model_build/tiny", self.by_name)

    def test_ids_split_at_the_final_slash(self):
        r = self.by_name["attention_forward_t16/baseline-mha"]
        self.assertEqual((r["group"], r["variant"]), ("attention_forward_t16", "baseline-mha"))
        self.assertEqual(bs.split_id("a/b/c"), ("a/b", "c"))
        self.assertEqual(bs.split_id("lonely"), ("(ungrouped)", "lonely"))

    def test_units_are_converted_to_nanoseconds(self):
        r = self.by_name["model_build/smoke"]
        self.assertAlmostEqual(r["mid_ns"], 2.3045e6, delta=1)  # "2.3045 ms"
        self.assertAlmostEqual(r["low_ns"], 2.2632e6, delta=1)
        self.assertAlmostEqual(r["high_ns"], 2.4140e6, delta=1)
        micro = self.by_name["attention_forward_t16/baseline-mha"]
        self.assertTrue(1e4 < micro["mid_ns"] < 1e6)  # a "µs" result

    def test_throughput_with_and_without_a_unit_prefix(self):
        units = {r["throughput"].split()[-1] for r in self.results if r["throughput"]}
        self.assertIn("Kelem/s", units)
        self.assertIn("elem/s", units)  # criterion pads this one with two spaces
        self.assertIn("Melem/s", units)
        plain = next(r for r in self.results if r["throughput"] and r["throughput"].endswith(" elem/s"))
        self.assertRegex(plain["throughput"], r"^[\d.]+ elem/s$")

    def test_outliers_are_attached_to_the_right_result(self):
        flagged = [r for r in self.results if r["samples"]]
        self.assertTrue(flagged)
        self.assertTrue(all(r["samples"] == 10 and 0 < r["outliers"] <= 10 for r in flagged))

    def test_a_clean_real_run_has_no_diagnostics(self):
        self.assertEqual(bs.diagnostics(self.text), [])


class Grouping(unittest.TestCase):
    def setUp(self):
        self.groups = {g["group"]: g for g in bs.group_results(bs.parse(fixture_text()))}

    def test_a_unit_variant_is_a_plain_multiple(self):
        g = self.groups["train_step_smoke_b4_t32"]
        self.assertEqual((g["kind"], g["denominator"]["variant"]), ("unit", "unit-forward"))
        rows = {r["variant"]: r for r in g["rows"]}
        self.assertIsNone(rows["unit-forward"]["ratio"])
        self.assertGreater(rows["forward-backward"]["ratio"], 2.0)  # backward costs more than forward
        expected = rows["forward-backward"]["mid_ns"] / rows["unit-forward"]["mid_ns"]
        self.assertAlmostEqual(rows["forward-backward"]["ratio"], expected)

    def test_a_baseline_variant_gets_badged_ratios(self):
        g = self.groups["attention_forward_t16"]
        self.assertEqual((g["kind"], g["denominator"]["variant"]), ("baseline", "baseline-mha"))
        self.assertEqual([r["variant"] for r in g["rows"]], ["baseline-mha", "gqa-2", "mqa"])

    def test_a_group_with_neither_has_no_ratio(self):
        g = self.groups["model_build"]
        self.assertIsNone(g["kind"])
        self.assertTrue(all(r["ratio"] is None for r in g["rows"]))

    def test_groups_keep_the_order_of_the_log(self):
        names = list(self.groups)
        self.assertEqual(names[0], "train_step_smoke_b4_t32")
        self.assertEqual(names.index("model_build"), names.index("train_step_tiny_b1_t32") + 1)

    def test_a_baseline_wins_over_a_unit_in_the_same_group(self):
        text = (
            "g/unit-a\n                        time:   [1.0 ms 1.0 ms 1.0 ms]\n"
            "g/baseline-b\n                        time:   [2.0 ms 2.0 ms 2.0 ms]\n"
            "g/other\n                        time:   [4.0 ms 4.0 ms 4.0 ms]\n"
        )
        g = bs.group_results(bs.parse(text))[0]
        self.assertEqual((g["kind"], g["denominator"]["variant"]), ("baseline", "baseline-b"))
        self.assertAlmostEqual(g["rows"][2]["ratio"], 2.0)


class Badges(unittest.TestCase):
    def test_thresholds(self):
        self.assertIn("faster", bs.badge(0.5))
        self.assertIn("parity", bs.badge(1.0))
        self.assertIn("parity", bs.badge(1.05))
        self.assertTrue(bs.badge(1.2).startswith("⚠️"))
        self.assertTrue(bs.badge(3.0).startswith("❌"))
        self.assertIn("slower", bs.badge(7.0))


class Markdown(unittest.TestCase):
    META = {"build": "7", "branch": "main", "commit": "abc", "rust_version": "rustc test",
            "runner": "ubuntu-latest", "package": "midman-bench", "filter": "", "note": "a note"}

    def render(self, text, outcome="success"):
        results = bs.parse(text)
        return bs.build_md(self.META, outcome, bs.group_results(results), len(results),
                           bs.diagnostics(text), bs.raw_embed(text, outcome, bool(results)))

    def test_the_real_run_renders_every_group_and_both_ratio_columns(self):
        md = self.render(fixture_text())
        for group in ("### train_step_smoke_b4_t32", "### model_build", "### attention_forward_t16"):
            self.assertIn(group, md)
        self.assertIn("x the unit variant", md)
        self.assertIn("vs baseline", md)
        self.assertIn("Note: a note", md)
        self.assertIn("Runner: `ubuntu-latest`", md)
        self.assertNotIn("End of raw log", md)  # a clean pass shows no raw text

    def test_a_unit_group_never_shows_a_badge(self):
        md = self.render(fixture_text())
        section = md.split("### train_step_smoke_b4_t32")[1].split("###")[0]
        for emoji in ("✅", "⚠️", "❌", "🔴"):
            self.assertNotIn(emoji, section)
        self.assertRegex(section, r"\d+\.\d\dx")

    def test_a_baseline_group_shows_badges(self):
        md = self.render(fixture_text())
        section = md.split("### attention_forward_t16")[1].split("###")[0]
        self.assertTrue(any(e in section for e in ("✅", "⚠️", "❌", "🔴")))

    def test_no_results_says_so_and_embeds_the_log(self):
        md = self.render("error: could not compile `x`\nsome rustc output\n", outcome="failure")
        self.assertIn("No criterion results were found", md)
        self.assertIn("could not compile", md)
        self.assertIn("❌ FAIL", md)


class RawEmbed(unittest.TestCase):
    PREAMBLE = "".join(f"warning: lint noise {i}\n" for i in range(5000))

    def test_build_noise_before_the_finished_line_is_dropped(self):
        text = self.PREAMBLE + "    Finished `bench` profile [optimized] in 1s\nbench output\n"
        out = bs.raw_embed(text, "failure", True)
        self.assertTrue(out.lstrip().startswith("Finished `bench`"))
        self.assertNotIn("lint noise", out)

    def test_the_older_cargo_finished_format_is_recognized_too(self):
        text = self.PREAMBLE + "    Finished bench [optimized + debuginfo] target(s) in 1.00s\nbench output\n"
        out = bs.raw_embed(text, "failure", True)
        self.assertTrue(out.lstrip().startswith("Finished bench"))
        self.assertNotIn("lint noise", out)

    def test_a_word_finished_inside_a_line_is_not_a_marker(self):
        text = "warning: x\ntest result: ok. finished in 0.1s\nwarning: y\n"
        self.assertEqual(bs.raw_embed(text, "failure", False), text)

    def test_without_the_marker_the_whole_log_is_kept(self):
        text = "error[E0432]: unresolved import\n" + self.PREAMBLE[:200]
        self.assertTrue(bs.raw_embed(text, "failure", False).startswith("error[E0432]"))

    def test_the_embed_is_capped(self):
        text = "x" * (bs.MAX_RAW_EMBED * 3)
        self.assertEqual(len(bs.raw_embed(text, "failure", False)), bs.MAX_RAW_EMBED)

    def test_a_clean_pass_embeds_nothing(self):
        self.assertEqual(bs.raw_embed(fixture_text(), "success", True), "")

    def test_a_failure_after_some_results_still_embeds_the_tail(self):
        self.assertNotEqual(bs.raw_embed(fixture_text(), "failure", True), "")


class Diagnostics(unittest.TestCase):
    def test_criterion_warnings_are_counted(self):
        text = (
            "Warning: Unable to complete 100 samples in 5.0s. You may wish to increase target time.\n"
            "thread 'main' panicked at src/x.rs:1:1\n"
            "benchmark took zero time\n"
        )
        kinds = {d["kind"]: d["count"] for d in bs.diagnostics(text)}
        self.assertEqual(kinds["unable to complete samples"], 1)
        self.assertEqual(kinds["criterion warning"], 1)
        self.assertEqual(kinds["panic"], 1)
        self.assertEqual(kinds["took zero time"], 1)


class CommandLine(unittest.TestCase):
    def test_end_to_end_writes_json_and_appends_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            json_path, md_path = os.path.join(tmp, "r.json"), os.path.join(tmp, "s.md")
            with open(md_path, "w", encoding="utf-8") as fh:
                fh.write("EXISTING\n")
            env = dict(os.environ, BUILD_NUM="12", BASELINE_NOTE="cli test", RUNNER_LABEL="test-runner")
            proc = subprocess.run(
                [sys.executable, os.path.join(HERE, "bench_summary.py"), "--log", FIXTURE,
                 "--outcome", "success", "--json", json_path, "--md", md_path],
                env=env, capture_output=True, text=True, check=True)
            self.assertIn("13 benchmark(s) parsed", proc.stderr)
            with open(json_path, encoding="utf-8") as fh:
                data = json.load(fh)
            self.assertEqual(data["meta"]["build"], "12")
            self.assertEqual(data["meta"]["runner"], "test-runner")
            self.assertEqual(sum(len(g["rows"]) for g in data["groups"]), 13)
            with open(md_path, encoding="utf-8") as fh:
                md = fh.read()
            self.assertTrue(md.startswith("EXISTING\n"))  # appended, not overwritten
            self.assertIn("cli test", md)

    def test_a_missing_log_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [sys.executable, os.path.join(HERE, "bench_summary.py"), "--log", os.path.join(tmp, "nope"),
                 "--outcome", "failure", "--json", os.path.join(tmp, "r.json")],
                capture_output=True, text=True, check=True)
            self.assertIn("No criterion results were found", proc.stdout)


if __name__ == "__main__":
    unittest.main()
