#!/usr/bin/env python3
# ============================================================================
# NOTICE: Full documentation, design decisions, and fix history for this file
# live in docs/ci-and-workflows.md, section "test_data_smoke_summary.py"
# ============================================================================
"""Self-tests for data_smoke_summary.py, run against a real log of the data-smoke
workflow's steps (scripts/fixtures/data-smoke-ubuntu.txt, captured by running the
workflow's own run blocks) and against logs made from it with one thing changed.
Run with:

    python3 -m unittest discover -s scripts -p 'test_*.py'
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data_smoke_summary as ds  # noqa: E402

FIXTURE = os.path.join(HERE, "fixtures", "data-smoke-ubuntu.txt")
SCRIPT = os.path.join(HERE, "data_smoke_summary.py")
ALL_OK = "build=success,corpus=success,train=success,shards=success,speed=success,oracle=success"
META = {"build": "7", "branch": "main", "commit": "0123456789abcdef", "rust_version": "rustc 1.99.1",
        "runner": "ubuntu-latest (Linux X64)", "note": ""}


def real():
    return read(FIXTURE)


def read(path, mode="r"):
    with open(path, mode) as handle:
        return handle.read()


def put(path, data):
    """Write text or bytes to a file and close it."""
    with open(path, "wb" if isinstance(data, bytes) else "w") as handle:
        handle.write(data)


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def result(text=None, outcomes=ALL_OK, tokenizer=None, shards=None, meta=None):
    return ds.build_result(real() if text is None else text, outcomes, tokenizer, shards, meta or META)


class ParseRealLog(unittest.TestCase):
    def setUp(self):
        self.parsed = ds.parse(real())

    def test_sections(self):
        names = list(ds.sections(real()))
        self.assertEqual(names, ["build", "corpus", "train", "shards", "scrub speed", "oracle"])

    def test_corpus_and_training(self):
        self.assertEqual(self.parsed["corpus_files"], 256)
        train = self.parsed["train"]
        self.assertEqual((train["files"], train["unreadable"], train["bytes"]), (256, 0, 1607185))
        self.assertEqual((train["merges"], train["merges_wanted"], train["vocab_size"]), (760, 760, 1024))

    def test_shards(self):
        s = self.parsed["shards"]
        self.assertEqual((s["files_read"], s["not_utf8"], s["empty"], s["duplicates"]), (256, 0, 0, 0))
        self.assertEqual((s["secrets"], s["secrets_by_rule"]), (1, {"url-password": 1}))
        self.assertEqual((s["documents"], s["bytes"], s["tokens"]), (256, 1607187, 713006))
        self.assertAlmostEqual(s["bytes_per_token"], 2.25)
        self.assertEqual((s["shard_count"], s["sequences"], s["seq_len"]), (4, 5568, 128))
        self.assertAlmostEqual(s["seconds"], 0.2)
        self.assertAlmostEqual(s["mb_per_s"], 8.0)
        self.assertEqual(s["batches"], 100)
        self.assertEqual(s["batch_digest"],
                         "f4fdebabe1204c97c083301d8eab933b5a236da0d87e9330d9234b655d56120c")

    def test_speed(self):
        self.assertAlmostEqual(self.parsed["scrub_mb_per_s"], 72.1)

    def test_oracle(self):
        o = self.parsed["oracle"]
        self.assertEqual(o["gitleaks_version"], "8.30.1")
        self.assertEqual(len(o["kinds"]), 24)
        self.assertTrue(o["ok"])
        self.assertEqual((o["ours_outside"], o["gitleaks_outside"]), (0, 0))
        by_kind = {k["kind"]: k for k in o["kinds"]}
        self.assertEqual(by_kind["url-password"]["gitleaks"], 0)
        self.assertEqual(by_kind["aws-secret"]["gitleaks"], 11)
        self.assertEqual(by_kind["aws-secret"]["ours"], 12)
        self.assertTrue(all(k["samples"] == 12 for k in o["kinds"]))

    def test_the_result_is_ok(self):
        r = result()
        self.assertTrue(r["ok"])
        self.assertEqual(r["outcomes"]["oracle"], "success")
        self.assertEqual(r["meta"]["runner"], "ubuntu-latest (Linux X64)")


class Robustness(unittest.TestCase):
    def test_empty_and_garbage_logs_do_not_raise_and_are_not_ok(self):
        for text in ("", "nothing here\n", "=== train ===\nnoise\n", "=== shards ===\n", "\x1b[31m=== oracle ===\x1b[0m\nx"):
            r = result(text, outcomes="")
            self.assertFalse(r["ok"], repr(text))
            self.assertIsNone(r["train"])

    def test_color_codes_are_ignored(self):
        colored = real().replace("learned", "\x1b[32mlearned\x1b[0m")
        self.assertEqual(ds.parse(colored)["train"]["merges"], 760)

    def test_outcome_parsing(self):
        self.assertEqual(ds.parse_outcomes("build=success, oracle=skipped,,junk"),
                         {"build": "success", "oracle": "skipped"})
        self.assertEqual(ds.parse_outcomes(""), {})


class Decision(unittest.TestCase):
    def test_a_failed_or_cancelled_step_is_not_ok(self):
        for step in ("build", "corpus", "train", "shards", "speed", "oracle"):
            for outcome in ("failure", "cancelled"):
                outcomes = ALL_OK.replace(f"{step}=success", f"{step}={outcome}")
                self.assertFalse(result(outcomes=outcomes)["ok"], f"{step} {outcome}")

    def test_a_skipped_step_is_ok_only_for_the_oracle(self):
        self.assertTrue(result(outcomes=ALL_OK.replace("oracle=success", "oracle=skipped"))["ok"])
        self.assertFalse(result(outcomes=ALL_OK.replace("speed=success", "speed=skipped"))["ok"])

    def test_off_the_oracle_runner_the_section_is_absent(self):
        text = real().split("=== oracle ===")[0]
        r = result(text, outcomes=ALL_OK.replace("oracle=success", "oracle=skipped"))
        self.assertTrue(r["ok"])
        self.assertIsNone(r["oracle"])
        self.assertIn("skipped here", ds.markdown(r, text))

    def test_an_oracle_that_succeeded_but_printed_nothing_is_not_ok(self):
        text = real().split("=== oracle ===")[0]
        self.assertFalse(result(text)["ok"])

    def test_a_missed_sample_fails_the_oracle(self):
        text = real().replace("github-pat             12    12  12", "github-pat             12    11  12")
        r = result(text)
        self.assertFalse(r["oracle"]["ok"])
        self.assertFalse(r["ok"])
        self.assertIn("\u274c", ds.markdown(r, text))

    def test_a_finding_outside_the_samples_fails_the_oracle(self):
        text = real().replace("findings of ours outside the samples: 0", "findings of ours outside the samples: 2")
        self.assertFalse(result(text)["ok"])

    def test_gitleaks_finding_extra_alone_does_not_fail_us(self):
        text = real().replace("findings of gitleaks outside the samples: 0", "findings of gitleaks outside the samples: 5")
        r = result(text)
        self.assertTrue(r["ok"])
        self.assertEqual(r["oracle"]["gitleaks_outside"], 5)

    def test_a_missing_batch_digest_or_speed_is_not_ok(self):
        no_digest = "\n".join(l for l in real().splitlines() if not l.startswith("first 100 batches"))
        self.assertFalse(result(no_digest)["ok"])
        no_speed = "\n".join(l for l in real().splitlines() if " MB in " not in l)
        self.assertFalse(result(no_speed)["ok"])

    def test_missing_pipeline_lines_are_not_ok(self):
        for prefix in ("learned ", "documents kept", "files read"):
            text = "\n".join(l for l in real().splitlines() if not l.startswith(prefix))
            self.assertFalse(result(text)["ok"], prefix)

    def test_secrets_by_rule_and_zero_secrets(self):
        text = real().replace('secrets replaced 1 {"url-password": 1}', 'secrets replaced 3 {"a-b": 2, "c": 1}')
        self.assertEqual(ds.parse(text)["shards"]["secrets_by_rule"], {"a-b": 2, "c": 1})
        text = real().replace('secrets replaced 1 {"url-password": 1}', "secrets replaced 0 {}")
        self.assertEqual(ds.parse(text)["shards"]["secrets"], 0)


class Digests(unittest.TestCase):
    def test_hashes_of_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tok = os.path.join(tmp, "tokenizer.txt")
            shards = os.path.join(tmp, "shards")
            os.mkdir(shards)
            put(tok, b"vocabulary")
            put(os.path.join(shards, "shard-00001.mmtokens"), b"two")
            put(os.path.join(shards, "shard-00000.mmtokens"), b"one")
            put(os.path.join(shards, "notes.txt"), b"ignored")
            d = ds.digests(tok, shards)
            self.assertEqual(d["vocabulary"], hashlib.sha256(b"vocabulary").hexdigest())
            self.assertEqual(d["shards"], [hashlib.sha256(b"one").hexdigest(), hashlib.sha256(b"two").hexdigest()])
            listing = "shard-00000.mmtokens\n%s\nshard-00001.mmtokens\n%s\n" % (
                hashlib.sha256(b"one").hexdigest(), hashlib.sha256(b"two").hexdigest())
            self.assertEqual(d["combined"], hashlib.sha256(listing.encode()).hexdigest())

    def test_a_different_file_changes_the_combined_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            shards = os.path.join(tmp, "s")
            os.mkdir(shards)
            path = os.path.join(shards, "shard-00000.mmtokens")
            put(path, b"a")
            first = ds.digests(None, shards)["combined"]
            put(path, b"b")
            self.assertNotEqual(first, ds.digests(None, shards)["combined"])

    def test_missing_paths_give_none(self):
        d = ds.digests("/no/such/file", "/no/such/dir")
        self.assertEqual((d["vocabulary"], d["shards"], d["combined"]), (None, [], None))
        d = ds.digests(None, None)
        self.assertIsNone(d["vocabulary"])


class Markdown(unittest.TestCase):
    def test_a_passing_summary(self):
        text = real()
        md = ds.markdown(result(text), text)
        for needle in ("## midman-ai data smoke: build #7", "Result: \u2705 pass", "| build | \u2705 success |",
                       "256 tracked text files", "1,024 ids, 760 merges", "713,006", "2.25 bytes per token",
                       "4, holding 5,568 sequences of 128 tokens", "72.1 MB/s",
                       "### Scrubber against gitleaks 8.30.1", "| aws-secret | 11 | 12 | 12 |",
                       "Findings outside the samples: scrubber 0, gitleaks 0", "`f4fdebabe120`"):
            self.assertIn(needle, md)
        self.assertNotIn("End of the raw log", md)

    def test_a_failing_summary_shows_the_end_of_the_log(self):
        text = real()
        md = ds.markdown(result(text, outcomes=ALL_OK.replace("train=success", "train=failure")), text)
        self.assertIn("Result: \u274c fail", md)
        self.assertIn("| train | \u274c failure |", md)
        self.assertIn("End of the raw log", md)

    def test_the_note_and_panics_are_shown(self):
        meta = dict(META, note="after the scrubber change")
        text = real() + "\nthread 'main' panicked at src/x.rs:1:1:\nboom\n"
        md = ds.markdown(result(text, meta=meta), text)
        self.assertIn("Note: after the scrubber change", md)
        self.assertIn("| panic | 1 |", md)

    def test_an_unparsable_log_says_so(self):
        md = ds.markdown(result("", outcomes="build=failure"), "")
        self.assertIn("does not hold the lines", md)

    def test_raw_embed_starts_at_the_last_finished_line_and_is_bounded(self):
        text = "Compiling a\n    Finished `release`\n" + "\n".join(f"line {n}" for n in range(500))
        embedded = ds.raw_embed(text)
        self.assertEqual(len(embedded.splitlines()), ds.TAIL_LINES)
        self.assertTrue(embedded.endswith("line 499"))
        self.assertNotIn("Compiling a", ds.raw_embed("Compiling a\n    Finished x\nok"))


class CommandLine(unittest.TestCase):
    def run_cli(self, *args, env=None):
        return subprocess.run([sys.executable, SCRIPT, *args], capture_output=True, text=True,
                              env=dict(os.environ, **(env or {})))

    def test_writing_json_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            log, out, md = (os.path.join(tmp, n) for n in ("raw.txt", "out.json", "summary.md"))
            put(log, real())
            done = self.run_cli("--log", log, "--outcomes", ALL_OK, "--tokenizer", "none", "--shards", "none",
                                "--build", "9", "--branch", "main", "--commit", "abc", "--rust-version", "rustc x",
                                "--json", out, "--md", md,
                                env={"RUNNER_LABEL": "ubuntu-latest (Linux X64)", "BASELINE_NOTE": "n"})
            self.assertEqual(done.returncode, 0, done.stderr)
            data = load(out)
            self.assertTrue(data["ok"])
            self.assertEqual((data["meta"]["build"], data["meta"]["note"]), ("9", "n"))
            self.assertIn("build #9", read(md))

    def test_a_missing_log_still_writes_a_failing_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out.json")
            done = self.run_cli("--log", os.path.join(tmp, "absent.txt"), "--json", out, "--md", os.path.join(tmp, "s.md"))
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertFalse(load(out)["ok"])

    def test_no_mode_is_an_error(self):
        self.assertNotEqual(self.run_cli().returncode, 0)


def runner_result(runner, **changes):
    base = load(os.path.join(HERE, "fixtures", "data-smoke-results.json"))
    base["meta"]["runner"] = runner
    for dotted, value in changes.items():
        node = base
        *path, last = dotted.split(".")
        for key in path:
            node = node[key]
        node[last] = value
    return base


class Compare(unittest.TestCase):
    def three(self, **changes_for_mac):
        return {
            "ubuntu-latest (Linux X64)": runner_result("ubuntu-latest (Linux X64)"),
            "macos-latest (macOS ARM64)": runner_result("macos-latest (macOS ARM64)", **changes_for_mac),
            "ubuntu-24.04-arm (Linux ARM64)": runner_result("ubuntu-24.04-arm (Linux ARM64)"),
        }

    def test_identical_runners_agree(self):
        runners, rows, differences, ok = ds.compare(self.three())
        self.assertTrue(ok)
        self.assertEqual(differences, [])
        self.assertEqual(len(runners), 3)
        self.assertTrue(all(same for _name, _values, same in rows))
        self.assertEqual([r[0] for r in rows][:3], ["vocabulary file", "all shards", "first batches"])

    def test_each_kind_of_difference_is_caught(self):
        digest = "0" * 64
        for field in ("digests.vocabulary", "digests.combined", "shards.batch_digest"):
            self.assertFalse(ds.compare(self.three(**{field: digest}))[3], field)
        for field, value in (("shards.documents", 1), ("shards.tokens", 1), ("shards.duplicates", 9),
                             ("shards.secrets", 9), ("shards.sequences", 1), ("train.merges", 1)):
            self.assertFalse(ds.compare(self.three(**{field: value}))[3], field)

    def test_a_single_shard_that_differs_or_is_missing_is_named(self):
        results = self.three()
        mac = results["macos-latest (macOS ARM64)"]
        mac["digests"]["shards"][1] = "f" * 64
        runners, rows, differences, ok = ds.compare(results)
        self.assertFalse(ok)
        self.assertEqual([d[0] for d in differences], ["shard 1"])
        # A shard that one runner did not make is a difference too.
        mac["digests"]["shards"] = mac["digests"]["shards"][:3]
        self.assertEqual([d[0] for d in ds.compare(results)[2]], ["shard 1", "shard 3"])

    def test_missing_values_are_not_agreement(self):
        results = self.three()
        for r in results.values():
            r["digests"]["vocabulary"] = None
        self.assertFalse(ds.compare(results)[3])

    def test_one_result_is_not_enough(self):
        one = {"only": runner_result("only")}
        self.assertFalse(ds.compare(one)[3])
        self.assertFalse(ds.compare({})[3])

    def test_the_table_and_verdict(self):
        results = self.three(**{"shards.batch_digest": "1" * 64})
        runners, rows, differences, ok = ds.compare(results)
        md = ds.compare_markdown(runners, rows, differences, ok, len(results))
        self.assertIn("| first batches |", md)
        self.assertIn("\u274c", md)
        self.assertIn("disagree", md)
        good = self.three()
        runners, rows, differences, ok = ds.compare(good)
        self.assertIn("byte-for-byte the same", ds.compare_markdown(runners, rows, differences, ok, 3))
        self.assertIn("at least two", ds.compare_markdown([], [], [], False, 1))

    def write_tree(self, tmp, results):
        for n, (runner, data) in enumerate(results.items()):
            folder = os.path.join(tmp, f"data-smoke-7-{n}")
            os.makedirs(folder)
            put(os.path.join(folder, ds.RESULT_NAME), json.dumps(data))

    def test_loading_a_tree_of_artifacts_and_the_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_tree(tmp, self.three())
            self.assertEqual(len(ds.load_results(tmp)), 3)
            md = os.path.join(tmp, "out.md")
            done = subprocess.run([sys.executable, SCRIPT, "--compare", tmp, "--md", md], capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr + done.stdout)
            self.assertIn("byte-for-byte the same", read(md))

    def test_the_exit_code_is_1_when_they_differ_or_nothing_is_there(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_tree(tmp, self.three(**{"digests.combined": "2" * 64}))
            done = subprocess.run([sys.executable, SCRIPT, "--compare", tmp], capture_output=True, text=True)
            self.assertEqual(done.returncode, 1)
        with tempfile.TemporaryDirectory() as empty:
            done = subprocess.run([sys.executable, SCRIPT, "--compare", empty], capture_output=True, text=True)
            self.assertEqual(done.returncode, 1)
            self.assertIn("at least two", done.stdout)


if __name__ == "__main__":
    unittest.main()
