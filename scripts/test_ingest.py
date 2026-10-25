#!/usr/bin/env python3
# ============================================================================
# NOTICE: Full documentation, design decisions, and fix history for this file
# live in docs/ci-and-workflows.md, section "test_ingest.py"
# ============================================================================
"""Self-tests for ingest.py. Standard library only; needs the `git` program.

Run with `python3 -m unittest discover -s scripts -p 'test_*.py'`.
"""
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ingest  # noqa: E402

FIXTURES = os.path.join(HERE, "fixtures")


def fixture(name):
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as handle:
        return handle.read()


def git(args, cwd):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"] + args,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


class GithubRepoTests(unittest.TestCase):
    def test_spellings_of_one_repository(self):
        for url in (
            "https://github.com/dtolnay/syn",
            "https://github.com/dtolnay/syn/",
            "https://github.com/dtolnay/syn.git",
            "http://github.com/dtolnay/syn",
            "https://www.github.com/dtolnay/syn",
            "git://github.com/dtolnay/syn.git",
            "https://github.com/dtolnay/syn/tree/master/src",
            "https://github.com/dtolnay/syn#readme",
            "  https://github.com/dtolnay/syn  ",
        ):
            self.assertEqual(ingest.github_repo(url), ("dtolnay", "syn"), url)

    def test_names_with_dots_dashes_and_case_are_kept(self):
        self.assertEqual(
            ingest.github_repo("https://github.com/Foo-Bar/baz.rs.git"), ("Foo-Bar", "baz.rs")
        )

    def test_other_hosts_and_junk_are_not_github(self):
        for url in (
            None,
            "",
            "https://gitlab.com/a/b",
            "https://example.com/github.com/a/b",
            "https://github.com/onlyowner",
            "https://github.com/",
            "not a url",
        ):
            self.assertIsNone(ingest.github_repo(url), url)


class LicenseTests(unittest.TestCase):
    def test_slash_spelling_means_or(self):
        self.assertEqual(ingest.normalize_license("MIT/Apache-2.0"), "MIT OR Apache-2.0")
        self.assertEqual(
            ingest.normalize_license("MIT/Apache-2.0/Unlicense"), "MIT OR Apache-2.0 OR Unlicense"
        )

    def test_spdx_expressions_are_left_alone(self):
        for text in (
            "MIT",
            "MIT OR Apache-2.0",
            "Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT",
            "(MIT OR Apache-2.0) AND Unicode-3.0",
        ):
            self.assertEqual(ingest.normalize_license(text), text)

    def test_blank_is_none_and_spaces_are_collapsed(self):
        self.assertIsNone(ingest.normalize_license(None))
        self.assertIsNone(ingest.normalize_license(""))
        self.assertIsNone(ingest.normalize_license("   "))
        self.assertEqual(ingest.normalize_license(" MIT   OR  Apache-2.0 "), "MIT OR Apache-2.0")

    def test_a_non_spdx_name_is_not_repaired(self):
        self.assertEqual(ingest.normalize_license("Other"), "Other")
        self.assertEqual(ingest.normalize_license("mit"), "mit")

    def test_merge_one_or_equal_licenses(self):
        self.assertEqual(ingest.merge_licenses(["MIT"]), "MIT")
        # One license that is itself an expression is not wrapped in brackets.
        self.assertEqual(ingest.merge_licenses(["MIT OR Apache-2.0"]), "MIT OR Apache-2.0")
        self.assertEqual(
            ingest.merge_licenses(["(MIT OR Apache-2.0) AND Unicode-3.0", None]),
            "(MIT OR Apache-2.0) AND Unicode-3.0",
        )
        self.assertEqual(ingest.merge_licenses(["MIT", "MIT", None, "MIT"]), "MIT")
        self.assertIsNone(ingest.merge_licenses([]))
        self.assertIsNone(ingest.merge_licenses([None, None]))

    def test_merge_different_licenses_needs_all_of_them(self):
        self.assertEqual(ingest.merge_licenses(["MIT", "Apache-2.0"]), "MIT AND Apache-2.0")
        self.assertEqual(
            ingest.merge_licenses(["MIT OR Apache-2.0", "MIT"]), "(MIT OR Apache-2.0) AND MIT"
        )
        self.assertEqual(
            ingest.merge_licenses(["A", "B AND C", "D OR E"]), "A AND (B AND C) AND (D OR E)"
        )


class CratesTests(unittest.TestCase):
    def setUp(self):
        self.page = json.loads(fixture("ingest-crates-page.json"))

    def test_real_page_is_read(self):
        crates = ingest.parse_crates_page(self.page)
        self.assertEqual(
            [c["name"] for c in crates],
            ["hashbrown", "syn", "getrandom", "bitflags", "rand_core", "rand"],
        )
        first = crates[0]
        self.assertEqual(first["repository"], "https://github.com/rust-lang/hashbrown")
        self.assertEqual(first["downloads"], 2838902678)
        self.assertEqual(first["version"], "0.17.1")
        self.assertTrue(all(isinstance(c["downloads"], int) for c in crates))

    def test_version_falls_back_to_newest_and_bad_entries_are_skipped(self):
        page = {
            "crates": [
                {"name": "a", "newest_version": "1.0.0", "downloads": 5},
                {"name": "b", "max_stable_version": "2.0.0", "newest_version": "3.0.0"},
                {"id": "c"},
                {"name": ""},
                {"name": 7},
                {},
            ]
        }
        crates = ingest.parse_crates_page(page)
        self.assertEqual([c["name"] for c in crates], ["a", "b", "c"])
        self.assertEqual([c["version"] for c in crates], ["1.0.0", "2.0.0", None])
        self.assertEqual(crates[0]["downloads"], 5)
        self.assertEqual(crates[1]["downloads"], 0)
        self.assertEqual(ingest.parse_crates_page({}), [])

    def test_real_version_document(self):
        doc = json.loads(fixture("ingest-crate-version.json"))
        license_text, yanked = ingest.parse_crate_version(doc)
        self.assertEqual(license_text, "(MIT OR Apache-2.0) AND Unicode-3.0")
        self.assertFalse(yanked)

    def test_version_document_edge_cases(self):
        self.assertEqual(
            ingest.parse_crate_version({"version": {"license": "MIT/Apache-2.0", "yanked": True}}),
            ("MIT OR Apache-2.0", True),
        )
        self.assertEqual(ingest.parse_crate_version({"version": {"license": None}}), (None, False))
        self.assertEqual(ingest.parse_crate_version({}), (None, False))

    def test_candidates_group_by_repository_and_keep_a_renamed_one_apart(self):
        crates = ingest.parse_crates_page(self.page)
        licenses = {c["name"]: "MIT OR Apache-2.0" for c in crates}
        out = ingest.build_crate_candidates(crates, licenses)
        ids = [c["id"] for c in out]
        # rand_core's listing still points at the old repository name: same project, two ids.
        self.assertIn("rust-lang-nursery/rand", ids)
        self.assertIn("rust-random/rand", ids)
        self.assertEqual(len(out), 6)
        self.assertEqual(out[0]["url"], "https://github.com/rust-lang/hashbrown")
        self.assertEqual(out[0]["category"], "rust")
        self.assertEqual(out[0]["origin"], "crates.io")

    def test_crates_of_one_repository_are_merged_with_all_licenses(self):
        crates = [
            {"name": "x", "repository": "https://github.com/o/r", "downloads": 10, "version": "1"},
            {"name": "y", "repository": "https://github.com/O/R.git", "downloads": 5, "version": "1"},
            {"name": "z", "repository": "https://gitlab.com/o/q", "downloads": 99, "version": "1"},
        ]
        out = ingest.build_crate_candidates(crates, {"x": "MIT", "y": "GPL-3.0-only", "z": "MIT"})
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["id"], "o/r")
        self.assertEqual(out[0]["crates"], ["x", "y"])
        self.assertEqual(out[0]["downloads"], 15)
        self.assertEqual(out[0]["license"], "MIT AND GPL-3.0-only")

    def test_a_crate_without_a_license_makes_the_repository_unasserted(self):
        crates = [{"name": "x", "repository": "https://github.com/o/r", "downloads": 1, "version": "1"}]
        out = ingest.build_crate_candidates(crates, {"x": None})
        self.assertEqual(out[0]["license"], "NOASSERTION")


class GithubSearchTests(unittest.TestCase):
    def test_real_reply_is_read(self):
        doc = json.loads(fixture("ingest-github-search.json"))
        out = ingest.parse_github_search(doc, "shell")
        self.assertEqual(
            [c["id"] for c in out],
            [
                "obra/superpowers",
                "mattpocock/skills",
                "ohmyzsh/ohmyzsh",
                "msitarzewski/agency-agents",
                "d3/d3",
                "papers-we-love/papers-we-love",
            ],
        )
        by_id = {c["id"]: c for c in out}
        self.assertEqual(by_id["ohmyzsh/ohmyzsh"]["license"], "MIT")
        self.assertEqual(by_id["d3/d3"]["license"], "ISC")
        self.assertEqual(by_id["ohmyzsh/ohmyzsh"]["stars"], 190155)
        self.assertEqual(by_id["ohmyzsh/ohmyzsh"]["category"], "shell")
        self.assertEqual(by_id["ohmyzsh/ohmyzsh"]["url"], "https://github.com/ohmyzsh/ohmyzsh")
        self.assertEqual(by_id["ohmyzsh/ohmyzsh"]["origin"], "github-search")

    def test_real_reply_has_a_repository_with_no_license_as_noassertion(self):
        # In the captured reply the license of papers-we-love is null.
        doc = json.loads(fixture("ingest-github-search.json"))
        out = {c["id"]: c["license"] for c in ingest.parse_github_search(doc, "shell")}
        self.assertEqual(out["papers-we-love/papers-we-love"], "NOASSERTION")

    # The cases below are written by hand: a captured reply does not have a fork or a name
    # that the manifest would refuse.
    def test_forks_odd_names_and_noassertion(self):
        doc = {
            "items": [
                {"full_name": "a/b", "fork": False, "stargazers_count": 3, "license": None},
                {"full_name": "someone/fork", "fork": True, "stargazers_count": 9, "license": None},
                {
                    "full_name": "bad name/x y",
                    "fork": False,
                    "stargazers_count": 1,
                    "license": {"spdx_id": "MIT"},
                },
                {
                    "full_name": "c/d",
                    "fork": False,
                    "stargazers_count": 2,
                    "license": {"spdx_id": "NOASSERTION"},
                },
                {"fork": False},
            ]
        }
        out = ingest.parse_github_search(doc, "shell")
        self.assertEqual([c["id"] for c in out], ["a/b", "c/d"])
        self.assertEqual([c["license"] for c in out], ["NOASSERTION", "NOASSERTION"])

    def test_empty_result(self):
        self.assertEqual(ingest.parse_github_search({}, "shell"), [])


class LsRemoteTests(unittest.TestCase):
    def test_real_output(self):
        self.assertEqual(
            ingest.parse_ls_remote(fixture("ingest-ls-remote.txt")),
            "14432a28e180ec717933714aed7828fc7ed2fd29",
        )

    def test_other_shapes_give_none(self):
        good = "14432a28e180ec717933714aed7828fc7ed2fd29"
        self.assertIsNone(ingest.parse_ls_remote(""))
        self.assertIsNone(ingest.parse_ls_remote(good + "\trefs/heads/main\n"))
        self.assertIsNone(ingest.parse_ls_remote(good[:39] + "\tHEAD\n"))
        self.assertIsNone(ingest.parse_ls_remote(good.upper() + "\tHEAD\n"))
        self.assertIsNone(ingest.parse_ls_remote("fatal: repository not found\n"))

    def test_a_64_digit_commit_is_accepted(self):
        long_id = "ab" * 32
        self.assertEqual(ingest.parse_ls_remote(long_id + "\tHEAD\n"), long_id)

    def test_head_line_is_found_after_other_lines(self):
        good = "14432a28e180ec717933714aed7828fc7ed2fd29"
        self.assertEqual(ingest.parse_ls_remote("warning: x\n" + good + "\tHEAD\n"), good)


class ManifestTextTests(unittest.TestCase):
    def test_real_manifest_round_trips_byte_for_byte(self):
        text = fixture("ingest-manifest.txt")
        records = ingest.read_manifest(text)
        self.assertEqual([r["id"] for r in records][:2], ["rust-lang/hashbrown", "dtolnay/syn"])
        self.assertEqual(len(records), 4)
        self.assertEqual(ingest.manifest_text(records), text)

    def test_empty_manifest(self):
        self.assertEqual(ingest.manifest_text([]), "midman-data-manifest 1\n")
        self.assertEqual(ingest.read_manifest("midman-data-manifest 1\n"), [])

    def test_the_reader_refuses_a_bad_frame(self):
        good = fixture("ingest-manifest.txt")
        cases = {
            "wrong header": good.replace("midman-data-manifest 1", "midman-data-manifest 2", 1),
            "no final newline": good[:-1],
            "no blank line": good.replace("\n\nid:", "\nid:", 1),
            "wrong key": good.replace("license: ", "licence: ", 1),
            "key order": good.replace("url: ", "xurl: ", 1),
            "short record": good.rsplit("\n", 2)[0] + "\n",
        }
        for name, text in cases.items():
            with self.assertRaises(ingest.IngestError, msg=name):
                ingest.read_manifest(text)

    def test_drop_same_commit_keeps_the_first(self):
        def src(ident, revision):
            return {"id": ident, "revision": revision}

        kept, dropped = ingest.drop_same_commit(
            [src("a/x", "1"), src("b/y", "2"), src("c/x", "1"), src("d/z", "1")]
        )
        self.assertEqual([s["id"] for s in kept], ["a/x", "b/y"])
        self.assertEqual(dropped, [("c/x", "a/x"), ("d/z", "a/x")])
        self.assertEqual(ingest.drop_same_commit([]), ([], []))


class SelectFilesTests(unittest.TestCase):
    def test_selection_rules(self):
        with tempfile.TemporaryDirectory() as root:
            def put(rel, text="fn main() {}\n"):
                path = os.path.join(root, rel)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(text)

            put("a/src/lib.rs")
            put("a/src/main.rs")
            put("a/README.md")
            put("a/target/debug/gen.rs")
            put("a/.git/hooks/x.rs")
            put("a/vendor/dep/lib.rs")
            put("a/node_modules/x/y.rs")
            put("a/src/empty.rs", "")
            put("a/src/with space.rs")
            put("a/src/big.rs", "x" * 2000)
            os.symlink(os.path.join(root, "a/src/lib.rs"), os.path.join(root, "a/src/link.rs"))
            put("a/src/exact.rs", "x" * 1000)
            found = ingest.select_files(root, "rust", 1000)
            names = [os.path.relpath(p, root) for p in found]
            # exact.rs is exactly the limit and is taken; big.rs is one byte over a limit
            # of 1999 and is refused at 1000, and taken at 2000.
            self.assertEqual(names, ["a/src/exact.rs", "a/src/lib.rs", "a/src/main.rs"])
            self.assertEqual(found, sorted(found))
            at_limit = [os.path.relpath(p, root) for p in ingest.select_files(root, "rust", 2000)]
            self.assertIn("a/src/big.rs", at_limit)
            below = [os.path.relpath(p, root) for p in ingest.select_files(root, "rust", 1999)]
            self.assertNotIn("a/src/big.rs", below)

    def test_shell_takes_suffixes_and_whole_names(self):
        with tempfile.TemporaryDirectory() as root:
            for rel in ("install.sh", "Makefile", "Dockerfile", "lib/x.bash", "notes.txt", "makefile.bak"):
                path = os.path.join(root, rel)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write("echo hi\n")
            names = sorted(os.path.relpath(p, root) for p in ingest.select_files(root, "shell", 1000))
            self.assertEqual(names, ["Dockerfile", "Makefile", "install.sh", "lib/x.bash"])

    def test_unknown_category_is_an_error(self):
        with self.assertRaises(ingest.IngestError):
            ingest.select_files(".", "cobol", 10)


class CommandTests(unittest.TestCase):
    """The commands that use git, run against repositories made on the spot."""

    def make_repo(self, root, name, text):
        path = os.path.join(root, name)
        os.makedirs(path)
        git(["init", "-q", "-b", "main"], path)
        with open(os.path.join(path, "lib.rs"), "w", encoding="utf-8") as handle:
            handle.write(text)
        git(["add", "."], path)
        git(["commit", "-q", "-m", "first"], path)
        return path

    def write_candidates(self, path, entries):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"schema": ingest.CANDIDATES_SCHEMA, "candidates": entries}, handle)

    def run_main(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = ingest.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_manifest_then_fetch(self):
        with tempfile.TemporaryDirectory() as root:
            one = self.make_repo(root, "one", "pub fn one() {}\n")
            two = self.make_repo(root, "two", "pub fn two() {}\n")
            twin = os.path.join(root, "twin")
            subprocess.run(["git", "clone", "-q", one, twin], check=True, capture_output=True)
            gone = os.path.join(root, "missing")
            entry = lambda ident, path, lic="MIT": {  # noqa: E731
                "id": ident,
                "url": "file://" + path,
                "category": "rust",
                "license": lic,
            }
            cands = os.path.join(root, "c.json")
            self.write_candidates(
                cands,
                [
                    entry("t/one", one),
                    entry("t/two", two, "MIT OR Apache-2.0"),
                    entry("t/twin", twin),
                    entry("t/gone", gone),
                    entry("T/ONE", one),
                    entry("t/skipme", two),
                    entry("Mixed/Case", two),
                    entry("bad id", two),
                ],
            )
            manifest = os.path.join(root, "m.txt")
            code, _out, err = self.run_main(
                [
                    "manifest", "--candidates", cands, "--out", manifest,
                    "--date", "2026-10-25", "--exclude", "T/SKIPME", "--exclude", "mixed/case",
                ]
            )
            self.assertEqual(code, 0, err)
            with open(manifest, encoding="utf-8") as handle:
                records = ingest.read_manifest(handle.read())
            self.assertEqual([r["id"] for r in records], ["t/one", "t/two"])
            self.assertEqual(records[0]["fetched"], "2026-10-25")
            self.assertEqual(records[1]["license"], "MIT OR Apache-2.0")
            self.assertRegex(records[0]["revision"], r"^[0-9a-f]{40}$")
            for why in ("same commit as t/one", "excluded by --exclude", "id not allowed"):
                self.assertIn(why, err)
            self.assertIn("t/gone", err)
            self.assertIn("Mixed/Case: excluded by --exclude", err)
            # A repeated id (here only the case differs) is dropped without a word.
            self.assertNotIn("T/ONE", err)

            accepted = os.path.join(root, "a.txt")
            with open(accepted, "w", encoding="utf-8") as handle:
                handle.write("t/one\nt/two\n")
            dest = os.path.join(root, "src")
            code, _out, err = self.run_main(
                ["fetch", "--manifest", manifest, "--accepted", accepted, "--dest", dest]
            )
            self.assertEqual(code, 0, err)
            with open(os.path.join(dest, "t__one", "lib.rs"), encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "pub fn one() {}\n")
            with open(os.path.join(dest, "t__two", ".midman-revision"), encoding="utf-8") as handle:
                self.assertEqual(handle.read().strip(), records[1]["revision"])

            # A second fetch finds both in place and does not fetch again.
            code, _out, err = self.run_main(
                ["fetch", "--manifest", manifest, "--accepted", accepted, "--dest", dest]
            )
            self.assertEqual(code, 0, err)
            self.assertEqual(err.count("have "), 2)

            code, out, _err = self.run_main(["list", "--dest", dest, "--category", "rust"])
            self.assertEqual(code, 0)
            self.assertEqual(
                [os.path.relpath(p, dest) for p in out.splitlines()],
                ["t__one/lib.rs", "t__two/lib.rs"],
            )

            # An id that is not in the manifest is refused before anything is fetched.
            with open(accepted, "w", encoding="utf-8") as handle:
                handle.write("t/unknown\n")
            code, _out, err = self.run_main(
                ["fetch", "--manifest", manifest, "--accepted", accepted, "--dest", dest]
            )
            self.assertEqual(code, 1)
            self.assertIn("t/unknown", err)

    def test_fetch_reports_a_source_that_cannot_be_fetched(self):
        with tempfile.TemporaryDirectory() as root:
            record = {
                "id": "t/none",
                "url": "file://" + os.path.join(root, "nothing"),
                "revision": "a" * 40,
                "license": "MIT",
                "fetched": "2026-10-25",
                "category": "rust",
            }
            manifest = os.path.join(root, "m.txt")
            with open(manifest, "w", encoding="utf-8") as handle:
                handle.write(ingest.manifest_text([record]))
            accepted = os.path.join(root, "a.txt")
            with open(accepted, "w", encoding="utf-8") as handle:
                handle.write("t/none\n")
            code, _out, err = self.run_main(
                ["fetch", "--manifest", manifest, "--accepted", accepted, "--dest", os.path.join(root, "d")]
            )
            self.assertEqual(code, 1)
            self.assertIn("FAILED t/none", err)
            self.assertIn("1 of 1 sources failed", err)

    def test_manifest_refuses_a_file_that_is_not_a_candidate_list(self):
        with tempfile.TemporaryDirectory() as root:
            bad = os.path.join(root, "bad.json")
            with open(bad, "w", encoding="utf-8") as handle:
                json.dump({"schema": "other/1", "candidates": []}, handle)
            code, _out, err = self.run_main(
                ["manifest", "--candidates", bad, "--out", os.path.join(root, "m.txt")]
            )
            self.assertEqual(code, 1)
            self.assertIn(ingest.CANDIDATES_SCHEMA, err)


if __name__ == "__main__":
    unittest.main()
