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
from unittest import mock

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

    def test_spaced_slash_spelling_means_or_too(self):
        # crates.io gives `Apache-2.0 / MIT` for fnv (a real value, see the fixture below).
        self.assertEqual(ingest.normalize_license("Apache-2.0 / MIT"), "Apache-2.0 OR MIT")
        self.assertEqual(ingest.normalize_license("MIT / Apache-2.0 / Unlicense"),
                         "MIT OR Apache-2.0 OR Unlicense")
        self.assertEqual(ingest.normalize_license(" MIT /Apache-2.0 "), "MIT OR Apache-2.0")
        # Empty parts are dropped, not joined.
        self.assertEqual(ingest.normalize_license("MIT//Apache-2.0"), "MIT OR Apache-2.0")
        self.assertEqual(ingest.normalize_license("MIT /"), "MIT")
        self.assertEqual(ingest.normalize_license("/MIT/ /Apache-2.0"), "MIT OR Apache-2.0")
        # A slash next to an operator or a bracket is not the old spelling; it is left as it is.
        self.assertEqual(ingest.normalize_license("MIT OR Apache-2.0 / BSD-3-Clause"),
                         "MIT OR Apache-2.0 / BSD-3-Clause")
        self.assertEqual(ingest.normalize_license("(MIT/Apache-2.0)"), "(MIT/Apache-2.0)")

    def test_real_crates_io_version_with_a_spaced_slash(self):
        doc = json.loads(fixture("ingest-crate-version-spaced-slash.json"))
        self.assertEqual(ingest.parse_crate_version(doc), ("Apache-2.0 OR MIT", False))

    def test_which_texts_have_the_shape_of_an_spdx_expression(self):
        for text in (
            "MIT", "MIT OR Apache-2.0", "(MIT OR Apache-2.0) AND Unicode-3.0",
            "Apache-2.0 WITH LLVM-exception", "Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT",
            "GPL-2.0-or-later", "LGPL-2.1+", "LicenseRef-in-house", "DocumentRef-x:LicenseRef-y",
            "NONE", "NOASSERTION", "((MIT))", "MIT AND (Apache-2.0 OR BSD-3-Clause)",
            "  MIT  OR   Apache-2.0 ", "BSD-2-Clause-Patent", "(MIT)",
        ):
            self.assertTrue(ingest.is_spdx_shaped(text), text)
        for text in (
            "", "   ", "/", "MIT /", "MIT / Apache-2.0", "MIT Apache-2.0", "MIT OR", "OR MIT",
            "MIT AND AND Apache-2.0", "(MIT", "MIT)", "()", "MIT WITH", "MIT WITH OR",
            "MIT WITH (x)", "http://example.com/license", "MIT, Apache-2.0", "mit or apache-2.0",
            "MIT OR (", "MIT (Apache-2.0)", "WITH MIT", "MIT WITH A WITH B", "see LICENSE file",
            "MIT OR Apache-2.0 OR", "(MIT OR Apache-2.0", "MIT OR Apache-2.0)", "MIT\u00a0OR\u2014Apache",
            # an operator or a bracket where an id is wanted
            "AND", "OR", "WITH", ")", "(", "MIT AND OR", "MIT OR WITH", "MIT OR )", "MIT AND ) OR MIT",
            "MIT WITH )", "MIT WITH (", "MIT WITH AND", "( ) MIT",
        ):
            self.assertFalse(ingest.is_spdx_shaped(text), repr(text))

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

    def test_real_reply_has_a_repository_with_no_license_as_none(self):
        # In the captured reply the license of papers-we-love is null: GitHub found none.
        doc = json.loads(fixture("ingest-github-search.json"))
        out = {c["id"]: c["license"] for c in ingest.parse_github_search(doc, "shell")}
        self.assertEqual(out["papers-we-love/papers-we-love"], "NONE")

    # The cases below are written by hand: a captured reply does not have a fork or a name
    # that the manifest would refuse.
    def test_forks_odd_names_none_and_noassertion(self):
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
                {"full_name": "e/f", "fork": False, "license": {"spdx_id": None}},
                {"full_name": "g/h", "fork": False, "license": {}},
                {"fork": False},
            ]
        }
        out = ingest.parse_github_search(doc, "shell")
        self.assertEqual([c["id"] for c in out], ["a/b", "c/d", "e/f", "g/h"])
        # No license at all is NONE. A license GitHub could not identify, or an object
        # without an id, is NOASSERTION: a license file exists, so it is not "none".
        self.assertEqual(
            [c["license"] for c in out], ["NONE", "NOASSERTION", "NOASSERTION", "NOASSERTION"]
        )

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

    def test_fetch_can_drop_the_git_directory(self):
        with tempfile.TemporaryDirectory() as root:
            repo = self.make_repo(root, "one", "pub fn one() {}\n")
            revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                                      capture_output=True, text=True).stdout.strip()
            record = {"id": "t/one", "url": "file://" + repo, "revision": revision, "license": "MIT",
                      "fetched": "2026-10-27", "category": "rust"}
            manifest = os.path.join(root, "m.txt")
            accepted = os.path.join(root, "a.txt")
            with open(manifest, "w", encoding="utf-8") as handle:
                handle.write(ingest.manifest_text([record]))
            with open(accepted, "w", encoding="utf-8") as handle:
                handle.write("t/one\n")
            kept = os.path.join(root, "kept")
            slim = os.path.join(root, "slim")
            for dest, extra in ((kept, []), (slim, ["--drop-git"])):
                code, _out, err = self.run_main(["fetch", "--manifest", manifest, "--accepted", accepted,
                                                 "--dest", dest] + extra)
                self.assertEqual(code, 0, err)
            self.assertTrue(os.path.isdir(os.path.join(kept, "t__one", ".git")))
            self.assertFalse(os.path.exists(os.path.join(slim, "t__one", ".git")))
            with open(os.path.join(slim, "t__one", "lib.rs"), encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "pub fn one() {}\n")
            with open(os.path.join(slim, "t__one", ".midman-revision"), encoding="utf-8") as handle:
                self.assertEqual(handle.read().strip(), revision)
            # A source fetched slim is recognised by its marker and not fetched again.
            code, _out, err = self.run_main(["fetch", "--manifest", manifest, "--accepted", accepted,
                                             "--dest", slim, "--drop-git"])
            self.assertEqual(code, 0, err)
            self.assertIn("have t/one", err)
            self.assertFalse(os.path.exists(os.path.join(slim, "t__one", ".git")))

    def test_a_license_is_made_fit_for_the_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            texts = {"t/slash": "Apache-2.0 / MIT", "t/prose": "see the LICENSE file",
                     "t/empty": "", "t/lower": "mit or apache-2.0", "t/good": "MIT OR Apache-2.0"}
            candidates = []
            for ident, text in texts.items():
                path = self.make_repo(root, ident.split("/")[1], "pub fn " + ident.split("/")[1] + "() {}\n")
                candidates.append({"id": ident, "url": "file://" + path, "category": "rust", "license": text})
            cands = os.path.join(root, "c.json")
            self.write_candidates(cands, candidates)
            manifest = os.path.join(root, "m.txt")
            code, _out, err = self.run_main(
                ["manifest", "--candidates", cands, "--out", manifest, "--date", "2026-10-27"]
            )
            self.assertEqual(code, 0, err)
            with open(manifest, encoding="utf-8") as handle:
                got = {r["id"]: r["license"] for r in ingest.read_manifest(handle.read())}
            self.assertEqual(got, {
                "t/slash": "Apache-2.0 OR MIT", "t/prose": "NOASSERTION", "t/empty": "NOASSERTION",
                "t/lower": "NOASSERTION", "t/good": "MIT OR Apache-2.0",
            })
            self.assertIn("the license of t/prose is `see the LICENSE file`, which is not an SPDX"
                          " expression; recorded as NOASSERTION", err)
            self.assertIn("the license of t/lower is `mit or apache-2.0`", err)
            self.assertNotIn("t/slash is", err)
            self.assertNotIn("t/empty is", err)
            self.assertNotIn("t/good is", err)

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


class MeasureHelperTests(unittest.TestCase):
    def test_extension_keys(self):
        self.assertEqual(ingest.extension_of("lib.rs"), ".rs")
        self.assertEqual(ingest.extension_of("LIB.RS"), ".rs")
        self.assertEqual(ingest.extension_of("Makefile"), "Makefile")
        self.assertEqual(ingest.extension_of("Dockerfile"), "Dockerfile")
        self.assertEqual(ingest.extension_of("LICENSE"), "(none)")
        self.assertEqual(ingest.extension_of(".gitignore"), "(none)")
        self.assertEqual(ingest.extension_of("a.tar.gz"), ".gz")

    def test_file_stats(self):
        with tempfile.TemporaryDirectory() as root:
            def put(name, data):
                path = os.path.join(root, name)
                with open(path, "wb") as handle:
                    handle.write(data)
                return path

            self.assertEqual(ingest.file_stats(put("a", b"hello\n")), (6, True, False))
            self.assertEqual(ingest.file_stats(put("b", b"\xff\xfe\n")), (3, False, False))
            self.assertEqual(ingest.file_stats(put("c", b"")), (0, True, False))
            at = b"x" * ingest.LONG_LINE
            self.assertEqual(ingest.file_stats(put("d", at + b"\nshort\n")), (len(at) + 7, True, False))
            over = b"x" * (ingest.LONG_LINE + 1)
            self.assertEqual(ingest.file_stats(put("e", b"short\n" + over))[2], True)
            self.assertIsNone(ingest.file_stats(os.path.join(root, "missing")))

    def test_extension_table_skips_what_selection_skips(self):
        with tempfile.TemporaryDirectory() as root:
            def put(rel, data=b"12345"):
                path = os.path.join(root, rel)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as handle:
                    handle.write(data)

            put("src/a.rs")
            put("src/b.RS", b"123")
            put("docs/x.md", b"1")
            put("Makefile", b"12")
            put(".midman-revision")
            put(".git/config")
            put("target/x.rs")
            put("node_modules/y.js")
            os.symlink(os.path.join(root, "src/a.rs"), os.path.join(root, "link.rs"))
            table = ingest.extension_table(root)
            self.assertEqual(table, {".rs": [2, 8], ".md": [1, 1], "Makefile": [1, 2]})

    def test_measure_source_counts_what_list_would_take(self):
        with tempfile.TemporaryDirectory() as root:
            def put(rel, data):
                with open(os.path.join(root, rel), "wb") as handle:
                    handle.write(data)

            put("a.rs", b"fn a() {}\n")
            put("c.rs", b"fn c() {}\n")
            put("b.rs", b"fn b() {}\n" + b"x" * 2000 + b"\n")
            put("bad.rs", b"\xff\xff")
            put("empty.rs", b"")
            put("readme.md", b"text")
            got = ingest.measure_source(root, "rust", 1_000_000)
            self.assertEqual(
                got,
                {"files": 3, "bytes": 10 + 10 + 10 + 2001, "not_utf8": 1, "long_line_files": 1,
                 "docs_files": 1, "docs_bytes": 4, "docs_not_utf8": 0},
            )
            self.assertEqual(
                ingest.measure_source(root, "shell", 1_000_000),
                {"files": 0, "bytes": 0, "not_utf8": 0, "long_line_files": 0,
                 "docs_files": 1, "docs_bytes": 4, "docs_not_utf8": 0},
            )


class MeasureSummaryTests(unittest.TestCase):
    def source(self, ident, category, files, size, bad=0, long_lines=0, docs=0, docs_size=0, docs_bad=0):
        return {"id": ident, "category": category, "revision": "a" * 40, "files": files,
                "bytes": size, "not_utf8": bad, "long_line_files": long_lines,
                "docs_files": docs, "docs_bytes": docs_size, "docs_not_utf8": docs_bad}

    def test_categories_add_up_and_empty_sources_are_counted(self):
        sources = [
            self.source("a/x", "rust", 3, 300, 1, 1, 4, 4000, 1),
            self.source("a/y", "rust", 0, 0, docs=2, docs_size=500),
            self.source("b/z", "shell", 2, 50, 0, 2, 1, 70),
        ]
        out = ingest.summarize_measurements(sources, {".rs": [4, 400], ".md": [1, 100]})
        self.assertEqual(out["schema"], ingest.RESULTS_SCHEMA)
        self.assertEqual(
            out["categories"]["rust"],
            {"sources": 2, "sources_without_files": 1, "files": 3, "bytes": 300,
             "not_utf8": 1, "long_line_files": 1, "docs_files": 6, "docs_bytes": 4500,
             "docs_not_utf8": 1},
        )
        self.assertEqual(out["categories"]["shell"]["docs_files"], 1)
        self.assertEqual(out["categories"]["shell"]["docs_bytes"], 70)
        self.assertEqual(out["categories"]["shell"]["sources_without_files"], 0)
        self.assertEqual(out["categories"]["shell"]["long_line_files"], 2)
        self.assertEqual(out["all_files"], 5)
        self.assertEqual(out["all_bytes"], 500)
        self.assertEqual(out["sources"], sources)

    def test_a_source_without_docs_keys_counts_as_none(self):
        old = {"id": "a/x", "category": "rust", "files": 1, "bytes": 5, "not_utf8": 0,
               "long_line_files": 0}
        out = ingest.summarize_measurements([old], {})
        self.assertEqual(out["categories"]["rust"]["docs_files"], 0)
        self.assertEqual(out["categories"]["rust"]["docs_bytes"], 0)

    def test_extensions_rank_by_bytes_then_name_and_are_cut(self):
        # Inserted in reverse, so only the tie-break puts `.a` first.
        table = {"." + chr(97 + i): [1, 10] for i in reversed(range(20))}
        table[".big"] = [1, 99]
        out = ingest.summarize_measurements([], table)
        names = [e["extension"] for e in out["extensions"]]
        self.assertEqual(len(names), ingest.TOP_EXTENSIONS)
        self.assertEqual(names[:3], [".big", ".a", ".b"])
        self.assertEqual(out["all_files"], 21)

    def test_real_results_are_self_consistent(self):
        doc = json.loads(fixture("ingest-results.json"))
        again = ingest.summarize_measurements(doc["sources"], {})
        self.assertEqual(again["categories"], doc["categories"])
        self.assertEqual(doc["categories"]["rust"]["sources"], 37)
        self.assertEqual(doc["categories"]["shell"]["sources"], 28)
        self.assertEqual(len(doc["sources"]), 65)
        self.assertEqual(sum(s["files"] for s in doc["sources"]), 3151 + 3593)
        self.assertEqual(sum(s["docs_files"] for s in doc["sources"]), 221 + 5180)
        self.assertEqual(doc["schema"], ingest.RESULTS_SCHEMA)

    def test_markdown_from_real_results(self):
        doc = json.loads(fixture("ingest-results.json"))
        text = ingest.results_markdown(doc, "a note")
        self.assertTrue(text.startswith("## midman-ai ingest"))
        self.assertIn("\na note\n", text)
        self.assertIn("| rust | 37 | 0 | 3151 | 29.9 | 0 | 1 | 221 | 0.6 |", text)
        self.assertIn("| shell | 28 | 0 | 3593 | 16.0 | 0 | 1 | 5180 | 37.7 |", text)
        self.assertIn("### Where the docs are (largest 8)", text)
        self.assertIn("| `omacom/omarchy` | shell | 2600 | 15.5 | 6.0 |", text)
        self.assertIn("| `rust-lang/regex` | rust | 227 | 5.6 |", text)
        self.assertIn("| `omacom/omarchy` | shell | 1357 | 6.0 |", text)
        self.assertIn("| `.mp4` | 19 | 86.2 | 16.2% |", text)
        self.assertTrue(text.endswith("All files fetched: 19612, 533.0 MB.\n\n"))
        self.assertNotIn("Sources with no file of their category", text)
        self.assertEqual(text.count("\n| `"), 15 + 8 + ingest.TOP_EXTENSIONS)

    def test_markdown_lists_sources_without_files_and_omits_an_empty_note(self):
        sources = [self.source("a/x", "rust", 0, 0), self.source("a/y", "rust", 1, 2_500_000)]
        doc = ingest.summarize_measurements(sources, {".rs": [1, 2_500_000]})
        text = ingest.results_markdown(doc)
        self.assertIn("### Sources with no file of their category\n\n- `a/x` (rust)\n", text)
        self.assertIn("| rust | 2 | 1 | 1 | 2.5 | 0 | 0 | 0 | 0.0 |", text)
        self.assertNotIn("Where the docs are", text)
        self.assertTrue(text.startswith("## midman-ai ingest: what was fetched and what would be used\n\n| Category"))
        self.assertEqual(ingest.megabytes(0), "0.0")
        self.assertEqual(ingest.megabytes(1_049_999), "1.0")
        self.assertEqual(ingest.megabytes(1_050_000), "1.1")


class PolicyReportTests(unittest.TestCase):
    def test_real_report_is_read(self):
        report = ingest.parse_policy_report(fixture("ingest-policy-report.tsv"))
        self.assertEqual(len(report["accepted"]), 66)
        self.assertEqual(len(report["rejected"]), 11)
        first = report["rejected"][0]
        self.assertEqual(first["id"], "tw93/Mole")
        self.assertEqual(first["category"], "shell")
        self.assertEqual(first["license"], "GPL-3.0")
        self.assertEqual(first["reason"], "`GPL-3.0` is not on the allow list")
        unidentified = [s for s in report["rejected"] if s["license"] == "NOASSERTION"]
        self.assertEqual(len(unidentified), 5)
        self.assertEqual(
            unidentified[0]["reason"], "the license could not be identified (`NOASSERTION`)"
        )
        unlicensed = [s["id"] for s in report["accepted"] if s["license"] == "NONE"]
        self.assertEqual(
            sorted(unlicensed),
            [
                "eip-work/kuboard-press",
                "kholia/OSX-KVM",
                "papers-we-love/papers-we-love",
                "powerline/fonts",
                "youngyangyang04/leetcode-master",
            ],
        )
        rust = [s for s in report["accepted"] if s["category"] == "rust"]
        self.assertEqual(len(rust), 37)
        self.assertEqual(report["accepted"][0]["id"], "rust-lang/hashbrown")

    def test_other_lines_are_ignored(self):
        text = "accepted\tonly/three\tfields\nrejected\ta\tb\tc\nnoise\n\naccepted\ta/b\trust\tMIT\n"
        report = ingest.parse_policy_report(text)
        self.assertEqual(report["accepted"], [{"id": "a/b", "category": "rust", "license": "MIT"}])
        self.assertEqual(report["rejected"], [])

    def test_markdown_from_the_real_report(self):
        report = ingest.parse_policy_report(fixture("ingest-policy-report.tsv"))
        text = ingest.policy_markdown(report, "a note")
        self.assertTrue(text.startswith("## midman-ai ingest: license policy\n\na note\n\n| Category"))
        self.assertIn("| rust | 37 | 0 |", text)
        self.assertIn("| shell | 29 | 11 |", text)
        self.assertIn("| all | 66 | 11 |", text)
        self.assertIn("| `tw93/Mole` | shell | GPL-3.0 | `GPL-3.0` is not on the allow list |", text)
        self.assertIn("\n5 of the accepted sources have no license.", text)
        self.assertEqual(text.count("\n| `"), 11)
        # A blank line ends it, so whatever is appended next starts a new block.
        self.assertTrue(text.endswith("|\n\n"))

    def test_no_refusals_means_no_table_and_pipes_are_escaped(self):
        report = {"accepted": [{"id": "a/b", "category": "rust", "license": "MIT"}], "rejected": []}
        text = ingest.policy_markdown(report)
        self.assertTrue(text.startswith("## midman-ai ingest: license policy\n\n| Category"))
        self.assertNotIn("### Refused", text)
        self.assertNotIn("no license", text)
        self.assertIn("| all | 1 | 0 |", text)
        # The count of sources with no license does not depend on the case of NONE.
        report["accepted"].append({"id": "e/f", "category": "rust", "license": "none"})
        self.assertIn("\n1 of the accepted sources have no license.", ingest.policy_markdown(report))
        report["rejected"].append({"id": "c/d", "category": "rust", "license": "A | B", "reason": "x|y"})
        text = ingest.policy_markdown(report)
        self.assertIn("| `c/d` | rust | A \\| B | x\\|y |", text)

    def test_report_command_appends(self):
        with tempfile.TemporaryDirectory() as root:
            report = os.path.join(root, "r.tsv")
            summary = os.path.join(root, "s.md")
            with open(report, "w", encoding="utf-8") as handle:
                handle.write(fixture("ingest-policy-report.tsv"))
            with open(summary, "w", encoding="utf-8") as handle:
                handle.write("before\n")
            err = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(err):
                code = ingest.main(["report", "--report", report, "--markdown", summary, "--note", "n"])
            self.assertEqual(code, 0)
            self.assertIn("report: 66 accepted, 11 rejected", err.getvalue())
            with open(summary, encoding="utf-8") as handle:
                text = handle.read()
            self.assertTrue(text.startswith("before\n## midman-ai ingest: license policy"))


def write_files(root, files):
    """Makes files under `root`: {relative path: text or bytes}."""
    for rel, data in files.items():
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(data if isinstance(data, bytes) else data.encode("utf-8"))


class DocsKindTests(unittest.TestCase):
    def names(self, root, category, kind, limit=1000):
        return [os.path.relpath(p, root) for p in ingest.select_files(root, category, limit, kind)]

    def test_which_files_are_docs(self):
        with tempfile.TemporaryDirectory() as root:
            write_files(root, {
                "README.md": "x", "docs/guide.md": "x", "docs/more.markdown": "x",
                "docs/index.rst": "x", "docs/UPPER.MD": "x",
                "CHANGELOG.md": "x", "changes.md": "x", "History.md": "x", "NEWS.md": "x",
                "RELEASE-NOTES.md": "x", "RELEASES.md": "x",
                "README": "x", "notes.txt": "x", "main.rs": "x",
                "vendor/dep/README.md": "x", "target/doc/x.md": "x", "node_modules/p/README.md": "x",
                "empty.md": "", "with space.md": "x", "big.md": "y" * 1001, "edge.md": "y" * 1000,
            })
            os.symlink(os.path.join(root, "README.md"), os.path.join(root, "link.md"))
            expected = ["README.md", "docs/UPPER.MD", "docs/guide.md", "docs/index.rst",
                        "docs/more.markdown", "edge.md"]
            self.assertEqual(self.names(root, "rust", "docs"), expected)
            # The category does not matter for docs, not even one that has no code rule.
            self.assertEqual(self.names(root, "shell", "docs"), expected)
            self.assertEqual(self.names(root, "anything", "docs"), expected)
            self.assertIn("big.md", self.names(root, "rust", "docs", 1001))

    def test_code_does_not_include_docs_and_a_bad_kind_is_an_error(self):
        with tempfile.TemporaryDirectory() as root:
            write_files(root, {"a.rs": "x", "README.md": "x"})
            self.assertEqual(self.names(root, "rust", "code"), ["a.rs"])
            self.assertEqual(self.names(root, "rust", "docs"), ["README.md"])
            with self.assertRaises(ingest.IngestError):
                ingest.select_files(root, "rust", 10, "everything")
            with self.assertRaises(ingest.IngestError):
                ingest.select_files(root, "cobol", 10, "code")


class ShebangTests(unittest.TestCase):
    def test_which_first_lines_name_a_shell(self):
        with tempfile.TemporaryDirectory() as root:
            for i, line in enumerate((
                "#!/bin/sh", "#!/bin/bash", "#!/usr/bin/env bash", "#!/usr/bin/env -S bash -e",
                "#! /bin/sh", "#!/bin/zsh -f", "#!/usr/bin/fish", "#!/usr/bin/env dash",
                "#!/bin/ksh", "#!/usr/local/bin/bash",
            )):
                write_files(root, {"ok%d" % i: line + "\necho hi\n"})
                self.assertTrue(ingest.has_shell_shebang(os.path.join(root, "ok%d" % i)), line)
            for i, line in enumerate((
                "#!/usr/bin/env python3", "#!/usr/bin/perl", "#!/bin/bashful", "#!/bin/shell",
                "# !/bin/sh", " #!/bin/sh", "echo hi", "",
            )):
                write_files(root, {"no%d" % i: line + "\n"})
                self.assertFalse(ingest.has_shell_shebang(os.path.join(root, "no%d" % i)), line)
            self.assertFalse(ingest.has_shell_shebang(os.path.join(root, "missing")))

    def test_extensionless_scripts_are_selected_for_shell_only(self):
        with tempfile.TemporaryDirectory() as root:
            write_files(root, {
                "bin/tool": "#!/usr/bin/env bash\necho hi\n",
                "bin/pytool": "#!/usr/bin/env python3\nprint(1)\n",
                "bin/plain": "just text\n",
                "bin/data.txt": "#!/bin/sh\n",
                "run.sh": "echo x\n",
                ".zshrc": "alias x=y\n",
                ".profile": "#!/bin/sh\n",
                "vendor/tool2": "#!/bin/sh\n",
                "bin/empty": "",
            })
            names = [os.path.relpath(p, root) for p in ingest.select_files(root, "shell", 1000)]
            self.assertEqual(names, [".profile", "bin/tool", "run.sh"])
            # Rust takes no extensionless files, whatever they start with.
            self.assertEqual(ingest.select_files(root, "rust", 1000), [])


class FilesCommandTests(unittest.TestCase):
    def run_main(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = ingest.main(argv)
        return code, out.getvalue(), err.getvalue()

    def make(self, root):
        records = []
        for ident, category in (("t/code", "rust"), ("t/scripts", "shell"), ("t/last", "rust")):
            records.append({"id": ident, "url": "https://github.com/" + ident, "revision": "a" * 40,
                            "license": "MIT", "fetched": "2026-10-25", "category": category})
        manifest = os.path.join(root, "m.txt")
        with open(manifest, "w", encoding="utf-8") as handle:
            handle.write(ingest.manifest_text(records))
        accepted = os.path.join(root, "a.txt")
        with open(accepted, "w", encoding="utf-8") as handle:
            handle.write("t/scripts\nt/code\nt/last\n")
        dest = os.path.join(root, "src")
        write_files(dest, {
            "t__code/a.rs": "fn a() {}\n", "t__code/bad.rs": b"\xff\xfe", "t__code/README.md": "# hi\n",
            "t__code/CHANGELOG.md": "x\n",
            "t__scripts/install": "#!/bin/sh\necho\n", "t__scripts/x.sh": "echo\n",
            "t__scripts/notes.md": "text\n", "t__scripts/bad.md": b"\xff",
            "t__last/b.rs": "fn b() {}\n",
        })
        return manifest, accepted, dest

    def test_files_in_accepted_order_code_before_docs(self):
        with tempfile.TemporaryDirectory() as root:
            manifest, accepted, dest = self.make(root)
            out = os.path.join(root, "f.tsv")
            code, _o, err = self.run_main(["files", "--manifest", manifest, "--accepted", accepted,
                                           "--dest", dest, "--out", out])
            self.assertEqual(code, 0, err)
            with open(out, encoding="utf-8") as handle:
                rows = [line.split("\t") for line in handle.read().splitlines()]
            rel = [(r[0], r[1], r[2], os.path.relpath(r[3], dest)) for r in rows]
            self.assertEqual(rel, [
                ("shell", "code", "t/scripts", "t__scripts/install"),
                ("shell", "code", "t/scripts", "t__scripts/x.sh"),
                ("shell", "docs", "t/scripts", "t__scripts/notes.md"),
                ("rust", "code", "t/code", "t__code/a.rs"),
                ("rust", "docs", "t/code", "t__code/README.md"),
                ("rust", "code", "t/last", "t__last/b.rs"),
            ])
            # The counts are listed sorted by category and then kind.
            self.assertEqual(
                [l for l in err.splitlines() if l.startswith("files:")],
                ["files: rust code: 2", "files: rust docs: 1", "files: shell code: 2",
                 "files: shell docs: 1"],
            )

    def test_one_kind_and_errors(self):
        with tempfile.TemporaryDirectory() as root:
            manifest, accepted, dest = self.make(root)
            out = os.path.join(root, "f.tsv")
            base = ["files", "--manifest", manifest, "--accepted", accepted, "--dest", dest, "--out", out]
            code, _o, err = self.run_main(base + ["--kind", "docs"])
            self.assertEqual(code, 0, err)
            with open(out, encoding="utf-8") as handle:
                kinds = {line.split("\t")[1] for line in handle.read().splitlines()}
            self.assertEqual(kinds, {"docs"})
            code, _o, err = self.run_main(base + ["--kind", "code"])
            with open(out, encoding="utf-8") as handle:
                kinds = {line.split("\t")[1] for line in handle.read().splitlines()}
            self.assertEqual(kinds, {"code"})
            os.rename(os.path.join(dest, "t__last"), os.path.join(dest, "elsewhere"))
            code, _o, err = self.run_main(base)
            self.assertEqual(code, 1)
            self.assertIn("has not been fetched", err)
            with open(accepted, "w", encoding="utf-8") as handle:
                handle.write("t/unknown\n")
            code, _o, err = self.run_main(base)
            self.assertEqual(code, 1)
            self.assertIn("t/unknown", err)

    def test_list_takes_a_kind(self):
        with tempfile.TemporaryDirectory() as root:
            write_files(root, {"a.rs": "x", "README.md": "x"})
            code, out, _e = self.run_main(["list", "--dest", root, "--category", "rust", "--kind", "docs"])
            self.assertEqual(code, 0)
            self.assertEqual([os.path.basename(p) for p in out.split()], ["README.md"])
            code, out, _e = self.run_main(["list", "--dest", root, "--category", "rust"])
            self.assertEqual([os.path.basename(p) for p in out.split()], ["a.rs"])


class CargoLicenseTests(unittest.TestCase):
    # The Cargo.toml files are real, fetched from the default branches on 2026-10-26.
    def test_license_in_the_package_table(self):
        self.assertEqual(ingest.cargo_license(fixture("ingest-cargo-ripgrep.toml")), "Unlicense OR MIT")

    def test_license_in_the_workspace_package_table(self):
        self.assertEqual(ingest.cargo_license(fixture("ingest-cargo-ruff.toml")), "MIT")

    def test_a_workspace_root_without_a_license_gives_none(self):
        self.assertIsNone(ingest.cargo_license(fixture("ingest-cargo-tokio.toml")))

    def test_an_error_page_is_not_toml(self):
        # A real reply of raw.githubusercontent.com when it rate-limited the request.
        self.assertIsNone(ingest.cargo_license(fixture("ingest-cargo-http-429.txt")))
        self.assertIsNone(ingest.cargo_license(""))

    def test_the_other_shapes(self):
        self.assertEqual(
            ingest.cargo_license('[package]\nname = "a"\nlicense = "MIT/Apache-2.0"\n'),
            "MIT OR Apache-2.0",
        )
        # license.workspace = true takes the workspace table.
        self.assertEqual(
            ingest.cargo_license(
                '[package]\nname = "a"\nlicense.workspace = true\n'
                '[workspace.package]\nlicense = "Apache-2.0"\n'
            ),
            "Apache-2.0",
        )
        # The package table wins over the workspace table.
        self.assertEqual(
            ingest.cargo_license('[package]\nlicense = "MIT"\n[workspace.package]\nlicense = "GPL-3.0"\n'),
            "MIT",
        )
        # A license file is not a license id.
        self.assertIsNone(ingest.cargo_license('[package]\nlicense-file = "LICENSE"\n'))
        self.assertIsNone(ingest.cargo_license('[package]\nlicense = 5\n'))
        self.assertIsNone(ingest.cargo_license("[workspace]\nmembers = []\n"))
        self.assertIsNone(ingest.cargo_license("workspace = 3\n"))

    def candidates(self):
        def one(ident, license_text):
            return {"id": ident, "license": license_text, "category": "rust"}

        return [one("a/none", "NONE"), one("b/dual", "NOASSERTION"), one("c/mit", "MIT"),
                one("d/gone", "NOASSERTION"), one("e/nolicense", "NONE"), one("f/gpl", "GPL-3.0")]

    def test_only_the_unidentified_are_looked_up_and_changed(self):
        replies = {
            "a/none": '[package]\nlicense = "MIT"\n',
            "b/dual": '[workspace.package]\nlicense = "MIT OR Apache-2.0"\n',
            "d/gone": None,
            "e/nolicense": '[package]\nname = "x"\n',
        }
        asked = []

        def fake_fetch(url):
            asked.append(url)
            ident = url.split("raw.githubusercontent.com/")[1].split("/HEAD/")[0]
            return replies[ident]

        items = self.candidates()
        err = io.StringIO()
        with mock.patch.object(ingest, "fetch_text", fake_fetch), \
                mock.patch.object(ingest.time, "sleep"), redirect_stderr(err):
            ingest.apply_cargo_licenses(items, "rust")
        by_id = {c["id"]: c for c in items}
        self.assertEqual(by_id["a/none"]["license"], "MIT")
        self.assertEqual(by_id["a/none"]["license_origin"], "Cargo.toml")
        self.assertEqual(by_id["b/dual"]["license"], "MIT OR Apache-2.0")
        self.assertEqual(by_id["c/mit"]["license"], "MIT")
        self.assertNotIn("license_origin", by_id["c/mit"])
        self.assertEqual(by_id["d/gone"]["license"], "NOASSERTION")
        self.assertEqual(by_id["e/nolicense"]["license"], "NONE")
        self.assertEqual(by_id["f/gpl"]["license"], "GPL-3.0")
        self.assertEqual(len(asked), 4)
        self.assertTrue(all(u.endswith("/HEAD/Cargo.toml") for u in asked))
        self.assertIn("2 replaced from Cargo.toml, 1 Cargo.toml files could not be read", err.getvalue())

    def test_other_categories_are_left_alone(self):
        items = self.candidates()
        err = io.StringIO()
        with mock.patch.object(ingest, "fetch_text", side_effect=AssertionError("asked")), \
                redirect_stderr(err):
            ingest.apply_cargo_licenses(items, "shell")
        self.assertEqual(items, self.candidates())
        self.assertIn("only used for the category `rust`", err.getvalue())

    def test_the_github_command_uses_it_when_asked(self):
        doc = {"items": [
            {"full_name": "o/dual", "fork": False, "stargazers_count": 9, "license": {"spdx_id": "NOASSERTION"}},
            {"full_name": "o/mit", "fork": False, "stargazers_count": 8, "license": {"spdx_id": "MIT"}},
        ]}
        with tempfile.TemporaryDirectory() as root:
            for flag, expected in (([], "NOASSERTION"), (["--cargo-license"], "MIT OR Apache-2.0")):
                out = os.path.join(root, "c.json")
                with mock.patch.object(ingest, "get_json", return_value=doc), \
                        mock.patch.object(ingest, "fetch_text", return_value='[package]\nlicense = "MIT OR Apache-2.0"\n'), \
                        mock.patch.object(ingest.time, "sleep"), \
                        redirect_stderr(io.StringIO()):
                    code = ingest.main(["github", "--query", "q", "--category", "rust", "--count", "2",
                                        "--out", out] + flag)
                self.assertEqual(code, 0)
                with open(out, encoding="utf-8") as handle:
                    got = {c["id"]: c["license"] for c in json.load(handle)["candidates"]}
                self.assertEqual(got, {"o/dual": expected, "o/mit": "MIT"})


class NoticeScanTests(unittest.TestCase):
    """The scan that looks, in a source with no license, for a statement against use."""

    def labels(self, text):
        return sorted({label for _line, label, _text in ingest.scan_notice_text(text)})

    # Sentences written by hand to show what each pattern is for, except the first, which is
    # the line of papers-we-love's README that the first real run found.
    def test_real_line_from_a_readme(self):
        line = (
            'The name "Papers We Love" and the logos for the organization are copyrighted, '
            "and under the ownership of Papers We Love Ltd, all rights reserved. When starting"
        )
        hits = ingest.scan_notice_text("# Title\n\n" + line + "\n")
        self.assertEqual([(n, label) for n, label, _ in hits], [(3, "all rights reserved")])

    def test_statements_that_are_caught(self):
        cases = {
            "Copyright (c) 2024 Someone. All Rights Reserved.": "all rights reserved",
            "ALL   RIGHTS\tRESERVED": "all rights reserved",
            "This is proprietary software.": "proprietary",
            "Free for non-commercial use.": "non-commercial",
            "Noncommercial use is fine, anything else is not.": "non-commercial",
            "For personal use only.": "personal or educational use only",
            "This is meant for educational purposes only": "personal or educational use only",
            "Do not use without my permission.": "permission",
            "You may not use this without written permission": "permission",
            "Permission has not been granted to reuse it": "permission",
            "This code may not be copied.": "may not be used or copied",
            "The files must not be redistributed": "may not be used or copied",
            "Please do not copy or steal my work": "may not be used or copied",
            "You are not allowed to reuse any of this": "may not be used or copied",
            "No license is granted to anyone": "no license granted",
            "This repository is not licensed for reuse": "no license granted",
            "No AI training on this repository": "against AI training",
            "Scraping this for machine learning datasets is prohibited": "against AI training",
            "Training of language models on this code is not allowed": "against AI training",
            "do not scrape this repo": "against AI training",
        }
        for sentence, label in cases.items():
            self.assertIn(label, self.labels(sentence), sentence)

    def test_ordinary_text_is_not_caught(self):
        for sentence in (
            "A collection of dotfiles and install scripts.",
            "Permission is hereby granted, free of charge, to any person obtaining a copy",
            "Feel free to copy anything you like.",
            "This project is MIT licensed.",
            "Use at your own risk.",
            "We train a model on the data in this folder.",
            "Pull requests are welcome; ask before you change the public interface.",
            "The commercial edition has more features.",
        ):
            self.assertEqual(self.labels(sentence), [], sentence)

    def test_one_hit_per_line_and_label(self):
        text = "All rights reserved. All rights reserved.\nnothing\nproprietary and all rights reserved"
        hits = ingest.scan_notice_text(text)
        self.assertEqual(
            [(n, label) for n, label, _ in hits],
            [(1, "all rights reserved"), (3, "all rights reserved"), (3, "proprietary")],
        )

    def test_the_text_of_a_hit_is_tidied_and_cut(self):
        long_line = "all   rights\treserved " + "x" * 400
        hit = ingest.scan_notice_text(long_line)[0]
        self.assertTrue(hit[2].startswith("all rights reserved xxxx"))
        self.assertEqual(len(hit[2]), 160)

    def test_which_files_are_read(self):
        with tempfile.TemporaryDirectory() as root:
            def put(name, size=10):
                with open(os.path.join(root, name), "w", encoding="utf-8") as handle:
                    handle.write("x" * size)

            for name in ("README.md", "readme.txt", "LICENSE", "Licence.md", "COPYING", "NOTICE",
                         "TERMS.md", "LEGAL", "DISCLAIMER", "COPYRIGHT", "README"):
                put(name)
            for name in ("main.rs", "CHANGELOG.md", "docs.md", "AUTHORS", "Makefile"):
                put(name)
            put("README-big.md", ingest.MAX_NOTICE_BYTES + 1)
            put("README-limit.md", ingest.MAX_NOTICE_BYTES)
            os.makedirs(os.path.join(root, "README-dir"))
            os.makedirs(os.path.join(root, "sub"))
            with open(os.path.join(root, "sub", "README.md"), "w", encoding="utf-8") as handle:
                handle.write("all rights reserved")
            os.symlink(os.path.join(root, "README.md"), os.path.join(root, "README-link.md"))
            names = [os.path.basename(p) for p in ingest.notice_files(root)]
            self.assertEqual(
                names,
                sorted(["COPYING", "COPYRIGHT", "DISCLAIMER", "LEGAL", "LICENSE", "Licence.md",
                        "NOTICE", "README", "README-limit.md", "README.md", "TERMS.md",
                        "readme.txt"]),
            )
            self.assertEqual(ingest.notice_files(os.path.join(root, "missing")), [])
            # Only the top level is read.
            self.assertEqual(ingest.scan_source(os.path.join(root, "sub")), [
                {"file": "README.md", "line": 1, "label": "all rights reserved",
                 "text": "all rights reserved"}])
            self.assertEqual(ingest.scan_source(root), [])

    def test_scan_source_reads_bad_utf8_and_names_the_file_and_line(self):
        with tempfile.TemporaryDirectory() as root:
            with open(os.path.join(root, "NOTICE"), "wb") as handle:
                handle.write(b"\xff proprietary\nfor personal use only\n")
            with open(os.path.join(root, "README.md"), "w", encoding="utf-8") as handle:
                handle.write("fine\nfine\nThis is proprietary.\n")
            hits = ingest.scan_source(root)
            self.assertEqual(
                [(h["file"], h["line"], h["label"]) for h in hits],
                [
                    ("NOTICE", 1, "proprietary"),
                    ("NOTICE", 2, "personal or educational use only"),
                    ("README.md", 3, "proprietary"),
                ],
            )
            # A byte that is not UTF-8 becomes the replacement character, not nothing.
            self.assertEqual(hits[0]["text"], "\ufffd proprietary")

    def test_markdown(self):
        hit = {"file": "README.md", "line": 4, "label": "proprietary", "text": "a | b"}
        many = [dict(hit, line=n) for n in range(1, 6)]
        text = ingest.notices_markdown(["a/x", "b/y", "c/z"], {"a/x": [hit], "b/y": many}, {"c/z": [hit]}, "n")
        self.assertTrue(text.startswith("## midman-ai ingest: license notices\n\nn\n\n"))
        self.assertIn("scanned: 3. Left out because their own files say otherwise: 2. Kept after review: 1.", text)
        self.assertIn("### Left out", text)
        self.assertIn("### Kept after review", text)
        self.assertIn("| `a/x` | README.md | 4 | proprietary | a \\| b |", text)
        self.assertEqual(text.count("| `b/y` | README.md |"), ingest.MAX_NOTICE_HITS_SHOWN)
        self.assertIn("| `b/y` | | | | and 2 more |", text)
        self.assertTrue(text.endswith("|\n\n"))
        # Exactly as many hits as are shown: no row saying there are more.
        exact = [dict(hit, line=n) for n in range(1, ingest.MAX_NOTICE_HITS_SHOWN + 1)]
        text = ingest.notices_markdown(["a/x"], {"a/x": exact}, {})
        self.assertEqual(text.count("| `a/x` | README.md |"), ingest.MAX_NOTICE_HITS_SHOWN)
        self.assertNotIn("more |", text)

    def test_markdown_when_nothing_was_scanned_or_nothing_was_found(self):
        none = ingest.notices_markdown([], {}, {})
        self.assertIn("No accepted source lacks a license, so nothing was scanned.", none)
        self.assertNotIn("###", none)
        clean = ingest.notices_markdown(["a/x"], {}, {})
        self.assertIn("scanned: 1. Left out because their own files say otherwise: 0. Kept after review: 0.", clean)
        self.assertNotIn("###", clean)


class NoticesCommandTests(unittest.TestCase):
    def make(self, root):
        def source(ident, license_text, readme):
            record = {"id": ident, "url": "https://github.com/" + ident, "revision": "a" * 40,
                      "license": license_text, "fetched": "2026-10-25", "category": "shell"}
            if readme is not None:
                folder = os.path.join(root, "src", ident.replace("/", "__"))
                os.makedirs(folder)
                with open(os.path.join(folder, "README.md"), "w", encoding="utf-8") as handle:
                    handle.write(readme)
            return record

        records = [
            source("t/licensed", "MIT", "All rights reserved."),
            source("t/clean", "NONE", "Some scripts.\n"),
            source("t/closed", "none", "Some scripts.\nAll rights reserved.\nproprietary\n"),
            source("t/reviewed", "NONE", "For personal use only.\n"),
            source("t/last", "Apache-2.0", "hello"),
        ]
        manifest = os.path.join(root, "m.txt")
        with open(manifest, "w", encoding="utf-8") as handle:
            handle.write(ingest.manifest_text(records))
        accepted = os.path.join(root, "a.txt")
        with open(accepted, "w", encoding="utf-8") as handle:
            handle.write("\n".join(r["id"] for r in records) + "\n")
        return manifest, accepted

    def run_main(self, argv):
        err = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(err):
            code = ingest.main(argv)
        return code, err.getvalue()

    def test_the_scan_leaves_out_what_says_otherwise(self):
        with tempfile.TemporaryDirectory() as root:
            manifest, accepted = self.make(root)
            usable = os.path.join(root, "u.txt")
            report = os.path.join(root, "r.tsv")
            summary = os.path.join(root, "s.md")
            reviewed = os.path.join(root, "rev.txt")
            with open(reviewed, "w", encoding="utf-8") as handle:
                handle.write("# reviewed by hand\nt/reviewed\n")
            with open(summary, "w", encoding="utf-8") as handle:
                handle.write("before\n")
            base = ["notices", "--manifest", manifest, "--accepted", accepted,
                    "--dest", os.path.join(root, "src"), "--out", usable]
            code, err = self.run_main(base + ["--report", report, "--markdown", summary,
                                              "--reviewed", reviewed])
            self.assertEqual(code, 0, err)
            self.assertIn("3 with no license scanned, 1 left out, 1 kept after review; 4 of 5 usable", err)
            with open(usable, encoding="utf-8") as handle:
                self.assertEqual(handle.read().split(), ["t/licensed", "t/clean", "t/reviewed", "t/last"])
            with open(report, encoding="utf-8") as handle:
                rows = [line.split("\t") for line in handle.read().splitlines()]
            self.assertEqual(
                [(r[0], r[1], r[2], r[3], r[4]) for r in rows],
                [("left out", "t/closed", "README.md", "2", "all rights reserved"),
                 ("left out", "t/closed", "README.md", "3", "proprietary"),
                 ("kept", "t/reviewed", "README.md", "1", "personal or educational use only")],
            )
            with open(summary, encoding="utf-8") as handle:
                text = handle.read()
            self.assertTrue(text.startswith("before\n## midman-ai ingest: license notices"))
            self.assertIn("| `t/closed` | README.md | 2 | all rights reserved |", text)
            # Without the review list the same source is left out.
            code, err = self.run_main(base)
            self.assertEqual(code, 0, err)
            self.assertIn("2 left out, 0 kept after review; 3 of 5 usable", err)
            with open(usable, encoding="utf-8") as handle:
                self.assertEqual(handle.read().split(), ["t/licensed", "t/clean", "t/last"])

    def test_errors(self):
        with tempfile.TemporaryDirectory() as root:
            manifest, accepted = self.make(root)
            usable = os.path.join(root, "u.txt")
            dest = os.path.join(root, "src")
            base = ["notices", "--manifest", manifest, "--dest", dest, "--out", usable]
            with open(accepted, "w", encoding="utf-8") as handle:
                handle.write("t/unknown\n")
            code, err = self.run_main(base + ["--accepted", accepted])
            self.assertEqual(code, 1)
            self.assertIn("t/unknown", err)
            # An unlicensed source that was not fetched is an error; a licensed one is not read.
            with open(accepted, "w", encoding="utf-8") as handle:
                handle.write("t/clean\n")
            os.rename(os.path.join(dest, "t__clean"), os.path.join(dest, "elsewhere"))
            code, err = self.run_main(base + ["--accepted", accepted])
            self.assertEqual(code, 1)
            self.assertIn("has not been fetched", err)
            with open(accepted, "w", encoding="utf-8") as handle:
                handle.write("t/licensed\n")
            os.rename(os.path.join(dest, "t__licensed"), os.path.join(dest, "elsewhere2"))
            code, err = self.run_main(base + ["--accepted", accepted])
            self.assertEqual(code, 0, err)
            self.assertIn("0 with no license scanned", err)


class MeasureCommandTests(unittest.TestCase):
    def test_measure_after_fetch(self):
        with tempfile.TemporaryDirectory() as root:
            repo = os.path.join(root, "repo")
            os.makedirs(repo)
            git(["init", "-q", "-b", "main"], repo)
            with open(os.path.join(repo, "lib.rs"), "w", encoding="utf-8") as handle:
                handle.write("pub fn f() {}\n")
            with open(os.path.join(repo, "bad.rs"), "wb") as handle:
                handle.write(b"\xff\xfe")
            with open(os.path.join(repo, "README.md"), "w", encoding="utf-8") as handle:
                handle.write("hello\n")
            git(["add", "."], repo)
            git(["commit", "-q", "-m", "x"], repo)
            record = {
                "id": "t/repo", "url": "file://" + repo,
                "revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                                           capture_output=True, text=True).stdout.strip(),
                "license": "MIT", "fetched": "2026-10-25", "category": "rust",
            }
            manifest = os.path.join(root, "m.txt")
            accepted = os.path.join(root, "a.txt")
            dest = os.path.join(root, "src")
            with open(manifest, "w", encoding="utf-8") as handle:
                handle.write(ingest.manifest_text([record]))
            with open(accepted, "w", encoding="utf-8") as handle:
                handle.write("t/repo\n")
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                self.assertEqual(
                    ingest.main(["fetch", "--manifest", manifest, "--accepted", accepted, "--dest", dest]), 0)
            result_path = os.path.join(root, "r.json")
            summary = os.path.join(root, "s.md")
            policy = os.path.join(root, "p.tsv")
            with open(policy, "w", encoding="utf-8") as handle:
                handle.write("accepted\tt/repo\trust\tMIT\n")
            with open(summary, "w", encoding="utf-8") as handle:
                handle.write("earlier text\n")

            def run_measure():
                err = io.StringIO()
                with redirect_stdout(io.StringIO()), redirect_stderr(err):
                    code = ingest.main(["measure", "--manifest", manifest, "--accepted", accepted,
                                        "--dest", dest, "--out", result_path, "--markdown", summary,
                                        "--report", policy, "--note", "hello"])
                return code, err.getvalue()

            code, err = run_measure()
            self.assertEqual(code, 0, err)
            self.assertIn("measure: rust: 1 sources, 1 files", err)
            with open(result_path, encoding="utf-8") as handle:
                doc = json.load(handle)
            self.assertEqual(doc["sources"][0]["id"], "t/repo")
            self.assertEqual(doc["sources"][0]["files"], 1)
            self.assertEqual(doc["sources"][0]["not_utf8"], 1)
            self.assertEqual(doc["sources"][0]["revision"], record["revision"])
            extensions = {e["extension"]: e["files"] for e in doc["extensions"]}
            self.assertEqual(extensions, {".rs": 2, ".md": 1})
            with open(summary, encoding="utf-8") as handle:
                text = handle.read()
            self.assertTrue(text.startswith("earlier text\n## midman-ai ingest: license policy"))
            self.assertIn("## midman-ai ingest: what was fetched", text)
            # The summary file is appended to, as $GITHUB_STEP_SUMMARY has to be.
            run_measure()
            with open(summary, encoding="utf-8") as handle:
                self.assertEqual(handle.read().count("## midman-ai ingest: license policy"), 2)

            with open(accepted, "w", encoding="utf-8") as handle:
                handle.write("t/unknown\n")
            code, err = run_measure()
            self.assertEqual(code, 1)
            self.assertIn("t/unknown", err)
            with open(accepted, "w", encoding="utf-8") as handle:
                handle.write("t/repo\n")
            os.rename(os.path.join(dest, "t__repo"), os.path.join(dest, "elsewhere"))
            code, err = run_measure()
            self.assertEqual(code, 1)
            self.assertIn("has not been fetched", err)


if __name__ == "__main__":
    unittest.main()
