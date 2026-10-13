#!/usr/bin/env python3
# ============================================================================
# NOTICE: Full documentation, design decisions, and fix history for this file
# live in docs/ci-and-workflows.md, section "test_assemble_site.py"
# ============================================================================
"""Self-tests for assemble_site.py. Run with:

    python3 -m unittest discover -s scripts -p 'test_*.py'
"""
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import assemble_site as a  # noqa: E402


def touch(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class Assemble(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.web, self.book, self.wasm, self.data, self.out = t / "web", t / "book", t / "build", t / "data", t / "dist"
        touch(self.web / "home" / "index.html", "home")
        touch(self.web / "shared" / "theme.css")
        touch(self.web / "lab" / "index.html", "lab")
        touch(self.web / "lab" / "engine.js")
        touch(self.web / "lab" / "test" / "engine.test.mjs")
        touch(self.web / "lab" / "test" / "support.mjs")
        touch(self.web / "tests" / "index.html")
        touch(self.web / "tests" / "test" / "fixtures" / "ci-results.json")
        touch(self.web / "benchmarks" / "index.html")
        touch(self.web / "benchmarks" / "test" / "render.test.mjs")
        touch(self.web / "_headers", "/*\n  X-Test: 1\n")
        touch(self.book / "index.html", "book")
        touch(self.wasm / "midman_api.wasm", "plain")
        touch(self.wasm / "midman_api.simd.wasm", "simd")
        touch(self.data / "ci-results.json", "{}")

    def tearDown(self):
        self.tmp.cleanup()

    def run_assemble(self):
        return a.assemble(self.web, self.book, self.wasm, self.data, self.out)

    def files(self):
        return sorted(str(p.relative_to(self.out)) for p in self.out.rglob("*") if p.is_file())

    def test_a_complete_site_lands_in_the_right_places(self):
        self.assertEqual(self.run_assemble(), [])
        self.assertEqual(
            self.files(),
            [
                "_headers",
                "benchmarks/index.html",
                "data/ci-results.json",
                "docs/index.html",
                "index.html",
                "lab/engine.js",
                "lab/index.html",
                "lab/midman_api.simd.wasm",
                "lab/midman_api.wasm",
                "shared/theme.css",
                "tests/index.html",
            ],
        )
        self.assertEqual((self.out / "index.html").read_text(), "home")
        self.assertEqual((self.out / "lab" / "midman_api.simd.wasm").read_text(), "simd")

    def test_tests_and_their_fixtures_are_not_published(self):
        self.run_assemble()
        self.assertFalse(any("test" in f.split("/")[-1].replace("tests", "") and f.endswith(".mjs") for f in self.files()))
        self.assertFalse((self.out / "lab" / "test").exists())
        self.assertFalse((self.out / "tests" / "test").exists())
        self.assertFalse((self.out / "benchmarks" / "test").exists())

    def test_a_missing_engine_is_an_error_naming_the_file(self):
        (self.wasm / "midman_api.simd.wasm").unlink()
        problems = self.run_assemble()
        self.assertEqual(len(problems), 1)
        self.assertIn("midman_api.simd.wasm", problems[0])
        self.assertFalse(self.out.exists(), "nothing is assembled when something is missing")

    def test_missing_docs_and_missing_web_parts_are_errors(self):
        (self.book / "index.html").unlink()
        (self.web / "lab" / "index.html").unlink()
        self.assertEqual(len(self.run_assemble()), 1, "only the book is reported; lab/ still exists as a directory")
        import shutil
        shutil.rmtree(self.web / "benchmarks")
        problems = self.run_assemble()
        self.assertTrue(any("web/benchmarks" in p for p in problems))
        self.assertTrue(any("built docs" in p for p in problems))

    def test_missing_data_is_fine(self):
        import shutil
        shutil.rmtree(self.data)
        self.assertEqual(self.run_assemble(), [])
        self.assertFalse((self.out / "data").exists())

    def test_an_old_output_is_replaced_not_merged(self):
        touch(self.out / "stale.html")
        self.assertEqual(self.run_assemble(), [])
        self.assertFalse((self.out / "stale.html").exists())

    def test_the_command_line_exits_nonzero_on_a_problem(self):
        import subprocess
        (self.wasm / "midman_api.wasm").unlink()
        r = subprocess.run(
            [sys.executable, str(HERE / "assemble_site.py"), "--web", str(self.web), "--book", str(self.book),
             "--wasm", str(self.wasm), "--data", str(self.data), "--out", str(self.out)],
            capture_output=True, text=True)
        self.assertEqual(r.returncode, 1)
        self.assertIn("midman_api.wasm", r.stderr)


if __name__ == "__main__":
    unittest.main()
