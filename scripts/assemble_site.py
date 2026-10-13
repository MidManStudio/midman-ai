#!/usr/bin/env python3
# ============================================================================
# NOTICE: Full documentation, design decisions, and fix history for this file
# live in docs/ci-and-workflows.md, section "assemble_site.py"
# ============================================================================
"""Assembles the deployable site directory from its parts. Standard library only.

  web/home/*            -> out/                 (the landing page)
  web/shared/           -> out/shared/
  web/lab/              -> out/lab/             (without tests)
  web/tests/            -> out/tests/           (without tests)
  web/benchmarks/       -> out/benchmarks/      (without tests)
  <book>                -> out/docs/            (the built mdBook)
  <wasm>/*.wasm         -> out/lab/
  <data>/*.json         -> out/data/            (optional: CI, benchmark and smoke results)
  web/_headers          -> out/_headers

It fails, naming what is missing, if the engine or the book is absent, because a
site published without them would look fine and not work. Missing data files are
not an error: the pages say that no results are published yet.
"""
import argparse
import shutil
import sys
from pathlib import Path

REQUIRED_WASM = ("midman_api.wasm", "midman_api.simd.wasm")


def copy_tree(src: Path, dst: Path) -> None:
    """Copies a directory, leaving out test directories and test files."""
    def skip(directory, names):
        return [n for n in names if n == "test" or n.endswith(".test.mjs") or n == "__pycache__"]
    shutil.copytree(src, dst, ignore=skip, dirs_exist_ok=True)


def assemble(web: Path, book: Path, wasm: Path, data: Path, out: Path) -> list[str]:
    """Builds `out` and returns the list of problems (empty on success)."""
    problems = []
    for name in REQUIRED_WASM:
        if not (wasm / name).is_file():
            problems.append(f"missing engine build: {wasm / name}")
    if not (book / "index.html").is_file():
        problems.append(f"missing built docs: {book / 'index.html'}")
    for part in ("home", "shared", "lab", "tests", "benchmarks"):
        if not (web / part).is_dir():
            problems.append(f"missing web/{part}")
    if problems:
        return problems

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    copy_tree(web / "home", out)
    copy_tree(web / "shared", out / "shared")
    for part in ("lab", "tests", "benchmarks"):
        copy_tree(web / part, out / part)
    shutil.copytree(book, out / "docs")
    for name in REQUIRED_WASM:
        shutil.copy2(wasm / name, out / "lab" / name)
    if data.is_dir():
        (out / "data").mkdir()
        for f in sorted(data.glob("*.json")):
            shutil.copy2(f, out / "data" / f.name)
    headers = web / "_headers"
    if headers.is_file():
        shutil.copy2(headers, out / "_headers")
    return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--web", default="web")
    ap.add_argument("--book", default="web/site/book")
    ap.add_argument("--wasm", default="build")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="dist")
    args = ap.parse_args()
    problems = assemble(Path(args.web), Path(args.book), Path(args.wasm), Path(args.data), Path(args.out))
    for p in problems:
        print(f"error: {p}", file=sys.stderr)
    if problems:
        return 1
    files = sum(1 for f in Path(args.out).rglob("*") if f.is_file())
    print(f"assembled {files} files into {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
