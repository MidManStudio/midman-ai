#!/usr/bin/env python3
# ============================================================================
# NOTICE: Full documentation, design decisions, and fix history for this file
# live in docs/ci-and-workflows.md, section "check_docs.py"
# ============================================================================
"""Checks that the docs and the source stay in step. Standard library only.

Hard failures:
  - a workspace member has no docs/<package-name>.md (or <member>/docs/<package-name>.md)
  - a NOTICE header points at a doc file that does not exist
  - a NOTICE header names a section that has no matching heading in that doc
  - a crate doc is not split into parts and its last "##" heading is not
    "Fixes and Problems"

Not failures (reported only):
  - .rs files with no NOTICE header (mid-engine adds them incrementally too),
    unless --require-headers is passed
  - em dashes in docs (the guidelines say not to use them)

Exit code is 1 if there is at least one hard failure.
"""
import argparse
import glob
import json
import os
import re
import sys

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    sys.exit("check_docs.py needs Python 3.11+ (tomllib)")

RE_NOTICE = re.compile(r'live in (docs/[^\s,]+\.md), section "([^"\n]+)')
RE_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
EM_DASH = "\u2014"
SOURCE_DIRS = ("src", "tests", "benches", "examples")


def norm(text):
    return re.sub(r"[`*_]", "", text).strip().lower()


def headings(path):
    out = []
    in_fence = False
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            m = RE_HEADING.match(line.rstrip("\n"))
            if m:
                out.append((len(m.group(1)), m.group(2)))
    return out


def section_resolves(section, heads):
    want = norm(section)
    token = re.split(r"[:\s]", want)[0]
    for _level, text in heads:
        have = norm(text)
        if have == want or (token and (have == token or have.startswith(token + ":"))):
            return True
    return False


def workspace_members(root):
    with open(os.path.join(root, "Cargo.toml"), "rb") as fh:
        data = tomllib.load(fh)
    members = []
    for pattern in data.get("workspace", {}).get("members", []):
        for hit in sorted(glob.glob(os.path.join(root, pattern))):
            if os.path.isfile(os.path.join(hit, "Cargo.toml")):
                members.append(os.path.relpath(hit, root))
    return members


def package_name(root, member):
    with open(os.path.join(root, member, "Cargo.toml"), "rb") as fh:
        return tomllib.load(fh).get("package", {}).get("name", os.path.basename(member))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--require-headers", action="store_true",
                    help="treat a missing NOTICE header in a .rs file as a failure")
    ap.add_argument("--json", dest="json_path", default="")
    ap.add_argument("--md", default="", help="file to append the markdown summary to")
    args = ap.parse_args()
    root = os.path.abspath(args.root)

    failures, notes = [], []
    checked_headers = 0
    missing_headers = []
    crate_rows = []

    for member in workspace_members(root):
        name = package_name(root, member)
        candidates = [os.path.join(root, "docs", f"{name}.md"),
                      os.path.join(root, member, "docs", f"{name}.md")]
        doc = next((c for c in candidates if os.path.isfile(c)), None)
        if doc is None:
            failures.append({"where": member, "problem": f"no docs/{name}.md"})
            crate_rows.append((name, "missing", 0))
            continue

        split = os.path.isdir(os.path.splitext(doc)[0])
        h2 = [t for lvl, t in headings(doc) if lvl == 2]
        if not split and (not h2 or norm(h2[-1]) != "fixes and problems"):
            failures.append({"where": os.path.relpath(doc, root),
                             "problem": 'last "##" heading is not "Fixes and Problems"'})

        n_files = 0
        for sub in SOURCE_DIRS:
            for path in sorted(glob.glob(os.path.join(root, member, sub, "**", "*.rs"), recursive=True)):
                n_files += 1
                rel = os.path.relpath(path, root)
                with open(path, encoding="utf-8", errors="replace") as fh:
                    head = "".join(fh.readline() for _ in range(6))
                m = RE_NOTICE.search(head)
                if not m:
                    missing_headers.append(rel)
                    continue
                checked_headers += 1
                ref, section = m.group(1), m.group(2)
                ref_path = os.path.join(root, ref)
                if not os.path.isfile(ref_path):
                    failures.append({"where": rel, "problem": f"NOTICE points at missing file {ref}"})
                elif not section_resolves(section, headings(ref_path)):
                    failures.append({"where": rel,
                                     "problem": f'{ref} has no heading for section "{section}"'})
        crate_rows.append((name, os.path.relpath(doc, root), n_files))

    docs_dir = os.path.join(root, "docs")
    em_dash_files = []
    if os.path.isdir(docs_dir):
        for path in sorted(glob.glob(os.path.join(docs_dir, "**", "*.md"), recursive=True)):
            with open(path, encoding="utf-8", errors="replace") as fh:
                if EM_DASH in fh.read():
                    em_dash_files.append(os.path.relpath(path, root))

    if missing_headers and args.require_headers:
        for rel in missing_headers:
            failures.append({"where": rel, "problem": "no NOTICE header (--require-headers)"})
    if missing_headers:
        notes.append(f"{len(missing_headers)} .rs file(s) have no NOTICE header")
    if em_dash_files:
        notes.append(f"em dash found in {len(em_dash_files)} doc(s): " + ", ".join(em_dash_files[:8]))

    result = {"ok": not failures, "crates": len(crate_rows), "headers_checked": checked_headers,
              "headers_missing": missing_headers, "failures": failures, "notes": notes}
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)

    out = ["## midman-ai docs conformance", "",
           f"Crates: {len(crate_rows)} &nbsp;|&nbsp; NOTICE headers resolved: {checked_headers} "
           f"&nbsp;|&nbsp; Failures: {len(failures)}", ""]
    if failures:
        out += ["| Where | Problem |", "|---|---|"]
        out += [f"| `{f['where']}` | {f['problem']} |" for f in failures[:80]]
        if len(failures) > 80:
            out.append(f"| ... | {len(failures) - 80} more |")
        out.append("")
    else:
        out += ["✅ Every crate has a doc, every NOTICE header resolves.", ""]
    for n in notes:
        out.append(f"- Note: {n}")
    if notes:
        out.append("")
    text = "\n".join(out) + "\n"
    if args.md:
        with open(args.md, "a", encoding="utf-8") as fh:
            fh.write(text)
    else:
        sys.stdout.write(text)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
