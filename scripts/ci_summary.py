#!/usr/bin/env python3
# ============================================================================
# NOTICE: Full documentation, design decisions, and fix history for this file
# live in docs/ci-and-workflows.md, section "ci_summary.py"
# ============================================================================
"""Turns the raw fmt / clippy / test logs from ci.yml into a JSON file and a
markdown job summary. Standard library only.

This script never fails the job. The gate step in ci.yml decides pass/fail
from the step outcomes; this only reports.
"""
import argparse
import json
import os
import re
import sys

RE_ANSI = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

# GitHub caps a job summary at 1024 KiB. Stay well under it.
MAX_SUMMARY_BYTES = 700_000
MAX_LINES_PER_ITEM = 12


def rel(path):
    """Show repo-relative paths in the summary instead of runner-absolute ones."""
    cwd = os.getcwd().rstrip(os.sep) + os.sep
    return path[len(cwd):] if path.startswith(cwd) else path


def read_log(path):
    if not path:
        return ""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return RE_ANSI.sub("", fh.read())
    except FileNotFoundError:
        return ""


# ---------------------------------------------------------------------------
# fmt
# ---------------------------------------------------------------------------
# rustfmt prints "Diff in <path>:<line>:" (newer) or "Diff in <path> at line <n>:" (older).
RE_FMT_DIFF = re.compile(r"^Diff in (.+?)(?: at line |:)(\d+):?\s*$")


def parse_fmt(text):
    files = {}
    errors = []
    for line in text.splitlines():
        m = RE_FMT_DIFF.match(line)
        if m:
            key = rel(m.group(1))
            files[key] = files.get(key, 0) + 1
        elif line.startswith("error"):
            errors.append(line.strip())
    return {
        "files": [{"path": p, "hunks": n} for p, n in sorted(files.items())],
        "errors": errors[:MAX_LINES_PER_ITEM],
    }


# ---------------------------------------------------------------------------
# clippy / rustc diagnostics
# ---------------------------------------------------------------------------
RE_DIAG = re.compile(r"^(warning|error)(?:\[(\w+)\])?: (.*)$")
RE_LOC = re.compile(r"^\s*--> (.+?):(\d+):(\d+)\s*$")
RE_LINT_NOTE = re.compile(r"`-D ([A-Za-z0-9_:-]+)` implied by")
RE_LINT = re.compile(r"(clippy::[A-Za-z0-9_-]+)")
# Lines cargo/rustc emit that are summaries of diagnostics, not diagnostics.
RE_DIAG_SUMMARY = re.compile(
    r"^(?:`[^`]+` \(|could not compile|aborting due to|build failed|"
    r"\d+ warnings? emitted|\d+ previous errors?)"
)


def parse_diagnostics(text):
    diags = []
    cur = None
    for line in text.splitlines():
        m = RE_DIAG.match(line)
        if m:
            level, code, msg = m.groups()
            if RE_DIAG_SUMMARY.match(msg):
                cur = None
                continue
            cur = {"level": level, "code": code or "", "message": msg.strip(),
                   "path": "", "line": 0, "lint": ""}
            diags.append(cur)
            continue
        if cur is None:
            continue
        loc = RE_LOC.match(line)
        if loc and not cur["path"]:
            cur["path"], cur["line"] = loc.group(1), int(loc.group(2))
            continue
        if not cur["lint"]:
            lint = RE_LINT_NOTE.search(line) or RE_LINT.search(line)
            if lint:
                cur["lint"] = lint.group(1).replace("-", "_")
    # rustc only prints the "implied by -D warnings" note on the first hit of a lint,
    # so later diagnostics with the same message inherit that lint name.
    known = {d["message"]: d["lint"] for d in diags if d["lint"]}
    for d in diags:
        if not d["lint"] and not d["code"]:
            d["lint"] = known.get(d["message"], "")
    return diags


def parse_clippy(text):
    diags = parse_diagnostics(text)
    return {
        "errors": sum(1 for d in diags if d["level"] == "error"),
        "warnings": sum(1 for d in diags if d["level"] == "warning"),
        "diagnostics": diags,
    }


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
RE_RUN = re.compile(r"Running\s+.*?\(\S*?/deps/([A-Za-z0-9_]+)-[0-9a-f]{8,}\)\s*$")
RE_DOC = re.compile(r"^\s*Doc-tests\s+(\S+)\s*$")
RE_TEST = re.compile(r"^test\s+(.+?)\s+\.\.\.\s+(ok|FAILED|ignored)(?:,.*)?\s*$")
RE_RESULT = re.compile(
    r"test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored;"
    r".*?finished in ([\d.]+)s"
)
RE_FAIL_BLOCK = re.compile(r"^---- (.+?) stdout ----$")


def parse_tests(text):
    suites = []
    cur = None
    failure_output = {}
    block_name = None
    block_lines = []

    def close_block():
        nonlocal block_name, block_lines
        if block_name is not None:
            kept = []
            for l in block_lines:
                if l.strip().startswith("stack backtrace:") or l.startswith("note: run with `RUST_BACKTRACE"):
                    break
                kept.append(l)
            failure_output[block_name] = "\n".join(kept).strip()
        block_name, block_lines = None, []

    for line in text.splitlines():
        m = RE_FAIL_BLOCK.match(line)
        if m:
            close_block()
            block_name = m.group(1)
            continue
        if block_name is not None:
            if line.startswith("failures:") or line.startswith("test result:"):
                close_block()
            else:
                block_lines.append(line)
                continue

        m = RE_RUN.search(line)
        if m:
            cur = {"name": m.group(1).replace("_", "-"), "passed": 0, "failed": 0,
                   "ignored": 0, "duration_s": 0.0, "failed_tests": []}
            suites.append(cur)
            continue
        m = RE_DOC.match(line)
        if m:
            cur = {"name": f"{m.group(1).replace('_', '-')} (doc-tests)", "passed": 0,
                   "failed": 0, "ignored": 0, "duration_s": 0.0, "failed_tests": []}
            suites.append(cur)
            continue
        m = RE_TEST.match(line)
        if m and cur is not None and m.group(2) == "FAILED":
            cur["failed_tests"].append(m.group(1))
            continue
        m = RE_RESULT.search(line)
        if m and cur is not None:
            cur["passed"], cur["failed"], cur["ignored"] = (int(m.group(i)) for i in (1, 2, 3))
            cur["duration_s"] = float(m.group(4))
    close_block()

    for s in suites:
        s["failure_output"] = {t: failure_output.get(t, "") for t in s["failed_tests"]}

    build_errors = []
    if not any(s["passed"] + s["failed"] + s["ignored"] for s in suites):
        build_errors = [d for d in parse_diagnostics(text) if d["level"] == "error"]

    return {
        "suites": suites,
        "passed": sum(s["passed"] for s in suites),
        "failed": sum(s["failed"] for s in suites),
        "ignored": sum(s["ignored"] for s in suites),
        "duration_s": round(sum(s["duration_s"] for s in suites), 3),
        "build_errors": build_errors[:MAX_LINES_PER_ITEM],
    }


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
def mark(outcome):
    return {"success": "✅ pass", "failure": "❌ FAIL", "skipped": "⏭️ skipped",
            "cancelled": "⚠️ cancelled"}.get(outcome, f"⚠️ {outcome or 'not run'}")


def first_lines(text, n=MAX_LINES_PER_ITEM):
    lines = [l for l in text.strip().splitlines() if l.strip()]
    return lines[:n]


def render(result):
    meta = result["meta"]
    out = []
    out.append(f"## midman-ai CI: build #{meta['build']}")
    out.append("")
    out.append(
        f"Package: `{meta['package']}` &nbsp;|&nbsp; Branch: `{meta['branch']}` "
        f"&nbsp;|&nbsp; Commit: `{meta['commit']}` &nbsp;|&nbsp; Rust: `{meta['rust_version']}`"
    )
    out.append("")
    out.append("| Check | Result |")
    out.append("|---|---|")
    for key, label in (("fmt", "Formatting"), ("clippy", "Clippy (-D warnings)"),
                       ("build", "Build (all targets)"), ("test", "Tests")):
        out.append(f"| {label} | {mark(result['outcomes'].get(key))} |")
    out.append("")

    fmt = result["fmt"]
    if fmt["files"] or fmt["errors"]:
        out.append("### Formatting differences")
        out.append("")
        for f in fmt["files"]:
            out.append(f"- `{f['path']}`: {f['hunks']} hunk(s)")
        for e in fmt["errors"]:
            out.append(f"- `{e}`")
        out.append("")
        out.append("Run `cargo fmt --all` locally to fix.")
        out.append("")

    clippy = result["clippy"]
    if clippy["diagnostics"]:
        out.append(f"### Clippy: {clippy['errors']} error(s), {clippy['warnings']} warning(s)")
        out.append("")
        for d in clippy["diagnostics"][:60]:
            where = f"`{d['path']}:{d['line']}`" if d["path"] else "(no location)"
            tag = d["lint"] or d["code"]
            tag = f" `{tag}`" if tag else ""
            out.append(f"- **{d['level']}**{tag} {where}: {d['message']}")
        if len(clippy["diagnostics"]) > 60:
            out.append(f"- ... and {len(clippy['diagnostics']) - 60} more, see the raw log artifact")
        out.append("")

    test = result["test"]
    out.append("### Tests")
    out.append("")
    if test["build_errors"]:
        out.append("The test binaries did not build:")
        out.append("")
        for d in test["build_errors"]:
            where = f"`{d['path']}:{d['line']}`" if d["path"] else ""
            out.append(f"- {d['code'] or 'error'} {where}: {d['message']}")
        out.append("")
    total = test["passed"] + test["failed"] + test["ignored"]
    if not test["suites"] and not test["build_errors"]:
        out.append("⚠️ No test suites were collected. Check the raw log.")
    elif total == 0 and not test["build_errors"]:
        out.append(f"⚠️ {len(test['suites'])} test suites ran, but none contain any tests yet.")
        out.append("")
        out.append("**Total: 0**")
    else:
        out.append(
            f"**Total: {total}** &nbsp;|&nbsp; Passed: {test['passed']} &nbsp;|&nbsp; "
            f"Failed: {test['failed']} &nbsp;|&nbsp; Ignored: {test['ignored']} "
            f"&nbsp;|&nbsp; Duration: {test['duration_s']:.2f}s"
        )
    out.append("")

    failing = [(s["name"], t, s["failure_output"].get(t, ""))
               for s in test["suites"] for t in s["failed_tests"]]
    if failing:
        out.append("#### Failures")
        out.append("")
        for suite, name, output in failing:
            out.append(f"- **{suite} :: {name}**")
            for l in first_lines(output, 6):
                out.append(f"  - `{l.strip()}`")
        out.append("")

    if test["suites"]:
        out.append("<details><summary>Per-suite results</summary>")
        out.append("")
        out.append("| Suite | Passed | Failed | Ignored | Duration |")
        out.append("|---|---|---|---|---|")
        for s in test["suites"]:
            icon = "✅" if s["failed"] == 0 else "❌"
            out.append(f"| {icon} {s['name']} | {s['passed']} | {s['failed']} | "
                       f"{s['ignored']} | {s['duration_s']:.2f}s |")
        out.append("")
        out.append("</details>")
        out.append("")

    text = "\n".join(out) + "\n"
    if len(text.encode("utf-8")) > MAX_SUMMARY_BYTES:
        text = text.encode("utf-8")[:MAX_SUMMARY_BYTES].decode("utf-8", "ignore")
        text += "\n\n_Summary truncated. See the raw log artifact._\n"
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fmt-log")
    ap.add_argument("--clippy-log")
    ap.add_argument("--build-log")
    ap.add_argument("--test-log")
    for name in ("fmt", "clippy", "build", "test"):
        ap.add_argument(f"--{name}-outcome", default="")
    ap.add_argument("--json", dest="json_path", default="ci-results.json")
    ap.add_argument("--md", default="", help="file to append the markdown summary to")
    args = ap.parse_args()

    result = {
        "meta": {
            "build": os.environ.get("BUILD_NUM", "0"),
            "branch": os.environ.get("BRANCH", "unknown"),
            "commit": os.environ.get("COMMIT", "unknown")[:8],
            "rust_version": os.environ.get("RUST_VERSION", "stable"),
            "package": os.environ.get("PACKAGE", "workspace"),
        },
        "outcomes": {k: getattr(args, f"{k}_outcome") for k in ("fmt", "clippy", "build", "test")},
        "fmt": parse_fmt(read_log(args.fmt_log)),
        "clippy": parse_clippy(read_log(args.clippy_log)),
        "test": parse_tests(read_log(args.test_log)),
    }

    with open(args.json_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    md = render(result)
    if args.md:
        with open(args.md, "a", encoding="utf-8") as fh:
            fh.write(md)
    else:
        sys.stdout.write(md)

    t = result["test"]
    print(f"fmt files: {len(result['fmt']['files'])}  clippy: "
          f"{result['clippy']['errors']}E/{result['clippy']['warnings']}W  "
          f"tests: {t['passed']} passed, {t['failed']} failed, {t['ignored']} ignored",
          file=sys.stderr)


if __name__ == "__main__":
    main()
