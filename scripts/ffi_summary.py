#!/usr/bin/env python3
# ============================================================================
# NOTICE: Full documentation, design decisions, and fix history for this file
# live in docs/ci-and-workflows.md, section "ffi_summary.py"
# ============================================================================
"""Turns the raw logs of the midman-api-ffi workflow into a JSON file and a
markdown job summary, and gates the run. Standard library only.

    ffi_summary.py parse --out results.json --md "$GITHUB_STEP_SUMMARY" \\
        --cargo-raw cargo-raw.txt \\
        --c-log "C11 test, shared library=so.txt" --c-log "C++17 header check=cxx.txt" \\
        --sanitizer-log "C11 test under ASan and UBSan=asan.txt" \\
        --valgrind-log "C11 test under valgrind=vg.txt" --valgrind-status 0
    ffi_summary.py gate results.json

The C programs (ffi-smoke-test/test.c and cxx_check.cpp) print one line per check,
`ok:   <message>` or `FAIL: <message> (line N)`, and end with `=== N check(s)
failed ===`. A log without that last line (a crash, a link error, a timeout) is one
failed check, "program ran to completion", never an empty passing suite.

`parse` never fails the job. `gate` exits 1 if any check failed or none were
collected at all: a run that produced no results is not a pass.
"""
import argparse
import json
import os
import re
import sys

RE_ANSI = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
RE_OK = re.compile(r"^ok:\s+(.*\S)\s*$")
RE_FAIL = re.compile(r"^FAIL:\s+(.*?)\s+\(line (\d+)\)\s*$")
RE_FINAL = re.compile(r"^=== (\d+) check\(s\) failed ===\s*$")
RE_CARGO_TEST = re.compile(r"^test (.+?) \.\.\. (ok|FAILED)\s*$")  # doctest names contain spaces
RE_ASAN = re.compile(r"ERROR: AddressSanitizer")
RE_LSAN = re.compile(r"ERROR: LeakSanitizer")
RE_UBSAN = re.compile(r"runtime error:")
RE_VALGRIND_SUMMARY = re.compile(r"ERROR SUMMARY: (\d+) errors from (\d+) contexts")
RE_VALGRIND_LOST = re.compile(r"definitely lost: ([\d,]+) bytes in ([\d,]+) blocks")
MAX_TAIL_CHARS = 20_000


def strip_ansi(text):
    return RE_ANSI.sub("", text)


def read(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return strip_ansi(fh.read())
    except OSError:
        return ""


def check(name, passed, detail=""):
    return {"name": name, "status": "passed" if passed else "failed", "detail": detail}


def suite(name, checks, log=""):
    return {
        "name": name,
        "passed": sum(1 for c in checks if c["status"] == "passed"),
        "failed": sum(1 for c in checks if c["status"] == "failed"),
        "checks": checks,
        "log": log,
    }


def c_checks(text):
    """One check per `ok:`/`FAIL:` line, plus a failed check if the program did not finish."""
    checks, closing = [], None
    for line in text.splitlines():
        ok = RE_OK.match(line)
        if ok:
            checks.append(check(ok.group(1), True))
            continue
        bad = RE_FAIL.match(line)
        if bad:
            checks.append(check(bad.group(1), False, f"line {bad.group(2)}"))
            continue
        final = RE_FINAL.match(line)
        if final:
            closing = int(final.group(1))
    reported_failures = sum(1 for c in checks if c["status"] == "failed")
    if closing is None:
        checks.append(check("program ran to completion", False,
                            "no closing line: a crash, a link error or a timeout"))
    elif closing != reported_failures:
        checks.append(check("the failure count agrees with the FAIL lines", False,
                            f"the program said {closing}, the log has {reported_failures}"))
    elif not checks:
        checks.append(check("ran at least one check", False, "the program printed no checks"))
    return checks


def sanitizer_checks(text):
    """What the sanitizers reported, as three checks that are always present."""
    ub_lines = [line.strip() for line in text.splitlines() if RE_UBSAN.search(line)]
    return [
        check("no AddressSanitizer errors", not RE_ASAN.search(text)),
        check("no LeakSanitizer reports", not RE_LSAN.search(text)),
        check("no UndefinedBehaviorSanitizer reports", not ub_lines, "; ".join(ub_lines[:3])),
    ]


def valgrind_checks(text, status):
    """Valgrind's own verdict. A log with no ERROR SUMMARY line is a failed run."""
    summary = RE_VALGRIND_SUMMARY.search(text)
    if not summary:
        return [check("valgrind ran to completion", False, "no ERROR SUMMARY line in the log")]
    errors = int(summary.group(1))
    lost = RE_VALGRIND_LOST.search(text)
    lost_bytes = int(lost.group(1).replace(",", "")) if lost else 0
    return [
        check("valgrind found no errors", errors == 0, f"{errors} error(s)" if errors else ""),
        check("nothing definitely lost", lost_bytes == 0, f"{lost_bytes} bytes" if lost_bytes else ""),
        check("valgrind exit status is 0", status == 0, f"exit status {status}" if status else ""),
    ]


def cargo_checks(text):
    """One check per `test name ... ok|FAILED` line; none at all is a failed check."""
    checks = []
    for line in text.splitlines():
        m = RE_CARGO_TEST.match(line.strip())
        if m:
            checks.append(check(m.group(1), m.group(2) == "ok"))
    if not checks:
        checks.append(check("collected at least one Rust test", False,
                            "cargo missing, a compile error, or a filter that matched nothing"))
    return checks


def summarize(suites):
    return {"total": sum(len(s["checks"]) for s in suites),
            "passed": sum(s["passed"] for s in suites),
            "failed": sum(s["failed"] for s in suites)}


def split_named(value):
    name, sep, path = value.rpartition("=")
    if not sep or not name:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {value!r}")
    return name, path


def tail(text):
    return text[-MAX_TAIL_CHARS:]


def build_md(meta, suites, summary):
    verdict = "\u2705 pass" if summary["failed"] == 0 and summary["total"] > 0 else "\u274c FAIL"
    out = [f"## midman-api C ABI: build #{meta['build']}", "",
           f"Branch: `{meta['branch']}` &nbsp;|&nbsp; Commit: `{meta['commit'][:10]}` "
           f"&nbsp;|&nbsp; Runner: `{meta['runner']}` &nbsp;|&nbsp; Rust: `{meta['rust_version'] or 'unknown'}`"
           f" &nbsp;|&nbsp; C: `{meta['cc'] or 'unknown'}` &nbsp;|&nbsp; C++: `{meta['cxx'] or 'unknown'}`", ""]
    if meta["note"]:
        out += [f"Note: {meta['note']}", ""]
    out += [f"{verdict}: {summary['passed']} of {summary['total']} checks passed.", "",
            "| Suite | Passed | Failed |", "|---|---|---|"]
    out += [f"| {s['name']} | {s['passed']} | {s['failed']} |" for s in suites]
    out.append("")
    failed = [(s, c) for s in suites for c in s["checks"] if c["status"] == "failed"]
    if failed:
        out += ["### Failed checks", ""]
        out += [f"- **{s['name']}**: {c['name']}" + (f" ({c['detail']})" if c["detail"] else "") for s, c in failed]
        out.append("")
        for s in suites:
            if s["failed"] and s["log"]:
                out += [f"<details><summary>End of the log: {s['name']}</summary>", "", "```", tail(s["log"]),
                        "```", "", "</details>", ""]
    if summary["total"] == 0:
        out += ["No checks were collected.", ""]
    return "\n".join(out) + "\n"


def cmd_parse(args):
    suites = []
    if args.cargo_raw:
        text = read(args.cargo_raw)
        suites.append(suite("Rust tests (cargo test -p midman-api)", cargo_checks(text), text))
    for name, path in args.c_log:
        text = read(path)
        suites.append(suite(name, c_checks(text), text))
    for name, path in args.sanitizer_log:
        text = read(path)
        suites.append(suite(name, c_checks(text) + sanitizer_checks(text), text))
    for name, path in args.valgrind_log:
        text = read(path)
        suites.append(suite(name, c_checks(text) + valgrind_checks(text, args.valgrind_status), text))
    summary = summarize(suites)
    meta = {"build": args.build, "branch": args.branch, "commit": args.commit,
            "rust_version": args.rust_version, "cc": args.cc_version, "cxx": args.cxx_version,
            "runner": os.environ.get("RUNNER_LABEL", "unknown"), "note": os.environ.get("BASELINE_NOTE", "")}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "summary": summary,
                   "suites": [{k: v for k, v in s.items() if k != "log"} for s in suites]}, fh, indent=2)
        fh.write("\n")
    md = build_md(meta, suites, summary)
    if args.md:
        with open(args.md, "a", encoding="utf-8") as fh:
            fh.write(md)
    else:
        sys.stdout.write(md)
    print(f"{summary['total']} check(s), {summary['failed']} failed", file=sys.stderr)


def cmd_gate(args):
    try:
        with open(args.results, encoding="utf-8") as fh:
            summary = json.load(fh)["summary"]
    except (OSError, ValueError, KeyError):
        print("No results file was produced.")
        return 1
    print(f"total={summary['total']} passed={summary['passed']} failed={summary['failed']}")
    if summary["total"] == 0:
        print("No checks were collected.")
        return 1
    if summary["failed"] > 0:
        print(f"{summary['failed']} check(s) failed. See the job summary.")
        return 1
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("parse", help="parse the logs into JSON and a markdown summary")
    p.add_argument("--out", default="midman-api-ffi-results.json")
    p.add_argument("--md", default="", help="file to append the markdown summary to")
    p.add_argument("--cargo-raw", default="")
    p.add_argument("--c-log", action="append", default=[], type=split_named, metavar="NAME=PATH")
    p.add_argument("--sanitizer-log", action="append", default=[], type=split_named, metavar="NAME=PATH")
    p.add_argument("--valgrind-log", action="append", default=[], type=split_named, metavar="NAME=PATH")
    p.add_argument("--valgrind-status", type=int, default=127)
    p.add_argument("--build", default="0")
    p.add_argument("--branch", default="unknown")
    p.add_argument("--commit", default="unknown")
    p.add_argument("--rust-version", default="")
    p.add_argument("--cc-version", default="")
    p.add_argument("--cxx-version", default="")
    p.set_defaults(run=cmd_parse)
    g = sub.add_parser("gate", help="exit 1 if any check failed or none were collected")
    g.add_argument("results")
    g.set_defaults(run=cmd_gate)
    args = ap.parse_args(argv)
    return args.run(args) or 0


if __name__ == "__main__":
    sys.exit(main())
