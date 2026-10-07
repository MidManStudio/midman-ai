#!/usr/bin/env python3
# ============================================================================
# NOTICE: Full documentation, design decisions, and fix history for this file
# live in docs/ci-and-workflows.md, section "train_smoke_summary.py"
# ============================================================================
"""Turns the raw log of the train-smoke workflow into a JSON file and a markdown
job summary. Standard library only.

The smoke test prints one line per run:

    loss 5.5674 -> 0.0005, accuracy 1.000

This script reads that line and the `test result:` line. It never fails the job;
the gate step in train-smoke.yml decides pass/fail from the step outcome.
"""
import argparse
import json
import re

RE_ANSI = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
RE_RESULT = re.compile(r"loss\s+([0-9.]+)\s+->\s+([0-9.]+),\s+accuracy\s+([0-9.]+)")
RE_SUMMARY = re.compile(r"^test result: (ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored")


def parse(text):
    text = RE_ANSI.sub("", text)
    runs = [tuple(float(x) for x in m.groups()) for m in RE_RESULT.finditer(text)]
    totals = None
    for line in text.splitlines():
        m = RE_SUMMARY.match(line.strip())
        if m:
            totals = {"status": m.group(1), "passed": int(m.group(2)),
                      "failed": int(m.group(3)), "ignored": int(m.group(4))}
    return runs, totals


def tail(text, n=25):
    lines = RE_ANSI.sub("", text).splitlines()
    return "\n".join(lines[-n:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--outcome", default="")
    ap.add_argument("--rust-version", default="")
    ap.add_argument("--build", default="0")
    ap.add_argument("--branch", default="unknown")
    ap.add_argument("--commit", default="unknown")
    ap.add_argument("--json", dest="json_path", default="train-smoke-results.json")
    ap.add_argument("--md", default="", help="file to append the markdown summary to")
    args = ap.parse_args()

    try:
        with open(args.log, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        text = ""
    runs, totals = parse(text)

    result = {
        "meta": {"build": args.build, "branch": args.branch, "commit": args.commit,
                 "rust_version": args.rust_version},
        "outcome": args.outcome,
        "runs": [{"first_loss": a, "last_loss": b, "accuracy": c} for a, b, c in runs],
        "tests": totals,
    }
    with open(args.json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    icon = "\u2705 pass" if args.outcome == "success" else "\u274c fail"
    out = [f"## midman-ai train smoke: build #{args.build}", "",
           f"Branch: `{args.branch}` &nbsp;|&nbsp; Commit: `{args.commit[:10]}` &nbsp;|&nbsp; "
           f"Rust: `{args.rust_version or 'unknown'}`", "",
           f"`cargo test --release -- --ignored`: {icon}", ""]
    if runs:
        out += ["| Run | Loss at start | Loss at end | Next-token accuracy |",
                "| --- | --- | --- | --- |"]
        for i, (a, b, c) in enumerate(runs, 1):
            out.append(f"| {i} | {a:.4f} | {b:.4f} | {c:.1%} |")
        out.append("")
    else:
        out += ["No `loss ... -> ..., accuracy ...` line was found in the log.", ""]
    if totals:
        out += [f"Tests: {totals['passed']} passed, {totals['failed']} failed, "
                f"{totals['ignored']} ignored.", ""]
    if not runs or args.outcome != "success":
        out += ["<details><summary>End of the raw log</summary>", "", "```", tail(text), "```",
                "", "</details>", ""]
    md = "\n".join(out)
    if args.md:
        with open(args.md, "a", encoding="utf-8") as f:
            f.write(md + "\n")
    else:
        print(md)


if __name__ == "__main__":
    main()
