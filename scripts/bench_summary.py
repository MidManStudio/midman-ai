#!/usr/bin/env python3
# ============================================================================
# NOTICE: Full documentation, design decisions, and fix history for this file
# live in docs/ci-and-workflows.md, section "bench_summary.py"
# ============================================================================
"""Turns the raw `cargo bench` log from benchmarks.yml into a JSON file and a
markdown job summary. Standard library only.

Parses criterion's `name  time:   [low mid high]` lines. Groups by the first
path component of the benchmark id (criterion's group name). Never fails the job; the gate step in
benchmarks.yml decides pass/fail from the cargo step outcome.
"""
import argparse
import json
import os
import re
import sys

RE_ANSI = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
RE_TIME = re.compile(
    r"^(?P<name>\S.*?)?\s*time:\s+\[(?P<lo>[\d.]+) (?P<lou>\S+) (?P<mid>[\d.]+) (?P<midu>\S+) "
    r"(?P<hi>[\d.]+) (?P<hiu>\S+)\]\s*$"
)
RE_CHANGE = re.compile(r"^\s*change:\s+\[.*?\]\s+\(p = ([\d.]+)")
RE_OUTLIERS = re.compile(r"^Found (\d+) outliers among (\d+) measurements")
DIAG_PATTERNS = [
    ("took zero time", re.compile(r"took zero time", re.I)),
    ("unable to complete samples", re.compile(r"Unable to complete \d+ samples", re.I)),
    ("panic", re.compile(r"panicked at")),
    ("criterion warning", re.compile(r"^Warning:", re.M)),
]
UNIT_NS = {"ps": 1e-3, "ns": 1.0, "µs": 1e3, "us": 1e3, "μs": 1e3, "ms": 1e6, "s": 1e9}
MAX_RAW_FALLBACK = 200_000


def to_ns(value, unit):
    return float(value) * UNIT_NS.get(unit, 1.0)


def pretty(ns):
    for limit, div, unit in ((1e9, 1e9, "s"), (1e6, 1e6, "ms"), (1e3, 1e3, "µs")):
        if ns >= limit:
            return f"{ns / div:.3f} {unit}"
    return f"{ns:.2f} ns"


def read(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return RE_ANSI.sub("", fh.read())
    except FileNotFoundError:
        return ""


def parse(text):
    results = []
    prev = ""
    for line in text.splitlines():
        m = RE_TIME.match(line)
        if m:
            name = (m.group("name") or "").strip() or prev
            results.append({
                "name": name,
                "low_ns": to_ns(m.group("lo"), m.group("lou")),
                "mid_ns": to_ns(m.group("mid"), m.group("midu")),
                "high_ns": to_ns(m.group("hi"), m.group("hiu")),
                "p_change": None,
                "outliers": 0,
                "samples": 0,
            })
            continue
        c = RE_CHANGE.match(line)
        if c and results:
            results[-1]["p_change"] = float(c.group(1))
        o = RE_OUTLIERS.match(line)
        if o and results:
            results[-1]["outliers"], results[-1]["samples"] = int(o.group(1)), int(o.group(2))
        if line.strip() and not line.startswith((" ", "\t")):
            prev = line.strip()
    return results


def diagnostics(text):
    found = []
    for label, rx in DIAG_PATTERNS:
        n = len(rx.findall(text))
        if n:
            found.append({"kind": label, "count": n})
    return found


def build_md(meta, outcome, results, diags, raw_tail):
    out = [f"## midman-ai benchmarks: build #{meta['build']}", "",
           f"Package: `{meta['package']}` &nbsp;|&nbsp; Filter: `{meta['filter'] or '(none)'}` "
           f"&nbsp;|&nbsp; Branch: `{meta['branch']}` &nbsp;|&nbsp; Commit: `{meta['commit']}` "
           f"&nbsp;|&nbsp; Rust: `{meta['rust_version']}`", ""]
    if meta["note"]:
        out += [f"Note: {meta['note']}", ""]
    out += [f"`cargo bench`: {'✅ pass' if outcome == 'success' else '❌ FAIL' if outcome == 'failure' else outcome or 'not run'}",
            ""]

    if not results:
        out += ["No criterion results were found in the log.", "",
                "This is expected while the workspace has no `[[bench]]` targets.", ""]
        if raw_tail:
            out += ["<details><summary>End of raw log</summary>", "", "```", raw_tail, "```", "", "</details>", ""]
    else:
        groups = {}
        for r in results:
            group, sep, variant = r["name"].partition("/")
            if not sep:
                group, variant = "(ungrouped)", r["name"]
            groups.setdefault(group, []).append({**r, "variant": variant})
        out += [f"**{len(results)} benchmarks in {len(groups)} group(s)**", ""]
        for group, rows in sorted(groups.items()):
            out += [f"### {group}", "", "| Benchmark | Median | Range | Outliers | vs previous run |",
                    "|---|---|---|---|---|"]
            for r in sorted(rows, key=lambda x: x["mid_ns"]):
                change = "n/a" if r["p_change"] is None else ("changed (p<0.05)" if r["p_change"] < 0.05 else "no change")
                outl = f"{r['outliers']}/{r['samples']}" if r["samples"] else "n/a"
                out.append(f"| `{r['variant']}` | {pretty(r['mid_ns'])} | "
                           f"{pretty(r['low_ns'])} to {pretty(r['high_ns'])} | {outl} | {change} |")
            out.append("")

    if diags:
        out += ["### Diagnostics", "", "| Kind | Count |", "|---|---|"]
        out += [f"| {d['kind']} | {d['count']} |" for d in diags]
        out.append("")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True)
    ap.add_argument("--outcome", default="")
    ap.add_argument("--json", dest="json_path", default="bench-results.json")
    ap.add_argument("--md", default="", help="file to append the markdown summary to")
    args = ap.parse_args()

    text = read(args.log)
    results = parse(text)
    diags = diagnostics(text)
    meta = {
        "build": os.environ.get("BUILD_NUM", "0"),
        "branch": os.environ.get("BRANCH", "unknown"),
        "commit": os.environ.get("COMMIT", "unknown")[:8],
        "rust_version": os.environ.get("RUST_VERSION", "stable"),
        "package": os.environ.get("PACKAGE", "workspace"),
        "filter": os.environ.get("BENCH_FILTER", ""),
        "note": os.environ.get("BASELINE_NOTE", ""),
    }

    with open(args.json_path, "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "outcome": args.outcome, "results": results, "diagnostics": diags}, fh, indent=2)

    tail = ""
    if not results:
        idx = text.rfind("Finished")
        tail = (text[idx:] if idx != -1 else text)[-MAX_RAW_FALLBACK:]
    md = build_md(meta, args.outcome, results, diags, tail)
    if args.md:
        with open(args.md, "a", encoding="utf-8") as fh:
            fh.write(md)
    else:
        sys.stdout.write(md)
    print(f"{len(results)} benchmark(s) parsed", file=sys.stderr)


if __name__ == "__main__":
    main()
