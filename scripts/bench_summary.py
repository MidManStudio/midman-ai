#!/usr/bin/env python3
# ============================================================================
# NOTICE: Full documentation, design decisions, and fix history for this file
# live in docs/ci-and-workflows.md, section "bench_summary.py"
# ============================================================================
"""Turns the raw `cargo bench` log from benchmarks.yml into a JSON file and a
markdown job summary. Standard library only.

Parses criterion's `time:` and `thrpt:` lines. A benchmark id is `group/variant`,
split at the final `/`. Within a group:

  * a variant named `baseline-*` is a reference implementation; every other
    variant gets a ratio against it and a badge;
  * otherwise a variant named `unit-*` is a cost denominator (for example the
    forward pass); every other variant gets a plain multiple, no badge.

Never fails the job; the gate step in benchmarks.yml decides pass/fail from the
cargo step outcome.
"""
import argparse
import json
import os
import re
import sys

RE_ANSI = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
RE_TIME = re.compile(
    r"^(?P<name>\S.*?)?\s*time:\s+\[(?P<lo>[\d.]+)\s+(?P<lou>\S+)\s+(?P<mid>[\d.]+)\s+(?P<midu>\S+)\s+"
    r"(?P<hi>[\d.]+)\s+(?P<hiu>\S+)\]\s*$"
)
# Criterion pads a unit that has no SI prefix with an extra space ("729.04  elem/s").
RE_THRPT = re.compile(
    r"^\s*thrpt:\s+\[[\d.]+\s+\S+\s+(?P<mid>[\d.]+)\s+(?P<unit>\S+)\s+[\d.]+\s+\S+\]\s*$"
)
RE_OUTLIERS = re.compile(r"^Found (\d+) outliers among (\d+) measurements")
DIAG_PATTERNS = [
    ("took zero time", re.compile(r"took zero time", re.I)),
    ("unable to complete samples", re.compile(r"Unable to complete \d+ samples", re.I)),
    ("completed N iterations", re.compile(r"Completed \d+ iterations")),
    ("panic", re.compile(r"panicked at")),
    ("criterion warning", re.compile(r"^Warning:", re.M)),
]
UNIT_NS = {"ps": 1e-3, "ns": 1.0, "µs": 1e3, "us": 1e3, "μs": 1e3, "ms": 1e6, "s": 1e9}
# Cargo prints `Finished `bench` profile ...` now and `Finished bench [optimized] ...` on older versions.
RE_FINISHED = re.compile(r"^[ \t]*Finished ", re.M)
MAX_RAW_EMBED = 200_000
BASELINE_PREFIX = "baseline-"
UNIT_PREFIX = "unit-"


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


def split_id(name):
    group, sep, variant = name.rpartition("/")
    if not sep:
        return "(ungrouped)", name
    return group, variant


def parse(text):
    """Returns one dict per benchmark, in the order they appear in the log."""
    results = []
    prev = ""
    for line in text.splitlines():
        m = RE_TIME.match(line)
        if m:
            name = (m.group("name") or "").strip() or prev
            group, variant = split_id(name)
            results.append({
                "name": name,
                "group": group,
                "variant": variant,
                "low_ns": to_ns(m.group("lo"), m.group("lou")),
                "mid_ns": to_ns(m.group("mid"), m.group("midu")),
                "high_ns": to_ns(m.group("hi"), m.group("hiu")),
                "throughput": None,
                "outliers": 0,
                "samples": 0,
            })
            continue
        t = RE_THRPT.match(line)
        if t and results:
            results[-1]["throughput"] = f"{t.group('mid')} {t.group('unit')}"
            continue
        o = RE_OUTLIERS.match(line)
        if o and results:
            results[-1]["outliers"], results[-1]["samples"] = int(o.group(1)), int(o.group(2))
        if line.strip() and not line.startswith((" ", "\t")):
            prev = line.strip()
    return results


def group_results(results):
    """Groups in log order, each with its denominator variant and kind (or None)."""
    groups = {}
    for r in results:
        groups.setdefault(r["group"], []).append(r)
    out = []
    for name, rows in groups.items():
        denominator, kind = None, None
        for prefix, label in ((BASELINE_PREFIX, "baseline"), (UNIT_PREFIX, "unit")):
            found = next((r for r in rows if r["variant"].startswith(prefix)), None)
            if found:
                denominator, kind = found, label
                break
        for r in rows:
            r["ratio"] = None
            if denominator is not None and r is not denominator and denominator["mid_ns"] > 0:
                r["ratio"] = r["mid_ns"] / denominator["mid_ns"]
        out.append({"group": name, "rows": rows, "denominator": denominator, "kind": kind})
    return out


def badge(ratio):
    if ratio <= 1.05:
        return f"✅ {ratio:.2f}x" + (" (faster)" if ratio < 0.95 else " (parity)")
    if ratio <= 1.5:
        return f"⚠️ {ratio:.2f}x"
    if ratio <= 5.0:
        return f"❌ {ratio:.2f}x"
    return f"🔴 {ratio:.1f}x slower"


def diagnostics(text):
    found = []
    for label, rx in DIAG_PATTERNS:
        n = len(rx.findall(text))
        if n:
            found.append({"kind": label, "count": n})
    return found


def raw_embed(text, outcome, has_results):
    """The slice of the raw log worth showing, or an empty string.

    Everything before cargo's own `Finished` line is build output, which can be
    thousands of warning lines and can push the job summary past GitHub's size
    limit, so it is dropped. If that marker is missing the build itself failed,
    and the whole log is the useful part.
    """
    if has_results and outcome == "success":
        return ""
    finished = list(RE_FINISHED.finditer(text))
    body = text[finished[-1].start():] if finished else text
    return body[-MAX_RAW_EMBED:]


def build_md(meta, outcome, grouped, total, diags, raw):
    status = "✅ pass" if outcome == "success" else "❌ FAIL" if outcome == "failure" else outcome or "not run"
    out = [f"## midman-ai benchmarks: build #{meta['build']}", "",
           f"Package: `{meta['package']}` &nbsp;|&nbsp; Filter: `{meta['filter'] or '(none)'}` "
           f"&nbsp;|&nbsp; Branch: `{meta['branch']}` &nbsp;|&nbsp; Commit: `{meta['commit']}` "
           f"&nbsp;|&nbsp; Runner: `{meta['runner']}` &nbsp;|&nbsp; Rust: `{meta['rust_version']}`", ""]
    if meta["note"]:
        out += [f"Note: {meta['note']}", ""]
    out += [f"`cargo bench`: {status}", ""]

    if not grouped:
        out += ["No criterion results were found in the log.", ""]
    else:
        out += [f"**{total} benchmarks in {len(grouped)} group(s)**", "",
                "Time is criterion's estimate with its 95% interval. Throughput counts elements per "
                "second: floating-point operations for matrix products, tokens for training steps. "
                "Ratios are against the group's `baseline-*` variant (badged) or, failing that, its "
                "`unit-*` variant (a plain multiple).", ""]
        for g in grouped:
            head = {"baseline": "vs baseline", "unit": "x the unit variant"}.get(g["kind"], "")
            cols = ["Variant", "Time", "95% interval", "Throughput"] + ([head] if head else []) + ["Outliers"]
            out += [f"### {g['group']}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
            for r in g["rows"]:
                cells = [f"`{r['variant']}`", pretty(r["mid_ns"]),
                         f"{pretty(r['low_ns'])} to {pretty(r['high_ns'])}", r["throughput"] or "n/a"]
                if head:
                    if r is g["denominator"]:
                        cells.append(f"({g['kind']})")
                    elif g["kind"] == "baseline":
                        cells.append(badge(r["ratio"]))
                    else:
                        cells.append(f"{r['ratio']:.2f}x")
                cells.append(f"{r['outliers']}/{r['samples']}" if r["samples"] else "0")
                out.append("| " + " | ".join(cells) + " |")
            out.append("")

    if diags:
        out += ["### Diagnostics", "", "| Kind | Count |", "|---|---|"]
        out += [f"| {d['kind']} | {d['count']} |" for d in diags]
        out.append("")
    if raw:
        out += ["<details><summary>End of raw log</summary>", "", "```", raw, "```", "", "</details>", ""]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True)
    ap.add_argument("--outcome", default="")
    ap.add_argument("--json", dest="json_path", default="bench-results.json")
    ap.add_argument("--md", default="", help="file to append the markdown summary to")
    args = ap.parse_args()

    text = read(args.log)
    results = parse(text)
    grouped = group_results(results)
    diags = diagnostics(text)
    meta = {
        "build": os.environ.get("BUILD_NUM", "0"),
        "branch": os.environ.get("BRANCH", "unknown"),
        "commit": os.environ.get("COMMIT", "unknown")[:8],
        "rust_version": os.environ.get("RUST_VERSION", "stable"),
        "runner": os.environ.get("RUNNER_LABEL", "unknown"),
        "package": os.environ.get("PACKAGE", "workspace"),
        "filter": os.environ.get("BENCH_FILTER", ""),
        "note": os.environ.get("BASELINE_NOTE", ""),
    }
    for g in grouped:  # keep the JSON free of object cross-references
        g["denominator"] = g["denominator"]["variant"] if g["denominator"] else None
    with open(args.json_path, "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "outcome": args.outcome, "groups": grouped, "diagnostics": diags}, fh, indent=2)
        fh.write("\n")

    md = build_md(meta, args.outcome, [
        {**g, "denominator": next((r for r in g["rows"] if r["variant"] == g["denominator"]), None)}
        for g in grouped
    ], len(results), diags, raw_embed(text, args.outcome, bool(results)))
    if args.md:
        with open(args.md, "a", encoding="utf-8") as fh:
            fh.write(md)
    else:
        sys.stdout.write(md)
    print(f"{len(results)} benchmark(s) parsed", file=sys.stderr)


if __name__ == "__main__":
    main()
