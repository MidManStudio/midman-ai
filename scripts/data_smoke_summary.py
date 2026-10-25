#!/usr/bin/env python3
# ============================================================================
# NOTICE: Full documentation, design decisions, and fix history for this file
# live in docs/ci-and-workflows.md, section "data_smoke_summary.py"
# ============================================================================
"""Turns the raw log of the data-smoke workflow into a JSON file and a markdown job
summary, and compares the results of several runners. Standard library only.

The log is one file with a section per step, each starting with a line `=== name ===`
(build, corpus, train, shards, scrub speed, oracle). Each section is read for the lines
the tools print:

    corpus: 412 files
    trained on 412 files (0 unreadable): ...
    learned 760 of 760 merges, vocabulary 1024, ...
    files read 412: not UTF-8 0, empty 0, duplicates 3
    secrets replaced 2 {"url-password": 2}
    documents kept 409: 5000000 bytes, 1700000 tokens (2.94 bytes per token)
    7 shards in shards; 13000 sequences of 128 tokens; 1.2 s, 4.1 MB/s
    first 100 batches (4 x 128 tokens, seed 1): sha256 <64 hex digits>
    5.1 MB in 0.060 s = 85.0 MB/s
    <the table printed by oracle_corpus.py compare>

Modes:

  --log FILE ...     write the JSON and the summary of one runner. Never fails the job;
                     the gate step in the workflow reads `ok` from the JSON.
  --compare DIR ...  read every data-smoke-results.json below DIR (one per runner) and
                     check that the vocabulary, every shard and the digest of the first
                     batches are the same on all of them. Exits 1 if they are not, or if
                     fewer than two results are found.
"""
import argparse
import hashlib
import json
import os
import re
import sys

RE_ANSI = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
RE_SECTION = re.compile(r"^=== (.+?) ===\s*$")
RE_CORPUS = re.compile(r"^corpus: (\d+) files")
RE_TRAIN = re.compile(
    r"trained on (\d+) files \((\d+) unreadable\): (\d+) bytes, (\d+) pieces, (\d+) distinct")
RE_MERGES = re.compile(r"learned (\d+) of (\d+) merges, vocabulary (\d+)")
RE_READ = re.compile(r"files read (\d+): not UTF-8 (\d+), empty (\d+), duplicates (\d+)")
RE_SECRETS = re.compile(r"secrets replaced (\d+)(?: (\{.*\}))?")
RE_RULE = re.compile(r'"([^"]+)":\s*(\d+)')
RE_DOCS = re.compile(r"documents kept (\d+): (\d+) bytes, (\d+) tokens \(([0-9.]+) bytes per token\)")
RE_SHARDS = re.compile(
    r"(\d+) shards in (\S+); (\d+) sequences of (\d+) tokens; ([0-9.]+) s, ([0-9.]+) MB/s")
RE_BATCHES = re.compile(
    r"first (\d+) batches \((\d+) x (\d+) tokens, seed (\d+)\): sha256 ([0-9a-f]{64})")
RE_SPEED = re.compile(r"([0-9.]+) MB in ([0-9.]+) s = ([0-9.]+) MB/s")
RE_GITLEAKS = re.compile(r"^gitleaks (\d+\.\d+\.\d+)", re.M)
RE_ORACLE_ROW = re.compile(r"^(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$")
RE_OUTSIDE_OURS = re.compile(r"findings of ours outside the samples: (\d+)")
RE_OUTSIDE_THEIRS = re.compile(r"findings of gitleaks outside the samples: (\d+)")
RE_FINISHED = re.compile(r"^[ \t]*Finished ", re.M)
RE_PANIC = re.compile(r"panicked at")
MAX_RAW_EMBED = 200_000
TAIL_LINES = 60
RESULT_NAME = "data-smoke-results.json"
STEPS = ("build", "corpus", "train", "shards", "speed", "oracle")


def strip_ansi(text):
    return RE_ANSI.sub("", text)


def sections(text):
    """The log split at its `=== name ===` lines: {name: text of the section}."""
    found, name, lines = {}, None, []
    for line in strip_ansi(text).splitlines():
        m = RE_SECTION.match(line)
        if m:
            if name is not None:
                found[name] = "\n".join(lines)
            name, lines = m.group(1), []
        elif name is not None:
            lines.append(line)
    if name is not None:
        found[name] = "\n".join(lines)
    return found


def parse_train(text):
    t, m = RE_TRAIN.search(text), RE_MERGES.search(text)
    if not (t and m):
        return None
    return {"files": int(t.group(1)), "unreadable": int(t.group(2)), "bytes": int(t.group(3)),
            "merges": int(m.group(1)), "merges_wanted": int(m.group(2)),
            "vocab_size": int(m.group(3))}


def parse_shards(text):
    read, docs, shards = RE_READ.search(text), RE_DOCS.search(text), RE_SHARDS.search(text)
    if not (read and docs and shards):
        return None
    secrets = RE_SECRETS.search(text)
    by_rule = {k: int(v) for k, v in RE_RULE.findall(secrets.group(2) or "")} if secrets else {}
    batches = RE_BATCHES.search(text)
    return {
        "files_read": int(read.group(1)), "not_utf8": int(read.group(2)),
        "empty": int(read.group(3)), "duplicates": int(read.group(4)),
        "secrets": int(secrets.group(1)) if secrets else 0, "secrets_by_rule": by_rule,
        "documents": int(docs.group(1)), "bytes": int(docs.group(2)),
        "tokens": int(docs.group(3)), "bytes_per_token": float(docs.group(4)),
        "shard_count": int(shards.group(1)), "sequences": int(shards.group(3)),
        "seq_len": int(shards.group(4)), "seconds": float(shards.group(5)),
        "mb_per_s": float(shards.group(6)),
        "batches": int(batches.group(1)) if batches else None,
        "batch_digest": batches.group(5) if batches else None,
    }


def parse_oracle(text):
    rows = []
    for line in text.splitlines():
        m = RE_ORACLE_ROW.match(line.strip())
        if m and m.group(1) != "kind":
            rows.append({"kind": m.group(1), "gitleaks": int(m.group(2)),
                         "ours": int(m.group(3)), "samples": int(m.group(4))})
    ours_out, theirs_out = RE_OUTSIDE_OURS.search(text), RE_OUTSIDE_THEIRS.search(text)
    if not rows or ours_out is None:
        return None
    version = RE_GITLEAKS.search(text)
    outside = int(ours_out.group(1))
    # The scrubber must find every sample of every kind and nothing outside them.
    ok = outside == 0 and all(r["ours"] == r["samples"] for r in rows)
    return {"gitleaks_version": version.group(1) if version else "",
            "kinds": rows, "ours_outside": outside,
            "gitleaks_outside": int(theirs_out.group(1)) if theirs_out else None, "ok": ok}


def parse(text):
    """Everything the log says: {corpus, train, shards, speed, oracle} (None if absent)."""
    parts = sections(text)
    corpus = RE_CORPUS.search(parts.get("corpus", ""))
    speed = RE_SPEED.search(parts.get("scrub speed", ""))
    return {
        "corpus_files": int(corpus.group(1)) if corpus else None,
        "train": parse_train(parts.get("train", "")),
        "shards": parse_shards(parts.get("shards", "")),
        "scrub_mb_per_s": float(speed.group(3)) if speed else None,
        "oracle": parse_oracle(parts.get("oracle", "")) if "oracle" in parts else None,
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def digests(vocabulary_path, shards_dir):
    """SHA-256 of the vocabulary file, of every shard in file-name order, and one over the
    names and digests of all the shards.

    The JSON keeps the shard digests as a plain list and the vocabulary under a neutral key:
    gitleaks reads `"tokenizer": "<64 hex digits>"` as a secret, and one committed line like
    that would fail its scan of the history for good. The shard names are covered by the
    combined digest."""
    result = {"vocabulary": None, "shards": [], "combined": None}
    if vocabulary_path and os.path.isfile(vocabulary_path):
        result["vocabulary"] = sha256_file(vocabulary_path)
    named = {}
    if shards_dir and os.path.isdir(shards_dir):
        for name in sorted(os.listdir(shards_dir)):
            if name.endswith(".mmtokens"):
                named[name] = sha256_file(os.path.join(shards_dir, name))
    result["shards"] = [digest for _name, digest in sorted(named.items())]
    if named:
        listing = "".join(f"{n}\n{d}\n" for n, d in sorted(named.items()))
        result["combined"] = hashlib.sha256(listing.encode()).hexdigest()
    return result


def parse_outcomes(text):
    """`build=success,oracle=skipped` as a dict. Missing steps are not in it."""
    found = {}
    for item in text.split(","):
        if "=" in item:
            name, outcome = item.split("=", 1)
            found[name.strip()] = outcome.strip()
    return found


def decide_ok(outcomes, parsed):
    """Whether every check passed. Steps listed in `outcomes` must have succeeded (the
    oracle may also have been skipped, as it is off the x86_64 Linux runner); what they
    should have printed must have been found; an oracle that ran must have agreed."""
    for step in STEPS:
        outcome = outcomes.get(step)
        if outcome is None:
            continue
        allowed = ("success", "skipped") if step == "oracle" else ("success",)
        if outcome not in allowed:
            return False
    if parsed["train"] is None or parsed["shards"] is None:
        return False
    if parsed["shards"]["batch_digest"] is None or parsed["scrub_mb_per_s"] is None:
        return False
    oracle = parsed["oracle"]
    if oracle is not None and not oracle["ok"]:
        return False
    if outcomes.get("oracle") == "success" and oracle is None:
        return False
    return True


def raw_embed(text):
    """The end of the log without the build output: from cargo's last `Finished` line."""
    text = strip_ansi(text)
    finished = list(RE_FINISHED.finditer(text))
    body = text[finished[-1].start():] if finished else text
    body = "\n".join(body.splitlines()[-TAIL_LINES:])
    return body[-MAX_RAW_EMBED:]


def icon(outcome):
    return {"success": "\u2705", "skipped": "\u2796", "failure": "\u274c",
            "cancelled": "\u26aa"}.get(outcome, "\u2753")


def short(digest):
    return f"`{digest[:12]}`" if digest else "none"


def fmt_int(n):
    return f"{n:,}"


def markdown(result, text):
    meta, parsed = result["meta"], result
    out = [f"## midman-ai data smoke: build #{meta['build']}", "",
           f"Branch: `{meta['branch']}` &nbsp;|&nbsp; Commit: `{str(meta['commit'])[:10]}` &nbsp;|&nbsp; "
           f"Runner: `{meta['runner']}` &nbsp;|&nbsp; Rust: `{meta['rust_version'] or 'unknown'}`", ""]
    if meta["note"]:
        out += [f"Note: {meta['note']}", ""]
    mark = "\u2705 pass" if result["ok"] else "\u274c fail"
    out += [f"Result: {mark}", ""]
    if result["outcomes"]:
        out += ["| Step | Outcome |", "|---|---|"]
        for step in STEPS:
            if step in result["outcomes"]:
                out.append(f"| {step} | {icon(result['outcomes'][step])} {result['outcomes'][step]} |")
        out.append("")
    train, shards = parsed["train"], parsed["shards"]
    out += ["### Pipeline", "", "| What | Value |", "|---|---|"]
    if parsed["corpus_files"] is not None:
        out.append(f"| Corpus | {fmt_int(parsed['corpus_files'])} tracked text files |")
    if train:
        out.append(f"| Vocabulary | {fmt_int(train['vocab_size'])} ids, {fmt_int(train['merges'])} merges "
                   f"learned from {fmt_int(train['files'])} files ({fmt_int(train['bytes'])} bytes) |")
    if shards:
        out += [
            f"| Files read | {fmt_int(shards['files_read'])}: not UTF-8 {shards['not_utf8']}, "
            f"empty {shards['empty']}, duplicates {shards['duplicates']} |",
            f"| Secrets replaced | {shards['secrets']} {shards['secrets_by_rule'] or ''} |",
            f"| Documents kept | {fmt_int(shards['documents'])} ({fmt_int(shards['bytes'])} bytes) |",
            f"| Tokens | {fmt_int(shards['tokens'])} ({shards['bytes_per_token']:.2f} bytes per token) |",
            f"| Shards | {shards['shard_count']}, holding {fmt_int(shards['sequences'])} "
            f"sequences of {shards['seq_len']} tokens |",
            f"| Whole path | {shards['seconds']:.1f} s, {shards['mb_per_s']:.1f} MB/s |",
        ]
    if parsed["scrub_mb_per_s"] is not None:
        out.append(f"| Scrubber alone | {parsed['scrub_mb_per_s']:.1f} MB/s |")
    out.append("")
    if not train or not shards:
        out += ["The log does not hold the lines the tools print; see the end of the raw log below.", ""]
    dig = result["digests"]
    out += ["### Output digests", "",
            "These must be the same on every runner (`platforms: all` compares them).", "",
            "| Output | SHA-256 (first 12 digits) |", "|---|---|",
            f"| vocabulary file | {short(dig['vocabulary'])} |",
            f"| all {len(dig['shards'])} shards | {short(dig['combined'])} |",
            f"| first {shards['batches'] if shards and shards['batches'] else 0} batches | "
            f"{short(shards['batch_digest'] if shards else None)} |", ""]
    oracle = parsed["oracle"]
    if oracle:
        verdict = "\u2705 found every sample and nothing else" if oracle["ok"] else "\u274c missed samples or found extra"
        out += [f"### Scrubber against gitleaks {oracle['gitleaks_version']}", "", f"{verdict}.", "",
                "| Kind | gitleaks | scrubber | samples |", "|---|---|---|---|"]
        for row in oracle["kinds"]:
            flag = "" if row["ours"] == row["samples"] else " \u274c"
            out.append(f"| {row['kind']} | {row['gitleaks']} | {row['ours']}{flag} | {row['samples']} |")
        out += ["", f"Findings outside the samples: scrubber {oracle['ours_outside']}, "
                    f"gitleaks {oracle['gitleaks_outside']}.", ""]
    elif result["outcomes"].get("oracle") == "skipped":
        out += ["The gitleaks comparison runs on x86_64 Linux only and was skipped here.", ""]
    panics = len(RE_PANIC.findall(strip_ansi(text)))
    if panics:
        out += ["### Diagnostics", "", "| Kind | Count |", "|---|---|", f"| panic | {panics} |", ""]
    if not result["ok"]:
        out += ["<details><summary>End of the raw log</summary>", "", "```", raw_embed(text), "```",
                "", "</details>", ""]
    return "\n".join(out)


def build_result(text, outcomes_text, tokenizer, shards_dir, meta):
    parsed = parse(text)
    outcomes = parse_outcomes(outcomes_text)
    result = {"meta": meta, "outcomes": outcomes, **parsed,
              "digests": digests(tokenizer, shards_dir)}
    result["ok"] = decide_ok(outcomes, parsed)
    return result


# ---------------------------------------------------------------------------
# Comparing runners
# ---------------------------------------------------------------------------

def comparable(result):
    """What must be identical on every runner."""
    shards = result.get("shards") or {}
    train = result.get("train") or {}
    dig = result.get("digests") or {}
    return {
        "vocabulary file": dig.get("vocabulary"),
        "all shards": dig.get("combined"),
        "first batches": shards.get("batch_digest"),
        "merges learned": train.get("merges"),
        "documents kept": shards.get("documents"),
        "tokens": shards.get("tokens"),
        "duplicates": shards.get("duplicates"),
        "secrets replaced": shards.get("secrets"),
        "sequences": shards.get("sequences"),
    }


def compare(results):
    """Returns (runners, rows, shard_differences, ok). `results` maps a runner to its result."""
    runners = sorted(results)
    per_runner = {r: comparable(results[r]) for r in runners}
    names = list(next(iter(per_runner.values())).keys()) if per_runner else []
    rows, ok = [], len(runners) >= 2
    for name in names:
        values = [per_runner[r][name] for r in runners]
        same = len(set(map(str, values))) == 1 and values[0] is not None
        ok = ok and same
        rows.append((name, values, same))
    shard_lists = {r: (results[r].get("digests") or {}).get("shards") or [] for r in runners}
    differences = []
    for index in range(max((len(s) for s in shard_lists.values()), default=0)):
        values = [s[index] if index < len(s) else None for s in (shard_lists[r] for r in runners)]
        if len(set(values)) != 1 or values[0] is None:
            differences.append((f"shard {index}", values))
    ok = ok and not differences
    return runners, rows, differences, ok


def load_results(directory):
    found = {}
    for root, _dirs, files in os.walk(directory):
        if RESULT_NAME in files:
            with open(os.path.join(root, RESULT_NAME), encoding="utf-8") as handle:
                data = json.load(handle)
            label = (data.get("meta") or {}).get("runner") or os.path.basename(root)
            found[label] = data
    return found


def compare_markdown(runners, rows, differences, ok, count):
    out = ["## midman-ai data smoke: the runners compared", ""]
    if count < 2:
        out += [f"Found {count} result(s); at least two are needed to compare.", ""]
    out += ["| What | " + " | ".join(f"`{r}`" for r in runners) + " | Same |",
            "|---|" + "---|" * len(runners) + "---|"]
    for name, values, same in rows:
        shown = [short(v) if isinstance(v, str) and len(v) == 64 else str(v) for v in values]
        mark = "\u2705" if same else "\u274c"
        out.append(f"| {name} | " + " | ".join(shown) + f" | {mark} |")
    out.append("")
    if differences:
        out += ["Shards that differ or are missing on some runner:", ""]
        for name, values in differences:
            out.append(f"- `{name}`: " + ", ".join(short(v) for v in values))
        out.append("")
    out.append("\u2705 The output is byte-for-byte the same on every runner." if ok
               else "\u274c The runners disagree, or there was nothing to compare.")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log")
    ap.add_argument("--outcomes", default="")
    ap.add_argument("--tokenizer", default="tokenizer.txt")
    ap.add_argument("--shards", default="shards")
    ap.add_argument("--rust-version", default="")
    ap.add_argument("--build", default="0")
    ap.add_argument("--branch", default="unknown")
    ap.add_argument("--commit", default="unknown")
    ap.add_argument("--json", dest="json_path", default=RESULT_NAME)
    ap.add_argument("--md", default="", help="file to append the markdown summary to")
    ap.add_argument("--compare", default="", help="directory holding one result per runner")
    args = ap.parse_args()

    if args.compare:
        results = load_results(args.compare)
        runners, rows, differences, ok = compare(results) if results else ([], [], [], False)
        md = compare_markdown(runners, rows, differences, ok, len(results))
        if args.md:
            with open(args.md, "a", encoding="utf-8") as handle:
                handle.write(md)
        else:
            print(md)
        sys.exit(0 if ok else 1)

    if not args.log:
        ap.error("--log or --compare is required")
    try:
        with open(args.log, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError:
        text = ""
    meta = {"build": args.build, "branch": args.branch, "commit": args.commit,
            "rust_version": args.rust_version,
            "runner": os.environ.get("RUNNER_LABEL", "unknown"),
            "note": os.environ.get("BASELINE_NOTE", "")}
    result = build_result(text, args.outcomes, args.tokenizer, args.shards, meta)
    with open(args.json_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    md = markdown(result, text)
    if args.md:
        with open(args.md, "a", encoding="utf-8") as handle:
            handle.write(md + "\n")
    else:
        print(md)


if __name__ == "__main__":
    main()
