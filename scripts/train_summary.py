#!/usr/bin/env python3
# ============================================================================
# NOTICE: Full documentation, design decisions, and fix history for this file
# live in docs/ci-and-workflows.md, section "train_summary.py"
# ============================================================================
"""Turns the logs of a `midman-training` run into a markdown report.

    train_summary.py --run-dir DIR [--markdown FILE] [--note TEXT] [--vocab N] [--rows N]

DIR holds what the `train` example writes: `train.tsv`, `eval.tsv` (optional) and
`summary.txt`. The report is appended to FILE (the job summary), or printed. Standard
library only.
"""
import argparse
import math
import os
import statistics
import sys

TRAIN_HEADER = ["step", "loss", "grad_norm", "lr", "tokens", "seconds"]
EVAL_HEADER = ["step", "eval_loss", "tokens"]


class SummaryError(Exception):
    """A log that is not what the `train` example writes."""


def parse_tsv(text, header, name):
    """Rows of a tab separated log as lists of numbers; the first line must be `header`."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines or lines[0].split("\t") != header:
        raise SummaryError(name + ": the first line is not " + "\t".join(header))
    rows = []
    for number, line in enumerate(lines[1:], start=2):
        parts = line.split("\t")
        if len(parts) != len(header):
            raise SummaryError("%s: line %d has %d fields, not %d" % (name, number, len(parts), len(header)))
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            raise SummaryError("%s: line %d has a field that is not a number" % (name, number))
    return rows


def parse_train(text):
    """Training log rows as dicts."""
    return [dict(zip(TRAIN_HEADER, row)) for row in parse_tsv(text, TRAIN_HEADER, "train.tsv")]


def parse_eval(text):
    """Evaluation log rows as dicts; a step that appears twice keeps its last line."""
    last = {}
    for row in parse_tsv(text, EVAL_HEADER, "eval.tsv"):
        last[row[0]] = dict(zip(EVAL_HEADER, row))
    return [last[step] for step in sorted(last)]


def parse_summary(text):
    """The `key value` lines of summary.txt as a dict."""
    out = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2:
            raise SummaryError("summary.txt: `" + line + "` is not `key value`")
        out[parts[0]] = parts[1]
    for key in ("steps", "total_steps", "tokens", "finished"):
        if key not in out:
            raise SummaryError("summary.txt has no `" + key + "`")
    if out["finished"] not in ("true", "false"):
        raise SummaryError("summary.txt: finished is `" + out["finished"] + "`")
    try:
        return {
            "steps": int(out["steps"]),
            "total_steps": int(out["total_steps"]),
            "tokens": int(out["tokens"]),
            "finished": out["finished"] == "true",
        }
    except ValueError:
        raise SummaryError("summary.txt: a number is not a number")


def pick_rows(rows, count):
    """At most `count` rows spread evenly over `rows`, the first and the last included."""
    if count < 2:
        raise SummaryError("at least two rows have to be shown")
    if len(rows) <= count:
        return list(rows)
    chosen = sorted({round(i * (len(rows) - 1) / (count - 1)) for i in range(count)})
    return [rows[i] for i in chosen]


def tokens_per_second(rows):
    """The median speed between consecutive log lines, or None.

    The seconds in the log start again at 0 each time a run is resumed, so only the
    intervals inside one stretch of the log are used.
    """
    rates = []
    for before, after in zip(rows, rows[1:]):
        seconds = after["seconds"] - before["seconds"]
        tokens = after["tokens"] - before["tokens"]
        if seconds > 0 and tokens > 0:
            rates.append(tokens / seconds)
    return statistics.median(rates) if rates else None


def format_loss(value):
    return "%.3f" % value


def report(train, evals, summary, note="", vocab=None, rows=12):
    """The markdown report."""
    lines = ["## midman-ai train", ""]
    if note:
        lines += [note, ""]
    if summary["finished"]:
        status = "finished: %d of %d steps" % (summary["steps"], summary["total_steps"])
    else:
        status = "stopped on the time budget at step %d of %d (the state can be resumed)" % (
            summary["steps"], summary["total_steps"])
    lines += ["%s, %s tokens trained on." % (status, format(summary["tokens"], ","))]
    speed = tokens_per_second(train)
    if speed is not None:
        lines += ["Speed: about %s tokens per second." % format(round(speed), ",")]
    if vocab:
        lines += ["A model that knows nothing scores ln(%d) = %.3f nats per token." % (vocab, math.log(vocab))]
    lines.append("")
    if evals:
        lines += ["### Held-out loss (sources the model was not trained on)", "",
                  "| Step | Loss | Tokens |", "|---:|---:|---:|"]
        for row in pick_rows(evals, rows):
            lines.append("| %d | %s | %s |" % (row["step"], format_loss(row["eval_loss"]), format(int(row["tokens"]), ",")))
        first, last = evals[0], evals[-1]
        if len(evals) > 1:
            lines += ["", "Held-out loss went from %s to %s." % (format_loss(first["eval_loss"]), format_loss(last["eval_loss"]))]
        lines.append("")
    if train:
        lines += ["### Training loss (mean over the steps since the previous line)", "",
                  "| Step | Loss | Gradient norm | Learning rate | Tokens |", "|---:|---:|---:|---:|---:|"]
        for row in pick_rows(train, rows):
            lines.append("| %d | %s | %.3f | %.2e | %s |" % (
                row["step"], format_loss(row["loss"]), row["grad_norm"], row["lr"], format(int(row["tokens"]), ",")))
        biggest = max(train, key=lambda r: r["grad_norm"])
        lines += ["", "Largest gradient norm logged: %.3f at step %d." % (biggest["grad_norm"], biggest["step"]), ""]
    else:
        lines += ["The training log has no lines yet.", ""]
    return "\n".join(lines) + "\n"


def read(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except OSError as err:
        raise SummaryError(str(err))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--markdown", help="append the report to this file (default: print it)")
    parser.add_argument("--note", default="")
    parser.add_argument("--vocab", type=int, help="vocabulary size, for the loss of a model that knows nothing")
    parser.add_argument("--rows", type=int, default=12)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        train = parse_train(read(os.path.join(args.run_dir, "train.tsv")))
        eval_path = os.path.join(args.run_dir, "eval.tsv")
        evals = parse_eval(read(eval_path)) if os.path.exists(eval_path) else []
        summary = parse_summary(read(os.path.join(args.run_dir, "summary.txt")))
        text = report(train, evals, summary, args.note, args.vocab, args.rows)
    except SummaryError as err:
        sys.stderr.write("train_summary: " + str(err) + "\n")
        return 1
    if args.markdown:
        with open(args.markdown, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
