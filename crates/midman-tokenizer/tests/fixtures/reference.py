#!/usr/bin/env python3
"""The independent reference for midman-tokenizer, and the script that writes its fixtures.

Everything here is written from the rules in `docs/midman-tokenizer.md` and shares no code
with the Rust implementation: the pre-tokenizer is one regular expression per setting,
and the trainer is the textbook loop that recounts every pair at every step. The fixtures
it writes are what `tests/reference_oracle.rs` checks the Rust code against.

    pip install regex
    python3 reference.py

Run it from this directory. The committed fixtures were made with `regex` 2026.9.29 on
Python 3.12. Both sides use Unicode properties, so a different Unicode version could in
principle move a character between classes; the fixtures use only long-stable characters.
"""
import itertools
import os
import random

import regex

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_PIECE = 128


def build_pattern(cfg):
    ansi, underscore, prefix, digits, punct_newlines, whitespace = cfg
    word = r"[\p{Alphabetic}_]" if underscore else r"\p{Alphabetic}"
    excluded = r"\p{White_Space}\p{Alphabetic}0-9" + ("_" if underscore else "") + (r"\x1b" if ansi else "")
    parts = []
    if ansi:
        parts.append(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b")
    if prefix == "space":
        parts.append(" ?" + word + "+")
    else:
        parts.append(r"[^\r\n\p{Alphabetic}0-9" + ("_" if underscore else "") + r"]?" + word + "+")
    quantifier = "+" if digits == 0 else "{1,%d}" % digits
    parts.append((" ?" if prefix == "space" else "") + "[0-9]" + quantifier)
    parts.append(" ?[^" + excluded + "]+" + (r"[\r\n]*" if punct_newlines else ""))
    if whitespace == "cl100k":
        parts.append(r"\p{White_Space}*[\r\n]+")
    parts.append(r"\p{White_Space}+(?!\P{White_Space})")
    parts.append(r"\p{White_Space}+")
    return regex.compile("|".join(parts))


_patterns = {}


def cap(piece):
    """Cut a match into chunks of at most MAX_PIECE bytes, moving each cut back by up to
    three bytes so it does not fall inside a multi-byte character."""
    out = []
    while len(piece) > MAX_PIECE:
        end = MAX_PIECE
        cut = end
        while end - cut < 3 and (piece[cut] & 0xC0) == 0x80:
            cut -= 1
        if (piece[cut] & 0xC0) != 0x80:
            end = cut
        out.append(piece[:end])
        piece = piece[end:]
    if piece:
        out.append(piece)
    return out


def pieces(cfg, text):
    pattern = _patterns.setdefault(cfg, build_pattern(cfg))
    out, covered = [], 0
    for m in pattern.finditer(text):
        assert m.start() == covered, "the pattern left a gap"
        covered = m.end()
        out.extend(cap(m.group().encode("utf-8")))
    assert covered == len(text), "the pattern stopped early"
    return out


def apply_merge(symbols, pair, new_id):
    out, i = [], 0
    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out


def train(cfg, docs, n_merges, min_frequency):
    counts = {}
    for doc in docs:
        for piece in pieces(cfg, doc):
            counts[piece] = counts.get(piece, 0) + 1
    words = {piece: list(piece) for piece in counts}
    merges = []
    while len(merges) < n_merges:
        stats = {}
        for piece, symbols in words.items():
            for a, b in zip(symbols, symbols[1:]):
                stats[(a, b)] = stats.get((a, b), 0) + counts[piece]
        if not stats:
            break
        # Highest count; among ties the smaller pair.
        pair, count = max(stats.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
        if count < min_frequency:
            break
        new_id = 256 + len(merges)
        merges.append(pair)
        for piece in words:
            words[piece] = apply_merge(words[piece], pair, new_id)
    return merges


def encode(cfg, merges, text):
    ids = []
    for piece in pieces(cfg, text):
        symbols = list(piece)
        for rank, pair in enumerate(merges):
            symbols = apply_merge(symbols, pair, 256 + rank)
        ids.extend(symbols)
    return ids


def cfg_line(cfg):
    ansi, underscore, prefix, digits, punct_newlines, whitespace = cfg
    return f"{int(ansi)},{int(underscore)},{prefix},{digits},{int(punct_newlines)},{whitespace}"


# ---- fixtures -------------------------------------------------------------------------

PRETOKENIZE_CONFIGS = [
    (False, False, "space", 0, False, "gpt2"),
    (False, False, "any", 3, True, "cl100k"),
    (True, True, "any", 3, True, "cl100k"),
    (True, True, "space", 1, False, "gpt2"),
    (True, False, "any", 0, True, "gpt2"),
    (False, True, "space", 3, True, "cl100k"),
    (True, True, "any", 1, False, "gpt2"),
    (False, False, "any", 0, False, "cl100k"),
]

EDGE_CASES = [
    "", " ", "  ", "\n", "\r\n", "\t", "a", "_", "0", "\x1b", "x\x1b", "\x1b[", "\x1b[0m", "\x1b[1;32mok\x1b[0m",
    "\x1b[?25h", "\x1b[2 q", "\x1b[31", "\x1b]0;title\x07", "a  b", "a  \n  b", "  \n\n    x", "x = 12345;\n",
    "snake_case _private __init__", "/usr/local/bin --release -v $HOME", "fn main() {\n    let x = 1;\n}\n",
    "héllo wörld 日本語 text", "a\u00a0b\u3000c", "a\u2028b\u0085c", "x\u0301y", "\u0663\u00b2\u2167\uff11",
    "ok \U0001F44D\U0001F44D", "tabs\t\tand\x0b\x0cforms", "x\x0b;\x0c;", "end of line   \n", "a.b(c)::d->e", "{\n}\n;\n)\n", "123 4567 89",
]
LONG_CASES = ["a" * 300, " " * 400, "=" * 250, "é" * 100, "x" * 127 + "é" * 40, "9" * 300, "\n" * 200,
              "a" * 125 + "\U00020000" * 40, "a" * 126 + "\U00020000" * 40]
ATOMS = ["a", "b", "Z", "_", "0", "7", " ", "  ", "\t", "\n", "\r\n", ".", ",", ";", "(", ")", "{", "}", "/", "-", "=",
         '"', "$", "::", "->", "\x1b[0m", "\x1b[1;32m", "\x1b", "é", "日", "→", "\U0001F44D", "\u00a0", "snake_case", "Foo", "1234"]


def write_pretokenize():
    rng = random.Random(20261018)
    lines = []
    for index, cfg in enumerate(PRETOKENIZE_CONFIGS):
        cases = list(EDGE_CASES)
        if index in (0, 2):
            cases += LONG_CASES
        cases += ["".join(rng.choice(ATOMS) for _ in range(rng.randint(1, 25))) for _ in range(12)]
        for text in cases:
            out = "|".join(p.hex() for p in pieces(cfg, text))
            lines.append(f"{cfg_line(cfg)}\t{text.encode('utf-8').hex()}\t{out}\n")
    with open(os.path.join(HERE, "pretokenize_golden.txt"), "w") as f:
        f.writelines(lines)
    print("pretokenize_golden.txt:", len(lines), "cases")


BPE_CONFIGS = {
    "a": (False, False, "space", 0, False, "gpt2"),
    "b": (True, True, "any", 3, True, "cl100k"),
}
BPE_TESTS = [
    "fn main() {\n    println!(\"hello\");\n}\n",
    "\x1b[1m\x1b[32m   Compiling\x1b[0m demo v0.2.0 (/home/dev/demo)\n",
    "$ git diff --stat\n src/lib.rs | 4 ++--\n",
    "store.insert(\"key\", \"value\").unwrap();",
    "naïve café — 日本語 → ok",
    "snake_case_names and CamelCaseNames and 12345 numbers",
    "",
    "   \n\n\t  mixed   whitespace  \r\n",
]


def read_corpus():
    text = open(os.path.join(HERE, "bpe_corpus.txt"), encoding="utf-8").read()
    text = text.replace("\\x1b", "\x1b")
    return text.split("\n=====\n")


def write_bpe():
    docs = read_corpus()
    for name, cfg in BPE_CONFIGS.items():
        merges = train(cfg, docs, 300, 2)
        lines = [f"config {cfg_line(cfg)}\n", "min_frequency 2\n", "requested_merges 300\n", f"merges {len(merges)}\n"]
        lines += [f"{a} {b}\n" for a, b in merges]
        lines.append("---\n")
        for text in BPE_TESTS:
            ids = encode(cfg, merges, text)
            lines.append(f"{text.encode('utf-8').hex()}\t{' '.join(map(str, ids))}\n")
        with open(os.path.join(HERE, f"bpe_golden_{name}.txt"), "w") as f:
            f.writelines(lines)
        print(f"bpe_golden_{name}.txt: {len(merges)} merges, {len(BPE_TESTS)} encodings")


if __name__ == "__main__":
    write_pretokenize()
    write_bpe()
