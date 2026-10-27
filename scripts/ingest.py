#!/usr/bin/env python3
# ============================================================================
# NOTICE: Full documentation, design decisions, and fix history for this file
# live in docs/ci-and-workflows.md, section "ingest.py"
# ============================================================================
"""Finds source repositories, pins each to a commit and writes a midman-data manifest.

Standard library only (plus the `git` program). Four commands:

    ingest.py crates   --count N --out candidates.json
    ingest.py github   --query Q --category C --count N --out candidates.json [--cargo-license]
    ingest.py manifest --candidates A.json [B.json ...] --out manifest.txt [--date YYYY-MM-DD]
                       [--exclude ID ...]
    ingest.py fetch    --manifest manifest.txt --accepted ids.txt --dest DIR [--drop-git]
    ingest.py list     --dest DIR --category C [--kind code|docs] [--max-bytes N]
    ingest.py files    --manifest manifest.txt --accepted ids.txt --dest DIR --out files.tsv
                       [--kind all|code|docs] [--max-bytes N]
    ingest.py measure  --manifest manifest.txt --accepted ids.txt --dest DIR --out results.json
                       [--markdown FILE] [--report check_manifest.tsv] [--note TEXT]
    ingest.py report   --report check_manifest.tsv --markdown FILE [--note TEXT]
    ingest.py notices  --manifest manifest.txt --accepted ids.txt --dest DIR --out usable.txt
                       [--report notices.tsv] [--markdown FILE] [--reviewed FILE]

The license policy is NOT decided here. This script only records what the listing says
(crates.io for Rust, the GitHub API for the rest). The Rust example `check_manifest` reads
the manifest the way training does and applies the policy; its list of accepted ids is
what `fetch` takes. That keeps one copy of the policy, in `midman-data`.
"""
import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

CANDIDATES_SCHEMA = "midman-ingest-candidates/1"
RESULTS_SCHEMA = "midman-ingest-results/2"
# A line longer than this many bytes marks a file as possibly minified or generated.
LONG_LINE = 1000
# The notice scan reads the top-level files of a source that has no license, looking for a
# statement against use. A file is read if its name starts with one of these (any case).
NOTICE_PREFIXES = (
    "readme", "license", "licence", "copying", "notice", "terms", "legal", "disclaimer",
    "copyright",
)
MAX_NOTICE_BYTES = 200_000
MAX_NOTICE_HITS_SHOWN = 3
TOP_EXTENSIONS = 12
MANIFEST_HEADER = "midman-data-manifest 1"
USER_AGENT = "midman-ai-ingest (https://github.com/MidManStudio/midman-ai)"
CRATES_API = "https://crates.io/api/v1"
GITHUB_API = "https://api.github.com"
# crates.io asks crawlers for at most one request a second.
CRATES_DELAY = 1.1

# What a manifest id may hold, as the Rust reader checks it.
ID_RE = re.compile(r"^[A-Za-z0-9._/-]{1,128}$")
REPO_RE = re.compile(
    r"^(?:https?://|git://|git\+https://)?(?:www\.)?github\.com[/:]"
    r"([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?(?:[/#?].*)?$"
)
COMMIT_RE = re.compile(r"^[0-9a-f]{40}([0-9a-f]{24})?$")

# Files taken from a fetched source, by category: plain suffixes, plus whole names.
# This is a first cut, not a quality filter: see "Not built yet" in docs/midman-data.md.
CATEGORY_FILES = {
    "rust": {"suffixes": (".rs",), "names": ()},
    "shell": {
        "suffixes": (".sh", ".bash", ".zsh", ".fish", ".bats"),
        "names": ("Makefile", "makefile", "GNUmakefile", "Dockerfile"),
    },
}
# Documentation is taken as a second kind of file from every source, whatever its category.
# Release notes are left out: they are long, repetitive and say little about the code.
DOC_SUFFIXES = (".md", ".markdown", ".rst")
DOC_SKIP_PREFIXES = ("CHANGELOG", "CHANGES", "HISTORY", "NEWS", "RELEASE")
# A shell script with no extension is recognised by its first line.
SHELL_SHEBANG = re.compile(
    rb"^#!\s*(?:/usr/bin/env\s+(?:-S\s+)?|/\S*/)?(?:sh|bash|zsh|dash|ksh|fish)\b"
)
SKIP_DIRS = {".git", "target", "node_modules", "vendor", "third_party", "build", "dist"}


class IngestError(Exception):
    """A problem the person running the script can act on."""


# ---------------------------------------------------------------------------
# Pure helpers (tested)
# ---------------------------------------------------------------------------


def github_repo(url):
    """Returns (owner, repo) for any GitHub URL spelling, or None if it is not GitHub."""
    if not url:
        return None
    match = REPO_RE.match(url.strip())
    if not match:
        return None
    owner, repo = match.groups()
    if not repo:
        return None
    return owner, repo


def normalize_license(text):
    """The license string as an SPDX expression, or None when the listing gave none.

    Cargo's older spelling `MIT/Apache-2.0` (also written `MIT / Apache-2.0`, as crates.io
    has it for `fnv`) means `MIT OR Apache-2.0`; it is rewritten here because the Rust reader
    only reads SPDX. Nothing else is changed: a name that is not SPDX stays as it is, and
    `manifest` records it as unidentified (see `is_spdx_shaped`).
    """
    if text is None:
        return None
    text = " ".join(str(text).split())
    if not text:
        return None
    if "/" in text and "(" not in text and not re.search(r"\b(AND|OR|WITH)\b", text):
        text = " OR ".join(part.strip() for part in text.split("/") if part.strip())
    return text


def merge_licenses(licenses):
    """One expression for a repository that holds several crates.

    The crates of one repository usually share a license. If they do not, every license
    has to be acceptable, so the expressions are joined with AND.
    """
    seen = []
    for lic in licenses:
        if lic and lic not in seen:
            seen.append(lic)
    if not seen:
        return None
    if len(seen) == 1:
        return seen[0]
    parts = []
    for lic in seen:
        wrapped = " OR " in lic or " AND " in lic
        parts.append("(" + lic + ")" if wrapped else lic)
    return " AND ".join(parts)


_SPDX_TOKEN = re.compile(r"\s*(\(|\)|[A-Za-z0-9.+:_-]+)")


def is_spdx_shaped(text):
    """Whether `text` has the shape of an SPDX license expression.

    It checks the grammar only (`id`, `id WITH id`, joined by `AND` or `OR`, in brackets as
    wanted; the operators in capitals), not that an id exists. The Rust reader refuses a whole
    manifest for one expression it cannot read, so a license text that fails this check is
    never written into one.
    """
    tokens = []
    position = 0
    text = text.strip()
    while position < len(text):
        match = _SPDX_TOKEN.match(text, position)
        if match is None:
            return False
        tokens.append(match.group(1))
        position = match.end()
    keywords = ("AND", "OR", "WITH")

    def term(at):
        if at >= len(tokens):
            return None
        if tokens[at] == "(":
            end = expression(at + 1)
            if end is None or end >= len(tokens) or tokens[end] != ")":
                return None
            return end + 1
        if tokens[at] in keywords or tokens[at] == ")":
            return None
        at += 1
        if at < len(tokens) and tokens[at] == "WITH":
            if at + 1 >= len(tokens) or tokens[at + 1] in keywords + ("(", ")"):
                return None
            at += 2
        return at

    def expression(at):
        at = term(at)
        while at is not None and at < len(tokens) and tokens[at] in ("AND", "OR"):
            at = term(at + 1)
        return at

    return expression(0) == len(tokens)


def cargo_license(text):
    """The license a root `Cargo.toml` declares, as an SPDX expression, or None.

    Looks at `[package] license`, then `[workspace.package] license` (a package that says
    `license.workspace = true` takes the second). A `license-file` is not a license id and
    gives None, as does text that is not TOML (for example an error page).
    """
    try:
        import tomllib
    except ImportError:
        raise IngestError("reading Cargo.toml needs Python 3.11 or newer")
    try:
        doc = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return None
    workspace = doc.get("workspace")
    for table in (doc.get("package"), workspace.get("package") if isinstance(workspace, dict) else None):
        if isinstance(table, dict) and isinstance(table.get("license"), str):
            return normalize_license(table["license"])
    return None


def parse_crates_page(page):
    """Reads one crates.io listing page into plain records (the data is untrusted)."""
    out = []
    for crate in page.get("crates", []):
        name = crate.get("name") or crate.get("id")
        if not isinstance(name, str) or not name:
            continue
        version = crate.get("max_stable_version") or crate.get("newest_version")
        out.append(
            {
                "name": name,
                "repository": crate.get("repository"),
                "downloads": int(crate.get("downloads") or 0),
                "version": version,
            }
        )
    return out


def parse_crate_version(doc):
    """The license of one crate version, from the version endpoint."""
    version = doc.get("version") or {}
    return normalize_license(version.get("license")), bool(version.get("yanked"))


def build_crate_candidates(crates, licenses):
    """Groups crates by repository. `licenses` maps crate name to its license (or None)."""
    by_repo = {}
    order = []
    for crate in crates:
        repo = github_repo(crate["repository"])
        if repo is None:
            continue
        key = (repo[0] + "/" + repo[1]).lower()
        if key not in by_repo:
            by_repo[key] = {
                "id": repo[0] + "/" + repo[1],
                "url": "https://github.com/" + repo[0] + "/" + repo[1],
                "category": "rust",
                "origin": "crates.io",
                "crates": [],
                "downloads": 0,
                "licenses": [],
            }
            order.append(key)
        entry = by_repo[key]
        entry["crates"].append(crate["name"])
        entry["downloads"] += crate["downloads"]
        entry["licenses"].append(licenses.get(crate["name"]))
    out = []
    for key in order:
        entry = by_repo[key]
        entry["license"] = merge_licenses(entry.pop("licenses")) or "NOASSERTION"
        out.append(entry)
    return out


def parse_github_search(doc, category):
    """Reads a GitHub repository search result into candidates. Forks are skipped."""
    out = []
    for item in doc.get("items", []):
        if item.get("fork"):
            continue
        full = item.get("full_name")
        if not isinstance(full, str) or not ID_RE.match(full):
            continue
        # GitHub gives `license: null` when it found no license at all, and an object with
        # the id NOASSERTION when there is a license file it could not identify. The first is
        # recorded as NONE, the second as NOASSERTION: they are not the same thing.
        listed = item.get("license")
        if listed is None:
            license_text = "NONE"
        else:
            license_text = normalize_license(listed.get("spdx_id")) or "NOASSERTION"
        out.append(
            {
                "id": full,
                "url": "https://github.com/" + full,
                "category": category,
                "origin": "github-search",
                "stars": int(item.get("stargazers_count") or 0),
                "license": license_text,
            }
        )
    return out


def parse_ls_remote(output):
    """The commit id on the first line of `git ls-remote URL HEAD`, or None."""
    for line in output.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == "HEAD" and COMMIT_RE.match(parts[0]):
            return parts[0]
    return None


def drop_same_commit(sources):
    """Removes a source whose commit an earlier source already has.

    A repository that was renamed or moved answers under both names with the same history,
    so the same HEAD commit means the same repository. Returns (kept, dropped) where
    `dropped` is a list of (id, id of the source it duplicates).
    """
    first = {}
    kept = []
    dropped = []
    for source in sources:
        owner = first.get(source["revision"])
        if owner is not None:
            dropped.append((source["id"], owner))
            continue
        first[source["revision"]] = source["id"]
        kept.append(source)
    return kept, dropped


def manifest_text(sources):
    """The manifest in the exact text form midman-data reads (fixed key order)."""
    keys = ("id", "url", "revision", "license", "fetched", "category")
    out = [MANIFEST_HEADER + "\n"]
    for source in sources:
        out.append("\n")
        for key in keys:
            out.append(key + ": " + source[key] + "\n")
    return "".join(out)


def read_manifest(text):
    """Reads a manifest written by `manifest_text` into a list of dicts (strict)."""
    lines = text.split("\n")
    if lines[0] != MANIFEST_HEADER or lines[-1] != "":
        raise IngestError("not a midman-data manifest (header or final newline)")
    body = lines[1:-1]
    keys = ("id", "url", "revision", "license", "fetched", "category")
    if len(body) % 7 != 0:
        raise IngestError("manifest records are not 7 lines each")
    out = []
    for at in range(0, len(body), 7):
        if body[at] != "":
            raise IngestError("expected a blank line at manifest line " + str(at + 2))
        record = {}
        for offset, key in enumerate(keys, start=1):
            prefix = key + ": "
            line = body[at + offset]
            if not line.startswith(prefix):
                raise IngestError("expected `" + prefix + "` at manifest line " + str(at + offset + 2))
            record[key] = line[len(prefix):]
        out.append(record)
    return out


def has_shell_shebang(path):
    """Whether the file starts with a `#!` line that names a shell."""
    try:
        with open(path, "rb") as handle:
            return SHELL_SHEBANG.match(handle.read(200)) is not None
    except OSError:
        return False


def select_files(dest, category, max_bytes, kind="code"):
    """Paths of files of one kind under `dest`, sorted, no directory in SKIP_DIRS.

    `kind` is `code` (the files of this category: by suffix or name, and for shell also an
    extensionless file that starts with a shell `#!` line) or `docs` (Markdown and
    reStructuredText, without release notes; the category does not matter).
    """
    if kind not in ("code", "docs"):
        raise IngestError("kind must be `code` or `docs`, not `" + str(kind) + "`")
    rule = None
    if kind == "code":
        rule = CATEGORY_FILES.get(category)
        if rule is None:
            raise IngestError("no file rule for category `" + category + "`")
    found = []
    for root, dirs, files in os.walk(dest):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(files):
            by_shebang = False
            if kind == "code":
                wanted = name.endswith(rule["suffixes"]) or name in rule["names"]
                by_shebang = category == "shell" and not wanted and os.path.splitext(name)[1] == ""
            else:
                wanted = name.lower().endswith(DOC_SUFFIXES) and not name.upper().startswith(
                    DOC_SKIP_PREFIXES
                )
            if not (wanted or by_shebang):
                continue
            path = os.path.join(root, name)
            if " " in path or os.path.islink(path):
                continue
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            if not 0 < size <= max_bytes:
                continue
            if by_shebang and not has_shell_shebang(path):
                continue
            found.append(path)
    return sorted(found)


def source_dir(dest, ident):
    """Where `fetch` puts a source: DIR/<owner>__<repo>."""
    return os.path.join(dest, ident.replace("/", "__"))


def file_stats(path):
    """(bytes, is_utf8, has_long_line) for one file, or None if it cannot be read."""
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except OSError:
        return None
    try:
        data.decode("utf-8")
        utf8 = True
    except UnicodeDecodeError:
        utf8 = False
    long_line = any(len(line) > LONG_LINE for line in data.split(b"\n"))
    return len(data), utf8, long_line


def extension_of(name):
    """The key a file is counted under in the extension table."""
    if name in ("Makefile", "makefile", "GNUmakefile", "Dockerfile"):
        return name
    ext = os.path.splitext(name)[1]
    return ext.lower() if ext else "(none)"


def extension_table(root):
    """Files and bytes by extension over everything under `root` outside SKIP_DIRS."""
    table = {}
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in files:
            path = os.path.join(base, name)
            if os.path.islink(path) or name == ".midman-revision":
                continue
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            entry = table.setdefault(extension_of(name), [0, 0])
            entry[0] += 1
            entry[1] += size
    return table


def count_files(paths):
    """Files, bytes, files that are not UTF-8 and files with a very long line.

    A file that is not UTF-8 is counted apart and not in the files and bytes.
    """
    files = bytes_ = not_utf8 = long_line_files = 0
    for path in paths:
        stats = file_stats(path)
        if stats is None:
            continue
        size, utf8, long_line = stats
        if not utf8:
            not_utf8 += 1
            continue
        files += 1
        bytes_ += size
        if long_line:
            long_line_files += 1
    return files, bytes_, not_utf8, long_line_files


def measure_source(root, category, max_bytes):
    """What `files` would take from one fetched source, counted: code, then docs."""
    files, bytes_, not_utf8, long_line_files = count_files(
        select_files(root, category, max_bytes, "code")
    )
    docs_files, docs_bytes, docs_not_utf8, _ = count_files(
        select_files(root, category, max_bytes, "docs")
    )
    return {
        "files": files,
        "bytes": bytes_,
        "not_utf8": not_utf8,
        "long_line_files": long_line_files,
        "docs_files": docs_files,
        "docs_bytes": docs_bytes,
        "docs_not_utf8": docs_not_utf8,
    }


def summarize_measurements(sources, extensions):
    """The results document: per source, per category and the extension table."""
    categories = {}
    for source in sources:
        entry = categories.setdefault(
            source["category"],
            {"sources": 0, "sources_without_files": 0, "files": 0, "bytes": 0,
             "not_utf8": 0, "long_line_files": 0, "docs_files": 0, "docs_bytes": 0,
             "docs_not_utf8": 0},
        )
        entry["sources"] += 1
        if source["files"] == 0:
            entry["sources_without_files"] += 1
        for key in ("files", "bytes", "not_utf8", "long_line_files", "docs_files", "docs_bytes",
                    "docs_not_utf8"):
            entry[key] += source.get(key, 0)
    ranked = sorted(extensions.items(), key=lambda item: (-item[1][1], item[0]))
    all_bytes = sum(v[1] for v in extensions.values())
    return {
        "schema": RESULTS_SCHEMA,
        "sources": sources,
        "categories": categories,
        "all_files": sum(v[0] for v in extensions.values()),
        "all_bytes": all_bytes,
        "extensions": [
            {"extension": ext, "files": v[0], "bytes": v[1]} for ext, v in ranked[:TOP_EXTENSIONS]
        ],
    }


def _compile(pattern):
    return re.compile(pattern, re.IGNORECASE)


# Each entry: what the match is called in the report, and the pattern. They look for a
# statement that rights are kept or use is limited; they are deliberately wide, because a
# false alarm costs one source and a missed one is the thing the scan is for.
NOTICE_PATTERNS = [
    ("all rights reserved", _compile(r"\ball\s+rights\s+reserved\b")),
    ("proprietary", _compile(r"\bproprietary\b")),
    ("non-commercial", _compile(r"\bnon[- ]?commercial\b")),
    (
        "personal or educational use only",
        _compile(
            r"\b(personal|private|educational|research|non-?profit)\s+(use|purposes?)\s+only\b"
            r"|\bfor\s+(personal|educational|private)\s+use\b"
        ),
    ),
    (
        "permission",
        _compile(
            r"\b(without|no)\s+(\w+\s+){0,2}permission\b"
            r"|\bpermission\s+(is|has)\s+not\s+(been\s+)?granted\b"
        ),
    ),
    (
        "may not be used or copied",
        _compile(
            r"\b(may|must|shall|can)\s*(not|n't)\s+be\s+(used|copied|reproduced|redistributed"
            r"|distributed|modified|sold|shared)\b"
            r"|\bdo\s+not\s+(copy|redistribute|reproduce|reupload|re-upload|steal|reuse|use\s+this)\b"
            r"|\b(not|never)\s+(allowed|permitted)\s+to\s+(copy|use|reuse|redistribute|modify)\b"
        ),
    ),
    (
        "no license granted",
        _compile(r"\bno\s+licen[sc]e\b[^.\n]{0,40}\b(granted|given|provided)\b|\bnot\s+licensed\b"),
    ),
    (
        "against AI training",
        _compile(
            r"\b(no|not|never|prohibit\w*|forbid\w*|forbidden|disallow\w*|banned?)\b[^.\n]{0,60}"
            r"\b(ai|a\.i\.|machine[- ]learning|llms?|language\s+models?|generative)\b[^.\n]{0,40}"
            r"\b(train\w*|scrap\w*|dataset\w*|corpus)\b"
            r"|\b(train\w*|scrap\w*)\b[^.\n]{0,40}\b(ai|machine[- ]learning|llms?|language\s+models?)\b"
            r"[^.\n]{0,40}\b(prohibit\w*|forbid\w*|forbidden|disallow\w*|not\s+(allowed|permitted))\b"
            r"|\bdo\s+not\s+(train|scrape)\b"
        ),
    ),
]


def notice_files(root):
    """The top-level files of a source that the notice scan reads, sorted by name."""
    found = []
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return found
    for name in names:
        if not name.lower().startswith(NOTICE_PREFIXES):
            continue
        path = os.path.join(root, name)
        if os.path.islink(path) or not os.path.isfile(path):
            continue
        try:
            if os.path.getsize(path) > MAX_NOTICE_BYTES:
                continue
        except OSError:
            continue
        found.append(path)
    return found


def scan_notice_text(text):
    """(line number, label, the line) for every line that matches a pattern, once per label."""
    hits = []
    for number, line in enumerate(text.splitlines(), start=1):
        for label, pattern in NOTICE_PATTERNS:
            if pattern.search(line):
                hits.append((number, label, " ".join(line.split())[:160]))
    return hits


def scan_source(root):
    """The notice scan of one fetched source: a list of hit dicts, file by file."""
    hits = []
    for path in notice_files(root):
        try:
            with open(path, "rb") as handle:
                text = handle.read().decode("utf-8", errors="replace")
        except OSError:
            continue
        for number, label, line in scan_notice_text(text):
            hits.append(
                {"file": os.path.basename(path), "line": number, "label": label, "text": line}
            )
    return hits


def notices_markdown(scanned, flagged, kept, note=""):
    """The result of the notice scan as markdown.

    `scanned` is the list of source ids that were scanned, `flagged` maps an id to its hits
    (left out) and `kept` maps an id to its hits (flagged, but listed as reviewed).
    """
    lines = ["## midman-ai ingest: license notices", ""]
    if note:
        lines += [note, ""]
    if not scanned:
        lines += ["No accepted source lacks a license, so nothing was scanned.", ""]
        return "\n".join(lines) + "\n"
    lines += [
        "Sources with no license that were scanned: %d. Left out because their own files say"
        " otherwise: %d. Kept after review: %d." % (len(scanned), len(flagged), len(kept)),
        "",
    ]
    for title, table in (("Left out", flagged), ("Kept after review", kept)):
        if not table:
            continue
        lines += ["### " + title, "", "| Source | File | Line | Found | Text |", "|---|---|---:|---|---|"]
        for ident in sorted(table):
            hits = table[ident]
            for hit in hits[:MAX_NOTICE_HITS_SHOWN]:
                lines.append(
                    "| `%s` | %s | %d | %s | %s |"
                    % (ident, cell(hit["file"]), hit["line"], hit["label"], cell(hit["text"]))
                )
            if len(hits) > MAX_NOTICE_HITS_SHOWN:
                lines.append(
                    "| `%s` | | | | and %d more |" % (ident, len(hits) - MAX_NOTICE_HITS_SHOWN)
                )
        lines.append("")
    return "\n".join(lines) + "\n"


def parse_policy_report(text):
    """Reads the standard output of `check_manifest` (tab-separated, one source per line)."""
    accepted = []
    rejected = []
    for line in text.splitlines():
        parts = line.split("\t")
        if parts[0] == "accepted" and len(parts) == 4:
            accepted.append({"id": parts[1], "category": parts[2], "license": parts[3]})
        elif parts[0] == "rejected" and len(parts) == 5:
            rejected.append(
                {"id": parts[1], "category": parts[2], "license": parts[3], "reason": parts[4]}
            )
    return {"accepted": accepted, "rejected": rejected}


def cell(text):
    """Text safe inside a markdown table cell."""
    return text.replace("|", "\\|")


def policy_markdown(report, note=""):
    """The license policy's verdicts as markdown: counts by category, then every refusal."""
    lines = ["## midman-ai ingest: license policy", ""]
    if note:
        lines += [note, ""]
    categories = {}
    for key in ("accepted", "rejected"):
        for source in report[key]:
            categories.setdefault(source["category"], {"accepted": 0, "rejected": 0})[key] += 1
    lines += ["| Category | Accepted | Rejected |", "|---|---:|---:|"]
    for name in sorted(categories):
        lines.append("| %s | %d | %d |" % (name, categories[name]["accepted"], categories[name]["rejected"]))
    lines.append("| all | %d | %d |" % (len(report["accepted"]), len(report["rejected"])))
    lines.append("")
    unlicensed = sum(1 for source in report["accepted"] if source["license"].upper() == "NONE")
    if unlicensed:
        lines += [
            "%d of the accepted sources have no license. The project accepts those unless their"
            " own files say otherwise, which is checked after they are fetched." % unlicensed,
            "",
        ]
    if report["rejected"]:
        lines += ["### Refused", "", "| Source | Category | License | Reason |", "|---|---|---|---|"]
        for source in report["rejected"]:
            lines.append(
                "| `%s` | %s | %s | %s |"
                % (source["id"], source["category"], cell(source["license"]), cell(source["reason"]))
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def megabytes(count):
    return "%.1f" % (count / 1_000_000)


def results_markdown(results, note=""):
    """The results as a markdown summary (tables only, no claims beyond the numbers)."""
    lines = ["## midman-ai ingest: what was fetched and what would be used", ""]
    if note:
        lines += [note, ""]
    lines += [
        "| Category | Sources | Without files | Files | MB | Not UTF-8 | Long-line files | Docs files | Docs MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in sorted(results["categories"]):
        c = results["categories"][name]
        lines.append(
            "| %s | %d | %d | %d | %s | %d | %d | %d | %s |"
            % (name, c["sources"], c["sources_without_files"], c["files"],
               megabytes(c["bytes"]), c["not_utf8"], c["long_line_files"],
               c["docs_files"], megabytes(c["docs_bytes"]))
        )
    lines += [
        "",
        "Files, MB and the two columns after them count code: by suffix, by name, and for shell"
        " also scripts with a shell `#!` line. Long-line files hold a line over %d bytes"
        " (possibly minified or generated). Docs are Markdown and reStructuredText from every"
        " source, without release notes; the category column is the category of the source they"
        " come from." % LONG_LINE,
        "",
    ]
    empty = [s for s in results["sources"] if s["files"] == 0]
    if empty:
        lines += ["### Sources with no file of their category", ""]
        lines += ["- `%s` (%s)" % (s["id"], s["category"]) for s in empty]
        lines.append("")
    lines += ["### Largest sources", "", "| Source | Category | Files | MB |", "|---|---|---:|---:|"]
    biggest = sorted(results["sources"], key=lambda s: (-s["bytes"], s["id"]))[:15]
    for s in biggest:
        lines.append("| `%s` | %s | %d | %s |" % (s["id"], s["category"], s["files"], megabytes(s["bytes"])))
    docs_sources = sorted(
        (s for s in results["sources"] if s.get("docs_bytes", 0) > 0),
        key=lambda s: (-s["docs_bytes"], s["id"]),
    )[:8]
    if docs_sources:
        lines += ["", "### Where the docs are (largest 8)", "",
                  "| Source | Category | Docs files | Docs MB | Code MB |", "|---|---|---:|---:|---:|"]
        for s in docs_sources:
            lines.append(
                "| `%s` | %s | %d | %s | %s |"
                % (s["id"], s["category"], s["docs_files"], megabytes(s["docs_bytes"]),
                   megabytes(s["bytes"]))
            )
    lines += ["", "### Everything fetched, by extension (top %d by size)" % TOP_EXTENSIONS, "",
              "| Extension | Files | MB | Share |", "|---|---:|---:|---:|"]
    total = results["all_bytes"] or 1
    for e in results["extensions"]:
        lines.append("| `%s` | %d | %s | %.1f%% |" % (e["extension"], e["files"], megabytes(e["bytes"]),
                                                      100.0 * e["bytes"] / total))
    lines += ["", "All files fetched: %d, %s MB." % (results["all_files"], megabytes(results["all_bytes"])), ""]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Network and git (not unit tested: they need the network)
# ---------------------------------------------------------------------------


def get_json(url, headers=None, retries=3):
    request_headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    request_headers.update(headers or {})
    last = None
    for attempt in range(retries):
        request = urllib.request.Request(url, headers=request_headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            if err.code in (403, 429) and "github.com" in url:
                reset = err.headers.get("x-ratelimit-reset")
                remaining = err.headers.get("x-ratelimit-remaining")
                if remaining == "0" and reset:
                    when = datetime.datetime.fromtimestamp(int(reset), datetime.timezone.utc)
                    raise IngestError(
                        "GitHub rate limit used up; it resets at "
                        + when.strftime("%Y-%m-%d %H:%M:%S")
                        + " UTC. Set GITHUB_TOKEN to get a much higher limit."
                    )
            last = err
            if err.code in (429, 500, 502, 503, 504):
                time.sleep(2 * (attempt + 1))
                continue
            raise IngestError("HTTP " + str(err.code) + " for " + url)
        except (urllib.error.URLError, TimeoutError) as err:
            last = err
            time.sleep(2 * (attempt + 1))
    raise IngestError("could not fetch " + url + ": " + str(last))


def fetch_text(url, retries=3):
    """The body of a URL as text; None if it does not exist or could not be fetched.

    A 404 is final. A 429 or a server error is tried again after a wait.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as err:
            if err.code in (429, 500, 502, 503, 504):
                time.sleep(3 * (attempt + 1))
                continue
            return None
        except (urllib.error.URLError, TimeoutError):
            time.sleep(3 * (attempt + 1))
    return None


def run_git(args, cwd=None, timeout=900):
    result = subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=dict(os.environ, GIT_TERMINAL_PROMPT="0"),
    )
    if result.returncode != 0:
        raise IngestError("git " + " ".join(args[:2]) + " failed: " + result.stderr.strip()[:300])
    return result.stdout


def today():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")


def write_json(path, doc):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(doc, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_crates(args):
    repos = []
    seen_repos = set()
    crates = []
    skipped_other_hosts = 0
    page = 1
    while len(seen_repos) < args.count and page <= args.max_pages:
        doc = get_json(
            CRATES_API + "/crates?page=" + str(page) + "&per_page=100&sort=downloads"
        )
        listed = parse_crates_page(doc)
        if not listed:
            break
        for crate in listed:
            repo = github_repo(crate["repository"])
            if repo is None:
                skipped_other_hosts += 1
                continue
            key = (repo[0] + "/" + repo[1]).lower()
            if key not in seen_repos and len(seen_repos) >= args.count:
                continue
            seen_repos.add(key)
            crates.append(crate)
        page += 1
        time.sleep(CRATES_DELAY)
    licenses = {}
    for crate in crates:
        if not crate["version"]:
            licenses[crate["name"]] = None
            continue
        doc = get_json(CRATES_API + "/crates/" + crate["name"] + "/" + crate["version"])
        licenses[crate["name"]], _yanked = parse_crate_version(doc)
        time.sleep(CRATES_DELAY)
    repos = build_crate_candidates(crates, licenses)
    write_json(
        args.out,
        {
            "schema": CANDIDATES_SCHEMA,
            "created": today(),
            "origin": "crates.io by downloads",
            "candidates": repos,
        },
    )
    sys.stderr.write(
        "crates: " + str(len(crates)) + " crates in " + str(len(repos)) + " repositories; "
        + str(skipped_other_hosts) + " listed crates are not on GitHub\n"
    )


def apply_cargo_licenses(candidates, category):
    """For Rust candidates GitHub could not give a license for, read the root `Cargo.toml`.

    GitHub reports a repository with two license files (the usual `MIT OR Apache-2.0`) as
    NOASSERTION, and one with none as NONE. The maintainers' own declaration in `Cargo.toml`
    is better evidence than either, so it replaces them. A license GitHub did identify is
    left alone. Candidates changed get `license_origin: Cargo.toml`.
    """
    if category != "rust":
        sys.stderr.write("cargo-license: only used for the category `rust`, skipped\n")
        return
    replaced = unreadable = 0
    for candidate in candidates:
        if candidate["license"] not in ("NONE", "NOASSERTION"):
            continue
        text = fetch_text("https://raw.githubusercontent.com/" + candidate["id"] + "/HEAD/Cargo.toml")
        if text is None:
            unreadable += 1
            continue
        declared = cargo_license(text)
        if declared:
            candidate["license"] = declared
            candidate["license_origin"] = "Cargo.toml"
            replaced += 1
        time.sleep(0.3)
    sys.stderr.write(
        "cargo-license: %d replaced from Cargo.toml, %d Cargo.toml files could not be read\n"
        % (replaced, unreadable)
    )


def cmd_github(args):
    headers = {"X-GitHub-Api-Version": "2022-11-28"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = "Bearer " + token
    out = []
    page = 1
    while len(out) < args.count and page <= 10:
        query = urllib.parse.urlencode(
            {"q": args.query, "sort": "stars", "order": "desc", "per_page": 100, "page": page}
        )
        doc = get_json(GITHUB_API + "/search/repositories?" + query, headers)
        found = parse_github_search(doc, args.category)
        if not found and not doc.get("items"):
            break
        out.extend(found)
        page += 1
        time.sleep(2)
    out = out[: args.count]
    if args.cargo_license:
        apply_cargo_licenses(out, args.category)
    write_json(
        args.out,
        {
            "schema": CANDIDATES_SCHEMA,
            "created": today(),
            "origin": "github search: " + args.query,
            "candidates": out,
        },
    )
    sys.stderr.write("github: " + str(len(out)) + " repositories\n")


def cmd_manifest(args):
    merged = []
    seen = set()
    for path in args.candidates:
        with open(path, encoding="utf-8") as handle:
            doc = json.load(handle)
        if doc.get("schema") != CANDIDATES_SCHEMA:
            raise IngestError(path + ": not a " + CANDIDATES_SCHEMA + " file")
        for candidate in doc["candidates"]:
            key = candidate["id"].lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(candidate)
    date = args.date or today()
    excluded = {ident.lower() for ident in args.exclude}
    sources = []
    skipped = []
    notes = []
    for candidate in merged:
        if candidate["id"].lower() in excluded:
            skipped.append((candidate["id"], "excluded by --exclude"))
            continue
        if not ID_RE.match(candidate["id"]):
            skipped.append((candidate["id"], "id not allowed"))
            continue
        license_text = normalize_license(candidate["license"]) or "NOASSERTION"
        if not is_spdx_shaped(license_text):
            notes.append((candidate["id"], license_text))
            license_text = "NOASSERTION"
        try:
            out = run_git(["ls-remote", candidate["url"], "HEAD"], timeout=120)
        except (IngestError, subprocess.TimeoutExpired) as err:
            skipped.append((candidate["id"], str(err)))
            continue
        revision = parse_ls_remote(out)
        if revision is None:
            skipped.append((candidate["id"], "no HEAD commit"))
            continue
        sources.append(
            {
                "id": candidate["id"],
                "url": candidate["url"],
                "revision": revision,
                "license": license_text,
                "fetched": date,
                "category": candidate["category"],
            }
        )
    sources, dropped = drop_same_commit(sources)
    for ident, owner in dropped:
        skipped.append((ident, "same commit as " + owner))
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(manifest_text(sources))
    os.replace(tmp, args.out)
    for ident, text in notes:
        sys.stderr.write(
            "manifest: the license of " + ident + " is `" + text
            + "`, which is not an SPDX expression; recorded as NOASSERTION\n"
        )
    sys.stderr.write("manifest: " + str(len(sources)) + " sources")
    if skipped:
        sys.stderr.write(", skipped " + str(len(skipped)) + ":")
        for ident, why in skipped:
            sys.stderr.write("\n  " + ident + ": " + why)
    sys.stderr.write("\n")


def cmd_fetch(args):
    with open(args.manifest, encoding="utf-8") as handle:
        records = {r["id"]: r for r in read_manifest(handle.read())}
    with open(args.accepted, encoding="utf-8") as handle:
        wanted = [line.strip() for line in handle if line.strip()]
    os.makedirs(args.dest, exist_ok=True)
    failed = []
    for ident in wanted:
        record = records.get(ident)
        if record is None:
            raise IngestError("`" + ident + "` is not in the manifest")
        target = source_dir(args.dest, ident)
        marker = os.path.join(target, ".midman-revision")
        if os.path.exists(marker):
            with open(marker, encoding="utf-8") as handle:
                if handle.read().strip() == record["revision"]:
                    sys.stderr.write("have " + ident + "\n")
                    continue
        try:
            os.makedirs(target, exist_ok=True)
            run_git(["init", "-q"], cwd=target)
            run_git(["fetch", "-q", "--depth", "1", record["url"], record["revision"]], cwd=target)
            run_git(["checkout", "-q", "--detach", "FETCH_HEAD"], cwd=target)
            head = run_git(["rev-parse", "HEAD"], cwd=target).strip()
            if head != record["revision"]:
                raise IngestError("checked out " + head + ", manifest says " + record["revision"])
            with open(marker, "w", encoding="utf-8") as handle:
                handle.write(record["revision"] + "\n")
            if args.drop_git:
                # The commit has been checked and recorded in the marker; the history is not
                # needed, and is about a third of what a shallow checkout takes.
                shutil.rmtree(os.path.join(target, ".git"), ignore_errors=True)
            sys.stderr.write("fetched " + ident + " at " + record["revision"][:10] + "\n")
        except (IngestError, subprocess.TimeoutExpired) as err:
            failed.append((ident, str(err)))
    for ident, why in failed:
        sys.stderr.write("FAILED " + ident + ": " + why + "\n")
    if failed:
        raise IngestError(str(len(failed)) + " of " + str(len(wanted)) + " sources failed")


def cmd_measure(args):
    with open(args.manifest, encoding="utf-8") as handle:
        records = {r["id"]: r for r in read_manifest(handle.read())}
    with open(args.accepted, encoding="utf-8") as handle:
        wanted = [line.strip() for line in handle if line.strip()]
    sources = []
    extensions = {}
    for ident in wanted:
        record = records.get(ident)
        if record is None:
            raise IngestError("`" + ident + "` is not in the manifest")
        root = source_dir(args.dest, ident)
        if not os.path.isdir(root):
            raise IngestError("`" + ident + "` has not been fetched into " + args.dest)
        entry = {"id": ident, "category": record["category"], "revision": record["revision"]}
        entry.update(measure_source(root, record["category"], args.max_bytes))
        sources.append(entry)
        for ext, (count, size) in extension_table(root).items():
            slot = extensions.setdefault(ext, [0, 0])
            slot[0] += count
            slot[1] += size
    results = summarize_measurements(sources, extensions)
    write_json(args.out, results)
    if args.markdown:
        text = ""
        if args.report:
            with open(args.report, encoding="utf-8") as handle:
                text = policy_markdown(parse_policy_report(handle.read()))
        text += results_markdown(results, args.note)
        with open(args.markdown, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    for name in sorted(results["categories"]):
        c = results["categories"][name]
        sys.stderr.write(
            "measure: %s: %d sources, %d files, %s MB (%d sources without files)\n"
            % (name, c["sources"], c["files"], megabytes(c["bytes"]), c["sources_without_files"])
        )


def cmd_notices(args):
    with open(args.manifest, encoding="utf-8") as handle:
        records = {r["id"]: r for r in read_manifest(handle.read())}
    with open(args.accepted, encoding="utf-8") as handle:
        wanted = [line.strip() for line in handle if line.strip()]
    reviewed = set()
    if args.reviewed:
        with open(args.reviewed, encoding="utf-8") as handle:
            reviewed = {line.strip() for line in handle if line.strip() and not line.startswith("#")}
    usable = []
    scanned = []
    flagged = {}
    kept = {}
    for ident in wanted:
        record = records.get(ident)
        if record is None:
            raise IngestError("`" + ident + "` is not in the manifest")
        if record["license"].strip().upper() != "NONE":
            usable.append(ident)
            continue
        root = source_dir(args.dest, ident)
        if not os.path.isdir(root):
            raise IngestError("`" + ident + "` has not been fetched into " + args.dest)
        scanned.append(ident)
        hits = scan_source(root)
        if not hits:
            usable.append(ident)
        elif ident in reviewed:
            kept[ident] = hits
            usable.append(ident)
        else:
            flagged[ident] = hits
    with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
        for ident in usable:
            handle.write(ident + "\n")
    if args.report:
        with open(args.report, "w", encoding="utf-8", newline="\n") as handle:
            for status, table in (("left out", flagged), ("kept", kept)):
                for ident in sorted(table):
                    for hit in table[ident]:
                        handle.write(
                            "\t".join(
                                [status, ident, hit["file"], str(hit["line"]), hit["label"], hit["text"]]
                            ).replace("\r", " ")
                            + "\n"
                        )
    if args.markdown:
        with open(args.markdown, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(notices_markdown(scanned, flagged, kept))
    sys.stderr.write(
        "notices: %d with no license scanned, %d left out, %d kept after review; %d of %d usable\n"
        % (len(scanned), len(flagged), len(kept), len(usable), len(wanted))
    )


def cmd_report(args):
    with open(args.report, encoding="utf-8") as handle:
        report = parse_policy_report(handle.read())
    with open(args.markdown, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(policy_markdown(report, args.note))
    sys.stderr.write(
        "report: %d accepted, %d rejected\n" % (len(report["accepted"]), len(report["rejected"]))
    )


def cmd_list(args):
    for path in select_files(args.dest, args.category, args.max_bytes, args.kind):
        sys.stdout.write(path + "\n")


def cmd_files(args):
    """Writes `category<TAB>kind<TAB>source<TAB>path` for every usable file of the sources."""
    with open(args.manifest, encoding="utf-8") as handle:
        records = {r["id"]: r for r in read_manifest(handle.read())}
    with open(args.accepted, encoding="utf-8") as handle:
        wanted = [line.strip() for line in handle if line.strip()]
    kinds = ("code", "docs") if args.kind == "all" else (args.kind,)
    counts = {}
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as out:
        for ident in wanted:
            record = records.get(ident)
            if record is None:
                raise IngestError("`" + ident + "` is not in the manifest")
            root = source_dir(args.dest, ident)
            if not os.path.isdir(root):
                raise IngestError("`" + ident + "` has not been fetched into " + args.dest)
            for kind in kinds:
                for path in select_files(root, record["category"], args.max_bytes, kind):
                    stats = file_stats(path)
                    if stats is None or not stats[1]:
                        continue
                    out.write("\t".join([record["category"], kind, ident, path]) + "\n")
                    key = (record["category"], kind)
                    counts[key] = counts.get(key, 0) + 1
    os.replace(tmp, args.out)
    for (category, kind), count in sorted(counts.items()):
        sys.stderr.write("files: %s %s: %d\n" % (category, kind, count))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    crates = sub.add_parser("crates", help="top crates.io crates by downloads, grouped by repository")
    crates.add_argument("--count", type=int, default=40, help="repositories wanted")
    crates.add_argument("--max-pages", type=int, default=10)
    crates.add_argument("--out", required=True)
    crates.set_defaults(run=cmd_crates)

    github = sub.add_parser("github", help="GitHub repository search, most starred first")
    github.add_argument("--query", required=True, help='for example "language:Shell stars:>500"')
    github.add_argument("--category", required=True)
    github.add_argument("--count", type=int, default=40)
    github.add_argument("--out", required=True)
    github.add_argument(
        "--cargo-license",
        action="store_true",
        help="for category rust, take the license from the root Cargo.toml when GitHub has none",
    )
    github.set_defaults(run=cmd_github)

    manifest = sub.add_parser("manifest", help="pin candidates to a commit and write a manifest")
    manifest.add_argument("--candidates", nargs="+", required=True)
    manifest.add_argument("--out", required=True)
    manifest.add_argument("--date")
    manifest.add_argument(
        "--exclude", action="append", default=[], metavar="ID", help="leave this source id out"
    )
    manifest.set_defaults(run=cmd_manifest)

    fetch = sub.add_parser("fetch", help="fetch accepted sources at their pinned commit")
    fetch.add_argument("--manifest", required=True)
    fetch.add_argument("--accepted", required=True, help="file with one source id per line")
    fetch.add_argument("--dest", required=True)
    fetch.add_argument(
        "--drop-git", action="store_true", help="delete each source's .git once it is checked out"
    )
    fetch.set_defaults(run=cmd_fetch)

    measure = sub.add_parser("measure", help="count what fetched sources would contribute")
    measure.add_argument("--manifest", required=True)
    measure.add_argument("--accepted", required=True, help="file with one source id per line")
    measure.add_argument("--dest", required=True)
    measure.add_argument("--out", required=True, help="results JSON")
    measure.add_argument("--markdown", help="append a markdown summary to this file")
    measure.add_argument("--report", help="check_manifest output; adds the license policy section")
    measure.add_argument("--note", default="", help="a line shown at the top of the summary")
    measure.add_argument("--max-bytes", type=int, default=1024 * 1024)
    measure.set_defaults(run=cmd_measure)

    notices = sub.add_parser(
        "notices", help="look in the files of unlicensed sources for a statement against use"
    )
    notices.add_argument("--manifest", required=True)
    notices.add_argument("--accepted", required=True, help="file with one source id per line")
    notices.add_argument("--dest", required=True)
    notices.add_argument("--out", required=True, help="the accepted ids that remain usable")
    notices.add_argument("--report", help="write the matching lines here (tab separated)")
    notices.add_argument("--markdown", help="append a markdown summary to this file")
    notices.add_argument("--reviewed", help="ids to keep although the scan found something")
    notices.set_defaults(run=cmd_notices)

    report = sub.add_parser("report", help="markdown summary of check_manifest's verdicts")
    report.add_argument("--report", required=True, help="the standard output of check_manifest")
    report.add_argument("--markdown", required=True, help="append the summary to this file")
    report.add_argument("--note", default="")
    report.set_defaults(run=cmd_report)

    listing = sub.add_parser("list", help="print the paths of one category's files")
    listing.add_argument("--dest", required=True)
    listing.add_argument("--category", required=True)
    listing.add_argument("--kind", choices=("code", "docs"), default="code")
    listing.add_argument("--max-bytes", type=int, default=1024 * 1024)
    listing.set_defaults(run=cmd_list)

    files = sub.add_parser("files", help="list the usable files of the accepted sources, by kind")
    files.add_argument("--manifest", required=True)
    files.add_argument("--accepted", required=True, help="file with one source id per line")
    files.add_argument("--dest", required=True)
    files.add_argument("--out", required=True, help="tab separated: category, kind, source, path")
    files.add_argument("--kind", choices=("all", "code", "docs"), default="all")
    files.add_argument("--max-bytes", type=int, default=1024 * 1024)
    files.set_defaults(run=cmd_files)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.run(args)
    except IngestError as err:
        sys.stderr.write("ingest: " + str(err) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
