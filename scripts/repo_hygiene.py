#!/usr/bin/env python3
# ============================================================================
# NOTICE: Full documentation, design decisions, and fix history for this file
# live in docs/ci-and-workflows.md, section "repo_hygiene.py"
# ============================================================================
"""Repository hygiene checks for security.yml. Standard library only.

Enforces the workspace conventions from the plan: no committed model weights,
bulk datasets, credentials or build artifacts, and every workspace member opts
in to the shared lint policy. Reads the tracked file list from git.

Exit code is 1 if there is at least one hard failure. The unsafe inventory is
informational.
"""
import argparse
import fnmatch
import glob
import json
import os
import re
import subprocess
import sys

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    sys.exit("repo_hygiene.py needs Python 3.11+ (tomllib)")

MAX_BYTES = 5 * 1024 * 1024

# The mdix replacements flow stores .tar.gz drops under .mdix/ and templates live
# under .github/, so neither is scanned for artifacts.
SKIP_PREFIXES = (".mdix/", ".github/")

ARTIFACT_EXTS = {
    ".safetensors", ".gguf", ".ggml", ".pt", ".pth", ".ckpt", ".onnx", ".bin",
    ".npz", ".npy", ".h5", ".hdf5", ".pkl", ".pickle", ".parquet", ".arrow",
    ".tflite", ".weights", ".tar", ".gz", ".tgz", ".zip", ".7z", ".rlib", ".rmeta",
}
SECRET_NAME_GLOBS = [
    ".env", ".env.*", "*.pem", "*.key", "*.p12", "*.pfx", "id_rsa*", "id_ed25519*",
    "credentials.json", "secrets.*", ".netrc", "*.enc",
]
SECRET_NAME_ALLOW = {".env.example"}
BUILD_DIRS = {"target", "node_modules", "__pycache__"}
RE_UNSAFE = re.compile(r"\bunsafe\b")


def tracked_files(root):
    out = subprocess.run(["git", "-C", root, "ls-files", "-z"], check=True,
                         capture_output=True).stdout.decode("utf-8", "replace")
    return [p for p in out.split("\0") if p]


def workspace_members(root):
    with open(os.path.join(root, "Cargo.toml"), "rb") as fh:
        data = tomllib.load(fh)
    members = []
    for pattern in data.get("workspace", {}).get("members", []):
        for hit in sorted(glob.glob(os.path.join(root, pattern))):
            if os.path.isfile(os.path.join(hit, "Cargo.toml")):
                members.append(os.path.relpath(hit, root).replace(os.sep, "/"))
    return members


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--json", dest="json_path", default="")
    ap.add_argument("--md", default="", help="file to append the markdown summary to")
    args = ap.parse_args()
    root = os.path.abspath(args.root)

    failures = []
    build_dir_hits = {}
    files = tracked_files(root)

    for rel in files:
        if rel.startswith(SKIP_PREFIXES):
            continue
        parts = rel.split("/")
        base = parts[-1]
        ext = os.path.splitext(base)[1].lower()
        hit_dirs = [i for i, part in enumerate(parts[:-1]) if part in BUILD_DIRS]
        if hit_dirs:
            top = "/".join(parts[: hit_dirs[0] + 1]) + "/"
            build_dir_hits[top] = build_dir_hits.get(top, 0) + 1
            continue
        if ext in ARTIFACT_EXTS:
            failures.append({"check": "weights, data or archive", "path": rel, "detail": f"extension {ext}"})
        if base not in SECRET_NAME_ALLOW and any(fnmatch.fnmatch(base, g) for g in SECRET_NAME_GLOBS):
            failures.append({"check": "secret-looking name", "path": rel, "detail": "matches a credential filename"})
        full = os.path.join(root, rel)
        try:
            size = os.path.getsize(full)
        except OSError:
            continue
        if size > MAX_BYTES:
            failures.append({"check": "large file", "path": rel,
                             "detail": f"{size / 1048576:.1f} MiB (limit {MAX_BYTES // 1048576} MiB)"})

    for top, count in sorted(build_dir_hits.items()):
        failures.append({"check": "build artifact", "path": top,
                         "detail": f"{count} tracked file(s) inside a build directory"})

    # Every member opts in to the shared lint policy, and no crate dir is left out of the workspace.
    members = workspace_members(root)
    for member in members:
        with open(os.path.join(root, member, "Cargo.toml"), "rb") as fh:
            manifest = tomllib.load(fh)
        if manifest.get("lints", {}).get("workspace") is not True:
            failures.append({"check": "workspace lints", "path": f"{member}/Cargo.toml",
                             "detail": "missing [lints] workspace = true"})
    listed = set(members)
    for top in ("crates", "apps"):
        for hit in sorted(glob.glob(os.path.join(root, top, "*", "Cargo.toml"))):
            member = os.path.relpath(os.path.dirname(hit), root).replace(os.sep, "/")
            if member not in listed:
                failures.append({"check": "orphan crate", "path": member,
                                 "detail": "has a Cargo.toml but is not in workspace members"})

    # Unsafe inventory (informational).
    unsafe_files = {}
    for rel in files:
        if not rel.endswith(".rs") or rel.startswith(SKIP_PREFIXES):
            continue
        try:
            with open(os.path.join(root, rel), encoding="utf-8", errors="replace") as fh:
                lines = fh.read().splitlines()
        except OSError:
            continue
        hits = 0
        opt_out = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("//"):
                continue
            if "allow(unsafe_code)" in stripped:
                opt_out = True
            elif RE_UNSAFE.search(stripped):
                hits += 1
        if hits or opt_out:
            unsafe_files[rel] = {"unsafe_uses": hits, "opts_out_of_lint": opt_out}

    result = {"ok": not failures, "tracked_files": len(files), "failures": failures,
              "unsafe_inventory": unsafe_files}
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)

    out = ["## midman-ai repository hygiene", "",
           f"Tracked files: {len(files)} &nbsp;|&nbsp; Workspace members: {len(members)} "
           f"&nbsp;|&nbsp; Failures: {len(failures)}", ""]
    if failures:
        out += ["| Check | Path | Detail |", "|---|---|---|"]
        out += [f"| {f['check']} | `{f['path']}` | {f['detail']} |" for f in failures[:100]]
        if len(failures) > 100:
            out.append(f"| ... | ... | {len(failures) - 100} more |")
        out.append("")
    else:
        out += ["✅ No weights, bulk data, archives, build output or credential-looking files are tracked.", ""]
    out.append("### Unsafe inventory")
    out.append("")
    if unsafe_files:
        out += ["| File | `unsafe` uses | Opts out of workspace lint |", "|---|---|---|"]
        out += [f"| `{p}` | {v['unsafe_uses']} | {'yes' if v['opts_out_of_lint'] else 'no'} |"
                for p, v in sorted(unsafe_files.items())]
    else:
        out.append("No Rust file uses `unsafe` or opts out of the `unsafe_code` lint.")
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
