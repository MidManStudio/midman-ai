#!/usr/bin/env python3
# ============================================================================
# NOTICE: Full documentation, design decisions, and fix history for this file
# live in docs/ci-and-workflows.md, section "security_summary.py"
# ============================================================================
"""Turns `cargo audit --json` and gitleaks JSON reports into a markdown job
summary for security.yml. Standard library only.

Never fails the job. The gate step in security.yml decides pass/fail from the
tool exit codes. Secret values are never printed: gitleaks runs with --redact
and this script only reads rule, file, line and commit.
"""
import argparse
import json
import sys


def load(path):
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read().strip()
        return json.loads(text) if text else None
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def audit_md(data, outcome):
    out = ["## midman-ai dependency audit (cargo audit)", ""]
    if data is None:
        out += [f"❌ No usable report was produced (step outcome: `{outcome or 'unknown'}`). "
                "Check the raw log artifact.", ""]
        return out
    db = data.get("database", {})
    deps = data.get("lockfile", {}).get("dependency-count", "?")
    vulns = data.get("vulnerabilities", {})
    warnings = data.get("warnings", {}) or {}
    out.append(f"Dependencies in Cargo.lock: {deps} &nbsp;|&nbsp; Advisories in database: "
               f"{db.get('advisory-count', '?')} &nbsp;|&nbsp; Database commit: "
               f"`{str(db.get('last-commit', '?'))[:8]}`")
    out.append("")
    count = vulns.get("count", 0)
    if count:
        out += [f"### ❌ {count} known vulnerabilit{'y' if count == 1 else 'ies'}", "",
                "| Advisory | Crate | Title | Patched in |", "|---|---|---|---|"]
        for v in vulns.get("list", []):
            adv, pkg = v.get("advisory", {}), v.get("package", {})
            patched = ", ".join(f"`{p}`" for p in v.get("versions", {}).get("patched", [])) or "none"
            out.append(f"| [{adv.get('id', '?')}](https://rustsec.org/advisories/{adv.get('id', '')}.html) | "
                       f"`{pkg.get('name', '?')} {pkg.get('version', '')}` | {adv.get('title', '')} | {patched} |")
        out.append("")
    else:
        out += ["✅ No known vulnerabilities in Cargo.lock.", ""]
    flat = [(kind, item) for kind, items in warnings.items() for item in (items or [])]
    if flat:
        out += [f"### ⚠️ {len(flat)} warning(s) (do not fail the job)", "",
                "| Kind | Crate | Advisory |", "|---|---|---|"]
        for kind, item in flat:
            pkg, adv = item.get("package", {}), item.get("advisory") or {}
            out.append(f"| {kind} | `{pkg.get('name', '?')} {pkg.get('version', '')}` | "
                       f"{adv.get('id', '')} {adv.get('title', '')} |")
        out.append("")
    return out


def gitleaks_md(data, outcome):
    out = ["## midman-ai secret scan (gitleaks, full git history)", ""]
    if data is None:
        if outcome == "success":
            out += ["✅ No secrets found in any commit.", ""]
        else:
            out += [f"❌ No usable report was produced (step outcome: `{outcome or 'unknown'}`). "
                    "Check the raw log artifact.", ""]
        return out
    if not data:
        out += ["✅ No secrets found in any commit.", ""]
        return out
    out += [f"### ❌ {len(data)} potential secret(s)", "",
            "Values are redacted. Rotate anything real, then remove it from history.", "",
            "| Rule | Location | Commit | Date |", "|---|---|---|---|"]
    for f in data[:100]:
        out.append(f"| `{f.get('RuleID', '?')}` | `{f.get('File', '?')}:{f.get('StartLine', '?')}` | "
                   f"`{str(f.get('Commit', ''))[:8]}` | {str(f.get('Date', ''))[:10]} |")
    if len(data) > 100:
        out.append(f"| ... | {len(data) - 100} more | | |")
    out.append("")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit", help="path to cargo audit --json output")
    ap.add_argument("--audit-outcome", default="")
    ap.add_argument("--gitleaks", help="path to gitleaks JSON report")
    ap.add_argument("--gitleaks-outcome", default="")
    ap.add_argument("--md", default="", help="file to append the markdown summary to")
    args = ap.parse_args()

    out = []
    if args.audit is not None:
        out += audit_md(load(args.audit), args.audit_outcome)
    if args.gitleaks is not None:
        out += gitleaks_md(load(args.gitleaks), args.gitleaks_outcome)
    text = "\n".join(out) + "\n"
    if args.md:
        with open(args.md, "a", encoding="utf-8") as fh:
            fh.write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
