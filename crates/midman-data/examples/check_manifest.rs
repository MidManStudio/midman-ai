// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-data.md, section "examples/check_manifest.rs"
// ============================================================================
//! Reads a manifest the way training does and applies the license policy to it.
//!
//! ```text
//! cargo run --release -p midman-data --example check_manifest -- MANIFEST [--accepted FILE] [--require-license]
//! ```
//!
//! The manifest is read with [`Manifest::load`], so a file that is not valid is refused with
//! the line that is wrong. Every source is then judged by [`LicensePolicy`]: permissive plus
//! weak copyleft, and a source whose listing says it has no license (`NONE`) is accepted too,
//! which is the project's rule. `--require-license` turns that off, so that `NONE` is refused
//! like any license that is not allowed. A license that exists but could not be identified
//! (`NOASSERTION`) is refused either way. The report on standard output lists each source as `accepted` or
//! `rejected` with the reason, then the totals by category. With `--accepted FILE` the ids of
//! the accepted sources are written to FILE, one per line, in manifest order: that is the
//! list `scripts/ingest.py fetch` takes. The exit status is 0 when the manifest is valid,
//! whatever the policy decided.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use midman_data::{LicensePolicy, Manifest};

const USAGE: &str = "usage: check_manifest MANIFEST [--accepted FILE] [--require-license]";

struct Args {
    manifest: PathBuf,
    accepted: Option<PathBuf>,
    require_license: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut args = std::env::args().skip(1);
    let mut manifest = None;
    let mut accepted = None;
    let mut require_license = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--require-license" => require_license = true,
            "--accepted" => {
                let value =
                    args.next().ok_or_else(|| format!("--accepted needs a value\n{USAGE}"))?;
                accepted = Some(PathBuf::from(value));
            }
            other if other.starts_with("--") => {
                return Err(format!("unknown argument `{other}`\n{USAGE}"));
            }
            other => {
                if manifest.replace(PathBuf::from(other)).is_some() {
                    return Err(format!("only one manifest can be given\n{USAGE}"));
                }
            }
        }
    }
    Ok(Args { manifest: manifest.ok_or_else(|| USAGE.to_string())?, accepted, require_license })
}

fn run(args: &Args) -> Result<(), String> {
    let manifest = Manifest::load(&args.manifest).map_err(|e| e.to_string())?;
    let mut policy = LicensePolicy::permissive_and_weak_copyleft();
    if !args.require_license {
        policy = policy.allow_unlicensed();
    }
    let partition = manifest.partition(&policy);

    for source in &partition.accepted {
        println!("accepted\t{}\t{}\t{}", source.id, source.category, source.license);
    }
    for refused in &partition.rejected {
        println!(
            "rejected\t{}\t{}\t{}\t{}",
            refused.source.id, refused.source.category, refused.source.license, refused.reason
        );
    }

    let mut by_category: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for source in &partition.accepted {
        by_category.entry(source.category.as_str()).or_default().0 += 1;
    }
    for refused in &partition.rejected {
        by_category.entry(refused.source.category.as_str()).or_default().1 += 1;
    }
    let unlicensed = partition
        .accepted
        .iter()
        .filter(|source| source.license.eq_ignore_ascii_case("none"))
        .count();
    eprintln!(
        "{} sources: {} accepted, {} rejected",
        manifest.len(),
        partition.accepted.len(),
        partition.rejected.len()
    );
    if unlicensed > 0 {
        eprintln!(
            "  {unlicensed} of the accepted have no license (accepted by the project's rule)"
        );
    }
    for (category, (accepted, rejected)) in &by_category {
        eprintln!("  {category}: {accepted} accepted, {rejected} rejected");
    }

    if let Some(path) = &args.accepted {
        let mut text = String::new();
        for source in &partition.accepted {
            text.push_str(&source.id);
            text.push('\n');
        }
        fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))?;
    }
    Ok(())
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::from(2);
        }
    };
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("check_manifest: {message}");
            ExitCode::FAILURE
        }
    }
}
