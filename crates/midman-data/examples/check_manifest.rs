// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-data.md, section "examples/check_manifest.rs"
// ============================================================================
//! Reads a manifest the way training does and applies the license policy to it.
//!
//! ```text
//! cargo run --release -p midman-data --example check_manifest -- MANIFEST [--accepted FILE]
//! ```
//!
//! The manifest is read with [`Manifest::load`], so a file that is not valid is refused with
//! the line that is wrong. Every source is then judged by [`LicensePolicy`] (permissive plus
//! weak copyleft). The report on standard output lists each source as `accepted` or
//! `rejected` with the reason, then the totals by category. With `--accepted FILE` the ids of
//! the accepted sources are written to FILE, one per line, in manifest order: that is the
//! list `scripts/ingest.py fetch` takes. The exit status is 0 when the manifest is valid,
//! whatever the policy decided.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use midman_data::{LicensePolicy, Manifest};

const USAGE: &str = "usage: check_manifest MANIFEST [--accepted FILE]";

struct Args {
    manifest: PathBuf,
    accepted: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut args = std::env::args().skip(1);
    let mut manifest = None;
    let mut accepted = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
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
    Ok(Args { manifest: manifest.ok_or_else(|| USAGE.to_string())?, accepted })
}

fn run(args: &Args) -> Result<(), String> {
    let manifest = Manifest::load(&args.manifest).map_err(|e| e.to_string())?;
    let policy = LicensePolicy::permissive_and_weak_copyleft();
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
    eprintln!(
        "{} sources: {} accepted, {} rejected",
        manifest.len(),
        partition.accepted.len(),
        partition.rejected.len()
    );
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
