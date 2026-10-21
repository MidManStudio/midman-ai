// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-data.md, section "examples/scan.rs"
// ============================================================================
//! Runs the secret scrubber over files and reports what it finds. Used to compare it with
//! gitleaks and to measure false positives and speed; see `docs/midman-data.md`.
//!
//! ```text
//! cargo run --release -p midman-data --example scan -- lines FILE
//! cargo run --release -p midman-data --example scan -- files   < list-of-paths
//! cargo run --release -p midman-data --example scan -- speed   < list-of-paths
//! ```
//!
//! `lines` prints `line<TAB>rule<TAB>start of the match` for every finding in one file.
//! `files` does the same for every file named on standard input (as `path:line`) and ends
//! with totals on standard error. `speed` times the scan of those files, best of five.
//! Add `--no-generic` to turn off the keyword-and-entropy rule. Matches are cut to 48
//! characters, so a report never holds a whole secret.

use std::collections::BTreeMap;
use std::io::BufRead;
use std::time::Instant;

use midman_data::{Finding, Scrubber};

const SHOWN: usize = 48;

fn describe(text: &str, finding: &Finding) -> (usize, String) {
    let line = text[..finding.start].matches('\n').count() + 1;
    let shown: String = text[finding.start..finding.end].chars().take(SHOWN).collect();
    (line, shown.replace('\n', "\\n"))
}

fn paths_from_stdin() -> Vec<String> {
    std::io::stdin().lock().lines().map_while(Result::ok).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let scrubber = if args.iter().any(|a| a == "--no-generic") {
        Scrubber::new().without_generic()
    } else {
        Scrubber::new()
    };
    match args.first().map(String::as_str) {
        Some("lines") if args.len() >= 2 => {
            let text = std::fs::read_to_string(&args[1]).expect("a readable UTF-8 file");
            for finding in scrubber.find(&text) {
                let (line, shown) = describe(&text, &finding);
                println!("{line}\t{}\t{shown}", finding.rule);
            }
        }
        Some("files") => {
            let (mut files, mut bytes, mut total) = (0u64, 0u64, 0u64);
            let mut by_rule: BTreeMap<&str, u64> = BTreeMap::new();
            for path in paths_from_stdin() {
                let Ok(text) = std::fs::read_to_string(&path) else { continue };
                files += 1;
                bytes += text.len() as u64;
                for finding in scrubber.find(&text) {
                    let (line, shown) = describe(&text, &finding);
                    println!("{path}:{line}\t{}\t{shown}", finding.rule);
                    total += 1;
                    *by_rule.entry(finding.rule).or_default() += 1;
                }
            }
            eprintln!(
                "files {files}, {:.1} MB, findings {total}, by rule {by_rule:?}",
                bytes as f64 / 1e6
            );
        }
        Some("speed") => {
            let texts: Vec<String> =
                paths_from_stdin().iter().filter_map(|p| std::fs::read_to_string(p).ok()).collect();
            let megabytes = texts.iter().map(String::len).sum::<usize>() as f64 / 1e6;
            let mut best = f64::MAX;
            for _ in 0..5 {
                let start = Instant::now();
                let found: usize = texts.iter().map(|t| scrubber.find(t).len()).sum();
                std::hint::black_box(found);
                best = best.min(start.elapsed().as_secs_f64());
            }
            println!("{megabytes:.1} MB in {best:.3} s = {:.1} MB/s", megabytes / best);
        }
        _ => {
            eprintln!("usage: scan lines FILE | scan files | scan speed   [--no-generic]");
            std::process::exit(2);
        }
    }
}
