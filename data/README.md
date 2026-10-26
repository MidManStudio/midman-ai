# data

Manifests and instructions only. Bulk datasets and model weights are not committed to this repository.

## `manifest.txt`

The sources the training corpus is built from: one record per repository, each with the URL, the commit it is pinned to, its license as an SPDX expression, the date it was listed and its category (`rust` or `shell`). The format is the one `midman-data` reads (`crates/midman-data/src/manifest.rs`, described in `docs/midman-data.md`).

- A license of `NONE` means the listing found no license. The project uses such a source unless its own files say otherwise (the `Ingest` workflow looks, and says what it found); a source that was flagged, that you have read and want to keep, goes in `notices-reviewed.txt`. A license of `NOASSERTION` means a license file exists that could not be identified, and that source is refused.
- It is a proposal until you have read it: a source is removed by deleting its record (the blank line and the six lines after it), and a source is refused at fetch time if the license policy does not accept its license, so a record that stays in is not necessarily used.
- To see which sources the policy accepts and what they would give, run the `midman-ai Ingest` workflow (mode `measure`). To make a new proposal, run it in mode `discover` and put the manifest it uploads here.
- Details of the commands and the workflow: `docs/ci-and-workflows.md`, section `ingest.py` and `ingest.yml`.
