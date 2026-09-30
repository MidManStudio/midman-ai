## What and why

<!-- What changed, and why it is needed. Link the issue if there is one. -->

## Crates touched

<!-- e.g. midman-tokenizer, midman-data -->

## Checklist

- [ ] `ci.yml` run for the touched package(s): fmt, clippy, build and tests pass
- [ ] Each new or changed source file has the NOTICE header, and a section in its `docs/<crate>.md`
- [ ] Fixes or problems found along the way are logged under "Fixes and Problems" in the relevant doc
- [ ] No model weights, datasets, archives, credentials or build output are committed (`security.yml`)
- [ ] Any `unsafe` has a SAFETY comment, and the crate opts out of the workspace lint on purpose
- [ ] Data changes: source URL, revision, license and retrieval date are recorded in the manifest
- [ ] Ubel data changes: the Ubel language version and compiler revision are recorded

## Notes for review

<!-- Anything the reviewer should look at first, or that you are unsure about. -->
