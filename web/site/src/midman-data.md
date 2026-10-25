# midman-data

**What it does:** everything that happens to training text before the model sees it. A license policy that reads SPDX expressions, a provenance manifest, a secret scrubber, exact deduplication, packing into token shards on disk, and a batcher that hands out training sequences in a seeded order a stopped run can resume.

**Example:** `scrubber.scrub(&text)`, `dedup.insert(text.as_bytes())`, `packer.add_text(&tokenizer, &text)?`, and later `batcher.next_batch()` for the `inputs` and `targets` the model's loss takes. The `build_shards` example does all of it over a list of files.

**Status:** done and tested. On real code it kept 25.7 million tokens at 2.94 bytes per token, and a small model's loss fell as it trained on the batches. The secret scrubber agrees with gitleaks on every sample of a made-up corpus. Not built yet: fetching sources, quality filters, near-duplicate detection, mixing sources by weight.
