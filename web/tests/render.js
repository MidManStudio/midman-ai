// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-web.md, section "tests/render.js"
// ============================================================================
// Renders ci-results.json (scripts/ci_summary.py), train-smoke-results.json
// (scripts/train_smoke_summary.py) and data-smoke-results.json (scripts/data_smoke_summary.py).

function el(doc, tag, text, className) {
  const e = doc.createElement(tag);
  if (text !== undefined) e.textContent = text;
  if (className) e.className = className;
  return e;
}

const mark = (outcome) => (outcome === "success" ? "pass" : outcome === "failure" ? "FAIL" : outcome || "not run");

export function renderCi(doc, data) {
  const m = data.meta;
  const t = data.test;
  const box = doc.getElementById("ci");
  box.replaceChildren();
  box.append(el(doc, "h2", `CI build #${m.build}`));
  box.append(el(doc, "p", `Package ${m.package}, ${m.rust_version}, commit ${m.commit}, branch ${m.branch}.`, "muted"));
  const list = el(doc, "table");
  for (const [name, outcome] of Object.entries(data.outcomes)) {
    const row = el(doc, "tr");
    row.append(el(doc, "td", name), el(doc, "td", mark(outcome), outcome === "success" ? "good" : "bad"));
    list.append(row);
  }
  box.append(list);
  box.append(
    el(
      doc,
      "p",
      `${t.passed} tests passed, ${t.failed} failed, ${t.ignored} ignored. Clippy: ${data.clippy.errors} errors, ${data.clippy.warnings} warnings. Files needing format: ${data.fmt.files.length}.`,
    ),
  );

  const suites = doc.getElementById("suites");
  suites.replaceChildren(el(doc, "h2", "Test suites"));
  const table = el(doc, "table");
  const head = el(doc, "tr");
  for (const h of ["Suite", "Passed", "Failed", "Ignored", "Seconds"]) head.append(el(doc, "th", h));
  table.append(head);
  const ordered = [...t.suites].sort((a, b) => b.failed - a.failed || a.name.localeCompare(b.name));
  for (const s of ordered) {
    const row = el(doc, "tr");
    row.append(
      el(doc, "td", s.name, "mono"),
      el(doc, "td", String(s.passed)),
      el(doc, "td", String(s.failed), s.failed ? "bad" : "muted"),
      el(doc, "td", String(s.ignored), "muted"),
      el(doc, "td", s.duration_s.toFixed(2), "muted"),
    );
    table.append(row);
    for (const name of s.failed_tests ?? []) {
      const fail = el(doc, "tr");
      fail.append(el(doc, "td", `failed: ${name}`, "bad"));
      table.append(fail);
    }
  }
  suites.append(table);
  suites.hidden = false;
}

export function renderSmoke(doc, data) {
  const box = doc.getElementById("smoke");
  box.replaceChildren(el(doc, "h2", `Training smoke test, build #${data.meta.build}`));
  box.append(
    el(doc, "p", `${mark(data.outcome)} on ${data.meta.runner}, ${data.meta.rust_version}, commit ${String(data.meta.commit).slice(0, 10)}.`, "muted"),
  );
  if (data.meta.note) box.append(el(doc, "p", `Note: ${data.meta.note}`));
  const table = el(doc, "table");
  const head = el(doc, "tr");
  for (const h of ["Loss at start", "Loss at end", "Next-token accuracy"]) head.append(el(doc, "th", h));
  table.append(head);
  for (const r of data.runs) {
    const row = el(doc, "tr");
    row.append(
      el(doc, "td", r.first_loss.toFixed(4)),
      el(doc, "td", r.last_loss.toFixed(4)),
      el(doc, "td", `${(r.accuracy * 100).toFixed(1)}%`, r.accuracy > 0.95 ? "good" : "bad"),
    );
    table.append(row);
  }
  box.append(table);
  box.hidden = false;
}

const count = (n) => Number(n).toLocaleString("en-US");
const prefix = (digest) => (digest ? String(digest).slice(0, 12) : "none");

export function renderData(doc, data) {
  const box = doc.getElementById("data");
  const meta = data.meta ?? {};
  box.replaceChildren(el(doc, "h2", `Data pipeline smoke test, build #${meta.build}`));
  box.append(
    el(
      doc,
      "p",
      `${data.ok ? "pass" : "FAIL"} on ${meta.runner}, ${meta.rust_version}, commit ${String(meta.commit).slice(0, 10)}.`,
      data.ok ? "good" : "bad",
    ),
  );
  if (meta.note) box.append(el(doc, "p", `Note: ${meta.note}`));

  const { train, shards, digests } = data;
  if (!train || !shards) {
    box.append(el(doc, "p", "The run is incomplete: the log does not hold the pipeline's numbers.", "bad"));
  } else {
    const rows = [
      ["Corpus", `${count(data.corpus_files)} tracked text files`],
      ["Vocabulary", `${count(train.vocab_size)} ids, ${count(train.merges)} merges`],
      ["Files read", `${count(shards.files_read)}: not UTF-8 ${shards.not_utf8}, empty ${shards.empty}, duplicates ${shards.duplicates}`],
      ["Secrets replaced", String(shards.secrets)],
      ["Documents kept", count(shards.documents)],
      ["Tokens", `${count(shards.tokens)} (${shards.bytes_per_token.toFixed(2)} bytes per token)`],
      ["Shards", `${shards.shard_count}, ${count(shards.sequences)} sequences of ${shards.seq_len} tokens`],
      ["Whole path", `${shards.seconds.toFixed(1)} s, ${shards.mb_per_s.toFixed(1)} MB/s`],
    ];
    if (typeof data.scrub_mb_per_s === "number") rows.push(["Scrubber alone", `${data.scrub_mb_per_s.toFixed(1)} MB/s`]);
    if (digests) {
      rows.push(["Vocabulary file (SHA-256)", prefix(digests.vocabulary)]);
      rows.push([`All ${(digests.shards ?? []).length} shards (SHA-256)`, prefix(digests.combined)]);
      rows.push([`First ${shards.batches ?? 0} batches (SHA-256)`, prefix(shards.batch_digest)]);
    }
    const table = el(doc, "table");
    const head = el(doc, "tr");
    for (const h of ["What", "Value"]) head.append(el(doc, "th", h));
    table.append(head);
    for (const [what, value] of rows) {
      const row = el(doc, "tr");
      row.append(el(doc, "td", what), el(doc, "td", value));
      table.append(row);
    }
    box.append(table);
  }

  const oracle = data.oracle;
  if (oracle) {
    const samples = oracle.kinds.reduce((sum, k) => sum + k.samples, 0);
    const found = oracle.kinds.reduce((sum, k) => sum + k.ours, 0);
    const missed = oracle.kinds.filter((k) => k.gitleaks < k.samples).map((k) => `${k.kind} (${k.gitleaks} of ${k.samples})`);
    box.append(
      el(
        doc,
        "p",
        `Secret scrubber against gitleaks ${oracle.gitleaks_version}: found ${found} of ${samples} samples of ${oracle.kinds.length} kinds, ${oracle.ours_outside} findings outside them.`,
        oracle.ok ? "good" : "bad",
      ),
    );
    if (missed.length) box.append(el(doc, "p", `Gitleaks found fewer samples of: ${missed.join(", ")}.`, "muted"));
  }
  box.hidden = false;
}

export function renderMissing(doc, message) {
  doc.getElementById("ci").textContent = message;
}
