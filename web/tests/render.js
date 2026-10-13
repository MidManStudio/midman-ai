// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-web.md, section "tests/render.js"
// ============================================================================
// Renders ci-results.json (scripts/ci_summary.py) and train-smoke-results.json
// (scripts/train_smoke_summary.py).

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

export function renderMissing(doc, message) {
  doc.getElementById("ci").textContent = message;
}
