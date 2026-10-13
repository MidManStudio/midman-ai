// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-web.md, section "benchmarks/render.js"
// ============================================================================
// Renders the JSON written by scripts/bench_summary.py. Pure DOM construction from a
// document and a data object, so a test can run it in a headless DOM.

/** Formats nanoseconds with a sensible unit. */
export function formatTime(ns) {
  if (ns >= 1e9) return `${(ns / 1e9).toFixed(3)} s`;
  if (ns >= 1e6) return `${(ns / 1e6).toFixed(3)} ms`;
  if (ns >= 1e3) return `${(ns / 1e3).toFixed(3)} µs`;
  return `${ns.toFixed(2)} ns`;
}

/** The same thresholds bench_summary.py uses for its badges. */
export function badge(ratio) {
  if (ratio <= 1.05) return { text: `${ratio.toFixed(2)}x ${ratio < 0.95 ? "(faster)" : "(parity)"}`, level: "ok" };
  if (ratio <= 1.5) return { text: `${ratio.toFixed(2)}x`, level: "warn" };
  if (ratio <= 5) return { text: `${ratio.toFixed(2)}x`, level: "bad" };
  return { text: `${ratio.toFixed(1)}x slower`, level: "bad" };
}

function cell(doc, tag, text, className) {
  const el = doc.createElement(tag);
  el.textContent = text;
  if (className) el.className = className;
  return el;
}

export function render(doc, data) {
  const summary = doc.getElementById("summary");
  const host = doc.getElementById("groups");
  host.replaceChildren();
  const m = data.meta ?? {};
  const total = data.groups.reduce((n, g) => n + g.rows.length, 0);
  summary.textContent =
    `Build #${m.build}, ${data.outcome === "success" ? "passed" : "FAILED"}. ${total} benchmarks in ${data.groups.length} groups. ` +
    `Package ${m.package}, runner ${m.runner}, ${m.rust_version}, commit ${m.commit}.` +
    (m.note ? ` Note: ${m.note}` : "");
  for (const g of data.groups) {
    const section = doc.createElement("section");
    section.className = "card group";
    section.append(cell(doc, "h2", g.group));
    const wrap = doc.createElement("div");
    wrap.className = "scroll";
    const table = doc.createElement("table");
    const head = doc.createElement("tr");
    const ratioHead = g.kind === "baseline" ? "vs baseline" : g.kind === "unit" ? "x the unit variant" : null;
    for (const h of ["Variant", "Time", "", "95% interval", "Throughput", ...(ratioHead ? [ratioHead] : [])]) {
      head.append(cell(doc, "th", h));
    }
    table.append(head);
    const slowest = Math.max(...g.rows.map((r) => r.mid_ns));
    for (const r of g.rows) {
      const tr = doc.createElement("tr");
      tr.append(cell(doc, "td", r.variant, "mono"), cell(doc, "td", formatTime(r.mid_ns)));
      const barCell = doc.createElement("td");
      barCell.style.width = "30%";
      const bar = doc.createElement("span");
      bar.className = r.variant === g.denominator ? "bar baseline" : "bar";
      bar.style.width = `${Math.max((r.mid_ns / slowest) * 100, 1)}%`;
      barCell.append(bar);
      tr.append(barCell, cell(doc, "td", `${formatTime(r.low_ns)} to ${formatTime(r.high_ns)}`), cell(doc, "td", r.throughput ?? "n/a"));
      if (ratioHead) {
        if (r.variant === g.denominator) tr.append(cell(doc, "td", `(${g.kind})`, "muted"));
        else if (g.kind === "baseline") {
          const b = badge(r.ratio);
          tr.append(cell(doc, "td", b.text, `ratio ${b.level}`));
        } else tr.append(cell(doc, "td", `${r.ratio.toFixed(2)}x`, "ratio"));
      }
      table.append(tr);
    }
    wrap.append(table);
    section.append(wrap);
    host.append(section);
  }
  if (data.diagnostics?.length) {
    const d = doc.createElement("section");
    d.className = "card";
    d.append(cell(doc, "h2", "Diagnostics"));
    for (const x of data.diagnostics) d.append(cell(doc, "p", `${x.kind}: ${x.count}`));
    host.append(d);
  }
}

export function renderMissing(doc, message) {
  doc.getElementById("summary").textContent = message;
}
