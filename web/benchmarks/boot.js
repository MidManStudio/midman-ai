// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-web.md, section "benchmarks/boot.js"
// ============================================================================
import { render, renderMissing } from "./render.js";

try {
  const response = await fetch("../data/bench-results.json");
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  render(document, await response.json());
} catch {
  renderMissing(
    document,
    "No benchmark results are published yet. Run the Benchmarks workflow, then deploy the site again.",
  );
}
