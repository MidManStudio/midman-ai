// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-web.md, section "tests/boot.js"
// ============================================================================
import { renderCi, renderData, renderMissing, renderSmoke } from "./render.js";

async function load(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

try {
  renderCi(document, await load("../data/ci-results.json"));
} catch {
  renderMissing(document, "No CI results are published yet. Run the CI workflow, then deploy the site again.");
}
try {
  renderSmoke(document, await load("../data/train-smoke-results.json"));
} catch {
  // The smoke test is optional; its card simply stays hidden.
}
try {
  renderData(document, await load("../data/data-smoke-results.json"));
} catch {
  // So is the data smoke test.
}
