// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-web.md, section "lab/boot.js"
// ============================================================================
// Loads the WebAssembly engine and starts the page. Kept out of the HTML so a strict
// Content-Security-Policy can forbid inline scripts.
import { Engine } from "./engine.js";
import { init } from "./lab.js";

const status = document.getElementById("status");
const error = document.getElementById("error");
try {
  const engine = await Engine.loadBest({
    simd: new URL("./midman_api.simd.wasm", import.meta.url).href,
    plain: new URL("./midman_api.wasm", import.meta.url).href,
  });
  await init(document, engine);
} catch (e) {
  status.textContent = "The engine could not start.";
  error.hidden = false;
  error.textContent =
    `${e.message} This page needs WebAssembly. If the file is missing, the site was built without the engine.`;
}
