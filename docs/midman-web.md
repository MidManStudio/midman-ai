# midman-web

## Overview
The website: a landing page, the lab, the docs book, and live test and benchmark pages. It is static HTML and JavaScript plus an mdBook, with no UI framework, the same shape as the mid-engine site. It is deployed to Cloudflare Pages by `deploy-site.yml`, by hand only.

The lab is the point of building the site early. `midman-api` is compiled to WebAssembly and a small JavaScript module wraps its C ABI, so the page runs the real model: every parameter count, loss, gradient and attention map comes from the same Rust code the tests check. Nothing is reimplemented in JavaScript. The engine is 256 KB with no imports, in two builds (SIMD and plain) chosen at load time; in Node, the SIMD build trained the Smoke preset 1.6 times faster, and both agree with the native build to within one float rounding step.

Layout under `web/`: `home/` (landing page), `lab/`, `tests/`, `benchmarks/`, `shared/` (theme), `site/` (the mdBook), and `_headers` (Cloudflare headers, with a Content-Security-Policy that allows only same-origin scripts, plus `wasm-unsafe-eval` on the lab page). `scripts/assemble_site.py` puts the parts together into `dist/`.

Test status when this was written: 60 JavaScript tests across five files (engine against the real WebAssembly binary, charts, the whole lab page in a headless DOM, and the two result pages) and 7 tests of the assembly script. **Not tested: the pages in a real browser.** The page tests use a headless DOM with recording canvases, so layout, real canvas rendering and touch behaviour have not been seen by anyone yet.

## Modules

### `lab/engine.js`
**What it does:** The JavaScript side of the C ABI: loads the WebAssembly module from a URL, bytes or a compiled module, picks the SIMD or plain build with `loadBest`, and wraps every export in plain methods (`describe`, `configReport`, `presetConfig`, `ropeTable`, `createLab` and a `Lab` with `step`, `eval`, `stats`, `attention`, `embedding`, `destroy`). No dependencies; the same file runs in a browser and in Node.

**Decisions:** Results are copied out of library memory immediately, because the next call replaces them and memory growth detaches old views. A negative status becomes an `EngineError` with the library's message. Input arrays are written into `midman_alloc` memory and freed after the call. The module has no imports, so it instantiates with an empty import object.

**Tests:** `test/engine.test.mjs`: 17 tests against the real binary.

### `lab/charts.js`
**What it does:** Canvas charts (line, heat map, bars, scatter) that draw onto any 2D context, plus pure helpers: round tick positions, the viridis color map, number formatting and a deterministic two-component PCA.

**Decisions:** Drawing functions take the context and size, so the same code runs on a real canvas and on the recording context the tests use. A log axis skips values with no logarithm and says there is no data yet when nothing is left.

**Tests:** `test/charts.test.mjs`: 14 tests, numerics directly and drawing against a recording context.

### `lab/lab.js`
**What it does:** The lab page logic: `init(document, engine)` wires every control to the engine and the charts (parameter calculator, training with live loss, gradient and learning-rate charts, attention maps per head, rotary embeddings, embedding PCA) and returns its internals.

**Decisions:** Every engine error is caught at the event handler and shown in a banner, never thrown into the page. Running to the end is chunked through a scheduler so the page stays responsive, and Pause takes effect between chunks. Canvases are sized to their on-screen width at the device pixel ratio.

**Tests:** `test/lab.dom.test.mjs`: 16 tests that load the real `index.html` in a headless DOM with the real engine and drive it by clicks and edits.

### `lab/boot.js`
**What it does:** Loads the engine and starts the page, or shows why it could not.

**Decisions:** Kept in its own file so a strict Content-Security-Policy can forbid inline scripts.

**Tests:** Covered indirectly: the DOM test calls `init` directly.

### `lab/index.html`
**What it does:** The lab page: five tabs, each a labelled panel with controls and canvases.

**Decisions:** Mobile first. Canvases have text alternatives that the page updates (for example the loss curve reports its start and end).

**Tests:** `test/lab.dom.test.mjs` loads this file.

### `lab/lab.css`
**What it does:** Styles for the lab's tabs, form fields, readouts and chart grids.

### `lab/test/engine.test.mjs`
**What it does:** Tests of `engine.js` against the real WebAssembly build: no imports, the described presets and exact counts, config reports, the rope table, that the browser build trains like the native build (the first five losses and gradient norms of a known run agree to 1e-4), the schedule, stats, attention and embeddings, results surviving memory growth, determinism, handle safety, bad labs, blow-ups, 64-bit seeds, and `loadBest`.

**Decisions:** The native numbers are pinned from one native run, so the two builds are tested as one system.

### `lab/test/charts.test.mjs`
**What it does:** Tests of `charts.js`: ticks, the color map, number formatting, PCA (direction, ordering, degenerate input) and every drawing function against a recording context.

### `lab/test/lab.dom.test.mjs`
**What it does:** The end-to-end page test described above.

**Decisions:** jsdom is a test-only dependency (`npm install --no-save jsdom`). Without it the test skips locally; with `REQUIRE_DOM=1`, as in the workflow, a missing jsdom is a failure.

### `lab/test/support.mjs`
**What it does:** The recording canvas context shared by the lab tests.

### `shared/theme.css`
**What it does:** The dark theme, header, cards, form controls and tables shared by every page.

**Decisions:** Colors, sizes and fonts are CSS variables so the site can be restyled in one place. There is no logo yet; the header is a text wordmark.

### `home/index.html`
**What it does:** The landing page: what the project is, links to the lab, docs, tests and benchmarks, and an honest note on what works and what is not started.

### `benchmarks/index.html`
**What it does:** The benchmarks page shell.

**Tests:** `test/render.test.mjs` loads this file.

### `benchmarks/render.js`
**What it does:** Renders `bench-results.json` (from `scripts/bench_summary.py`): a table per group with time, a relative bar, the 95% interval, throughput, and a badge against the `baseline-*` variant or a plain multiple against the `unit-*` variant.

**Decisions:** The badge thresholds are the same as the script's. Rendering is pure DOM construction from a document and data, so a test runs it headlessly.

**Tests:** `test/render.test.mjs`: 8 tests, rendering a real run (26 groups, 51 benchmarks).

### `benchmarks/boot.js`
**What it does:** Fetches `../data/bench-results.json` and renders it, or says that no results are published yet.

### `benchmarks/bench.css`
**What it does:** Styles for the relative bars and the ratio colors.

### `benchmarks/test/render.test.mjs`
**What it does:** Tests of the benchmarks renderer against a real result file.

### `tests/index.html`
**What it does:** The tests page shell, with a card for the CI results, one for the training smoke test and one for the data pipeline smoke test, and a lede that says what each checks.

**Tests:** `test/render.test.mjs` loads this file.

### `tests/render.js`
**What it does:** Renders `ci-results.json` (from `scripts/ci_summary.py`): each check's outcome, the totals, and a table of suites with failures first; the training smoke test's losses and accuracy; and the data pipeline smoke test (`data-smoke-results.json` from `scripts/data_smoke_summary.py`): the pass or fail line with the runner and commit, a table of the pipeline numbers and the first twelve digits of the output digests, and the scrubber-against-gitleaks line with the kinds where gitleaks found fewer samples. A result with no pipeline numbers still renders and says that the run is incomplete.

### `tests/boot.js`
**What it does:** Fetches the result files and renders them. A missing smoke-test file (training or data) only hides its card.

### `tests/test/render.test.mjs`
**What it does:** Tests of the tests-page renderer against a real CI result (291 passing tests in 20 suites), a training smoke-test result and a data smoke-test result (`test/fixtures/data-smoke-results.json`, from a local run of the workflow's steps).

**Tests:** 10 tests. The five added are for the data card: a real result with its numbers, digests and gitleaks comparison; a failed run marked FAIL with the note shown; no gitleaks line when the comparison did not run; an incomplete result; and the hidden container and the lede.

## CI and Workflows
`deploy-site.yml` builds the engine, runs these tests, builds the book, pulls the latest CI, benchmark and smoke-test (training and data) results from earlier runs, assembles `dist/` and deploys it. Run the tests locally with `MIDMAN_WASM=path/to/midman_api.wasm node --test web/lab/test/engine.test.mjs` (and the other files). See `docs/ci-and-workflows.md`.

## Fixes and Problems

### `lab/charts.js`
- **PCA reported rounding noise as signal.** The mean was computed as a sum of `value / rows` terms, which is not exact in floating point, so a cloud of identical points had residuals around 1e-16 and was reported as "100% of the variance explained". The mean is now summed first and divided once.

### `lab/test/lab.dom.test.mjs`
- **A test counted draws cumulatively.** A redraw reuses the same recording context, so the count after a second draw included the first. The test now counts only what the redraw added.

### `lab/test/engine.test.mjs`
- **Node 22's `--test` takes files, not directories.** The command in the header comment names the file.

### `benchmarks/test/render.test.mjs`
- **A wrong expectation:** `2.3045 ms` formats as `2.304` in floating point, not `2.305`. The test uses a value that does not sit on a rounding boundary.

### `home/index.html`
- **Open: there is no logo.** mid-engine's site uses a vectorized logo you supplied; none exists for midman-ai, so the header is a text wordmark.
- **Open: Cloudflare Pages versus Workers.** Cloudflare now recommends Workers with static assets for new projects and keeps Pages fully supported. This site uses Pages to match mid-engine and your existing token; switching is a different deploy command and a small config file.
