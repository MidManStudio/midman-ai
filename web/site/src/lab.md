# The lab

The lab is a page that runs the real Rust code in your browser. It has a parameter calculator, training with live loss and gradient charts, attention heat maps, a rotary-embedding explorer, and an embedding plot.

**How it works:** `midman-api` is compiled to WebAssembly (a SIMD build and a plain build, chosen at load time). A small dependency-free JavaScript module wraps its C ABI. Every number on the page comes from Rust; nothing is reimplemented in JavaScript.

**Status:** tested end to end in CI: the real WebAssembly binary, the real page in a headless browser DOM, and agreement with the native build.
