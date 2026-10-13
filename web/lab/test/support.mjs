// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-web.md, section "lab/test/support.mjs"
// ============================================================================
// Shared helpers for the lab tests.

/** A 2D canvas context that records every call and property assignment. */
export function recorder() {
  const calls = [];
  const target = {};
  return new Proxy(target, {
    get(_, name) {
      if (name === "calls") return calls;
      if (name in target) return target[name];
      return (...args) => {
        calls.push([name, ...args]);
      };
    },
    set(_, name, value) {
      target[name] = value;
      return true;
    },
  });
}

/** How many times the recorded context was asked to do `name`. */
export function count(ctx, name) {
  return ctx.calls.filter((c) => c[0] === name).length;
}
