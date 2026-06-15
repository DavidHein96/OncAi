// Tests for the pure front-end helpers in app.js, run with Node's built-in
// test runner (no dependencies):  node --test web/
//
// app.js exports these only under Node; in the browser it boots the app instead.

const { test } = require("node:test");
const assert = require("node:assert/strict");

const {
  canon,
  sameValue,
  shortenPath,
  gatherEvidence,
  buildHighlightedNote,
  escapeHtml,
} = require("./app.js");

// --------------------------------------------------------------------------- //
// canon / sameValue — "did this field actually change?"
// --------------------------------------------------------------------------- //

test("sameValue is insensitive to object key order", () => {
  assert.ok(sameValue({ date: "2026-02", precision: 2 }, { precision: 2, date: "2026-02" }));
});

test("sameValue distinguishes genuinely different values", () => {
  assert.ok(!sameValue({ a: 1 }, { a: 2 }));
  assert.ok(
    !sameValue({ date: "2026", precision: 1, anchor: "x" }, { date: "2026", precision: 1 })
  );
});

test("sameValue treats undefined and null as equal", () => {
  assert.ok(sameValue(undefined, null));
});

test("canon recurses into nested arrays and objects", () => {
  assert.equal(canon([{ b: 2, a: 1 }]), canon([{ a: 1, b: 2 }]));
});

// --------------------------------------------------------------------------- //
// shortenPath — header display of the reviews file location
// --------------------------------------------------------------------------- //

test("shortenPath keeps the last two segments with an ellipsis", () => {
  assert.equal(
    shortenPath("/Users/me/Documents/oncai_reviews/demo.reviews.jsonl"),
    "…/oncai_reviews/demo.reviews.jsonl"
  );
});

test("shortenPath leaves short paths untouched", () => {
  assert.equal(shortenPath("a/b"), "a/b");
  assert.equal(shortenPath("solo"), "solo");
});

test("shortenPath handles Windows backslash separators", () => {
  assert.equal(shortenPath("C:\\Users\\me\\Documents\\x.jsonl"), "…/Documents/x.jsonl");
});

// --------------------------------------------------------------------------- //
// gatherEvidence — flatten explicit provenance snippets for highlighting
// --------------------------------------------------------------------------- //

test("gatherEvidence flattens evidence snippets and review anchors", () => {
  const ev = {
    fields: {
      review_anchor: ["review quote"],
      evidence: ["a quote", "b ||| c"],
      histology: ["regular list field"],
    },
  };
  assert.deepEqual(gatherEvidence(ev), ["a quote", "b", "c", "review quote"]);
});

test("gatherEvidence accepts scalar evidence", () => {
  const ev = {
    fields: {
      evidence: "diagnosis quote",
    },
  };
  assert.deepEqual(gatherEvidence(ev), ["diagnosis quote"]);
});

test("gatherEvidence returns nothing when there is no evidence field", () => {
  assert.deepEqual(gatherEvidence({ fields: { ordinary_list: ["not evidence"] } }), []);
});

// --------------------------------------------------------------------------- //
// escapeHtml
// --------------------------------------------------------------------------- //

test("escapeHtml neutralizes HTML metacharacters", () => {
  assert.equal(escapeHtml("<b>&\"'"), "&lt;b&gt;&amp;&quot;&#39;");
});

// --------------------------------------------------------------------------- //
// buildHighlightedNote — the whitespace-flexible evidence highlighter
// --------------------------------------------------------------------------- //

test("buildHighlightedNote wraps an exact match in <mark>", () => {
  const html = buildHighlightedNote("The CAIX is positive here.", ["CAIX is positive"]);
  assert.ok(html.includes('<mark class="hl">CAIX is positive</mark>'));
});

test("buildHighlightedNote matches across differing whitespace, case-insensitively", () => {
  // note has a newline + double space where the quote has single spaces
  const html = buildHighlightedNote("caix  is\npositive", ["CAIX is positive"]);
  assert.ok(html.includes("<mark"));
});

test("buildHighlightedNote escapes note text and leaves non-matches unmarked", () => {
  const html = buildHighlightedNote("a <tag> with no quote", ["zzz"]);
  assert.ok(html.includes("&lt;tag&gt;"));
  assert.ok(!html.includes("<mark"));
});

test("buildHighlightedNote handles missing text", () => {
  assert.equal(buildHighlightedNote("", ["anything"]), "(note text unavailable)");
});

test("buildHighlightedNote merges overlapping matches into a single span", () => {
  // two quotes whose matches overlap should not produce nested/duplicate marks
  const html = buildHighlightedNote("alpha beta gamma", ["alpha beta", "beta gamma"]);
  assert.equal((html.match(/<mark/g) || []).length, 1);
});
